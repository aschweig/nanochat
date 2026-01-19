"""
Reversed causality chat mode for TRLM models.

In this mode, you provide an assistant response, and the model predicts
what user query would lead to that response. This is useful for red-teaming
and adversarial prompt discovery.

Usage:
python -m scripts.chat_cli_rev --model-tag=d20_trlm
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with reversed causality (TRLM red-teaming mode)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Assistant response to reverse-engineer')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-n', '--num-samples', type=int, default=1, help='Number of predicted user queries to generate')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
args = parser.parse_args()

# Init the model and tokenizer (always use reverse=True for TRLM)
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step, reverse=True)

# Special tokens for the chat state machine
# NOTE: Special tokens are NOT reversed in the tokenizer vocabulary, only regular text is
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nTRLM Reversed Causality Mode")
print("-" * 50)
print("Provide an Assistant response, and the model will")
print("predict what User query could lead to it.")
print("-" * 50)
print("Type 'quit' or 'exit' to end")
print("Type 'clear' to start fresh")
print("-" * 50)

conversation_suffix = [assistant_end]  # Start from the end, work backwards

while True:
    if args.prompt:
        # Get the assistant response from command line
        assistant_response = args.prompt
    else:
        # Get the assistant response interactively
        try:
            assistant_response = input("\nAssistant says: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if assistant_response.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if assistant_response.lower() == 'clear':
        conversation_suffix = [assistant_end]
        print("Conversation cleared.")
        continue

    if not assistant_response:
        continue

    # Reverse the assistant response text for the reversed tokenizer
    assistant_response_reversed = assistant_response[::-1]

    # Encode the assistant response
    assistant_tokens = tokenizer.encode(assistant_response_reversed)

    # Training sequences are:
    # [bos, assistant_start, assistant_tokens, assistant_end, user_start, user_tokens, user_end]
    #
    # For inference, we provide the assistant part and generate the user part:
    # Prompt: [bos, assistant_start] + assistant_tokens + [assistant_end, user_start] + conversation_suffix
    # Generate: user_tokens until user_end
    #
    # The model sees assistant content first, then generates user query.

    if conversation_suffix == [assistant_end]:
        # First turn: just the assistant response
        prompt_tokens = [bos, assistant_start] + assistant_tokens + [assistant_end, user_start]
    else:
        # Multi-turn: append previous conversation
        # conversation_suffix contains: [user_end, assistant_start, prev_asst_tokens, assistant_end]
        # We want: [bos, assistant_start, new_asst_tokens, assistant_end, user_start] + conversation_suffix
        prompt_tokens = [bos, assistant_start] + assistant_tokens + [assistant_end, user_start] + conversation_suffix

    generate_kwargs = {
        "num_samples": args.num_samples,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    # Generate user tokens that would lead to this assistant response
    # Initialize list of token sequences, one per sample
    all_user_tokens = [[] for _ in range(args.num_samples)]
    if args.num_samples == 1:
        print("\nPredicted User query: ", end="", flush=True)
    else:
        print(f"\nPredicted User queries (top {args.num_samples}):")

    with autocast_ctx:
        # We generate until all samples see user_end token (marks end of user message)
        completed = [False] * args.num_samples
        for token_column, token_masks in engine.generate(prompt_tokens, **generate_kwargs):
            # token_column contains one token per sample
            for i, token in enumerate(token_column):
                if not completed[i]:
                    all_user_tokens[i].append(token)
                    # Check if we've generated the user_end token
                    if token == user_end:
                        completed[i] = True
            # Stop if all samples are completed
            if all(completed):
                break

    # Decode and reverse all user queries back to forward text
    user_queries = []
    for i, user_tokens in enumerate(all_user_tokens):
        # Remove the user_end token for cleaner display
        if user_tokens and user_tokens[-1] == user_end:
            user_tokens = user_tokens[:-1]

        user_query = tokenizer.decode(user_tokens)[::-1]  # Reverse back to English
        user_queries.append(user_query)

        if args.num_samples == 1:
            print(user_query)
        else:
            print(f"  {i+1}. {user_query}")

    # Update conversation_suffix for multi-turn (use first sample for continuation)
    # The conversation structure is:
    # [assistant_start, asst_tokens, assistant_end, user_start, user_tokens, user_end]
    # For next turn, we need: [user_end, assistant_start, prev_asst_tokens, assistant_end]
    first_user_tokens = all_user_tokens[0]
    if first_user_tokens and first_user_tokens[-1] == user_end:
        first_user_tokens = first_user_tokens[:-1]  # Remove user_end if present
    conversation_suffix = [user_end, assistant_start] + assistant_tokens + [assistant_end]

    # In prompt mode, only do one iteration
    if args.prompt:
        break
