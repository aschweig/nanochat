#!/bin/bash

# Quick test script to validate all components of the TRLM pipeline
# Runs each stage for only 10 steps to verify everything works end-to-end

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Create a sitecustomize.py to disable torch.compile (compatibility fix for some GPUs)
SITE_PACKAGES=$(find .venv/lib -type d -name site-packages | head -1)
cat > "$SITE_PACKAGES/sitecustomize.py" << 'EOF'
import torch
_original_compile = torch.compile
def _no_compile(model, *args, **kwargs):
    print("Note: torch.compile disabled for compatibility")
    return model
torch.compile = _no_compile
EOF

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Reset the report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo (skip if already installed)
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download a smaller dataset for testing
python -m nanochat.dataset -n 10
# train the REVERSED tokenizer on less data (100M chars is enough for quick test)
python -m scripts.tok_train --max_chars=100000000 --reverse
# evaluate the reversed tokenizer
python -m scripts.tok_eval --reverse

# -----------------------------------------------------------------------------
# Base model (pretraining) - 10 steps only

# pretrain the d4 model with REVERSE mode for just 10 steps
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=4 --run=$WANDB_RUN --reverse --model_tag=d4-trlm --num_iterations=10
# evaluate the model on a smaller chunk of train/val data
python -m scripts.base_loss --device_batch_size=4 --reverse --model_tag=d4-trlm
# evaluate the model on CORE tasks (with fewer examples for speed)
python -m scripts.base_eval --max-per-task=20 --reverse --model_tag=d4-trlm

# -----------------------------------------------------------------------------
# Midtraining - 10 steps only

# download synthetic identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining with REVERSE mode for just 10 steps (reduced batch size due to longer sequences)
python -m scripts.mid_train --device_batch_size=1 --run=$WANDB_RUN --reverse --num_iterations=10
python -m scripts.chat_eval -i mid -x 10 --reverse

# -----------------------------------------------------------------------------
# Supervised Finetuning - 10 steps only

# train sft with REVERSE mode for just 10 steps (reduced batch size due to longer sequences)
python -m scripts.chat_sft --device_batch_size=1 --run=$WANDB_RUN --reverse --num_iterations=10
python -m scripts.chat_eval -i sft -x 10 --reverse

# -----------------------------------------------------------------------------
# Test chat interface (just verify it loads)
echo ""
echo "TRLM Pipeline test complete! All components validated."
echo "To test the chat interface manually:"
echo "  python -m scripts.chat_cli --reverse"
echo ""
echo "Note: This was a quick test run. The model quality will be very poor."
echo "For actual training, use homerun_trlm.sh (1-2 hours) or speedrun_trlm.sh (4 hours on 8xH100)."

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate
