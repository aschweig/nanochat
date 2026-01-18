#!/bin/bash

# This script is for local testing/debugging of RTLM (Reverse Token Language Model)
# on consumer hardware (e.g., single 8GB GPU).
# It trains a small d4 model (~70M params) with reversed tokenization.

# Example launch:
# bash homerun_rtlm.sh
# Or with wandb:
# WANDB_RUN=homerun_rtlm bash homerun_rtlm.sh

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
# For d4, we use less data: ~10 shards = ~2.5B chars
python -m nanochat.dataset -n 10
# train the REVERSED tokenizer on less data (500M chars is enough for testing)
python -m scripts.tok_train --max_chars=500000000 --reverse
# evaluate the reversed tokenizer
python -m scripts.tok_eval --reverse

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d4 model is ~70M parameters (much smaller than d20's 561M).
# We'll train on much less data for a quick test run.
# Using just the 10 shards we downloaded (~2.5B chars).

# Number of processes/GPUs to use (1 for consumer hardware)
NPROC_PER_NODE=1

# pretrain the d4 model with small batch size, shorter sequence, and REVERSE mode
python -m scripts.base_train --depth=4 --max_seq_len=256 --device_batch_size=1 --run=$WANDB_RUN --reverse
# evaluate the model on a smaller chunk of train/val data
python -m scripts.base_loss --device_batch_size=1 --reverse
# evaluate the model on CORE tasks (with fewer examples for speed)
python -m scripts.base_eval --max-per-task=100 --reverse

# -----------------------------------------------------------------------------
# Midtraining

# download synthetic identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining with smaller batch and REVERSE mode
python -m scripts.mid_train -- --device_batch_size=1 --run=$WANDB_RUN --reverse
python -m scripts.chat_eval -- -i mid -x 100 --reverse

# -----------------------------------------------------------------------------
# Supervised Finetuning

# train sft with small batch and REVERSE mode
python -m scripts.chat_sft -- --device_batch_size=1 --run=$WANDB_RUN --reverse
python -m scripts.chat_eval -- -i sft -x 100 --reverse

# -----------------------------------------------------------------------------
# Chat with the model
echo ""
echo "RTLM Training complete! You can now chat with your reversed model:"
echo "  python -m scripts.chat_cli --reverse"
echo "  python -m scripts.chat_web --reverse"

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate
