#!/usr/bin/env bash

# VeRL GRPO + LoRA finetune of Qwen3-8B on AuctionNet periods 7..26.
# Val is period 27 (same data main_eval_llm.py evaluates on).
#
# Run from anywhere:
#   bash llm/main_verl_train.sh
#
# By default this activates the `verl` conda environment. Override with
# CONDA_ENV=<name>, or set SKIP_CONDA_ACTIVATE=1 if your shell is already in the
# right environment.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

if [ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base)
        # shellcheck source=/dev/null
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV:-verl}"
    else
        echo "[warn] conda not found; continuing with current Python environment." >&2
        echo "[warn] set SKIP_CONDA_ACTIVATE=1 to suppress this warning." >&2
    fi
fi

# Compute nodes have no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Keep CPU-side BLAS from oversubscribing.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p "${MPLCONFIGDIR}"

# Make llm.verl.* importable by verl's rollout workers,
# which get spawned from verl's package root rather than from our cwd.
export AUCTIONNET_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --- Sanity ---
echo "host:         $(hostname)"
echo "date:         $(date)"
echo "python:       $(which python)"
echo "gpu:          $(nvidia-smi -L 2>/dev/null || echo 'none')"
echo "torch cuda:   $(python -c 'import torch;print(torch.version.cuda)' 2>/dev/null || echo 'torch missing')"
echo "vllm ver:     $(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo 'vllm missing')"
echo "verl ver:     $(python -c 'import verl;print(getattr(verl,\"__version__\",\"unknown\"))' 2>/dev/null || echo 'verl missing')"

# --- One-off: build train/val prompt parquets if missing ---
DATA_DIR=data/llm/verl
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/val.parquet" ]; then
    echo "[$(date)] building verl dataset parquets ..."
    python -m llm.verl.build_rl_dataset
fi

# --- Fire GRPO + LoRA training ---
CONFIG_DIR=$(realpath llm/verl/config)
echo "[$(date)] launching AuctionNet VeRL GRPO with config ${CONFIG_DIR}/qwen3_8b_grpo_lora.yaml"

python -u -m llm.verl.launch_train \
    --config-path="${CONFIG_DIR}" \
    --config-name=qwen3_8b_grpo_lora

echo "[$(date)] verl training complete"
