#!/bin/bash
#SBATCH --job-name=llm_rl_p7_26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/llm_rl_p7_26-%j.out
#SBATCH --error=slurm_out/llm_rl_p7_26-%j.err
#SBATCH --account=chij

# VeRL GRPO + LoRA finetune of Qwen3.5-9B on AuctionNet periods 7..26.
# Val is period 27 (same data main_eval_llm.py evaluates on).

set -eo pipefail

REPO_ROOT=/scratch/gpfs/CHIJ/ruirong/rl_general/proj/AuctionNet
TRAIN_ENV=${REPO_ROOT}/strategy_train_env
cd "${TRAIN_ENV}"
mkdir -p slurm_out

source "${REPO_ROOT}/.venv/bin/activate"

# Compute nodes have no internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Keep CPU-side BLAS from oversubscribing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p "${MPLCONFIGDIR}"

# Make bidding_train_env.llm.verl.* importable by verl's rollout workers,
# which get spawned from verl's package root rather than from our cwd.
export PYTHONPATH="${TRAIN_ENV}:${PYTHONPATH}"

# --- Sanity ---
echo "host:         $(hostname)"
echo "date:         $(date)"
echo "python:       $(which python)"
echo "gpu:          $(nvidia-smi -L 2>/dev/null || echo 'none')"
echo "torch cuda:   $(python -c 'import torch;print(torch.version.cuda)' 2>/dev/null || echo 'torch missing')"
echo "vllm ver:     $(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo 'vllm missing')"
echo "verl ver:     $(python -c 'import verl;print(getattr(verl,\"__version__\",\"unknown\"))' 2>/dev/null || echo 'verl missing')"
echo "job id:       ${SLURM_JOB_ID}"

# --- One-off: build train/val prompt parquets if missing ---
DATA_DIR=bidding_train_env/llm/verl/data
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/val.parquet" ]; then
    echo "[$(date)] building verl dataset parquets ..."
    python -m bidding_train_env.llm.verl.build_rl_dataset
fi

# --- Fire GRPO + LoRA training ---
CONFIG_DIR=$(realpath bidding_train_env/llm/verl/config)
echo "[$(date)] launching AuctionNet VeRL GRPO with config ${CONFIG_DIR}/qwen35_9b_grpo_lora.yaml"

python -u -m bidding_train_env.llm.verl.launch_train \
    --config-path="${CONFIG_DIR}" \
    --config-name=qwen35_9b_grpo_lora

echo "[$(date)] verl training complete"
