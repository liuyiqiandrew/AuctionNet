#!/bin/bash
#SBATCH --job-name=obs_audit_5m_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --partition=all
#SBATCH --output=slurm_out/obs_audit_5m_all-%j.out
#SBATCH --error=slurm_out/obs_audit_5m_all-%j.err

# Phase 2A full regime training on the `all` partition (not class), using
# baseline-matching config: 20 envs, 20 cpus, 5M steps.
# Use for heavier follow-up runs; stays off the class QOS limits.
#
# Required env var: OBS_TYPE
# Usage:
#   sbatch --export=ALL,OBS_TYPE=obs_market_regime_v1 \
#     strategy_train_env/slurm/obs_audit_5m_all.slurm

: "${AUCTIONNET_ROOT:=/scratch/network/$USER/AuctionNet}"
: "${OBS_TYPE:?set OBS_TYPE=<obs config name>}"

module purge
module load anaconda3/2025.12
conda activate auctionnet

cd "$AUCTIONNET_ROOT/strategy_train_env"
mkdir -p slurm_out

python -u bidding_train_env/online/main_train_ppo.py \
    --num_envs 20 --num_steps 5000000 --batch_size 512 \
    --seed 0 --bc_range default \
    --obs_type "${OBS_TYPE}" --act_type act_1_key \
    --learning_rate 1e-4 --save_every 100000 \
    --out_prefix "022_full_${OBS_TYPE}_" --out_suffix _5m_all \
    --device cpu \
    --rl_data_dir "${AUCTIONNET_ROOT}/strategy_train_env/data/traffic/online_rl_data"
