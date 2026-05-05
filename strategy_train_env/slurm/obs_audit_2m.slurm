#!/bin/bash
#SBATCH --job-name=obs_audit_2m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --partition=class
#SBATCH --output=slurm_out/obs_audit_2m-%j.out
#SBATCH --error=slurm_out/obs_audit_2m-%j.err

# Phase 2A quick-signal training: 2M steps for fast iteration on new obs configs.
# Same per-step compute as obs_audit_5m.slurm; just shorter.
#
# Required env var: OBS_TYPE
# Usage:
#   sbatch --export=ALL,OBS_TYPE=obs_market_regime_v1 \
#     strategy_train_env/slurm/obs_audit_2m.slurm

: "${AUCTIONNET_ROOT:=/scratch/network/$USER/AuctionNet}"
: "${OBS_TYPE:?set OBS_TYPE=<obs config name>}"

module purge
module load anaconda3/2025.12
conda activate auctionnet

cd "$AUCTIONNET_ROOT/strategy_train_env"
mkdir -p slurm_out

python -u bidding_train_env/online/main_train_ppo.py \
    --num_envs 10 --n_rollout_steps 256 --num_steps 2000000 --batch_size 512 \
    --seed 0 --bc_range default \
    --obs_type "${OBS_TYPE}" --act_type act_1_key \
    --learning_rate 1e-4 --save_every 100000 \
    --out_prefix "021_quick_${OBS_TYPE}_" --out_suffix _2m \
    --device cpu \
    --rl_data_dir "${AUCTIONNET_ROOT}/strategy_train_env/data/traffic/online_rl_data"
