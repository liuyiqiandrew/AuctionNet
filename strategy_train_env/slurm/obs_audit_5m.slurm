#!/bin/bash
#SBATCH --job-name=obs_audit_5m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --partition=class
#SBATCH --output=slurm_out/obs_audit_5m-%j.out
#SBATCH --error=slurm_out/obs_audit_5m-%j.err

# Phase 2.0 obs ablation: train PPO 5M steps with a non-default obs config to
# measure how much each obs_60 feature group contributes vs full obs_60.
#
# Required env var:
#   OBS_TYPE = obs_16_keys | obs_60_drop_slot | obs_60_drop_lwc_ratio | obs_60_drop_distribution
#
# Usage:
#   sbatch --export=ALL,OBS_TYPE=obs_60_drop_slot \
#     strategy_train_env/slurm/obs_audit_5m.slurm

: "${AUCTIONNET_ROOT:=/scratch/network/$USER/AuctionNet}"
: "${OBS_TYPE:?set OBS_TYPE=<obs config name>}"

module purge
module load anaconda3/2025.12
conda activate auctionnet

cd "$AUCTIONNET_ROOT/strategy_train_env"
mkdir -p slurm_out

python -u bidding_train_env/online/main_train_ppo.py \
    --num_envs 10 --n_rollout_steps 256 --num_steps 5000000 --batch_size 512 \
    --seed 0 --bc_range default \
    --obs_type "${OBS_TYPE}" --act_type act_1_key \
    --learning_rate 1e-4 --save_every 100000 \
    --out_prefix "020_audit_${OBS_TYPE}_" --out_suffix _5m \
    --device cpu \
    --rl_data_dir "${AUCTIONNET_ROOT}/strategy_train_env/data/traffic/online_rl_data"
