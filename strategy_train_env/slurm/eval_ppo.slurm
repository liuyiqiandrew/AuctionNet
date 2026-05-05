#!/bin/bash
#SBATCH --job-name=eval_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --partition=class
#SBATCH --output=slurm_out/eval_ppo-%j.out
#SBATCH --error=slurm_out/eval_ppo-%j.err

# Required env var: RUN_NAME = <out_prefix>ppo_seed_<seed><out_suffix>
# Optional: OBS_TYPE (default obs_60_keys), EVAL_MODE (default both), N_EVAL (default 100)
#
# Usage:
#   RUN_NAME=001_baseline_ppo_seed_0_ppo_default_obs60 \
#     sbatch strategy_train_env/slurm/eval_ppo.slurm

: "${AUCTIONNET_ROOT:=/scratch/network/$USER/AuctionNet}"
: "${RUN_NAME:?set RUN_NAME=<out_prefix>ppo_seed_<seed><out_suffix>}"
: "${OBS_TYPE:=obs_60_keys}"
: "${EVAL_MODE:=both}"
: "${N_EVAL:=100}"

module purge
module load anaconda3/2025.12
conda activate auctionnet

cd "$AUCTIONNET_ROOT/strategy_train_env"
mkdir -p slurm_out

python -u bidding_train_env/online/main_eval_ppo.py \
    --load_path "../output/online/training/ongoing/${RUN_NAME}" \
    --eval_mode "${EVAL_MODE}" --n_eval_episodes "${N_EVAL}" \
    --obs_type "${OBS_TYPE}" --act_type act_1_key \
    --bc_range default --eval_period 27 \
    --rl_data_dir "${AUCTIONNET_ROOT}/strategy_train_env/data/traffic/online_rl_data"
