#!/bin/bash
#SBATCH --job-name=regime_v1_10m_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/regime_v1_10m_all-%j.out
#SBATCH --error=slurm_out/regime_v1_10m_all-%j.err

# Phase 2A canonical training run for obs_market_regime_v1 (Bonnie's extension).
# Matches the team's baseline_ppo_obs60_10m.slurm config exactly except for
# --obs_type, so eval comparisons are apples-to-apples.
#
# Submit (from repo root):
#   sbatch strategy_train_env/slurm/regime_v1_10m_all.slurm
#
# Resume: just resubmit; main_train_ppo.py auto-resumes from latest checkpoint
# in the run dir.

: "${AUCTIONNET_ROOT:=/scratch/network/$USER/AuctionNet}"

module purge
module load anaconda3/2025.6
conda activate auctionnet

cd "$AUCTIONNET_ROOT/strategy_train_env"
mkdir -p slurm_out

python -u bidding_train_env/online/main_train_ppo.py \
    --num_envs 20 --num_steps 10000000 --batch_size 512 \
    --seed 0 --bc_range default \
    --obs_type obs_market_regime_v1 --act_type act_1_key \
    --learning_rate 1e-4 --save_every 1000000 \
    --out_prefix 030_regime_v1_ --out_suffix _10m_all \
    --device cpu \
    --rl_data_dir "${AUCTIONNET_ROOT}/strategy_train_env/data/traffic/online_rl_data"
