#!/bin/bash
#SBATCH --job-name=eval_regime_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_out/eval_regime_v1-%j.out
#SBATCH --error=slurm_out/eval_regime_v1-%j.err

# Evaluate the obs_market_regime_v1 PPO checkpoint on period 27.
# Runs both `random` (100 sampled campaigns) and `sweep` (one episode per
# advertiser using raw budget/CPAConstraint) modes.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${AUCTIONNET_ROOT:=$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

module purge
module load anaconda3/2025.6
conda activate auctionnet

cd "$AUCTIONNET_ROOT"
mkdir -p slurm_out

python -u ppo/main_eval_ppo.py \
    --load_path output/ppo/training/ongoing/030_regime_v1_ppo_seed_0_10m_all \
    --eval_mode both \
    --n_eval_episodes 100 \
    --eval_period 27 \
    --obs_type obs_market_regime_v1 \
    --act_type act_1_key \
    --bc_range default \
    --rl_data_dir "${AUCTIONNET_ROOT}/data/traffic/online_rl_data"
