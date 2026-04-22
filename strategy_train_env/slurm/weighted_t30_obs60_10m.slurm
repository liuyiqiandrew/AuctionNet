#!/bin/bash
#SBATCH --job-name=ppo_weighted_t30_obs60_10m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm_out/ppo_weighted_t30_obs60_10m-%j.out
#SBATCH --error=slurm_out/ppo_weighted_t30_obs60_10m-%j.err

# Override AUCTIONNET_ROOT if your checkout lives elsewhere:
#   AUCTIONNET_ROOT=/your/path sbatch slurm/weighted_t30_obs60_10m.slurm
: "${AUCTIONNET_ROOT:=/scratch/network/$USER/AuctionNet}"

module load anaconda3/2025.12
conda activate auctionnet

cd "$AUCTIONNET_ROOT/strategy_train_env"
mkdir -p slurm_out

python bidding_train_env/online/main_train_ppo_weighted.py \
    --num_envs 20 --num_steps 10000000 --batch_size 512 \
    --seed 0 --bc_range default \
    --obs_type obs_60_keys --act_type act_1_key \
    --learning_rate 2e-5 --save_every 10000 \
    --temperature 30 --alpha 0.9 \
    --out_prefix 004_t30_ --out_suffix _ppo_default_obs60 \
    --device cpu --use_dummy_vec_env
