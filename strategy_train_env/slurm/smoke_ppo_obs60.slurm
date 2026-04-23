#!/bin/bash
#SBATCH --job-name=ppo_smoke_obs60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --partition=class
#SBATCH --output=slurm_out/ppo_smoke_obs60-%j.out
#SBATCH --error=slurm_out/ppo_smoke_obs60-%j.err

: "${AUCTIONNET_ROOT:=/scratch/network/$USER/AuctionNet}"

module purge
module load anaconda3/2025.12
conda activate auctionnet

cd "$AUCTIONNET_ROOT/strategy_train_env"
mkdir -p slurm_out

python -u bidding_train_env/online/main_train_ppo.py \
    --num_envs 4 --num_steps 100000 --batch_size 128 --n_rollout_steps 128 \
    --seed 0 --bc_range default \
    --obs_type obs_60_keys --act_type act_1_key \
    --learning_rate 1e-4 --save_every 10000 \
    --out_prefix 000_smoke_ --out_suffix _ppo_default_obs60 \
    --device cpu \
    --rl_data_dir "${AUCTIONNET_ROOT}/strategy_train_env/data/traffic/online_rl_data"
