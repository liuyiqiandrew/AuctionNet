#!/bin/bash
#SBATCH --job-name=ppo_baseline_obs60_10m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --partition=class
#SBATCH --output=slurm_out/ppo_baseline_obs60_10m-%j.out
#SBATCH --error=slurm_out/ppo_baseline_obs60_10m-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=bl0919@princeton.edu

# Override AUCTIONNET_ROOT if your checkout lives elsewhere:
#   AUCTIONNET_ROOT=/your/path sbatch ppo/slurm/baseline_ppo_obs60_10m.slurm
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${AUCTIONNET_ROOT:=$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

module purge
module load anaconda3/2025.12
conda activate auctionnet

cd "$AUCTIONNET_ROOT"
mkdir -p slurm_out

python -u ppo/main_train_ppo.py \
    --num_envs 20 --num_steps 10000000 --batch_size 512 \
    --seed 0 --bc_range default \
    --obs_type obs_60_keys --act_type act_1_key \
    --learning_rate 1e-4 --save_every 100000 \
    --out_prefix 001_baseline_ --out_suffix _ppo_default_obs60 \
    --device cpu \
    --rl_data_dir "${AUCTIONNET_ROOT}/data/traffic/online_rl_data"
