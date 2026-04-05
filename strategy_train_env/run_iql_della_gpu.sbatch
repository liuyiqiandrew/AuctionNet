#!/bin/bash
#SBATCH --job-name=iql_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -eo pipefail

cd /scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet/strategy_train_env

module purge
module load anaconda3/2025.6
conda activate rl

srun python main/main_iql.py
