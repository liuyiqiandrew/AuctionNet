#!/bin/bash
#SBATCH --job-name=auctionnet_main_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet/strategy_train_env/slurm_out/eval-%j.out

set -eo pipefail

cd /scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet

module purge
module load anaconda3/2025.6
conda activate rl

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"
echo "Host: $(hostname)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Start time: $(date)"

srun python main_test.py
