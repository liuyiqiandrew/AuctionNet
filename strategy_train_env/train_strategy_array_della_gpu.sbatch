#!/bin/bash
#SBATCH --job-name=strategy_train_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=3-4
#SBATCH --output=/scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet/strategy_train_env/slurm_out/train-%A_%a.out

set -eo pipefail

## Array index mapping:
## 0 -> iql
## 1 -> bc
## 2 -> bcq
## 3 -> cql
## 4 -> td3_bc
## 5 -> dt
## 6 -> onlinelp
ALGORITHMS=(iql bc bcq cql td3_bc dt onlinelp)

ALGO="${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}"

case "${ALGO}" in
    iql)
        ENTRYPOINT="main/main_iql.py"
        ;;
    bc)
        ENTRYPOINT="main/main_bc.py"
        ;;
    bcq)
        ENTRYPOINT="main/main_bcq.py"
        ;;
    cql)
        ENTRYPOINT="main/main_cql.py"
        ;;
    td3_bc)
        ENTRYPOINT="main/main_td3_bc.py"
        ;;
    dt)
        ENTRYPOINT="main/main_decision_transformer.py"
        ;;
    onlinelp)
        ENTRYPOINT="main/main_onlineLp.py"
        ;;
    *)
        echo "Unsupported algorithm: ${ALGO}" >&2
        exit 2
        ;;
esac

cd /scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet/strategy_train_env

module purge
module load anaconda3/2025.6
conda activate rl

echo "Array task id: ${SLURM_ARRAY_TASK_ID}"
echo "Training baseline: ${ALGO}"
echo "Entry point: ${ENTRYPOINT}"
echo "Working directory: $(pwd)"
echo "Host: $(hostname)"

srun python "${ENTRYPOINT}"
