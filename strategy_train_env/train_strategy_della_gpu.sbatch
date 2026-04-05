#!/bin/bash
#SBATCH --job-name=strategy_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet/strategy_train_env/slurm_out/train-%j.out

set -eo pipefail

ALGO="${1:-iql}"

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
    dt|decision_transformer)
        ENTRYPOINT="main/main_decision_transformer.py"
        ;;
    onlinelp|online_lp)
        ENTRYPOINT="main/main_onlineLp.py"
        ;;
    *)
        echo "Unsupported algorithm: ${ALGO}" >&2
        echo "Supported values: iql, bc, bcq, cql, td3_bc, dt, onlinelp" >&2
        exit 2
        ;;
esac

cd /scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet/strategy_train_env

module purge
module load anaconda3/2025.6
conda activate rl

echo "Training baseline: ${ALGO}"
echo "Entry point: ${ENTRYPOINT}"
echo "Working directory: $(pwd)"
echo "Host: $(hostname)"

srun python "${ENTRYPOINT}"
