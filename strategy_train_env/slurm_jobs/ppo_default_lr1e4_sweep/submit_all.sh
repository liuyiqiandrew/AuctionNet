#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/scratch/gpfs/SIMONSOBS/users/yl9946/scratch/AuctionNet/strategy_train_env/slurm_out/ppo_default_lr1e4_sweep"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]"
    exit 2
fi

mkdir -p "${LOG_DIR}"

for job in "${SCRIPT_DIR}"/*.sbatch; do
    if (( dry_run )); then
        printf 'sbatch %q\n' "${job}"
    else
        sbatch "${job}"
    fi
done
