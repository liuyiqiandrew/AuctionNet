#!/bin/bash
#SBATCH --job-name=prep_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=02:00:00
#SBATCH --partition=class
#SBATCH --output=slurm_out/prep_data-%j.out
#SBATCH --error=slurm_out/prep_data-%j.err

# One-time data preparation:
#   1. Unzip raw period parquets into strategy_train_env/data/traffic/
#   2. Run prepare_data.py to derive online_rl_data/period-{N}_{pvalues,bids,constraints}.parquet
#
# Skips work that's already done (idempotent).

: "${AUCTIONNET_ROOT:=/scratch/network/$USER/AuctionNet}"

module purge
module load anaconda3/2025.12
conda activate auctionnet

cd "$AUCTIONNET_ROOT"
mkdir -p slurm_out strategy_train_env/data/traffic

ZIP=specs/Data/raw_data_period_7_to_27.zip
RAW_DIR=strategy_train_env/data/traffic

if [[ ! -f "$RAW_DIR/period-7.parquet" ]]; then
    echo "[prep_data] Unzipping $ZIP -> $RAW_DIR"
    unzip -n -d "$RAW_DIR" "$ZIP"
else
    echo "[prep_data] Raw parquets already present in $RAW_DIR; skipping unzip"
fi

cd strategy_train_env
mkdir -p slurm_out

RL_DIR=data/traffic/online_rl_data
if [[ ! -f "$RL_DIR/period-7_pvalues.parquet" ]]; then
    echo "[prep_data] Running prepare_data.py for periods 7..27"
    python -u bidding_train_env/online/prepare_data.py --first_period 7 --last_period 27
else
    echo "[prep_data] $RL_DIR already populated; skipping prepare_data.py"
fi

echo "[prep_data] Done. Files in $RL_DIR:"
ls "$RL_DIR" | head -20
