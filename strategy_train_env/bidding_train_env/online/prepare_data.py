"""CLI entry point: per-period raw parquet -> (pvalues, bids, constraints) parquets.

Usage (from AuctionNet/strategy_train_env/):
    python bidding_train_env/online/prepare_data.py --first_period 7 --last_period 27
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from bidding_train_env.online.data_generator import (
    generate_advertiser_constraints_df,
    generate_bids_df,
    generate_pvalue_df,
)
from bidding_train_env.online.definitions import RAW_DATA_DIR, RL_DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_period", type=int, default=7)
    parser.add_argument("--last_period", type=int, default=27)
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--out_dir", type=str, default=str(RL_DATA_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in range(args.first_period, args.last_period + 1):
        src = raw_dir / f"period-{p}.parquet"
        pv_out = out_dir / f"period-{p}_pvalues.parquet"
        bd_out = out_dir / f"period-{p}_bids.parquet"
        ac_out = out_dir / f"period-{p}_constraints.parquet"

        if pv_out.exists() and bd_out.exists() and ac_out.exists():
            print(f"[skip] period {p}: all outputs exist")
            continue

        t0 = time.time()
        print(f"[load] {src}")
        data = pd.read_parquet(src)

        if not pv_out.exists():
            pv_df = generate_pvalue_df(data)
            pv_df.to_parquet(pv_out)
            print(f"  -> {pv_out.name} ({len(pv_df)} rows)")
        if not bd_out.exists():
            bd_df = generate_bids_df(data)
            bd_df.to_parquet(bd_out)
            print(f"  -> {bd_out.name} ({len(bd_df)} rows)")
        if not ac_out.exists():
            ac_df = generate_advertiser_constraints_df(data)
            ac_df.to_parquet(ac_out)
            print(f"  -> {ac_out.name} ({len(ac_df)} rows)")

        del data
        gc.collect()
        print(f"  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
