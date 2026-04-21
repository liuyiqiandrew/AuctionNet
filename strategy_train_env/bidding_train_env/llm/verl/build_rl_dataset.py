"""Build VeRL train/val prompt parquets for the AuctionNet LLM bidding agent.

One row per (period, advertiser). Train = periods 7..26, val = period 27.

Each row stores the full initial chat needed by the async VLLM completion
callback:
- system prompt
- tick-0 user state
- a small machine-readable metadata header so the callback can reconstruct the
  AuctionNet replay environment for this advertiser/period.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Let this file be run as `python -m ...` from the package root or as a plain
# script from anywhere: push the repo's strategy_train_env onto sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bidding_train_env.llm.prompt import SYSTEM_PROMPT, build_user_message
from bidding_train_env.online.definitions import EPISODE_LENGTH, RL_DATA_DIR, load_act_keys, load_obs_keys
from bidding_train_env.online.online_env import EnvironmentFactory

TRAIN_PERIODS = list(range(7, 27))  # 7..26 inclusive
VAL_PERIODS = [27]

OUT_DIR = Path(__file__).resolve().parent / "data"
META_TAG = "auction_meta"


def _build_env(period: int, obs_keys: list, act_keys: list):
    return EnvironmentFactory.create(
        env_name="BiddingEnv",
        pvalues_df_path=str(RL_DATA_DIR / f"period-{period}_pvalues.parquet"),
        bids_df_path=str(RL_DATA_DIR / f"period-{period}_bids.parquet"),
        constraints_df_path=str(RL_DATA_DIR / f"period-{period}_constraints.parquet"),
        obs_keys=obs_keys,
        act_keys=act_keys,
        budget_range=None,
        target_cpa_range=None,
        seed=0,
    )


def _state_dict(env):
    inner = env.unwrapped
    pvalues, sigma = inner._get_pvalues_and_sigma()
    return inner._get_state_dict(pvalues, sigma)


def _with_meta(user_content: str, meta: dict) -> str:
    meta_json = json.dumps(meta, separators=(",", ":"), sort_keys=True)
    return f"<{META_TAG}>{meta_json}</{META_TAG}>\n{user_content}"


def _build(periods: list[int]) -> pd.DataFrame:
    obs_keys = load_obs_keys("obs_16_keys")
    act_keys = load_act_keys("act_1_key")
    rows: list[dict] = []
    for p in periods:
        cons_path = RL_DATA_DIR / f"period-{p}_constraints.parquet"
        if not cons_path.exists():
            raise FileNotFoundError(f"missing constraints parquet: {cons_path}")
        cons = pd.read_parquet(cons_path)
        cons = cons[cons.deliveryPeriodIndex == p]
        env = _build_env(p, obs_keys, act_keys)
        for _, r in cons.iterrows():
            advertiser = int(r.advertiserNumber)
            budget = float(r.budget)
            target_cpa = float(r.CPAConstraint)
            meta = {
                "period": int(p),
                "advertiser": advertiser,
                "budget": budget,
                "target_cpa": target_cpa,
                # Fixed per-row seed keeps the environment deterministic across
                # GRPO group samples for the same prompt.
                "seed": int(p * 1000 + advertiser),
            }
            env.reset()
            env.unwrapped.set_campaign(
                advertiser=advertiser,
                budget=budget,
                target_cpa=target_cpa,
                period=int(p),
            )
            initial_user = build_user_message(_state_dict(env), tick=0, episode_length=EPISODE_LENGTH)
            rows.append(
                {
                    "data_source": "auctionnet_bidding",
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _with_meta(initial_user["content"], meta)},
                    ],
                    "ability": "auto_bidding",
                    "reward_model": {"style": "env", "ground_truth": ""},
                    "extra_info": meta,
                }
            )
        env.close()
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = _build(TRAIN_PERIODS)
    val = _build(VAL_PERIODS)
    train.to_parquet(OUT_DIR / "train.parquet")
    val.to_parquet(OUT_DIR / "val.parquet")
    print(f"[build_rl_dataset] train rows={len(train)} -> {OUT_DIR / 'train.parquet'}")
    print(f"[build_rl_dataset] val   rows={len(val)}   -> {OUT_DIR / 'val.parquet'}")


if __name__ == "__main__":
    main()
