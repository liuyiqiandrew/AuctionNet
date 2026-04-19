"""Shared constants, paths, observation/action key schemas, and PPO defaults."""

import json
from pathlib import Path

from stable_baselines3 import PPO

ONLINE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = ONLINE_DIR / "configs"

AUCTIONNET_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = AUCTIONNET_ROOT / "strategy_train_env" / "data" / "traffic"
RL_DATA_DIR = RAW_DATA_DIR / "online_rl_data"
OUTPUT_DIR = AUCTIONNET_ROOT / "output" / "online"

EPISODE_LENGTH = 48

# Pluggable algorithm registry. Add custom variants here.
ALGO_CLASS_DICT = {"ppo": PPO}

# Budget / target_cpa sampling ranges by name.
# "default" is a sentinel — env reads raw per-row (budget, CPAConstraint) from parquet.
BC_RANGES = {
    "dense":   dict(budget_range=(400, 12000), target_cpa_range=(6, 12)),
    "sparse":  dict(budget_range=(1000, 6000), target_cpa_range=(50, 150)),
    "default": dict(budget_range=None,         target_cpa_range=None),
}

# Known-stable PPO hyperparams (ported from oil's config).
DEFAULT_PPO_KWARGS = dict(
    learning_rate=2e-5, clip_range=0.3, gamma=0.99, gae_lambda=0.9,
    ent_coef=3e-6, vf_coef=0.5, max_grad_norm=0.7, n_epochs=10,
    n_steps=128, batch_size=256,
)

# Per-timestep aggregates tracked in history_info. Used for last_/last_three_/historical_ views.
HISTORY_KEYS = [
    "least_winning_cost_mean",
    "least_winning_cost_10_pct",
    "least_winning_cost_01_pct",
    "cpa_exceedence_rate",
    "pvalues_mean",
    "conversion_mean",
    "conversion_count",
    "bid_success_mean",
    "successful_bid_position_mean",
    "bid_over_lwc_mean",
    "pv_over_lwc_mean",
    "pv_over_lwc_90_pct",
    "pv_over_lwc_99_pct",
    "pv_num",
    "exposure_count",
    "cost_sum",
]

# Per-slot aggregates (one entry per slot 1..3 plus overall).
HISTORY_AND_SLOT_KEYS = [
    "bid_mean",
    "cost_mean",
    "bid_success_count",
    "exposure_mean",
]

# Scalar state entries that don't use a history-based view.
NO_HISTORY_KEYS = [
    "time_left",
    "budget_left",
    "budget",
    "cpa",
    "category",
    "total_conversions",
    "total_cost",
    "total_cpa",
    "current_pvalues_mean",
    "current_pvalues_90_pct",
    "current_pvalues_99_pct",
    "current_pv_num",
]

DEFAULT_RWD_WEIGHTS = {"dense": 1.0, "sparse": 0.0}

# Scalar info-dict keys the training callback averages per-rollout. Mix of per-step
# keys (present every step) and terminal-only keys (filtered by None in the callback).
INFO_KEYWORDS = (
    "conversions", "cost", "cpa", "target_cpa", "budget",
    "avg_pvalues", "score_over_pvalue", "score_over_budget", "score_over_cpa",
    "cost_over_budget", "target_cpa_over_cpa", "score",
    "sparse", "dense", "action", "bid", "overspend",
)


def load_obs_keys(obs_type: str) -> list:
    with open(CONFIGS_DIR / f"{obs_type}.json") as f:
        return json.load(f)


def load_act_keys(act_type: str) -> list:
    with open(CONFIGS_DIR / f"{act_type}.json") as f:
        return json.load(f)
