"""Replay-based bidding environment for PPO training.

Replays logged auction data from per-period CSVs. Each episode corresponds to
one advertiser's 48-tick auction sequence. The agent outputs a scalar alpha;
bids are alpha * pValues; wins are determined by comparing against the logged
leastWinningCost. This is orders of magnitude faster than running the full
48-agent live simulator.
"""

import math
import os
import logging

import numpy as np

from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv
from bidding_train_env.baseline.ppo.state_builder import (
    build_state, apply_normalize, NUM_TICK,
)

logger = logging.getLogger(__name__)


class BiddingEnv:
    """Replay environment for PPO training.

    At construction, loads all (period, advertiser) episodes from the specified
    periods and caches the per-tick numpy arrays. Each reset() samples a random
    episode; each step() replays one tick of that episode.
    """

    def __init__(self, periods, data_dir, normalize_dict):
        self._normalize_dict = normalize_dict
        self._offline_env = OfflineEnv()
        self._rng = np.random.default_rng()

        # Pre-extract all episode data into a compact dict.
        self._episodes = {}
        for p in periods:
            csv_path = os.path.join(data_dir, f"period-{p}.csv")
            if not os.path.exists(csv_path):
                logger.warning(f"Skipping missing file: {csv_path}")
                continue
            loader = TestDataLoader(file_path=csv_path)
            for key in loader.keys:
                num_ticks, pValues, pValueSigmas, lwc = loader.mock_data(key)
                row = loader.test_dict[key].iloc[0]
                budget = float(row["budget"])
                cpa_constraint = float(row["CPAConstraint"])
                self._episodes[key] = (
                    num_ticks, pValues, pValueSigmas, lwc, budget, cpa_constraint,
                )
            # Free the raw DataFrame.
            del loader
        self._keys = list(self._episodes.keys())
        if not self._keys:
            raise RuntimeError(f"No episode data loaded from periods {periods}")
        print(f"[BiddingEnv] Loaded {len(self._keys)} episodes "
              f"from {len(periods)} periods")

        # Episode state (set by reset).
        self._tick = 0
        self._num_ticks = 0
        self._pValues = None
        self._pValueSigmas = None
        self._lwc = None
        self._budget = 0.0
        self._cpa_constraint = 0.0
        self._remaining_budget = 0.0
        self._history_pv_info = []
        self._history_bid = []
        self._history_auc = []
        self._history_imp = []
        self._history_lwc = []
        self._total_cost = 0.0
        self._total_conversions = 0.0
        self._total_value = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, key=None):
        """Start a new episode. Returns the initial 16-dim observation."""
        if key is None:
            key = self._keys[self._rng.integers(len(self._keys))]
        ep = self._episodes[key]
        self._num_ticks = ep[0]
        self._pValues = ep[1]
        self._pValueSigmas = ep[2]
        self._lwc = ep[3]
        self._budget = ep[4]
        self._cpa_constraint = ep[5]

        self._tick = 0
        self._remaining_budget = self._budget
        self._history_pv_info = []
        self._history_bid = []
        self._history_auc = []
        self._history_imp = []
        self._history_lwc = []
        self._total_cost = 0.0
        self._total_conversions = 0.0
        self._total_value = 0.0

        obs = build_state(
            0, self._pValues[0],
            self._history_pv_info, self._history_bid,
            self._history_auc, self._history_imp,
            self._history_lwc, self._budget, self._remaining_budget,
        )
        return apply_normalize(obs, self._normalize_dict)

    def step(self, alpha):
        """Execute one tick. Returns (next_obs, reward, done, info)."""
        t = self._tick
        pv = self._pValues[t]
        sigma = self._pValueSigmas[t]
        lwc = self._lwc[t]

        # Compute bids from policy output.
        bids = np.maximum(float(alpha) * pv, 0.0)

        # Simulate auction.
        tick_value, tick_cost, tick_status, tick_conversion = (
            self._offline_env.simulate_ad_bidding(pv, sigma, bids, lwc)
        )

        # Budget enforcement: drop random wins until cost fits.
        while tick_cost.sum() > self._remaining_budget:
            ratio = max(
                (tick_cost.sum() - self._remaining_budget)
                / (tick_cost.sum() + 1e-4),
                0,
            )
            won_idx = np.where(tick_status)[0]
            if won_idx.size == 0:
                break
            n_drop = max(1, math.ceil(won_idx.size * ratio))
            drop = self._rng.choice(won_idx, n_drop, replace=False)
            bids[drop] = 0
            tick_value, tick_cost, tick_status, tick_conversion = (
                self._offline_env.simulate_ad_bidding(pv, sigma, bids, lwc)
            )

        # Tick aggregates.
        cost_this_tick = float(tick_cost.sum())
        value_this_tick = float((pv * tick_status).sum())
        conv_this_tick = float(tick_conversion.sum())

        # Update accumulators.
        self._remaining_budget -= cost_this_tick
        self._total_cost += cost_this_tick
        self._total_value += value_this_tick
        self._total_conversions += conv_this_tick

        # Build history arrays (must match run_evaluate.py format).
        self._history_pv_info.append(np.column_stack([pv, sigma]))
        self._history_bid.append(bids)
        self._history_auc.append(
            np.column_stack([tick_status.astype(float),
                             tick_status.astype(float),
                             tick_cost])
        )
        self._history_imp.append(
            np.column_stack([tick_conversion.astype(float),
                             tick_conversion.astype(float)])
        )
        self._history_lwc.append(lwc)

        # Advance tick and check termination.
        self._tick += 1
        done = (
            self._tick >= self._num_ticks
            or self._remaining_budget < self._offline_env.min_remaining_budget
        )

        # Build next observation.
        if not done:
            next_obs = build_state(
                self._tick, self._pValues[self._tick],
                self._history_pv_info, self._history_bid,
                self._history_auc, self._history_imp,
                self._history_lwc, self._budget, self._remaining_budget,
            )
            next_obs = apply_normalize(next_obs, self._normalize_dict)
        else:
            next_obs = np.zeros(16, dtype=np.float32)

        info = {
            "cost": cost_this_tick,
            "conversions": conv_this_tick,
            "value": value_this_tick,
            "total_cost": self._total_cost,
            "total_conversions": self._total_conversions,
            "total_value": self._total_value,
            "remaining_budget": self._remaining_budget,
            "budget": self._budget,
            "cpa_constraint": self._cpa_constraint,
            "num_ticks": self._tick,
        }
        return next_obs, value_this_tick, done, info
