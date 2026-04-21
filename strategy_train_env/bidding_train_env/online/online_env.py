"""Gymnasium bidding environment for the AuctionNet online PPO pipeline.

Replays one advertiser's bidding problem over an episode of 48 timesteps.
Per step, the policy emits a log-bid-coefficient; the env computes per-impression
bids, settles a second-price auction against the pre-aggregated top-3 competing
bids, drops winners to stay within budget, samples conversions, and returns a
CPA-clipped reward.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd

from bidding_train_env.online.definitions import (
    DEFAULT_RWD_WEIGHTS,
    EPISODE_LENGTH,
    HISTORY_AND_SLOT_KEYS,
    HISTORY_KEYS,
    NO_HISTORY_KEYS,
)
from bidding_train_env.online.helpers import safe_mean


class BiddingEnv(gym.Env):
    EPS = 1e-6

    def __init__(
        self,
        pvalues_df_path: str,
        bids_df_path: str,
        constraints_df_path: str | None = None,
        obs_keys: list | None = None,
        act_keys: list | None = None,
        rwd_weights: dict | None = None,
        budget_range: tuple | None = (400, 12000),
        target_cpa_range: tuple | None = (6, 12),
        deterministic_conversion: bool = False,
        temporal_seq_len: int = 1,
        seed: int = 0,
    ):
        if obs_keys is None or act_keys is None:
            raise ValueError("obs_keys and act_keys are required")
        self.obs_keys = obs_keys
        self.act_keys = act_keys
        self.obs_dim = len(obs_keys)
        self.temporal_seq_len = int(temporal_seq_len)
        if self.temporal_seq_len < 1:
            raise ValueError(f"temporal_seq_len must be >= 1, got {temporal_seq_len}")
        self._state_history: list[np.ndarray] = []
        self.pvalues_key_pos = self.act_keys.index("pvalue")
        self.rwd_weights = rwd_weights if rwd_weights is not None else DEFAULT_RWD_WEIGHTS
        self.deterministic_conversion = deterministic_conversion

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim * self.temporal_seq_len,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-10, high=10, shape=(len(act_keys),), dtype=np.float32
        )

        self.pvalues_df = self._load_pvalues_df(pvalues_df_path)
        self.bids_df = self._load_bids_df(bids_df_path)
        # "default" bc_range mode: sample raw per-row budget/CPAConstraint at reset.
        self.constraints_df = (
            pd.read_parquet(constraints_df_path) if (budget_range is None and constraints_df_path) else None
        )

        self.episode_length = EPISODE_LENGTH
        self.budget_range = budget_range
        self.target_cpa_range = target_cpa_range

        self.advertiser_list = list(self.pvalues_df.advertiserNumber.unique())
        self.period_list = list(self.pvalues_df.deliveryPeriodIndex.unique())
        cat = self.pvalues_df.groupby("advertiserNumber").advertiserCategoryIndex.first()
        self.advertiser_category_dict = dict(zip(cat.index, cat.values))

        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_campaign_params()
        pvalues, sigma = self._get_pvalues_and_sigma()
        state = self._get_state(self._get_state_dict(pvalues, sigma))
        return state, {}

    def step(self, action):
        pvalues, sigma = self._get_pvalues_and_sigma()
        bid_coef = self._compute_bid_coef(action, pvalues, sigma)
        advertiser_bids = bid_coef * self.target_cpa
        return self._place_bids(advertiser_bids, pvalues, sigma)

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #

    def _reset_campaign_params(self, advertiser=None, budget=None, target_cpa=None, period=None):
        self.advertiser = advertiser if advertiser is not None else self.np_random.choice(self.advertiser_list)
        self.period = period if period is not None else self.np_random.choice(self.period_list)
        if budget is not None and target_cpa is not None:
            self.total_budget = float(budget)
            self.target_cpa = float(target_cpa)
        elif self.budget_range is not None:
            self.total_budget = float(self.np_random.uniform(*self.budget_range))
            self.target_cpa = float(self.np_random.uniform(*self.target_cpa_range))
        else:
            row = self.constraints_df[
                (self.constraints_df.deliveryPeriodIndex == self.period)
                & (self.constraints_df.advertiserNumber == self.advertiser)
            ]
            if len(row) == 0:
                raise RuntimeError(
                    f"No constraints for advertiser={self.advertiser} in period={self.period}. "
                    "Regenerate constraints parquet or use a non-default bc_range."
                )
            self.total_budget = float(row.budget.iloc[0])
            self.target_cpa = float(row.CPAConstraint.iloc[0])

        self.time_step = 0
        self.remaining_budget = self.total_budget
        self.total_conversions = 0.0
        self.total_cost = 0.0
        self.total_cpa = 0.0
        self.pv_num_total = 0

        self.history_info = {k: [] for k in HISTORY_KEYS + NO_HISTORY_KEYS + HISTORY_AND_SLOT_KEYS}
        for k in HISTORY_AND_SLOT_KEYS:
            for slot in range(1, 4):
                self.history_info[f"{k}_slot_{slot}"] = []

        self.episode_bids_df = self._get_episode_bids_df()
        self.episode_pvalues_df = self._get_episode_pvalues_df()
        self._state_history = []

    def set_campaign(self, advertiser: int, budget: float, target_cpa: float, period: int | None = None):
        """For eval sweep mode: override sampled params without re-seeding RNG."""
        self._reset_campaign_params(
            advertiser=advertiser, budget=budget, target_cpa=target_cpa, period=period
        )
        pvalues, sigma = self._get_pvalues_and_sigma()
        return self._get_state(self._get_state_dict(pvalues, sigma))

    # ------------------------------------------------------------------ #
    # Bidding pipeline
    # ------------------------------------------------------------------ #

    def _compute_bid_coef(self, action, pvalues, pvalues_sigma):
        action = np.atleast_2d(np.asarray(action, dtype=np.float32)).copy()
        action[:, self.pvalues_key_pos] = np.exp(action[:, self.pvalues_key_pos])
        bid_basis = self._get_bid_basis(pvalues, pvalues_sigma)
        # Broadcast scalar-per-key action over all impressions.
        if action.shape[0] == 1:
            bid_coef = np.clip(np.einsum("k,nk->n", action[0], bid_basis), 0, np.inf)
        else:
            bid_coef = np.clip(np.einsum("nk,nk->n", action, bid_basis), 0, np.inf)
        return bid_coef

    def _get_bid_basis(self, pvalues, pvalues_sigma):
        basis_dict = {"pvalue": pvalues, "pvalue_sigma": pvalues_sigma}
        return np.stack([basis_dict[k] for k in self.act_keys], axis=1)

    def _place_bids(self, advertiser_bids, pvalues, pvalues_sigma):
        bid_row = self.episode_bids_df[self.episode_bids_df.timeStepIndex == self.time_step].iloc[0]
        top_bids = np.asarray(bid_row.bid)
        top_bids_cost = np.asarray(bid_row.cost)
        top_bids_exposed = np.asarray(bid_row.isExposed)
        least_winning_cost = top_bids_cost[:, 0]

        bid_success, bid_position, bid_exposed, bid_cost = self._compute_success_exposition_cost(
            advertiser_bids, top_bids, top_bids_cost, top_bids_exposed
        )
        bid_success, bid_position, bid_exposed, bid_cost = self._handle_overcost(
            bid_success, bid_position, bid_exposed, bid_cost,
            advertiser_bids, top_bids, top_bids_cost, top_bids_exposed,
        )

        # Conversion: stochastic by default. Both pValue AND pValueSigma matter —
        # p_sampled is drawn from N(pvalues, sigma) clipped to [0,1], then a Bernoulli
        # draw gated by exposure gives the binary conversion outcome.
        if self.deterministic_conversion:
            bid_conversion = pvalues * bid_exposed
        else:
            p_sampled = np.clip(self.np_random.normal(pvalues, pvalues_sigma), 0.0, 1.0)
            bid_conversion = self.np_random.binomial(n=1, p=p_sampled) * bid_exposed

        # Update step-level cumulatives.
        self.time_step += 1
        step_cost = float(np.sum(bid_cost))
        step_conv = float(np.sum(bid_conversion))
        self.total_cost += step_cost
        self.total_conversions += step_conv
        self.total_cpa = self.total_cost / self.total_conversions if self.total_conversions > 0 else 0.0
        self.pv_num_total += len(pvalues)
        self.remaining_budget -= step_cost

        terminated = self.time_step >= self.episode_length
        dense_reward = self._compute_score(step_cost, step_conv)
        sparse_reward = self._compute_score(self.total_cost, self.total_conversions) if terminated else 0.0

        self._update_history(
            pvalues, advertiser_bids, least_winning_cost,
            bid_success, bid_position, bid_exposed, bid_cost, bid_conversion,
        )

        reward = self._compute_reward({"dense": dense_reward, "sparse": sparse_reward})

        info = {
            "dense": dense_reward,
            "sparse": sparse_reward,
            "bid": float(np.mean(advertiser_bids)) if len(advertiser_bids) else 0.0,
            "action": float(np.sum(advertiser_bids) / (np.sum(pvalues) + self.EPS) / self.target_cpa),
        }
        if terminated:
            cpa = self.total_cpa
            pv_hist = self.history_info["pvalues_mean"]
            avg_pvalues = float(np.mean(pv_hist)) if pv_hist else 0.0
            info.update(
                score=sparse_reward,
                conversions=self.total_conversions,
                cost=self.total_cost,
                cpa=cpa,
                target_cpa=self.target_cpa,
                budget=self.total_budget,
                avg_pvalues=avg_pvalues,
                score_over_pvalue=(sparse_reward / avg_pvalues) if avg_pvalues > 0 else 0.0,
                score_over_budget=(sparse_reward / self.total_budget) if self.total_budget > 0 else 0.0,
                score_over_cpa=(sparse_reward / cpa) if cpa > 0 else 0.0,
                cost_over_budget=self.total_cost / max(self.total_budget, self.EPS),
                target_cpa_over_cpa=(self.target_cpa / cpa) if cpa > 0 else 0.0,
                advertiser=int(self.advertiser),
                period=int(self.period),
            )
            # Episode-over: pvalues/sigma passed to next-state builder are unused.
            new_pv, new_sigma = pvalues, pvalues_sigma
        else:
            new_pv, new_sigma = self._get_pvalues_and_sigma()

        state = self._get_state(self._get_state_dict(new_pv, new_sigma))
        return state, reward, terminated, False, info

    def _compute_success_exposition_cost(self, advertiser_bids, top_bids, top_bids_cost, top_bids_exposed):
        higher = advertiser_bids[:, None] >= top_bids
        bid_success = higher.any(axis=1)
        bid_position = np.sum(higher, axis=1) - 1  # 0..2, with 2 = top slot

        bid_exposed = np.zeros_like(bid_position, dtype=np.float32)
        bid_exposed[bid_success] = top_bids_exposed[bid_success, bid_position[bid_success]]

        # Second-price: cost = cost of the slot we outbid.
        bid_cost = top_bids_cost[np.arange(len(bid_position)), bid_position] * bid_exposed
        return bid_success, bid_position, bid_exposed, bid_cost

    def _handle_overcost(self, bid_success, bid_position, bid_exposed, bid_cost,
                         advertiser_bids, top_bids, top_bids_cost, top_bids_exposed):
        total_cost = float(np.sum(bid_cost))
        advertiser_bids = advertiser_bids.copy()
        while total_cost > self.remaining_budget:
            over_ratio = (total_cost - self.remaining_budget) / total_cost
            winners = np.where(bid_success)[0]
            if winners.size == 0:
                break
            n_drop = int(np.ceil(winners.size * over_ratio))
            drop_idx = self.np_random.choice(winners, n_drop, replace=False)
            advertiser_bids[drop_idx] = 0
            bid_success, bid_position, bid_exposed, bid_cost = self._compute_success_exposition_cost(
                advertiser_bids, top_bids, top_bids_cost, top_bids_exposed
            )
            total_cost = float(np.sum(bid_cost))
        return bid_success, bid_position, bid_exposed, bid_cost

    def _compute_score(self, cost, conversions):
        if conversions <= 0:
            return 0.0
        cpa = cost / conversions
        if cpa <= 0:
            return 0.0
        cpa_coeff = min(1.0, (self.target_cpa / cpa) ** 2)
        return float(cpa_coeff * conversions)

    def _compute_reward(self, reward_dict):
        return sum(self.rwd_weights.get(k, 0.0) * v for k, v in reward_dict.items())

    # ------------------------------------------------------------------ #
    # History / state construction
    # ------------------------------------------------------------------ #

    def _update_history(self, pvalues, advertiser_bids, least_winning_cost,
                        bid_success, bid_position, bid_exposed, bid_cost, bid_conversion):
        eps = self.EPS
        update = {
            "least_winning_cost_mean": float(np.mean(least_winning_cost)),
            "least_winning_cost_10_pct": float(np.percentile(least_winning_cost, 10)),
            "least_winning_cost_01_pct": float(np.percentile(least_winning_cost, 1)),
            "cpa_exceedence_rate": (self.total_cpa - self.target_cpa) / self.target_cpa,
            "pvalues_mean": float(np.mean(pvalues)),
            "conversion_mean": float(np.mean(bid_conversion)),
            "conversion_count": float(np.sum(bid_conversion)),
            "bid_success_mean": float(np.mean(bid_success)),
            "successful_bid_position_mean": safe_mean(bid_position[bid_success]),
            "bid_over_lwc_mean": float(np.mean(advertiser_bids / (least_winning_cost + eps))),
            "pv_over_lwc_mean": float(np.mean(pvalues / (least_winning_cost + eps))),
            "pv_over_lwc_90_pct": float(np.percentile(pvalues / (least_winning_cost + eps), 90)),
            "pv_over_lwc_99_pct": float(np.percentile(pvalues / (least_winning_cost + eps), 99)),
            "pv_num": len(pvalues),
            "exposure_count": float(np.sum(bid_exposed)),
            "cost_sum": float(np.sum(bid_cost)),
        }
        # Per-slot splits (slots 1..3; 3 = top-slot, 1 = worst-winning slot).
        slot_src = {
            "bid_mean": (advertiser_bids, safe_mean, bid_success),
            "cost_mean": (bid_cost, safe_mean, bid_exposed.astype(bool)),
            "bid_success_count": (bid_success.astype(float), np.sum, True),
            "exposure_mean": (bid_exposed, safe_mean, True),
        }
        for key, (data, fn, cond) in slot_src.items():
            update[key] = float(fn(data))
            for slot in range(3):
                # bid_position=0 -> slot_3 (worst), bid_position=2 -> slot_1 (top)
                mask = (bid_position == slot)
                if cond is not True:
                    mask = np.logical_and(mask, cond)
                sel = data[mask]
                update[f"{key}_slot_{3 - slot}"] = float(fn(sel))

        for k, v in update.items():
            self.history_info[k].append(v)

    def _get_state_dict(self, pvalues, pvalues_sigma):
        state_dict = {
            "time_left": (self.episode_length - self.time_step) / self.episode_length,
            "budget_left": max(self.remaining_budget, 0) / max(self.total_budget, self.EPS),
            "budget": self.total_budget,
            "cpa": self.target_cpa,
            "category": float(self.advertiser_category_dict[self.advertiser]),
            "total_conversions": self.total_conversions,
            "total_cost": self.total_cost,
            "total_cpa": self.total_cpa,
            "pv_num_total": self.pv_num_total,
            "current_pvalues_mean": float(np.mean(pvalues)),
            "current_pvalues_90_pct": float(np.percentile(pvalues, 90)),
            "current_pvalues_99_pct": float(np.percentile(pvalues, 99)),
            "current_pv_num": len(pvalues),
        }
        for key, info in self.history_info.items():
            state_dict[f"last_{key}"] = safe_mean(info[-1:])
            state_dict[f"last_three_{key}"] = safe_mean(info[-3:])
            state_dict[f"historical_{key}"] = safe_mean(info)

        # last_three_pv_num: reported as a sum (not mean) for backward compat with obs_16.
        state_dict["last_three_pv_num"] = float(np.sum(self.history_info["pv_num"][-3:]))

        # Deprecated-key aliases so obs_60 (which uses some legacy names) resolves cleanly.
        _alias = {
            "least_winning_cost_mean": "historical_least_winning_cost_mean",
            "least_winning_cost_10_pct": "historical_least_winning_cost_10_pct",
            "least_winning_cost_01_pct": "historical_least_winning_cost_01_pct",
            "pvalues_mean": "historical_pvalues_mean",
            "conversion_mean": "historical_conversion_mean",
            "bid_success_mean": "historical_bid_success_mean",
            "last_bid_success": "last_bid_success_mean",
            "historical_cost_slot_1_mean": "historical_cost_mean_slot_1",
            "historical_cost_slot_2_mean": "historical_cost_mean_slot_2",
            "historical_cost_slot_3_mean": "historical_cost_mean_slot_3",
            "last_cost_slot_1_mean": "last_cost_mean_slot_1",
            "last_cost_slot_2_mean": "last_cost_mean_slot_2",
            "last_cost_slot_3_mean": "last_cost_mean_slot_3",
            "last_three_cost_slot_1_mean": "last_three_cost_mean_slot_1",
            "last_three_cost_slot_2_mean": "last_three_cost_mean_slot_2",
            "last_three_cost_slot_3_mean": "last_three_cost_mean_slot_3",
        }
        for dst, src in _alias.items():
            if src in state_dict:
                state_dict[dst] = state_dict[src]
        return state_dict

    def _get_state(self, state_dict):
        missing = [k for k in self.obs_keys if k not in state_dict]
        if missing:
            raise KeyError(f"obs_keys missing from state_dict: {missing}")
        base_state = np.asarray([state_dict[k] for k in self.obs_keys], dtype=np.float32)
        return self._get_temporal_state(base_state)

    def _get_temporal_state(self, base_state):
        if self.temporal_seq_len == 1:
            return base_state

        self._state_history.append(base_state.copy())
        if len(self._state_history) > self.temporal_seq_len:
            self._state_history = self._state_history[-self.temporal_seq_len:]

        pad_count = self.temporal_seq_len - len(self._state_history)
        padded_history = [self._state_history[0]] * pad_count + self._state_history
        return np.concatenate(padded_history).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Data loading / episode slicing
    # ------------------------------------------------------------------ #

    def _get_pvalues_and_sigma(self):
        row = self.episode_pvalues_df[self.episode_pvalues_df.timeStepIndex == self.time_step]
        return row.pValue.item(), row.pValueSigma.item()

    def _get_episode_bids_df(self):
        return self.bids_df[self.bids_df.deliveryPeriodIndex == self.period].copy()

    def _get_episode_pvalues_df(self):
        return self.pvalues_df[
            (self.pvalues_df.advertiserNumber == self.advertiser)
            & (self.pvalues_df.deliveryPeriodIndex == self.period)
        ].copy()

    def _load_pvalues_df(self, path):
        print(f"Loading pvalues from {path}")
        df = pd.read_parquet(path)
        df["pValue"] = df["pValue"].apply(np.array)
        df["pValueSigma"] = df["pValueSigma"].apply(np.array)
        return df

    def _load_bids_df(self, path):
        print(f"Loading bids from {path}")
        df = pd.read_parquet(path)
        df["bid"] = df["bid"].apply(np.stack)
        df["isExposed"] = df["isExposed"].apply(np.stack)
        df["cost"] = df["cost"].apply(np.stack)
        return df


# --------------------------------------------------------------------- #
# Gym registration + factory
# --------------------------------------------------------------------- #

from gymnasium.envs.registration import register  # noqa: E402

try:
    register(id="BiddingEnv-v0", entry_point=f"{__name__}:BiddingEnv")
except gym.error.Error:
    pass  # Already registered.


class EnvironmentFactory:
    """Static factory so callers can write `EnvironmentFactory.create(env_name="BiddingEnv", ...)`."""

    ENV_NAME_TO_ID = {"BiddingEnv": "BiddingEnv-v0"}

    @staticmethod
    def create(env_name: str, **kwargs):
        env_id = EnvironmentFactory.ENV_NAME_TO_ID.get(env_name, env_name)
        return gym.make(env_id, **kwargs)
