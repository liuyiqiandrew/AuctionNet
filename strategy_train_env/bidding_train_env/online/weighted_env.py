"""BiddingEnv variant that samples advertisers non-uniformly by historical score."""

from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium.envs.registration import register

from bidding_train_env.online.online_env import BiddingEnv


class WeightedBiddingEnv(BiddingEnv):
    def __init__(self, *args, advertiser_weights: np.ndarray | list, **kwargs):
        self._advertiser_weights = np.asarray(advertiser_weights, dtype=np.float64)
        super().__init__(*args, **kwargs)
        if len(self._advertiser_weights) != len(self.advertiser_list):
            raise ValueError(
                f"advertiser_weights len ({len(self._advertiser_weights)}) "
                f"!= advertiser_list len ({len(self.advertiser_list)})"
            )

    def _reset_campaign_params(self, advertiser=None, budget=None, target_cpa=None, period=None):
        if advertiser is None:
            advertiser = self.np_random.choice(self.advertiser_list, p=self._advertiser_weights)
        super()._reset_campaign_params(
            advertiser=advertiser, budget=budget, target_cpa=target_cpa, period=period
        )


try:
    register(id="WeightedBiddingEnv-v0", entry_point=f"{__name__}:WeightedBiddingEnv")
except gym.error.Error:
    pass
