import os
import pickle
import numpy as np
import torch

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.baseline.ppo.state_builder import build_state, apply_normalize


class PpoBiddingStrategy(BaseBiddingStrategy):
    """PPO inference wrapper. Mirrors IqlBiddingStrategy but loads
    saved_model/PPOtest/ppo_model.pth and delegates state construction
    to the shared state_builder helpers."""

    def __init__(self, budget=100, name="Ppo-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        here = os.path.dirname(os.path.realpath(__file__))
        root = os.path.dirname(os.path.dirname(here))  # strategy_train_env/
        model_path = os.path.join(root, "saved_model", "PPOtest", "ppo_model.pth")
        dict_path = os.path.join(root, "saved_model", "PPOtest", "normalize_dict.pkl")
        self.model = torch.jit.load(model_path)
        with open(dict_path, "rb") as f:
            self.normalize_dict = pickle.load(f)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        raw = build_state(
            timeStepIndex, pValues, historyPValueInfo, historyBid,
            historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
            self.budget, self.remaining_budget,
        )
        state = apply_normalize(raw, self.normalize_dict)
        state_t = torch.tensor(state, dtype=torch.float32)
        alpha = float(self.model(state_t).item())
        return alpha * np.asarray(pValues)
