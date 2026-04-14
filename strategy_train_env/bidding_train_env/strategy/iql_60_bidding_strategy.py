import os
import pickle

import numpy as np
import torch

from bidding_train_env.baseline.iql_60.state_builder import (
    apply_normalize,
    build_state_60_from_current,
)
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class Iql60BiddingStrategy(BaseBiddingStrategy):
    """Offline-eval strategy wrapper for the IQL60 extension."""

    def __init__(self, budget=100, name="Iql60-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(os.path.dirname(file_name))
        model_path = os.path.join(dir_name, "saved_model", "IQL60test", "iql_model.pth")
        dict_path = os.path.join(dir_name, "saved_model", "IQL60test", "normalize_dict.pkl")
        self.model = torch.jit.load(model_path)
        with open(dict_path, "rb") as file:
            self.normalize_dict = pickle.load(file)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        current_lwc = np.asarray(historyLeastWinningCost[-1], dtype=np.float32) if False else None
        del current_lwc
        raise RuntimeError(
            "Iql60BiddingStrategy requires the caller to provide current leastWinningCost. "
            "Use the validation path in run_iql_60.py or adapt the evaluator wrapper."
        )

    def bidding_with_current_lwc(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        currentLeastWinningCost,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        state = build_state_60_from_current(
            timeStepIndex=timeStepIndex,
            pValues=pValues,
            pValueSigmas=pValueSigmas,
            currentLeastWinningCost=currentLeastWinningCost,
            historyPValueInfo=historyPValueInfo,
            historyBid=historyBid,
            historyAuctionResult=historyAuctionResult,
            historyImpressionResult=historyImpressionResult,
            historyLeastWinningCost=historyLeastWinningCost,
            budget=self.budget,
            remaining_budget=self.remaining_budget,
            target_cpa=self.cpa,
            category=self.category,
        )
        state = apply_normalize(state, self.normalize_dict)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        alpha = self.model(state_tensor).cpu().numpy()
        return alpha * pValues
