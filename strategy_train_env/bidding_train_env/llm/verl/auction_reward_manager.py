"""Custom VeRL reward manager for the AuctionNet multi-turn bidding rollout.

The async VLLM completion callback runs the AuctionNet replay environment and
stores per-episode metrics in `non_tensor_batch["reward_scores"]`. This reward
manager converts the terminal episode score into VeRL's token-level reward
tensor while surfacing useful metrics for logging.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager import register


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


@register("auction_env")
class AuctionEnvRewardManager:
    """Consume rollout-side AuctionNet metrics produced by the completion callback."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False):
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        metric_keys = (
            "episode_score",
            "total_reward",
            "malformed_count",
            "malformed_rate",
            "conversions",
            "cost",
            "cpa",
            "target_cpa",
            "budget",
            "cost_over_budget",
            "target_cpa_over_cpa",
        )

        response_length = responses.shape[-1]
        if "loss_mask" in data.batch:
            response_mask = data.batch["loss_mask"][:, -response_length:]
        else:
            response_mask = data.batch["attention_mask"][:, -response_length:]

        reward_scores = data.non_tensor_batch.get("reward_scores")
        if reward_scores is None:
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            return reward_tensor

        for i in range(len(data)):
            raw_scores = reward_scores[i]
            scores = raw_scores.item() if isinstance(raw_scores, np.ndarray) and raw_scores.shape == () else raw_scores
            if not isinstance(scores, dict):
                scores = {}

            reward = _to_float(scores.get("episode_score", 0.0))
            valid_positions = torch.nonzero(response_mask[i] > 0, as_tuple=False).flatten()
            if len(valid_positions) == 0:
                valid_positions = torch.nonzero(data.batch["attention_mask"][i, -response_length:] > 0, as_tuple=False).flatten()
            if len(valid_positions) > 0:
                reward_tensor[i, int(valid_positions[-1])] = reward

            for key in metric_keys:
                reward_extra_info[key].append(_to_float(scores.get(key, 0.0)))

            if self.num_examine > 0 and i < self.num_examine:
                prompt = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                response = self.tokenizer.decode(data.batch["responses"][i], skip_special_tokens=True)
                print("[prompt]", prompt)
                print("[response]", response)
                print("[episode_score]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
