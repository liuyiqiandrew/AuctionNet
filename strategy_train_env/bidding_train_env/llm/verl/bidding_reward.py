"""Reward function consumed by VeRL's `custom_reward_function` hook.

VeRL calls this after each rollout. Our `BiddingAgentLoop` already computes
the terminal auction score inside the env — the score is passed back via
`AgentLoopOutput.metrics["episode_score"]` and surfaced to this function as
`extra_info["__agent_loop_metrics__"]` (or directly under `extra_info`
depending on the VeRL version). We check both.

AuctionNet episode score:
    score = min(1, (target_cpa / realized_cpa)^2) * conversions
"""
from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    if extra_info is None:
        return 0.0
    # Newer verl: metrics piped under __agent_loop_metrics__.
    loop_metrics = extra_info.get("__agent_loop_metrics__") or {}
    score = loop_metrics.get("episode_score")
    if score is None:
        # Older verl: metrics merged directly into extra_info.
        score = extra_info.get("episode_score", 0.0)
    return float(score)
