"""Metrics tracking and plotting for PPO training and evaluation."""

import json
import os
from dataclasses import dataclass, asdict, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class IterationMetrics:
    iteration: int
    mean_episode_reward: float
    mean_episode_cost: float
    mean_episode_conversions: float
    mean_episode_score: float
    mean_budget_utilization: float
    policy_loss: float
    value_loss: float
    entropy: float
    elapsed_seconds: float


@dataclass
class EvalEpisodeMetrics:
    player_index: int
    episode_seed: int
    reward: float
    cost: float
    conversions: float
    cpa: float
    cpa_constraint: float
    score: float
    budget: float
    budget_utilization: float
    num_ticks_active: int


class MetricsTracker:
    def __init__(self):
        self.iterations: list[IterationMetrics] = []
        self.eval_episodes: list[EvalEpisodeMetrics] = []

    def log_iteration(self, **kwargs) -> IterationMetrics:
        m = IterationMetrics(**kwargs)
        self.iterations.append(m)
        return m

    def log_eval_episode(self, **kwargs) -> EvalEpisodeMetrics:
        m = EvalEpisodeMetrics(**kwargs)
        self.eval_episodes.append(m)
        return m

    def save(self, path: str):
        data = {
            "iterations": [asdict(m) for m in self.iterations],
            "eval_episodes": [asdict(m) for m in self.eval_episodes],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        with open(path) as f:
            data = json.load(f)
        tracker = cls()
        tracker.iterations = [IterationMetrics(**d) for d in data.get("iterations", [])]
        tracker.eval_episodes = [EvalEpisodeMetrics(**d) for d in data.get("eval_episodes", [])]
        return tracker


def compute_score(reward: float, cpa: float, cpa_constraint: float) -> float:
    """NeurIPS scoring: penalize CPA exceedance quadratically."""
    if reward <= 0:
        return 0.0
    penalty = 1.0
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = coef ** 2
    return penalty * reward


def plot_training_curves(tracker: MetricsTracker, save_dir: str):
    if not tracker.iterations:
        return
    iters = [m.iteration for m in tracker.iterations]
    fields = [
        ("mean_episode_reward", "Episode Reward"),
        ("mean_episode_score", "Episode Score"),
        ("mean_budget_utilization", "Budget Utilization"),
        ("policy_loss", "Policy Loss"),
        ("value_loss", "Value Loss"),
        ("entropy", "Entropy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (attr, title) in zip(axes.flat, fields):
        vals = [getattr(m, attr) for m in tracker.iterations]
        ax.plot(iters, vals, linewidth=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("PPO Training Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eval_summary(tracker: MetricsTracker, save_dir: str):
    if not tracker.eval_episodes:
        return
    # Group by player_index
    by_player: dict[int, list[EvalEpisodeMetrics]] = {}
    for m in tracker.eval_episodes:
        by_player.setdefault(m.player_index, []).append(m)
    player_ids = sorted(by_player.keys())

    mean_scores = [np.mean([m.score for m in by_player[p]]) for p in player_ids]
    sem_scores = [np.std([m.score for m in by_player[p]]) / max(1, len(by_player[p]) ** 0.5) for p in player_ids]
    mean_butil = [np.mean([m.budget_utilization for m in by_player[p]]) for p in player_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(player_ids))
    ax1.bar(x, mean_scores, yerr=sem_scores, capsize=4, color="steelblue", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in player_ids])
    ax1.set_xlabel("Player Index")
    ax1.set_ylabel("Score")
    ax1.set_title("Evaluation Score by Player")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, mean_butil, color="darkorange", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(p) for p in player_ids])
    ax2.set_xlabel("Player Index")
    ax2.set_ylabel("Budget Utilization")
    ax2.set_title("Budget Utilization by Player")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis="y")

    all_scores = [m.score for m in tracker.eval_episodes]
    all_conv = [m.conversions for m in tracker.eval_episodes]
    all_cpa_exceed = [1 for m in tracker.eval_episodes if m.cpa > m.cpa_constraint]
    summary = (
        f"Avg Score: {np.mean(all_scores):.2f} +/- {np.std(all_scores) / len(all_scores)**0.5:.2f}\n"
        f"Avg Conversions: {np.mean(all_conv):.1f}\n"
        f"CPA Exceedance: {len(all_cpa_exceed)}/{len(tracker.eval_episodes)} episodes"
    )
    fig.text(0.5, -0.02, summary, ha="center", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("PPO Online Evaluation Summary", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = os.path.join(save_dir, "eval_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
