"""Metrics tracking and plotting for IQL offline validation during training."""

import json
import os
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class IterationMetrics:
    step: int
    mean_eval_reward: float
    mean_eval_conversions: float
    mean_eval_score: float
    mean_budget_utilization: float
    continuous_score: float
    cpa_exceedance: float
    train_loss: float
    q_loss: float
    v_loss: float
    a_loss: float
    num_groups: int


class MetricsTracker:
    def __init__(self):
        self.iterations: list[IterationMetrics] = []

    def log_iteration(self, **kwargs) -> IterationMetrics:
        metrics = IterationMetrics(**kwargs)
        self.iterations.append(metrics)
        return metrics

    def save(self, path: str):
        data = {"iterations": [asdict(metrics) for metrics in self.iterations]}
        with open(path, "w", encoding="utf-8") as metrics_file:
            json.dump(data, metrics_file, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        with open(path, encoding="utf-8") as metrics_file:
            data = json.load(metrics_file)
        tracker = cls()
        tracker.iterations = [IterationMetrics(**entry) for entry in data.get("iterations", [])]
        return tracker


def plot_training_curves(tracker: MetricsTracker, save_dir: str):
    if not tracker.iterations:
        return

    steps = [metrics.step for metrics in tracker.iterations]
    fields = [
        ("mean_eval_score", "score"),
        ("mean_eval_reward", "rollout/ep_rew_mean"),
        ("train_loss", "train/loss"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    latest = tracker.iterations[-1]
    for ax, (attr, title) in zip(axes, fields):
        values = [getattr(metrics, attr) for metrics in tracker.iterations]
        latest_value = getattr(latest, attr)
        ax.plot(steps, values, linewidth=1.0)
        ax.set_ylabel(title)
        ax.set_title(f"{title}  (latest: {latest_value:.4f} @ step {latest.step})")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Steps")
    fig.suptitle("IQL Offline Validation", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
