import csv
import json
import logging
import math
import os
from typing import Callable, Dict, Optional

import numpy as np

from bidding_train_env.baseline.iql.metrics import (
    MetricsTracker,
    plot_training_curves,
)
from bidding_train_env.offline_eval.offline_env import OfflineEnv
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader


logger = logging.getLogger(__name__)


def get_score_neurips(reward: float, cpa: float, cpa_constraint: float) -> float:
    penalty = 1.0
    if math.isfinite(cpa) and cpa > cpa_constraint:
        penalty = (cpa_constraint / (cpa + 1e-10)) ** 2
    elif not math.isfinite(cpa):
        penalty = 0.0
    return penalty * reward


def _compute_cpa(all_cost: float, reward: float) -> float:
    if reward > 0:
        return all_cost / (reward + 1e-10)
    if all_cost > 0:
        return float("inf")
    return 0.0


def _compute_cpa_exceedance(cpa: float, cpa_constraint: float) -> float:
    if math.isfinite(cpa):
        return (cpa - cpa_constraint) / (cpa_constraint + 1e-10)
    return float("inf")


def evaluate_offline_bidding_strategy(
    strategy_factory: Callable[[], object],
    file_path: str,
    max_groups: Optional[int] = None,
    random_seed: Optional[int] = 1,
) -> Dict[str, float]:
    """
    Evaluate a bidding strategy on held-out offline data and return low-variance
    checkpoint metrics.

    The three main metrics are:
    - continuous_raw_score
    - continuous_reward
    - cpa_exceedance_rate

    The returned metrics are averaged across validation groups.
    """

    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv()
    keys = data_loader.keys[:max_groups] if max_groups is not None else data_loader.keys

    group_metrics = []

    for group_idx, key in enumerate(keys):
        if random_seed is not None:
            np.random.seed(random_seed + group_idx)

        group_df = data_loader.test_dict[key]
        agent = strategy_factory()

        if "budget" in group_df.columns:
            agent.budget = float(group_df["budget"].iloc[0])
        if "CPAConstraint" in group_df.columns:
            agent.cpa = float(group_df["CPAConstraint"].iloc[0])
        if "advertiserCategoryIndex" in group_df.columns:
            agent.category = int(group_df["advertiserCategoryIndex"].iloc[0])
        agent.reset()

        num_time_steps, p_values, p_value_sigmas, least_winning_costs = data_loader.mock_data(key)

        sparse_reward_total = 0.0
        continuous_reward_total = 0.0
        history = {
            "historyBids": [],
            "historyAuctionResult": [],
            "historyImpressionResult": [],
            "historyLeastWinningCost": [],
            "historyPValueInfo": [],
        }

        for time_step_index in range(num_time_steps):
            p_value = p_values[time_step_index]
            p_value_sigma = p_value_sigmas[time_step_index]
            least_winning_cost = least_winning_costs[time_step_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(p_value.shape[0])
            else:
                bid = agent.bidding(
                    time_step_index,
                    p_value,
                    p_value_sigma,
                    history["historyPValueInfo"],
                    history["historyBids"],
                    history["historyAuctionResult"],
                    history["historyImpressionResult"],
                    history["historyLeastWinningCost"],
                )

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                p_value,
                p_value_sigma,
                bid,
                least_winning_cost,
            )

            over_cost_ratio = max(
                (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4),
                0,
            )
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                if pv_index.size == 0:
                    break
                dropped_pv_index = np.random.choice(
                    pv_index,
                    int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                    replace=False,
                )
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                    p_value,
                    p_value_sigma,
                    bid,
                    least_winning_cost,
                )
                over_cost_ratio = max(
                    (np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4),
                    0,
                )

            agent.remaining_budget -= np.sum(tick_cost)
            sparse_reward_total += float(np.sum(tick_conversion))
            continuous_reward_total += float(np.sum(p_value * tick_status))

            tick_pvalue_info = np.array([(p_value[i], p_value_sigma[i]) for i in range(p_value.shape[0])])
            history["historyPValueInfo"].append(tick_pvalue_info)
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(least_winning_cost)
            tick_auction_result = np.array(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])]
            )
            history["historyAuctionResult"].append(tick_auction_result)
            tick_impression_result = np.array(
                [(tick_conversion[i], tick_conversion[i]) for i in range(p_value.shape[0])]
            )
            history["historyImpressionResult"].append(tick_impression_result)

        all_cost = float(agent.budget - agent.remaining_budget)
        continuous_cpa = _compute_cpa(all_cost, continuous_reward_total)
        sparse_cpa = _compute_cpa(all_cost, sparse_reward_total)
        continuous_raw_score = get_score_neurips(continuous_reward_total, continuous_cpa, agent.cpa)
        sparse_raw_score = get_score_neurips(sparse_reward_total, sparse_cpa, agent.cpa)
        cpa_exceedance_rate = _compute_cpa_exceedance(continuous_cpa, agent.cpa)
        budget_consumer_ratio = all_cost / agent.budget if agent.budget > 0 else 0.0

        group_metrics.append(
            {
                "continuous_raw_score": continuous_raw_score,
                "continuous_reward": continuous_reward_total,
                "cpa_exceedance_rate": cpa_exceedance_rate,
                "budget_consumer_ratio": budget_consumer_ratio,
                "sparse_raw_score": sparse_raw_score,
                "sparse_reward": sparse_reward_total,
            }
        )

    if not group_metrics:
        raise ValueError(f"No validation groups found in {file_path}")

    metric_names = list(group_metrics[0].keys())
    return {
        metric_name: float(np.mean([metrics[metric_name] for metrics in group_metrics]))
        for metric_name in metric_names
    } | {"num_groups": len(group_metrics)}


class OfflineMetricTracker:
    """
    Track checkpoint metrics during training and remember the best checkpoint for
    each monitored metric.
    """

    def __init__(
        self,
        output_dir: str,
        evaluator: Callable[[], Dict[str, float]],
        checkpoint_saver: Optional[Callable[[int], str]] = None,
        eval_interval: int = 1000,
    ):
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.checkpoint_saver = checkpoint_saver
        self.eval_interval = eval_interval
        self.csv_path = os.path.join(output_dir, "training_curve.csv")
        self.best_path = os.path.join(output_dir, "best_metrics.json")
        self.metrics_path = os.path.join(output_dir, "metrics.json")
        self.training_tracker = MetricsTracker()
        self.metric_modes = {
            "continuous_raw_score": "max",
            "sparse_raw_score": "max",
            "continuous_reward": "max",
            "cpa_exceedance_rate": "min",
        }
        self.best_metrics = {
            metric_name: {
                "mode": mode,
                "value": None,
                "step": None,
                "checkpoint_path": None,
            }
            for metric_name, mode in self.metric_modes.items()
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def should_evaluate(self, step: int, force: bool = False) -> bool:
        return force or (self.eval_interval > 0 and step % self.eval_interval == 0)

    def maybe_evaluate(
        self,
        step: int,
        extra_metrics: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> Optional[Dict[str, float]]:
        if not self.should_evaluate(step, force=force):
            return None

        checkpoint_path = self.checkpoint_saver(step) if self.checkpoint_saver else None
        metrics = dict(self.evaluator())
        if extra_metrics:
            metrics.update(extra_metrics)
        metrics["step"] = step
        if checkpoint_path is not None:
            metrics["checkpoint_path"] = checkpoint_path

        self._append_metrics(metrics)
        self._log_training_iteration(metrics)
        updated_metrics = self._update_best_metrics(metrics)
        self._write_best_metrics()
        self.training_tracker.save(self.metrics_path)
        plot_training_curves(self.training_tracker, self.output_dir)

        if updated_metrics:
            logger.info("Updated best validation metrics at step %s: %s", step, ", ".join(updated_metrics))
        logger.info(
            "Validation metrics at step %s: sparse_raw_score=%.6f continuous_raw_score=%.6f continuous_reward=%.6f cpa_exceedance_rate=%.6f",
            step,
            metrics["sparse_raw_score"],
            metrics["continuous_raw_score"],
            metrics["continuous_reward"],
            metrics["cpa_exceedance_rate"],
        )
        return metrics

    def _append_metrics(self, metrics: Dict[str, float]) -> None:
        write_header = not os.path.exists(self.csv_path)
        fieldnames = list(metrics.keys())
        with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    def _log_training_iteration(self, metrics: Dict[str, float]) -> None:
        q_loss = float(metrics.get("train_q_loss", 0.0))
        v_loss = float(metrics.get("train_v_loss", 0.0))
        a_loss = float(metrics.get("train_a_loss", 0.0))
        self.training_tracker.log_iteration(
            step=int(metrics["step"]),
            mean_eval_reward=float(metrics["continuous_reward"]),
            mean_eval_conversions=float(metrics["sparse_reward"]),
            mean_eval_score=float(metrics["sparse_raw_score"]),
            mean_budget_utilization=float(metrics["budget_consumer_ratio"]),
            continuous_score=float(metrics["continuous_raw_score"]),
            cpa_exceedance=float(metrics["cpa_exceedance_rate"]),
            train_loss=(q_loss + v_loss + a_loss) / 3.0,
            q_loss=q_loss,
            v_loss=v_loss,
            a_loss=a_loss,
            num_groups=int(metrics.get("num_groups", 0)),
        )

    def _update_best_metrics(self, metrics: Dict[str, float]) -> list[str]:
        updated_metrics = []
        for metric_name, mode in self.metric_modes.items():
            current_value = metrics.get(metric_name)
            if current_value is None:
                continue

            best_entry = self.best_metrics[metric_name]
            best_value = best_entry["value"]
            is_better = False
            if best_value is None:
                is_better = True
            elif mode == "max":
                is_better = current_value > best_value
            elif mode == "min":
                is_better = current_value < best_value

            if is_better:
                best_entry["value"] = current_value
                best_entry["step"] = metrics["step"]
                best_entry["checkpoint_path"] = metrics.get("checkpoint_path")
                updated_metrics.append(metric_name)
        return updated_metrics

    def _write_best_metrics(self) -> None:
        with open(self.best_path, "w", encoding="utf-8") as json_file:
            json.dump(self.best_metrics, json_file, indent=2)
