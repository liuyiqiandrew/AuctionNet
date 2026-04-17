import argparse
import concurrent.futures
import logging
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

from bidding_train_env.baseline.iql.iql import IQL
from bidding_train_env.baseline.iql_60.replay_buffer import ArrayReplayBuffer
from bidding_train_env.baseline.iql_60.state_builder import (
    FEATURE_NAMES,
    STATE_DIM,
    build_state_60_from_current,
)
from bidding_train_env.common.utils import save_normalize_dict
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from run.offline_metric_tracker import OfflineMetricTracker

np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TRAIN_PERIODS = tuple(range(7, 27))
STATE_NORMALIZE_INDICES = list(range(STATE_DIM))
DEFAULT_RAW_DATA_DIR = "./data/traffic"
DEFAULT_CACHE_DIR = "./data/traffic/training_data_60_rlData_folder"
DEFAULT_VALIDATION_DATA_PATH = "./data/traffic/period-27.csv"
DEFAULT_TRAIN_STEPS = 100000
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_LOG_INTERVAL = 100
DEFAULT_MAX_VALIDATION_GROUPS = 8
DEFAULT_SAVE_ROOT = "saved_model/IQL60test"
TRAIN_DATA_FIELDS = (
    "states",
    "actions",
    "rewards_continuous",
    "next_states",
    "dones",
)
NORMALIZED_CACHE_FIELDS = (
    "normalize_state",
    "action",
    "normalize_reward",
    "normalize_nextstate",
    "done",
    "state_min",
    "state_max",
    "state_mean",
    "state_std",
)


def _set_random_seeds(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _period_cache_path(cache_dir, period):
    return os.path.join(cache_dir, f"period-{period}-rlData60.npz")


def _combined_cache_base(cache_dir, periods):
    period_tag = f"period-{periods[0]}-{periods[-1]}"
    return os.path.join(cache_dir, f"training_data_{period_tag}-rlData60")


def _get_cpu_thread_count():
    env_val = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv("OMP_NUM_THREADS")
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _configure_torch_cpu_threads():
    cpu_threads = _get_cpu_thread_count()
    try:
        requested_threads = max(1, int(os.getenv("TORCH_NUM_THREADS", cpu_threads)))
    except ValueError:
        requested_threads = cpu_threads
    torch.set_num_threads(requested_threads)
    try:
        torch.set_num_interop_threads(max(1, min(8, requested_threads // 2 or 1)))
    except RuntimeError:
        # set_num_interop_threads can only be called once per process.
        pass
    logger.info(
        "Configured torch CPU threading: requested=%s num_threads=%s num_interop_threads=%s",
        requested_threads,
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )


def _serialize_transition_frame(training_data):
    states = np.asarray(training_data["state"].to_list(), dtype=np.float32)
    next_states = np.zeros((len(training_data), STATE_DIM), dtype=np.float32)
    valid_next = training_data["next_state"].notna().to_numpy()
    if valid_next.any():
        next_states[valid_next] = np.asarray(
            training_data.loc[valid_next, "next_state"].to_list(),
            dtype=np.float32,
        )
    return {
        "states": states,
        "actions": training_data["action"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "rewards_continuous": training_data["reward_continuous"].to_numpy(dtype=np.float32).reshape(-1, 1),
        "next_states": next_states,
        "dones": training_data["done"].to_numpy(dtype=np.float32).reshape(-1, 1),
    }


def _load_npz_payload(npz_path, expected_fields):
    with np.load(npz_path, allow_pickle=False) as payload:
        return {field: payload[field] for field in expected_fields}


def _build_normalize_dict(state_min, state_max, state_mean, state_std):
    return {
        i: {
            "min": float(state_min[i]),
            "max": float(state_max[i]),
            "mean": float(state_mean[i]),
            "std": float(state_std[i]),
        }
        for i in STATE_NORMALIZE_INDICES
    }


def _normalize_training_arrays(combined_cache_path, normalized_cache_path):
    payload = _load_npz_payload(combined_cache_path, TRAIN_DATA_FIELDS)
    states = payload["states"]
    next_states = payload["next_states"]
    rewards = payload["rewards_continuous"]

    state_min = states.min(axis=0)
    state_max = states.max(axis=0)
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0)
    denom = (state_max - state_min + 0.01).astype(np.float32, copy=False)

    normalized_states = ((states - state_min) / denom).astype(np.float32, copy=False)
    normalized_next_states = ((next_states - state_min) / denom).astype(np.float32, copy=False)

    reward_min = float(rewards.min())
    reward_range = float(rewards.max() - reward_min + 1e-8)
    normalized_rewards = ((rewards - reward_min) / reward_range).astype(np.float32, copy=False)

    np.savez(
        normalized_cache_path,
        normalize_state=normalized_states,
        action=payload["actions"],
        normalize_reward=normalized_rewards,
        normalize_nextstate=normalized_next_states,
        done=payload["dones"],
        state_min=state_min.astype(np.float32, copy=False),
        state_max=state_max.astype(np.float32, copy=False),
        state_mean=state_mean.astype(np.float32, copy=False),
        state_std=state_std.astype(np.float32, copy=False),
    )
    logger.info("Saved normalized IQL60 cache to %s", normalized_cache_path)


def _load_or_build_normalized_training_cache(
    raw_data_dir=DEFAULT_RAW_DATA_DIR,
    cache_dir=DEFAULT_CACHE_DIR,
    periods=TRAIN_PERIODS,
    cache_workers=None,
):
    combined_cache_base = _combined_cache_base(cache_dir, periods)
    combined_cache_path = f"{combined_cache_base}.npz"
    normalized_cache_path = f"{combined_cache_base}-normalized.npz"

    if not os.path.exists(normalized_cache_path):
        combined_cache_path = ensure_train_data_60(
            raw_data_dir=raw_data_dir,
            cache_dir=cache_dir,
            periods=periods,
            cache_workers=cache_workers,
        )
        _normalize_training_arrays(combined_cache_path, normalized_cache_path)

    payload = _load_npz_payload(normalized_cache_path, NORMALIZED_CACHE_FIELDS)
    normalize_dict = _build_normalize_dict(
        payload["state_min"],
        payload["state_max"],
        payload["state_mean"],
        payload["state_std"],
    )
    return payload, normalize_dict


class Iql60ValidationStrategy(BaseBiddingStrategy):
    """In-memory validation strategy that uses the 60-dim state."""

    def __init__(self, model, normalize_dict, budget=100, name="Iql60-ValidationStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        self.model = model
        self.normalize_dict = normalize_dict

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
        raise RuntimeError("Use bidding_with_current_lwc in the offline evaluation wrapper.")

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
        for key, value in self.normalize_dict.items():
            min_value = value["min"]
            max_value = value["max"]
            state[key] = (
                (state[key] - min_value) / (max_value - min_value + 0.01)
                if max_value >= min_value
                else 0.0
            )

        with torch.no_grad():
            model_was_training = self.model.training
            self.model.eval()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            alpha = self.model(state_tensor).detach().cpu().numpy()
            if model_was_training:
                self.model.train()
        return alpha * np.asarray(pValues)


def _evaluate_offline_bidding_strategy_60(
    strategy_factory,
    file_path,
    max_groups=None,
    random_seed=1,
):
    from bidding_train_env.offline_eval.offline_env import OfflineEnv
    from bidding_train_env.offline_eval.test_dataloader import TestDataLoader

    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv()
    keys = data_loader.keys[:max_groups] if max_groups is not None else data_loader.keys

    def get_score_neurips(reward, cpa, cpa_constraint):
        penalty = 1.0
        if np.isfinite(cpa) and cpa > cpa_constraint:
            penalty = (cpa_constraint / (cpa + 1e-10)) ** 2
        elif not np.isfinite(cpa):
            penalty = 0.0
        return penalty * reward

    def compute_cpa(all_cost, reward):
        if reward > 0:
            return all_cost / (reward + 1e-10)
        if all_cost > 0:
            return float("inf")
        return 0.0

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
                bid = agent.bidding_with_current_lwc(
                    time_step_index,
                    p_value,
                    p_value_sigma,
                    least_winning_cost,
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
                dropped = np.random.choice(
                    pv_index,
                    int(np.ceil(pv_index.shape[0] * over_cost_ratio)),
                    replace=False,
                )
                bid[dropped] = 0
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

            history["historyPValueInfo"].append(np.column_stack([p_value, p_value_sigma]))
            history["historyBids"].append(np.asarray(bid))
            history["historyLeastWinningCost"].append(np.asarray(least_winning_cost))
            history["historyAuctionResult"].append(
                np.column_stack([tick_status.astype(float), tick_status.astype(float), tick_cost])
            )
            history["historyImpressionResult"].append(
                np.column_stack([tick_conversion.astype(float), tick_conversion.astype(float)])
            )

        all_cost = float(agent.budget - agent.remaining_budget)
        continuous_cpa = compute_cpa(all_cost, continuous_reward_total)
        sparse_cpa = compute_cpa(all_cost, sparse_reward_total)
        continuous_raw_score = get_score_neurips(continuous_reward_total, continuous_cpa, agent.cpa)
        sparse_raw_score = get_score_neurips(sparse_reward_total, sparse_cpa, agent.cpa)
        cpa_exceedance_rate = (
            (continuous_cpa - agent.cpa) / (agent.cpa + 1e-10)
            if np.isfinite(continuous_cpa)
            else float("inf")
        )
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


def save_iql60_checkpoint(model, normalize_dict, save_root, step):
    checkpoint_dir = os.path.join(save_root, "checkpoints", f"step_{step:05d}")
    model.save_jit(checkpoint_dir)
    save_normalize_dict(normalize_dict, checkpoint_dir)
    return checkpoint_dir


def build_iql60_metric_tracker(
    model,
    normalize_dict,
    save_root,
    validation_data_path=DEFAULT_VALIDATION_DATA_PATH,
    eval_interval=DEFAULT_EVAL_INTERVAL,
    max_validation_groups=DEFAULT_MAX_VALIDATION_GROUPS,
    save_checkpoints=True,
):
    if eval_interval <= 0:
        return None

    def strategy_factory():
        return Iql60ValidationStrategy(model=model, normalize_dict=normalize_dict)

    def evaluator():
        return _evaluate_offline_bidding_strategy_60(
            strategy_factory=strategy_factory,
            file_path=validation_data_path,
            max_groups=max_validation_groups,
        )

    def checkpoint_saver(step):
        return save_iql60_checkpoint(model, normalize_dict, save_root, step)

    logger.info(
        (
            "IQL60 offline metric tracking enabled: eval_interval=%s "
            "validation_data_path=%s max_validation_groups=%s save_checkpoints=%s"
        ),
        eval_interval,
        validation_data_path,
        max_validation_groups,
        save_checkpoints,
    )
    return OfflineMetricTracker(
        output_dir=save_root,
        evaluator=evaluator,
        checkpoint_saver=checkpoint_saver if save_checkpoints else (lambda step: ""),
        eval_interval=eval_interval,
    )


def _period_summary(periods):
    periods = sorted({int(period) for period in periods})
    if not periods:
        return "unknown"
    if len(periods) == 1:
        return f"period-{periods[0]}"
    return f"period-{periods[0]} to {periods[-1]}"


def _validation_summary(validation_data_path):
    match = re.search(r"period-(\d+)", os.path.basename(validation_data_path))
    if match:
        return f"period-{match.group(1)}"
    return validation_data_path


def _write_training_report(save_root, model, metric_tracker, step_num, validation_data_path):
    report_path = os.path.join(save_root, "training_report.md")
    image_path = os.path.join(save_root, "training_curves.png")
    visualization = "[image #1](training_curves.png)" if os.path.exists(image_path) else "N/A"

    report_lines = [
        "Method: IQL60",
        f"Training data: {_period_summary(TRAIN_PERIODS)}",
        f"Testing data: {_validation_summary(validation_data_path)}",
        (
            "Data preprocessing techniques: "
            f"60-dim state ({', '.join(FEATURE_NAMES[:6])}, ...); "
            "states rebuilt from raw period CSVs; min-max normalization on all 60 features; "
            "continuous reward normalization used for training"
        ),
        (
            "Learning rate: "
            f"actor_lr={model.actor_lr}, critic_lr={model.critic_lr}, value_lr={model.V_lr}"
        ),
        f"Number of steps trained: {step_num}",
        (
            "Training visualization (if any, incl. Score / Ep. reward / Loss / Entropy plots): "
            f"{visualization}"
        ),
        "Other important metrics / comments",
    ]

    if metric_tracker is not None and metric_tracker.training_tracker.iterations:
        latest = metric_tracker.training_tracker.iterations[-1]
        best_score = metric_tracker.best_metrics.get("sparse_raw_score", {}).get("value")
        best_step = metric_tracker.best_metrics.get("sparse_raw_score", {}).get("step")
        report_lines.extend(
            [
                f"- Latest score: {latest.mean_eval_score:.4f} @ step {latest.step}",
                f"- Latest episode reward: {latest.mean_eval_reward:.4f}",
                f"- Latest train/loss: {latest.train_loss:.4f}",
                f"- Latest budget utilization: {latest.mean_budget_utilization:.4f}",
                f"- Validation groups per eval: {latest.num_groups}",
            ]
        )
        if best_score is not None and best_step is not None:
            report_lines.append(f"- Best score: {best_score:.4f} @ step {best_step}")
    else:
        report_lines.append("- No offline validation metrics were recorded.")

    report_lines.append(
        f"- Training/validation split follows PPO-style holdout: train {_period_summary(TRAIN_PERIODS)}, validate {_validation_summary(validation_data_path)}."
    )
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")


def _generate_period_train_data_60(period_csv_path):
    raw_df = pd.read_csv(period_csv_path)
    rows = []
    group_cols = [
        "deliveryPeriodIndex",
        "advertiserNumber",
        "advertiserCategoryIndex",
        "budget",
        "CPAConstraint",
    ]
    for group_key, group in raw_df.groupby(group_cols):
        (
            delivery_period_index,
            advertiser_number,
            advertiser_category_index,
            budget,
            cpa_constraint,
        ) = group_key
        group = group.sort_values(["timeStepIndex", "pvIndex"])
        real_all_cost = float((group["isExposed"] * group["cost"]).sum())
        real_all_conversion = float(group["conversionAction"].sum())
        history = {
            "historyPValueInfo": [],
            "historyBids": [],
            "historyAuctionResult": [],
            "historyImpressionResult": [],
            "historyLeastWinningCost": [],
        }

        for time_step_index in sorted(group["timeStepIndex"].unique()):
            current = group[group["timeStepIndex"] == time_step_index].copy()
            current.fillna(0.0, inplace=True)
            pvalues = current["pValue"].to_numpy(dtype=np.float32)
            sigmas = current["pValueSigma"].to_numpy(dtype=np.float32)
            least_winning_cost = current["leastWinningCost"].to_numpy(dtype=np.float32)
            bids = current["bid"].to_numpy(dtype=np.float32)
            win = current["xi"].to_numpy(dtype=np.float32)
            costs = current["cost"].to_numpy(dtype=np.float32)
            conversion = current["conversionAction"].to_numpy(dtype=np.float32)
            remaining_budget = float(current["remainingBudget"].iloc[0])

            state = build_state_60_from_current(
                timeStepIndex=time_step_index,
                pValues=pvalues,
                pValueSigmas=sigmas,
                currentLeastWinningCost=least_winning_cost,
                historyPValueInfo=history["historyPValueInfo"],
                historyBid=history["historyBids"],
                historyAuctionResult=history["historyAuctionResult"],
                historyImpressionResult=history["historyImpressionResult"],
                historyLeastWinningCost=history["historyLeastWinningCost"],
                budget=float(budget),
                remaining_budget=remaining_budget,
                target_cpa=float(cpa_constraint),
                category=float(advertiser_category_index),
            )

            total_bid = float(np.sum(bids))
            total_value = float(np.sum(pvalues))
            action = total_bid / total_value if total_value > 0 else 0.0
            reward = float(current.loc[current["isExposed"] == 1, "conversionAction"].sum())
            reward_continuous = float(current.loc[current["isExposed"] == 1, "pValue"].sum())
            done = 1 if time_step_index == 47 or float(current["isEnd"].iloc[0]) == 1 else 0

            rows.append(
                {
                    "deliveryPeriodIndex": delivery_period_index,
                    "advertiserNumber": advertiser_number,
                    "advertiserCategoryIndex": advertiser_category_index,
                    "budget": budget,
                    "CPAConstraint": cpa_constraint,
                    "realAllCost": real_all_cost,
                    "realAllConversion": real_all_conversion,
                    "timeStepIndex": time_step_index,
                    "state": tuple(float(x) for x in state.tolist()),
                    "action": action,
                    "reward": reward,
                    "reward_continuous": reward_continuous,
                    "done": done,
                }
            )

            history["historyPValueInfo"].append(np.column_stack([pvalues, sigmas]))
            history["historyBids"].append(bids)
            history["historyLeastWinningCost"].append(least_winning_cost)
            history["historyAuctionResult"].append(np.column_stack([win, win, costs]))
            history["historyImpressionResult"].append(np.column_stack([conversion, conversion]))

    training_data = pd.DataFrame(rows)
    training_data = training_data.sort_values(
        by=["deliveryPeriodIndex", "advertiserNumber", "timeStepIndex"]
    )
    training_data["next_state"] = training_data.groupby(
        ["deliveryPeriodIndex", "advertiserNumber"]
    )["state"].shift(-1)
    training_data.loc[training_data["done"] == 1, "next_state"] = None
    return training_data


def _build_period_cache_task(period, raw_data_dir, cache_dir):
    period_csv_path = os.path.join(raw_data_dir, f"period-{period}.csv")
    cache_path = _period_cache_path(cache_dir, period)
    frame = _generate_period_train_data_60(period_csv_path)
    payload = _serialize_transition_frame(frame)
    np.savez(cache_path, **payload)
    return cache_path, len(frame)


def ensure_train_data_60(
    raw_data_dir=DEFAULT_RAW_DATA_DIR,
    cache_dir=DEFAULT_CACHE_DIR,
    periods=TRAIN_PERIODS,
    cache_workers=None,
):
    os.makedirs(cache_dir, exist_ok=True)
    combined_path = f"{_combined_cache_base(cache_dir, periods)}.npz"
    if os.path.exists(combined_path):
        return combined_path

    missing_periods = [
        period
        for period in periods
        if not os.path.exists(_period_cache_path(cache_dir, period))
    ]
    if missing_periods:
        if cache_workers is None:
            max_workers = max(1, min(8, _get_cpu_thread_count()))
        else:
            max_workers = max(1, int(cache_workers))
        max_workers = min(len(missing_periods), max_workers)
        logger.info(
            "Building %s missing 60-dim period caches with %s workers",
            len(missing_periods),
            max_workers,
        )
        if max_workers == 1:
            for period in missing_periods:
                cache_path, row_count = _build_period_cache_task(period, raw_data_dir, cache_dir)
                logger.info("Built %s (%s rows)", cache_path, row_count)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_build_period_cache_task, period, raw_data_dir, cache_dir)
                    for period in missing_periods
                ]
                for future in concurrent.futures.as_completed(futures):
                    cache_path, row_count = future.result()
                    logger.info("Built %s (%s rows)", cache_path, row_count)

    payload_parts = {field: [] for field in TRAIN_DATA_FIELDS}
    for period in periods:
        cache_path = _period_cache_path(cache_dir, period)
        payload = _load_npz_payload(cache_path, TRAIN_DATA_FIELDS)
        for field in TRAIN_DATA_FIELDS:
            payload_parts[field].append(payload[field])

    combined_payload = {
        field: np.concatenate(payload_parts[field], axis=0)
        for field in TRAIN_DATA_FIELDS
    }
    np.savez(combined_path, **combined_payload)
    logger.info("Saved combined 60-dim RL data to %s", combined_path)
    return combined_path


def train_model_steps(
    model,
    replay_buffer,
    metric_tracker=None,
    step_num=DEFAULT_TRAIN_STEPS,
    batch_size=DEFAULT_BATCH_SIZE,
    log_interval=DEFAULT_LOG_INTERVAL,
):
    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size)
        q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_states, terminals)
        step = i + 1
        if step == 1 or (log_interval > 0 and step % log_interval == 0) or step == step_num:
            logger.info("Step: %s Q_loss: %s V_loss: %s A_loss: %s", step, q_loss, v_loss, a_loss)
        if metric_tracker is not None:
            metric_tracker.maybe_evaluate(
                step,
                extra_metrics={
                    "train_q_loss": float(q_loss),
                    "train_v_loss": float(v_loss),
                    "train_a_loss": float(a_loss),
                },
                force=(step == step_num),
            )


def test_trained_model(model, replay_buffer):
    states, actions, rewards, next_states, terminals = replay_buffer.sample(100)
    pred_actions = model.take_actions(states)
    actions = actions.cpu().detach().numpy()
    _ = rewards, next_states, terminals
    logger.info("action VS pred action: %s", np.concatenate((actions, pred_actions), axis=1))


def train_iql60_model(
    step_num=DEFAULT_TRAIN_STEPS,
    eval_interval=DEFAULT_EVAL_INTERVAL,
    validation_data_path=DEFAULT_VALIDATION_DATA_PATH,
    max_validation_groups=DEFAULT_MAX_VALIDATION_GROUPS,
    batch_size=DEFAULT_BATCH_SIZE,
    log_interval=DEFAULT_LOG_INTERVAL,
    save_root=DEFAULT_SAVE_ROOT,
    seed=1,
    save_eval_checkpoints=True,
    cache_workers=None,
):
    _set_random_seeds(seed)
    _configure_torch_cpu_threads()
    t0 = time.time()
    training_payload, normalize_dict = _load_or_build_normalized_training_cache(cache_workers=cache_workers)
    logger.info("IQL60 cache preparation finished in %.1fs", time.time() - t0)

    t1 = time.time()
    save_normalize_dict(normalize_dict, save_root)
    logger.info("Loaded normalized cached IQL60 arrays in %.1fs", time.time() - t1)

    t2 = time.time()
    replay_buffer = ArrayReplayBuffer(
        states=training_payload["normalize_state"],
        actions=training_payload["action"],
        rewards=training_payload["normalize_reward"],
        next_states=training_payload["normalize_nextstate"],
        dones=training_payload["done"],
    )
    logger.info(
        "Loaded %s IQL60 transitions into contiguous replay arrays in %.1fs",
        len(replay_buffer),
        time.time() - t2,
    )

    model = IQL(dim_obs=STATE_DIM)
    metric_tracker = build_iql60_metric_tracker(
        model=model,
        normalize_dict=normalize_dict,
        save_root=save_root,
        validation_data_path=validation_data_path,
        eval_interval=eval_interval,
        max_validation_groups=max_validation_groups,
        save_checkpoints=save_eval_checkpoints,
    )
    train_model_steps(
        model,
        replay_buffer,
        metric_tracker=metric_tracker,
        step_num=step_num,
        batch_size=batch_size,
        log_interval=log_interval,
    )
    model.save_jit(save_root)
    _write_training_report(save_root, model, metric_tracker, step_num, validation_data_path)
    test_trained_model(model, replay_buffer)


def run_iql_60():
    print(sys.path)
    train_iql60_model()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train IQL60 on CPU resources.")
    parser.add_argument("--steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--max-validation-groups", type=int, default=DEFAULT_MAX_VALIDATION_GROUPS)
    parser.add_argument("--validation-data-path", default=DEFAULT_VALIDATION_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache-workers", type=int, default=None)
    parser.add_argument(
        "--no-eval-checkpoints",
        action="store_true",
        help="Record validation metrics without saving per-eval model checkpoint directories.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_iql60_model(
        step_num=args.steps,
        eval_interval=args.eval_interval,
        validation_data_path=args.validation_data_path,
        max_validation_groups=args.max_validation_groups,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        save_root=args.save_root,
        seed=args.seed,
        save_eval_checkpoints=not args.no_eval_checkpoints,
        cache_workers=args.cache_workers,
    )


if __name__ == "__main__":
    main()
