import argparse
import numpy as np
import logging
import os
import random
import re
import torch
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.iql.iql import IQL
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from run.offline_metric_tracker import OfflineMetricTracker, evaluate_offline_bidding_strategy
import sys
import pandas as pd
import ast

np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16
NORMALIZE_INDICES = [13, 14, 15]
TRAIN_PERIODS = tuple(range(7, 27))
DEFAULT_TRAIN_DATA_DIR = "./data/traffic/training_data_rlData_folder"
DEFAULT_VALIDATION_DATA_PATH = "./data/traffic/period-27.csv"
DEFAULT_TRAIN_STEPS = 100000
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_BATCH_SIZE = 100
DEFAULT_LOG_INTERVAL = 100
DEFAULT_MAX_VALIDATION_GROUPS = 8
DEFAULT_SAVE_ROOT = "saved_model/IQLtest"


def _set_random_seeds(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_data_cache_path(train_data_dir=DEFAULT_TRAIN_DATA_DIR, periods=TRAIN_PERIODS):
    return os.path.join(
        train_data_dir,
        f"training_data_period-{periods[0]}-{periods[-1]}-rlData.csv",
    )


def ensure_train_data_cache(train_data_dir=DEFAULT_TRAIN_DATA_DIR, periods=TRAIN_PERIODS):
    cache_path = _train_data_cache_path(train_data_dir=train_data_dir, periods=periods)
    if os.path.exists(cache_path):
        return cache_path

    frames = []
    for period in periods:
        period_path = os.path.join(train_data_dir, f"period-{period}-rlData.csv")
        if not os.path.exists(period_path):
            raise FileNotFoundError(f"Missing training cache for period-{period}: {period_path}")
        frames.append(pd.read_csv(period_path))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(cache_path, index=False)
    logger.info("Saved IQL training cache for periods %s-%s to %s", periods[0], periods[-1], cache_path)
    return cache_path


class IqlValidationStrategy(BaseBiddingStrategy):
    """
    Lightweight in-memory strategy wrapper used for checkpoint validation during
    training.
    """

    def __init__(self, model, normalize_dict, budget=100, name="Iql-ValidationStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        self.model = model
        self.normalize_dict = normalize_dict

    def reset(self):
        self.remaining_budget = self.budget

    @staticmethod
    def _mean_of_last_n_elements(history, n):
        last_n_data = history[max(0, len(history) - n):]
        if len(last_n_data) == 0:
            return 0.0
        return float(np.mean([np.mean(data) for data in last_n_data]))

    def _build_state(
        self,
        timeStepIndex,
        pValues,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pvalue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0.0
        historical_conversion_mean = (
            np.mean([np.mean(reward) for reward in history_conversion]) if history_conversion else 0.0
        )
        historical_least_winning_cost_mean = (
            np.mean([np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0.0
        )
        historical_pvalues_mean = np.mean([np.mean(value) for value in history_pvalue]) if history_pvalue else 0.0
        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0.0

        last_three_xi_mean = self._mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = self._mean_of_last_n_elements(history_conversion, 3)
        last_three_least_winning_cost_mean = self._mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pvalues_mean = self._mean_of_last_n_elements(history_pvalue, 3)
        last_three_bid_mean = self._mean_of_last_n_elements(historyBid, 3)

        current_pvalues_mean = float(np.mean(pValues))
        current_pv_num = len(pValues)
        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_pv_num_total = sum(len(bids) for bids in historyBid[-3:]) if historyBid else 0

        test_state = np.array(
            [
                time_left,
                budget_left,
                historical_bid_mean,
                last_three_bid_mean,
                historical_least_winning_cost_mean,
                historical_pvalues_mean,
                historical_conversion_mean,
                historical_xi_mean,
                last_three_least_winning_cost_mean,
                last_three_pvalues_mean,
                last_three_conversion_mean,
                last_three_xi_mean,
                current_pvalues_mean,
                current_pv_num,
                last_three_pv_num_total,
                historical_pv_num_total,
            ],
            dtype=np.float32,
        )

        for key, value in self.normalize_dict.items():
            min_value = value["min"]
            max_value = value["max"]
            test_state[key] = (
                (test_state[key] - min_value) / (max_value - min_value)
                if max_value > min_value
                else 0.0
            )

        return test_state

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
        del pValueSigmas
        state = self._build_state(
            timeStepIndex,
            pValues,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )

        with torch.no_grad():
            model_was_training = self.model.training
            self.model.eval()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            alpha = self.model(state_tensor).detach().cpu().numpy()
            if model_was_training:
                self.model.train()

        return alpha * pValues


def save_iql_checkpoint(model, normalize_dict, save_root, step):
    checkpoint_dir = os.path.join(save_root, "checkpoints", f"step_{step:05d}")
    model.save_jit(checkpoint_dir)
    save_normalize_dict(normalize_dict, checkpoint_dir)
    return checkpoint_dir


def build_iql_metric_tracker(
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
        return IqlValidationStrategy(model=model, normalize_dict=normalize_dict)

    def evaluator():
        return evaluate_offline_bidding_strategy(
            strategy_factory=strategy_factory,
            file_path=validation_data_path,
            max_groups=max_validation_groups,
        )

    def checkpoint_saver(step):
        return save_iql_checkpoint(model, normalize_dict, save_root, step)

    logger.info(
        (
            "Offline metric tracking enabled: eval_interval=%s "
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


def _format_period_summary(period_values):
    unique_periods = sorted({int(period) for period in period_values if pd.notna(period)})
    if not unique_periods:
        return "unknown"
    if len(unique_periods) == 1:
        return f"period-{unique_periods[0]}"
    contiguous = all(
        next_period - current_period == 1
        for current_period, next_period in zip(unique_periods, unique_periods[1:])
    )
    if contiguous:
        return f"period-{unique_periods[0]} to {unique_periods[-1]}"
    return ", ".join(f"period-{period}" for period in unique_periods)


def _format_test_data_summary(validation_data_path):
    match = re.search(r"period-(\d+)", os.path.basename(validation_data_path))
    if match:
        return f"period-{match.group(1)}"
    return validation_data_path


def _write_training_report(
    save_root,
    training_data,
    validation_data_path,
    model,
    metric_tracker,
    step_num,
):
    report_path = os.path.join(save_root, "training_report.md")
    image_path = os.path.join(save_root, "training_curves.png")
    train_period_summary = _format_period_summary(training_data.get("deliveryPeriodIndex", pd.Series(dtype=float)))
    test_period_summary = _format_test_data_summary(validation_data_path)

    preprocessing = (
        f"16-dim state features; serialized state/next_state parsed from CSV; "
        f"min-max normalization on feature indices {NORMALIZE_INDICES}; "
        f"continuous reward normalization used for training"
    )
    learning_rate = (
        f"actor_lr={model.actor_lr}, critic_lr={model.critic_lr}, value_lr={model.V_lr}"
    )

    if metric_tracker is not None and metric_tracker.training_tracker.iterations:
        latest = metric_tracker.training_tracker.iterations[-1]
        best_score = metric_tracker.best_metrics.get("sparse_raw_score", {}).get("value")
        best_score_step = metric_tracker.best_metrics.get("sparse_raw_score", {}).get("step")
        other_metrics = [
            f"Latest score: {latest.mean_eval_score:.4f} @ step {latest.step}",
            f"Latest episode reward: {latest.mean_eval_reward:.4f}",
            f"Latest train/loss: {latest.train_loss:.4f}",
            f"Latest budget utilization: {latest.mean_budget_utilization:.4f}",
            f"Validation groups per eval: {latest.num_groups}",
        ]
        if best_score is not None and best_score_step is not None:
            other_metrics.append(f"Best score: {best_score:.4f} @ step {best_score_step}")
    else:
        other_metrics = ["No offline validation metrics were recorded."]

    train_periods = {
        int(period)
        for period in training_data.get("deliveryPeriodIndex", pd.Series(dtype=float))
        if pd.notna(period)
    }
    test_match = re.search(r"period-(\d+)", test_period_summary)
    if test_match and int(test_match.group(1)) in train_periods:
        other_metrics.append(
            f"Current default validation data ({test_period_summary}) overlaps the training data range ({train_period_summary})."
        )
    else:
        other_metrics.append(
            f"Training/validation split follows PPO-style holdout: train {train_period_summary}, validate {test_period_summary}."
        )

    if os.path.exists(image_path):
        visualization = f"[image #1](training_curves.png)"
    else:
        visualization = "N/A"

    report_lines = [
        f"Method: IQL",
        f"Training data: {train_period_summary}",
        f"Testing data: {test_period_summary}",
        f"Data preprocessing techniques: {preprocessing}",
        f"Learning rate: {learning_rate}",
        f"Number of steps trained: {step_num}",
        (
            "Training visualization (if any, incl. Score / Ep. reward / Loss / Entropy plots): "
            f"{visualization}"
        ),
        "Other important metrics / comments",
    ]
    report_lines.extend(f"- {metric}" for metric in other_metrics)

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")


def train_iql_model(
    step_num=DEFAULT_TRAIN_STEPS,
    eval_interval=DEFAULT_EVAL_INTERVAL,
    validation_data_path=DEFAULT_VALIDATION_DATA_PATH,
    max_validation_groups=DEFAULT_MAX_VALIDATION_GROUPS,
    batch_size=DEFAULT_BATCH_SIZE,
    log_interval=DEFAULT_LOG_INTERVAL,
    save_root=DEFAULT_SAVE_ROOT,
    seed=1,
    save_eval_checkpoints=True,
):
    """
    Train the IQL model.
    """
    _set_random_seeds(seed)
    train_data_path = ensure_train_data_cache()
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            try:
                return eval(val, {"__builtins__": {}}, {"np": np})
            except Exception:
                return val

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    is_normalize = True
    normalize_dic = {}

    if is_normalize:
        normalize_dic = normalize_state(training_data, STATE_DIM, normalize_indices=NORMALIZE_INDICES)
        # select use continuous reward
        training_data['reward'] = normalize_reward(training_data, "reward_continuous")
        # select use sparse reward
        # training_data['reward'] = normalize_reward(training_data, "reward")
        save_normalize_dict(normalize_dic, save_root)

    # Build replay buffer
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    # Train model
    model = IQL(dim_obs=STATE_DIM)
    metric_tracker = build_iql_metric_tracker(
        model=model,
        normalize_dict=normalize_dic,
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

    # Save model
    model.save_jit(save_root)

    _write_training_report(
        save_root=save_root,
        training_data=training_data,
        validation_data_path=validation_data_path,
        model=model,
        metric_tracker=metric_tracker,
        step_num=step_num,
    )

    # Test trained model
    test_trained_model(model, replay_buffer)


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


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
            logger.info(f'Step: {step} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}')
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
    tem = np.concatenate((actions, pred_actions), axis=1)
    print("action VS pred action:", tem)


def run_iql():
    print(sys.path)
    """
    Run IQL model training and evaluation.
    """
    train_iql_model()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train IQL on CPU or GPU resources.")
    parser.add_argument("--steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--max-validation-groups", type=int, default=DEFAULT_MAX_VALIDATION_GROUPS)
    parser.add_argument("--validation-data-path", default=DEFAULT_VALIDATION_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-eval-checkpoints",
        action="store_true",
        help="Record validation metrics without saving per-eval model checkpoint directories.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_iql_model(
        step_num=args.steps,
        eval_interval=args.eval_interval,
        validation_data_path=args.validation_data_path,
        max_validation_groups=args.max_validation_groups,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        save_root=args.save_root,
        seed=args.seed,
        save_eval_checkpoints=not args.no_eval_checkpoints,
    )


if __name__ == '__main__':
    main()
