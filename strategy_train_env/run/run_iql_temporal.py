import json
import logging
import os
import re

import numpy as np
import pandas as pd
import torch

from bidding_train_env.baseline.iql_temporal.gru_iql import GRUIQL
from bidding_train_env.baseline.iql_temporal.sequence_utils import (
    TemporalContextBuffer,
    TemporalReplayBuffer,
    apply_normalize,
    build_iql_flat_state,
    safe_literal_eval,
)
from bidding_train_env.common.utils import normalize_reward, normalize_state, save_normalize_dict
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from run.offline_metric_tracker import OfflineMetricTracker, evaluate_offline_bidding_strategy

np.set_printoptions(suppress=True, precision=4)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

STATE_DIM = 16
NORMALIZE_INDICES = [13, 14, 15]
TRAIN_PERIODS = tuple(range(7, 27))
DEFAULT_TRAIN_DATA_DIR = "./data/traffic/training_data_rlData_folder"
DEFAULT_VALIDATION_DATA_PATH = "./data/traffic/period-27.csv"
DEFAULT_TRAIN_STEPS = 100000
DEFAULT_SEQUENCE_LENGTH = 8
DEFAULT_BATCH_SIZE = 256
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LOG_INTERVAL = 100
DEFAULT_SAVE_ROOT = "saved_model/IQLTemporaltest"


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
    logger.info(
        "Saved temporal IQL training cache for periods %s-%s to %s",
        periods[0],
        periods[-1],
        cache_path,
    )
    return cache_path


class TemporalIqlValidationStrategy(BaseBiddingStrategy):
    """Validation strategy that runs a GRU over the last K normalized states."""

    def __init__(
        self,
        model,
        normalize_dict,
        sequence_length,
        state_dim=STATE_DIM,
        budget=100,
        name="TemporalIql-ValidationStrategy",
        cpa=2,
        category=1,
    ):
        super().__init__(budget, name, cpa, category)
        self.model = model
        self.normalize_dict = normalize_dict
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.context_buffer = TemporalContextBuffer(seq_len=sequence_length, state_dim=state_dim)

    def reset(self):
        self.remaining_budget = self.budget
        self.context_buffer.reset()

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
        flat_state = build_iql_flat_state(
            time_step_index=timeStepIndex,
            p_values=pValues,
            history_pvalue_info=historyPValueInfo,
            history_bid=historyBid,
            history_auction_result=historyAuctionResult,
            history_impression_result=historyImpressionResult,
            history_least_winning_cost=historyLeastWinningCost,
            budget=self.budget,
            remaining_budget=self.remaining_budget,
        )
        normalized_state = apply_normalize(flat_state, self.normalize_dict)
        self.context_buffer.append(normalized_state)
        state_sequence, sequence_length = self.context_buffer.as_padded_sequence()

        with torch.no_grad():
            model_was_training = self.model.training
            self.model.eval()
            state_tensor = torch.tensor(
                state_sequence,
                dtype=torch.float32,
                device=self.model.device,
            ).unsqueeze(0)
            length_tensor = torch.tensor([sequence_length], dtype=torch.long, device=self.model.device)
            alpha = float(self.model(state_tensor, length_tensor).detach().cpu().numpy().reshape(-1)[0])
            if model_was_training:
                self.model.train()

        return alpha * np.asarray(pValues, dtype=np.float32)


def _write_model_config(save_dir, model, sequence_length):
    os.makedirs(save_dir, exist_ok=True)
    config = {
        "method": "GRU-IQL",
        "state_dim": STATE_DIM,
        "sequence_length": sequence_length,
        "encoder_hidden_dim": model.encoder_hidden_dim,
        "actor_lr": model.actor_lr,
        "critic_lr": model.critic_lr,
        "value_lr": model.V_lr,
        "expectile": model.expectile,
        "temperature": model.temperature,
        "gamma": model.GAMMA,
        "tau": model.tau,
    }
    with open(os.path.join(save_dir, "model_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)


def save_temporal_iql_checkpoint(model, normalize_dict, save_root, step, sequence_length):
    checkpoint_dir = os.path.join(save_root, "checkpoints", f"step_{step:05d}")
    model.save_checkpoint(checkpoint_dir)
    save_normalize_dict(normalize_dict, checkpoint_dir)
    _write_model_config(checkpoint_dir, model, sequence_length)
    return checkpoint_dir


def build_temporal_iql_metric_tracker(
    model,
    normalize_dict,
    save_root,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
    validation_data_path=DEFAULT_VALIDATION_DATA_PATH,
    eval_interval=1000,
    max_validation_groups=32,
):
    if eval_interval <= 0:
        return None

    def strategy_factory():
        return TemporalIqlValidationStrategy(
            model=model,
            normalize_dict=normalize_dict,
            sequence_length=sequence_length,
        )

    def evaluator():
        return evaluate_offline_bidding_strategy(
            strategy_factory=strategy_factory,
            file_path=validation_data_path,
            max_groups=max_validation_groups,
        )

    def checkpoint_saver(step):
        return save_temporal_iql_checkpoint(
            model=model,
            normalize_dict=normalize_dict,
            save_root=save_root,
            step=step,
            sequence_length=sequence_length,
        )

    logger.info(
        (
            "Temporal IQL offline metric tracking enabled: eval_interval=%s "
            "validation_data_path=%s max_validation_groups=%s sequence_length=%s"
        ),
        eval_interval,
        validation_data_path,
        max_validation_groups,
        sequence_length,
    )
    return OfflineMetricTracker(
        output_dir=save_root,
        evaluator=evaluator,
        checkpoint_saver=checkpoint_saver,
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
    sequence_length,
):
    report_path = os.path.join(save_root, "training_report.md")
    image_path = os.path.join(save_root, "training_curves.png")
    train_period_summary = _format_period_summary(training_data.get("deliveryPeriodIndex", pd.Series(dtype=float)))
    test_period_summary = _format_test_data_summary(validation_data_path)

    preprocessing = (
        f"16-dim base state features; {sequence_length}-step padded GRU sequences; "
        f"min-max normalization on feature indices {NORMALIZE_INDICES}; "
        f"continuous reward normalization used for training"
    )
    learning_rate = (
        f"actor_lr={model.actor_lr}, critic_lr={model.critic_lr}, "
        f"value_lr={model.V_lr}, encoder_hidden_dim={model.encoder_hidden_dim}"
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

    other_metrics.extend(
        [
            f"Training/validation split follows PPO-style holdout: train {train_period_summary}, validate {test_period_summary}.",
            f"Temporal encoder uses the last {sequence_length} normalized states only; past actions/rewards are not separate GRU inputs.",
        ]
    )

    visualization = "[image #1](training_curves.png)" if os.path.exists(image_path) else "N/A"

    report_lines = [
        "Method: GRU-IQL",
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


def train_model_steps(
    model,
    replay_buffer,
    metric_tracker=None,
    step_num=DEFAULT_TRAIN_STEPS,
    batch_size=DEFAULT_BATCH_SIZE,
    log_interval=DEFAULT_LOG_INTERVAL,
):
    for i in range(step_num):
        (
            state_sequences,
            sequence_lengths,
            actions,
            rewards,
            next_state_sequences,
            next_sequence_lengths,
            terminals,
        ) = replay_buffer.sample(batch_size)
        q_loss, v_loss, a_loss = model.step(
            state_sequences,
            sequence_lengths,
            actions,
            rewards,
            next_state_sequences,
            next_sequence_lengths,
            terminals,
        )
        step = i + 1
        if step == 1 or step % log_interval == 0 or step == step_num:
            logger.info(
                "Step: %s Q_loss: %s V_loss: %s A_loss: %s",
                step,
                q_loss,
                v_loss,
                a_loss,
            )
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
    (
        state_sequences,
        sequence_lengths,
        actions,
        rewards,
        next_state_sequences,
        next_sequence_lengths,
        terminals,
    ) = replay_buffer.sample(100)
    pred_actions = model.take_actions(state_sequences, sequence_lengths)
    _ = rewards, next_state_sequences, next_sequence_lengths, terminals
    action_pairs = np.concatenate((actions.cpu().detach().numpy(), pred_actions), axis=1)
    logger.info("action VS pred action: %s", action_pairs)


def train_iql_temporal_model(
    step_num=DEFAULT_TRAIN_STEPS,
    eval_interval=1000,
    validation_data_path=DEFAULT_VALIDATION_DATA_PATH,
    max_validation_groups=32,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    encoder_hidden_dim=DEFAULT_HIDDEN_DIM,
    log_interval=DEFAULT_LOG_INTERVAL,
):
    train_data_path = ensure_train_data_cache()
    training_data = pd.read_csv(train_data_path)
    save_root = DEFAULT_SAVE_ROOT

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)

    normalize_dict = normalize_state(
        training_data,
        STATE_DIM,
        normalize_indices=NORMALIZE_INDICES,
    )
    normalize_reward(training_data, "reward_continuous")
    save_normalize_dict(normalize_dict, save_root)

    replay_buffer = TemporalReplayBuffer.from_dataframe(
        training_data=training_data,
        seq_len=sequence_length,
        state_dim=STATE_DIM,
        is_normalize=True,
    )
    logger.info("Temporal replay buffer size: %s", len(replay_buffer))

    model = GRUIQL(
        dim_obs=STATE_DIM,
        seq_len=sequence_length,
        encoder_hidden_dim=encoder_hidden_dim,
    )
    _write_model_config(save_root, model, sequence_length)

    metric_tracker = build_temporal_iql_metric_tracker(
        model=model,
        normalize_dict=normalize_dict,
        save_root=save_root,
        sequence_length=sequence_length,
        validation_data_path=validation_data_path,
        eval_interval=eval_interval,
        max_validation_groups=max_validation_groups,
    )
    train_model_steps(
        model=model,
        replay_buffer=replay_buffer,
        metric_tracker=metric_tracker,
        step_num=step_num,
        batch_size=batch_size,
        log_interval=log_interval,
    )

    model.save_checkpoint(save_root)
    save_normalize_dict(normalize_dict, save_root)
    _write_model_config(save_root, model, sequence_length)
    _write_training_report(
        save_root=save_root,
        training_data=training_data,
        validation_data_path=validation_data_path,
        model=model,
        metric_tracker=metric_tracker,
        step_num=step_num,
        sequence_length=sequence_length,
    )
    test_trained_model(model, replay_buffer)


def run_iql_temporal():
    train_iql_temporal_model()


if __name__ == "__main__":
    run_iql_temporal()
