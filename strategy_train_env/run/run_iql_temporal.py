import argparse
import json
import logging
import os
import pickle
import re
import shutil
import signal

import numpy as np
import pandas as pd
import torch

from bidding_train_env.baseline.iql_temporal.gru_iql import GRUIQL
from bidding_train_env.baseline.iql_temporal.sequence_utils import (
    BASE_STATE_DIM,
    STATE_ONLY_SCHEMA,
    TOKEN_SCHEMA_DIMS,
    TOKEN_SCHEMA_FIELDS,
    TRANSITION_V1_SCHEMA,
    TRANSITION_V2_SCHEMA,
    TemporalContextBuffer,
    TemporalReplayBuffer,
    add_temporal_token_columns,
    apply_normalize,
    apply_token_normalize,
    build_iql_flat_state,
    build_temporal_token,
    previous_features_from_history,
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
DEFAULT_RAW_DATA_DIR = "./data/traffic"
DEFAULT_VALIDATION_DATA_PATH = "./data/traffic/period-27.csv"
DEFAULT_TRAIN_STEPS = 100000
DEFAULT_SEQUENCE_LENGTH = 8
DEFAULT_BATCH_SIZE = 256
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LOG_INTERVAL = 100
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_STATE_SAVE_INTERVAL = 1000
DEFAULT_SAVE_ROOT = "saved_model/IQLTemporaltest"

PHASE_CONFIGS = {
    1: {
        "phase_name": "checkpoint_hygiene",
        "save_root": "saved_model/IQLTemporal_phase1_checkpoint_hygiene",
        "architecture_summary": "Legacy separate-encoder GRU-IQL; checkpoint selection/reporting only.",
        "token_schema": STATE_ONLY_SCHEMA,
        "token_dim": STATE_DIM,
        "use_residual_latest_state": False,
        "shared_encoder": False,
        "notes": ["Changes selection/reporting only; model architecture remains the original GRU-IQL."],
    },
    2: {
        "phase_name": "residual_gru",
        "save_root": "saved_model/IQLTemporal_phase2_residual_gru",
        "architecture_summary": "Separate-encoder GRU-IQL with residual latest-token path and LayerNorm.",
        "token_schema": STATE_ONLY_SCHEMA,
        "token_dim": STATE_DIM,
        "use_residual_latest_state": True,
        "shared_encoder": False,
        "notes": ["Tests whether the GRU bottleneck hides important instantaneous bidding state."],
    },
    3: {
        "phase_name": "transition_v1",
        "save_root": "saved_model/IQLTemporal_phase3_transition_v1",
        "architecture_summary": "Residual separate-encoder GRU-IQL over transition-aware v1 temporal tokens.",
        "token_schema": TRANSITION_V1_SCHEMA,
        "token_dim": TOKEN_SCHEMA_DIMS[TRANSITION_V1_SCHEMA],
        "use_residual_latest_state": True,
        "shared_encoder": False,
        "notes": ["Adds previous action/reward/budget/done features from the RL cache."],
    },
    4: {
        "phase_name": "transition_v2",
        "save_root": "saved_model/IQLTemporal_phase4_transition_v2",
        "architecture_summary": "Residual separate-encoder GRU-IQL over raw-outcome transition v2 tokens.",
        "token_schema": TRANSITION_V2_SCHEMA,
        "token_dim": TOKEN_SCHEMA_DIMS[TRANSITION_V2_SCHEMA],
        "use_residual_latest_state": True,
        "shared_encoder": False,
        "notes": ["Adds previous cost, win-rate, market-price, and volume fields from raw period CSVs."],
    },
    5: {
        "phase_name": "shared_encoder",
        "save_root": "saved_model/IQLTemporal_phase5_shared_encoder",
        "architecture_summary": "Shared temporal encoder with separate actor/value/Q heads over transition v2 tokens.",
        "token_schema": TRANSITION_V2_SCHEMA,
        "token_dim": TOKEN_SCHEMA_DIMS[TRANSITION_V2_SCHEMA],
        "use_residual_latest_state": True,
        "shared_encoder": True,
        "notes": ["Uses one shared temporal representation and a combined optimizer update."],
    },
}

_STOP_REQUESTED = False
_STOP_SIGNAL = None


def _set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _request_stop(signum, frame):
    del frame
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum


def _install_signal_handlers():
    for signal_name in ("SIGUSR1", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _request_stop)


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


def _temporal_outcome_cache_path(train_data_dir=DEFAULT_TRAIN_DATA_DIR, periods=TRAIN_PERIODS):
    return os.path.join(
        train_data_dir,
        f"training_data_period-{periods[0]}-{periods[-1]}-temporalOutcome.csv",
    )


def ensure_temporal_outcome_cache(
    training_data,
    raw_data_dir=DEFAULT_RAW_DATA_DIR,
    train_data_dir=DEFAULT_TRAIN_DATA_DIR,
    periods=TRAIN_PERIODS,
):
    """Merge per-tick raw outcome summaries needed by transition_v2 tokens."""
    cache_path = _temporal_outcome_cache_path(train_data_dir=train_data_dir, periods=periods)
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    outcome_frames = []
    group_cols = ["deliveryPeriodIndex", "advertiserNumber", "timeStepIndex"]
    for period in periods:
        period_path = os.path.join(raw_data_dir, f"period-{period}.csv")
        if not os.path.exists(period_path):
            raise FileNotFoundError(f"Missing raw period data for transition_v2: {period_path}")
        raw_period = pd.read_csv(period_path)
        outcome = raw_period.groupby(group_cols).agg(
            tick_cost=("cost", "sum"),
            tick_win_rate=("xi", "mean"),
            tick_mean_lwc=("leastWinningCost", "mean"),
            tick_num_pv=("pvIndex", "count"),
        ).reset_index()
        outcome_frames.append(outcome)

    outcomes = pd.concat(outcome_frames, ignore_index=True)
    merged = training_data.merge(outcomes, on=group_cols, how="left")
    for column in ["tick_cost", "tick_win_rate", "tick_mean_lwc", "tick_num_pv"]:
        merged[column] = merged[column].fillna(0.0)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        merged.to_csv(cache_path, index=False)
        logger.info("Saved temporal outcome cache to %s", cache_path)
    except OSError as exc:
        logger.warning("Could not write temporal outcome cache to %s: %s", cache_path, exc)
    return merged


class TemporalIqlValidationStrategy(BaseBiddingStrategy):
    """Validation strategy that runs temporal IQL over state or transition tokens."""

    def __init__(
        self,
        model,
        normalize_dict,
        sequence_length,
        state_dim=STATE_DIM,
        token_schema=STATE_ONLY_SCHEMA,
        token_normalize_dict=None,
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
        self.token_schema = token_schema
        self.token_normalize_dict = token_normalize_dict
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
        if self.token_schema == STATE_ONLY_SCHEMA:
            token = apply_normalize(flat_state, self.normalize_dict)
        else:
            previous_features = previous_features_from_history(
                token_schema=self.token_schema,
                budget=self.budget,
                history_pvalue_info=historyPValueInfo,
                history_bid=historyBid,
                history_auction_result=historyAuctionResult,
                history_impression_result=historyImpressionResult,
                history_least_winning_cost=historyLeastWinningCost,
            )
            token = build_temporal_token(flat_state, self.token_schema, previous_features)
            token = apply_token_normalize(token, self.token_normalize_dict)
        self.context_buffer.append(token)
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


def _model_config(model, sequence_length, phase_config=None):
    phase_config = phase_config or {}
    return {
        "method": "GRU-IQL",
        "state_dim": STATE_DIM,
        "base_state_dim": BASE_STATE_DIM,
        "token_dim": model.num_of_states,
        "token_schema": phase_config.get("token_schema", STATE_ONLY_SCHEMA),
        "normalization_schema": (
            "legacy_state_normalization"
            if phase_config.get("token_schema", STATE_ONLY_SCHEMA) == STATE_ONLY_SCHEMA
            else "temporal_token_minmax"
        ),
        "sequence_length": sequence_length,
        "encoder_hidden_dim": model.encoder_hidden_dim,
        "use_residual_latest_state": model.use_residual_latest_state,
        "shared_encoder": model.shared_encoder,
        "actor_lr": model.actor_lr,
        "critic_lr": model.critic_lr,
        "value_lr": model.V_lr,
        "expectile": model.expectile,
        "temperature": model.temperature,
        "gamma": model.GAMMA,
        "tau": model.tau,
    }


def _write_model_config(save_dir, model, sequence_length, phase_config=None):
    os.makedirs(save_dir, exist_ok=True)
    config = _model_config(model, sequence_length, phase_config=phase_config)
    with open(os.path.join(save_dir, "model_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)


def save_token_normalize_dict(token_normalize_dict, save_dir):
    if token_normalize_dict is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "token_normalize_dict.pkl"), "wb") as file:
        pickle.dump(token_normalize_dict, file)


def save_temporal_iql_checkpoint(
    model,
    normalize_dict,
    save_root,
    step,
    sequence_length,
    phase_config=None,
    token_normalize_dict=None,
):
    checkpoint_dir = os.path.join(save_root, "checkpoints", f"step_{step:05d}")
    model.save_checkpoint(checkpoint_dir)
    save_normalize_dict(normalize_dict, checkpoint_dir)
    save_token_normalize_dict(token_normalize_dict, checkpoint_dir)
    _write_model_config(checkpoint_dir, model, sequence_length, phase_config=phase_config)
    return checkpoint_dir


def promote_best_sparse_checkpoint(metrics, save_root):
    checkpoint_path = metrics.get("checkpoint_path")
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        logger.warning("Cannot promote best sparse checkpoint; missing checkpoint_path=%s", checkpoint_path)
        return
    best_dir = os.path.join(save_root, "best_sparse_model")
    tmp_dir = f"{best_dir}.tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    shutil.copytree(checkpoint_path, tmp_dir)
    metadata_path = os.path.join(tmp_dir, "best_sparse_metrics.json")
    with open(metadata_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    os.replace(tmp_dir, best_dir)
    logger.info("Promoted best sparse checkpoint to %s", best_dir)


def _optimizer_state_dicts(model):
    return model.optimizer_state_dicts()


def _move_optimizer_state_to_device(optimizer, device):
    def move_value(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move_value(item) for item in value)
        return value

    optimizer.state = move_value(optimizer.state)


def _load_optimizer_state_dicts(model, optimizer_states):
    for name, optimizer in model.optimizer_map().items():
        state_dict = optimizer_states.get(name)
        if state_dict is None:
            logger.warning("Training state did not contain optimizer state for %s", name)
            continue
        optimizer.load_state_dict(state_dict)
        _move_optimizer_state_to_device(optimizer, model.device)


def _capture_rng_state():
    rng_state = {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    else:
        rng_state["torch_cuda_all"] = None
    return rng_state


def _restore_rng_state(rng_state):
    if not rng_state:
        logger.warning("Training state did not contain RNG state")
        return
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch_cpu") is not None:
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
    cuda_rng_state = rng_state.get("torch_cuda_all")
    if cuda_rng_state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        except RuntimeError as exc:
            logger.warning("Could not restore CUDA RNG state: %s", exc)


def _atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _latest_training_state_path(save_root):
    return os.path.join(save_root, "training_state_latest.pt")


def _step_training_state_path(save_root, step):
    return os.path.join(save_root, "checkpoints", f"step_{step:05d}", "training_state.pt")


def _resolve_resume_state_path(save_root, resume):
    if resume is None:
        return None
    resume_value = str(resume).strip()
    if resume_value.lower() in {"", "none", "false", "0", "no"}:
        return None
    if resume_value.lower() == "latest":
        latest_path = _latest_training_state_path(save_root)
        return latest_path if os.path.exists(latest_path) else None
    if os.path.isdir(resume_value):
        candidate = os.path.join(resume_value, "training_state.pt")
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(resume_value, "training_state_latest.pt")
        if os.path.exists(candidate):
            return candidate
    return resume_value


def load_temporal_iql_training_state(state_path, map_location="cpu"):
    if state_path is None:
        return None
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Could not find temporal IQL training state: {state_path}")
    return _torch_load(state_path, map_location=map_location)


def save_temporal_iql_training_state(
    model,
    normalize_dict,
    save_root,
    step,
    sequence_length,
    total_steps,
    phase_config=None,
    token_normalize_dict=None,
    save_step_copy=True,
):
    payload = {
        "schema_version": 1,
        "global_step": int(step),
        "total_steps": int(total_steps),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": _optimizer_state_dicts(model),
        "model_config": _model_config(model, sequence_length, phase_config=phase_config),
        "phase_config": phase_config,
        "normalize_dict": normalize_dict,
        "token_normalize_dict": token_normalize_dict,
        "rng_state": _capture_rng_state(),
    }

    if save_step_copy:
        step_path = _step_training_state_path(save_root, step)
        _atomic_torch_save(payload, step_path)
        logger.info("Saved temporal IQL recovery state to %s", step_path)

    latest_path = _latest_training_state_path(save_root)
    _atomic_torch_save(payload, latest_path)
    logger.info("Updated temporal IQL latest recovery state at step %s: %s", step, latest_path)
    return latest_path


def restore_temporal_iql_training_state(model, state_payload):
    model.load_state_dict(state_payload["model_state_dict"])
    _load_optimizer_state_dicts(model, state_payload.get("optimizer_state_dicts", {}))
    _restore_rng_state(state_payload.get("rng_state"))
    model.to(model.device)
    return int(state_payload.get("global_step", 0))


def _restore_metric_tracker_state(metric_tracker):
    if metric_tracker is None:
        return
    if os.path.exists(metric_tracker.metrics_path):
        metric_tracker.training_tracker = metric_tracker.training_tracker.__class__.load(metric_tracker.metrics_path)
        logger.info("Restored %s previous validation metric rows", len(metric_tracker.training_tracker.iterations))
    if os.path.exists(metric_tracker.best_path):
        with open(metric_tracker.best_path, encoding="utf-8") as best_file:
            best_metrics = json.load(best_file)
        for metric_name, best_entry in best_metrics.items():
            if metric_name in metric_tracker.best_metrics:
                metric_tracker.best_metrics[metric_name].update(best_entry)
        logger.info("Restored previous best validation metrics from %s", metric_tracker.best_path)


def build_temporal_iql_metric_tracker(
    model,
    normalize_dict,
    save_root,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
    state_dim=STATE_DIM,
    token_schema=STATE_ONLY_SCHEMA,
    token_normalize_dict=None,
    phase_config=None,
    validation_data_path=DEFAULT_VALIDATION_DATA_PATH,
    eval_interval=DEFAULT_EVAL_INTERVAL,
    max_validation_groups=32,
):
    if eval_interval <= 0:
        return None

    def strategy_factory():
        return TemporalIqlValidationStrategy(
            model=model,
            normalize_dict=normalize_dict,
            sequence_length=sequence_length,
            state_dim=state_dim,
            token_schema=token_schema,
            token_normalize_dict=token_normalize_dict,
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
            phase_config=phase_config,
            token_normalize_dict=token_normalize_dict,
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
        best_checkpoint_promoter=lambda metrics: promote_best_sparse_checkpoint(metrics, save_root),
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


def resolve_phase_config(phase, save_root=None):
    phase = int(phase)
    if phase not in PHASE_CONFIGS:
        raise ValueError(f"Unknown temporal IQL phase {phase}; expected one of {sorted(PHASE_CONFIGS)}")
    phase_config = dict(PHASE_CONFIGS[phase])
    phase_config["phase"] = phase
    if save_root is not None:
        phase_config["save_root"] = save_root
    return phase_config


def _best_sparse_entry(metric_tracker):
    if metric_tracker is None:
        return {}
    return metric_tracker.best_metrics.get("sparse_raw_score", {}) or {}


def _latest_iteration(metric_tracker):
    if metric_tracker is None or not metric_tracker.training_tracker.iterations:
        return None
    return metric_tracker.training_tracker.iterations[-1]


def _baseline_best_score(save_root, baseline_name):
    saved_model_root = os.path.dirname(os.path.abspath(save_root))
    best_path = os.path.join(saved_model_root, baseline_name, "best_metrics.json")
    if not os.path.exists(best_path):
        return None
    with open(best_path, encoding="utf-8") as best_file:
        best_metrics = json.load(best_file)
    return best_metrics.get("sparse_raw_score", {}).get("value")


def _prior_phase_best_score(phase, save_root):
    if phase <= 1:
        return None
    prior_config = PHASE_CONFIGS.get(phase - 1)
    if prior_config is None:
        return None
    prior_root = os.path.join(os.path.dirname(os.path.abspath(save_root)), os.path.basename(prior_config["save_root"]))
    best_path = os.path.join(prior_root, "best_metrics.json")
    if not os.path.exists(best_path):
        return None
    with open(best_path, encoding="utf-8") as best_file:
        best_metrics = json.load(best_file)
    return best_metrics.get("sparse_raw_score", {}).get("value")


def _comparison_text(score, baseline_score, label):
    if score is None:
        return f"{label}: current score unavailable."
    if baseline_score is None:
        return f"{label}: baseline not found."
    delta = score - baseline_score
    outcome = "beat" if delta > 0 else "did not beat"
    return f"{label}: {outcome} baseline by {delta:.4f} ({score:.4f} vs {baseline_score:.4f})."


def _write_phase_metadata(
    save_root,
    phase_config,
    model,
    metric_tracker,
    training_data,
    validation_data_path,
    sequence_length,
):
    best_entry = _best_sparse_entry(metric_tracker)
    metadata = {
        "phase": phase_config["phase"],
        "phase_name": phase_config["phase_name"],
        "parent_baseline": "IQLTemporaltest",
        "architecture_summary": phase_config["architecture_summary"],
        "token_schema": phase_config["token_schema"],
        "token_fields": TOKEN_SCHEMA_FIELDS.get(phase_config["token_schema"], []),
        "state_dim": STATE_DIM,
        "base_state_dim": BASE_STATE_DIM,
        "token_dim": model.num_of_states,
        "sequence_length": sequence_length,
        "selection_metric": "sparse_raw_score",
        "constraint_metric": "cpa_violation_rate",
        "train_periods": _format_period_summary(training_data.get("deliveryPeriodIndex", pd.Series(dtype=float))),
        "validation_period": _format_test_data_summary(validation_data_path),
        "best_checkpoint": best_entry.get("checkpoint_path"),
        "best_sparse_raw_score": best_entry.get("value"),
        "use_residual_latest_state": model.use_residual_latest_state,
        "shared_encoder": model.shared_encoder,
        "notes": phase_config.get("notes", []),
    }
    with open(os.path.join(save_root, "phase_metadata.json"), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def _write_phase_report(
    save_root,
    phase_config,
    model,
    metric_tracker,
    training_data,
    validation_data_path,
    final_step,
    sequence_length,
    batch_size,
    eval_interval,
):
    best_entry = _best_sparse_entry(metric_tracker)
    latest = _latest_iteration(metric_tracker)
    best_score = best_entry.get("value")
    prior_score = _prior_phase_best_score(phase_config["phase"], save_root)
    plain_iql_score = _baseline_best_score(save_root, "IQLtest")
    phase_report_path = os.path.join(save_root, "phase_report.md")
    lines = [
        f"# Temporal IQL Phase {phase_config['phase']}: {phase_config['phase_name']}",
        "",
        "## What Changed",
        f"- {phase_config['architecture_summary']}",
        f"- Token schema: `{phase_config['token_schema']}` with token_dim={model.num_of_states}.",
        f"- Residual latest-state path: {model.use_residual_latest_state}.",
        f"- Shared encoder: {model.shared_encoder}.",
        "",
        "## Compatibility",
        "- Uses the same period-7 to period-26 training split and period-27 validation split.",
        "- Keeps the actor output as a one-dimensional bid multiplier.",
        "- Saves final model separately from `best_sparse_model/`.",
        "",
        "## Training Config",
        f"- total/final step: {final_step}",
        f"- batch_size: {batch_size}",
        f"- sequence_length: {sequence_length}",
        f"- eval_interval: {eval_interval}",
        f"- actor_lr={model.actor_lr}, critic_lr={model.critic_lr}, value_lr={model.V_lr}",
        "",
        "## Metrics",
    ]
    if latest is not None:
        lines.extend(
            [
                f"- Final validation score: {latest.mean_eval_score:.4f} @ step {latest.step}",
                f"- Final reward: {latest.mean_eval_reward:.4f}",
                f"- Final budget utilization: {latest.mean_budget_utilization:.4f}",
                f"- Final train/loss: {latest.train_loss:.4f}",
            ]
        )
    else:
        lines.append("- Final validation score: unavailable.")
    if best_score is not None:
        lines.append(f"- Best sparse score: {best_score:.4f} @ step {best_entry.get('step')}")
        lines.append(f"- Best checkpoint: {best_entry.get('checkpoint_path')}")
    else:
        lines.append("- Best sparse score: unavailable.")
    lines.extend(
        [
            "",
            "## Comparison",
            f"- {_comparison_text(best_score, prior_score, 'Prior phase')}",
            f"- {_comparison_text(best_score, plain_iql_score, 'Plain IQLtest')}",
            "",
            "## Notes",
        ]
    )
    lines.extend(f"- {note}" for note in phase_config.get("notes", []))
    lines.append("")
    with open(phase_report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))


def _write_training_report(
    save_root,
    training_data,
    validation_data_path,
    model,
    metric_tracker,
    step_num,
    sequence_length,
    phase_config=None,
):
    report_path = os.path.join(save_root, "training_report.md")
    image_path = os.path.join(save_root, "training_curves.png")
    train_period_summary = _format_period_summary(training_data.get("deliveryPeriodIndex", pd.Series(dtype=float)))
    test_period_summary = _format_test_data_summary(validation_data_path)

    phase_config = phase_config or {"token_schema": STATE_ONLY_SCHEMA}
    if phase_config.get("token_schema") == STATE_ONLY_SCHEMA:
        preprocessing = (
            f"16-dim base state features; {sequence_length}-step padded GRU sequences; "
            f"min-max normalization on feature indices {NORMALIZE_INDICES}; "
            f"continuous reward normalization used for training"
        )
    else:
        preprocessing = (
            f"{phase_config.get('token_schema')} temporal tokens; token_dim={model.num_of_states}; "
            "token-level min-max normalization; continuous reward normalization used for training"
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
            f"Temporal encoder schema: {phase_config.get('token_schema', STATE_ONLY_SCHEMA)}.",
        ]
    )

    visualization = "[image #1](training_curves.png)" if os.path.exists(image_path) else "N/A"

    report_lines = [
        f"Method: GRU-IQL phase {phase_config.get('phase', 'legacy')} ({phase_config.get('phase_name', 'legacy')})",
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
    total_steps=DEFAULT_TRAIN_STEPS,
    start_step=0,
    batch_size=DEFAULT_BATCH_SIZE,
    log_interval=DEFAULT_LOG_INTERVAL,
    state_save_interval=DEFAULT_STATE_SAVE_INTERVAL,
    state_saver=None,
):
    last_step = start_step
    for step in range(start_step + 1, total_steps + 1):
        if _STOP_REQUESTED:
            signal_name = signal.Signals(_STOP_SIGNAL).name if _STOP_SIGNAL is not None else "unknown"
            logger.warning(
                "Received %s; saving recovery state at completed step %s and stopping early",
                signal_name,
                last_step,
            )
            if state_saver is not None:
                state_saver(last_step)
            return last_step, True

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
        last_step = step
        if step == 1 or step % log_interval == 0 or step == total_steps:
            logger.info(
                "Step: %s Q_loss: %s V_loss: %s A_loss: %s",
                step,
                q_loss,
                v_loss,
                a_loss,
            )

        if _STOP_REQUESTED:
            signal_name = signal.Signals(_STOP_SIGNAL).name if _STOP_SIGNAL is not None else "unknown"
            logger.warning("Received %s; saving recovery state at step %s and stopping early", signal_name, step)
            if state_saver is not None:
                state_saver(step)
            return step, True

        if (
            state_saver is not None
            and state_save_interval > 0
            and (step % state_save_interval == 0 or step == total_steps)
        ):
            state_saver(step)

        if metric_tracker is not None:
            metric_tracker.maybe_evaluate(
                step,
                extra_metrics={
                    "train_q_loss": float(q_loss),
                    "train_v_loss": float(v_loss),
                    "train_a_loss": float(a_loss),
                },
                force=(step == total_steps),
            )
    return last_step, False


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
    total_steps=None,
    step_num=None,
    resume="none",
    save_root=None,
    phase=1,
    eval_interval=DEFAULT_EVAL_INTERVAL,
    validation_data_path=DEFAULT_VALIDATION_DATA_PATH,
    max_validation_groups=32,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    encoder_hidden_dim=DEFAULT_HIDDEN_DIM,
    log_interval=DEFAULT_LOG_INTERVAL,
    state_save_interval=DEFAULT_STATE_SAVE_INTERVAL,
    raw_data_dir=DEFAULT_RAW_DATA_DIR,
    seed=1,
):
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = False
    _STOP_SIGNAL = None
    _install_signal_handlers()
    phase_config = resolve_phase_config(phase, save_root=save_root)
    save_root = phase_config["save_root"]
    os.makedirs(os.path.join(save_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "best_sparse_model"), exist_ok=True)

    if total_steps is None:
        total_steps = step_num if step_num is not None else DEFAULT_TRAIN_STEPS
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    if seed is not None:
        _set_random_seeds(seed)

    resume_state_path = _resolve_resume_state_path(save_root, resume)
    resume_payload = None
    start_step = 0
    if resume_state_path is not None:
        resume_payload = load_temporal_iql_training_state(resume_state_path, map_location="cpu")
        start_step = int(resume_payload.get("global_step", 0))
        resume_config = resume_payload.get("model_config", {})
        saved_phase_config = resume_payload.get("phase_config")
        if saved_phase_config is not None:
            phase_config.update(saved_phase_config)
        saved_sequence_length = resume_config.get("sequence_length")
        saved_hidden_dim = resume_config.get("encoder_hidden_dim")
        saved_token_schema = resume_config.get("token_schema")
        saved_token_dim = resume_config.get("token_dim")
        saved_residual = resume_config.get("use_residual_latest_state")
        saved_shared = resume_config.get("shared_encoder")
        if saved_sequence_length is not None and int(saved_sequence_length) != sequence_length:
            logger.warning(
                "Using sequence_length=%s from resume state instead of requested sequence_length=%s",
                saved_sequence_length,
                sequence_length,
            )
            sequence_length = int(saved_sequence_length)
        if saved_hidden_dim is not None and int(saved_hidden_dim) != encoder_hidden_dim:
            logger.warning(
                "Using encoder_hidden_dim=%s from resume state instead of requested encoder_hidden_dim=%s",
                saved_hidden_dim,
                encoder_hidden_dim,
            )
            encoder_hidden_dim = int(saved_hidden_dim)
        if saved_token_schema is not None and saved_token_schema != phase_config["token_schema"]:
            logger.warning(
                "Using token_schema=%s from resume state instead of requested token_schema=%s",
                saved_token_schema,
                phase_config["token_schema"],
            )
            phase_config["token_schema"] = saved_token_schema
        if saved_token_dim is not None:
            phase_config["token_dim"] = int(saved_token_dim)
        if saved_residual is not None:
            phase_config["use_residual_latest_state"] = bool(saved_residual)
        if saved_shared is not None:
            phase_config["shared_encoder"] = bool(saved_shared)
        logger.info("Loaded temporal IQL recovery state from %s at global step %s", resume_state_path, start_step)
    elif str(resume).strip().lower() == "latest":
        logger.info("No latest temporal IQL recovery state found under %s; starting from scratch", save_root)

    if start_step >= total_steps:
        logger.info(
            "Temporal IQL already reached target: global_step=%s total_steps=%s. Nothing to train.",
            start_step,
            total_steps,
        )
        return

    train_data_path = ensure_train_data_cache()
    training_data = pd.read_csv(train_data_path)
    if phase_config["token_schema"] == TRANSITION_V2_SCHEMA:
        training_data = ensure_temporal_outcome_cache(
            training_data=training_data,
            raw_data_dir=raw_data_dir,
        )

    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)

    normalize_dict = normalize_state(
        training_data,
        STATE_DIM,
        normalize_indices=NORMALIZE_INDICES,
    )
    normalize_reward(training_data, "reward_continuous")
    if resume_payload is not None and resume_payload.get("normalize_dict") is not None:
        normalize_dict = resume_payload["normalize_dict"]
    save_normalize_dict(normalize_dict, save_root)
    token_normalize_dict = None
    replay_token_col = None
    replay_state_dim = STATE_DIM
    if phase_config["token_schema"] != STATE_ONLY_SCHEMA:
        token_normalize_dict = resume_payload.get("token_normalize_dict") if resume_payload is not None else None
        token_normalize_dict = add_temporal_token_columns(
            training_data,
            token_schema=phase_config["token_schema"],
            token_normalize_dict=token_normalize_dict,
        )
        save_token_normalize_dict(token_normalize_dict, save_root)
        replay_token_col = "normalize_token"
        replay_state_dim = phase_config["token_dim"]

    replay_buffer = TemporalReplayBuffer.from_dataframe(
        training_data=training_data,
        seq_len=sequence_length,
        state_dim=replay_state_dim,
        is_normalize=True,
        token_col=replay_token_col,
    )
    logger.info("Temporal replay buffer size: %s", len(replay_buffer))

    model = GRUIQL(
        dim_obs=replay_state_dim,
        seq_len=sequence_length,
        encoder_hidden_dim=encoder_hidden_dim,
        use_residual_latest_state=phase_config["use_residual_latest_state"],
        shared_encoder=phase_config["shared_encoder"],
    )
    if resume_payload is not None:
        start_step = restore_temporal_iql_training_state(model, resume_payload)
    _write_model_config(save_root, model, sequence_length, phase_config=phase_config)

    metric_tracker = build_temporal_iql_metric_tracker(
        model=model,
        normalize_dict=normalize_dict,
        save_root=save_root,
        sequence_length=sequence_length,
        state_dim=replay_state_dim,
        token_schema=phase_config["token_schema"],
        token_normalize_dict=token_normalize_dict,
        phase_config=phase_config,
        validation_data_path=validation_data_path,
        eval_interval=eval_interval,
        max_validation_groups=max_validation_groups,
    )
    if resume_payload is not None:
        _restore_metric_tracker_state(metric_tracker)

    def state_saver(step):
        return save_temporal_iql_training_state(
            model=model,
            normalize_dict=normalize_dict,
            save_root=save_root,
            step=step,
            sequence_length=sequence_length,
            total_steps=total_steps,
            phase_config=phase_config,
            token_normalize_dict=token_normalize_dict,
        )

    logger.info(
        (
            "Training temporal IQL from global step %s to %s "
            "(batch_size=%s, state_save_interval=%s, eval_interval=%s)"
        ),
        start_step,
        total_steps,
        batch_size,
        state_save_interval,
        eval_interval,
    )
    final_step, stopped_early = train_model_steps(
        model=model,
        replay_buffer=replay_buffer,
        metric_tracker=metric_tracker,
        total_steps=total_steps,
        start_step=start_step,
        batch_size=batch_size,
        log_interval=log_interval,
        state_save_interval=state_save_interval,
        state_saver=state_saver,
    )
    saved_in_loop = stopped_early or (
        state_save_interval > 0 and (final_step % state_save_interval == 0 or final_step == total_steps)
    )
    if final_step > start_step and not saved_in_loop:
        state_saver(final_step)
    if metric_tracker is not None:
        latest = _latest_iteration(metric_tracker)
        if final_step > 0 and (latest is None or int(latest.step) != int(final_step)):
            metric_tracker.maybe_evaluate(final_step, force=True)

    model.save_checkpoint(save_root)
    save_normalize_dict(normalize_dict, save_root)
    save_token_normalize_dict(token_normalize_dict, save_root)
    _write_model_config(save_root, model, sequence_length, phase_config=phase_config)
    _write_training_report(
        save_root=save_root,
        training_data=training_data,
        validation_data_path=validation_data_path,
        model=model,
        metric_tracker=metric_tracker,
        step_num=final_step,
        sequence_length=sequence_length,
        phase_config=phase_config,
    )
    _write_phase_metadata(
        save_root=save_root,
        phase_config=phase_config,
        model=model,
        metric_tracker=metric_tracker,
        training_data=training_data,
        validation_data_path=validation_data_path,
        sequence_length=sequence_length,
    )
    _write_phase_report(
        save_root=save_root,
        phase_config=phase_config,
        model=model,
        metric_tracker=metric_tracker,
        training_data=training_data,
        validation_data_path=validation_data_path,
        final_step=final_step,
        sequence_length=sequence_length,
        batch_size=batch_size,
        eval_interval=eval_interval,
    )
    if stopped_early:
        logger.info("Stopped early at global step %s; resubmit with --resume latest to continue", final_step)
    else:
        test_trained_model(model, replay_buffer)


def run_iql_temporal(**kwargs):
    train_iql_temporal_model(**kwargs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train temporal GRU-IQL with resumable recovery checkpoints.")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--phase", type=int, choices=sorted(PHASE_CONFIGS), default=1)
    parser.add_argument(
        "--resume",
        default="none",
        help="Use 'latest' to resume save_root/training_state_latest.pt, 'none' to start fresh, or pass a state path.",
    )
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--state-save-interval", type=int, default=DEFAULT_STATE_SAVE_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-validation-groups", type=int, default=32)
    parser.add_argument("--validation-data-path", default=DEFAULT_VALIDATION_DATA_PATH)
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--encoder-hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_iql_temporal_model(
        total_steps=args.total_steps,
        resume=args.resume,
        save_root=args.save_root,
        phase=args.phase,
        eval_interval=args.eval_interval,
        validation_data_path=args.validation_data_path,
        max_validation_groups=args.max_validation_groups,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        encoder_hidden_dim=args.encoder_hidden_dim,
        log_interval=args.log_interval,
        state_save_interval=args.state_save_interval,
        raw_data_dir=args.raw_data_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
