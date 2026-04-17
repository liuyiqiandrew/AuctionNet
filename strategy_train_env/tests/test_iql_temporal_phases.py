import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bidding_train_env.baseline.iql_temporal.gru_iql import GRUIQL
from bidding_train_env.baseline.iql_temporal.sequence_utils import (
    PREV_DONE_INDEX,
    TRANSITION_V1_SCHEMA,
    TRANSITION_V2_SCHEMA,
    TemporalReplayBuffer,
    add_temporal_token_columns,
)
from run.run_iql_temporal import (
    _write_phase_metadata,
    _write_phase_report,
    promote_best_sparse_checkpoint,
    resolve_phase_config,
)
from run.offline_metric_tracker import OfflineMetricTracker


def _state(budget_left):
    values = np.zeros(16, dtype=np.float32)
    values[0] = 1.0
    values[1] = budget_left
    values[12] = 0.1
    values[13] = 100.0
    return tuple(float(value) for value in values)


def _training_frame():
    return pd.DataFrame(
        [
            {
                "deliveryPeriodIndex": 7,
                "advertiserNumber": 1,
                "timeStepIndex": 0,
                "budget": 100.0,
                "state": _state(1.0),
                "next_state": _state(0.8),
                "action": 10.0,
                "reward": 1.0,
                "reward_continuous": 0.5,
                "done": 0.0,
                "normalize_reward": 0.5,
            },
            {
                "deliveryPeriodIndex": 7,
                "advertiserNumber": 1,
                "timeStepIndex": 1,
                "budget": 100.0,
                "state": _state(0.8),
                "next_state": _state(0.7),
                "action": 20.0,
                "reward": 2.0,
                "reward_continuous": 1.5,
                "done": 1.0,
                "normalize_reward": 1.0,
            },
            {
                "deliveryPeriodIndex": 7,
                "advertiserNumber": 2,
                "timeStepIndex": 0,
                "budget": 100.0,
                "state": _state(1.0),
                "next_state": _state(0.9),
                "action": 30.0,
                "reward": 3.0,
                "reward_continuous": 2.5,
                "done": 0.0,
                "normalize_reward": 0.75,
            },
        ]
    )


def test_transition_v1_previous_fields_align_to_prior_row_and_reset_by_group():
    frame = _training_frame()
    add_temporal_token_columns(frame, TRANSITION_V1_SCHEMA)

    first_token = np.asarray(frame.loc[0, "temporal_token"], dtype=np.float32)
    second_token = np.asarray(frame.loc[1, "temporal_token"], dtype=np.float32)
    reset_token = np.asarray(frame.loc[2, "temporal_token"], dtype=np.float32)

    assert first_token[16:21].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert second_token[16] == 10.0
    assert second_token[17] == 1.0
    assert second_token[18] == 0.5
    assert np.isclose(second_token[19], 0.2)
    assert second_token[PREV_DONE_INDEX] == 0.0
    assert reset_token[16:21].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_transition_v2_token_dim_and_replay_padding():
    frame = _training_frame()
    frame["tick_cost"] = [5.0, 6.0, 7.0]
    frame["tick_win_rate"] = [0.1, 0.2, 0.3]
    frame["tick_mean_lwc"] = [0.01, 0.02, 0.03]
    frame["tick_num_pv"] = [100, 200, 300]
    add_temporal_token_columns(frame, TRANSITION_V2_SCHEMA)

    replay = TemporalReplayBuffer.from_dataframe(
        frame,
        seq_len=2,
        state_dim=25,
        is_normalize=True,
        token_col="normalize_token",
    )

    assert replay.state_sequences.shape == (3, 2, 25)
    assert replay.sequence_lengths.tolist() == [1, 2, 1]
    assert replay.next_sequence_lengths.tolist() == [2, 1, 1]


def test_residual_and_shared_models_produce_expected_shapes():
    sequences = torch.zeros((4, 3, 21), dtype=torch.float32)
    lengths = torch.tensor([1, 2, 3, 3], dtype=torch.long)
    actions = torch.ones((4, 1), dtype=torch.float32)
    rewards = torch.ones((4, 1), dtype=torch.float32)
    dones = torch.zeros((4, 1), dtype=torch.float32)

    residual = GRUIQL(dim_obs=21, seq_len=3, use_residual_latest_state=True)
    assert residual.take_actions(sequences, lengths).shape == (4, 1)
    q_loss, v_loss, a_loss = residual.step(sequences, lengths, actions, rewards, sequences, lengths, dones)
    assert np.isfinite(float(q_loss))
    assert np.isfinite(float(v_loss))
    assert np.isfinite(float(a_loss))

    shared = GRUIQL(dim_obs=21, seq_len=3, use_residual_latest_state=True, shared_encoder=True)
    assert shared.take_actions(sequences, lengths).shape == (4, 1)
    assert list(shared.optimizer_state_dicts()) == ["shared_optimizer"]
    q_loss, v_loss, a_loss = shared.step(sequences, lengths, actions, rewards, sequences, lengths, dones)
    assert np.isfinite(float(q_loss))
    assert np.isfinite(float(v_loss))
    assert np.isfinite(float(a_loss))


def test_phase_metadata_and_report_are_written():
    frame = _training_frame()
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        phase_config = resolve_phase_config(3, save_root=str(tmp_path))
        model = GRUIQL(dim_obs=21, seq_len=2, use_residual_latest_state=True)

        _write_phase_metadata(
            save_root=str(tmp_path),
            phase_config=phase_config,
            model=model,
            metric_tracker=None,
            training_data=frame,
            validation_data_path="./data/traffic/period-27.csv",
            sequence_length=2,
        )
        _write_phase_report(
            save_root=str(tmp_path),
            phase_config=phase_config,
            model=model,
            metric_tracker=None,
            training_data=frame,
            validation_data_path="./data/traffic/period-27.csv",
            final_step=1,
            sequence_length=2,
            batch_size=4,
            eval_interval=1,
        )

        metadata = json.loads((tmp_path / "phase_metadata.json").read_text())
        assert metadata["phase"] == 3
        assert metadata["token_schema"] == TRANSITION_V1_SCHEMA
        assert metadata["token_dim"] == 21
        assert (tmp_path / "phase_report.md").exists()


def test_metric_tracker_promotes_best_sparse_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        def checkpoint_saver(step):
            checkpoint = root / "checkpoints" / f"step_{step:05d}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "marker.txt").write_text("checkpoint")
            return str(checkpoint)

        def evaluator():
            return {
                "continuous_raw_score": 8.0,
                "continuous_reward": 8.0,
                "cpa_exceedance_rate": -0.1,
                "cpa_violation_rate": 0.0,
                "budget_consumer_ratio": 0.9,
                "sparse_raw_score": 9.0,
                "sparse_reward": 9.0,
                "num_groups": 1,
            }

        tracker = OfflineMetricTracker(
            output_dir=str(root),
            evaluator=evaluator,
            checkpoint_saver=checkpoint_saver,
            best_checkpoint_promoter=lambda metrics: promote_best_sparse_checkpoint(metrics, str(root)),
            eval_interval=1,
        )
        tracker.maybe_evaluate(1, force=True)

        assert (root / "best_metrics.json").exists()
        assert (root / "metrics.json").exists()
        assert (root / "training_curve.csv").exists()
        assert (root / "training_curves.png").exists()
        assert (root / "best_sparse_model" / "marker.txt").read_text() == "checkpoint"


if __name__ == "__main__":
    test_transition_v1_previous_fields_align_to_prior_row_and_reset_by_group()
    test_transition_v2_token_dim_and_replay_padding()
    test_residual_and_shared_models_produce_expected_shapes()
    test_phase_metadata_and_report_are_written()
    test_metric_tracker_promotes_best_sparse_checkpoint()
