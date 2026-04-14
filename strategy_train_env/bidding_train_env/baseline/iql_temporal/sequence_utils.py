import ast
from collections import deque

import numpy as np
import torch


def safe_literal_eval(val):
    """Safely parse a serialized Python literal used in cached RL data.

    Parameters
    ----------
    val : object
        Raw value read from a pandas column. This is usually a stringified tuple
        or ``NaN``.

    Returns
    -------
    object
        Parsed Python object when possible, otherwise the original value.
    """
    if val is None:
        return val
    if isinstance(val, float) and np.isnan(val):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        try:
            return eval(val, {"__builtins__": {}}, {"np": np})
        except Exception:
            return val


def _mean_of_last_n_elements(history, n):
    """Average the means of the last ``n`` history entries.

    Parameters
    ----------
    history : list[array-like]
        Sequence of past per-tick arrays.
    n : int
        Number of most recent entries to include.

    Returns
    -------
    float
        Mean over the selected history window, or ``0.0`` when the history is
        empty.
    """
    last_n_data = history[max(0, len(history) - n):]
    if len(last_n_data) == 0:
        return 0.0
    return float(np.mean([np.mean(data) for data in last_n_data]))


def build_iql_flat_state(
    time_step_index,
    p_values,
    history_pvalue_info,
    history_bid,
    history_auction_result,
    history_impression_result,
    history_least_winning_cost,
    budget,
    remaining_budget,
):
    """Rebuild the original 16-dim IQL state from online evaluation history.

    Parameters
    ----------
    time_step_index : int
        Current tick index in the delivery period.
    p_values : array-like
        Current predicted conversion probabilities.
    history_pvalue_info : list[np.ndarray]
        Historical per-tick ``(pValue, pValueSigma)`` arrays.
    history_bid : list[np.ndarray]
        Historical bid arrays.
    history_auction_result : list[np.ndarray]
        Historical auction result arrays.
    history_impression_result : list[np.ndarray]
        Historical impression result arrays.
    history_least_winning_cost : list[np.ndarray]
        Historical least-winning-cost arrays.
    budget : float
        Total campaign budget.
    remaining_budget : float
        Remaining campaign budget before the current decision.

    Returns
    -------
    numpy.ndarray
        Flat 16-dimensional state used by the original IQL implementation.
    """
    time_left = (48 - time_step_index) / 48
    budget_left = remaining_budget / budget if budget > 0 else 0.0
    history_xi = [result[:, 0] for result in history_auction_result]
    history_pvalue = [result[:, 0] for result in history_pvalue_info]
    history_conversion = [result[:, 1] for result in history_impression_result]

    historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0.0
    historical_conversion_mean = (
        np.mean([np.mean(reward) for reward in history_conversion]) if history_conversion else 0.0
    )
    historical_least_winning_cost_mean = (
        np.mean([np.mean(price) for price in history_least_winning_cost]) if history_least_winning_cost else 0.0
    )
    historical_pvalues_mean = np.mean([np.mean(value) for value in history_pvalue]) if history_pvalue else 0.0
    historical_bid_mean = np.mean([np.mean(bid) for bid in history_bid]) if history_bid else 0.0

    last_three_xi_mean = _mean_of_last_n_elements(history_xi, 3)
    last_three_conversion_mean = _mean_of_last_n_elements(history_conversion, 3)
    last_three_least_winning_cost_mean = _mean_of_last_n_elements(history_least_winning_cost, 3)
    last_three_pvalues_mean = _mean_of_last_n_elements(history_pvalue, 3)
    last_three_bid_mean = _mean_of_last_n_elements(history_bid, 3)

    current_pvalues_mean = float(np.mean(p_values))
    current_pv_num = len(p_values)
    historical_pv_num_total = sum(len(bids) for bids in history_bid) if history_bid else 0
    last_three_pv_num_total = sum(len(bids) for bids in history_bid[-3:]) if history_bid else 0

    return np.array(
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


def apply_normalize(state, normalize_dict):
    """Apply per-feature min-max normalization using a saved stats dict.

    Parameters
    ----------
    state : array-like
        Flat state vector.
    normalize_dict : dict
        Mapping from feature index to normalization statistics with ``min`` and
        ``max`` entries.

    Returns
    -------
    numpy.ndarray
        Normalized copy of ``state``.
    """
    state = np.asarray(state, dtype=np.float32).copy()
    for key, value in normalize_dict.items():
        min_value = value["min"]
        max_value = value["max"]
        state[key] = (
            (state[key] - min_value) / (max_value - min_value + 0.01)
            if max_value >= min_value
            else 0.0
        )
    return state


class TemporalContextBuffer:
    """Maintain the last ``K`` normalized states for online inference.

    Parameters
    ----------
    seq_len : int
        Maximum number of states to retain.
    state_dim : int
        Dimension of each per-step state vector.
    """

    def __init__(self, seq_len, state_dim):
        self.seq_len = seq_len
        self.state_dim = state_dim
        self._buffer = deque(maxlen=seq_len)

    def reset(self):
        """Clear all cached history for a new episode."""
        self._buffer.clear()

    def append(self, state):
        """Append one normalized state to the history buffer.

        Parameters
        ----------
        state : array-like
            Per-step state vector to append.
        """
        self._buffer.append(np.asarray(state, dtype=np.float32))

    def as_padded_sequence(self):
        """Return the current history as a left-padded fixed-length sequence.

        Returns
        -------
        tuple[numpy.ndarray, int]
            Padded sequence of shape ``(seq_len, state_dim)`` and the number of
            valid, unpadded states it contains.
        """
        sequence = np.zeros((self.seq_len, self.state_dim), dtype=np.float32)
        if not self._buffer:
            return sequence, 1
        buffered = np.asarray(list(self._buffer), dtype=np.float32)
        # Left-pad with zeros so the most recent state stays at the end.
        sequence[-buffered.shape[0] :] = buffered
        return sequence, buffered.shape[0]


class TemporalReplayBuffer:
    """Replay buffer of fixed-length temporal windows for GRU-IQL.

    Parameters
    ----------
    state_sequences : array-like
        Current-state padded sequences.
    sequence_lengths : array-like
        Valid lengths for ``state_sequences``.
    actions : array-like
        Action batch.
    rewards : array-like
        Reward batch.
    next_state_sequences : array-like
        Next-state padded sequences.
    next_sequence_lengths : array-like
        Valid lengths for ``next_state_sequences``.
    dones : array-like
        Terminal indicators.
    """

    def __init__(
        self,
        state_sequences,
        sequence_lengths,
        actions,
        rewards,
        next_state_sequences,
        next_sequence_lengths,
        dones,
    ):
        self.state_sequences = np.ascontiguousarray(state_sequences, dtype=np.float32)
        self.sequence_lengths = np.ascontiguousarray(sequence_lengths, dtype=np.int64)
        self.actions = np.ascontiguousarray(actions, dtype=np.float32)
        self.rewards = np.ascontiguousarray(rewards, dtype=np.float32)
        self.next_state_sequences = np.ascontiguousarray(next_state_sequences, dtype=np.float32)
        self.next_sequence_lengths = np.ascontiguousarray(next_sequence_lengths, dtype=np.int64)
        self.dones = np.ascontiguousarray(dones, dtype=np.float32)

    @classmethod
    def from_dataframe(cls, training_data, seq_len, state_dim, is_normalize=True):
        """Build temporal transition windows from per-tick RL training data.

        Parameters
        ----------
        training_data : pandas.DataFrame
            RL dataframe containing per-tick states, rewards, actions, and
            episode identifiers.
        seq_len : int
            Maximum temporal context length.
        state_dim : int
            Dimension of each per-step state vector.
        is_normalize : bool, optional
            When ``True``, read ``normalize_state`` and ``normalize_reward``.
            Otherwise use raw ``state`` and ``reward`` columns.

        Returns
        -------
        TemporalReplayBuffer
            Buffer populated with one temporal transition per dataframe row.
        """
        state_col = "normalize_state" if is_normalize else "state"
        reward_col = "normalize_reward" if is_normalize else "reward"
        group_cols = ["deliveryPeriodIndex", "advertiserNumber"]
        sorted_data = training_data.sort_values(group_cols + ["timeStepIndex"])

        state_sequences = []
        sequence_lengths = []
        actions = []
        rewards = []
        next_state_sequences = []
        next_sequence_lengths = []
        dones = []

        for _, trajectory in sorted_data.groupby(group_cols, sort=False):
            trajectory_states = np.asarray(trajectory[state_col].to_list(), dtype=np.float32)
            trajectory_actions = trajectory["action"].to_numpy(dtype=np.float32).reshape(-1, 1)
            trajectory_rewards = trajectory[reward_col].to_numpy(dtype=np.float32).reshape(-1, 1)
            trajectory_dones = trajectory["done"].to_numpy(dtype=np.float32).reshape(-1, 1)
            num_steps = trajectory_states.shape[0]

            for idx in range(num_steps):
                # Build the current window ending at the current step.
                current_start = max(0, idx - seq_len + 1)
                current_slice = trajectory_states[current_start : idx + 1]
                current_length = current_slice.shape[0]
                current_sequence = np.zeros((seq_len, state_dim), dtype=np.float32)
                current_sequence[-current_length:] = current_slice

                if idx + 1 < num_steps and trajectory_dones[idx, 0] != 1:
                    # Shift the window by one step when a valid next state exists.
                    next_start = max(0, idx - seq_len + 2)
                    next_slice = trajectory_states[next_start : idx + 2]
                    next_length = next_slice.shape[0]
                    next_sequence = np.zeros((seq_len, state_dim), dtype=np.float32)
                    next_sequence[-next_length:] = next_slice
                else:
                    # Terminal transitions use an all-zero next sequence.
                    next_sequence = np.zeros((seq_len, state_dim), dtype=np.float32)
                    next_length = 1

                state_sequences.append(current_sequence)
                sequence_lengths.append(current_length)
                actions.append(trajectory_actions[idx])
                rewards.append(trajectory_rewards[idx])
                next_state_sequences.append(next_sequence)
                next_sequence_lengths.append(next_length)
                dones.append(trajectory_dones[idx])

        return cls(
            state_sequences=np.stack(state_sequences),
            sequence_lengths=np.asarray(sequence_lengths),
            actions=np.stack(actions),
            rewards=np.stack(rewards),
            next_state_sequences=np.stack(next_state_sequences),
            next_sequence_lengths=np.asarray(next_sequence_lengths),
            dones=np.stack(dones),
        )

    def sample(self, batch_size):
        """Sample a random batch of temporal transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Sequence tensors and scalars ready for the temporal IQL update.
        """
        indices = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.from_numpy(self.state_sequences[indices]),
            torch.from_numpy(self.sequence_lengths[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_state_sequences[indices]),
            torch.from_numpy(self.next_sequence_lengths[indices]),
            torch.from_numpy(self.dones[indices]),
        )

    def __len__(self):
        """Return the number of stored temporal transitions."""
        return self.state_sequences.shape[0]
