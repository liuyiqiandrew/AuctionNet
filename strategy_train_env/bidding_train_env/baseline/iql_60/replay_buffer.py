import numpy as np
import torch


class ArrayReplayBuffer:
    """Contiguous replay buffer optimized for CPU training throughput."""

    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = np.ascontiguousarray(states, dtype=np.float32)
        self.actions = np.ascontiguousarray(actions, dtype=np.float32)
        self.rewards = np.ascontiguousarray(rewards, dtype=np.float32)
        self.next_states = np.ascontiguousarray(next_states, dtype=np.float32)
        self.dones = np.ascontiguousarray(dones, dtype=np.float32)

    @classmethod
    def from_dataframe(cls, training_data, is_normalize=True):
        state_col = "normalize_state" if is_normalize else "state"
        next_state_col = "normalize_nextstate" if is_normalize else "next_state"
        reward_col = "normalize_reward" if is_normalize else "reward"

        states = np.stack(training_data[state_col].values).astype(np.float32)
        actions = training_data["action"].to_numpy(dtype=np.float32).reshape(-1, 1)
        rewards = training_data[reward_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        next_states = np.stack(training_data[next_state_col].values).astype(np.float32)
        dones = training_data["done"].to_numpy(dtype=np.float32).reshape(-1, 1)
        return cls(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self.states.shape[0]
