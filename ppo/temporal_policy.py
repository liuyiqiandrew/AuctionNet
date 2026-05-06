"""Residual-GRU feature extractors for online PPO."""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TemporalGRUFeaturesExtractor(BaseFeaturesExtractor):
    """Encode stacked PPO observations with an IQL phase-2-style residual GRU."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        obs_dim: int,
        seq_len: int,
        hidden_dim: int = 64,
        use_residual_latest_state: bool = True,
    ):
        self.obs_dim = int(obs_dim)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.use_residual_latest_state = bool(use_residual_latest_state)
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if self.seq_len <= 1:
            raise ValueError(f"TemporalGRUFeaturesExtractor requires seq_len > 1, got {seq_len}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        expected_dim = self.obs_dim * self.seq_len
        if observation_space.shape != (expected_dim,):
            raise ValueError(
                "Temporal observation shape mismatch: "
                f"expected {(expected_dim,)}, got {observation_space.shape}"
            )

        features_dim = self.hidden_dim + (self.obs_dim if self.use_residual_latest_state else 0)
        super().__init__(observation_space, features_dim=features_dim)

        self.encoder = nn.GRU(
            input_size=self.obs_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.feature_norm = nn.LayerNorm(features_dim) if self.use_residual_latest_state else nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sequences = observations.reshape(observations.shape[0], self.seq_len, self.obs_dim)
        _, hidden = self.encoder(sequences)
        features = hidden[-1]
        if self.use_residual_latest_state:
            latest_token = sequences[:, -1, :]
            features = torch.cat([features, latest_token], dim=-1)
        return self.feature_norm(features)
