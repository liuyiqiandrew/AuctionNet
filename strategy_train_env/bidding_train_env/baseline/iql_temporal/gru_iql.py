import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam


def _prepare_lengths(lengths, max_seq_len, device):
    """Convert arbitrary sequence length input into a bounded tensor.

    Parameters
    ----------
    lengths : array-like or torch.Tensor
        Sequence lengths before padding.
    max_seq_len : int
        Maximum valid sequence length in the current batch.
    device : torch.device
        Device on which the returned tensor should live.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(batch_size,)`` with values clipped to the range
        ``[1, max_seq_len]``.
    """
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    lengths = lengths.to(device=device, dtype=torch.long).view(-1)
    return torch.clamp(lengths, min=1, max=max_seq_len)


def _encode_sequence(gru, sequences, lengths):
    """Encode a padded batch of sequences with a GRU.

    Parameters
    ----------
    gru : torch.nn.GRU
        GRU module used to summarize the sequence.
    sequences : torch.Tensor
        Tensor of shape ``(batch_size, seq_len, obs_dim)`` or
        ``(seq_len, obs_dim)``.
    lengths : array-like or torch.Tensor
        Unpadded lengths for each sequence in the batch.

    Returns
    -------
    torch.Tensor
        Final hidden state for each sequence with shape
        ``(batch_size, hidden_dim)``.
    """
    if sequences.dim() == 2:
        sequences = sequences.unsqueeze(0)
    lengths = _prepare_lengths(lengths, sequences.size(1), sequences.device)
    # Packing tells the GRU to ignore the right-padding used in the replay buffer.
    packed = pack_padded_sequence(
        sequences,
        lengths.detach().cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    _, hidden = gru(packed)
    return hidden[-1]


def _latest_valid_token(sequences, lengths):
    """Return the last unpadded token from each sequence."""
    if sequences.dim() == 2:
        sequences = sequences.unsqueeze(0)
    lengths = _prepare_lengths(lengths, sequences.size(1), sequences.device)
    batch_index = torch.arange(sequences.size(0), device=sequences.device)
    token_index = lengths - 1
    return sequences[batch_index, token_index]


class TemporalFeatureMixin:
    """Shared GRU feature extraction for separate temporal IQL heads."""

    def _init_temporal_features(self, dim_observation, encoder_hidden_dim, use_residual_latest_state):
        self.use_residual_latest_state = use_residual_latest_state
        self.feature_dim = encoder_hidden_dim + (dim_observation if use_residual_latest_state else 0)
        self.feature_norm = nn.LayerNorm(self.feature_dim) if use_residual_latest_state else nn.Identity()

    def _temporal_features(self, state_sequences, sequence_lengths):
        sequence_embedding = _encode_sequence(self.encoder, state_sequences, sequence_lengths)
        if self.use_residual_latest_state:
            latest_token = _latest_valid_token(state_sequences, sequence_lengths)
            sequence_embedding = torch.cat([sequence_embedding, latest_token], dim=-1)
        return self.feature_norm(sequence_embedding)


class SequenceQ(nn.Module):
    """Sequence-conditioned Q network for temporal IQL.

    Parameters
    ----------
    dim_observation : int
        Per-step observation dimension.
    dim_action : int
        Action dimension.
    encoder_hidden_dim : int
        Hidden size used by the GRU encoder.
    """

    def __init__(self, dim_observation, dim_action, encoder_hidden_dim, use_residual_latest_state=False):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=dim_observation,
            hidden_size=encoder_hidden_dim,
            batch_first=True,
        )
        TemporalFeatureMixin._init_temporal_features(
            self,
            dim_observation,
            encoder_hidden_dim,
            use_residual_latest_state,
        )
        self.action_fc = nn.Linear(dim_action, 64)
        self.fc1 = nn.Linear(self.feature_dim + 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state_sequences, sequence_lengths, actions):
        """Estimate Q-values for sequence-conditioned state-action pairs.

        Parameters
        ----------
        state_sequences : torch.Tensor
            Padded state history with shape ``(batch_size, seq_len, obs_dim)``.
        sequence_lengths : torch.Tensor
            Valid lengths for each sequence.
        actions : torch.Tensor
            Action tensor with shape ``(batch_size, action_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted Q-values with shape ``(batch_size, 1)``.
        """
        sequence_embedding = TemporalFeatureMixin._temporal_features(self, state_sequences, sequence_lengths)
        action_embedding = self.action_fc(actions)
        embedding = torch.cat([sequence_embedding, action_embedding], dim=-1)
        q = self.fc3(F.relu(self.fc2(F.relu(self.fc1(embedding)))))
        return q


class SequenceV(nn.Module):
    """Sequence-conditioned value network for temporal IQL.

    Parameters
    ----------
    dim_observation : int
        Per-step observation dimension.
    encoder_hidden_dim : int
        Hidden size used by the GRU encoder.
    """

    def __init__(self, dim_observation, encoder_hidden_dim, use_residual_latest_state=False):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=dim_observation,
            hidden_size=encoder_hidden_dim,
            batch_first=True,
        )
        TemporalFeatureMixin._init_temporal_features(
            self,
            dim_observation,
            encoder_hidden_dim,
            use_residual_latest_state,
        )
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, state_sequences, sequence_lengths):
        """Estimate state values from padded state histories.

        Parameters
        ----------
        state_sequences : torch.Tensor
            Padded state history with shape ``(batch_size, seq_len, obs_dim)``.
        sequence_lengths : torch.Tensor
            Valid lengths for each sequence.

        Returns
        -------
        torch.Tensor
            Predicted state values with shape ``(batch_size, 1)``.
        """
        sequence_embedding = TemporalFeatureMixin._temporal_features(self, state_sequences, sequence_lengths)
        result = F.relu(self.fc1(sequence_embedding))
        result = F.relu(self.fc2(result))
        result = F.relu(self.fc3(result))
        return self.fc4(result)


class SequenceActor(nn.Module):
    """Sequence-conditioned Gaussian actor for temporal IQL.

    Parameters
    ----------
    dim_observation : int
        Per-step observation dimension.
    dim_action : int
        Action dimension.
    encoder_hidden_dim : int
        Hidden size used by the GRU encoder.
    log_std_min : float, optional
        Lower clamp for the Gaussian log standard deviation.
    log_std_max : float, optional
        Upper clamp for the Gaussian log standard deviation.
    """

    def __init__(
        self,
        dim_observation,
        dim_action,
        encoder_hidden_dim,
        log_std_min=-10,
        log_std_max=2,
        use_residual_latest_state=False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.encoder = nn.GRU(
            input_size=dim_observation,
            hidden_size=encoder_hidden_dim,
            batch_first=True,
        )
        TemporalFeatureMixin._init_temporal_features(
            self,
            dim_observation,
            encoder_hidden_dim,
            use_residual_latest_state,
        )
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, dim_action)
        self.fc_std = nn.Linear(64, dim_action)

    def forward(self, state_sequences, sequence_lengths):
        """Compute Gaussian action parameters from state sequences.

        Parameters
        ----------
        state_sequences : torch.Tensor
            Padded state history with shape ``(batch_size, seq_len, obs_dim)``.
        sequence_lengths : torch.Tensor
            Valid lengths for each sequence.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean and log standard deviation tensors, each of shape
            ``(batch_size, action_dim)``.
        """
        sequence_embedding = TemporalFeatureMixin._temporal_features(self, state_sequences, sequence_lengths)
        x = F.relu(self.fc1(sequence_embedding))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, state_sequences, sequence_lengths):
        """Sample actions and return the corresponding Gaussian distribution."""
        mu, log_std = self.forward(state_sequences, sequence_lengths)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action, dist

    def get_action(self, state_sequences, sequence_lengths):
        """Sample an action from the actor and move it to CPU."""
        mu, log_std = self.forward(state_sequences, sequence_lengths)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action.detach().cpu()

    def get_det_action(self, state_sequences, sequence_lengths):
        """Return the deterministic mean action and move it to CPU."""
        mu, _ = self.forward(state_sequences, sequence_lengths)
        return mu.detach().cpu()


class SharedTemporalEncoder(nn.Module):
    """Shared temporal encoder used by the phase-5 model variant."""

    def __init__(self, dim_observation, encoder_hidden_dim, use_residual_latest_state=True):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=dim_observation,
            hidden_size=encoder_hidden_dim,
            batch_first=True,
        )
        self.use_residual_latest_state = use_residual_latest_state
        self.output_dim = encoder_hidden_dim + (dim_observation if use_residual_latest_state else 0)
        self.feature_norm = nn.LayerNorm(self.output_dim) if use_residual_latest_state else nn.Identity()

    def forward(self, state_sequences, sequence_lengths):
        sequence_embedding = _encode_sequence(self.encoder, state_sequences, sequence_lengths)
        if self.use_residual_latest_state:
            latest_token = _latest_valid_token(state_sequences, sequence_lengths)
            sequence_embedding = torch.cat([sequence_embedding, latest_token], dim=-1)
        return self.feature_norm(sequence_embedding)


class SharedValueHead(nn.Module):
    """Value head over a shared temporal feature vector."""

    def __init__(self, feature_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, features):
        result = F.relu(self.fc1(features))
        result = F.relu(self.fc2(result))
        result = F.relu(self.fc3(result))
        return self.fc4(result)


class SharedQHead(nn.Module):
    """Q head over a shared temporal feature vector and action."""

    def __init__(self, feature_dim, dim_action):
        super().__init__()
        self.action_fc = nn.Linear(dim_action, 64)
        self.fc1 = nn.Linear(feature_dim + 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, features, actions):
        action_embedding = self.action_fc(actions)
        embedding = torch.cat([features, action_embedding], dim=-1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(embedding)))))


class SharedActorHead(nn.Module):
    """Gaussian actor head over a shared temporal feature vector."""

    def __init__(self, feature_dim, dim_action, log_std_min=-10, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, dim_action)
        self.fc_std = nn.Linear(64, dim_action)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = torch.clamp(self.fc_std(x), self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, features):
        mu, log_std = self.forward(features)
        dist = Normal(mu, log_std.exp())
        action = dist.rsample()
        return action, dist

    def get_action(self, features):
        action, _ = self.evaluate(features)
        return action.detach().cpu()

    def get_det_action(self, features):
        mu, _ = self.forward(features)
        return mu.detach().cpu()


class GRUIQL(nn.Module):
    """Implicit Q-Learning with GRU sequence encoders.

    This variant keeps the original IQL objectives but replaces the flat
    feed-forward state encoder with GRU summaries over the last ``K`` states.

    Parameters
    ----------
    dim_obs : int, optional
        Per-step observation dimension.
    seq_len : int, optional
        Maximum history length used by the temporal encoder.
    dim_actions : int, optional
        Action dimension.
    gamma : float, optional
        Discount factor.
    tau : float, optional
        Soft target-network update coefficient.
    V_lr : float, optional
        Learning rate for the value network.
    critic_lr : float, optional
        Learning rate for both Q networks.
    actor_lr : float, optional
        Learning rate for the actor.
    network_random_seed : int, optional
        Random seed for network initialization.
    expectile : float, optional
        Expectile used by the value loss.
    temperature : float, optional
        Weighting temperature used in the actor loss.
    encoder_hidden_dim : int, optional
        Hidden size of each GRU encoder.
    """

    def __init__(
        self,
        dim_obs=16,
        seq_len=8,
        dim_actions=1,
        gamma=0.99,
        tau=0.01,
        V_lr=1e-4,
        critic_lr=1e-4,
        actor_lr=1e-4,
        network_random_seed=1,
        expectile=0.7,
        temperature=3.0,
        encoder_hidden_dim=64,
        use_residual_latest_state=False,
        shared_encoder=False,
    ):
        super().__init__()
        self.num_of_states = dim_obs
        self.seq_len = seq_len
        self.num_of_actions = dim_actions
        self.V_lr = V_lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.network_random_seed = network_random_seed
        self.expectile = expectile
        self.temperature = temperature
        self.encoder_hidden_dim = encoder_hidden_dim
        self.use_residual_latest_state = use_residual_latest_state
        self.shared_encoder = shared_encoder
        self.GAMMA = gamma
        self.tau = tau

        torch.random.manual_seed(self.network_random_seed)
        if self.shared_encoder:
            self.temporal_encoder = SharedTemporalEncoder(
                self.num_of_states,
                self.encoder_hidden_dim,
                use_residual_latest_state=self.use_residual_latest_state,
            )
            self.temporal_encoder_target = deepcopy(self.temporal_encoder)
            shared_feature_dim = self.temporal_encoder.output_dim
            self.value_net = SharedValueHead(shared_feature_dim)
            self.critic1 = SharedQHead(shared_feature_dim, self.num_of_actions)
            self.critic2 = SharedQHead(shared_feature_dim, self.num_of_actions)
            self.critic1_target = deepcopy(self.critic1)
            self.critic2_target = deepcopy(self.critic2)
            self.actors = SharedActorHead(shared_feature_dim, self.num_of_actions)
            self.shared_optimizer = Adam(
                list(self.temporal_encoder.parameters())
                + list(self.value_net.parameters())
                + list(self.critic1.parameters())
                + list(self.critic2.parameters())
                + list(self.actors.parameters()),
                lr=self.actor_lr,
            )
        else:
            self.value_net = SequenceV(
                self.num_of_states,
                self.encoder_hidden_dim,
                use_residual_latest_state=self.use_residual_latest_state,
            )
            self.critic1 = SequenceQ(
                self.num_of_states,
                self.num_of_actions,
                self.encoder_hidden_dim,
                use_residual_latest_state=self.use_residual_latest_state,
            )
            self.critic2 = SequenceQ(
                self.num_of_states,
                self.num_of_actions,
                self.encoder_hidden_dim,
                use_residual_latest_state=self.use_residual_latest_state,
            )
            self.critic1_target = SequenceQ(
                self.num_of_states,
                self.num_of_actions,
                self.encoder_hidden_dim,
                use_residual_latest_state=self.use_residual_latest_state,
            )
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target = SequenceQ(
                self.num_of_states,
                self.num_of_actions,
                self.encoder_hidden_dim,
                use_residual_latest_state=self.use_residual_latest_state,
            )
            self.critic2_target.load_state_dict(self.critic2.state_dict())
            self.actors = SequenceActor(
                self.num_of_states,
                self.num_of_actions,
                self.encoder_hidden_dim,
                use_residual_latest_state=self.use_residual_latest_state,
            )
            self.value_optimizer = Adam(self.value_net.parameters(), lr=self.V_lr)
            self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.critic_lr)
            self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
            self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)
        self.deterministic_action = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.to(self.device)

    def optimizer_state_dicts(self):
        if self.shared_encoder:
            return {"shared_optimizer": self.shared_optimizer.state_dict()}
        return {
            "value_optimizer": self.value_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def optimizer_map(self):
        if self.shared_encoder:
            return {"shared_optimizer": self.shared_optimizer}
        return {
            "value_optimizer": self.value_optimizer,
            "critic1_optimizer": self.critic1_optimizer,
            "critic2_optimizer": self.critic2_optimizer,
            "actor_optimizer": self.actor_optimizer,
        }

    def step(
        self,
        state_sequences,
        sequence_lengths,
        actions,
        rewards,
        next_state_sequences,
        next_sequence_lengths,
        dones,
    ):
        """Run one temporal IQL optimization step.

        Parameters
        ----------
        state_sequences : torch.Tensor
            Current padded state sequences with shape
            ``(batch_size, seq_len, obs_dim)``.
        sequence_lengths : torch.Tensor
            Valid lengths for ``state_sequences``.
        actions : torch.Tensor
            Batch of actions with shape ``(batch_size, action_dim)``.
        rewards : torch.Tensor
            Batch of scalar rewards with shape ``(batch_size, 1)``.
        next_state_sequences : torch.Tensor
            Next-state padded sequences with shape
            ``(batch_size, seq_len, obs_dim)``.
        next_sequence_lengths : torch.Tensor
            Valid lengths for ``next_state_sequences``.
        dones : torch.Tensor
            Terminal indicators with shape ``(batch_size, 1)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Critic, value, and actor losses returned as NumPy scalars for
            logging consistency with the existing IQL runners.
        """
        state_sequences = state_sequences.to(self.device)
        sequence_lengths = sequence_lengths.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_state_sequences = next_state_sequences.to(self.device)
        next_sequence_lengths = next_sequence_lengths.to(self.device)
        dones = dones.to(self.device)

        if self.shared_encoder:
            return self._shared_step(
                state_sequences,
                sequence_lengths,
                actions,
                rewards,
                next_state_sequences,
                next_sequence_lengths,
                dones,
            )

        # IQL updates V, then policy, then Q-functions using the same batch.
        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(state_sequences, sequence_lengths, actions)
        value_loss.backward()
        self.value_optimizer.step()

        actor_loss = self.calc_policy_loss(state_sequences, sequence_lengths, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic1_loss, critic2_loss = self.calc_q_loss(
            state_sequences,
            sequence_lengths,
            actions,
            rewards,
            dones,
            next_state_sequences,
            next_sequence_lengths,
        )
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)

        return (
            critic1_loss.detach().cpu().numpy(),
            value_loss.detach().cpu().numpy(),
            actor_loss.detach().cpu().numpy(),
        )

    def _shared_step(
        self,
        state_sequences,
        sequence_lengths,
        actions,
        rewards,
        next_state_sequences,
        next_sequence_lengths,
        dones,
    ):
        """Run one combined optimizer update for the shared-encoder variant."""
        value_loss = self.calc_value_loss(state_sequences, sequence_lengths, actions)
        actor_loss = self.calc_policy_loss(state_sequences, sequence_lengths, actions)
        critic1_loss, critic2_loss = self.calc_q_loss(
            state_sequences,
            sequence_lengths,
            actions,
            rewards,
            dones,
            next_state_sequences,
            next_sequence_lengths,
        )
        total_loss = value_loss + actor_loss + critic1_loss + critic2_loss
        self.shared_optimizer.zero_grad()
        total_loss.backward()
        self.shared_optimizer.step()

        self.update_target(self.temporal_encoder, self.temporal_encoder_target)
        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)

        return (
            critic1_loss.detach().cpu().numpy(),
            value_loss.detach().cpu().numpy(),
            actor_loss.detach().cpu().numpy(),
        )

    def take_actions(self, state_sequences, sequence_lengths):
        """Produce actions from temporal state histories.

        Parameters
        ----------
        state_sequences : array-like or torch.Tensor
            Padded state history batch.
        sequence_lengths : array-like or torch.Tensor
            Valid lengths for each sequence.

        Returns
        -------
        numpy.ndarray
            Non-negative action predictions with shape
            ``(batch_size, action_dim)``.
        """
        if not torch.is_tensor(state_sequences):
            state_sequences = torch.tensor(state_sequences, dtype=torch.float32)
        if not torch.is_tensor(sequence_lengths):
            sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)
        state_sequences = state_sequences.to(self.device)
        sequence_lengths = sequence_lengths.to(self.device)
        if self.shared_encoder:
            features = self.temporal_encoder(state_sequences, sequence_lengths)
            if self.deterministic_action:
                actions = self.actors.get_det_action(features)
            else:
                actions = self.actors.get_action(features)
        elif self.deterministic_action:
            actions = self.actors.get_det_action(state_sequences, sequence_lengths)
        else:
            actions = self.actors.get_action(state_sequences, sequence_lengths)
        actions = torch.clamp(actions, min=0)
        return actions.cpu().numpy()

    def forward(self, state_sequences, sequence_lengths):
        """Return deterministic actions for scripted or eval-time inference."""
        if self.shared_encoder:
            features = self.temporal_encoder(state_sequences, sequence_lengths)
            actions = self.actors.get_det_action(features)
        else:
            actions = self.actors.get_det_action(state_sequences, sequence_lengths)
        actions = torch.clamp(actions, min=0)
        return actions

    def calc_policy_loss(self, state_sequences, sequence_lengths, actions):
        """Compute the temporal IQL actor objective.

        Parameters
        ----------
        state_sequences : torch.Tensor
            Current padded state sequences.
        sequence_lengths : torch.Tensor
            Valid lengths for each current sequence.
        actions : torch.Tensor
            Dataset actions.

        Returns
        -------
        torch.Tensor
            Scalar actor loss.
        """
        if self.shared_encoder:
            with torch.no_grad():
                features_for_advantage = self.temporal_encoder(state_sequences, sequence_lengths)
                target_features = self.temporal_encoder_target(state_sequences, sequence_lengths)
                v = self.value_net(features_for_advantage)
                q1 = self.critic1_target(target_features, actions)
                q2 = self.critic2_target(target_features, actions)
                min_q = torch.min(q1, q2)

            exp_a = torch.exp(min_q - v) * self.temperature
            exp_a = torch.min(exp_a, torch.tensor([100.0], device=self.device))
            actor_features = self.temporal_encoder(state_sequences, sequence_lengths)
            _, dist = self.actors.evaluate(actor_features)
            log_probs = dist.log_prob(actions)
            return -(exp_a * log_probs).mean()

        with torch.no_grad():
            v = self.value_net(state_sequences, sequence_lengths)
            q1 = self.critic1_target(state_sequences, sequence_lengths, actions)
            q2 = self.critic2_target(state_sequences, sequence_lengths, actions)
            min_q = torch.min(q1, q2)

        exp_a = torch.exp(min_q - v) * self.temperature
        exp_a = torch.min(exp_a, torch.tensor([100.0], device=self.device))

        _, dist = self.actors.evaluate(state_sequences, sequence_lengths)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss

    def calc_value_loss(self, state_sequences, sequence_lengths, actions):
        """Compute the expectile value regression loss.

        Parameters
        ----------
        state_sequences : torch.Tensor
            Current padded state sequences.
        sequence_lengths : torch.Tensor
            Valid lengths for each current sequence.
        actions : torch.Tensor
            Dataset actions used to query the target Q networks.

        Returns
        -------
        torch.Tensor
            Scalar value loss.
        """
        if self.shared_encoder:
            with torch.no_grad():
                target_features = self.temporal_encoder_target(state_sequences, sequence_lengths)
                q1 = self.critic1_target(target_features, actions)
                q2 = self.critic2_target(target_features, actions)
                min_q = torch.min(q1, q2)

            features = self.temporal_encoder(state_sequences, sequence_lengths)
            value = self.value_net(features)
            return self.l2_loss(min_q - value, self.expectile).mean()

        with torch.no_grad():
            q1 = self.critic1_target(state_sequences, sequence_lengths, actions)
            q2 = self.critic2_target(state_sequences, sequence_lengths, actions)
            min_q = torch.min(q1, q2)

        value = self.value_net(state_sequences, sequence_lengths)
        value_loss = self.l2_loss(min_q - value, self.expectile).mean()
        return value_loss

    def calc_q_loss(
        self,
        state_sequences,
        sequence_lengths,
        actions,
        rewards,
        dones,
        next_state_sequences,
        next_sequence_lengths,
    ):
        """Compute the temporal Bellman regression loss for both Q networks.

        Parameters
        ----------
        state_sequences : torch.Tensor
            Current padded state sequences.
        sequence_lengths : torch.Tensor
            Valid lengths for each current sequence.
        actions : torch.Tensor
            Dataset actions.
        rewards : torch.Tensor
            Immediate rewards.
        dones : torch.Tensor
            Terminal indicators.
        next_state_sequences : torch.Tensor
            Next-state padded sequences.
        next_sequence_lengths : torch.Tensor
            Valid lengths for each next-state sequence.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Critic losses for ``critic1`` and ``critic2``.
        """
        if self.shared_encoder:
            with torch.no_grad():
                next_features = self.temporal_encoder(next_state_sequences, next_sequence_lengths)
                next_v = self.value_net(next_features)
                q_target = rewards + (self.GAMMA * (1 - dones) * next_v)

            features = self.temporal_encoder(state_sequences, sequence_lengths)
            q1 = self.critic1(features, actions)
            q2 = self.critic2(features, actions)
            return ((q1 - q_target) ** 2).mean(), ((q2 - q_target) ** 2).mean()

        with torch.no_grad():
            next_v = self.value_net(next_state_sequences, next_sequence_lengths)
            q_target = rewards + (self.GAMMA * (1 - dones) * next_v)

        q1 = self.critic1(state_sequences, sequence_lengths, actions)
        q2 = self.critic2(state_sequences, sequence_lengths, actions)
        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()
        return critic1_loss, critic2_loss

    def update_target(self, local_model, target_model):
        """Soft-update a target network toward its online counterpart."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * local_param.data)

    def save_checkpoint(self, save_path):
        """Save model weights and constructor config for later reload.

        Parameters
        ----------
        save_path : str
            Directory where the checkpoint file should be written.
        """
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "state_dict": deepcopy(self).cpu().state_dict(),
                "config": {
                    "dim_obs": self.num_of_states,
                    "seq_len": self.seq_len,
                    "dim_actions": self.num_of_actions,
                    "gamma": self.GAMMA,
                    "tau": self.tau,
                    "V_lr": self.V_lr,
                    "critic_lr": self.critic_lr,
                    "actor_lr": self.actor_lr,
                    "network_random_seed": self.network_random_seed,
                    "expectile": self.expectile,
                    "temperature": self.temperature,
                    "encoder_hidden_dim": self.encoder_hidden_dim,
                    "use_residual_latest_state": self.use_residual_latest_state,
                    "shared_encoder": self.shared_encoder,
                },
            },
            os.path.join(save_path, "gru_iql_model.pt"),
        )

    @classmethod
    def load_checkpoint(cls, load_path, map_location="cpu"):
        """Load a temporal IQL checkpoint from disk.

        Parameters
        ----------
        load_path : str
            Directory containing ``gru_iql_model.pt``.
        map_location : str or torch.device, optional
            Device mapping passed to ``torch.load``.

        Returns
        -------
        GRUIQL
            Reconstructed model with weights restored.
        """
        checkpoint = torch.load(os.path.join(load_path, "gru_iql_model.pt"), map_location=map_location)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        target_device = map_location if isinstance(map_location, torch.device) else None
        if target_device is None and isinstance(map_location, str):
            try:
                target_device = torch.device(map_location)
            except (TypeError, RuntimeError):
                target_device = None

        if target_device is None:
            target_device = model.device
        if target_device.type == "cuda" and not torch.cuda.is_available():
            target_device = torch.device("cpu")

        model.use_cuda = target_device.type == "cuda"
        model.device = target_device
        model.to(target_device)
        return model

    @staticmethod
    def l2_loss(diff, expectile=0.8):
        """Compute the expectile-weighted squared error.

        Parameters
        ----------
        diff : torch.Tensor
            Difference term, typically ``target_q - value``.
        expectile : float, optional
            Expectile coefficient in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Elementwise weighted squared error.
        """
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)
