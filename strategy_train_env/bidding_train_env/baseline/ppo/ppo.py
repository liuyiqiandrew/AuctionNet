import logging
import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    # Declared as torch.jit constants so they survive torch.jit.script.
    __constants__ = ["LOG_ALPHA_MIN", "LOG_ALPHA_MAX", "LOG_STD_MIN", "LOG_STD_MAX"]
    # Clamp range for log(alpha) to keep alpha = exp(log_alpha) numerically
    # safe in float32 (exp(15) ~ 3.3e6 — well above any plausible bid scale).
    LOG_ALPHA_MIN: float = -10.0
    LOG_ALPHA_MAX: float = 15.0
    # Bounds on the policy log-std so std stays in a numerically safe band.
    # exp(-5)~6.7e-3, exp(2)~7.4: enough range for both narrow and broad policies.
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 2.0

    def __init__(self, dim_obs: int = 16, hidden: int = 64, log_std_init: float = -0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim_obs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.ones(1) * log_std_init)

    def _mu_std(self, h):
        mu = self.mu_head(h).squeeze(-1)
        mu = torch.clamp(mu, self.LOG_ALPHA_MIN, self.LOG_ALPHA_MAX)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mu)
        return mu, std

    def forward(self, obs):
        # Inference path used by JIT-saved model: returns alpha (>= 0).
        h = self.trunk(obs)
        log_alpha = self.mu_head(h).squeeze(-1)
        log_alpha = torch.clamp(log_alpha, self.LOG_ALPHA_MIN, self.LOG_ALPHA_MAX)
        return torch.exp(log_alpha)

    def act(self, obs):
        # Training rollout: stochastic, returns (alpha, log_prob, value, log_alpha).
        h = self.trunk(obs)
        mu, std = self._mu_std(h)
        dist = Normal(mu, std)
        log_alpha = dist.sample()
        log_alpha = torch.clamp(log_alpha, self.LOG_ALPHA_MIN, self.LOG_ALPHA_MAX)
        log_prob = dist.log_prob(log_alpha)
        value = self.v_head(h).squeeze(-1)
        return torch.exp(log_alpha), log_prob, value, log_alpha

    def evaluate(self, obs, log_alpha):
        h = self.trunk(obs)
        mu, std = self._mu_std(h)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(log_alpha)
        entropy = dist.entropy()
        value = self.v_head(h).squeeze(-1)
        return log_prob, entropy, value


class PPO(nn.Module):
    """PPO trainer. self.forward(state) -> alpha is the JIT inference contract,
    matching IQL.forward in baseline/iql/iql.py:186-190."""

    def __init__(self, dim_obs: int = 16, lr: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, clip_eps: float = 0.2, vf_coef: float = 0.25,
                 ent_coef: float = 0.01, max_grad_norm: float = 0.5):
        super().__init__()
        self.ac = ActorCritic(dim_obs=dim_obs)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.ac.to(self.device)

    def forward(self, state):
        # JIT-scripted inference entry: state is a (16,) or (B, 16) float tensor.
        return self.ac(state)

    @torch.no_grad()
    def act(self, state_np):
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        squeeze = False
        if s.dim() == 1:
            s = s.unsqueeze(0)
            squeeze = True
        alpha, logp, value, log_alpha = self.ac.act(s)
        if squeeze:
            return (float(alpha.squeeze(0).cpu().numpy()),
                    float(logp.squeeze(0).cpu().numpy()),
                    float(value.squeeze(0).cpu().numpy()),
                    float(log_alpha.squeeze(0).cpu().numpy()))
        return (alpha.cpu().numpy(), logp.cpu().numpy(),
                value.cpu().numpy(), log_alpha.cpu().numpy())

    def update(self, batch, n_epochs: int = 4, minibatch_size: int = 256):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        log_act = torch.as_tensor(batch["log_act"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["ret"], dtype=torch.float32, device=self.device)
        old_val = torch.as_tensor(batch["val"], dtype=torch.float32, device=self.device)
        advs = torch.as_tensor(batch["adv"], dtype=torch.float32, device=self.device)

        # Finite-safe advantage normalization. A degenerate batch (all-equal advs,
        # or any inf slipping through) used to silently NaN every subsequent step.
        advs = torch.nan_to_num(advs, nan=0.0, posinf=0.0, neginf=0.0)
        returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        old_val = torch.nan_to_num(old_val, nan=0.0, posinf=0.0, neginf=0.0)
        if advs.numel() > 1 and advs.std().item() > 1e-6:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        else:
            advs = advs - advs.mean()

        N = obs.shape[0]
        idx = np.arange(N)
        pi_losses, v_losses, ent_vals = [], [], []
        for _ in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, minibatch_size):
                mb = idx[start:start + minibatch_size]
                logp, entropy, value = self.ac.evaluate(obs[mb], log_act[mb])
                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * advs[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs[mb]
                pi_loss = -torch.min(surr1, surr2).mean()
                # Clipped value loss (PPO2-style): caps how much one minibatch
                # can pull the critic, which protects the shared trunk.
                v_pred_clipped = old_val[mb] + (value - old_val[mb]).clamp(
                    -self.clip_eps, self.clip_eps
                )
                v_loss = 0.5 * torch.max(
                    (value - returns[mb]) ** 2,
                    (v_pred_clipped - returns[mb]) ** 2,
                ).mean()
                ent = entropy.mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                if not torch.isfinite(loss):
                    logger.warning("Non-finite PPO loss; skipping minibatch")
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    logger.warning("Non-finite grad norm; skipping minibatch")
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.opt.step()
                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                ent_vals.append(ent.item())
        if not pi_losses:
            return 0.0, 0.0, 0.0
        return float(np.mean(pi_losses)), float(np.mean(v_losses)), float(np.mean(ent_vals))

    def bc_pretrain(self, states, actions, epochs: int = 5, batch_size: int = 256):
        """Warm-start the actor by regressing log(alpha) against offline (s, a) pairs."""
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.log(np.clip(actions, 1e-6, None)),
                            dtype=torch.float32, device=self.device)
        N = s.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(N)
            running = 0.0
            for start in range(0, N, batch_size):
                mb = idx[start:start + batch_size]
                h = self.ac.trunk(s[mb])
                mu = self.ac.mu_head(h).squeeze(-1)
                loss = F.mse_loss(mu, a[mb])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                running += loss.item() * len(mb)
            print(f"[BC warm-start] epoch {ep} loss={running / N:.4f}")

    def save_jit(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        m = deepcopy(self).cpu()
        m.use_cuda = False
        m.device = torch.device("cpu")
        m.ac.to("cpu")
        jit_model = torch.jit.script(m)
        torch.jit.save(jit_model, os.path.join(save_path, "ppo_model.pth"))
