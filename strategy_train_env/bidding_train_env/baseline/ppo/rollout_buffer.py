import numpy as np


class RolloutBuffer:
    """On-policy buffer with GAE. Use add() per timestep, record_reward() per
    timestep, finish_path() at the end of each episode, build_batch() to drain."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self._adv_chunks = []
        self._ret_chunks = []
        self._clear()

    def _clear(self):
        self.obs = []
        self.logp = []
        self.val = []
        self.log_act = []
        self.rew = []
        self.done = []

    def add(self, obs, logp, value, log_act):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.logp.append(float(logp))
        self.val.append(float(value))
        self.log_act.append(float(log_act))

    def record_reward(self, reward, done):
        self.rew.append(float(reward))
        self.done.append(float(done))

    def finish_path(self, last_value: float = 0.0):
        """Compute GAE advantages and returns over the in-progress episode."""
        T = len(self.rew)
        if T == 0:
            return
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        # Pull the corresponding tail of self.val (one value per timestep added).
        values = np.array(self.val[-T:] + [last_value], dtype=np.float32)
        rews = np.array(self.rew, dtype=np.float32)
        dones = np.array(self.done, dtype=np.float32)
        for t in reversed(range(T)):
            delta = rews[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        self._adv_chunks.append(adv)
        self._ret_chunks.append(adv + values[:-1])
        # Clear the per-episode reward/done lists; obs/logp/val/log_act stay
        # because build_batch consumes them across episodes.
        self.rew = []
        self.done = []

    def build_batch(self):
        obs = np.stack(self.obs).astype(np.float32)
        logp = np.array(self.logp, dtype=np.float32)
        log_act = np.array(self.log_act, dtype=np.float32)
        adv = np.concatenate(self._adv_chunks).astype(np.float32)
        ret = np.concatenate(self._ret_chunks).astype(np.float32)
        self._clear()
        self._adv_chunks = []
        self._ret_chunks = []
        return {"obs": obs, "logp": logp, "log_act": log_act, "adv": adv, "ret": ret}
