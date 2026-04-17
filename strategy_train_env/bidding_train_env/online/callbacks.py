"""Checkpoint + stdout logging callbacks."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class CustomCheckpointCallback(CheckpointCallback):
    """Saves model + VecNormalize per checkpoint, excluding obs buffers to cut file size."""

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path, exclude=["_last_obs", "_last_original_obs"])
            if self.verbose >= 2:
                print(f"[ckpt] saved model to {model_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                vec_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                vecnorm = self.model.get_vec_normalize_env()
                vecnorm.old_obs = None
                vecnorm.save(vec_path)
                if self.verbose >= 2:
                    print(f"[ckpt] saved vecnormalize to {vec_path}")
        return True


class StdoutEpisodeCallback(BaseCallback):
    """Prints mean rollout/episode stats at each rollout end."""

    def __init__(self, log_interval: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.rollouts = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts += 1
        if self.rollouts % self.log_interval != 0:
            return
        ep_info = self.model.ep_info_buffer
        if not ep_info:
            return
        rew = float(np.mean([e["r"] for e in ep_info]))
        length = float(np.mean([e["l"] for e in ep_info]))
        print(
            f"[rollout {self.rollouts}] steps={self.num_timesteps} "
            f"ep_rew_mean={rew:.4f} ep_len_mean={length:.1f} "
            f"n_ep_buf={len(ep_info)}"
        )
