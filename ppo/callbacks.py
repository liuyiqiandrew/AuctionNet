"""Checkpoint + stdout logging callbacks."""

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class CustomCheckpointCallback(CheckpointCallback):
    """Saves model + VecNormalize per checkpoint, excluding obs buffers to cut file size.

    Additionally reads rollout_log.jsonl (written by JsonRolloutCallback) and
    renders a two-panel training curve (ep_rew_mean + score) to
    rollout_curve.png alongside the checkpoint.
    """

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

            self._plot_training_curve()
        return True

    def _plot_training_curve(self) -> None:
        jsonl_path = Path(self.save_path) / "rollout_log.jsonl"
        if not jsonl_path.exists():
            return
        rows = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            if self.verbose:
                print("[ckpt] matplotlib unavailable; skipping plot")
            return

        steps = [r.get("timesteps") for r in rows]
        ep_rew = [r.get("ep_rew_mean") for r in rows]
        score = [r.get("score") for r in rows]

        fig, (ax_rew, ax_score) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        rew_xy = [(s, v) for s, v in zip(steps, ep_rew) if s is not None and v is not None]
        if rew_xy:
            xs, ys = zip(*rew_xy)
            ax_rew.plot(xs, ys, lw=1.2)
        ax_rew.set_ylabel("ep_rew_mean")
        ax_rew.grid(alpha=0.3)

        score_xy = [(s, v) for s, v in zip(steps, score) if s is not None and v is not None]
        if score_xy:
            xs, ys = zip(*score_xy)
            ax_score.plot(xs, ys, lw=1.2, color="tab:orange")
        ax_score.set_ylabel("score")
        ax_score.set_xlabel("env steps")
        ax_score.grid(alpha=0.3)

        fig.suptitle(f"training curve @ step {self.num_timesteps}")
        fig.tight_layout()
        plot_path = Path(self.save_path) / "rollout_curve.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        if self.verbose >= 2:
            print(f"[ckpt] saved training curve to {plot_path}")


class JsonRolloutCallback(BaseCallback):
    """Per-rollout mean over INFO_KEYWORDS across envs; appends JSONL + prints stdout.

    Mirrors oil's TensorboardCallback: collect dict-of-lists across every step of
    the current rollout, filter None (terminal-only keys show up rarely), emit
    means at rollout end. Also pulls ep_rew_mean/ep_len_mean from sb3's
    ep_info_buffer so one row carries both the training signal and the
    benchmark metric (score).
    """

    def __init__(
        self,
        info_keywords,
        log_path,
        log_interval: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.info_keywords = tuple(info_keywords)
        self.log_path = Path(log_path)
        self.log_interval = log_interval
        self.rollouts = 0
        self.rollout_info: dict = {}

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_rollout_start(self) -> None:
        self.rollout_info = {k: [] for k in self.info_keywords}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k in self.info_keywords:
                v = info.get(k)
                if v is not None:
                    self.rollout_info[k].append(float(v))
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts += 1
        row: dict = {
            "rollout": self.rollouts,
            "timesteps": int(self.num_timesteps),
        }
        for k in self.info_keywords:
            xs = self.rollout_info[k]
            row[k] = float(np.mean(xs)) if xs else None
            row[f"{k}_n"] = len(xs)

        ep_buf = self.model.ep_info_buffer
        if ep_buf:
            row["ep_rew_mean"] = float(np.mean([e["r"] for e in ep_buf]))
            row["ep_len_mean"] = float(np.mean([e["l"] for e in ep_buf]))
            row["ep_n"] = len(ep_buf)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if self.rollouts % self.log_interval == 0:
            def _fmt(v, prec=4):
                return "na" if v is None else f"{v:.{prec}f}"
            print(
                f"[rollout {self.rollouts}] steps={self.num_timesteps} "
                f"ep_rew_mean={_fmt(row.get('ep_rew_mean'))} "
                f"score={_fmt(row.get('score'))} "
                f"cost_over_budget={_fmt(row.get('cost_over_budget'), 3)} "
                f"target_cpa_over_cpa={_fmt(row.get('target_cpa_over_cpa'), 3)} "
                f"n_terminal={row.get('score_n', 0)}"
            )
