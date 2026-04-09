"""PPO training with replay-based environment interaction.

Data split: periods 7-26 for BC warmstart + online training,
period 27 reserved for evaluation. Training episodes replay logged
auction data from per-period CSVs — the agent bids against recorded
market prices (leastWinningCost), making rollouts fast.
"""

import os
import ast
import time
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from bidding_train_env.common.utils import normalize_state, save_normalize_dict
from bidding_train_env.baseline.ppo.ppo import PPO
from bidding_train_env.baseline.ppo.rollout_buffer import RolloutBuffer
from bidding_train_env.baseline.ppo.state_builder import STATE_DIM
from bidding_train_env.baseline.ppo.bidding_env import BiddingEnv
from bidding_train_env.baseline.ppo.metrics import (
    MetricsTracker, compute_score, plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TRAIN_PERIODS = tuple(range(7, 27))  # periods 7-26; period 27 held out
REWARD_SCALE = 1.0 / 20.0


@dataclass
class TrainConfig:
    num_iterations: int = 20000
    episodes_per_iter: int = 20
    ppo_epochs: int = 4
    minibatch_size: int = 256
    bc_epochs: int = 5
    bc_batch_size: int = 512
    reward_scale: float = REWARD_SCALE
    lr: float = 5e-5
    cpa_penalty_coef: float = 10.0
    checkpoint_interval: int = 50
    save_dir: str = "saved_model/PPOtest"
    seed: int = 1
    data_dir: str = "./data/traffic"


@dataclass
class EpisodeResult:
    reward: float
    cost: float
    conversions: float
    num_ticks: int
    budget: float
    cpa_constraint: float

    @property
    def cpa(self) -> float:
        return self.cost / self.conversions if self.conversions > 0 else float("inf")

    @property
    def score(self) -> float:
        return compute_score(self.conversions, self.cpa, self.cpa_constraint)

    @property
    def budget_utilization(self) -> float:
        return self.cost / self.budget if self.budget > 0 else 0.0


# ---------------------------------------------------------------------------
# Data loading (BC warmstart)
# ---------------------------------------------------------------------------

def _safe_literal(x):
    if pd.isna(x):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        try:
            return eval(x, {"__builtins__": {}}, {"np": np})
        except Exception:
            return x


def load_offline_for_warmstart(periods=TRAIN_PERIODS):
    """Load per-period RL data for BC warmstart, excluding held-out periods."""
    data_dir = "./data/traffic/training_data_rlData_folder"
    frames = []
    for p in periods:
        path = os.path.join(data_dir, f"period-{p}-rlData.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No RL data found for periods {periods} in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["state"] = df["state"].apply(_safe_literal)
    df["next_state"] = df["next_state"].apply(_safe_literal)
    normalize_dict = normalize_state(df, STATE_DIM, normalize_indices=[13, 14, 15])
    states = np.stack(df["normalize_state"].values).astype(np.float32)
    actions = df["action"].values.astype(np.float32)
    logger.info(f"Loaded {len(df)} transitions from {len(frames)} periods for BC warmstart")
    return states, actions, normalize_dict


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_one_episode(env: BiddingEnv, ppo: PPO, buffer: RolloutBuffer,
                    config: TrainConfig) -> EpisodeResult:
    """Run one episode in the replay environment, collecting PPO rollout data."""
    obs = env.reset()
    done = False
    while not done:
        alpha, logp, value, log_alpha = ppo.act(obs)
        buffer.add(obs, logp, value, log_alpha)
        obs, reward, done, info = env.step(alpha)
        buffer.record_reward(reward * config.reward_scale, done)

    # Terminal CPA shaping.
    total_cost = info["total_cost"]
    total_conv = info["total_conversions"]
    cpa_cap = info["cpa_constraint"]
    realized_cpa = total_cost / total_conv if total_conv > 0 else float("inf")
    if realized_cpa > cpa_cap and len(buffer.rew) > 0:
        overrun = (realized_cpa - cpa_cap) / max(cpa_cap, 1e-6)
        buffer.rew[-1] -= config.cpa_penalty_coef * overrun * config.reward_scale

    buffer.finish_path(last_value=0.0)
    return EpisodeResult(
        reward=info["total_value"],
        cost=total_cost,
        conversions=total_conv,
        num_ticks=info["num_ticks"],
        budget=info["budget"],
        cpa_constraint=cpa_cap,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_ppo(config: TrainConfig | None = None) -> MetricsTracker:
    if config is None:
        config = TrainConfig()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    # BC warmstart on periods 7-26
    logger.info("Loading offline data for BC warmstart (periods 7-26)...")
    states, actions, normalize_dict = load_offline_for_warmstart(TRAIN_PERIODS)
    save_normalize_dict(normalize_dict, config.save_dir)

    ppo = PPO(dim_obs=STATE_DIM, lr=config.lr)
    print(f"[device] PPO on: {ppo.device}  (CUDA available: {torch.cuda.is_available()})")
    logger.info("Behavioral-cloning warmstart...")
    ppo.bc_pretrain(states, actions, epochs=config.bc_epochs, batch_size=config.bc_batch_size)

    # Build replay environment from training periods.
    logger.info("Loading replay environment (periods 7-26)...")
    env = BiddingEnv(
        periods=TRAIN_PERIODS,
        data_dir=config.data_dir,
        normalize_dict=normalize_dict,
    )

    buffer = RolloutBuffer(gamma=ppo.gamma, lam=ppo.lam)
    tracker = MetricsTracker()

    for it in range(config.num_iterations):
        t0 = time.time()
        results: list[EpisodeResult] = []

        for k in range(config.episodes_per_iter):
            result = run_one_episode(env, ppo, buffer, config)
            results.append(result)

        t_rollout = time.time() - t0
        t1 = time.time()
        batch = buffer.build_batch()
        pi, v, ent = ppo.update(batch, n_epochs=config.ppo_epochs,
                                minibatch_size=config.minibatch_size)
        t_update = time.time() - t1
        elapsed = time.time() - t0
        # print(f"[timing] iter {it:03d}: rollout={t_rollout:.1f}s  update={t_update:.3f}s  "
        #       f"batch_size={batch['obs'].shape[0]}  total={elapsed:.1f}s")

        tracker.log_iteration(
            iteration=it,
            mean_episode_reward=np.mean([r.reward for r in results]),
            mean_episode_cost=np.mean([r.cost for r in results]),
            mean_episode_conversions=np.mean([r.conversions for r in results]),
            mean_episode_score=np.mean([r.score for r in results]),
            mean_budget_utilization=np.mean([r.budget_utilization for r in results]),
            policy_loss=pi, value_loss=v, entropy=ent,
            elapsed_seconds=elapsed,
        )
        logger.info(
            f"iter {it:03d} | reward={tracker.iterations[-1].mean_episode_reward:.1f} "
            f"score={tracker.iterations[-1].mean_episode_score:.1f} "
            f"butil={tracker.iterations[-1].mean_budget_utilization:.2f} "
            f"pi={pi:.4f} v={v:.4f} ent={ent:.3f} ({elapsed:.1f}s)"
        )

        # Periodic checkpoint
        if (it + 1) % config.checkpoint_interval == 0:
            ckpt_dir = os.path.join(config.save_dir, f"checkpoint_{it+1}")
            ppo.save_jit(ckpt_dir)
            logger.info(f"Checkpoint saved to {ckpt_dir}")

    # Final save
    ppo.save_jit(config.save_dir)
    metrics_path = os.path.join(config.save_dir, "metrics.json")
    tracker.save(metrics_path)
    plot_training_curves(tracker, config.save_dir)
    logger.info(f"Training complete. Model: {config.save_dir}/ppo_model.pth  "
                f"Metrics: {metrics_path}")
    return tracker


if __name__ == "__main__":
    run_ppo()
