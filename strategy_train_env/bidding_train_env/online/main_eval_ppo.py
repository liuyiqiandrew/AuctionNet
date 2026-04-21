"""Online PPO evaluation entry point.

Two modes (both supported simultaneously with --eval_mode both):
  - random: N random (advertiser, budget, target_cpa) rollouts using --bc_range
  - sweep:  one rollout per advertiser in the eval period, using each advertiser's
            raw (budget, CPAConstraint) from the constraints parquet

Example:
    python bidding_train_env/online/main_eval_ppo.py \
        --load_path output/online/training/ongoing/001_ppo_seed_0 \
        --eval_mode both --n_eval_episodes 100 --bc_range dense
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bidding_train_env.online.definitions import (
    BC_RANGES,
    OUTPUT_DIR,
    RL_DATA_DIR,
    load_act_keys,
    load_obs_keys,
)
from bidding_train_env.online.helpers import get_last_checkpoint
from bidding_train_env.online.online_env import EnvironmentFactory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load_path", required=True,
                   help="Training run dir (contains rl_model_*_steps.zip)")
    p.add_argument("--checkpoint_num", type=int)
    p.add_argument("--eval_mode", choices=["random", "sweep", "both"], default="both")
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--eval_period", type=int, default=27)
    p.add_argument("--obs_type", default="obs_16_keys")
    p.add_argument("--act_type", default="act_1_key")
    p.add_argument("--bc_range", default="dense", choices=list(BC_RANGES))
    p.add_argument("--dense_weight", type=float, default=1.0)
    p.add_argument("--sparse_weight", type=float, default=0.0)
    p.add_argument("--rl_data_dir", type=str, default=str(RL_DATA_DIR))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temporal_seq_len", type=int, default=1,
                   help="must match training when evaluating temporal PPO checkpoints")
    args = p.parse_args()
    if args.temporal_seq_len < 1:
        p.error(f"--temporal_seq_len must be >= 1, got {args.temporal_seq_len}")
    return args


def build_dummy_vec_env(period, obs_keys, act_keys, bc, rwd_weights, rl_data_dir, seed, temporal_seq_len):
    def _thunk():
        return EnvironmentFactory.create(
            env_name="BiddingEnv",
            pvalues_df_path=str(rl_data_dir / f"period-{period}_pvalues.parquet"),
            bids_df_path=str(rl_data_dir / f"period-{period}_bids.parquet"),
            constraints_df_path=str(rl_data_dir / f"period-{period}_constraints.parquet"),
            obs_keys=obs_keys,
            act_keys=act_keys,
            rwd_weights=rwd_weights,
            budget_range=bc["budget_range"],
            target_cpa_range=bc["target_cpa_range"],
            temporal_seq_len=temporal_seq_len,
            seed=seed,
        )
    return DummyVecEnv([_thunk])


def _unwrap_inner_env(vec_env):
    """Drill through VecNormalize -> DummyVecEnv -> Monitor -> gym.Env -> BiddingEnv."""
    return vec_env.venv.envs[0].unwrapped


def run_rollout(model, vec_env, override=None):
    """Run one deterministic episode. `override` = dict(advertiser, budget, target_cpa) for sweep."""
    if override is not None:
        # Reset first to clear any auto-reset state from the previous episode, then
        # overwrite the random campaign params with the requested advertiser/budget/cpa.
        vec_env.reset()
        inner = _unwrap_inner_env(vec_env)
        state = inner.set_campaign(
            advertiser=override["advertiser"],
            budget=override["budget"],
            target_cpa=override["target_cpa"],
        )
        # Run the overridden raw state through VecNormalize so the policy sees
        # the same obs distribution it was trained on.
        obs = vec_env.normalize_obs(state[None])
    else:
        obs = vec_env.reset()

    total_r = 0.0
    info = {}
    done = np.array([False])
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = vec_env.step(action)
        total_r += float(reward[0])
        if done[0]:
            info = infos[0]
    info["total_reward"] = total_r
    return info


def aggregate(infos, keys=("score", "cost_over_budget", "target_cpa_over_cpa",
                           "conversions", "total_reward")):
    out = {}
    for k in keys:
        xs = np.array([i.get(k, 0.0) for i in infos], dtype=np.float64)
        out[k] = {
            "mean": float(xs.mean()) if len(xs) else 0.0,
            "std": float(xs.std(ddof=0)) if len(xs) else 0.0,
            "sem": float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0,
            "n": int(len(xs)),
        }
    return out


def _json_dump(obj, path):
    def _encoder(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return "<ns>"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_encoder)


def main():
    args = parse_args()
    obs_keys = load_obs_keys(args.obs_type)
    act_keys = load_act_keys(args.act_type)
    bc = BC_RANGES[args.bc_range]
    rwd = {"dense": args.dense_weight, "sparse": args.sparse_weight}

    load_path = Path(args.load_path)
    n = args.checkpoint_num or get_last_checkpoint(load_path)
    if n is None:
        # Fall back to final_model if no step checkpoints are present.
        model_zip = load_path / "final_model.zip"
        vnorm_pkl = load_path / "final_vecnormalize.pkl"
        if not model_zip.exists():
            raise FileNotFoundError(f"No checkpoints or final model at {load_path}")
    else:
        model_zip = load_path / f"rl_model_{n}_steps.zip"
        vnorm_pkl = load_path / f"rl_model_vecnormalize_{n}_steps.pkl"
        if not vnorm_pkl.exists():
            raise FileNotFoundError(f"Missing vecnormalize pkl: {vnorm_pkl}")

    rl_data_dir = Path(args.rl_data_dir)
    vec_env = build_dummy_vec_env(
        args.eval_period, obs_keys, act_keys, bc, rwd, rl_data_dir, args.seed, args.temporal_seq_len
    )
    vec_env = VecNormalize.load(str(vnorm_pkl), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(str(model_zip), env=vec_env)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUTPUT_DIR) / "testing" / load_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_mode in ("random", "both"):
        infos = [run_rollout(model, vec_env) for _ in range(args.n_eval_episodes)]
        result = {
            "mode": "random",
            "n": args.n_eval_episodes,
            "eval_period": args.eval_period,
            "bc_range": args.bc_range,
            "agg": aggregate(infos),
            "per_episode": infos,
        }
        _json_dump(result, out_dir / f"results_random_{ts}.json")
        print(f"[random] score_mean={result['agg']['score']['mean']:.4f} "
              f"(n={args.n_eval_episodes})")

    if args.eval_mode in ("sweep", "both"):
        cons = pd.read_parquet(rl_data_dir / f"period-{args.eval_period}_constraints.parquet")
        cons = cons[cons.deliveryPeriodIndex == args.eval_period]
        infos = []
        for _, c in cons.iterrows():
            override = {
                "advertiser": int(c.advertiserNumber),
                "budget": float(c.budget),
                "target_cpa": float(c.CPAConstraint),
            }
            info = run_rollout(model, vec_env, override=override)
            info.update(override)
            infos.append(info)
        result = {
            "mode": "sweep",
            "n": len(infos),
            "eval_period": args.eval_period,
            "agg": aggregate(infos),
            "per_episode": infos,
        }
        _json_dump(result, out_dir / f"results_sweep_{ts}.json")
        print(f"[sweep] score_mean={result['agg']['score']['mean']:.4f} (n={len(infos)})")


if __name__ == "__main__":
    main()
