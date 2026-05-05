"""Drop-in eval for PPO checkpoints that can't be loaded via `PPO.load`.

Use when the checkpoint was trained under a different torch/cloudpickle version
and `PPO.load(...)` segfaults during pickle rehydration. We rebuild a fresh PPO
from `model_config.json` + `env_config.json` and load only the tensor
`state_dict` from the zip — no cloudpickle roundtrip.

Same CLI as `main_eval_ppo.py` — e.g.:

    python bidding_train_env/online/main_eval_ppo_v2.py \
      --load_path ../output/online/training/ongoing/002_lambda0.0001_ppo_seed_0_ppo_default_obs16/ \
      --eval_mode both --n_eval_episodes 100 --obs_type obs_16_keys --bc_range default
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bidding_train_env.online.config import BC_RANGES, OUTPUT_DIR
from bidding_train_env.online.definitions import RL_DATA_DIR, load_act_keys, load_obs_keys
from bidding_train_env.online.online_env import EnvironmentFactory

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load_path", required=True)
    p.add_argument("--checkpoint_num", type=int)
    p.add_argument("--eval_mode", choices=["random", "sweep", "both"], default="both")
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--eval_period", type=int, default=27)
    p.add_argument("--obs_type", default="obs_16_keys")
    p.add_argument("--act_type", default="act_1_key")
    p.add_argument("--bc_range", default="default", choices=list(BC_RANGES))
    p.add_argument("--dense_weight", type=float, default=1.0)
    p.add_argument("--sparse_weight", type=float, default=0.0)
    p.add_argument("--rl_data_dir", type=str, default=str(RL_DATA_DIR))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--activation_fn", default="tanh", choices=list(_ACTIVATIONS),
                   help="model_config.json drops the activation class; re-supply here")
    return p.parse_args()


def build_dummy_vec_env(period, obs_keys, act_keys, bc, rwd, rl_data_dir, seed):
    def thunk():
        return EnvironmentFactory.create(
            env_name="BiddingEnv",
            pvalues_df_path=str(rl_data_dir / f"period-{period}_pvalues.parquet"),
            bids_df_path=str(rl_data_dir / f"period-{period}_bids.parquet"),
            constraints_df_path=str(rl_data_dir / f"period-{period}_constraints.parquet"),
            obs_keys=obs_keys, act_keys=act_keys, rwd_weights=rwd,
            budget_range=bc["budget_range"], target_cpa_range=bc["target_cpa_range"],
            seed=seed,
        )
    return DummyVecEnv([thunk])


def _unwrap_inner_env(vec_env):
    return vec_env.venv.envs[0].unwrapped


def load_ppo_from_state_dict(zip_path: Path, env, activation_fn: str) -> PPO:
    """Build a fresh PPO with the architecture from model_config.json,
    then overwrite its weights with the zip's `policy.pth` tensor state_dict."""
    cfg = json.loads((zip_path.parent / "model_config.json").read_text())
    pkw = dict(cfg["policy_kwargs"])
    pkw["activation_fn"] = _ACTIVATIONS[activation_fn]

    # Non-inference-critical fields — we just need a well-formed PPO.
    model = PPO(
        policy=cfg["policy"],
        env=env,
        policy_kwargs=pkw,
        learning_rate=3e-4, n_steps=cfg.get("n_steps", 128),
        batch_size=cfg.get("batch_size", 64), n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 0.99), gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2), ent_coef=cfg.get("ent_coef", 0.0),
        vf_coef=cfg.get("vf_coef", 0.5), max_grad_norm=cfg.get("max_grad_norm", 0.5),
        seed=cfg.get("seed", 0), device="cpu",
    )
    with zipfile.ZipFile(zip_path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")),
                        weights_only=True, map_location="cpu")
    model.policy.load_state_dict(sd)
    model.policy.eval()
    return model


def run_rollout(model, vec_env, override=None):
    if override is not None:
        vec_env.reset()
        inner = _unwrap_inner_env(vec_env)
        state = inner.set_campaign(
            advertiser=override["advertiser"],
            budget=override["budget"],
            target_cpa=override["target_cpa"],
        )
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


def main():
    args = parse_args()
    obs_keys = load_obs_keys(args.obs_type)
    act_keys = load_act_keys(args.act_type)
    bc = BC_RANGES[args.bc_range]
    rwd = {"dense": args.dense_weight, "sparse": args.sparse_weight}

    load_path = Path(args.load_path)
    zip_path = load_path / (
        f"rl_model_{args.checkpoint_num}_steps.zip" if args.checkpoint_num else "final_model.zip"
    )
    vnorm_pkl = load_path / (
        f"rl_model_vecnormalize_{args.checkpoint_num}_steps.pkl"
        if args.checkpoint_num else "final_vecnormalize.pkl"
    )
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    rl_data_dir = Path(args.rl_data_dir)
    vec_env = build_dummy_vec_env(
        args.eval_period, obs_keys, act_keys, bc, rwd, rl_data_dir, args.seed
    )
    vec_env = VecNormalize.load(str(vnorm_pkl), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = load_ppo_from_state_dict(zip_path, vec_env, args.activation_fn)
    print(f"[load] reconstructed PPO from state_dict at {zip_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUTPUT_DIR) / "testing" / load_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _dump(obj, path):
        def _enc(o):
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return "<ns>"
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_enc)

    if args.eval_mode in ("random", "both"):
        infos = [run_rollout(model, vec_env) for _ in range(args.n_eval_episodes)]
        result = {"mode": "random", "n": args.n_eval_episodes,
                  "eval_period": args.eval_period, "bc_range": args.bc_range,
                  "agg": aggregate(infos), "per_episode": infos}
        _dump(result, out_dir / f"results_random_v2_{ts}.json")
        print(f"[random] score_mean={result['agg']['score']['mean']:.4f} "
              f"(n={args.n_eval_episodes})")

    if args.eval_mode in ("sweep", "both"):
        cons = pd.read_parquet(
            rl_data_dir / f"period-{args.eval_period}_constraints.parquet"
        )
        cons = cons[cons.deliveryPeriodIndex == args.eval_period]
        infos = []
        for _, c in cons.iterrows():
            override = {"advertiser": int(c.advertiserNumber),
                        "budget": float(c.budget),
                        "target_cpa": float(c.CPAConstraint)}
            info = run_rollout(model, vec_env, override=override)
            info.update(override)
            infos.append(info)
        result = {"mode": "sweep", "n": len(infos),
                  "eval_period": args.eval_period,
                  "agg": aggregate(infos), "per_episode": infos}
        _dump(result, out_dir / f"results_sweep_v2_{ts}.json")
        print(f"[sweep] score_mean={result['agg']['score']['mean']:.4f} (n={len(infos)})")


if __name__ == "__main__":
    main()
