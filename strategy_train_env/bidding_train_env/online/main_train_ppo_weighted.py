"""Online PPO training with score-weighted advertiser sampling.

Same as main_train_ppo.py but samples advertisers in env reset by historical score:
high-scoring advertisers (under the original logged policy) are sampled more often.

Example:
    python bidding_train_env/online/main_train_ppo_weighted.py \
        --num_envs 20 --num_steps 10_000_000 --batch_size 512 \
        --seed 0 --bc_range default --out_prefix 001_weighted_T10_ \
        --obs_type obs_16_keys --learning_rate 1e-4 --save_every 10000 \
        --temperature 10 --alpha 0.9
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from bidding_train_env.online.callbacks import (
    CustomCheckpointCallback,
    JsonRolloutCallback,
)
from bidding_train_env.online.definitions import (
    ALGO_CLASS_DICT,
    BC_RANGES,
    INFO_KEYWORDS,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    RL_DATA_DIR,
    load_act_keys,
    load_obs_keys,
)
from bidding_train_env.online.helpers import get_model_and_env_path
from bidding_train_env.online.online_trainer import OnlineTrainer
from bidding_train_env.online.score_weights import (
    compute_advertiser_weights,
    compute_period_advertiser_scores,
    effective_sample_size,
)
from bidding_train_env.online.weighted_env import WeightedBiddingEnv  # noqa: F401  (registers env)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo", choices=list(ALGO_CLASS_DICT))
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--first_period", type=int, default=7)
    p.add_argument("--num_envs", type=int, default=20)
    p.add_argument("--rl_data_dir", type=str, default=str(RL_DATA_DIR))
    p.add_argument("--raw_data_dir", type=str, default=str(RAW_DATA_DIR))

    p.add_argument("--num_steps", type=int, default=10_000_000)
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--device", default="cuda")

    p.add_argument("--obs_type", default="obs_16_keys")
    p.add_argument("--act_type", default="act_1_key")
    p.add_argument("--bc_range", default="default", choices=list(BC_RANGES))
    p.add_argument("--dense_weight", type=float, default=1.0)
    p.add_argument("--sparse_weight", type=float, default=0.0)
    p.add_argument("--lambda_cpa", type=float, default=0.0)
    p.add_argument("--deterministic_conversion", action="store_true")

    p.add_argument("--net_arch", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_rollout_steps", type=int, default=128)
    p.add_argument("--ent_coef", type=float, default=3e-6)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--clip_range", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.9)
    p.add_argument("--max_grad_norm", type=float, default=0.7)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_std_init", type=float, default=0.0)

    p.add_argument("--out_prefix", default="")
    p.add_argument("--out_suffix", default="")
    p.add_argument("--load_path")
    p.add_argument("--checkpoint_num", type=int)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--use_dummy_vec_env", action="store_true")

    # Score-weighted sampling.
    p.add_argument("--temperature", type=float, default=10.0,
                   help="softmax temperature on per-advertiser score (lower = sharper).")
    p.add_argument("--alpha", type=float, default=0.9,
                   help="mix weight: alpha*softmax + (1-alpha)*uniform.")

    return p.parse_args()


def build_env_configs(args, obs_keys, act_keys):
    bc = BC_RANGES[args.bc_range]
    rl_data_dir = Path(args.rl_data_dir)
    raw_data_dir = Path(args.raw_data_dir)
    cfgs = []
    weight_summary = []
    for i in range(args.num_envs):
        period = args.first_period + i

        # Determine the advertiser_list for this period from raw data
        # (it always contains all 48 advertisers, but we read once to be safe).
        scores_df = compute_period_advertiser_scores(period, raw_data_dir)
        advertiser_list = sorted(scores_df["advertiserNumber"].astype(float).tolist())
        weights = compute_advertiser_weights(
            period=period,
            raw_data_dir=raw_data_dir,
            temperature=args.temperature,
            alpha=args.alpha,
            advertiser_list=advertiser_list,
        )
        weight_summary.append({
            "period": period,
            "n_advertisers": len(advertiser_list),
            "ess": effective_sample_size(weights),
            "max_weight": float(weights.max()),
            "min_weight": float(weights.min()),
        })

        cfgs.append(
            dict(
                env_name="WeightedBiddingEnv",
                pvalues_df_path=str(rl_data_dir / f"period-{period}_pvalues.parquet"),
                bids_df_path=str(rl_data_dir / f"period-{period}_bids.parquet"),
                constraints_df_path=str(rl_data_dir / f"period-{period}_constraints.parquet"),
                obs_keys=obs_keys,
                act_keys=act_keys,
                rwd_weights={"dense": args.dense_weight, "sparse": args.sparse_weight},
                budget_range=bc["budget_range"],
                target_cpa_range=bc["target_cpa_range"],
                deterministic_conversion=args.deterministic_conversion,
                lambda_cpa=args.lambda_cpa,
                seed=args.seed + i,
                advertiser_weights=weights.tolist(),
            )
        )
    return cfgs, weight_summary


def make_vec_env(cfgs, log_dir, use_dummy=False):
    def make_thunk(cfg):
        def _thunk():
            local_cfg = dict(cfg)
            local_cfg.pop("env_name", None)
            env = gym.make("WeightedBiddingEnv-v0", **local_cfg)
            return Monitor(env, log_dir, info_keywords=INFO_KEYWORDS)
        return _thunk

    thunks = [make_thunk(c) for c in cfgs]
    return DummyVecEnv(thunks) if use_dummy else SubprocVecEnv(thunks)


def main():
    args = parse_args()
    obs_keys = load_obs_keys(args.obs_type)
    act_keys = load_act_keys(args.act_type)

    run_name = f"{args.out_prefix}{args.algo}_seed_{args.seed}{args.out_suffix}"
    log_dir = Path(OUTPUT_DIR) / "training" / "ongoing" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(__file__, log_dir / Path(__file__).name)
    with open(log_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    cfgs, weight_summary = build_env_configs(args, obs_keys, act_keys)

    serializable = {k: v for k, v in cfgs[0].items()}
    with open(log_dir / "env_config.json", "w") as f:
        json.dump(serializable, f, indent=2, default=lambda _: "<ns>")
    with open(log_dir / "weight_summary.json", "w") as f:
        json.dump(weight_summary, f, indent=2)

    print("[weighted] per-period sampling weight summary:")
    for s in weight_summary:
        print(f"  period {s['period']:>2}: ess={s['ess']:.1f}/{s['n_advertisers']}  "
              f"max={s['max_weight']:.4f}  min={s['min_weight']:.4f}")

    model_cfg = dict(
        policy="MlpPolicy",
        device=args.device,
        batch_size=args.batch_size,
        n_steps=args.n_rollout_steps,
        learning_rate=lambda x: x * args.learning_rate,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        seed=args.seed,
        policy_kwargs=dict(
            log_std_init=args.log_std_init,
            activation_fn=nn.ReLU,
            net_arch=dict(pi=list(args.net_arch), vf=list(args.net_arch)),
        ),
    )

    envs = make_vec_env(cfgs, str(log_dir), use_dummy=args.use_dummy_vec_env)
    model_path, env_path = get_model_and_env_path(
        str(log_dir), args.load_path, args.checkpoint_num
    )
    envs = VecNormalize.load(env_path, envs) if env_path else VecNormalize(envs)

    save_freq_per_env = max(1, args.save_every // max(args.num_envs, 1))
    ckpt_cb = CustomCheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=str(log_dir),
        name_prefix="rl_model",
        save_vecnormalize=True,
        verbose=2,
    )
    rollout_cb = JsonRolloutCallback(
        info_keywords=INFO_KEYWORDS,
        log_path=log_dir / "rollout_log.jsonl",
        log_interval=args.log_interval,
    )

    trainer = OnlineTrainer(
        algo=args.algo,
        envs=envs,
        load_model_path=model_path,
        log_dir=str(log_dir),
        model_config=model_cfg,
        callbacks=[rollout_cb, ckpt_cb],
        timesteps=args.num_steps,
    )
    trainer.train()
    trainer.save()


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
