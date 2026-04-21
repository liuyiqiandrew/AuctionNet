"""Online PPO training entry point for shared temporal GRU feature sweeps.

This variant intentionally keeps the same CLI as ``main_train_ppo.py``. The only
behavioral difference is that temporal PPO uses one shared GRU feature extractor
for the actor and value networks instead of separate extractors.
"""

import json
import shutil
import sys
from pathlib import Path

# Make `bidding_train_env` importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize

from bidding_train_env.online.callbacks import (
    CustomCheckpointCallback,
    JsonRolloutCallback,
)
from bidding_train_env.online.definitions import (
    INFO_KEYWORDS,
    OUTPUT_DIR,
    load_act_keys,
    load_obs_keys,
)
from bidding_train_env.online.helpers import get_model_and_env_path
from bidding_train_env.online.main_train_ppo import (
    build_env_configs,
    make_vec_env,
    parse_args,
)
from bidding_train_env.online.online_trainer import OnlineTrainer
from bidding_train_env.online.temporal_policy import TemporalGRUFeaturesExtractor


def main():
    args = parse_args()
    obs_keys = load_obs_keys(args.obs_type)
    act_keys = load_act_keys(args.act_type)

    run_name = f"{args.out_prefix}{args.algo}_seed_{args.seed}{args.out_suffix}"
    log_dir = Path(OUTPUT_DIR) / "training" / "ongoing" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the launcher + parsed args for reproducibility.
    shutil.copy(__file__, log_dir / Path(__file__).name)
    with open(log_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    cfgs = build_env_configs(args, obs_keys, act_keys)
    # Dump one env_config so it is obvious what data each worker sees.
    serializable = {k: v for k, v in cfgs[0].items()}
    with open(log_dir / "env_config.json", "w") as f:
        json.dump(serializable, f, indent=2, default=lambda _: "<ns>")

    policy_kwargs = dict(
        log_std_init=args.log_std_init,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=list(args.net_arch), vf=list(args.net_arch)),
    )
    if args.temporal_seq_len > 1:
        policy_kwargs.update(
            features_extractor_class=TemporalGRUFeaturesExtractor,
            share_features_extractor=True,
            features_extractor_kwargs=dict(
                obs_dim=len(obs_keys),
                seq_len=args.temporal_seq_len,
                hidden_dim=args.temporal_hidden_dim,
                use_residual_latest_state=not args.no_temporal_residual_latest_state,
            ),
        )

    model_cfg = dict(
        policy="MlpPolicy",
        device=args.device,
        batch_size=args.batch_size,
        n_steps=args.n_rollout_steps,
        learning_rate=lambda x: x * args.learning_rate,  # linear schedule
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
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
