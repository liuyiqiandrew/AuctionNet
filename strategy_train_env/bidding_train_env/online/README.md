# Online PPO for AuctionNet

Minimal `stable_baselines3` PPO pipeline over a replay-based `BiddingEnv`.
Train on periods 7–26; evaluate on period 27.

## Layout

```
online/
  online_env.py        # BiddingEnv (gym.Env) + registration + EnvironmentFactory
  online_trainer.py    # OnlineTrainer + ALGO_CLASS_DICT (drop-in for custom variants)
  callbacks.py         # CustomCheckpointCallback, JsonRolloutCallback
  data_generator.py    # aggregation helpers (raw parquet -> pvalues/bids/constraints)
  prepare_data.py      # CLI wrapping data_generator
  main_train_ppo.py    # training entry point
  main_eval_ppo.py     # evaluation entry point (random + sweep modes)
  temporal_policy.py   # optional GRU feature extractor for temporal PPO
  definitions.py       # paths, PPO defaults, HISTORY/NO_HISTORY keys, BC_RANGES
  helpers.py           # safe_mean/max, get_last_checkpoint, get_model_and_env_path
  configs/
    obs_16_keys.json   # default observation schema
    obs_60_keys.json   # richer observation schema
    act_1_key.json     # ["pvalue"] log-bid-coefficient action
```

All commands below are run from `AuctionNet/strategy_train_env/`.

## 1. Prepare data (once)

Aggregate raw per-impression parquets into per-timestep pvalues + bids + per-advertiser constraints:

```bash
python bidding_train_env/online/prepare_data.py \
    --first_period 7 --last_period 27
```

Writes to `AuctionNet/strategy_train_env/data/traffic/online_rl_data/period-{N}_{pvalues,bids,constraints}.parquet`
(63 files for periods 7..27).

## 2. Train

```bash
python bidding_train_env/online/main_train_ppo.py \
    --num_envs 20 --num_steps 10000000 --batch_size 512 \
    --seed 0 --bc_range dense \
    --obs_type obs_16_keys --act_type act_1_key \
    --learning_rate 2e-5 --save_every 10000 \
    --out_prefix 001_ --out_suffix _ppo_dense_obs16
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--num_envs 20` | one `SubprocVecEnv` worker per period 7..26 (add `--use_dummy_vec_env` for debugging) |
| `--bc_range {dense,sparse,default}` | budget/target_cpa sampling range. `default` uses each advertiser's raw `(budget, CPAConstraint)` |
| `--obs_type {obs_16_keys,obs_60_keys}` | which observation schema (default: 16) |
| `--dense_weight / --sparse_weight` | per-step vs end-of-episode reward mix (default 1.0 / 0.0) |
| `--deterministic_conversion` | disable stochastic Bernoulli(N(pV, pVSigma)) sampling |
| `--temporal_seq_len K` | optional state-only temporal PPO; stack last K observations and encode them with a GRU |
| `--temporal_hidden_dim N` | hidden dimension for the temporal GRU feature extractor (default 64) |
| `--save_every N` | checkpoint every N env steps |
| `--load_path DIR --checkpoint_num K` | resume from checkpoint K in DIR (or latest if K omitted) |

Outputs land in `AuctionNet/output/online/training/ongoing/{out_prefix}ppo_seed_{seed}{out_suffix}/`:

```
args.json  env_config.json  model_config.json  main_train_ppo.py (snapshot)
rl_model_{N}_steps.zip  +  rl_model_vecnormalize_{N}_steps.pkl
final_model.zip         +  final_vecnormalize.pkl
```

Training auto-resumes from the latest checkpoint in its own log dir if interrupted.

### Optional temporal PPO

Temporal PPO is an opt-in residual-GRU encoder over the last `K` PPO
observations. The feature list is always the PPO observation schema selected by
`--obs_type`, for example `obs_16_keys` or `obs_60_keys`; it does not rebuild the
IQL testing-state feature list. The IQL branch only supplies the architecture
pattern: GRU history encoding plus a residual latest-observation path and
LayerNorm. Temporal mode uses separate actor/value GRU feature extractors to
mirror the residual-GRU IQL phase as closely as standard SB3 PPO allows.

This mode does not use IQL objectives, offline replay buffers, or
transition-v1/v2 tokens.

```bash
python bidding_train_env/online/main_train_ppo.py \
    --num_envs 20 --num_steps 10000000 --batch_size 512 \
    --seed 0 --bc_range dense \
    --obs_type obs_16_keys --act_type act_1_key \
    --learning_rate 2e-5 --save_every 10000 \
    --out_prefix 001_ --out_suffix _ppo_dense_obs16_temporal8 \
    --device cpu \
    --temporal_seq_len 8 --temporal_hidden_dim 64
```

## 3. Evaluate

```bash
python bidding_train_env/online/main_eval_ppo.py \
    --load_path ../output/online/training/ongoing/001_ppo_seed_0_ppo_dense_obs16 \
    --eval_mode both --n_eval_episodes 100 \
    --obs_type obs_16_keys --bc_range dense
```

For temporal checkpoints, pass the same observation schema and sequence length
used during training:

```bash
python bidding_train_env/online/main_eval_ppo.py \
    --load_path ../output/online/training/ongoing/001_ppo_seed_0_ppo_dense_obs16_temporal8 \
    --eval_mode both --n_eval_episodes 100 \
    --obs_type obs_16_keys --bc_range dense \
    --temporal_seq_len 8
```

`--eval_mode`:
- `random`: N random `(advertiser, budget, target_cpa)` rollouts per `--bc_range`.
- `sweep`:  one deterministic rollout per advertiser in `--eval_period` using raw `(budget, CPAConstraint)`.
- `both`:   runs both, writes two JSONs.

Results land in `AuctionNet/output/online/testing/{run_name}/results_{mode}_{timestamp}.json`
with per-episode rows plus aggregated `score`, `cost_over_budget`, `target_cpa_over_cpa`, `conversions`.

## Smoke test

```bash
python bidding_train_env/online/main_train_ppo.py \
    --num_envs 2 --num_steps 5000 --batch_size 64 --n_rollout_steps 64 \
    --save_every 1000 --device cpu --bc_range dense \
    --use_dummy_vec_env --out_prefix smoke_

python bidding_train_env/online/main_eval_ppo.py \
    --load_path ../output/online/training/ongoing/smoke_ppo_seed_0 \
    --eval_mode random --n_eval_episodes 5
```

Temporal smoke test:

```bash
python bidding_train_env/online/main_train_ppo.py \
    --num_envs 2 --num_steps 5000 --batch_size 64 --n_rollout_steps 64 \
    --save_every 1000 --device cpu --bc_range dense \
    --use_dummy_vec_env --out_prefix smoke_temporal_ \
    --temporal_seq_len 4 --temporal_hidden_dim 32

python bidding_train_env/online/main_eval_ppo.py \
    --load_path ../output/online/training/ongoing/smoke_temporal_ppo_seed_0 \
    --eval_mode random --n_eval_episodes 5 \
    --temporal_seq_len 4
```

## Adding a new algorithm

Register it in `definitions.py::ALGO_CLASS_DICT` and pass `--algo <name>` to
`main_train_ppo.py`. The class must be `stable_baselines3`-compatible (same
`learn` / `load` / `save` / `predict` API as `PPO`).

## Notes

- Action `[-10, 10]` is a log-bid-coefficient; `bid = exp(action) * pvalue * target_cpa`.
- Conversions are stochastic by default: `p ~ clip(N(pValue, pValueSigma), 0, 1)`,
  then `conversion = Bernoulli(p) * bid_exposed`. Both `pValue` and `pValueSigma`
  matter.
- Over-cost correction randomly drops winning bids until step cost ≤ remaining budget.
- PPO hyperparams default to known-stable values from the reference implementation
  (`clip_range=0.3`, `lr=2e-5` linear, `vf_coef=0.5`, `ent_coef=3e-6`,
  `max_grad_norm=0.7`, `gae_lambda=0.9`, `n_epochs=10`, `log_std_init=0.0`).
