# Online PPO for AuctionNet

Replay-based `stable_baselines3` PPO training for AuctionNet bidding. Train on
periods 7-26 and evaluate on period 27. All commands below are run from
`AuctionNet/strategy_train_env/`.

## Layout

```text
online/
  prepare_data.py      # raw parquet -> replay-ready pvalues/bids/constraints
  online_env.py        # BiddingEnv, WeightedBiddingEnv dispatch, observation logic
  main_train_ppo.py    # PPO training entry point
  main_eval_ppo.py     # random and sweep evaluation entry point
  temporal_policy.py   # optional residual-GRU feature extractor
  score_weights.py     # score-weighted advertiser sampling helpers
  weighted_env.py      # WeightedBiddingEnv reset-time advertiser sampler
  configs/
    obs_16_keys.json
    obs_60_keys.json
    obs_market_regime_v1.json
    act_1_key.json
```

## 1. Prepare Data

```bash
python bidding_train_env/online/prepare_data.py \
  --first_period 7 --last_period 27
```

This writes replay files to
`data/traffic/online_rl_data/period-{N}_{pvalues,bids,constraints}.parquet`.

## 2. Train and Evaluate

Training outputs are written to
`../output/online/training/ongoing/{out_prefix}ppo_seed_{seed}{out_suffix}/`.
Evaluation outputs are written to `../output/online/testing/{run_name}/`.

`main_eval_ppo.py` supports:

- `random`: sampled campaigns from `--bc_range`
- `sweep`: one deterministic rollout per advertiser in `--eval_period`
- `both`: run both modes

### Baseline PPO

Train:

```bash
python bidding_train_env/online/main_train_ppo.py \
  --num_envs 20 \
  --num_steps 10000000 \
  --batch_size 512 \
  --seed 0 \
  --bc_range default \
  --obs_type obs_16_keys \
  --act_type act_1_key \
  --learning_rate 2e-5 \
  --save_every 100000 \
  --out_prefix 001_ \
  --out_suffix _ppo_default_obs16
```

Evaluate:

```bash
python bidding_train_env/online/main_eval_ppo.py \
  --load_path ../output/online/training/ongoing/001_ppo_seed_0_ppo_default_obs16 \
  --eval_mode both \
  --n_eval_episodes 100 \
  --eval_period 27 \
  --obs_type obs_16_keys \
  --act_type act_1_key \
  --bc_range default \
  --rl_data_dir data/traffic/online_rl_data
```

Base arguments:

- `--num_envs 20`: train one vectorized environment per period, starting at
  `--first_period 7`, so the default covers periods 7-26.
- `--bc_range default`: use each advertiser's raw budget and CPA constraint.
- `--obs_type`: choose the observation schema JSON in `online/configs/`.
- `--act_type act_1_key`: one log bid coefficient action.
- `--dense_weight` / `--sparse_weight`: per-step vs terminal reward mix,
  defaulting to `1.0 / 0.0`.
- `--load_path`: training run directory containing checkpoints or
  `final_model.zip` and `final_vecnormalize.pkl`.

### Temporal PPO

Temporal PPO keeps PPO unchanged but returns the last `K` selected observations
as a flattened stack and encodes them with a residual GRU feature extractor.

Train:

```bash
python bidding_train_env/online/main_train_ppo.py \
  --num_envs 20 \
  --num_steps 10000000 \
  --batch_size 512 \
  --seed 0 \
  --bc_range default \
  --obs_type obs_16_keys \
  --act_type act_1_key \
  --learning_rate 2e-5 \
  --save_every 100000 \
  --out_prefix 001_temporal8_ \
  --out_suffix _ppo_default_obs16 \
  --temporal_seq_len 8 \
  --temporal_hidden_dim 64
```

Evaluate:

```bash
python bidding_train_env/online/main_eval_ppo.py \
  --load_path ../output/online/training/ongoing/001_temporal8_ppo_seed_0_ppo_default_obs16 \
  --eval_mode both \
  --n_eval_episodes 100 \
  --eval_period 27 \
  --obs_type obs_16_keys \
  --act_type act_1_key \
  --bc_range default \
  --temporal_seq_len 8 \
  --rl_data_dir data/traffic/online_rl_data
```

Temporal-specific arguments:

- `--temporal_seq_len K`: stack the last `K` PPO observations; default `1`
  disables temporal mode.
- `--temporal_hidden_dim N`: GRU hidden dimension, default `64`.
- `--no_temporal_residual_latest_state`: use only the GRU hidden state instead
  of concatenating the latest observation.
- Evaluation must use the same `--obs_type` and `--temporal_seq_len` as
  training.

### Score-Weighted PPO

Score-weighted PPO keeps the policy, action, reward, and eval flow unchanged,
but samples advertisers at `env.reset()` from
`alpha * softmax(score / temperature) + (1 - alpha) * uniform`, where score is
the logged-policy NeurIPS score computed from raw period parquets.

Train:

```bash
python bidding_train_env/online/main_train_ppo.py \
  --num_envs 20 \
  --num_steps 10000000 \
  --batch_size 512 \
  --seed 0 \
  --bc_range default \
  --obs_type obs_16_keys \
  --act_type act_1_key \
  --learning_rate 2e-5 \
  --save_every 100000 \
  --out_prefix 003_t30_ \
  --out_suffix _ppo_default_obs16 \
  --weighted-sampling \
  --temperature 30 \
  --alpha 0.9 \
  --raw_data_dir data/traffic
```

Evaluate:

```bash
python bidding_train_env/online/main_eval_ppo.py \
  --load_path ../output/online/training/ongoing/003_t30_ppo_seed_0_ppo_default_obs16 \
  --eval_mode both \
  --n_eval_episodes 100 \
  --eval_period 27 \
  --obs_type obs_16_keys \
  --act_type act_1_key \
  --bc_range default \
  --rl_data_dir data/traffic/online_rl_data
```

Weighted-specific arguments:

- `--weighted-sampling`: switch reset-time advertiser sampling from uniform to
  score-weighted.
- `--temperature`: softmax temperature; lower values make sampling sharper.
- `--alpha`: mixture weight between score-weighted sampling and a uniform
  floor.
- `--raw_data_dir`: raw `period-{N}.parquet` directory used to compute scores.
- Evaluation uses the normal PPO evaluator; no weighted-specific eval flag is
  needed.

### Reward-Shaped PPO

Reward-shaped PPO adds a per-step CPA overspend penalty to the dense reward:
`reward -= lambda_cpa * max(0, step_cost - target_cpa * step_conversions)`.
`lambda_cpa=0.0` reproduces baseline PPO.

Train:

```bash
python bidding_train_env/online/main_train_ppo.py \
  --num_envs 20 \
  --num_steps 10000000 \
  --batch_size 512 \
  --seed 0 \
  --bc_range default \
  --obs_type obs_16_keys \
  --act_type act_1_key \
  --learning_rate 2e-5 \
  --save_every 100000 \
  --out_prefix 002_lambda0.5_ \
  --out_suffix _ppo_default_obs16 \
  --lambda_cpa 0.5
```

Evaluate:

```bash
python bidding_train_env/online/main_eval_ppo.py \
  --load_path ../output/online/training/ongoing/002_lambda0.5_ppo_seed_0_ppo_default_obs16 \
  --eval_mode both \
  --n_eval_episodes 100 \
  --eval_period 27 \
  --obs_type obs_16_keys \
  --act_type act_1_key \
  --bc_range default \
  --rl_data_dir data/traffic/online_rl_data
```

Reward-shaping-specific arguments:

- `--lambda_cpa`: penalty weight for per-step CPA overspend. Sweep values such
  as `0.0`, `0.5`, `1.0`, and `2.0` against a matched `bc_range` baseline.
- Evaluation uses the normal PPO evaluator; compare `sweep` `score`,
  `cost_over_budget`, and `target_cpa_over_cpa`.
- If all shaped runs collapse, try smaller λ values before changing other PPO
  hyperparameters.

### Market-Regime PPO

Market-regime PPO changes only the observation schema. `obs_market_regime_v1`
extends `obs_60_keys` with CPA trend, slot exposure/win counts, cumulative
episode quantities, and derived opponent/regime features from `history_info`.

Train:

```bash
python bidding_train_env/online/main_train_ppo.py \
  --num_envs 20 \
  --num_steps 5000000 \
  --batch_size 512 \
  --seed 0 \
  --bc_range default \
  --obs_type obs_market_regime_v1 \
  --act_type act_1_key \
  --learning_rate 1e-4 \
  --save_every 1000000 \
  --out_prefix 022_full_obs_market_regime_v1_ \
  --out_suffix _5m_all \
  --rl_data_dir data/traffic/online_rl_data
```

Evaluate:

```bash
python bidding_train_env/online/main_eval_ppo.py \
  --load_path ../output/online/training/ongoing/022_full_obs_market_regime_v1_ppo_seed_0_5m_all \
  --eval_mode both \
  --n_eval_episodes 100 \
  --eval_period 27 \
  --obs_type obs_market_regime_v1 \
  --act_type act_1_key \
  --bc_range default \
  --temporal_seq_len 1 \
  --rl_data_dir data/traffic/online_rl_data
```

Market-regime-specific arguments:

- `--obs_type obs_market_regime_v1`: selects the 98-key market-regime
  observation vector.
- No action, reward, reset, bidding, or evaluation logic changes are required.
- Compare against an `obs_60_keys` baseline at the same step count, preferably
  using `sweep` metrics on period 27.
