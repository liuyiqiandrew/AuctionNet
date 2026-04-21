# Temporal PPO Changes

This note documents the optional temporal PPO extension added on top of the
original online PPO implementation in `ppo_new`.

## Summary

The original online PPO path used one flat observation at each timestep:

```text
obs_t -> MLP policy/value network -> log bid coefficient
```

The temporal PPO path is opt-in and keeps the same PPO algorithm, action space,
reward logic, checkpoint format, and evaluation flow. When enabled, the
environment returns the last `K` PPO observations as one flattened vector, and
the policy uses residual-GRU feature extractors before the usual PPO actor/value
heads:

```text
[obs_{t-K+1}, ..., obs_t] -> GRU + latest-observation residual -> PPO heads
```

This ports the residual-GRU architecture pattern from temporal IQL phase 2, not
the IQL feature list. PPO temporal mode uses the PPO feature list selected by
`--obs_type`, such as `obs_16_keys` or `obs_60_keys`. It does not port IQL
objectives, offline replay buffers, transition-v1/v2 tokens, or IQL-specific
checkpoint selection.

## What Stayed The Same

- Plain PPO commands still run unchanged because `--temporal_seq_len` defaults
  to `1`.
- The action remains the same one-dimensional log bid coefficient used by
  `act_1_key`.
- The bidding environment still computes bids, costs, conversions, dense reward,
  sparse reward, and terminal metrics the same way.
- Stable Baselines3 `PPO` is still the algorithm class. No `sb3-contrib` or
  recurrent PPO dependency was added.
- Existing checkpoint/resume behavior still uses `rl_model_*_steps.zip` plus
  `rl_model_vecnormalize_*_steps.pkl`.

## Files Changed

### `online_env.py`

Added `temporal_seq_len`, defaulting to `1`.

When `temporal_seq_len == 1`, the environment returns exactly the original flat
observation with shape:

```text
(obs_dim,)
```

When `temporal_seq_len > 1`, the environment stores the recent selected
PPO `obs_keys` states and returns a flattened chronological stack with shape:

```text
(temporal_seq_len * obs_dim,)
```

Early timesteps are padded by repeating the earliest available observation:

```text
t = 0, K = 4:
[obs_0, obs_0, obs_0, obs_0]

t = 1, K = 4:
[obs_0, obs_0, obs_0, obs_1]
```

This avoids a separate mask or zero-padding convention and keeps `VecNormalize`
simple.

### `temporal_policy.py`

Added `TemporalGRUFeaturesExtractor`, a Stable Baselines3
`BaseFeaturesExtractor`.

It:

- validates that the flattened observation shape matches `seq_len * obs_dim`
- reshapes `(batch, seq_len * obs_dim)` to `(batch, seq_len, obs_dim)`
- encodes the sequence with `nn.GRU(batch_first=True)`
- concatenates the GRU hidden state with the latest observation by default
- applies `LayerNorm` to the residual feature

The residual feature follows the temporal IQL phase-2 idea:

```text
h_t = GRU(sequence)
x_t = latest observation
z_t = LayerNorm(concat(h_t, x_t))
```

If `--no_temporal_residual_latest_state` is passed during training, the feature
is only the GRU hidden state.

### `main_train_ppo.py`

Added temporal training flags:

```bash
--temporal_seq_len
--temporal_hidden_dim
--no_temporal_residual_latest_state
```

The temporal feature extractor is only installed into `policy_kwargs` when
`--temporal_seq_len > 1`. Temporal mode also sets
`share_features_extractor=False`, so the PPO actor and value networks each get
their own residual-GRU feature extractor. This is the closest PPO analogue to
the separate residual-GRU encoders used by the temporal IQL phase-2 actor/value
heads. Otherwise, `policy_kwargs` stays equivalent to the original MLP PPO
configuration.

The env config written to each training run now includes `temporal_seq_len`, so
the run directory records whether the checkpoint was trained with flat or
temporal observations.

### `main_eval_ppo.py`

Added:

```bash
--temporal_seq_len
```

Evaluation must use the same `--obs_type` and `--temporal_seq_len` as training.
If a temporal checkpoint is evaluated with `--temporal_seq_len 1`, the
observation shape will not match the saved PPO policy.

### `README.md`

Documented optional temporal PPO usage, temporal smoke-test commands, and the
distinction between PPO-observation residual-GRU temporal PPO and temporal IQL
transition-token experiments.

## How To Run

Plain PPO remains unchanged:

```bash
python bidding_train_env/online/main_train_ppo.py \
    --num_envs 20 --num_steps 10000000 --batch_size 512 \
    --seed 0 --bc_range dense \
    --obs_type obs_16_keys --act_type act_1_key \
    --learning_rate 2e-5 --save_every 10000 \
    --out_prefix 001_ --out_suffix _ppo_dense_obs16
```

Temporal PPO:

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

Temporal evaluation:

```bash
python bidding_train_env/online/main_eval_ppo.py \
    --load_path ../output/online/training/ongoing/001_ppo_seed_0_ppo_dense_obs16_temporal8 \
    --eval_mode both --n_eval_episodes 100 \
    --obs_type obs_16_keys --bc_range dense \
    --temporal_seq_len 8
```

## Verification Performed

Static checks:

```bash
python -m py_compile \
    bidding_train_env/online/online_env.py \
    bidding_train_env/online/main_train_ppo.py \
    bidding_train_env/online/main_eval_ppo.py \
    bidding_train_env/online/temporal_policy.py

bash -n strategy_train_env/train_eval_ppo_cpu.sbatch
git diff --check
```

Runtime smoke checks were also run in the `AuctionNet` conda environment:

- flat PPO trained for 16 CPU steps and evaluated for one random episode
- temporal PPO trained for 16 CPU steps with `--temporal_seq_len 4`
  and evaluated for one random episode with the same sequence length
- temporal PPO was checked with `obs_60_keys` to verify the residual GRU follows
  the selected PPO feature list rather than a hardcoded IQL feature list

Both paths loaded checkpoints and evaluation environments without observation
shape errors.

## Design Notes

- Repeating the earliest observation at episode start keeps the sequence fully
  valid at every timestep and avoids introducing a mask into SB3 PPO.
- The GRU feature extractor keeps PPO feed-forward from SB3's perspective. PPO
  still samples rollouts and minibatches normally; temporal context is supplied
  by the environment observation itself.
- Temporal mode uses separate actor/value feature extractors via
  `share_features_extractor=False`; plain PPO keeps the original shared default.
- `VecNormalize` normalizes the flattened temporal observation. This means each
  lag position receives its own running statistics.
- Transition-aware temporal tokens from IQL can be added later, but they would
  require defining previous-action/reward/outcome fields inside the online PPO
  env and ensuring train/eval parity.
