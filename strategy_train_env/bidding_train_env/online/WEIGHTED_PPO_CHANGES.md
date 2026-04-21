# Score-Weighted PPO Changes

This note documents the optional score-weighted advertiser sampling extension
added on top of the original online PPO implementation in `ppo_new`.

## Summary

The original online PPO path samples advertisers **uniformly** at each
`env.reset()`:

```text
reset -> uniform over advertiser_list -> rollout
```

The score-weighted path replaces that uniform draw with a **softmax over
per-advertiser historical NeurIPS score** under the original logged policy. The
PPO algorithm, action space, reward logic, observation keys, checkpoint format,
and evaluation flow are unchanged.

```text
reset -> p = alpha * softmax(score / T) + (1 - alpha) * uniform -> rollout
```

High-scoring advertisers (under the logged policy) are sampled more often during
training. This is intended as a form of difficulty / informative-trajectory
weighting analogous to prioritized sampling in offline RL.

## What Stayed The Same

- PPO algorithm, hyperparameters, checkpoint format, and evaluation flow are
  untouched. `main_eval_ppo.py` can evaluate these runs with no changes.
- Action remains the one-dimensional log bid coefficient (`act_1_key`).
- Reward, observation keys (`obs_16_keys` / `obs_60_keys`), and environment
  dynamics are identical to plain PPO.
- Stable Baselines3 `PPO` is still the algorithm class.
- Existing checkpoint/resume behavior still uses `rl_model_*_steps.zip` plus
  `rl_model_vecnormalize_*_steps.pkl`.

## Files Added

### `score_weights.py`

Computes per-advertiser sampling weights for a single period.

- `compute_period_advertiser_scores(period, raw_data_dir)`: per-advertiser
  NeurIPS score = `conversions * min(1, (CPAConstraint / realCPA)^2)`, where
  `realCPA = sum(cost) / sum(conversions)`. Reads
  `raw_data_dir/period-{period}.parquet`.
- `compute_advertiser_weights(period, raw_data_dir, temperature, alpha,
  advertiser_list)`: returns `p = alpha * softmax(score / T) + (1 - alpha) *
  uniform`.
- `effective_sample_size(weights)`: `1 / sum(w^2)` — logged per period at
  launch to sanity-check how peaked the distribution is.

### `weighted_env.py`

`WeightedBiddingEnv(BiddingEnv)`: overrides `_reset_campaign_params` to sample
`advertiser` from `advertiser_list` using a pre-computed `advertiser_weights`
vector instead of `np_random.choice` uniform. Validates that the weight vector
length matches `advertiser_list` length. Registers as `WeightedBiddingEnv-v0`.

### `main_train_ppo_weighted.py`

Launcher. Mirrors `main_train_ppo.py` with two added flags:

```bash
--temperature   # softmax temperature on per-advertiser score (lower = sharper)
--alpha         # mix weight: alpha * softmax + (1 - alpha) * uniform
```

At startup it computes per-period weights via `compute_advertiser_weights`,
passes them into each `WeightedBiddingEnv` via `gym.make(...)`, and logs an
ESS summary per period to `weight_summary.json`. Everything else (vec-env
construction, callbacks, checkpoints, `VecNormalize`) is identical to
`main_train_ppo.py`.

## How To Run

Local smoke (CPU):

```bash
python bidding_train_env/online/main_train_ppo_weighted.py \
    --num_envs 20 --num_steps 200000 --batch_size 512 \
    --seed 0 --bc_range default \
    --obs_type obs_16_keys --act_type act_1_key \
    --learning_rate 2e-5 --save_every 10000 \
    --temperature 30 --alpha 0.9 \
    --out_prefix 003_t30_ --out_suffix _ppo_default_obs16 \
    --device cpu --use_dummy_vec_env
```

Full 10M-step run on Adroit:

```bash
sbatch slurm/weighted_t30_10m.slurm
```

The slurm script reads `$AUCTIONNET_ROOT` with a default of
`/scratch/network/$USER/AuctionNet`; override this before `sbatch` if your
checkout is elsewhere:

```bash
AUCTIONNET_ROOT=/your/path sbatch slurm/weighted_t30_10m.slurm
```

Evaluation uses the unchanged `main_eval_ppo.py`:

```bash
python bidding_train_env/online/main_eval_ppo.py \
    --load_path ../output/online/training/ongoing/003_t30_ppo_seed_0_ppo_default_obs16 \
    --eval_mode both --n_eval_episodes 100 \
    --obs_type obs_16_keys --bc_range default
```

## Hyperparameter Selection

Temperature `T` was swept in `{10, 20, 30, 50}` with `alpha = 0.9` using a
50k and (some on) 200k-step local smoke run per setting:

- `T = 30` and `T = 50` tied for the highest evaluation score.
- `T = 50` yields a near-uniform sampling distribution (very high ESS), so its
  behavior is effectively the same as the uniform baseline — a tie at `T = 50`
  does not demonstrate that weighted sampling is doing anything.
- `T = 30` gives a meaningfully non-uniform sampling distribution while
  matching the best 200k-scale score, so it is the most informative setting to
  scale up.

Selected `T = 30, alpha = 0.9` for the 10M-step run.

## Verification Performed

- 200k-step local CPU runs completed for `T in {30, 50}`.
  Evaluation scores at 200k were below the uniform-sampling PPO baseline at
  the same budget; this is expected — 200k is a smoke-test budget and PPO on
  this task only begins converging at several million steps.
- 10M-step run launched on Adroit with `slurm/weighted_t30_10m.slurm`,
  currently pending; final eval vs. baseline will be the real comparison.

## Design Notes

- Weights are computed **once** per period at launch (from the raw period
  parquet) and kept fixed through training. This matches how the logged policy
  score is defined and keeps the vec-env construction deterministic.
- `alpha < 1` preserves a uniform floor so every advertiser is still reachable,
  avoiding collapse onto a tiny subset of periods/advertisers.
- `WeightedBiddingEnv` only overrides advertiser sampling in reset. Everything
  downstream (budget, target CPA, rollout dynamics, reward) is inherited
  unchanged from `BiddingEnv`, so plain PPO eval code works on these
  checkpoints without modification.
