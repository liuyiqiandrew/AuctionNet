# Online PPO Baseline — Implementation Reference

This document describes the PPO baseline implemented in
`strategy_train_env/bidding_train_env/online/`. It is a verbatim reading of the
code, covering training objectives, data processing, and the hyperparameters
that ship as defaults.

Entry points (run from `AuctionNet/strategy_train_env/`):

- `bidding_train_env/online/prepare_data.py` — raw-parquet → per-period RL data
- `bidding_train_env/online/main_train_ppo.py` — train PPO
- `bidding_train_env/online/main_eval_ppo.py` — evaluate a trained checkpoint

Algorithm: `stable_baselines3.PPO` with `MlpPolicy` (Gaussian head over a
continuous `Box` action), wrapped in `VecNormalize` for online observation /
reward whitening. The trainer is registered through a tiny algorithm registry
(`ALGO_CLASS_DICT = {"ppo": PPO}` in `definitions.py`); custom variants drop in
by adding to that dict and passing `--algo`.

---

## 1. Data processing

### 1.1 Source layout

Raw inputs live at `strategy_train_env/data/traffic/period-{N}.parquet`, one
parquet per delivery period. Each row is a single ad opportunity for one
advertiser at one impression `pvIndex` at one `timeStepIndex`, with columns:

- `deliveryPeriodIndex`, `timeStepIndex`, `pvIndex`
- `advertiserNumber`, `advertiserCategoryIndex`
- `pValue`, `pValueSigma` — the per-opportunity conversion-probability mean
  and stdev for that advertiser
- `bid`, `cost`, `isExposed`, `xi` — bid placed in the historical auction,
  realized cost, whether the impression was exposed, and a top-3 indicator
  (`xi == 1` ↔ this row is one of the top-3 bidders for that impression)
- `budget`, `CPAConstraint` — per-(period, advertiser) campaign constraints

### 1.2 Aggregation (`data_generator.py`, `prepare_data.py`)

`prepare_data.py --first_period 7 --last_period 27` writes three parquet files
per period to `data/traffic/online_rl_data/`:

**`period-{N}_pvalues.parquet`** — one row per `(deliveryPeriodIndex,
timeStepIndex, advertiserNumber, advertiserCategoryIndex)`. The two value
columns `pValue` and `pValueSigma` are aggregated as Python lists of all
opportunities that advertiser saw at that timestep.

**`period-{N}_bids.parquet`** — one row per `(deliveryPeriodIndex,
timeStepIndex)`. Built in two groupby passes:

1. Filter `xi == 1` (top-3 rows only), then group by
   `(deliveryPeriodIndex, timeStepIndex, pvIndex)` aggregating `bid`,
   `isExposed`, `cost`, `advertiserNumber` to lists. After this pass, each row
   is one impression with the top-3 winning bidders.
2. Group again by `(deliveryPeriodIndex, timeStepIndex)`, producing one row per
   timestep whose `bid`/`cost`/`isExposed`/`advertiserNumber` columns are
   list-of-lists (`num_pv × 3`). Within each impression, the four lists are
   re-sorted in ascending order of `bid` using `np.argsort` and
   `_reorder_list_of_lists`, so column index `-1` is the highest competing
   bid and column index `0` is the lowest of the top-3 (the *least winning
   cost*).

**`period-{N}_constraints.parquet`** — one row per `(deliveryPeriodIndex,
advertiserNumber)` carrying the raw `(budget, CPAConstraint)`. Used when the
training/eval invocation selects `--bc_range default` (env reads the
advertiser's authentic constraints rather than sampling).

### 1.3 Train/eval split

`main_train_ppo.py` uses one env per period; the default
`--first_period 7 --num_envs 20` gives periods 7..26. `main_eval_ppo.py`
defaults to `--eval_period 27`. So the canonical split is **periods 7–26 train,
period 27 evaluate** (63 parquet files = 21 periods × 3 files).

### 1.4 Episode loading inside `BiddingEnv`

On construction, `BiddingEnv` (`online_env.py`) loads the full pvalues and bids
parquets for one period. `pValue` / `pValueSigma` cells are converted from
Python lists to `np.array` (`_load_pvalues_df`); `bid` / `cost` / `isExposed`
cells are stacked from list-of-lists to `np.ndarray` of shape `(num_pv, 3)`
(`_load_bids_df`). At `reset` the env caches the per-(advertiser, period)
slice of pvalues and the per-period bids slice as `episode_pvalues_df` /
`episode_bids_df`.

Episode length is fixed at `EPISODE_LENGTH = 48` ticks (`definitions.py`).
Each step pulls the row at `timeStepIndex == self.time_step` from those two
slices.

---

## 2. Environment dynamics

`BiddingEnv` is a `gymnasium.Env` registered as `BiddingEnv-v0`.

### 2.1 Spaces

- **Observation**: `Box(-inf, +inf, shape=(len(obs_keys),), float32)`. The
  observation schema is JSON-driven (`configs/obs_16_keys.json` or
  `configs/obs_60_keys.json`). See §2.6.
- **Action**: `Box(-10, +10, shape=(len(act_keys),), float32)`. The default
  schema `configs/act_1_key.json` is `["pvalue"]`, so the action is a single
  scalar in `[-10, 10]`. Larger schemas are supported but unused by the
  baseline.

### 2.2 Episode setup (`_reset_campaign_params`)

At each reset the env samples:

- `advertiser` uniformly from `pvalues_df.advertiserNumber.unique()`
- `period` uniformly from `pvalues_df.deliveryPeriodIndex.unique()` (each
  parquet contains a single period, so this is degenerate)
- `(total_budget, target_cpa)` according to `--bc_range`:
  - `dense`: `budget ~ U(400, 12000)`, `target_cpa ~ U(6, 12)`
  - `sparse`: `budget ~ U(1000, 6000)`, `target_cpa ~ U(50, 150)`
  - `default`: read raw `(budget, CPAConstraint)` from the constraints parquet
    for the sampled `(period, advertiser)`. **This is the canonical setting**
    (it is the README example and the `REWARD_SHAPING.md` recommendation).

`set_campaign(...)` is a hook used by sweep-mode evaluation to override the
sampled triple deterministically.

### 2.3 Action → bid pipeline (`step`, `_compute_bid_coef`)

For each timestep:

1. Retrieve `pvalues, sigma` arrays of length `num_pv` for this `(advertiser,
   period, timestep)`.
2. Apply the per-key transform: position `pvalues_key_pos` of the action
   vector is exponentiated. With the default `act_keys=["pvalue"]`, this means
   the scalar action `a` becomes `exp(a)`.
3. Build `bid_basis` per impression by stacking
   `{"pvalue": pvalues, "pvalue_sigma": pvalues_sigma}[k]` for each
   `k in act_keys`. With the default schema, `bid_basis = pvalues[:, None]`.
4. `bid_coef = clip(einsum("k,nk->n", action, bid_basis), 0, +∞)`. With one
   key, this collapses to `bid_coef = max(0, exp(a) * pvalues)`.
5. `advertiser_bids = bid_coef * target_cpa` (length `num_pv`).

Net effect of the default schema: **`bid = exp(action) · pValue · target_cpa`**
per impression. The action is the log-multiplier on the standard
"value × CPA" bidding formula.

### 2.4 Auction settlement (`_compute_success_exposition_cost`)

Against the cached per-impression top-3 competing bids
`top_bids ∈ ℝ^{num_pv×3}`:

- `higher = advertiser_bids[:, None] >= top_bids` — pairwise comparison
- `bid_success = higher.any(axis=1)` — won the impression if it beats at least
  one of the top-3
- `bid_position = higher.sum(axis=1) - 1`, taking values in `{0, 1, 2}` (the
  comment in code reads `0..2 with 2 = top slot`, but note the bid-list is
  pre-sorted *ascending*, so `bid_position == 2` means "outbid all three of
  the top-3 historical bidders" and `bid_position == 0` means "outbid only the
  lowest of the top-3"). This index is used both to look up exposure and cost.
- `bid_exposed[bid_success] = top_bids_exposed[bid_success, bid_position[bid_success]]`
  — exposure is replayed from the historical record at the slot the agent
  would occupy
- `bid_cost = top_bids_cost[arange, bid_position] * bid_exposed` — second-price
  charge: the cost of the slot the agent outbid, gated on actual exposure.

### 2.5 Over-cost correction (`_handle_overcost`)

If the realized step cost would exceed `remaining_budget`, the env runs a
fixed-point loop:

```
while total_cost > remaining_budget:
    over_ratio = (total_cost - remaining_budget) / total_cost
    n_drop    = ceil(num_winners * over_ratio)
    drop_idx  = np_random.choice(winners, n_drop, replace=False)
    advertiser_bids[drop_idx] = 0
    re-run _compute_success_exposition_cost
```

Effectively: at random, drop a fraction of winning impressions until the step
fits under the remaining budget. Mirrors the simulator-side budget logic in
`run/run_test.py`.

### 2.6 Conversion sampling

After settlement, conversions are computed per impression:

- `--deterministic_conversion`: `bid_conversion = pvalues * bid_exposed`
  (expectation, no noise).
- Default (stochastic): `p_sampled = clip(N(pvalues, pvalues_sigma), 0, 1)`
  followed by `bid_conversion = Bernoulli(p_sampled) * bid_exposed`. **Both
  `pValue` and `pValueSigma` matter**; deterministic mode silently ignores the
  latter.

### 2.7 Cumulatives, score, and reward

After each step the env updates:

```
total_cost          += sum(bid_cost)
total_conversions   += sum(bid_conversion)
total_cpa            = total_cost / total_conversions   (0 if no conversions)
remaining_budget    -= sum(bid_cost)
pv_num_total        += len(pvalues)
time_step           += 1
terminated           = (time_step >= 48)
```

The CPA-clipped score function is:

```
score(cost, conversions) = min(1, (target_cpa / cpa)**2) * conversions
                           where cpa = cost / conversions
                           (returns 0 if conversions <= 0 or cpa <= 0)
```

This is the AuctionNet competition score: it equals raw conversions when
`cpa <= target_cpa`, and decays quadratically once realized CPA exceeds the
target.

The per-step reward decomposition is:

```
dense_reward_t  = score(step_cost_t, step_conv_t)
sparse_reward_T = score(total_cost, total_conversions)   if t == T-1 else 0
overspend_t     = max(0, step_cost_t - target_cpa * step_conv_t)
reward_t        = dense_weight  * dense_reward_t
                + sparse_weight * sparse_reward_T
                - lambda_cpa    * overspend_t
```

with default weights `{"dense": 1.0, "sparse": 0.0}` and `lambda_cpa = 0.0`.
The `lambda_cpa > 0` branch is the optional Lagrangian CPA-overspend penalty
documented in `REWARD_SHAPING.md`; setting `lambda_cpa = 0` reproduces the
baseline exactly. With the defaults, the agent's training signal is
**purely the per-step CPA-clipped score**, with no explicit terminal bonus.

### 2.8 History tracking and observation construction

After each step, `_update_history` appends a per-step entry to `history_info`
for each key in `HISTORY_KEYS` (15 scalar aggregates over the step:
`pvalues_mean`, `least_winning_cost_mean`/`10_pct`/`01_pct`,
`cpa_exceedence_rate`, `bid_success_mean`, `bid_over_lwc_mean`,
`pv_over_lwc_{mean,90_pct,99_pct}`, `successful_bid_position_mean`,
`exposure_count`, `cost_sum`, `conversion_{mean,count}`, `pv_num`) and for
each key in `HISTORY_AND_SLOT_KEYS` (`bid_mean`, `cost_mean`,
`bid_success_count`, `exposure_mean`) which additionally are tracked split by
slot 1..3 (`slot_3` = top, `slot_1` = worst-winning). See `definitions.py` for
the exact lists.

`_get_state_dict` builds three views of every history scalar
`{last_, last_three_, historical_}{key}` (last-step value, mean of last 3,
mean over the whole episode). Together with the no-history scalars
(`time_left`, `budget_left`, `budget`, `cpa`, `category`, `total_*`, plus
`current_pvalues_{mean,90_pct,99_pct}` and `current_pv_num` from the
*upcoming* step's pvalues) plus a small alias table for legacy key names
used in `obs_60_keys`, this dict can resolve both schemas.

The two shipped observation schemas:

- `obs_16_keys.json` (default, 16-dim): `time_left`, `budget_left`,
  `historical_bid_mean`, `last_three_bid_mean`, `least_winning_cost_mean`,
  `pvalues_mean`, `conversion_mean`, `bid_success_mean`,
  `last_three_least_winning_cost_mean`, `last_three_pvalues_mean`,
  `last_three_conversion_mean`, `last_three_bid_success_mean`,
  `current_pvalues_mean`, `current_pv_num`, `last_three_pv_num`,
  `pv_num_total`. Notable: most "historical" entries here resolve through the
  alias table from the bare key (e.g. `least_winning_cost_mean` →
  `historical_least_winning_cost_mean`), and `last_three_pv_num` is the
  **sum** (not mean) of the last 3 entries for backward compat.
- `obs_60_keys.json` (60-dim): adds `budget`, `cpa`, `category`, all three
  views of the slot-split cost aggregates, percentile views of LWC and
  pv/lwc ratios, current pvalues percentiles, etc.

The final observation is the float32 vector `[state_dict[k] for k in
obs_keys]`, then passed through `VecNormalize`, then to the policy.

### 2.9 `info` dict

Per step: `dense`, `sparse`, `overspend`, `bid` (mean of advertiser bids),
`action` (`sum(bids) / sum(pvalues) / target_cpa`, ≈ the realized
multiplicative coefficient).

At terminal step, additional keys are emitted: `score` (= `sparse_reward`),
`conversions`, `cost`, `cpa`, `target_cpa`, `budget`, `avg_pvalues`,
`score_over_pvalue`, `score_over_budget`, `score_over_cpa`,
`cost_over_budget`, `target_cpa_over_cpa`, `advertiser`, `period`. These are
what `JsonRolloutCallback` averages per-rollout into `rollout_log.jsonl`.

---

## 3. Training objective

### 3.1 PPO

`stable_baselines3.PPO` uses the standard clipped surrogate objective with a
value loss and entropy bonus:

```
L(θ) = E_t [
    min( r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t )
  - vf_coef * (V_θ(s_t) - V̂_t)^2
  + ent_coef * H(π_θ(·|s_t))
]
```

with `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)` and `Â_t` the GAE
advantage. The policy distribution is a diagonal Gaussian (default for
continuous `Box` actions); its initial log-stdev is `log_std_init = 0.0`
(unit variance at initialization).

In our wiring (`main_train_ppo.py::main`):

- `policy = "MlpPolicy"` with `net_arch=dict(pi=[256,256,256], vf=[256,256,256])`,
  `activation_fn=ReLU`, `log_std_init=0.0`.
- `clip_range = 0.3`, `vf_coef = 0.5`, `ent_coef = 3e-6`,
  `max_grad_norm = 0.7`, `gamma = 0.99`, `gae_lambda = 0.9`.
- `n_steps = 128`, `batch_size = 512`, `n_epochs = 10`. With `num_envs = 20`,
  one rollout collects `128 × 20 = 2560` transitions, then PPO runs 10 epochs
  of minibatch gradient descent over those, batch size 512 → 5 minibatches per
  epoch.
- Learning rate is **linearly annealed** from `2e-5` to 0 over training:
  `learning_rate=lambda x: x * args.learning_rate`, where SB3 passes `x` as
  remaining-progress (`1.0 → 0.0`).

`VecNormalize` wraps the vectorized env with running-mean/var observation
normalization and reward normalization (SB3 defaults — `norm_obs=True`,
`norm_reward=True`, both clipped to ±10). At eval time both are switched off
for reward (`vec_env.norm_reward = False`) but obs normalization is preserved
by loading the training-time running stats from the saved `.pkl`.

### 3.2 Vectorization and rollouts

`make_vec_env` builds `num_envs` thunks, each instantiating one
`BiddingEnv` for one period and wrapping it in `Monitor(env, log_dir,
info_keywords=INFO_KEYWORDS)`. The thunks are launched into either
`SubprocVecEnv` (default) or `DummyVecEnv` (`--use_dummy_vec_env`). With
`--num_envs 20 --first_period 7`, env *i* trains exclusively on period
`7 + i`, i.e. periods 7..26. Episodes run independently in each subprocess and
auto-reset on terminal.

### 3.3 Reward shaping (optional)

`--lambda_cpa λ` adds the per-step Lagrangian penalty
`-λ · max(0, step_cost - target_cpa · step_conv)` to the reward
(`online_env.py:213-214`). The default `λ = 0` is the baseline.
`REWARD_SHAPING.md` documents a sweep over `λ ∈ {0, 0.5, 1.0, 2.0}` against the
`--bc_range default` setting; intent is to push `target_cpa_over_cpa` toward 1
(baseline runs collapsed CPA to ~5% of target → score 4.83 in sweep mode).

---

## 4. Hyperparameters (defaults)

The CLI defaults in `main_train_ppo.py::parse_args` are the canonical
hyperparameters. (Note: `definitions.py::DEFAULT_PPO_KWARGS` lists slightly
different values — `n_steps=128, batch_size=256, n_epochs=10` — but the
trainer constructs `model_cfg` directly from CLI args, so those constants are
*reference values* only, not what is loaded.)

### 4.1 PPO algorithm

| Hyperparameter | Default | Source |
|---|---|---|
| `policy` | `"MlpPolicy"` | hardcoded in `model_cfg` |
| `net_arch` (pi & vf) | `[256, 256, 256]` | `--net_arch` |
| `activation_fn` | `nn.ReLU` | hardcoded in `policy_kwargs` |
| `log_std_init` | `0.0` | `--log_std_init` |
| `learning_rate` | `2e-5`, linearly annealed to 0 | `--learning_rate`; schedule wired as `lambda x: x * lr` |
| `n_steps` (per env) | `128` | `--n_rollout_steps` |
| `batch_size` | `512` | `--batch_size` |
| `n_epochs` | `10` | `--n_epochs` |
| `gamma` | `0.99` | `--gamma` |
| `gae_lambda` | `0.9` | `--gae_lambda` |
| `clip_range` | `0.3` | `--clip_range` |
| `ent_coef` | `3e-6` | `--ent_coef` |
| `vf_coef` | `0.5` | `--vf_coef` |
| `max_grad_norm` | `0.7` | `--max_grad_norm` |
| `seed` | `0` | `--seed` |
| `device` | `cuda` | `--device` |

### 4.2 Vectorization & schedule

| Hyperparameter | Default |
|---|---|
| `num_envs` | `20` (one per period 7..26) |
| `first_period` | `7` |
| `vec env class` | `SubprocVecEnv` (`--use_dummy_vec_env` swaps to `DummyVecEnv`) |
| `total_timesteps` | `10_000_000` |
| `save_every` (env steps) | `10_000`; ckpt callback uses `save_freq = save_every // num_envs` per worker |
| `VecNormalize` | obs + reward norm with SB3 defaults; reload from `.pkl` on resume |

So one rollout = `n_steps × num_envs = 128 × 20 = 2560` transitions; one
training run = `10M / 2560 ≈ 3906` PPO updates × 10 epochs × 5 minibatches.

### 4.3 Environment

| Hyperparameter | Default |
|---|---|
| `episode_length` | `48` ticks |
| `obs_type` | `obs_16_keys` (16-dim) |
| `act_type` | `act_1_key` (1-dim, `pvalue` log-coefficient) |
| action bounds | `Box(-10, +10)` |
| `bc_range` | `default` (raw `(budget, CPAConstraint)` from constraints parquet) |
| `dense_weight` | `1.0` |
| `sparse_weight` | `0.0` |
| `lambda_cpa` | `0.0` (baseline; `>0` enables Lagrangian shaping) |
| `deterministic_conversion` | `False` (Bernoulli sampling of conversion) |

### 4.4 Logging & I/O

Each run writes to
`output/online/training/ongoing/{out_prefix}{algo}_seed_{seed}{out_suffix}/`:

- `args.json` — parsed CLI args
- `env_config.json` — first env's `kwargs` dict
- `model_config.json` — PPO `model_cfg` (note: `learning_rate` lambda
  serializes as `"<not serializable>"`; `activation_fn` similarly drops to a
  string — `main_eval_ppo_v2.py` works around this when the standard
  `PPO.load` path can't deserialize the pickle)
- `main_train_ppo.py` — snapshot of the launcher script
- `rl_model_{N}_steps.zip` + `rl_model_vecnormalize_{N}_steps.pkl` per
  checkpoint
- `final_model.zip` + `final_vecnormalize.pkl`
- `rollout_log.jsonl` — one JSON line per PPO rollout, with mean over
  `INFO_KEYWORDS` plus `ep_rew_mean`/`ep_len_mean` from
  `model.ep_info_buffer`
- `rollout_curve.png` — two-panel `ep_rew_mean` + `score` plot, refreshed at
  every checkpoint

`get_model_and_env_path` makes training **auto-resume** from the latest
checkpoint in its own log dir if interrupted; explicit `--load_path` is only
consulted when the log dir has no checkpoints.

---

## 5. Evaluation

`main_eval_ppo.py` loads the checkpoint plus its frozen `VecNormalize` stats
(`vec_env.training=False`, `vec_env.norm_reward=False`, but obs whitening
preserved) and runs deterministic policy rollouts (`model.predict(obs,
deterministic=True)`) on `--eval_period 27`:

- **`random` mode** — `n_eval_episodes` rollouts with the period-27 env's
  default sampling under the chosen `--bc_range`.
- **`sweep` mode** — iterate over every advertiser in
  `period-27_constraints.parquet`, calling `inner.set_campaign(advertiser,
  budget, target_cpa)` to override the random sample with that row's raw
  `(budget, CPAConstraint)`. The overridden raw state is pushed through
  `vec_env.normalize_obs` so the policy sees the same obs distribution it was
  trained on.
- **`both`** — runs random then sweep, writes two JSONs.

Per-episode `info` rows plus aggregate `mean/std/sem/n` over `score`,
`cost_over_budget`, `target_cpa_over_cpa`, `conversions`, `total_reward` are
written to `output/online/testing/{run_name}/results_{mode}_{ts}.json`. Sweep
score is the comparable headline metric against the AuctionNet leaderboard.

---

## 6. Quick reference — control-flow summary

1. Raw parquet (one row per ad-opportunity) →
   `prepare_data.py` → 3 parquets per period (`pvalues`, `bids`,
   `constraints`).
2. `main_train_ppo.py` builds 20 `BiddingEnv` workers (period 7..26), wraps in
   `Monitor` → `SubprocVecEnv` → `VecNormalize`, instantiates `PPO`, runs
   `learn(total_timesteps=10M)` with `JsonRolloutCallback` +
   `CustomCheckpointCallback`.
3. Per env step: sample/lookup pvalues, `bid = exp(action) · pValue ·
   target_cpa`, settle second-price auction against pre-aggregated top-3,
   randomly drop winners until budget holds, sample Bernoulli conversions
   from `clip(N(pValue, pValueSigma), 0, 1)`, emit
   `dense - λ·overspend` reward, update history, build next observation.
4. Per rollout (2560 transitions): PPO clipped-surrogate update for 10
   epochs, batch 512.
5. Eval: load ckpt + saved `VecNormalize`, deterministic rollouts on period
   27, dump per-episode `info` and aggregate stats.
