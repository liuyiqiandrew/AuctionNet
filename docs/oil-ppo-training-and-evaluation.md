# OIL Repo: PPO Training and Evaluation for Ad Auction Bidding

This document analyzes how PPO is trained and evaluated in the OIL repo (`/scratch/gpfs/CHIJ/ruirong/rl_general/proj/oil/`), based on `online/main_train_ppo.py` and `online/main_eval.py`.

## 1. Overview

OIL uses Stable Baselines 3 (SB3) PPO — extended with optional behavioral cloning — to train an ad bidding agent. The agent interacts with a Gymnasium environment (`BiddingEnv`) that replays historical auction data from parquet files. At each tick, the agent outputs a bid multiplier; the environment simulates the auction against logged competitor bids and returns a reward based on conversions won.

Key distinguishing features vs. AuctionNet's offline RL approach:
- **Online RL**: The agent collects rollouts by interacting with the environment, not from a fixed dataset.
- **Parquet-based replay environment**: Instead of a full multi-agent simulator, the environment replays logged competitor bids from parquet files and checks whether the agent's bid would have won each slot.
- **Oracle-guided training**: An optional behavioral cloning loss steers the policy toward a computed oracle action.
- **Flexible observation/action configs**: Observation dimensionality (16, 60, 145 keys) and action structure (1-dim or 3-dim) are configurable via JSON files.

## 2. Environment: BiddingEnv

**Source**: `online/envs/bidding_env.py`

### 2.1 Data Loading

Each environment instance loads two parquet files for a given delivery period:
- `period-X_pvalues.parquet`: Per-timestep arrays of `pValue` and `pValueSigma` for all ad opportunities
- `period-X_bids.parquet`: Per-timestep arrays of the top-3 competitor bids, their costs, exposure flags, and `leastWinningCost`

At episode start, the environment samples:
- A random `advertiser` (from the 48 agents) and its `budget`, `target_cpa`, `category`
- A random `period` (delivery period index)

The episode then replays that advertiser's 48-tick sequence from the parquet data.

### 2.2 Episode Reset

On `reset()`:
- A budget is sampled uniformly from `[budget_min, budget_max]`
- A target CPA is sampled uniformly from `[target_cpa_min, target_cpa_max]`
- An advertiser and period are selected (random or specified)
- `remaining_budget = budget`
- All history accumulators are zeroed

### 2.3 Observation Space

Observations are configurable via JSON config files. The two main configurations:

**obs_16_keys** (16-dim, similar to AuctionNet's state):
- `time_left`, `budget_left`
- Historical aggregates: `historical_bid_mean`, `last_three_bid_mean`, `least_winning_cost_mean`, `pvalues_mean`, `conversion_mean`, `bid_success_mean`
- Current tick: `current_pvalues_mean`, `current_pv_num`, `last_three_pv_num`, `historical_pv_num`

**obs_60_keys** (60-dim, the default for PPO training):
- Everything in obs_16_keys, plus:
- Per-slot breakdown: `cost_mean_slot_1/2/3`, `bid_mean_slot_1/2/3`
- Percentile features: `pv_over_lwc_90_pct`, `pv_over_lwc_99_pct`, `least_winning_cost_10_pct`, `least_winning_cost_01_pct`
- Additional history windows: `last_*` and `last_three_*` variants for all metrics
- Campaign metadata: `budget`, `target_cpa`, `category`

All observations are normalized at runtime via SB3's `VecNormalize` wrapper (running mean/std).

### 2.4 Action Space

**Standard mode** (`act_1_key`, 1-dim action):
- `gym.spaces.Box(low=-10, high=10, shape=(1,))`
- The scalar action `a` is transformed to a bid coefficient: `bid_coef = exp(a)`
- Final per-PV bid: `bid_i = bid_coef * target_cpa`

**Two-slopes mode** (`two_slopes_action=True`, 3-dim action):
- `gym.spaces.Box(low=-10, high=10, shape=(3,))`
- Action = `(log_y_0, x_0, slope)` defines a piecewise linear function of pValue:
  ```
  y_pred = exp(log_y_0) + slope * max(0, pvalue - x_0)
  bid_coef = 1 / y_pred
  ```
- This allows the agent to bid differently for high-value vs. low-value impressions within a single tick.

### 2.5 Step Function (Auction Simulation)

At each `step(action)`:

1. **Compute bids**: Transform the action into per-PV bids (via `compute_bid_coef()`)
2. **Load auction context**: Read competitor top-3 bids, their costs, and exposure flags from the parquet data for this tick
3. **Simulate auction**: For each PV:
   - Compare the agent's bid against the logged top-3 competitor bids
   - Determine which slot (if any) the agent would win
   - Apply second-price cost (pay the next-highest bid)
   - If `stochastic_exposure=True`: Bernoulli draw for whether the ad is shown (slot 1: 100%, slot 2: 80%, slot 3: 60%)
   - If exposed: sample conversion from `Bernoulli(N(pValue, pValueSigma))`
4. **Budget enforcement**: If total cost exceeds remaining budget, randomly drop winning bids until cost fits
5. **Update state**: Decrement `remaining_budget`, update history accumulators
6. **Compute reward**: See below
7. **Return**: `(next_obs, reward, terminated, truncated, info)`

The `exclude_self_bids` flag removes the agent's own historical bids from the competitor bid data, preventing the agent from competing against its own logged behavior.

### 2.6 Reward Function

Two reward signals, blended by configurable weights:

**Dense reward** (computed every step):
```python
cpa = cost / conversions  # at this tick
score = min(1, (target_cpa / cpa)^2) * conversions
```
This penalizes overspending (actual CPA > target) while rewarding conversions.

**Sparse reward** (computed at episode end only):
Same formula but applied to total episode cost and total episode conversions.

**Blending**:
```python
reward = dense_weight * dense_score + sparse_weight * sparse_score
```
The default configuration uses **dense_weight=1, sparse_weight=0** (pure dense reward).

### 2.7 Oracle Actions

The environment can compute oracle (expert) actions for behavioral cloning:

- **`get_oracle_action()`**: Simple oracle — computes optimal bid for each slot
- **`get_oracle_upgrade_action()`**: Sophisticated oracle — sorts all (impression, slot) pairs by pValue/cost ratio, greedily selects the best set subject to budget and CPA constraints
- These oracle actions are collected during rollouts when `imitation_coef > 0`

## 3. Training Pipeline

**Source**: `online/main_train_ppo.py`

### 3.1 Architecture

```
┌──────────────────────────────────────────────────┐
│  main_train_ppo.py                               │
│                                                   │
│  1. Parse args (budget range, CPA range, obs/act │
│     type, reward weights, network arch, etc.)     │
│  2. Load obs_keys and act_keys from JSON configs  │
│  3. Build config_list (one per parallel env)      │
│  4. Create SubprocVecEnv + VecNormalize           │
│  5. Initialize SingleEnvTrainer with BCPPO        │
│  6. trainer.train() → SB3 PPO learn loop          │
│  7. trainer.save()                                │
└──────────────────────────────────────────────────┘
```

### 3.2 Parallel Environments

The script creates `num_envs` parallel environments (default 20) via SB3's `SubprocVecEnv`. Each environment loads a different delivery period (periods 7 through 7+num_envs). This provides diverse training data as each period has different market dynamics.

Each environment is wrapped with:
- `OracleMonitor` (extends SB3's `Monitor`): Logs episode stats and provides oracle action access
- `VecNormalize`: Normalizes observations (running mean/std) and optionally rewards

### 3.3 Algorithm: BCPPO

The training uses `BCPPO` (Behavioral Cloning PPO), a custom extension of SB3's PPO:

**Standard PPO loss**:
```
L_policy = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
L_value  = MSE(V(s), R_t)
L_entropy = -H(pi(·|s))
```

**Added behavioral cloning loss**:
```
L_imitation = MSE(a_predicted, a_oracle)
```

**Combined loss**:
```
L_total = pg_coef * L_policy + vf_coef * L_value + ent_coef * L_entropy + imitation_coef * L_imitation
```

When `imitation_coef=0` (the default in most experiments), this reduces to standard PPO.

During rollout collection, the BCPPO algorithm also queries each environment for the oracle action via `env.env_method("get_oracle_action")` and stores it in an `OracleRolloutBuffer` alongside the standard PPO data.

### 3.4 Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 (512 in practice) | Mini-batch size for PPO updates |
| `n_steps` | 128 | Rollout horizon per environment |
| `learning_rate` | 2e-5 (linear decay) | `lr(progress) = progress * base_lr` |
| `ent_coef` | 3e-6 | Entropy bonus coefficient |
| `vf_coef` | 0.5 | Value function loss weight |
| `clip_range` | 0.3 | PPO clipping parameter |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.9 | GAE lambda |
| `max_grad_norm` | 0.7 | Gradient clipping |
| `n_epochs` | 10 | SGD epochs per rollout |
| `net_arch` | [256, 256, 256] | Shared by policy and value networks |
| `activation_fn` | ReLU | Network activation |
| `log_std_init` | 0.0 | Initial exploration noise |

### 3.5 Logging and Checkpointing

- **WandB**: Syncs TensorBoard metrics automatically
- **TensorBoard**: Logs per-rollout metrics (conversions, cost, cpa, score, action, bid)
- **Checkpoints**: Saved every `save_every` steps (default 10k). Each checkpoint includes:
  - Model weights (`.zip`)
  - VecNormalize statistics (`.pkl`)
- **Config**: Saves `args.json` and `env_config.json` to the output directory

### 3.6 Training Data Cycle

Each PPO rollout cycle works as follows:
```
For each of n_steps=128 steps across num_envs=20 environments:
    1. Observe state from all 20 envs
    2. Policy predicts actions (+ oracle actions collected if imitation_coef > 0)
    3. Environments step: simulate auctions, return rewards
    4. Store (s, a, r, s', done, oracle_a) in rollout buffer

Total transitions per rollout = 128 * 20 = 2,560

Then perform n_epochs=10 SGD passes over the buffer:
    - Shuffle and split into mini-batches of size 512
    - Compute PPO loss (+ optional BC loss)
    - Update policy and value networks
```

## 4. Evaluation Pipeline

**Source**: `online/main_eval.py`

### 4.1 Checkpoint Selection

If no specific checkpoint is given:
1. Load TensorBoard logs from the training run
2. Extract the `score` time series
3. Apply Savitzky-Golay smoothing filter
4. Select the checkpoint closest to the peak smoothed score

Alternatively, `--all_checkpoints` evaluates all checkpoints and reports the best.

### 4.2 Evaluation Loop

For each checkpoint:
```
For each of num_episodes=100 episodes:
    1. Reset environment (sample advertiser, budget, CPA)
    2. Load VecNormalize (training=False, no stat updates)
    3. While not done (up to 48 steps):
        a. Normalize observation using saved running stats
        b. model.predict(norm_obs, deterministic=True/False)
        c. env.step(action)
        d. Accumulate episode reward
    4. Record episode metrics
```

### 4.3 Comparison Baselines

The evaluation script can simultaneously run three comparison strategies on the same episodes:

| Baseline | Description |
|----------|-------------|
| **Baseline** | `get_baseline_action()` — a simple budget-pacing strategy |
| **Topline (Oracle)** | `get_oracle_action()` — optimal slot-based bidding with full information |
| **Flex Topline** | `get_oracle_upgrade_action()` — sophisticated oracle with piecewise-linear bid function |

All baselines are reset with the same budget, CPA, advertiser, and period as the agent, ensuring a fair comparison.

### 4.4 Evaluation Metrics

Per-episode metrics:
- **score**: The reward function value (main metric for checkpoint selection)
- **conversions**: Total conversions achieved
- **cost_over_budget**: `total_cost / budget` (should be <= 1)
- **target_cpa_over_cpa**: `target_cpa / actual_cpa` (should be >= 1 for CPA compliance)

Aggregated across episodes:
- Mean and SEM (standard error of mean) for all metrics
- Best checkpoint tracked across all evaluated checkpoints

### 4.5 Default Evaluation Data

Training uses periods 7 through 7+num_envs. Evaluation defaults to **period-27** (an unseen test period), ensuring the agent is tested on out-of-distribution market conditions.

### 4.6 Example Evaluation Command

```bash
python online/main_eval.py \
    --experiment_path=output/training/ongoing/006_ppo_seed_0_sparse_dataset_dense_reward_two_slopes \
    --checkpoint=5250000 \
    --num_episodes=100 \
    --deterministic
```

## 5. Key Design Decisions

### 5.1 Dense vs. Sparse Reward

The experiments explored both:
- **Dense reward** (`dense_weight=1, sparse_weight=0`): Score computed per tick. Provides frequent learning signal.
- **Sparse reward** (`dense_weight=0, sparse_weight=1`): Score computed only at episode end. Harder to learn from, but directly optimizes the evaluation metric.

Most successful runs use dense reward.

### 5.2 One-Slope vs. Two-Slopes Action

- **One-slope** (1-dim action): A single bid multiplier applied uniformly to all PVs. Simple but cannot differentiate between high/low-value impressions within a tick.
- **Two-slopes** (3-dim action): A piecewise-linear bid function of pValue, allowing the agent to bid more aggressively for high-value impressions. Produces better results in practice.

### 5.3 Oracle Upgrade

When `oracle_upgrade=True`, the environment computes a sophisticated oracle action at each step. This can be used:
- As a behavioral cloning target (when `imitation_coef > 0`)
- As a comparison topline during evaluation
- As a warmstart signal for the policy

### 5.4 Self-Bid Exclusion

When `exclude_self_bids=True`, the environment removes the agent's own historical bids from the competitor data. This prevents the unrealistic scenario where the agent competes against its own past behavior and avoids a feedback loop.

## 6. Comparison with AuctionNet's RL Pipeline

| Aspect | AuctionNet | OIL |
|--------|------------|-----|
| **RL paradigm** | Offline (fixed dataset) | Online (environment interaction) |
| **Environment** | No environment; uses pre-generated `(s,a,r,s')` tuples | Gymnasium env replaying parquet data |
| **Competitor modeling** | Full 48-agent simulator or logged data | Logged competitor bids from parquet |
| **Observation** | Fixed 16-dim state | Configurable 16/60/145-dim |
| **Action** | Scalar alpha (bid multiplier) | 1-dim or 3-dim (piecewise-linear) |
| **Reward** | Conversions (discrete) | Score function with CPA penalty |
| **Algorithm** | IQL, BCQ, CQL, TD3+BC, BC, DT | PPO, BCPPO (on-policy) |
| **Normalization** | Manual (per-dim in training scripts) | SB3 VecNormalize (running stats) |
| **Training data** | Aggregated per-tick CSV | Raw per-PV parquet |
| **Oracle guidance** | None | Optional BC loss with oracle actions |
| **Parallelism** | Single process | 20 parallel SubprocVecEnv |

## 7. File Reference

| File | Purpose |
|------|---------|
| `online/main_train_ppo.py` | Training entry point — args, env setup, trainer invocation |
| `online/main_eval.py` | Evaluation entry point — checkpoint selection, rollout collection, metric reporting |
| `online/envs/bidding_env.py` | Gymnasium environment — auction simulation, state/reward/transition |
| `online/envs/environment_factory.py` | Factory for creating BiddingEnv instances |
| `online/train/trainer.py` | `SingleEnvTrainer` — wraps SB3 model init/train/save |
| `online/algos/ppo.py` | `BCPPO` — PPO extended with behavioral cloning loss |
| `online/algos/buffers.py` | `OracleRolloutBuffer` — stores expert actions alongside PPO data |
| `online/callbacks/custom_callbacks.py` | Checkpoint saving with VecNormalize |
| `online/metrics/custom_callbacks.py` | TensorBoard metric logging |
| `online/helpers.py` | Checkpoint loading, TensorBoard parsing, best-checkpoint selection |
| `online/policies/actor.py` | Custom action distributions (time-aware Gaussian) |
| `definitions.py` | Root paths, algorithm registry, default configs |
| `data/obs_configs/obs_*.json` | Observation key configurations |
| `data/act_configs/act_*.json` | Action key configurations |
