# AuctionNet Basic IQL Baseline

This note explains the original non-temporal IQL baseline in this repo.

The goal of this document is twofold:

- explain how the files interact in the implementation
- explain the math well enough that a reader can understand what the model is optimizing

This is the baseline IQL path, not the GRU temporal extension.

## 1. What This Baseline Is

The basic IQL baseline is an offline reinforcement learning agent that learns a scalar bid multiplier from logged AuctionNet data.

At a high level, the agent learns:

```text
state s_t -> action alpha_t
```

and then converts that multiplier into actual per-opportunity bids by:

```text
bid_i = alpha_t * pValue_i
```

So the model does not predict one bid per impression directly.

Instead, at each tick it predicts a single pacing multiplier `alpha_t`, and the auction-facing bid vector is formed by scaling the current `pValue` vector.

## 2. Files Involved

The basic IQL path is spread across a few files.

### Data generation

- `strategy_train_env/bidding_train_env/train_data_generator/train_data_generator.py`

This file turns raw per-impression csv data into per-tick reinforcement learning transitions:

- `state`
- `action`
- `reward`
- `reward_continuous`
- `done`
- `next_state`

### Training runner

- `strategy_train_env/run/run_iql.py`

This is the orchestration layer. It:

- loads the cached RL dataframe
- normalizes state features
- normalizes continuous reward
- fills the replay buffer
- creates the `IQL` model
- trains it
- runs held-out offline evaluation during training
- writes checkpoints, plots, and reports

### Replay buffer

- `strategy_train_env/bidding_train_env/baseline/iql/replay_buffer.py`

This file stores and samples iid offline transitions.

### IQL model

- `strategy_train_env/bidding_train_env/baseline/iql/iql.py`

This file defines the neural networks and training losses:

- `Q`
- `V`
- `Actor`
- `IQL`

### Deployment / inference wrapper

- `strategy_train_env/bidding_train_env/strategy/iql_bidding_strategy.py`

This file loads the saved model and normalization dictionary and turns the predicted scalar `alpha` into AuctionNet bids.

### Reused evaluation utilities

The basic IQL path also reuses:

- `strategy_train_env/run/offline_metric_tracker.py`
- `strategy_train_env/bidding_train_env/baseline/iql/metrics.py`

These provide:

- held-out offline checkpoint evaluation
- metric logging
- progress plots

## 3. High-Level Training Flow

The basic IQL flow is:

```text
raw traffic csv
  -> train_data_generator.py
  -> period-*-rlData.csv
  -> run_iql.py
      -> normalize state features
      -> normalize reward_continuous
      -> ReplayBuffer
      -> IQL
      -> train_model_steps()
      -> OfflineMetricTracker.maybe_evaluate()
      -> save checkpoint / plots / report
```

So the main pipeline is:

1. raw auction logs become per-tick RL transitions
2. those transitions are sampled iid from a replay buffer
3. IQL learns from logged `(s, a, r, s', d)` tuples
4. during training, checkpoints are evaluated offline on held-out data

## 4. Training And Validation Split

The current default split in `run_iql.py` is:

- training periods: `period-7` to `period-26`
- validation period: `period-27`

This matches the PPO-style holdout setup used elsewhere in the repo.

The combined training cache is built as:

```text
strategy_train_env/data/traffic/training_data_rlData_folder/
  training_data_period-7-26-rlData.csv
```

## 5. What A Transition Looks Like

The IQL learner is trained on logged transitions:

\[
\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1}, d_t)\}
\]

where:

- \(s_t\) is the 16-dimensional pacing state
- \(a_t\) is the scalar logged bid multiplier
- \(r_t\) is the normalized reward used for training
- \(s_{t+1}\) is the next state
- \(d_t \in \{0,1\}\) is the terminal flag

In the current implementation:

- the training reward is based on `reward_continuous`
- `reward_continuous` is the sum of `pValue` on exposed impressions in the tick
- that reward is min-max normalized before training

So the IQL critic is trained on a normalized continuous-reward proxy, not on sparse conversion count directly.

## 6. How The 16-Dimensional State Is Built

The state is created in:

- `train_data_generator.py`

Each tick gets one 16-dimensional vector:

1. time left
2. budget left
3. average bid over all previous ticks
4. average bid over the last 3 ticks
5. average least winning cost over all previous ticks
6. average pValue over all previous ticks
7. average conversionAction over all previous ticks
8. average xi over all previous ticks
9. average least winning cost over the last 3 ticks
10. average pValue over the last 3 ticks
11. average conversionAction over the last 3 ticks
12. average xi over the last 3 ticks
13. current tick mean pValue
14. current tick traffic volume
15. traffic volume over the last 3 ticks
16. cumulative historical traffic volume

This means the original IQL baseline is not memoryless in the strict sense.

It does not use a learned recurrent encoder, but its state already contains hand-crafted history summaries.

## 7. Action Definition

The logged action is:

\[
a_t = \frac{\sum_i \text{bid}_{t,i}}{\sum_i pValue_{t,i}}
\]

when the denominator is positive, otherwise:

\[
a_t = 0
\]

So the action is a scalar multiplier that approximately says:

> how aggressively should this advertiser bid relative to predicted value at this tick?

At inference time, the model predicts `alpha_t`, and bids are formed by:

\[
\text{bid}_{t,i} = \alpha_t \cdot pValue_{t,i}
\]

## 8. Replay Buffer

The replay buffer is in:

- `baseline/iql/replay_buffer.py`

It stores transitions as:

- `state`
- `action`
- `reward`
- `next_state`
- `done`

and samples random mini-batches uniformly.

This is important:

- the basic IQL baseline is not sequence-based
- it does not preserve temporal order inside the batch
- it learns from iid transition samples

## 9. Neural Networks

The networks are defined in:

- `strategy_train_env/bidding_train_env/baseline/iql/iql.py`

There are three main learned modules.

## 9.1 `Q`

Purpose:

- estimate \(Q(s, a)\)

Inputs:

- `obs`
  shape: `(batch_size, state_dim)`
- `acts`
  shape: `(batch_size, action_dim)`

Internal structure:

1. linear layer on state
2. linear layer on action
3. concatenate the two embeddings
4. pass through an MLP

Output:

- scalar Q-value
  shape: `(batch_size, 1)`

There are two copies:

- `critic1`
- `critic2`

and two target copies:

- `critic1_target`
- `critic2_target`

The two-critic structure is used to reduce overestimation and stabilize the actor/value targets.

## 9.2 `V`

Purpose:

- estimate \(V(s)\)

Input:

- `obs`
  shape: `(batch_size, state_dim)`

Internal structure:

- standard feed-forward MLP

Output:

- scalar state value
  shape: `(batch_size, 1)`

## 9.3 `Actor`

Purpose:

- model a Gaussian policy \(\pi(a \mid s)\)

Input:

- `obs`
  shape: `(batch_size, state_dim)`

Outputs from `forward(...)`:

- `mu`
  shape: `(batch_size, action_dim)`
- `log_std`
  shape: `(batch_size, action_dim)`

The actor defines a Normal distribution:

\[
\pi(a \mid s) = \mathcal{N}(\mu(s), \sigma(s))
\]

where:

\[
\sigma(s) = \exp(\log \sigma(s))
\]

The code provides:

- `evaluate(...)`
  returns a sampled action and the distribution
- `get_action(...)`
  returns a sampled action
- `get_det_action(...)`
  returns the deterministic mean action

At inference time, the baseline uses the deterministic mean action and clamps it to be non-negative.

## 10. The `IQL` Class

The `IQL` class contains:

- `value_net`
- `critic1`
- `critic2`
- `critic1_target`
- `critic2_target`
- `actors`

and the optimizers for each component.

Its central public training method is:

- `step(states, actions, rewards, next_states, dones)`

Input shapes:

- `states`
  `(batch_size, state_dim)`
- `actions`
  `(batch_size, 1)`
- `rewards`
  `(batch_size, 1)`
- `next_states`
  `(batch_size, state_dim)`
- `dones`
  `(batch_size, 1)`

Output:

- `(critic1_loss, value_loss, actor_loss)`

returned as NumPy scalars for logging.

## 11. What IQL Is Doing Mathematically

Basic IQL learns three objects:

- two Q-functions \(Q_1, Q_2\)
- one value function \(V\)
- one policy \(\pi\)

The high-level intuition is:

1. learn which logged actions looked good according to the critic
2. learn a value baseline for each state
3. train the policy to imitate better-than-average logged actions

That is why IQL is often described as:

- offline RL
- but without explicit out-of-distribution action maximization

The actor is not directly maximizing over arbitrary actions.

Instead, it learns from logged actions, weighted by how advantageous those actions appear.

## 12. Critic Loss

The Bellman target implemented in this repo is:

\[
y_t = r_t + \gamma (1 - d_t) V(s_{t+1})
\]

Then each critic minimizes:

\[
\mathcal{L}_{Q_i}
=
\mathbb{E}_{(s,a,r,s',d)\sim\mathcal{D}}
\left[
\left(Q_i(s,a) - y_t\right)^2
\right]
\]

In the code this is in:

- `calc_q_loss(...)`

with:

- `q1 = critic1(states, actions)`
- `q2 = critic2(states, actions)`
- `q_target = rewards + gamma * (1 - dones) * V(next_states)`

and then squared error to the target.

## 13. Value Loss

The value network is trained by expectile regression against the lower of the two target critics:

\[
u(s,a) = \min(Q_1^{target}(s,a), Q_2^{target}(s,a)) - V(s)
\]

The expectile loss is:

\[
\mathcal{L}_V
=
\mathbb{E}_{(s,a)\sim\mathcal{D}}
\left[
\rho_\tau(u)
\right]
\]

where:

\[
\rho_\tau(u)
=
|\tau - \mathbf{1}[u < 0]| \cdot u^2
\]

In the code, this is implemented through:

- `l2_loss(diff, expectile)`

with:

\[
w(u) =
\begin{cases}
\tau & \text{if } u > 0 \\
1-\tau & \text{if } u \le 0
\end{cases}
\]

and:

\[
\rho_\tau(u) = w(u)u^2
\]

Interpretation:

- if the critic says the logged action is better than the current value estimate, the value is pushed upward more strongly
- if it is worse, the value is pushed downward more conservatively

This makes \(V(s)\) behave like an asymmetric lower-ish baseline for logged-action values.

## 14. Actor Loss

The actor loss is the key idea in IQL.

The policy does not maximize Q over arbitrary actions directly.

Instead, it performs weighted behavior cloning over logged actions.

The conceptual IQL actor objective is:

\[
\mathcal{L}_\pi
=
-\mathbb{E}_{(s,a)\sim\mathcal{D}}
\left[
w(s,a)\log \pi(a \mid s)
\right]
\]

where the weight depends on the advantage:

\[
A(s,a) = Q(s,a) - V(s)
\]

In many IQL expositions, the weight is written as:

\[
w(s,a) = \exp(\beta A(s,a))
\]

### Important repo-specific note

The code in this repo uses a slightly different form.

It computes:

\[
\tilde{w}(s,a)
=
\min\left(\exp(\min(Q_1,Q_2)-V)\cdot \text{temperature}, 100\right)
\]

That means:

- the exponent is `minQ - V`
- then the result is multiplied by `temperature`
- then clipped at `100`

So in this implementation, `temperature` is not inside the exponent.

This is an important detail if someone is comparing the code to a paper implementation.

The actual actor loss in this repo is therefore:

\[
\mathcal{L}_\pi
=
-\mathbb{E}_{(s,a)\sim\mathcal{D}}
\left[
\tilde{w}(s,a)\log \pi(a \mid s)
\right]
\]

Interpretation:

- if a logged action looks better than the state value baseline, it gets more weight
- if it looks mediocre or bad, it gets less weight
- the actor remains tied to logged data instead of chasing unsupported actions

## 15. Target Networks

After critic updates, the target critics are soft-updated:

\[
\theta^{target}
\leftarrow
(1-\tau)\theta^{target} + \tau\theta
\]

This happens in:

- `update_target(...)`

for both critic targets.

These target networks stabilize:

- value regression
- actor weighting
- Bellman targets

## 16. Training Step Order

Inside `IQL.step(...)`, the order is:

1. update value network
2. update actor
3. update both critics
4. soft-update target critics

So the training loop is:

```text
sample batch
  -> update V
  -> update actor
  -> update Q1 and Q2
  -> update target critics
```

This is the actual implementation order in the repo.

## 17. Inference Path

The deployment path is in:

- `strategy/iql_bidding_strategy.py`

Its flow is:

```text
current auction history
  -> rebuild 16-dim state
  -> normalize selected features
  -> model(state)
  -> alpha
  -> bids = alpha * pValues
```

At inference time:

- the model uses the deterministic actor mean
- the returned action is clamped to be non-negative

So the deployed behavior is deterministic by default.

## 18. What `run_iql.py` Adds On Top Of The Model

`run_iql.py` is not just a thin trainer.

It adds several practical training-system responsibilities.

### `ensure_train_data_cache(...)`

Responsibility:

- combine `period-7` to `period-26` RL caches into one training csv if needed

### `train_iql_model(...)`

Responsibility:

- full training orchestration

It does:

1. load cached RL data
2. parse serialized tuples using `safe_literal_eval`
3. normalize state features
4. normalize `reward_continuous`
5. populate the replay buffer
6. instantiate `IQL`
7. train for `step_num` updates
8. save model and report

### `IqlValidationStrategy`

Responsibility:

- make the in-memory IQL model look like a bidding strategy for the offline evaluator

This class rebuilds the same 16-dim state from evaluation histories and feeds it into the model.

### `build_iql_metric_tracker(...)`

Responsibility:

- connect the training loop to held-out offline evaluation

The tracker periodically evaluates checkpoints on `period-27` and writes:

- `training_curve.csv`
- `best_metrics.json`
- `metrics.json`
- `training_curves.png`

## 19. What Gets Saved

The default output directory is:

```text
strategy_train_env/saved_model/IQLtest/
```

Typical artifacts are:

```text
IQLtest/
├── iql_model.pth
├── normalize_dict.pkl
├── training_curves.png
├── metrics.json
├── training_curve.csv
├── best_metrics.json
├── training_report.md
└── checkpoints/
    ├── step_01000/
    │   ├── iql_model.pth
    │   └── normalize_dict.pkl
    └── ...
```

## 20. What The Baseline Is Actually Optimizing

This is a useful summary because people often conflate:

- business score
- training reward
- neural-network loss

They are not the same.

### Training reward

The IQL trainer uses normalized `reward_continuous`, which is based on exposed `pValue`.

So the per-transition reward proxy is closer to expected conversion mass than to realized sparse conversions.

### Actor / critic objective

The neural losses are:

- Bellman regression for critics
- expectile regression for value
- advantage-weighted behavior cloning for actor

### Validation score

Checkpoint evaluation during training uses the offline metric tracker and reports:

- sparse score
- continuous reward
- CPA-related diagnostics

So:

- training objective is not directly the benchmark score
- validation score is a held-out proxy for model quality

This separation is normal in offline RL.

## 21. What This Baseline Does Not Do

The basic IQL baseline does not:

- use a recurrent model
- use a transformer
- preserve temporal order inside minibatches
- optimize the benchmark score directly
- learn per-impression bids directly

It is better understood as:

- a logged-data offline RL baseline
- over a hand-crafted per-tick pacing state
- that learns a scalar bid multiplier

## 22. Mental Model For Readers

If you want one sentence to remember the design:

> Basic IQL in this repo learns a scalar bid multiplier from offline logged transitions by fitting twin critics, an expectile value function, and an advantage-weighted actor on a hand-crafted 16-dimensional pacing state.

And if you want the shortest file-level summary:

```text
train_data_generator.py
  builds per-tick RL transitions

run_iql.py
  orchestrates training, evaluation, saving, and reports

replay_buffer.py
  samples iid offline transitions

iql.py
  defines Q, V, Actor, and the IQL losses

iql_bidding_strategy.py
  loads the saved model and turns alpha into bids
```

That is the basic IQL baseline.
