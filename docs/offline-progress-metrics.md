# AuctionNet Offline Progress Metrics

This note answers a practical question:

> When tracking training progress, should we visualize raw score or reward?

Short answer:

- For final reporting, use raw score.
- For training-progress curves, use an offline held-out continuous raw score.
- Use reward only as a secondary diagnostic, not as the main metric.

The reason is simple:

- reward alone ignores the CPA constraint
- raw score is aligned with the benchmark objective
- but sparse realized conversions are noisy and realization-dependent

So the best compromise is:

- evaluate offline on held-out data
- replace sparse binary conversion reward with a continuous expected-conversion surrogate
- then apply the same CPA penalty used by the benchmark score

## 1. Recommendation

If the goal is to see whether an agent is improving during training, the best single metric is:

```text
continuous_raw_score
  = continuous_reward * min(1, cpa_target / cpa_continuous)^2
```

with:

```text
continuous_reward = expected conversions on won impressions
cpa_continuous = spend / continuous_reward
```

In AuctionNet terms, the cleanest low-variance version is to use the same continuous target the training code already uses:

- `reward_continuous`
- based on `pValue`, not on realized binary `conversionAction`

This gives you a validation curve that is:

- aligned with the benchmark objective
- much less noisy than sparse cumulative conversions
- more informative than reward alone

## 1.1 How Reward And Raw Score Are Calculated

There are three related quantities in this repo:

1. sparse reward
2. continuous reward
3. raw score

### Sparse reward

In the training-data generator, the sparse per-tick reward is the realized number of conversions in that tick:

```text
reward_t = sum(conversionAction_i)
```

over exposed impressions for that advertiser and tick.

Episode sparse reward is then:

```text
all_reward = sum_t reward_t
```

This is the business outcome people care about, but it is noisy because conversions are binary realizations.

### Continuous reward

The repo also constructs a continuous proxy:

```text
reward_continuous_t = sum(pValue_i)
```

again over exposed impressions for that advertiser and tick.

This is best interpreted as expected conversion mass rather than realized conversions.

That is why it is attractive for offline validation:

- it is much lower variance
- it still tracks value
- most offline RL training entrypoints already use it

### CPA

Once you have total cost and total reward, CPA is:

```text
cpa = all_cost / all_reward
```

For the continuous analog:

```text
cpa_continuous = all_cost / all_continuous_reward
```

### Raw score

The benchmark-style raw score in this repo is:

```text
raw_score = all_reward * min(1, cpa_target / cpa)^2
```

So:

- if `cpa <= cpa_target`, raw score equals reward
- if `cpa > cpa_target`, raw score is penalized quadratically

This is the right final scalar if you want a single number to rank policies.

### Continuous raw score

For training curves, the recommended low-variance analog is:

```text
continuous_raw_score
  = all_continuous_reward * min(1, cpa_target / cpa_continuous)^2
```

with:

```text
all_continuous_reward = sum_t reward_continuous_t
```

This is the metric I would use for checkpoint selection.

### One subtlety in the repo

`PlayerAnalysis._get_score_neurips(...)` computes the raw CPA-penalized score, but `get_return_res(...)` later divides the returned aggregate score by `20000`.

That scaling is fine for benchmark reporting, but for learning curves you should use the unscaled raw score.

## 2. Why Reward Alone Is Not Enough

Reward by itself is not the right main curve for this repo.

If you only plot reward:

- an agent can look better simply because it spends too aggressively
- an agent can increase conversions while violating CPA badly
- the curve does not reflect the actual constrained objective

This is especially problematic in AuctionNet, because the benchmark is not “maximize conversions at any cost.” It is “maximize conversions subject to CPA pressure.”

So reward is still useful, but only as a secondary diagnostic.

## 3. Why Raw Score Is Better Aligned

The repo already uses a CPA-penalized score.

In `simul_bidding_env/Tracker/PlayerAnalysis.py`, the score is:

```text
score = reward * min(1, cpa_constraint / cpa)^2
```

This is better than reward alone because:

- if CPA is within the target, score tracks reward
- if CPA exceeds the target, score penalizes the agent

This is the right direction for model selection.

## 4. Why Sparse Raw Score Is Still Not Ideal For Training Curves

The problem with plotting raw score from online cumulative conversions is variance.

The current online plots are realization-dependent because:

- conversions are sampled binary outcomes
- one good or bad realization can move the curve a lot
- cumulative conversion curves can look different even for the same policy

That makes sparse raw score a weak signal for “is the model genuinely getting better?”

So:

- sparse raw score is good for final evaluation
- sparse raw score is not the best main learning curve

## 5. What The Baselines Actually Optimize

Most baselines in this repo do not directly optimize benchmark raw score.

They optimize surrogate learning objectives.

That is not a flaw. It is the expected design for offline learning in this setting.

Intuitively:

- the real objective depends on long-horizon pacing, cost, and CPA
- the data is fixed and offline
- direct score optimization from logged trajectories is difficult and unstable
- so each baseline uses a proxy objective that is easier to optimize and still tries to produce a good bidding multiplier

In this repo, the learned policies usually output a scalar action `alpha`, and the final bid is roughly:

```text
bid = alpha * pValue
```

So the surrogate losses are really trying to answer:

> Given this pacing state, what bid multiplier should I choose?

### BC

`run/run_bc.py` normalizes `reward_continuous`, but the model itself trains by action regression.

Training objective:

- predict the logged scalar action `alpha`
- minimize MSE between predicted and logged action

Intuition in this auction setting:

- if the logged bidding policy was already reasonable, the simplest safe baseline is to imitate it
- BC does not try to infer hidden counterfactuals
- it just learns which multiplier was historically used in a similar pacing state
- that makes it a strong sanity-check baseline and a low-risk offline method

Relevant files:

- `strategy_train_env/run/run_bc.py`
- `strategy_train_env/bidding_train_env/baseline/bc/behavior_clone.py`

### IQL

`run/run_iql.py` uses normalized `reward_continuous`.

Training objective:

- Q-function regression
- expectile value regression
- advantage-weighted policy fitting

Intuition in this auction setting:

- some logged actions were better than others even in similar pacing states
- IQL tries to identify those better actions through a critic
- then it leans the actor toward the more advantageous observed actions
- that is a natural fit for offline bidding because it can improve over imitation without needing online exploration

Relevant files:

- `strategy_train_env/run/run_iql.py`
- `strategy_train_env/bidding_train_env/baseline/iql/iql.py`

### BCQ

`run/run_bcq.py` also uses normalized `reward_continuous`.

Training objective:

- VAE action reconstruction plus KL regularization
- critic update on bootstrapped targets
- actor update through the critic under a behavior-constrained action model

Intuition in this auction setting:

- pure offline critics can recommend unrealistic bid multipliers that never appeared in data
- BCQ says: only search for improvements near the support of historical behavior
- in budget-constrained bidding, that is a reasonable safeguard against extreme out-of-distribution bids and unstable pacing

Relevant files:

- `strategy_train_env/run/run_bcq.py`
- `strategy_train_env/bidding_train_env/baseline/bcq/bcq.py`

### CQL

`run/run_cql.py` also uses normalized `reward_continuous`.

Training objective:

- policy update with entropy temperature
- critic update with conservative penalty
- expectile-style critic fitting in this implementation

Intuition in this auction setting:

- in offline RL, overestimating unseen actions is dangerous
- CQL explicitly pushes the critic to be pessimistic on unsupported actions
- that conservative bias is appealing in bidding because critic overestimation can quickly turn into overspending and CPA violations

Relevant files:

- `strategy_train_env/run/run_cql.py`
- `strategy_train_env/bidding_train_env/baseline/cql/cql.py`

### TD3_BC

`run/run_td3_bc.py` also uses normalized `reward_continuous`.

Training objective:

- bootstrapped critic update
- actor objective combining Q maximization and behavior cloning

Intuition in this auction setting:

- pure Q-based improvement can exploit critic errors
- pure imitation can be too conservative
- TD3_BC mixes the two, which is a pragmatic choice for learning a better bid multiplier while staying close to the offline data distribution

Relevant files:

- `strategy_train_env/run/run_td3_bc.py`
- `strategy_train_env/bidding_train_env/baseline/td3_bc/td3_bc.py`

### Decision Transformer

DT does not optimize raw score either.

It trains by sequence modeling:

- condition on return-to-go
- predict the logged action sequence
- optimize action prediction loss

Intuition in this auction setting:

- bidding is a temporal control problem, not just a one-step regression problem
- useful behavior often looks like a pacing pattern across ticks
- DT tries to capture those trajectory-level patterns by learning which action sequences tend to appear in higher-return trajectories

Relevant files:

- `strategy_train_env/run/run_decision_transformer.py`
- `strategy_train_env/bidding_train_env/baseline/dt/dt.py`
- `strategy_train_env/bidding_train_env/baseline/dt/utils.py`

### OnlineLp

OnlineLp is not a neural offline RL baseline.

It builds a bidding lookup table from:

- `realCPA = leastWinningCost / pValue`
- cumulative cost thresholds by tick and category

Intuition in this auction setting:

- this is the operations-research view of the problem
- if you can estimate acquisition efficiency from market price and value, you can rank opportunities by their implied CPA
- then the task becomes budget allocation over time, rather than function approximation

Relevant files:

- `strategy_train_env/run/run_onlinelp.py`
- `strategy_train_env/bidding_train_env/baseline/onlineLp/onlineLp.py`

## 6. What The Data Pipeline Suggests

The training pipeline itself strongly suggests that a continuous metric is the right progress signal.

In `train_data_generator.py`, each row stores both:

- `reward`
- `reward_continuous`

where `reward_continuous` is based on continuous value rather than realized sparse conversions.

Then the main offline RL training scripts mostly choose:

```python
normalize_reward(training_data, "reward_continuous")
```

That means the training code is already favoring a lower-variance expected-value target over sparse conversions.

So if you want the validation metric to track what the algorithms are actually learning, a continuous score is the better choice.

## 7. Best Metric For Training Progress

My recommendation is to track three offline validation metrics on held-out data:

1. `continuous_raw_score`
This should be the main model-selection metric.

2. `continuous_reward`
This tells you whether the model is finding more value at all.

3. `cpa_exceedance_rate`
This tells you whether the score change is coming from better pacing versus simple overspending.

If you want only one curve, use `continuous_raw_score`.

These three metrics work well together:

- `continuous_raw_score`: main objective-aligned checkpoint metric
- `continuous_reward`: tells you whether the policy is capturing more expected value
- `cpa_exceedance_rate`: tells you whether gains are coming from real efficiency or from violating the CPA target

## 7.1 Exact Definitions Used By The Tracker

The tracker implementation in `strategy_train_env/run/offline_metric_tracker.py` computes these metrics group by group, then averages them across held-out groups.

Let:

- `g` index a held-out validation group
- `t` index the tick
- `i` index the PV within a tick
- `x_{g,t,i} in {0,1}` be the final win indicator after the offline replay and budget-adjustment logic
- `p_{g,t,i}` be `pValue`
- `m_{g,t,i}` be `leastWinningCost`
- `C_g` be the CPA constraint for group `g`
- `G` be the set of evaluated validation groups

Then the per-group spend is:

```text
S_g = sum_t sum_i m_{g,t,i} x_{g,t,i}
```

The per-group continuous reward is:

```text
R_g^cont = sum_t sum_i p_{g,t,i} x_{g,t,i}
```

The per-group continuous CPA is:

```text
CPA_g^cont =
  S_g / (R_g^cont + 1e-10),   if R_g^cont > 0
  +inf,                       if R_g^cont = 0 and S_g > 0
  0,                          if R_g^cont = 0 and S_g = 0
```

The per-group CPA penalty is:

```text
penalty_g =
  1,                                   if CPA_g^cont <= C_g
  (C_g / (CPA_g^cont + 1e-10))^2,      if CPA_g^cont > C_g and finite
  0,                                   if CPA_g^cont = +inf
```

So the per-group `continuous_raw_score` is:

```text
continuous_raw_score_g = penalty_g * R_g^cont
```

The per-group `continuous_reward` is simply:

```text
continuous_reward_g = R_g^cont
```

The per-group `cpa_exceedance_rate` is:

```text
cpa_exceedance_rate_g =
  (CPA_g^cont - C_g) / (C_g + 1e-10),   if CPA_g^cont is finite
  +inf,                                 otherwise
```

Finally, the tracker reports the mean across validation groups:

```text
continuous_raw_score = (1 / |G|) sum_{g in G} continuous_raw_score_g
continuous_reward = (1 / |G|) sum_{g in G} continuous_reward_g
cpa_exceedance_rate = (1 / |G|) sum_{g in G} cpa_exceedance_rate_g
```

So the exact interpretation is:

- `continuous_reward` measures average expected conversion mass on won impressions
- `continuous_raw_score` applies the same quadratic CPA penalty used for score-style evaluation
- `cpa_exceedance_rate` tells you how far above or below the CPA target the checkpoint sits on average

## 7.2 Other Useful Progress Metrics

There are other useful metrics for visualizing training progress, but I would treat them as complementary metrics rather than replacements for `continuous_raw_score`.

### Normalized continuous score

```text
normalized_continuous_score
  = continuous_raw_score / baseline_continuous_raw_score
```

Why it is useful:

- easier to compare runs and algorithms
- more intuitive than absolute score if scales differ across validation sets
- makes progress easier to read when you care about relative gain over a baseline

### Value density

```text
value_density = continuous_reward / spend
```

This is effectively an efficiency metric.

Why it is useful:

- tells you whether the policy is buying better traffic, not just more traffic
- helps separate “better selection” from “more aggressive spending”

### Budget pacing error

This measures how far realized cumulative spend is from a target pacing curve.

Why it is useful:

- many AuctionNet failures are pacing failures
- an agent can have decent reward but poor pacing discipline
- this is especially informative for methods that learn a scalar bid multiplier over time

### Held-out action error

For imitation-heavy methods such as BC or DT, you can also track:

```text
action_error = MSE(predicted_alpha, logged_alpha)
```

on held-out data.

Why it is useful:

- useful for debugging whether the model is even fitting the action mapping
- not a good final objective metric
- should be treated as a model-fit metric, not a performance metric

### Win rate against a baseline across validation periods

Instead of only averaging scores, compute:

```text
period_win_rate = fraction of held-out periods where model score > baseline score
```

Why it is useful:

- robust to outlier periods
- shows whether gains are broad and consistent
- often more trustworthy than mean alone when validation periods are heterogeneous

### Median continuous raw score

Why it is useful:

- more robust than mean score if some periods are unusually easy or hard
- a good companion to mean score in heavy-tailed settings

### Area under the learning curve

If you compare algorithms rather than checkpoints, compute the area under:

```text
continuous_raw_score vs training_step
```

Why it is useful:

- rewards both final quality and speed of improvement
- better than only comparing the last checkpoint

### Time to threshold

Measure the first training step when a run exceeds a target validation score.

Why it is useful:

- gives a simple sample-efficiency summary
- easy to compare across seeds and algorithms

### Recommended usage

If you want a practical multi-metric view, I would use:

1. main curve: `normalized_continuous_score`
2. diagnostic: `continuous_reward`
3. diagnostic: `cpa_exceedance_rate`
4. diagnostic: `budget_pacing_error`

If you want one additional robust summary across held-out periods, add:

5. `period_win_rate`

So yes, there are better metrics for certain failure modes, but not really a single universally better replacement. The best default remains `continuous_raw_score`, ideally normalized to a baseline for visualization.

## 8. What To Use For Final Reporting

For final reporting, use the benchmark-aligned sparse score as well.

That means:

- report the raw CPA-penalized score
- report reward
- report CPA exceedance

But for checkpoint-by-checkpoint training progress, the main curve should still be the continuous version.

So the split is:

- training curve: continuous raw score
- final benchmark table: sparse raw score plus diagnostics

## 8.1 Where To Change The Training Code For Checkpoint Metrics

If you want to visualize these metrics during training, the right implementation is:

1. save checkpoints during training
2. run held-out offline validation at each checkpoint
3. append one row per checkpoint to `training_curve.csv`
4. plot that CSV after training

### Where to add the checkpoint hook

For the iterative baselines, the hook belongs in the main training loop:

- `strategy_train_env/run/run_iql.py`
  - add the hook inside `train_model_steps(...)`
- `strategy_train_env/run/run_bc.py`
  - add it inside the `for i in range(step_num):` loop in `train_model()`
- `strategy_train_env/run/run_bcq.py`
  - add it inside `train_model_steps(...)`
- `strategy_train_env/run/run_cql.py`
  - add it inside `train_model_steps(...)`
- `strategy_train_env/run/run_td3_bc.py`
  - add it inside `train_model_steps(...)`
- `strategy_train_env/run/run_decision_transformer.py`
  - add it inside the dataloader loop in `train_model()`

`OnlineLp` is different:

- it is not trained through iterative gradient updates
- so it does not naturally produce a step-by-step learning curve
- for that method, a final offline evaluation is usually enough

### What the checkpoint hook should do

Every `eval_interval` steps:

1. save a checkpoint into `saved_model/<Algo>/checkpoints/`
2. run offline validation on a held-out file or period
3. compute:
   - `continuous_raw_score`
   - `continuous_reward`
   - `cpa_exceedance_rate`
   - optionally `budget_consumer_ratio`
4. append the result to `saved_model/<Algo>/training_curve.csv`

### Where the validation logic should live

The current offline evaluation code in `strategy_train_env/run/run_evaluate.py` is script-shaped and always builds `PlayerBiddingStrategy`.

For checkpoint curves, the cleanest refactor is to add a reusable function in:

- `strategy_train_env/run/run_evaluate.py`
- or a new helper such as `strategy_train_env/run/offline_validation.py`

That function should accept:

- a strategy object or checkpoint path
- a validation data file path
- a flag for sparse versus continuous reward

and return a metrics dictionary instead of only printing logs.

### What should be kept fixed during validation

To make the learning curve meaningful, keep these fixed across checkpoints:

- validation file or validation periods
- normalization method
- metric definitions
- random seed if you still use any stochastic components

### Suggested artifact layout

```text
saved_model/<Algo>/
├── latest model files
├── checkpoints/
│   ├── step_00500.*
│   ├── step_01000.*
│   └── ...
└── training_curve.csv
```

This makes it easy to reload the best checkpoint later.

## 9. Why Offline Evaluation Is Good Enough For This

For progress tracking, offline validation is a good choice.

You do not need full online evaluation at every checkpoint because:

- it is expensive
- it is noisy
- it is slower to iterate on

A fixed held-out offline set gives:

- faster feedback
- lower variance
- better comparability across checkpoints

This is exactly what you want for a learning curve.

## 10. What I Would Avoid

I would avoid using these as the main progress curve:

1. Training loss only
Loss is useful for debugging, but it is not the real objective.

2. Reward only
Reward ignores CPA and can reward bad pacing.

3. Online cumulative conversions
This is too realization-dependent to be the main training-improvement curve.

## 11. Practical Validation Protocol

If you want a reliable progress plot, use this protocol:

1. Split training periods and held-out validation periods.
Do not evaluate progress on the same data used for fitting.

2. Save checkpoints every fixed number of training steps.
For example every 500 or 1000 updates.

3. For each checkpoint, run offline evaluation on the held-out split.

4. Compute:

- continuous raw score
- continuous reward
- CPA exceedance
- budget use

5. Plot score versus training step.

If needed, also plot reward and CPA exceedance as secondary curves.

## 12. Simple Decision Rule

If you want one sentence:

> Use held-out offline continuous raw score as the main learning-curve metric, and use sparse raw score only for final reporting.

That is the best tradeoff between:

- objective alignment
- stability
- interpretability

## 13. Takeaway

So, between raw score and reward:

- raw score is the better objective-aligned metric
- but for training curves, use a continuous low-variance raw score instead of sparse realized reward

That gives you a progress signal that is much more trustworthy than cumulative online conversions.
