# AuctionNet Training Metric Tracker

This note explains the new offline checkpoint metric tracker and how to use it in a baseline training loop.

The tracker is designed for the three offline validation metrics discussed earlier:

- `continuous_raw_score`
- `continuous_reward`
- `cpa_exceedance_rate`

It is implemented in:

- `strategy_train_env/run/offline_metric_tracker.py`

and wired into IQL as a working example in:

- `strategy_train_env/run/run_iql.py`

## 1. What The Tracker Does

The tracker solves four related tasks:

1. periodically evaluate a model during training on held-out offline data
2. record validation metrics into `training_curve.csv`
3. track the best checkpoint for each of the three main metrics
4. write a machine-readable summary into `best_metrics.json`

The three tracked “best” metrics are:

- best `continuous_raw_score`
- best `continuous_reward`
- best `cpa_exceedance_rate`

The first two are maximized.

`cpa_exceedance_rate` is minimized.

## 2. Files Added

### `strategy_train_env/run/offline_metric_tracker.py`

This file provides two main pieces:

1. `evaluate_offline_bidding_strategy(...)`

This runs held-out offline validation for any object implementing the local bidding strategy API.

It returns averaged validation metrics such as:

- `continuous_raw_score`
- `continuous_reward`
- `cpa_exceedance_rate`
- `budget_consumer_ratio`
- `sparse_raw_score`
- `sparse_reward`

2. `OfflineMetricTracker`

This is the checkpoint tracker itself.

It:

- evaluates every `eval_interval` steps
- appends rows to `training_curve.csv`
- updates `best_metrics.json`
- stores the checkpoint path associated with the current best metric

### `strategy_train_env/run/run_iql.py`

This now includes:

- `IqlValidationStrategy`
- `save_iql_checkpoint(...)`
- `build_iql_metric_tracker(...)`
- metric tracking inside `train_model_steps(...)`

So IQL is the working example for how to adopt the tracker in other baselines.

## 3. Output Artifacts

For IQL, the tracker writes under:

```text
strategy_train_env/saved_model/IQLtest/
```

The relevant artifacts are:

```text
saved_model/IQLtest/
├── iql_model.pth
├── normalize_dict.pkl
├── checkpoints/
│   ├── step_01000/
│   │   ├── iql_model.pth
│   │   └── normalize_dict.pkl
│   ├── step_02000/
│   └── ...
├── training_curve.csv
└── best_metrics.json
```

`training_curve.csv` stores one row per validation checkpoint.

`best_metrics.json` stores the best value, best step, and checkpoint path for each tracked metric.

For a plotting utility that visualizes these checkpoint curves, see:

- `plot_training_checkpoints.py`
- `docs/training-checkpoint-plots.md`

## 4. How IQL Uses The Tracker

The integration pattern in IQL is:

1. train the model as before
2. create a validation strategy wrapper around the in-memory model
3. create a checkpoint saver
4. instantiate `OfflineMetricTracker`
5. call `maybe_evaluate(...)` inside the training loop

### Validation wrapper

The tracker does not know anything about IQL specifically.

So `run_iql.py` adds a small adapter:

- `IqlValidationStrategy`

This class:

- implements `reset()`
- implements `bidding(...)`
- rebuilds the same 16-dimensional pacing state used by the regular IQL strategy wrapper
- uses the current in-memory IQL model to produce `alpha`

That is the only IQL-specific part of the tracker integration.

### Checkpoint saver

`save_iql_checkpoint(...)` writes:

- JIT model
- normalization dictionary

into a step-specific directory under:

```text
saved_model/IQLtest/checkpoints/
```

### Training-loop hook

Inside `train_model_steps(...)`, IQL now calls:

```python
metric_tracker.maybe_evaluate(
    step,
    extra_metrics={
        "train_q_loss": float(q_loss),
        "train_v_loss": float(v_loss),
        "train_a_loss": float(a_loss),
    },
    force=(step == step_num),
)
```

This means:

- validation runs every `eval_interval`
- the final step is always evaluated
- training losses are stored in the CSV beside validation metrics

## 5. How To Use The Tracker In Another Agent

For another baseline, the integration steps are the same.

### Step 1. Import the tracker

In your `run_<agent>.py` file:

```python
from run.offline_metric_tracker import OfflineMetricTracker, evaluate_offline_bidding_strategy
```

### Step 2. Create a validation strategy wrapper

You need a lightweight strategy object that implements the standard local API:

- `reset()`
- `bidding(...)`

This wrapper can:

- use the in-memory model directly
- or load from a checkpoint

The simplest pattern is the IQL pattern:

- keep a small adapter class inside `run_<agent>.py`
- rebuild the inference features
- call the current model

### Step 3. Create a checkpoint saver

You need a function like:

```python
def save_my_agent_checkpoint(model, metadata, save_root, step):
    checkpoint_dir = os.path.join(save_root, "checkpoints", f"step_{step:05d}")
    ...
    return checkpoint_dir
```

This path is what the tracker stores inside `best_metrics.json`.

### Step 4. Create the tracker

Pattern:

```python
def strategy_factory():
    return MyValidationStrategy(model=model, metadata=metadata)

def evaluator():
    return evaluate_offline_bidding_strategy(
        strategy_factory=strategy_factory,
        file_path=validation_data_path,
        max_groups=max_validation_groups,
    )

metric_tracker = OfflineMetricTracker(
    output_dir=save_root,
    evaluator=evaluator,
    checkpoint_saver=checkpoint_saver,
    eval_interval=eval_interval,
)
```

### Step 5. Call the tracker in the training loop

Inside the training loop:

```python
metric_tracker.maybe_evaluate(
    step,
    extra_metrics={...},
    force=(step == step_num),
)
```

That is the full integration.

## 6. Requirements For A Validation Strategy

The validation strategy passed through `strategy_factory()` should behave like the regular evaluation strategy.

In practice it should:

- have `budget`, `remaining_budget`, `cpa`, and `category`
- implement `reset()`
- implement `bidding(...)`
- return one bid per PV

The generic offline evaluator then sets:

- `budget`
- `cpa`
- `category`

from the held-out validation group before evaluation starts.

That is better than the current script-style offline evaluation, because it makes validation respect the held-out advertiser configuration.

## 7. What Metrics The Tracker Records

The tracker currently writes these validation metrics:

- `continuous_raw_score`
- `continuous_reward`
- `cpa_exceedance_rate`
- `budget_consumer_ratio`
- `sparse_raw_score`
- `sparse_reward`
- `num_groups`

It also stores any extra training metrics you pass in `extra_metrics`.

For IQL that means:

- `train_q_loss`
- `train_v_loss`
- `train_a_loss`

## 8. How Best Metrics Are Defined

The tracker uses these directions:

- `continuous_raw_score`: maximize
- `continuous_reward`: maximize
- `cpa_exceedance_rate`: minimize

So the saved best checkpoint for each metric may be different.

That is intentional.

For example:

- one checkpoint may give the best score
- another may give the best pure value
- another may be the safest on CPA

## 9. Recommended Usage

For practical runs:

- use `continuous_raw_score` as the main checkpoint-selection metric
- use `continuous_reward` and `cpa_exceedance_rate` as diagnostics
- keep `max_validation_groups` small at first if training is too slow
- increase it later for more reliable model selection

For example:

- quick iteration: `max_validation_groups=16` or `32`
- stronger selection: `max_validation_groups=None`

## 10. Current IQL Defaults

The IQL example currently uses:

- `eval_interval=1000`
- `validation_data_path="./data/traffic/period-7.csv"`
- `max_validation_groups=32`

These defaults are chosen to make the example usable without making training prohibitively slow.

If you want more faithful validation, increase `max_validation_groups` or evaluate all groups.

## 11. How To Read The Outputs

### `training_curve.csv`

Use this file to make plots such as:

- `continuous_raw_score` vs `step`
- `continuous_reward` vs `step`
- `cpa_exceedance_rate` vs `step`
- training losses vs `step`

### `best_metrics.json`

Use this file when you want to:

- recover the best checkpoint by score
- recover the safest checkpoint by CPA
- compare which step was best for different criteria

## 12. Suggested Next Extensions

Once you are happy with the IQL example, the next natural extensions are:

1. wire the same tracker into BC
2. wire it into CQL / TD3_BC / BCQ
3. create a small plotting script for `training_curve.csv`
4. optionally add normalized score and period win rate to the tracker output
