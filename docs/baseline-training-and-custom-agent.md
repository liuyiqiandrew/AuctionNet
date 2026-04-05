# AuctionNet Baseline Training, Testing, and Custom Agent Guide

This note explains how the baseline agents in AuctionNet are trained and evaluated, and what you need to implement if you want to add your own agent and use it in the same workflow.

## 1. Mental Model

There are two different evaluation paths in this repo:

1. `strategy_train_env/main/main_test.py`
This is the offline evaluation path. It replays logged traffic from `strategy_train_env/data/traffic/period-7.csv` and evaluates one `PlayerBiddingStrategy` against logged least-winning costs.

2. `main_test.py` at the repo root
This is the online simulator evaluation path. It builds the full 48-agent auction simulator and replaces one player slot with your custom `PlayerBiddingStrategy`.

There is also a separate training path under `strategy_train_env`:

3. `strategy_train_env/main/main_<algo>.py`
These are the baseline training entrypoints such as `main_iql.py`, `main_bc.py`, `main_cql.py`, and so on.

The most important design point is this:

- For local experimentation, the player agent used by both offline eval and root online eval is imported from `strategy_train_env/bidding_train_env/strategy/__init__.py`.
- The classes under `simul_bidding_env/strategy/` are mostly the built-in background competitors used by the simulator `Controller`.

That means:

- If you want to train and test your own agent like the baselines, the minimum integration point is `strategy_train_env/bidding_train_env/strategy/`.
- You only need to edit `simul_bidding_env/strategy/` if you want your custom agent to be part of the fixed competitor pool, not just the injected player.

## 2. End-to-End Baseline Pipeline

The baseline workflow has three stages:

1. Generate or collect per-PV logs.
2. Convert those logs into one row per advertiser per tick for training.
3. Train a model or fit a policy artifact, then evaluate it offline or online.

### 2.1 Raw log format

The simulator can generate raw logs through `simul_bidding_env/Tracker/BiddingTracker.py`.

Each logged row is at ad-opportunity granularity and includes columns such as:

- `deliveryPeriodIndex`
- `advertiserNumber`
- `advertiserCategoryIndex`
- `budget`
- `CPAConstraint`
- `timeStepIndex`
- `remainingBudget`
- `pvIndex`
- `pValue`
- `pValueSigma`
- `bid`
- `xi`
- `adSlot`
- `cost`
- `isExposed`
- `conversionAction`
- `leastWinningCost`
- `isEnd`

This is the format used to build training data.

### 2.2 Training data generation

`strategy_train_env/bidding_train_env/train_data_generator/train_data_generator.py` converts raw per-PV logs into one row per advertiser per tick.

Each training row contains:

- `state`
- `action`
- `reward`
- `reward_continuous`
- `done`
- `next_state`

The default state is a 16-dimensional tuple:

1. `timeleft`
2. `bgtleft`
3. `avg_bid_all`
4. `avg_bid_last_3`
5. `avg_leastWinningCost_all`
6. `avg_pValue_all`
7. `avg_conversionAction_all`
8. `avg_xi_all`
9. `avg_leastWinningCost_last_3`
10. `avg_pValue_last_3`
11. `avg_conversionAction_last_3`
12. `avg_xi_last_3`
13. current tick mean `pValue`
14. current tick PV count
15. last 3 ticks PV count
16. historical PV count

The default action target is:

```text
action = sum(bid) / sum(pValue)
```

In other words, most learned baselines do not predict a bid directly for each PV. They predict a scalar multiplier `alpha`, and the final bid is:

```text
bid = alpha * pValue
```

The default rewards are:

- `reward`: realized conversions in the tick
- `reward_continuous`: sum of exposed `pValue` in the tick

### 2.3 Baseline training scripts

Most learned baselines follow the same pattern:

1. Read `./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv`
2. Parse `state` and `next_state`
3. Normalize selected state dimensions
4. Normalize reward if needed
5. Build a replay buffer or dataset
6. Train the model
7. Save model artifacts under `strategy_train_env/saved_model/<Algo>test/`

Examples:

- `run/run_iql.py`
- `run/run_bc.py`
- `run/run_bcq.py`
- `run/run_cql.py`
- `run/run_td3_bc.py`
- `run/run_decision_transformer.py`

Most learned agents save:

- a model artifact such as `*.pth` or `*.pt`
- normalization metadata such as `normalize_dict.pkl`

One exception is `OnlineLp`, which saves a tabular artifact instead of a neural model:

- `saved_model/onlineLpTest/period.csv`

## 3. How Baselines Are Evaluated

## 3.1 Offline evaluation

Command:

```bash
cd strategy_train_env
python main/main_test.py
```

This calls `run/run_evaluate.py`.

What happens:

1. It loads `./data/traffic/period-7.csv`
2. It groups the data by `(deliveryPeriodIndex, advertiserNumber)`
3. It currently evaluates only `keys[0]`
4. It instantiates `PlayerBiddingStrategy`
5. For each tick, it calls `agent.bidding(...)`
6. It simulates wins against logged `leastWinningCost`
7. It updates `remaining_budget`
8. It computes total reward, cost, CPA, and score

Important limitation:

- This is a replay-style offline check, not a full 48-agent simulator benchmark.
- It is useful as a quick sanity check, but it is weaker evidence than the root online evaluation.

## 3.2 Online simulator evaluation

Command:

```bash
python main_test.py
```

This calls `run/run_test.py`.

What happens:

1. `run_test()` imports `PlayerBiddingStrategy` from `bidding_train_env.strategy`
2. `Controller` creates the built-in competitor agents
3. The player slot `player_index` is replaced with your `PlayerBiddingStrategy`
4. For each tick, the simulator calls `agent.bidding(...)` for every agent
5. The environment runs the auction, exposure, and conversion simulation
6. `PlayerAnalysis` aggregates metrics

The main metrics returned are:

- `score`
- `reward`
- `win_pv_ratio`
- `budget_consumer_ratio`
- `second_price_ratio`
- `cpa_exceedance_Rate`
- `last_compete_tick_index`

`config/test.gin` controls online evaluation settings such as:

- `PVNUM`
- `NUM_EPISODE`
- `NUM_TICK`
- `GENERATE_LOG`

Important distinction:

- `GENERATE_LOG = True` in `config/test.gin` affects the root online evaluation path.
- It does not affect `strategy_train_env/main/main_test.py`.

## 4. The Practical Agent API

For local training and evaluation, the real API is the `BaseBiddingStrategy` interface.

The class contract is:

```python
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import numpy as np


class MyAgentBiddingStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="MyAgent", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        # Load model, table, or hand-coded parameters here.

    def reset(self):
        self.remaining_budget = self.budget
        # Reset any recurrent or per-episode state here.

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        bids = self.cpa * pValues
        return bids
```

### 4.1 Required fields

These fields already exist in `BaseBiddingStrategy`:

- `budget`
- `remaining_budget`
- `name`
- `cpa`
- `category`

You do not need the constructor defaults to be exact. The simulator and evaluation code overwrite `budget`, `cpa`, and `category` before running.

### 4.2 Required methods

You need to implement:

1. `reset(self)`
Reset per-episode state and set `self.remaining_budget = self.budget`.

2. `bidding(...)`
Return one bid per PV for the current tick.

### 4.3 Input contract for `bidding(...)`

The arguments are:

- `timeStepIndex`
Current tick index.

- `pValues`
NumPy array of predicted conversion values for the current tick.

- `pValueSigmas`
NumPy array of uncertainty estimates for the current tick.

- `historyPValueInfo`
List of prior-tick arrays. Each element is usually shaped `(num_pv_in_tick, 2)` and contains `(pValue, pValueSigma)`.

- `historyBid`
List of prior-tick bid arrays.

- `historyAuctionResult`
List of prior-tick arrays. In online eval, each row is effectively `(xi, slot, cost)`. In offline eval, the repo builds a simpler compatibility version, and most baseline strategies only use the first column anyway.

- `historyImpressionResult`
List of prior-tick arrays. In online eval, it is effectively `(isExposed, conversionAction)`. In offline eval, the compatibility representation is simpler.

- `historyLeastWinningCost`
List of prior-tick least-winning-cost arrays.

### 4.4 Output contract for `bidding(...)`

Your method should return:

- a NumPy array
- length equal to `len(pValues)`
- one bid per PV
- preferably non-negative values

The online evaluator clips negative bids to zero, but your strategy should return valid bids directly.

### 4.5 Budget handling

Your strategy should read `self.remaining_budget`, but it usually should not decrement it manually.

The evaluation loops update budget outside the strategy:

- offline in `run/run_evaluate.py`
- online in `run/run_test.py`

So the normal pattern is:

- read `self.remaining_budget`
- decide bids
- let the environment and runner update the remaining budget afterward

## 5. What Local Code Actually Calls

This repo includes a `PlayerAgentWrapper` with an `action(...)` method, but the local training and evaluation scripts do not use that method.

For local development in this repo, the path that matters is:

- implement `reset()`
- implement `bidding()`

The local scripts call `agent.bidding(...)` directly.

So if your goal is:

- train locally
- run offline eval locally
- run root online eval locally

then `bidding()` is the API you need to support.

## 6. Recommended Structure for a New Learned Agent

If you want your own trainable agent to follow the same pattern as the baselines, this is the cleanest structure:

```text
strategy_train_env/
├── bidding_train_env/
│   ├── baseline/
│   │   └── my_agent/
│   │       └── my_agent.py
│   └── strategy/
│       └── my_agent_bidding_strategy.py
├── main/
│   └── main_my_agent.py
├── run/
│   └── run_my_agent.py
└── saved_model/
    └── MyAgentTest/
```

Recommended responsibilities:

1. `baseline/my_agent/my_agent.py`
Implement the trainable model or fitting logic.

2. `run/run_my_agent.py`
Load training data, preprocess states, train the model, and save artifacts.

3. `main/main_my_agent.py`
Thin entrypoint that imports and runs `run_my_agent()`.

4. `strategy/my_agent_bidding_strategy.py`
Load the saved artifact and implement the `bidding(...)` interface used at evaluation time.

5. `saved_model/MyAgentTest/`
Store model weights, normalization stats, lookup tables, or any other inference-time assets.

## 7. Fastest Way to Add Your Own Agent

If you want the least friction, reuse the baseline state/action contract:

1. Keep the same 16-dimensional state.
2. Predict one scalar `alpha` per tick.
3. Return `bids = alpha * pValues`.
4. Save any normalization data next to the model.
5. In your strategy wrapper, rebuild the exact same 16-dimensional state from the histories.

This is the pattern used by IQL, BC, BCQ, CQL, and TD3_BC strategy wrappers.

Why this is the easiest path:

- the data generator already produces the state
- the training scripts already assume that format
- the bidding wrappers already show how to reconstruct the state online and offline

## 8. If You Want a Different State or Action Definition

You can do that, but then you need to own the full contract.

If you change the state definition, you should also change:

1. the training data generator
2. the training script
3. the normalization metadata
4. the inference-time featurization inside your bidding strategy

If you change the action definition, you should also make sure:

1. the model output still maps to one bid per PV at inference time
2. the offline and online evaluators can call your strategy without any special-case code

In practice, the most important rule is:

- training-time featurization and inference-time featurization must match exactly

## 9. Minimal Rule-Based Agent Example

If you want to test the interface before building a learner, start with a simple rule-based policy:

```python
import numpy as np
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy


class MyRuleAgentBiddingStrategy(BaseBiddingStrategy):
    def __init__(self, budget=100, name="MyRuleAgent", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        alpha = self.cpa
        if budget_left > time_left:
            alpha *= 0.8
        else:
            alpha *= 1.1
        bids = alpha * pValues
        bids[bids < 0] = 0
        return bids
```

This is enough to:

- run `strategy_train_env/main/main_test.py`
- run root `main_test.py`

without any training code.

## 10. How to Register Your Agent

For local testing, the main switch is:

`strategy_train_env/bidding_train_env/strategy/__init__.py`

Example:

```python
from .my_agent_bidding_strategy import MyAgentBiddingStrategy as PlayerBiddingStrategy
```

After that:

- `strategy_train_env/main/main_test.py` uses your agent for offline eval
- root `main_test.py` uses your agent as the injected player in online eval

You do not need to modify `Controller` for this case.

## 11. When You Do Need to Modify `Controller`

Modify `simul_bidding_env/Controller/Controller.py` only if you want your new agent to appear as one of the built-in background competitors in the 48-agent simulator.

That is a different goal from evaluating your custom player.

For normal local experiments, the player replacement mechanism in `run/run_test.py` is enough.

## 12. Common Gotchas

1. `strategy_train_env/main/main_test.py` and root `main_test.py` are not the same thing.
The first is offline replay evaluation. The second is full simulator evaluation.

2. `GENERATE_LOG = True` in `config/test.gin` only affects root online evaluation.

3. The root online evaluation currently loops over `player_index in range(0, 2)` in `main_test.py`.
If you want different player slots, change that loop.

4. Saved model paths are hard-coded inside each strategy wrapper.
If your training code saves to a different directory, your wrapper must load from that same directory.

5. Offline evaluation currently uses only the first grouped key in `period-7.csv`.
That makes it a quick check, not a full benchmark.

6. The repo's local strategy API is `bidding(...)`, not `action(...)`.

7. If you change the feature definition, make sure the strategy wrapper reconstructs the exact same features at inference time.

## 13. Suggested Development Workflow

If you are building a new agent from scratch, the safest order is:

1. Start with a no-training rule-based `bidding(...)` implementation.
2. Verify that it runs in `strategy_train_env/main/main_test.py`.
3. Verify that it runs in root `main_test.py`.
4. Add your training code under `strategy_train_env/run/` and `strategy_train_env/main/`.
5. Save artifacts under `strategy_train_env/saved_model/`.
6. Update your strategy wrapper to load those artifacts.
7. Re-run offline eval first, then full online eval.

## 14. Command Summary

Generate training data:

```bash
cd strategy_train_env
python bidding_train_env/train_data_generator/train_data_generator.py
```

Train a baseline:

```bash
cd strategy_train_env
python main/main_iql.py
```

Offline evaluation:

```bash
cd strategy_train_env
python main/main_test.py
```

Online evaluation:

```bash
cd ..
python main_test.py
```

## 15. Short Version

If you want to add your own agent with minimal friction:

1. Create `strategy_train_env/bidding_train_env/strategy/my_agent_bidding_strategy.py`
2. Implement `reset()` and `bidding(...)`
3. Export it as `PlayerBiddingStrategy` in `strategy_train_env/bidding_train_env/strategy/__init__.py`
4. If it is learned, add `run/run_my_agent.py` and `main/main_my_agent.py`
5. Save your inference assets under `strategy_train_env/saved_model/MyAgentTest/`
6. Run offline eval first, then root online eval

That is the same local pattern used by the provided baseline agents.
