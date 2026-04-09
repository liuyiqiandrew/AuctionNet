# Data Processing Pipeline: From Raw Auction Logs to RL Training Data

This document explains how the raw per-ad-opportunity auction logs are structured, and how `train_data_generator.py` aggregates them into the standard `(s, a, r, s', done)` RL training format.

## 1. Raw Data: Per-Ad-Opportunity Auction Logs

### 1.1 Origin

The raw CSV files (e.g., `period-7.csv`) are produced by the auction simulator's `BiddingTracker` (in `simul_bidding_env/Tracker/BiddingTracker.py`). During each simulation tick, the tracker logs one row **per (agent, ad-opportunity) pair**. Since every agent participates in every ad opportunity's auction, a single tick with `N` ad opportunities and 48 agents produces `N * 48` rows.

### 1.2 Scale

A typical raw file like `period-7.csv` contains ~24 million rows. This is because:

- 48 agents
- 48 time steps per episode
- Thousands of ad opportunities (PVs) per time step (varying per step, e.g., 1,400 to 12,500)
- Every agent sees every PV: `rows = sum_over_steps(num_pvs_at_step) * 48_agents`

### 1.3 Column Definitions

Each row represents one agent's participation in one ad opportunity's auction:

| Column | Type | Description |
|--------|------|-------------|
| `deliveryPeriodIndex` | float | Episode index (e.g., 7.0 for period-7). One CSV = one episode. |
| `advertiserNumber` | float | Agent index (0–47). There are 48 agents total (6 categories x 8 agents). |
| `advertiserCategoryIndex` | float | Category of the agent (0–5). |
| `budget` | float | Total budget for this agent in this episode. Fixed throughout. |
| `CPAConstraint` | float | Target cost-per-acquisition constraint for this agent. Fixed throughout. |
| `timeStepIndex` | float | Tick index (0–47). Each episode has 48 ticks. |
| `remainingBudget` | float | Agent's remaining budget at the **start** of this tick. |
| `pvIndex` | float | Global index of the ad opportunity (PV). |
| `pValue` | float | Predicted conversion probability of this PV for this agent. |
| `pValueSigma` | float | Uncertainty (std dev) of the pValue estimate. |
| `bid` | float | The bid this agent placed for this PV. |
| `xi` | float | 1 if the agent was among the top-3 bidders (won a slot), 0 otherwise. |
| `adSlot` | float | Slot position won (1=best, 2, 3), or 0 if no slot won. |
| `cost` | float | Second-price cost paid if the agent won a slot, 0 otherwise. |
| `isExposed` | float | 1 if the ad was actually shown (stochastic exposure based on slot position), 0 otherwise. |
| `conversionAction` | float | 1 if a conversion occurred (stochastic, based on pValue), 0 otherwise. Only possible if `isExposed=1`. |
| `leastWinningCost` | float | The market price of the last (3rd) slot — the minimum bid needed to win any slot. Same for all agents at a given PV. |
| `isEnd` | float | 1 if this agent's episode ended at this tick (budget exhausted), 0 otherwise. |

### 1.4 Auction Mechanics (from `BiddingEnv`)

Understanding how these columns are produced helps interpret the data:

1. **Slot assignment**: For each PV, all 48 agents bid. The top 3 bids win slots 1, 2, 3. The rest get `adSlot=0`, `xi=0`.
2. **Second-price cost**: Each winner pays the next-highest bid (generalized second-price). Specifically, slot `k` pays the bid of slot `k+1`. If the resulting price is below the reserve price (0.01), the slot is treated as unsold.
3. **Stochastic exposure**: Winning a slot does not guarantee the ad is shown. Exposure probability depends on slot: slot 1 = 100%, slot 2 = 80%, slot 3 = 60%. A Bernoulli draw determines `isExposed`. If slot 2 is not exposed, slot 3 is also forced to not be exposed (continuity rule).
4. **Conversion**: If exposed, a conversion happens with probability = `pValue` (more precisely, a truncated-normal sample around `pValue`). `conversionAction` is the Bernoulli outcome.
5. **Unsold slots**: If cost equals the reserve price, the slot is treated as unsold — `xi`, `adSlot`, `cost`, `isExposed`, and `conversionAction` are all zeroed out.

### 1.5 Key Structural Properties

- **All agents see all PVs**: At a given time step, every agent has exactly the same set of PV rows. Agent 0 at step 0 sees 12,516 PVs, and so does agent 1, agent 2, etc.
- **PV volume varies across steps**: The number of ad opportunities per tick is not constant (e.g., step 0 might have 12,516 PVs, step 5 might have 1,469).
- **One row = one auction participation, not one auction win**: Most rows have `isExposed=0` because most agents lose most auctions. Winning is sparse.

## 2. RL Data: Per-Agent-Per-Tick Trajectories

### 2.1 The Aggregation Problem

The raw data has **thousands of rows per (agent, time step)** — one per ad opportunity. RL algorithms need **one transition per (agent, time step)**. The data generator's job is to aggregate the per-PV auction outcomes into a single `(state, action, reward, next_state, done)` tuple per tick.

### 2.2 Processing Pipeline

The generator (`_generate_train_data` in `train_data_generator.py`) processes each episode's data as follows:

#### Step 1: Group by Agent Identity

```python
df.groupby(['deliveryPeriodIndex', 'advertiserNumber',
            'advertiserCategoryIndex', 'budget', 'CPAConstraint'])
```

Each group is one agent's full trajectory across all 48 ticks, containing thousands of per-PV rows.

#### Step 2: Compute Per-Tick Volume Features

Within each agent's trajectory:

- **`timeStepIndex_volume`**: Number of PVs at each tick (same across agents, since all agents see all PVs).
- **`historical_volume`**: Cumulative PV count from all previous ticks (shifted by 1 — does not include current tick).
- **`last_3_timeStepIndexs_volume`**: Sum of PV counts in the previous 3 ticks (rolling window, shifted by 1).

#### Step 3: Compute Per-Tick Aggregate Statistics

The generator computes per-tick means of five key columns:

```python
group_agg = group.groupby('timeStepIndex').agg({
    'bid': 'mean',
    'leastWinningCost': 'mean',
    'conversionAction': 'mean',
    'xi': 'mean',
    'pValue': 'mean',
    'timeStepIndex_volume': 'first'
})
```

Each of these is the **mean across all PVs at that tick** for this particular agent. For example, `mean(bid)` is the agent's average bid across all ~N ad opportunities at that tick.

#### Step 4: Compute Historical Rolling Features

For each of the 5 aggregated columns (`bid`, `leastWinningCost`, `conversionAction`, `xi`, `pValue`), two historical statistics are computed:

- **`avg_<col>_all`**: Expanding (cumulative) mean of per-tick averages from all **previous** ticks. This is a "lifetime average up to now" — using `expanding().mean().shift(1)`.
- **`avg_<col>_last_3`**: Rolling-3-window mean of per-tick averages from the **previous** ticks. This is a "recent trend" — using `rolling(window=3).mean().shift(1)`.

The `.shift(1)` is critical: it ensures these features only use **past information** (no lookahead). At `t=0`, all historical features are 0 (filled by `fillna(0)`).

#### Step 5: Assemble State, Action, Reward, Done

For each tick within each agent's trajectory, the generator reads the aggregated/rolled features and produces one RL transition row.

### 2.3 State Space (16 dimensions)

The state vector captures the agent's budget status, historical market behavior, and current market conditions:

| Dim | Name | Source | Interpretation |
|-----|------|--------|----------------|
| 0 | `timeleft` | `(48 - t) / 48` | Fraction of episode remaining. 1.0 at start, ~0.02 at last step. |
| 1 | `bgtleft` | `remainingBudget / budget` | Fraction of budget remaining. Decreases as agent wins auctions. |
| 2 | `avg_bid_all` | expanding mean of per-tick mean bid | Agent's overall average bidding intensity across all past ticks. |
| 3 | `avg_bid_last_3` | rolling-3 mean of per-tick mean bid | Agent's recent bidding intensity (short-term trend). |
| 4 | `avg_leastWinningCost_all` | expanding mean of per-tick mean market price | Overall average market competitiveness. Higher = more expensive market. |
| 5 | `avg_pValue_all` | expanding mean of per-tick mean pValue | Overall average quality of ad opportunities seen. |
| 6 | `avg_conversionAction_all` | expanding mean of per-tick mean conversion rate | Agent's overall conversion success rate. |
| 7 | `avg_xi_all` | expanding mean of per-tick mean win rate | Agent's overall auction win rate (fraction of auctions where agent got a slot). |
| 8 | `avg_leastWinningCost_last_3` | rolling-3 mean of per-tick mean market price | Recent market competitiveness trend. |
| 9 | `avg_pValue_last_3` | rolling-3 mean of per-tick mean pValue | Recent ad opportunity quality trend. |
| 10 | `avg_conversionAction_last_3` | rolling-3 mean of per-tick mean conversion rate | Recent conversion success trend. |
| 11 | `avg_xi_last_3` | rolling-3 mean of per-tick mean win rate | Recent auction win rate trend. |
| 12 | `pValue_agg` | mean pValue at **current** tick | Quality of ad opportunities available right now. |
| 13 | `timeStepIndex_volume_agg` | PV count at **current** tick | How many ad opportunities are available right now. |
| 14 | `last_3_timeStepIndexs_volume` | sum of PV counts in previous 3 ticks | Recent market activity level. |
| 15 | `historical_volume` | cumulative PV count from all previous ticks | Total market activity seen so far. |

**Design rationale**:
- Dims 0–1 encode the agent's own resource status (time and budget).
- Dims 2–11 encode historical performance and market conditions, split into "all history" (long-term) and "last 3 steps" (short-term trend) views.
- Dims 12–15 encode the current tick's market snapshot.
- The `.shift(1)` on dims 2–11 prevents information leakage — the agent cannot see the current tick's auction outcomes when deciding how to bid.
- Dims 12–15 do describe the current tick (PV values and counts are known before bidding). This is realistic: the agent knows what opportunities are available before placing bids.

### 2.4 Action (scalar)

```python
action = sum(bid) / sum(pValue)
```

The action is a **bid multiplier (alpha)** — the ratio of total bids to total predicted value across all PVs at this tick. It represents the agent's overall bidding aggressiveness:

- `alpha = 1.0`: bidding exactly at predicted value
- `alpha > 1.0`: bidding above value (aggressive, spending budget faster)
- `alpha < 1.0`: bidding below value (conservative, preserving budget)

This is why learned baselines predict a single scalar `alpha` per tick, and the final per-PV bid is `bid_i = alpha * pValue_i`.

### 2.5 Reward

Two reward signals are recorded:

| Reward | Formula | Meaning |
|--------|---------|---------|
| `reward` | `sum(conversionAction where isExposed=1)` | **Discrete**: number of actual conversions won at this tick. |
| `reward_continuous` | `sum(pValue where isExposed=1)` | **Continuous**: total expected conversion value of impressions won at this tick. |

Both only count opportunities where the agent actually won the auction **and** was exposed (the ad was shown). The discrete reward is the standard RL objective — maximize total conversions. The continuous reward provides a smoother learning signal.

### 2.6 Done Signal

```python
done = 1  if  timeStepIndex == 47  or  isEnd == 1  else  0
```

The episode terminates when:
- All 48 ticks are exhausted (`timeStepIndex == 47`), or
- The agent's budget was depleted (`isEnd == 1` in the raw data).

### 2.7 Next State

```python
next_state = state of same agent at timeStepIndex + 1
```

Computed via a grouped shift:

```python
training_data.groupby(['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(-1)
```

When `done == 1`, `next_state` is set to `None` (terminal state).

### 2.8 Episode-Level Metadata

Each row also carries episode-level metadata for analysis and potential reward shaping:

- `realAllCost`: total cost incurred by this agent across **all** ticks in the episode.
- `realAllConversion`: total conversions achieved by this agent across **all** ticks.

These are **not** used by the RL state/reward directly but can be useful for post-hoc analysis or alternative reward formulations.

## 3. The RL Process

Putting it all together, the MDP defined by this data generator is:

```
Environment:  Ad auction with 48 competing agents
Agent:        One advertiser with a fixed budget and CPA constraint
Episode:      One delivery period (48 ticks)

For each tick t = 0, 1, ..., 47:

    1. OBSERVE state s_t:
       - How much time and budget remain (dims 0-1)
       - Historical summary of own bidding, market prices, win rates,
         conversion rates, and opportunity quality (dims 2-11)
       - Current tick's opportunity landscape: mean PV value and
         volume of available ad opportunities (dims 12-15)

    2. SELECT action a_t:
       - A scalar multiplier alpha
       - Applied uniformly: bid_i = alpha * pValue_i for each PV

    3. EXECUTE auction:
       - All 48 agents bid on all PVs simultaneously
       - Top-3 bids per PV win slots (generalized second-price)
       - Exposure is stochastic (slot-dependent probability)
       - Conversions are stochastic (pValue-dependent probability)
       - Agent pays second-price cost for won slots

    4. RECEIVE reward r_t:
       - Number of conversions from exposed impressions at this tick

    5. TRANSITION to s_{t+1}:
       - Budget decreases by total cost paid
       - Historical statistics update with new observations
       - PV landscape changes (new set of ad opportunities)

    6. TERMINATION:
       - If t == 47 (time exhausted) or budget depleted: done = 1

Objective: Maximize cumulative conversions over the episode
           subject to the budget constraint.
```

## 4. Output Format

The final training CSV has one row per (agent, tick) with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `deliveryPeriodIndex` | float | Episode index |
| `advertiserNumber` | float | Agent index (0-47) |
| `advertiserCategoryIndex` | float | Agent category (0-5) |
| `budget` | float | Total budget |
| `CPAConstraint` | float | CPA target |
| `realAllCost` | float | Episode-total cost (metadata) |
| `realAllConversion` | float | Episode-total conversions (metadata) |
| `timeStepIndex` | float | Tick index (0-47) |
| `state` | tuple(16) | 16-dim state vector (stored as string repr of Python tuple) |
| `action` | float | Bid multiplier alpha |
| `reward` | float | Conversions at this tick |
| `reward_continuous` | float | Sum of pValues of exposed impressions |
| `done` | int | 1 if terminal, 0 otherwise |
| `next_state` | tuple(16) or None | Next tick's state, None if terminal |

A single-period file has `48 agents * 48 ticks = 2,304 rows`. The generator also produces a combined file (`training_data_all-rlData.csv`) concatenating all periods.

## 5. Data Flow Diagram

```
Raw CSV (period-7.csv)                    RL CSV (period-7-rlData.csv)
~24M rows                                 2,304 rows
one row per (agent, PV, tick)             one row per (agent, tick)
┌──────────────────────────┐              ┌─────────────────────────┐
│ agent 0, tick 0, PV 0    │              │ agent 0, tick 0         │
│ agent 0, tick 0, PV 1    │              │   state: (16-dim)       │
│ ...                      │  aggregate   │   action: alpha         │
│ agent 0, tick 0, PV 12515│ ──────────>  │   reward: conversions   │
│ agent 1, tick 0, PV 0    │   per tick   │   next_state: (16-dim)  │
│ ...                      │   per agent  │   done: 0               │
│ agent 47, tick 47, PV N  │              │ ...                     │
└──────────────────────────┘              │ agent 47, tick 47       │
                                          │   done: 1               │
                                          └─────────────────────────┘
```
