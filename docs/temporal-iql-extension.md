# AuctionNet Temporal IQL Extension

This note explains the separate temporal IQL extension added for AuctionNet.

The goal of this extension is simple:

- keep the original IQL idea
- keep the original 16-dimensional pacing state
- but replace the flat one-step state encoder with a learned GRU sequence encoder

In other words, the temporal extension changes the policy from:

```text
one state s_t -> MLP -> action alpha_t
```

to:

```text
last K states [s_{t-K+1}, ..., s_t] -> GRU -> action alpha_t
```

The implementation is separate from the existing IQL and IQL60 paths, so it does not overwrite the original algorithms.

## 1. What Problem This Extension Solves

The original IQL path in this repo already uses a hand-crafted 16-dimensional state that summarizes some history:

- historical means
- last-3-step means
- current pValue summary
- traffic counts

That state is useful, but it is still a single flat vector.

The temporal extension adds a learned sequence model on top of those states.

Instead of asking:

> What should I do from this one summarized state?

it asks:

> What should I do after seeing the last K summarized states in order?

This lets the model learn temporal patterns such as:

- whether spend is accelerating or decelerating
- whether traffic is rising or falling
- whether the campaign is pacing smoothly or reacting late
- whether recent states mean something different from older states

## 2. Files Involved

The temporal extension is split across a few files.

### Entry point

- `strategy_train_env/main/main_iql_temporal.py`

This is the small launcher. It only:

- sets seeds
- imports `run_iql_temporal`
- calls `run_iql_temporal()`

### Training runner

- `strategy_train_env/run/run_iql_temporal.py`

This is the orchestration layer. It:

- chooses training and validation data
- loads the RL dataframe
- normalizes the 16-dim state features
- builds temporal replay windows
- creates the `GRUIQL` model
- trains it
- runs offline checkpoint evaluation during training
- writes plots, reports, and checkpoints

### Temporal utilities

- `strategy_train_env/bidding_train_env/baseline/iql_temporal/sequence_utils.py`

This file handles:

- parsing serialized states from csv
- rebuilding the original 16-dim state at evaluation time
- applying normalization
- storing online temporal context during validation
- building sequence windows for offline training

### Temporal model

- `strategy_train_env/bidding_train_env/baseline/iql_temporal/gru_iql.py`

This file defines the actual neural networks:

- `SequenceQ`
- `SequenceV`
- `SequenceActor`
- `GRUIQL`

### Reused existing utilities

The temporal extension also reuses these existing files:

- `strategy_train_env/bidding_train_env/common/utils.py`
- `strategy_train_env/run/offline_metric_tracker.py`
- `strategy_train_env/bidding_train_env/baseline/iql/metrics.py`

Those reused pieces handle:

- feature normalization
- reward normalization
- checkpoint-time offline validation
- metric logging
- plot generation

## 3. High-Level Training Flow

The training flow in `run_iql_temporal.py` is:

```text
main_iql_temporal.py
  -> run_iql_temporal()
  -> train_iql_temporal_model()
      -> ensure_train_data_cache()
      -> read training csv
      -> safe_literal_eval() on state / next_state
      -> normalize_state()
      -> normalize_reward()
      -> TemporalReplayBuffer.from_dataframe()
      -> GRUIQL(...)
      -> build_temporal_iql_metric_tracker(...)
      -> train_model_steps(...)
          -> replay_buffer.sample(...)
          -> model.step(...)
          -> metric_tracker.maybe_evaluate(...)
      -> model.save_checkpoint(...)
      -> write report and config
```

So the temporal part enters in two places:

1. the replay buffer now returns K-step sequences instead of single states
2. the model now consumes those sequences with a GRU encoder

## 4. Data Representation

### 4.1 Original per-step state

This extension still uses the original 16-dimensional IQL pacing state.

That state contains:

1. time left
2. budget left
3. historical bid mean
4. last-three bid mean
5. historical least winning cost mean
6. historical pValue mean
7. historical conversion mean
8. historical xi mean
9. last-three least winning cost mean
10. last-three pValue mean
11. last-three conversion mean
12. last-three xi mean
13. current pValue mean
14. current pv count
15. last-three pv count total
16. historical pv count total

That reconstruction logic is in:

- `build_iql_flat_state(...)`

inside:

- `strategy_train_env/bidding_train_env/baseline/iql_temporal/sequence_utils.py`

### 4.2 Temporal state window

For training, we do not give the model only one state vector.

Instead, for each step `t`, we build:

```text
X_t = [s_{t-K+1}, ..., s_t]
```

where:

- each `s_i` is a 16-dimensional normalized state
- `K` is `sequence_length`
- if we do not yet have K steps of history, we left-pad with zeros

So each model input state is:

```text
state_sequence shape = (K, 16)
```

and each training batch has:

```text
state_sequences shape = (batch_size, K, 16)
```

The replay buffer also stores:

- `sequence_lengths`
- `actions`
- `rewards`
- `next_state_sequences`
- `next_sequence_lengths`
- `dones`

## 5. How The Sequence Windows Are Built

The sequence construction happens in:

- `TemporalReplayBuffer.from_dataframe(...)`

inside:

- `strategy_train_env/bidding_train_env/baseline/iql_temporal/sequence_utils.py`

For each advertiser trajectory:

1. sort the rows by `timeStepIndex`
2. walk through each step `t`
3. build the current padded window ending at `t`
4. build the next padded window ending at `t+1`
5. if the transition is terminal, use an all-zero next window

Conceptually:

```text
t = 0
current = [0, 0, 0, 0, s0]
next    = [0, 0, 0, s0, s1]

t = 1
current = [0, 0, 0, s0, s1]
next    = [0, 0, s0, s1, s2]

t = last
current = [..., s_{T-1}]
next    = zeros
```

The replay buffer therefore still produces one transition per original row, but each transition now contains temporal context.

## 6. How Offline Validation Works

Training uses the cached RL dataframe.

Offline validation during training uses held-out replay data through:

- `run/offline_metric_tracker.py`

To plug the temporal model into that system, `run_iql_temporal.py` defines:

- `TemporalIqlValidationStrategy`

This wrapper has to do online sequence maintenance during validation because the offline evaluator calls `bidding(...)` one step at a time.

Its flow is:

```text
offline evaluator gives current history
  -> build_iql_flat_state(...)
  -> apply_normalize(...)
  -> TemporalContextBuffer.append(...)
  -> TemporalContextBuffer.as_padded_sequence()
  -> model(state_sequence, sequence_length)
  -> alpha
  -> bid = alpha * pValues
```

So:

- training sequences are prebuilt from the cached RL dataframe
- validation sequences are rebuilt online from the evaluator history

This keeps training-time and eval-time behavior aligned.

## 7. Neural Network Structure

The temporal model is in:

- `strategy_train_env/bidding_train_env/baseline/iql_temporal/gru_iql.py`

There are four main neural-network classes.

## 7.1 `SequenceQ`

Purpose:

- estimate `Q(sequence, action)`

Inputs:

- `state_sequences`
  shape: `(batch_size, seq_len, obs_dim)`
- `sequence_lengths`
  shape: `(batch_size,)`
- `actions`
  shape: `(batch_size, action_dim)`

Internal steps:

1. run the state sequence through a GRU encoder
2. take the final hidden state
3. embed the action with a linear layer
4. concatenate sequence embedding and action embedding
5. pass through an MLP

Output:

- `q`
  shape: `(batch_size, 1)`

Interpretation:

- this is the critic estimate of how good the logged action is under the temporal context

## 7.2 `SequenceV`

Purpose:

- estimate `V(sequence)`

Inputs:

- `state_sequences`
  shape: `(batch_size, seq_len, obs_dim)`
- `sequence_lengths`
  shape: `(batch_size,)`

Internal steps:

1. run the sequence through a GRU encoder
2. take the final hidden state
3. pass through an MLP

Output:

- `value`
  shape: `(batch_size, 1)`

Interpretation:

- this is the temporal state value estimate used by IQL

## 7.3 `SequenceActor`

Purpose:

- define the policy `pi(a | sequence)`

Inputs:

- `state_sequences`
  shape: `(batch_size, seq_len, obs_dim)`
- `sequence_lengths`
  shape: `(batch_size,)`

Internal steps:

1. run the sequence through a GRU encoder
2. take the final hidden state
3. pass through an MLP
4. output:
   - action mean `mu`
   - action log std `log_std`

Outputs from `forward(...)`:

- `mu`
  shape: `(batch_size, action_dim)`
- `log_std`
  shape: `(batch_size, action_dim)`

Outputs from `evaluate(...)`:

- sampled action
  shape: `(batch_size, action_dim)`
- Gaussian distribution object

Outputs from `get_det_action(...)`:

- deterministic action mean
  shape: `(batch_size, action_dim)`

Interpretation:

- the actor predicts the bid multiplier `alpha`
- the final auction bid is still:

```text
bid = alpha * pValue
```

## 7.4 `GRUIQL`

Purpose:

- coordinate the temporal actor, value network, and twin critics
- implement the IQL update rules

Contained modules:

- `value_net`
- `critic1`
- `critic2`
- `critic1_target`
- `critic2_target`
- `actors`

Its `step(...)` function expects:

- `state_sequences`
  shape: `(batch_size, seq_len, obs_dim)`
- `sequence_lengths`
  shape: `(batch_size,)`
- `actions`
  shape: `(batch_size, action_dim)`
- `rewards`
  shape: `(batch_size, 1)`
- `next_state_sequences`
  shape: `(batch_size, seq_len, obs_dim)`
- `next_sequence_lengths`
  shape: `(batch_size,)`
- `dones`
  shape: `(batch_size, 1)`

Its `step(...)` output is:

- `critic1_loss`
- `value_loss`
- `actor_loss`

returned as NumPy scalars for logging consistency with the existing runners.

## 8. How The Temporal IQL Update Works

The update order inside `GRUIQL.step(...)` is:

1. update `V`
2. update actor
3. update `Q1` and `Q2`
4. soft-update target critics

### 8.1 Value loss

`calc_value_loss(...)` computes the expectile regression target:

```text
min(Q1_target, Q2_target) - V
```

Inputs:

- current sequence
- current action

Output:

- scalar value loss

### 8.2 Actor loss

`calc_policy_loss(...)` computes the advantage-weighted policy loss:

```text
-(exp(minQ - V) * log pi(a | sequence)).mean()
```

Inputs:

- current sequence
- logged action

Output:

- scalar actor loss

Interpretation:

- if the logged action looks better than the state value, the actor is pushed toward it more strongly

### 8.3 Critic loss

`calc_q_loss(...)` computes Bellman regression:

```text
target = reward + gamma * (1 - done) * V(next_sequence)
```

Then:

```text
Q1 loss = (Q1(sequence, action) - target)^2
Q2 loss = (Q2(sequence, action) - target)^2
```

Inputs:

- current sequence
- current action
- reward
- done
- next sequence

Output:

- `(critic1_loss, critic2_loss)`

## 9. Why `sequence_lengths` Are Needed

The replay buffer pads short histories with zeros.

That means the model must know how many steps are real and how many are padding.

`sequence_lengths` are used by:

- `_prepare_lengths(...)`
- `_encode_sequence(...)`

and then:

- `pack_padded_sequence(...)`

tells the GRU to ignore left-padding.

Without that, the model would treat zeros as real history, which would blur the temporal signal.

## 10. Role Of `TemporalContextBuffer`

`TemporalContextBuffer` is only for online-style inference during validation.

It stores the last `K` normalized states and returns:

- padded sequence
- valid sequence length

This object is necessary because during validation the evaluator does not hand us a prebuilt sequence window. It only gives the current timestep and historical arrays.

So the wrapper rebuilds the state one step at a time and the buffer keeps the rolling temporal context.

## 11. Role Of `build_iql_flat_state(...)`

This function is important because it keeps the temporal extension grounded in the original IQL feature definition.

It does not invent a new base state.

Instead, it rebuilds the exact same 16-dimensional summary that the original IQL path uses conceptually, then the temporal extension learns over sequences of those states.

So the temporal extension can be understood as:

```text
original 16-dim IQL state
  -> normalized
  -> stacked over time
  -> encoded by a GRU
  -> used by IQL actor / Q / V
```

## 12. Runner Responsibilities In Detail

`run_iql_temporal.py` contains a few important functions.

### `ensure_train_data_cache(...)`

Responsibility:

- build or load the combined training dataframe for periods 7 to 26

Input:

- directory of period RL csv files

Output:

- one combined training csv path

### `train_iql_temporal_model(...)`

Responsibility:

- full training orchestration

Inputs:

- `step_num`
- `eval_interval`
- `validation_data_path`
- `max_validation_groups`
- `sequence_length`
- `batch_size`
- `encoder_hidden_dim`
- `log_interval`

Outputs:

- model checkpoints under `saved_model/IQLTemporaltest`
- `training_curves.png`
- `metrics.json`
- `training_curve.csv`
- `best_metrics.json`
- `training_report.md`
- `model_config.json`

### `train_model_steps(...)`

Responsibility:

- repeatedly sample temporal transitions and call `model.step(...)`

Inputs:

- temporal replay buffer
- model
- optional metric tracker

Outputs:

- none directly
- updates model parameters
- logs losses
- triggers periodic evaluation

### `build_temporal_iql_metric_tracker(...)`

Responsibility:

- connect the temporal model to the existing offline metric tracker

Input:

- in-memory model
- normalization stats
- sequence length
- validation path

Output:

- `OfflineMetricTracker`

### `save_temporal_iql_checkpoint(...)`

Responsibility:

- write one checkpoint directory during training

Writes:

- `gru_iql_model.pt`
- `normalize_dict.pkl`
- `model_config.json`

## 13. What Gets Saved

The default save directory is:

```text
strategy_train_env/saved_model/IQLTemporaltest/
```

Typical artifacts are:

```text
IQLTemporaltest/
├── gru_iql_model.pt
├── normalize_dict.pkl
├── model_config.json
├── training_curves.png
├── metrics.json
├── training_curve.csv
├── best_metrics.json
├── training_report.md
└── checkpoints/
    ├── step_01000/
    │   ├── gru_iql_model.pt
    │   ├── normalize_dict.pkl
    │   └── model_config.json
    └── ...
```

## 14. What This Extension Does Not Do

This first temporal extension is intentionally narrow.

It does not yet:

- feed past actions into the GRU
- feed past rewards into the GRU
- share one encoder instance across actor and critics
- add an online evaluation strategy for the full 48-agent simulator
- replace the original IQL or IQL60 implementations

So the current design is:

- temporal over past normalized states only
- separate GRU encoders inside actor, value, and each critic
- offline replay training

That makes it easier to reason about and keeps the implementation closer to the original IQL structure.

## 15. Mental Model For Readers

If you want one sentence to remember the design:

> Temporal IQL in this repo is the original 16-dim IQL state, lifted from one-step transitions into K-step padded state sequences, then passed through GRU encoders inside the actor and critics before applying the usual IQL losses.

And if you want the shortest file-level mental model:

```text
main_iql_temporal.py
  launches

run_iql_temporal.py
  orchestrates data loading, training, evaluation, saving

sequence_utils.py
  builds and manages temporal state windows

gru_iql.py
  defines the temporal actor, critics, value net, and IQL update

offline_metric_tracker.py
  evaluates checkpoints and writes progress artifacts
```

That is the full extension.
