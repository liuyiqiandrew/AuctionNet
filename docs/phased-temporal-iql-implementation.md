# Phased Temporal IQL Implementation

This document describes the phased temporal IQL implementation in
`strategy_train_env`, the rationale for each change, the artifacts each phase
writes, and how to run and verify the phases.

It extends the original temporal IQL design documented in
`docs/temporal-iql-extension.md`. The original temporal model encoded a window
of 16-dimensional aggregated IQL states with GRUs. The phased implementation
keeps that baseline available, then adds checkpoint hygiene, residual temporal
features, transition-aware tokens, raw-outcome tokens, and a shared-encoder
experiment path.

## Why This Change Was Needed

The first temporal IQL run showed two practical problems:

1. The best checkpoint was not the final checkpoint.
   The run peaked around step 60000, while the root final model was worse. That
   made it easy to deploy or compare the wrong model.

2. The temporal encoder did not add enough information.
   The original GRU saw only the last `K` normalized 16-dimensional states. Those
   states already contain historical means and last-3 summaries, but the GRU did
   not directly see previous action, reward, budget movement, cost, win-rate, or
   raw outcome information.

The phased implementation addresses those issues without replacing the stable
plain IQL baseline. Each phase has a separate output directory and a documented
contract so experiments can be compared cleanly.

## Files Changed Or Added

### `strategy_train_env/run/run_iql_temporal.py`

This is now the phase-aware temporal IQL runner. It handles:

- phase selection through `--phase 1..5`
- phase-specific default save roots
- phase-specific token schema selection
- model construction for residual and shared-encoder variants
- token normalization artifacts
- best sparse checkpoint promotion
- final-step validation before report writing
- phase metadata and human-readable phase reports
- raw-outcome cache generation for transition v2 tokens

### `strategy_train_env/run/offline_metric_tracker.py`

The offline metric tracker now includes:

- `cpa_violation_rate`
- signed `cpa_exceedance_rate` retained as a diagnostic
- best-checkpoint promotion callback support

The important selection behavior is now:

```text
selection metric: sparse_raw_score, maximize
constraint metric: cpa_violation_rate, minimize
diagnostic metric: cpa_exceedance_rate, signed
```

### `strategy_train_env/bidding_train_env/baseline/iql_temporal/sequence_utils.py`

This file now contains the temporal token machinery:

- state-only schema
- `transition_v1` schema
- `transition_v2` schema
- token-level min-max normalization
- online previous-transition feature construction
- replay-buffer support for arbitrary temporal token columns

### `strategy_train_env/bidding_train_env/baseline/iql_temporal/gru_iql.py`

The GRU-IQL model now supports:

- the original separate-encoder GRU architecture
- residual latest-token features
- LayerNorm over temporal features
- a shared temporal encoder variant for phase 5
- optimizer state save/restore for both separate and shared variants

### `strategy_train_env/tests/test_iql_temporal_phases.py`

This test file covers the phase-specific behavior. It is compatible with
`pytest`, but it can also be run directly with Python because the current `rl`
environment does not include pytest.

### `strategy_train_env/train_iql_temporal_phase_gpu_test.sbatch`

This Slurm script runs a selected phase on the `gpu-test` QoS. The phase is
passed as the first sbatch argument.

## Phase Output Contract

Each phase writes to its own default directory:

```text
saved_model/IQLTemporal_phase1_checkpoint_hygiene/
saved_model/IQLTemporal_phase2_residual_gru/
saved_model/IQLTemporal_phase3_transition_v1/
saved_model/IQLTemporal_phase4_transition_v2/
saved_model/IQLTemporal_phase5_shared_encoder/
```

Each phase directory is expected to contain:

```text
model_config.json
phase_metadata.json
phase_report.md
best_metrics.json
metrics.json
training_curve.csv
training_curves.png
checkpoints/
best_sparse_model/
```

Some files depend on validation being enabled. For example, `best_metrics.json`,
`metrics.json`, `training_curve.csv`, `training_curves.png`, and
`best_sparse_model/` are produced or populated when `eval_interval > 0`.

## Phase Metadata

Each phase writes `phase_metadata.json`. This is the machine-readable record of
what was trained. It includes:

```json
{
  "phase": 3,
  "phase_name": "transition_v1",
  "parent_baseline": "IQLTemporaltest",
  "architecture_summary": "Residual separate-encoder GRU-IQL over transition-aware v1 temporal tokens.",
  "token_schema": "transition_v1",
  "token_fields": ["state_0", "...", "prev_done"],
  "state_dim": 16,
  "base_state_dim": 16,
  "token_dim": 21,
  "sequence_length": 8,
  "selection_metric": "sparse_raw_score",
  "constraint_metric": "cpa_violation_rate",
  "train_periods": "period-7 to 26",
  "validation_period": "period-27",
  "best_checkpoint": "saved_model/.../checkpoints/step_60000",
  "best_sparse_raw_score": 38.0,
  "use_residual_latest_state": true,
  "shared_encoder": false,
  "notes": []
}
```

The rationale for this file is reproducibility. A saved model should say which
phase produced it, which token schema it used, whether it used a shared encoder,
and which checkpoint was best.

## Phase Report

Each phase writes `phase_report.md`. This is the human-readable companion to
`phase_metadata.json`. It records:

- what changed in the phase
- what stayed compatible
- training configuration
- final validation metrics
- best sparse checkpoint metrics
- comparison against the previous phase when available
- comparison against `saved_model/IQLtest` when available

The rationale is to make each run understandable without opening logs or reading
code.

## Phase 1: Checkpoint Hygiene

Default save root:

```text
saved_model/IQLTemporal_phase1_checkpoint_hygiene/
```

Architecture:

```text
original separate-encoder GRU-IQL
state-only token schema
token_dim = 16
residual latest-token path = false
shared encoder = false
```

What changed:

- checkpoint selection and reporting only
- no architecture change
- best sparse checkpoint promotion to `best_sparse_model/`
- final model remains separate from best model
- `cpa_violation_rate` added as a proper non-negative constraint metric
- final-step validation forced before reports are written

Rationale:

- The previous temporal run showed that the final root model could be worse than
  an earlier checkpoint.
- Best sparse checkpoint promotion prevents accidentally treating the final
  checkpoint as the best policy.
- Signed CPA exceedance was not a good best-selection metric because negative
  values reward under-spending or low CPA too strongly. The new
  `cpa_violation_rate` is zero when CPA is within constraint and positive only
  when it violates the constraint.

## Phase 2: Residual GRU

Default save root:

```text
saved_model/IQLTemporal_phase2_residual_gru/
```

Architecture:

```text
separate actor/value/Q GRU encoders
state-only token schema
token_dim = 16
residual latest-token path = true
shared encoder = false
```

Feature computation:

```text
h_t = GRU(sequence)
x_t = latest valid token
z_t = LayerNorm(concat(h_t, x_t))
head_input = z_t
```

What changed:

- each separate actor/value/Q temporal network now has access to both the GRU
  summary and the latest token
- LayerNorm is applied to the concatenated temporal feature
- input schema remains compatible with phase 1

Rationale:

- A GRU hidden state can bottleneck important instantaneous bidding signals such
  as `time_left`, `budget_left`, and current traffic volume.
- Concatenating the latest token lets the model use both temporal context and the
  current state directly.
- Keeping separate encoders preserves the original IQL update structure and
  avoids shared-optimizer complexity in this phase.

## Phase 3: Transition Token V1

Default save root:

```text
saved_model/IQLTemporal_phase3_transition_v1/
```

Architecture:

```text
separate actor/value/Q GRU encoders
transition_v1 token schema
token_dim = 21
residual latest-token path = true
shared encoder = false
```

Token schema:

```text
token_t =
  16 current state features
  + prev_action
  + prev_sparse_reward
  + prev_continuous_reward
  + prev_budget_delta
  + prev_done
```

Field definitions:

- `prev_action`: prior row action in the same `(deliveryPeriodIndex, advertiserNumber)` group
- `prev_sparse_reward`: prior row `reward`
- `prev_continuous_reward`: prior row `reward_continuous`
- `prev_budget_delta`: previous budget-left minus current budget-left, clipped at `>= 0`
- `prev_done`: prior row `done`

First-token behavior:

```text
first token in each advertiser-period group:
  prev_action = 0
  prev_sparse_reward = 0
  prev_continuous_reward = 0
  prev_budget_delta = 0
  prev_done = 0
```

Rationale:

- The original GRU only observed a sequence of aggregated states.
- Previous action and outcome are important for pacing because the same current
  state can mean different things depending on what the agent just bid and what
  happened.
- `transition_v1` uses only fields already present in the existing RL cache, so
  it adds temporal information without regenerating training data from raw CSVs.

Normalization:

- Transition tokens use a new `token_normalize_dict.pkl`.
- All token dimensions except binary `prev_done` are min-max normalized.
- This avoids overloading the old `normalize_dict.pkl`, which was designed for
  the original 16-dimensional state.

## Phase 4: Transition Token V2

Default save root:

```text
saved_model/IQLTemporal_phase4_transition_v2/
```

Architecture:

```text
separate actor/value/Q GRU encoders
transition_v2 token schema
token_dim = 25
residual latest-token path = true
shared encoder = false
```

Token schema:

```text
transition_v1 fields
+ prev_cost_ratio
+ prev_win_rate
+ prev_mean_lwc
+ prev_num_pv_norm
```

Raw field sources:

- `prev_cost_ratio`: previous tick cost divided by campaign budget
- `prev_win_rate`: previous tick mean `xi`
- `prev_mean_lwc`: previous tick mean `leastWinningCost`
- `prev_num_pv_norm`: previous tick row count

These fields are built from raw period CSV columns:

```text
cost
xi
leastWinningCost
pvIndex
```

Rationale:

- Phase 3 uses only existing RL cache columns. That is deliberately low risk but
  still hides useful auction-outcome information.
- Phase 4 adds the previous tick's spend and market interaction features so the
  model can learn whether prior bids were too aggressive or too conservative.
- Keeping phase 3 available makes it possible to measure whether raw outcomes
  actually help.

Cache behavior:

- The runner tries to save a merged temporal outcome cache under
  `data/traffic/training_data_rlData_folder/`.
- If that directory is not writable, training continues using the in-memory
  merged frame and logs a warning.

The rationale for opportunistic cache writing is that Slurm jobs should benefit
from caching, but local or sandboxed checks should not fail just because the data
directory cannot be written.

## Phase 5: Shared Encoder

Default save root:

```text
saved_model/IQLTemporal_phase5_shared_encoder/
```

Architecture:

```text
shared TemporalEncoder(history) -> z_t
ActorHead(z_t) -> action
ValueHead(z_t) -> V
QHead1(z_t, action) -> Q1
QHead2(z_t, action) -> Q2
```

Configuration:

```text
transition_v2 token schema
token_dim = 25
residual latest-token path = true
shared encoder = true
```

Optimizer behavior:

- The shared-encoder variant uses one combined optimizer.
- The combined update optimizes value, actor, Q1, and Q2 losses together.
- The target critics and target temporal encoder are soft-updated.

Rationale:

- The separate-encoder design can learn inconsistent actor/value/Q temporal
  representations.
- A shared encoder may be more sample efficient with the small temporal dataset.
- It is isolated as phase 5 because changing encoder ownership changes optimizer
  semantics and should not be mixed with phase 3 or phase 4 comparisons.

## Online Validation Semantics

The offline training tokens and online validation tokens are intentionally
matched.

For phase 1 and phase 2:

```text
validation token = normalized 16-dimensional state
```

For phase 3 and phase 4:

```text
validation token = current 16-dimensional state + previous rollout outcome fields
```

During offline validation, previous-transition fields are reconstructed from the
history lists maintained by the validation loop:

- previous bid array
- previous auction result
- previous impression result
- previous pValue information
- previous least winning cost

This matters because a mismatch between training-token semantics and
validation-token semantics would make validation scores unreliable.

## Best Checkpoint Promotion

When a validation row improves `sparse_raw_score`, the checkpoint tracker calls
the phase runner's promotion hook. The hook copies the step checkpoint into:

```text
best_sparse_model/
```

and writes:

```text
best_sparse_model/best_sparse_metrics.json
```

The root model remains the final model. The best sparse model is the selected
model for comparison and deployment.

Rationale:

- The earlier temporal run peaked before the final checkpoint.
- Keeping both final and best checkpoint avoids ambiguity.
- `phase_report.md` records both final and best metrics.

## Running Locally

From `strategy_train_env`:

```bash
conda run -n rl python -B -m run.run_iql_temporal --phase 1
conda run -n rl python -B -m run.run_iql_temporal --phase 2
conda run -n rl python -B -m run.run_iql_temporal --phase 3
conda run -n rl python -B -m run.run_iql_temporal --phase 4
conda run -n rl python -B -m run.run_iql_temporal --phase 5
```

Useful overrides:

```bash
conda run -n rl python -B -m run.run_iql_temporal \
  --phase 3 \
  --total-steps 60000 \
  --eval-interval 500 \
  --max-validation-groups 8 \
  --sequence-length 8 \
  --batch-size 256
```

Use a custom save root:

```bash
conda run -n rl python -B -m run.run_iql_temporal \
  --phase 3 \
  --save-root saved_model/my_phase3_run
```

Resume a phase:

```bash
conda run -n rl python -B -m run.run_iql_temporal \
  --phase 3 \
  --resume latest
```

## Running On `gpu-test`

Use:

```bash
cd /scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet/strategy_train_env
sbatch train_iql_temporal_phase_gpu_test.sbatch 3
```

The first argument is the phase:

```bash
sbatch train_iql_temporal_phase_gpu_test.sbatch 1
sbatch train_iql_temporal_phase_gpu_test.sbatch 2
sbatch train_iql_temporal_phase_gpu_test.sbatch 3
sbatch train_iql_temporal_phase_gpu_test.sbatch 4
sbatch train_iql_temporal_phase_gpu_test.sbatch 5
```

Environment overrides:

```bash
IQL_TEMPORAL_TOTAL_STEPS=60000 \
IQL_TEMPORAL_EVAL_INTERVAL=500 \
IQL_TEMPORAL_VALIDATION_GROUPS=8 \
sbatch train_iql_temporal_phase_gpu_test.sbatch 3
```

Optional custom save root:

```bash
IQL_TEMPORAL_SAVE_ROOT=saved_model/phase3_seed2 \
IQL_TEMPORAL_SEED=2 \
sbatch train_iql_temporal_phase_gpu_test.sbatch 3
```

The sbatch script uses:

```text
qos: gpu-test
gpu: 1
time: 00:55:00
conda env: rl
```

It also uses `timeout --signal=USR1` so the Python training loop gets a chance
to save a recovery checkpoint before the test allocation ends.

## Recommended Experiment Order

Run phases in this order:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5

Compare each phase against:

```text
saved_model/IQLTemporaltest
saved_model/IQLtest
```

Use:

```text
primary score: sparse_raw_score
constraint: cpa_violation_rate == 0
diagnostics: continuous_raw_score, continuous_reward, budget_consumer_ratio, signed cpa_exceedance_rate
```

Phase 3 is the first phase expected to materially change temporal behavior.
Phase 4 and phase 5 are more expensive and should be run after phase 3 has a
stable comparison point.

## Testing And Verification

The focused test file is:

```text
strategy_train_env/tests/test_iql_temporal_phases.py
```

Run it directly:

```bash
cd /scratch/gpfs/SIMONSOBS/users/yl9946/scratch/rl_project/AuctionNet/strategy_train_env
conda run -n rl python -B tests/test_iql_temporal_phases.py
```

The tests cover:

- transition v1 previous-field alignment
- group-boundary reset behavior
- transition v2 token dimensionality
- replay padding and sequence lengths
- residual model forward and one training step
- shared-encoder model forward and one training step
- phase metadata writing
- phase report writing
- best sparse checkpoint promotion

The direct Python entrypoint exists because the current `rl` environment does
not include `pytest`.

## Known Operational Notes

### Phase 4 and phase 5 are heavier

The transition v2 path reads and groups raw period CSVs to build previous
outcome summaries. This can take materially longer than phase 3, especially the
first time before the temporal outcome cache exists.

### Cache write may be skipped in restricted environments

If the raw-outcome cache cannot be written, the runner logs a warning and uses
the in-memory merged data. This is expected in read-only or sandboxed contexts.
On a normal Slurm job with writable project storage, the cache should be saved.

### Legacy checkpoints are not automatically compatible with new phases

Phase 1 and legacy temporal checkpoints use 16-dimensional tokens. Phases 3-5
use 21- or 25-dimensional tokens. Resume only within the same phase/schema unless
you explicitly know the checkpoint config matches the requested model.

### Existing plain IQL remains the benchmark

Plain `IQLtest` previously outperformed the first temporal run. The purpose of
these phases is to find out whether richer temporal encoding can beat that
baseline, not to replace it by assumption.

## Rationale Summary By Change

| Change | Rationale |
| --- | --- |
| `phase_metadata.json` | Makes each saved model self-describing and comparable. |
| `phase_report.md` | Records human-readable change summary and metrics per run. |
| `best_sparse_model/` | Prevents deploying a worse final checkpoint when validation peaked earlier. |
| `cpa_violation_rate` | Separates constraint violation from signed CPA diagnostics. |
| residual latest token | Avoids hiding important instantaneous state behind the GRU bottleneck. |
| transition v1 tokens | Adds causal previous action and outcome context from the existing RL cache. |
| transition v2 tokens | Adds raw auction outcome signals that the RL cache does not expose directly. |
| shared encoder phase | Tests whether one temporal representation improves sample efficiency. |
| phase-specific save roots | Prevents artifacts from different experiments from overwriting each other. |
| gpu-test sbatch argument | Makes quick phase-specific Slurm runs easy and reproducible. |
