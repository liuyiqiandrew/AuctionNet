# AuctionNet Training Checkpoint Plots

This note explains how to visualize the checkpoint metrics written by the offline metric tracker during baseline training.

The plotting entrypoint is:

- `plot_training_checkpoints.py`

It reads the artifacts produced by the tracker in:

- `strategy_train_env/run/offline_metric_tracker.py`
- `strategy_train_env/run/run_iql.py`

and turns them into checkpoint-level plots.

## 1. What The Script Plots

The script focuses on the three main checkpoint metrics:

- `continuous_raw_score`
- `continuous_reward`
- `cpa_exceedance_rate`

These come from `training_curve.csv`, which stores one row per validation checkpoint.

If the CSV also contains training losses, the script will additionally produce a second figure for:

- `train_q_loss`
- `train_v_loss`
- `train_a_loss`

## 2. Expected Inputs

For a tracked run such as IQL, the default artifact directory is:

```text
strategy_train_env/saved_model/IQLtest/
```

The plotting script expects:

```text
strategy_train_env/saved_model/IQLtest/
├── training_curve.csv
├── best_metrics.json
└── checkpoints/
    ├── step_01000/
    ├── step_02000/
    └── ...
```

`training_curve.csv` is required.

`best_metrics.json` is optional but recommended. If it is missing, the plotting script will infer the best checkpoint for each of the three main metrics directly from the CSV.

## 3. Basic Usage

From the repo root:

```bash
conda activate AuctionNet
python plot_training_checkpoints.py
```

By default, that command looks at:

```text
strategy_train_env/saved_model/IQLtest/
```

and saves figures under:

```text
strategy_train_env/saved_model/IQLtest/plots/
```

## 4. Useful Examples

Plot the default IQL run:

```bash
python plot_training_checkpoints.py \
  --run-dir strategy_train_env/saved_model/IQLtest
```

Write the plots somewhere else:

```bash
python plot_training_checkpoints.py \
  --run-dir strategy_train_env/saved_model/IQLtest \
  --output-dir figures/iql_checkpoint_plots
```

Override the CSV and best-metric files manually:

```bash
python plot_training_checkpoints.py \
  --csv strategy_train_env/saved_model/IQLtest/training_curve.csv \
  --best-json strategy_train_env/saved_model/IQLtest/best_metrics.json \
  --output-dir figures/iql_checkpoint_plots
```

Add a custom title:

```bash
python plot_training_checkpoints.py \
  --run-dir strategy_train_env/saved_model/IQLtest \
  --title "IQL Offline Validation"
```

## 5. Output Files

The script writes:

```text
<output-dir>/
├── checkpoint_metrics.png
└── training_losses.png
```

`checkpoint_metrics.png` contains the three main checkpoint metrics.

`training_losses.png` is only written if the CSV contains loss columns.

The script also prints a short checkpoint summary to the terminal, including the best step and checkpoint path for each tracked metric.

## 6. How To Read The Main Plot

Each point corresponds to one saved validation checkpoint.

For the default IQL tracker, that usually means:

- one row every `eval_interval` steps
- one saved checkpoint directory under `checkpoints/step_<xxxxx>/`

The figure marks the best checkpoint for each metric using the entries from `best_metrics.json`.

### `continuous_raw_score`

This is the main low-variance model-selection metric.

- higher is better
- it includes the CPA penalty
- it is the best single curve for checkpoint selection

### `continuous_reward`

This tracks expected conversion value without the CPA penalty.

- higher is better
- useful as a secondary diagnostic
- can improve even when CPA behavior gets worse

### `cpa_exceedance_rate`

This tracks whether the checkpoint is overspending relative to the CPA target.

- lower is better
- `0` means exactly on the CPA boundary
- negative values mean the checkpoint is under the CPA target
- positive values mean the checkpoint exceeds the CPA target

The plot draws a horizontal line at `0` to make the CPA boundary easy to see.

## 7. When The Script Is Most Useful

This plot is especially helpful for:

- deciding which checkpoint to export for evaluation
- seeing whether reward gains are actually aligned with score gains
- checking whether CPA control improves or degrades over training
- spotting runs where the policy is not spending at all in early checkpoints

## 8. Notes

- The script is headless and uses the `Agg` backend, so it works well on remote machines and Slurm jobs.
- If the run is still in progress, you can rerun the script at any time and it will plot whatever rows are currently present in `training_curve.csv`.
- The plotting script does not load model checkpoints themselves. It only reads the tracker outputs and uses the stored checkpoint metadata for annotations.
