"""Plot PPO sweep rollout curves on shared axes.

Example from the repo root:

    python strategy_train_env/plot_ppo_sweep_curves.py

Useful focused views:

    python strategy_train_env/plot_ppo_sweep_curves.py --filter obs16
    python strategy_train_env/plot_ppo_sweep_curves.py --filter obs60
    python strategy_train_env/plot_ppo_sweep_curves.py --metrics score ep_rew_mean
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import deque
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "score": "Score",
    "ep_rew_mean": "Episode reward mean",
    "dense": "Dense reward",
    "sparse": "Sparse reward",
    "conversions": "Conversions",
    "cost_over_budget": "Cost / budget",
    "target_cpa_over_cpa": "Target CPA / CPA",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_training_root = root / "output" / "online" / "training" / "ongoing"
    default_output = root / "output" / "online" / "training" / "ongoing" / (
        "ppo_default_lr1e4_sweep_score_comparison.png"
    )

    parser = argparse.ArgumentParser(
        description="Overlay PPO sweep rollout curves from rollout_log.jsonl files."
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=default_training_root,
        help="Directory containing PPO run directories.",
    )
    parser.add_argument(
        "--run-glob",
        action="append",
        default=["sweep*_ppo_seed_0*"],
        help="Run directory glob relative to --training-root. Repeat to include multiple glob patterns.",
    )
    parser.add_argument(
        "--extra-run",
        action="append",
        type=Path,
        default=[],
        help=(
            "Extra run directory to include. May be absolute, or relative to --training-root. "
            "Repeat to include multiple controls."
        ),
    )
    parser.add_argument(
        "--filter",
        choices=[
            "all",
            "obs16",
            "obs16_temporal",
            "obs16_shared",
            "obs16_nonshared",
            "obs60",
            "obs60_temporal",
        ],
        default="all",
        help="Convenience filter for the default sweep run names.",
    )
    parser.add_argument(
        "--include-regex",
        help="Optional regex that run directory names must match.",
    )
    parser.add_argument(
        "--exclude-regex",
        help="Optional regex that run directory names must not match.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["score"],
        help="Rollout log metrics to plot. Multiple metrics produce stacked panels.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Causal rolling mean window in rollout rows. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also draw faint unsmoothed curves behind the smoothed curves.",
    )
    parser.add_argument(
        "--max-steps",
        type=float,
        help="Optional maximum env step value to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="PPO default bc_range / lr=1e-4 sweep",
        help="Figure title.",
    )
    parser.add_argument(
        "--legend-cols",
        type=int,
        default=1,
        help="Number of legend columns.",
    )
    return parser.parse_args()


def matches_filter(name: str, filter_name: str) -> bool:
    if filter_name == "all":
        return True
    if filter_name == "obs16":
        return "obs16" in name
    if filter_name == "obs16_temporal":
        return "obs16" in name and "tseq" in name
    if filter_name == "obs16_shared":
        return "obs16" in name and "sharedfe" in name
    if filter_name == "obs16_nonshared":
        return "obs16" in name and ("sharedfe" not in name)
    if filter_name == "obs60":
        return "obs60" in name
    if filter_name == "obs60_temporal":
        return "obs60" in name and "tseq" in name
    raise ValueError(f"Unhandled filter: {filter_name}")


def discover_runs(args: argparse.Namespace) -> list[Path]:
    runs: dict[Path, None] = {}
    for pattern in args.run_glob:
        for run_dir in args.training_root.glob(pattern):
            if run_dir.is_dir():
                runs[run_dir] = None
    for run_dir in args.extra_run:
        resolved = run_dir if run_dir.is_absolute() else args.training_root / run_dir
        if resolved.is_dir():
            runs[resolved] = None

    include = re.compile(args.include_regex) if args.include_regex else None
    exclude = re.compile(args.exclude_regex) if args.exclude_regex else None

    selected = []
    for run_dir in sorted(runs):
        name = run_dir.name
        if not matches_filter(name, args.filter):
            continue
        if include and not include.search(name):
            continue
        if exclude and exclude.search(name):
            continue
        if (run_dir / "rollout_log.jsonl").exists():
            selected.append(run_dir)
    return selected


def read_rollout_rows(run_dir: Path) -> list[dict]:
    rows = []
    with open(run_dir / "rollout_log.jsonl", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    q: deque[float] = deque()
    total = 0.0
    smoothed = []
    for value in values:
        q.append(value)
        total += value
        if len(q) > window:
            total -= q.popleft()
        smoothed.append(total / len(q))
    return smoothed


def metric_xy(rows: list[dict], metric: str, max_steps: float | None) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for row in rows:
        step = row.get("timesteps")
        value = row.get(metric)
        if step is None or value is None:
            continue
        if max_steps is not None and step > max_steps:
            continue
        xs.append(float(step))
        ys.append(float(value))
    return xs, ys


def short_label(name: str) -> str:
    prefix_match = re.match(r"sweep(\d+)_", name)
    if prefix_match:
        prefix = prefix_match.group(1)
    else:
        numeric_prefix_match = re.match(r"(\d{3})_", name)
        prefix = numeric_prefix_match.group(1) if numeric_prefix_match else ""
    obs_match = re.search(r"obs(16|60)", name)
    obs = f"obs{obs_match.group(1)}" if obs_match else "obs?"

    if "tseq" not in name:
        return f"{prefix} flat {obs}".strip()

    seq_match = re.search(r"tseq(\d+)", name)
    hidden_match = re.search(r"_h(\d+)_", name)
    seq = f"t{seq_match.group(1)}" if seq_match else "t?"
    hidden = f"h{hidden_match.group(1)}" if hidden_match else "h?"
    shared = " shared" if "sharedfe" in name else ""
    return f"{prefix} {obs} {seq} {hidden}{shared}".strip()


def style_for(name: str) -> dict:
    style = {"linewidth": 1.8, "alpha": 0.95}
    if "obs60" in name:
        style["linestyle"] = "-."
    elif "sharedfe" in name:
        style["linestyle"] = "--"
    else:
        style["linestyle"] = "-"
    if "tseq" not in name:
        style["color"] = "black"
        style["linewidth"] = 2.6
        style["linestyle"] = "-"
    return style


def plot_runs(args: argparse.Namespace, run_dirs: list[Path]) -> Path:
    if not run_dirs:
        raise ValueError("No run directories with rollout_log.jsonl matched the requested filters.")

    rows_by_run = {run_dir: read_rollout_rows(run_dir) for run_dir in run_dirs}
    metrics = list(args.metrics)

    fig_height = max(5.0, 3.8 * len(metrics))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13.5, fig_height), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for run_dir in run_dirs:
            xs, ys = metric_xy(rows_by_run[run_dir], metric, args.max_steps)
            if not xs:
                continue
            label = short_label(run_dir.name)
            style = style_for(run_dir.name)
            if args.show_raw and args.smooth_window > 1:
                raw_style = dict(style)
                raw_style.update({"linewidth": 0.7, "alpha": 0.18})
                ax.plot(xs, ys, **raw_style)
            ax.plot(xs, rolling_mean(ys, args.smooth_window), label=label, **style)

        ylabel = METRIC_LABELS.get(metric, metric)
        if args.smooth_window > 1:
            ylabel = f"{ylabel} ({args.smooth_window}-rollout mean)"
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Env steps")
    fig.suptitle(args.title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.81, 0.5),
        bbox_transform=fig.transFigure,
        frameon=False,
        ncol=args.legend_cols,
        title="Runs",
        fontsize="small",
        title_fontsize="small",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return args.output


def main() -> None:
    args = parse_args()
    run_dirs = discover_runs(args)
    out_path = plot_runs(args, run_dirs)
    print(f"Plotted {len(run_dirs)} runs to {out_path}")


if __name__ == "__main__":
    main()
