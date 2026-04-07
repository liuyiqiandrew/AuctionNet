import os
import argparse
import json
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MAIN_METRICS = [
    {
        "key": "continuous_raw_score",
        "title": "Continuous Raw Score",
        "ylabel": "Score",
        "mode": "max",
        "color": "tab:blue",
    },
    {
        "key": "continuous_reward",
        "title": "Continuous Reward",
        "ylabel": "Reward",
        "mode": "max",
        "color": "tab:green",
    },
    {
        "key": "cpa_exceedance_rate",
        "title": "CPA Exceedance Rate",
        "ylabel": "Rate",
        "mode": "min",
        "color": "tab:red",
    },
]

LOSS_METRICS = [
    {
        "key": "train_q_loss",
        "title": "Train Q Loss",
        "ylabel": "Loss",
        "color": "tab:purple",
    },
    {
        "key": "train_v_loss",
        "title": "Train V Loss",
        "ylabel": "Loss",
        "color": "tab:orange",
    },
    {
        "key": "train_a_loss",
        "title": "Train Actor Loss",
        "ylabel": "Loss",
        "color": "tab:brown",
    },
]


def resolve_path(path_str: str | None, default: Path) -> Path:
    if path_str is None:
        return default
    return Path(path_str).expanduser().resolve()


def load_training_curve(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find training curve CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError(f"Expected a 'step' column in {csv_path}")

    numeric_columns = [
        "step",
        "continuous_raw_score",
        "continuous_reward",
        "cpa_exceedance_rate",
        "budget_consumer_ratio",
        "sparse_raw_score",
        "sparse_reward",
        "num_groups",
        "train_q_loss",
        "train_v_loss",
        "train_a_loss",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["step"]).sort_values("step").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No checkpoint rows found in {csv_path}")
    return df


def load_best_metrics(best_json_path: Path) -> dict:
    if not best_json_path.exists():
        return {}
    with open(best_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def infer_best_metrics(df: pd.DataFrame) -> dict:
    inferred = {}
    for metric in MAIN_METRICS:
        key = metric["key"]
        if key not in df.columns:
            continue

        metric_df = df[["step", key]].dropna()
        if metric_df.empty:
            continue

        if metric["mode"] == "max":
            best_row = metric_df.loc[metric_df[key].idxmax()]
        else:
            best_row = metric_df.loc[metric_df[key].idxmin()]

        checkpoint_path = None
        if "checkpoint_path" in df.columns:
            checkpoint_matches = df.loc[df["step"] == best_row["step"], "checkpoint_path"]
            if not checkpoint_matches.empty:
                checkpoint_path = checkpoint_matches.iloc[0]

        inferred[key] = {
            "mode": metric["mode"],
            "value": float(best_row[key]),
            "step": int(best_row["step"]),
            "checkpoint_path": checkpoint_path,
        }
    return inferred


def format_metric_value(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def draw_metric_subplot(ax, df: pd.DataFrame, metric: dict, best_metrics: dict) -> None:
    key = metric["key"]
    ax.grid(alpha=0.3)
    ax.set_title(metric["title"])
    ax.set_ylabel(metric["ylabel"])

    if key not in df.columns:
        ax.text(0.5, 0.5, f"{key} not found", ha="center", va="center", transform=ax.transAxes)
        return

    metric_df = df[["step", key]].dropna()
    if metric_df.empty:
        ax.text(0.5, 0.5, f"{key} has no values", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot(
        metric_df["step"],
        metric_df[key],
        color=metric["color"],
        marker="o",
        markersize=4,
        linewidth=2,
    )

    if key == "cpa_exceedance_rate":
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)

    best_entry = best_metrics.get(key)
    if best_entry and best_entry.get("step") is not None and best_entry.get("value") is not None:
        best_step = int(best_entry["step"])
        best_value = float(best_entry["value"])
        ax.axvline(best_step, color="0.4", linestyle=":", linewidth=1)
        ax.scatter([best_step], [best_value], color="black", s=45, zorder=3)
        annotation = f"best step {best_step}\nvalue {best_value:.4f}"
        ax.annotate(
            annotation,
            xy=(best_step, best_value),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )


def make_main_figure(
    df: pd.DataFrame,
    best_metrics: dict,
    output_path: Path,
    run_dir: Path,
    title: str | None,
) -> None:
    fig, axes = plt.subplots(len(MAIN_METRICS), 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    figure_title = title or f"Checkpoint Metrics: {run_dir.name}"
    fig.suptitle(figure_title, fontsize=14)

    for ax, metric in zip(axes, MAIN_METRICS):
        draw_metric_subplot(ax, df, metric, best_metrics)

    axes[-1].set_xlabel("Checkpoint step")

    summary_lines = [
        f"Run dir: {run_dir}",
        f"Checkpoints plotted: {len(df)}",
        f"Last step: {int(df['step'].iloc[-1])}",
    ]
    if "checkpoint_path" in df.columns:
        summary_lines.append(f"Last checkpoint: {df['checkpoint_path'].iloc[-1]}")
    axes[0].text(
        1.01,
        1.0,
        "\n".join(summary_lines),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_loss_figure(df: pd.DataFrame, output_path: Path, run_dir: Path, title: str | None) -> bool:
    available_metrics = [metric for metric in LOSS_METRICS if metric["key"] in df.columns and not df[metric["key"]].dropna().empty]
    if not available_metrics:
        return False

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    if len(available_metrics) == 1:
        axes = [axes]

    figure_title = title or f"Training Losses: {run_dir.name}"
    fig.suptitle(figure_title, fontsize=14)

    for ax, metric in zip(axes, available_metrics):
        metric_df = df[["step", metric["key"]]].dropna()
        ax.plot(
            metric_df["step"],
            metric_df[metric["key"]],
            color=metric["color"],
            marker="o",
            markersize=4,
            linewidth=2,
        )
        ax.set_title(metric["title"])
        ax.set_ylabel(metric["ylabel"])
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Checkpoint step")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def print_summary(df: pd.DataFrame, best_metrics: dict) -> None:
    print(f"Loaded {len(df)} checkpoint rows from {int(df['step'].iloc[0])} to {int(df['step'].iloc[-1])}.")
    print("Best checkpoint summary:")
    for metric in MAIN_METRICS:
        key = metric["key"]
        best_entry = best_metrics.get(key)
        if not best_entry:
            print(f"  - {key}: not available")
            continue
        checkpoint_path = best_entry.get("checkpoint_path") or "n/a"
        print(
            f"  - {key}: step={best_entry.get('step')} "
            f"value={format_metric_value(best_entry.get('value'))} "
            f"checkpoint={checkpoint_path}"
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_run_dir = repo_root / "strategy_train_env" / "saved_model" / "IQLtest"

    parser = argparse.ArgumentParser(
        description="Plot AuctionNet training checkpoint metrics from training_curve.csv and best_metrics.json."
    )
    parser.add_argument(
        "--run-dir",
        default=str(default_run_dir),
        help="Directory containing training_curve.csv and best_metrics.json. Default: strategy_train_env/saved_model/IQLtest",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to training_curve.csv. Overrides --run-dir/training_curve.csv.",
    )
    parser.add_argument(
        "--best-json",
        default=None,
        help="Optional path to best_metrics.json. Overrides --run-dir/best_metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save figures. Default: <run-dir>/plots",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title prefix for the saved figures.",
    )
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir, default_run_dir)
    csv_path = resolve_path(args.csv, run_dir / "training_curve.csv")
    best_json_path = resolve_path(args.best_json, run_dir / "best_metrics.json")
    output_dir = resolve_path(args.output_dir, run_dir / "plots")

    df = load_training_curve(csv_path)
    best_metrics = load_best_metrics(best_json_path)
    if not best_metrics:
        best_metrics = infer_best_metrics(df)

    main_output = output_dir / "checkpoint_metrics.png"
    losses_output = output_dir / "training_losses.png"

    make_main_figure(df, best_metrics, main_output, run_dir, args.title)
    has_losses = make_loss_figure(df, losses_output, run_dir, args.title)

    print_summary(df, best_metrics)
    print(f"Saved metric plot to {main_output}")
    if has_losses:
        print(f"Saved loss plot to {losses_output}")
    else:
        print("Skipped loss plot because no training loss columns were found.")


if __name__ == "__main__":
    main()
