import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def infer_player_index(path: Path):
    match = re.search(r"player_(\d+)_episode_\d+\.csv$", path.name)
    if match:
        return int(match.group(1))
    return None


def infer_algo_name(path: Path):
    match = re.match(r"([A-Za-z0-9_]+)_player_\d+_episode_\d+\.csv$", path.name)
    if match:
        return match.group(1)
    return None


def build_tick_summary(df: pd.DataFrame, player_index: int) -> pd.DataFrame:
    player_df = df[df["advertiserNumber"].astype(int) == int(player_index)].copy()
    if player_df.empty:
        raise ValueError(f"No rows found for advertiserNumber/player_index={player_index}")
    player_df["real_spend"] = player_df["cost"] * player_df["isExposed"]

    tick_df = (
        player_df.groupby("timeStepIndex", as_index=False)
        .agg(
            spend=("real_spend", "sum"),
            conversions=("conversionAction", "sum"),
            exposures=("isExposed", "sum"),
            opportunities=("pvIndex", "count"),
            avg_bid=("bid", "mean"),
            avg_market=("leastWinningCost", "mean"),
            start_remaining_budget=("remainingBudget", "mean"),
            avg_pvalue=("pValue", "mean"),
            budget=("budget", "first"),
            cpa_constraint=("CPAConstraint", "first"),
        )
        .sort_values("timeStepIndex")
    )

    tick_df["cum_spend"] = tick_df["spend"].cumsum()
    tick_df["end_remaining_budget"] = tick_df["start_remaining_budget"] - tick_df["spend"]
    tick_df["cum_conversions"] = tick_df["conversions"].cumsum()
    tick_df["win_rate"] = tick_df["exposures"] / tick_df["opportunities"].clip(lower=1)
    tick_df["conversion_rate_given_exposure"] = (
        tick_df["conversions"] / tick_df["exposures"].clip(lower=1)
    )
    return tick_df


def compute_summary_metrics(tick_df: pd.DataFrame) -> dict:
    reward = float(tick_df["conversions"].sum())
    all_cost = float(tick_df["spend"].sum())
    budget = float(tick_df["budget"].iloc[0])
    cpa_constraint = float(tick_df["cpa_constraint"].iloc[0])
    all_exposures = float(tick_df["exposures"].sum())
    all_opportunities = float(tick_df["opportunities"].sum())

    if reward > 0:
        cpa = all_cost / reward
    elif all_cost > 0:
        cpa = float("inf")
    else:
        cpa = 0.0

    penalty = 1.0
    if math.isfinite(cpa) and cpa > cpa_constraint:
        penalty = (cpa_constraint / (cpa + 1e-10)) ** 2
    elif not math.isfinite(cpa):
        penalty = 0.0

    score = penalty * reward
    cpa_exceedance_rate = (cpa - cpa_constraint) / (cpa_constraint + 1e-10) if math.isfinite(cpa) else float("inf")
    budget_consumer_ratio = all_cost / budget if budget > 0 else 0.0
    win_rate = all_exposures / all_opportunities if all_opportunities > 0 else 0.0

    return {
        "score": score,
        "reward": reward,
        "all_cost": all_cost,
        "budget": budget,
        "cpa": cpa,
        "cpa_constraint": cpa_constraint,
        "cpa_exceedance_rate": cpa_exceedance_rate,
        "budget_consumer_ratio": budget_consumer_ratio,
        "win_rate": win_rate,
    }


def make_figure(
    tick_df: pd.DataFrame,
    output_path: Path,
    player_index: int,
    source_name: str,
    algo_name: str,
):
    summary = compute_summary_metrics(tick_df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"{algo_name} Evaluation Summary: player {player_index}\n{source_name}", fontsize=13)

    x = tick_df["timeStepIndex"]

    axes[0, 0].plot(x, tick_df["cum_spend"], marker="o", label="Cumulative real spend")
    axes[0, 0].plot(x, tick_df["end_remaining_budget"], marker="o", label="Remaining budget after tick")
    axes[0, 0].axhline(tick_df["budget"].iloc[0], color="tab:gray", linestyle="--", linewidth=1, label="Initial budget")
    axes[0, 0].set_title("Budget Pacing")
    axes[0, 0].set_xlabel("Tick")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, tick_df["cum_conversions"], marker="o", color="tab:green")
    axes[0, 1].set_title("Cumulative Conversions")
    axes[0, 1].set_xlabel("Tick")
    axes[0, 1].grid(alpha=0.3)
    cpa_text = f"{summary['cpa']:.2f}" if math.isfinite(summary["cpa"]) else "inf"
    cpa_exceed_text = (
        f"{summary['cpa_exceedance_rate']:.1%}"
        if math.isfinite(summary["cpa_exceedance_rate"])
        else "inf"
    )
    summary_text = (
        f"Episode score: {summary['score']:.2f}\n"
        f"Reward: {summary['reward']:.0f}\n"
        f"CPA: {cpa_text} / {summary['cpa_constraint']:.2f}\n"
        f"CPA exceed: {cpa_exceed_text}\n"
        f"Spend: {summary['all_cost']:.1f} / {summary['budget']:.1f}\n"
        f"Budget used: {summary['budget_consumer_ratio']:.1%}\n"
        f"Win rate: {summary['win_rate']:.1%}"
    )
    axes[0, 1].text(
        0.02,
        0.98,
        summary_text,
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    axes[1, 0].plot(x, tick_df["avg_bid"], marker="o", label="Average bid")
    axes[1, 0].plot(x, tick_df["avg_market"], marker="o", label="Average least winning cost")
    axes[1, 0].set_title("Bid vs Market Price")
    axes[1, 0].set_xlabel("Tick")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, tick_df["win_rate"], marker="o", label="Exposure ratio")
    axes[1, 1].plot(
        x,
        tick_df["conversion_rate_given_exposure"],
        marker="o",
        label="Conversion rate | exposure",
    )
    axes[1, 1].set_title("Auction Efficiency")
    axes[1, 1].set_xlabel("Tick")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot AuctionNet evaluation log for one player.")
    parser.add_argument("--input", required=True, help="Path to one evaluation log CSV.")
    parser.add_argument(
        "--player-index",
        type=int,
        default=None,
        help="Advertiser/player index to plot. If omitted, infer from filename.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: figures/<input_stem>.png",
    )
    parser.add_argument(
        "--algo-name",
        default=None,
        help="Algorithm name for the figure title. If omitted, try to infer from the input filename.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    player_index = args.player_index if args.player_index is not None else infer_player_index(input_path)
    if player_index is None:
        raise ValueError("Could not infer player index from filename. Pass --player-index explicitly.")
    algo_name = args.algo_name if args.algo_name is not None else infer_algo_name(input_path)
    if algo_name is None:
        algo_name = "Strategy"

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path("figures") / f"{input_path.stem}.png"
    )

    df = pd.read_csv(input_path)
    tick_df = build_tick_summary(df, player_index)
    make_figure(tick_df, output_path, player_index, input_path.name, algo_name)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
