"""60-dim state builder for the IQL60 extension.

This is inspired by the PPO/OIL obs_60_keys idea, but adapted to the data that
AuctionNet's offline environment actually exposes at training/eval time:
current pValues/pValueSigmas/leastWinningCost plus historical bids, wins,
costs, conversions, and least-winning-cost arrays.

It is intentionally separate from the existing 16-dim IQL state so the current
main-line implementation remains untouched.
"""

from __future__ import annotations

import numpy as np

STATE_DIM = 60
NUM_TICK = 48
EPS = 1e-8

FEATURE_NAMES = [
    "time_left",
    "budget_left",
    "budget",
    "target_cpa",
    "category",
    "current_pvalue_mean",
    "current_pvalue_std",
    "current_pvalue_p50",
    "current_pvalue_p90",
    "current_pvalue_p99",
    "current_sigma_mean",
    "current_sigma_std",
    "current_lwc_mean",
    "current_lwc_std",
    "current_lwc_p10",
    "current_lwc_p01",
    "current_pv_over_lwc_mean",
    "current_pv_over_lwc_p50",
    "current_pv_over_lwc_p90",
    "current_pv_over_lwc_p99",
    "current_pv_num",
    "hist_bid_mean",
    "hist_bid_std",
    "hist_lwc_mean",
    "hist_lwc_std",
    "hist_pvalue_mean",
    "hist_pvalue_std",
    "hist_conversion_mean",
    "hist_win_mean",
    "hist_cost_mean",
    "hist_cost_std",
    "hist_pv_over_lwc_mean",
    "hist_pv_num_mean",
    "hist_pv_num_total",
    "hist_cpa_proxy",
    "last_bid_mean",
    "last_bid_std",
    "last_lwc_mean",
    "last_lwc_std",
    "last_pvalue_mean",
    "last_pvalue_std",
    "last_conversion_mean",
    "last_win_mean",
    "last_cost_mean",
    "last_cost_std",
    "last_pv_over_lwc_mean",
    "last_pv_num",
    "last3_bid_mean",
    "last3_bid_std",
    "last3_lwc_mean",
    "last3_lwc_std",
    "last3_pvalue_mean",
    "last3_pvalue_std",
    "last3_conversion_mean",
    "last3_win_mean",
    "last3_cost_mean",
    "last3_cost_std",
    "last3_pv_over_lwc_mean",
    "last3_pv_num_total",
    "spent_ratio",
]


def _to_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_mean(x) -> float:
    arr = _to_float_array(x)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _safe_std(x) -> float:
    arr = _to_float_array(x)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr))


def _safe_percentile(x, q: float) -> float:
    arr = _to_float_array(x)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _mean_of_means(arrays: list[np.ndarray]) -> float:
    if not arrays:
        return 0.0
    return float(np.mean([_safe_mean(arr) for arr in arrays]))


def _mean_of_stds(arrays: list[np.ndarray]) -> float:
    if not arrays:
        return 0.0
    return float(np.mean([_safe_std(arr) for arr in arrays]))


def _last_array(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    return _to_float_array(arrays[-1])


def _last_n_arrays(arrays: list[np.ndarray], n: int) -> list[np.ndarray]:
    if not arrays:
        return []
    return [_to_float_array(arr) for arr in arrays[-n:]]


def _sum_lengths(arrays: list[np.ndarray]) -> float:
    return float(sum(len(arr) for arr in arrays)) if arrays else 0.0


def _mean_lengths(arrays: list[np.ndarray]) -> float:
    if not arrays:
        return 0.0
    return float(np.mean([len(arr) for arr in arrays]))


def _history_cpa_proxy(cost_arrays: list[np.ndarray], conversion_arrays: list[np.ndarray]) -> float:
    if not cost_arrays:
        return 0.0
    total_cost = float(sum(np.sum(arr) for arr in cost_arrays))
    total_conv = float(sum(np.sum(arr) for arr in conversion_arrays))
    if total_conv <= 0:
        return total_cost
    return total_cost / total_conv


def _extract_history_views(
    historyPValueInfo,
    historyBid,
    historyAuctionResult,
    historyImpressionResult,
    historyLeastWinningCost,
):
    history_pvalue = [_to_float_array(row[:, 0]) for row in historyPValueInfo]
    history_bid = [_to_float_array(row) for row in historyBid]
    history_win = [_to_float_array(row[:, 0]) for row in historyAuctionResult]
    history_cost = [_to_float_array(row[:, 2]) for row in historyAuctionResult]
    history_conv = [_to_float_array(row[:, 1]) for row in historyImpressionResult]
    history_lwc = [_to_float_array(row) for row in historyLeastWinningCost]
    history_ratio = [
        pvalue / np.maximum(lwc, EPS)
        for pvalue, lwc in zip(history_pvalue, history_lwc)
    ]
    return {
        "pvalue": history_pvalue,
        "bid": history_bid,
        "win": history_win,
        "cost": history_cost,
        "conversion": history_conv,
        "lwc": history_lwc,
        "ratio": history_ratio,
    }


def build_state_60_from_current(
    timeStepIndex,
    pValues,
    pValueSigmas,
    currentLeastWinningCost,
    historyPValueInfo,
    historyBid,
    historyAuctionResult,
    historyImpressionResult,
    historyLeastWinningCost,
    budget,
    remaining_budget,
    target_cpa,
    category,
):
    pvalues = _to_float_array(pValues)
    sigmas = _to_float_array(pValueSigmas)
    current_lwc = _to_float_array(currentLeastWinningCost)
    current_ratio = pvalues / np.maximum(current_lwc, EPS)

    history = _extract_history_views(
        historyPValueInfo=historyPValueInfo,
        historyBid=historyBid,
        historyAuctionResult=historyAuctionResult,
        historyImpressionResult=historyImpressionResult,
        historyLeastWinningCost=historyLeastWinningCost,
    )

    last3_bid = _last_n_arrays(history["bid"], 3)
    last3_lwc = _last_n_arrays(history["lwc"], 3)
    last3_pvalue = _last_n_arrays(history["pvalue"], 3)
    last3_conversion = _last_n_arrays(history["conversion"], 3)
    last3_win = _last_n_arrays(history["win"], 3)
    last3_cost = _last_n_arrays(history["cost"], 3)
    last3_ratio = _last_n_arrays(history["ratio"], 3)

    spent_ratio = (budget - remaining_budget) / budget if budget > 0 else 0.0

    state = np.array(
        [
            (NUM_TICK - timeStepIndex) / NUM_TICK,
            remaining_budget / budget if budget > 0 else 0.0,
            float(budget),
            float(target_cpa),
            float(category),
            _safe_mean(pvalues),
            _safe_std(pvalues),
            _safe_percentile(pvalues, 50),
            _safe_percentile(pvalues, 90),
            _safe_percentile(pvalues, 99),
            _safe_mean(sigmas),
            _safe_std(sigmas),
            _safe_mean(current_lwc),
            _safe_std(current_lwc),
            _safe_percentile(current_lwc, 10),
            _safe_percentile(current_lwc, 1),
            _safe_mean(current_ratio),
            _safe_percentile(current_ratio, 50),
            _safe_percentile(current_ratio, 90),
            _safe_percentile(current_ratio, 99),
            float(pvalues.size),
            _mean_of_means(history["bid"]),
            _mean_of_stds(history["bid"]),
            _mean_of_means(history["lwc"]),
            _mean_of_stds(history["lwc"]),
            _mean_of_means(history["pvalue"]),
            _mean_of_stds(history["pvalue"]),
            _mean_of_means(history["conversion"]),
            _mean_of_means(history["win"]),
            _mean_of_means(history["cost"]),
            _mean_of_stds(history["cost"]),
            _mean_of_means(history["ratio"]),
            _mean_lengths(history["bid"]),
            _sum_lengths(history["bid"]),
            _history_cpa_proxy(history["cost"], history["conversion"]),
            _safe_mean(_last_array(history["bid"])),
            _safe_std(_last_array(history["bid"])),
            _safe_mean(_last_array(history["lwc"])),
            _safe_std(_last_array(history["lwc"])),
            _safe_mean(_last_array(history["pvalue"])),
            _safe_std(_last_array(history["pvalue"])),
            _safe_mean(_last_array(history["conversion"])),
            _safe_mean(_last_array(history["win"])),
            _safe_mean(_last_array(history["cost"])),
            _safe_std(_last_array(history["cost"])),
            _safe_mean(_last_array(history["ratio"])),
            float(len(_last_array(history["bid"]))),
            _mean_of_means(last3_bid),
            _mean_of_stds(last3_bid),
            _mean_of_means(last3_lwc),
            _mean_of_stds(last3_lwc),
            _mean_of_means(last3_pvalue),
            _mean_of_stds(last3_pvalue),
            _mean_of_means(last3_conversion),
            _mean_of_means(last3_win),
            _mean_of_means(last3_cost),
            _mean_of_stds(last3_cost),
            _mean_of_means(last3_ratio),
            _sum_lengths(last3_bid),
            float(spent_ratio),
        ],
        dtype=np.float32,
    )
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM} features, got {state.shape[0]}")
    return state


def apply_normalize(state, normalize_dict):
    state = state.copy()
    for key, value in normalize_dict.items():
        min_value = value["min"]
        max_value = value["max"]
        state[key] = (
            (state[key] - min_value) / (max_value - min_value + 0.01)
            if max_value >= min_value
            else 0.0
        )
    return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
