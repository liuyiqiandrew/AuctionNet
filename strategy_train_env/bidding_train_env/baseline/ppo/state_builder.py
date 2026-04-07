import numpy as np

STATE_DIM = 16
NUM_TICK = 48


def build_state(timeStepIndex, pValues, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
                budget, remaining_budget):
    """Construct the 16-dim state used by every offline baseline + the PPO wrapper.

    Mirrors the inline formula in iql_bidding_strategy.py:46-92 so that
    training-time and inference-time states are bit-for-bit identical.
    """
    time_left = (NUM_TICK - timeStepIndex) / NUM_TICK
    budget_left = remaining_budget / budget if budget > 0 else 0

    history_xi = [r[:, 0] for r in historyAuctionResult]
    history_pValue = [r[:, 0] for r in historyPValueInfo]
    history_conversion = [r[:, 1] for r in historyImpressionResult]

    def _safe_mean(x):
        # np.mean of an empty array is nan; treat as 0 to avoid poisoning the state.
        x = np.asarray(x)
        if x.size == 0:
            return 0.0
        m = float(np.mean(x))
        return 0.0 if not np.isfinite(m) else m

    def _mean_of_means(h):
        if not h:
            return 0.0
        vals = [_safe_mean(x) for x in h]
        return float(np.mean(vals)) if vals else 0.0

    def _last3(h):
        if not h:
            return 0.0
        sub = h[-3:]
        vals = [_safe_mean(x) for x in sub]
        return float(np.mean(vals)) if vals else 0.0

    historical_pv_num_total = float(sum(len(b) for b in historyBid)) if historyBid else 0.0
    last_three_pv_num_total = float(
        sum(len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex))
    ) if historyBid else 0.0

    pv_arr = np.asarray(pValues)
    cur_pvalue_mean = float(np.mean(pv_arr)) if pv_arr.size > 0 else 0.0
    if not np.isfinite(cur_pvalue_mean):
        cur_pvalue_mean = 0.0

    state = np.array([
        time_left,
        budget_left,
        _mean_of_means(historyBid),
        _last3(historyBid),
        _mean_of_means(historyLeastWinningCost),
        _mean_of_means(history_pValue),
        _mean_of_means(history_conversion),
        _mean_of_means(history_xi),
        _last3(historyLeastWinningCost),
        _last3(history_pValue),
        _last3(history_conversion),
        _last3(history_xi),
        cur_pvalue_mean,
        float(pv_arr.size),
        last_three_pv_num_total,
        historical_pv_num_total,
    ], dtype=np.float32)
    # Final safety net.
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    return state


def apply_normalize(state, normalize_dict):
    """Min-max normalize the indices recorded in normalize_dict (typically [13,14,15])."""
    state = state.copy()
    for k, v in normalize_dict.items():
        lo, hi = v["min"], v["max"]
        state[k] = (state[k] - lo) / (hi - lo) if hi > lo else 0.0
    return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
