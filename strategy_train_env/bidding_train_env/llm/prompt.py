"""Prompt templates + response parser for the LLM bidding agent.

Action contract (matches `online_env.BiddingEnv` with `act_keys=["pvalue"]`):
    bid_i = exp(alpha) * pvalues_i * target_cpa,   alpha in [-10, 10]
    alpha = 0 is the neutral baseline (bid = pvalue * target_cpa).
"""

from __future__ import annotations

import re

ACTION_CLIP = (-10.0, 10.0)

SYSTEM_PROMPT = """You are an autonomous bidding agent in a 48-tick ad-auction episode.

Each tick you observe the current state and must output ONE decision: a log-bid-multiplier alpha (Greek letter alpha, written as `alpha` in your answer).

Action contract
---------------
Your bids for the current tick's impressions are computed as:
    bid_i = exp(alpha) * pvalue_i * target_cpa
Here `pvalue_i` is the per-impression conversion probability and `target_cpa` is the campaign CPA target. alpha must be a real number in [-10, 10].

- alpha = 0  -> bid = pvalue * target_cpa   (neutral baseline, same as an untrained strategy)
- alpha > 0  -> bid higher than baseline    (win more slots, spend faster)
- alpha < 0  -> bid lower than baseline     (win fewer slots, spend slower)

Objective
---------
Maximize `score = min(1, (target_cpa / realized_cpa)^2) * conversions` at the end of the episode. Two implicit constraints:
1. Total cost must not exceed `budget`. If you overspend, winning bids get dropped at random.
2. Realized CPA = total_cost / total_conversions should stay <= target_cpa; score is penalized quadratically whenever realized CPA exceeds the target.

Rules of thumb
--------------
- If the budget is under-spent relative to time elapsed (budget_used << time_used), raise alpha to bid more aggressively.
- If you are over-spending (budget_used >> time_used) or realized CPA is above target, lower alpha.
- Current-tick pvalues much higher than the recent least-winning-cost is a strong signal to bid up; much lower is a signal to bid down.

Output format
-------------
Respond with EXACTLY one line, nothing else:

<alpha>X</alpha>

where X is a decimal number in [-10, 10]. Do not add any other text, reasoning, punctuation, code fences, or explanation."""


_ALPHA_RE = re.compile(r"<\s*alpha\s*>\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*<\s*/\s*alpha\s*>")


def _fmt(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def build_user_message(state_dict: dict, tick: int, episode_length: int = 48) -> dict:
    """Render the BiddingEnv state dict into a compact human-readable user turn.

    Uses a subset of obs_16-style fields, plus absolute budget / target_cpa so the
    LLM can reason about pacing in concrete units. Total length is well under 300 tokens.
    """
    s = state_dict
    budget_used = 1.0 - float(s.get("budget_left", 0.0))
    time_used = 1.0 - float(s.get("time_left", 0.0))
    target_cpa = float(s.get("cpa", 0.0))
    realized_cpa = float(s.get("total_cpa", 0.0))
    cpa_ratio = (realized_cpa / target_cpa) if target_cpa > 0 else 0.0

    lines = [
        f"Tick {tick}/{episode_length}. Budget used {_fmt(budget_used)} of {_fmt(s.get('budget'), 0)}; "
        f"time used {_fmt(time_used)}.",
        f"Target CPA {_fmt(target_cpa, 2)}. Realized CPA so far {_fmt(realized_cpa, 2)} "
        f"(ratio realized/target = {_fmt(cpa_ratio, 2)}).",
        f"Total conversions so far: {_fmt(s.get('total_conversions', 0.0), 1)}. "
        f"Category {int(s.get('category', 0))}.",
        "",
        "This tick's impression batch:",
        f"  count = {int(s.get('current_pv_num', 0))}, "
        f"mean pvalue = {_fmt(s.get('current_pvalues_mean', 0.0), 5)}, "
        f"p90 = {_fmt(s.get('current_pvalues_90_pct', 0.0), 5)}, "
        f"p99 = {_fmt(s.get('current_pvalues_99_pct', 0.0), 5)}.",
        "",
        "Recent-history aggregates (None if the episode just started):",
        f"  last-tick bid mean      = {_fmt(s.get('last_bid_mean'))}",
        f"  last-3 bid mean         = {_fmt(s.get('last_three_bid_mean'))}",
        f"  historical bid mean     = {_fmt(s.get('historical_bid_mean'))}",
        f"  last-3 win rate         = {_fmt(s.get('last_three_bid_success_mean'))}",
        f"  historical win rate     = {_fmt(s.get('historical_bid_success_mean'))}",
        f"  last-3 least-winning-cost mean = {_fmt(s.get('last_three_least_winning_cost_mean'))}",
        f"  historical LWC mean            = {_fmt(s.get('historical_least_winning_cost_mean'))}",
        f"  last-3 conversions/tick = {_fmt(s.get('last_three_conversion_mean'))}",
        f"  historical pvalue mean  = {_fmt(s.get('historical_pvalues_mean'))}",
        "",
        "Decide alpha now. Remember: output only `<alpha>X</alpha>` with a single number X in [-10, 10].",
    ]
    return {"role": "user", "content": "\n".join(lines)}


def parse_alpha(
    text: str,
    clip: tuple[float, float] = ACTION_CLIP,
    default: float = 0.0,
) -> tuple[float, bool]:
    """Extract alpha from the model's response.

    Returns (alpha, is_malformed). Strict regex `<alpha>X</alpha>`; any parse or
    numeric failure falls back to `default` with `is_malformed=True`. Clamps to `clip`.
    """
    if not isinstance(text, str):
        return default, True
    m = _ALPHA_RE.search(text)
    if m is None:
        return default, True
    try:
        val = float(m.group(1))
    except ValueError:
        return default, True
    lo, hi = clip
    clamped = max(lo, min(hi, val))
    return clamped, False
