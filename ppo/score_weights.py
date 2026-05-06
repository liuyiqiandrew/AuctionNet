"""Compute per-advertiser sampling weights from historical NeurIPS scores.

For each (period, advertiser) trajectory we have in the raw period parquet,
compute the NeurIPS score the original logged policy achieved:

    score = conversions * min(1, (CPAConstraint / realCPA)^2)
    realCPA = sum(cost) / sum(conversions)   (0 if no conversions)

Then convert per-advertiser scores to a sampling distribution via:

    p_softmax = softmax(score / T)
    p = alpha * p_softmax + (1 - alpha) * uniform

T -> 0:  hard max (only top advertiser sampled)
T -> inf: uniform
alpha=1: pure softmax;  alpha=0: pure uniform
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_period_advertiser_scores(period: int, raw_data_dir: Path) -> pd.DataFrame:
    path = Path(raw_data_dir) / f"period-{period}.parquet"
    df = pd.read_parquet(
        path, columns=["advertiserNumber", "CPAConstraint", "cost", "conversionAction"]
    )
    grp = df.groupby("advertiserNumber").agg(
        conversions=("conversionAction", "sum"),
        cost=("cost", "sum"),
        cpa_constraint=("CPAConstraint", "first"),
    ).reset_index()
    grp["real_cpa"] = np.where(grp["conversions"] > 0, grp["cost"] / grp["conversions"], 0.0)
    cpa_coeff = np.where(
        grp["real_cpa"] > 0,
        np.minimum(1.0, (grp["cpa_constraint"] / grp["real_cpa"]) ** 2),
        0.0,
    )
    grp["score"] = grp["conversions"] * cpa_coeff
    return grp


def compute_advertiser_weights(
    period: int, raw_data_dir: Path, temperature: float, alpha: float, advertiser_list: list
) -> np.ndarray:
    scores_df = compute_period_advertiser_scores(period, raw_data_dir)
    scores_df = scores_df.set_index("advertiserNumber").loc[
        [float(a) for a in advertiser_list]
    ]
    scores = scores_df["score"].to_numpy(dtype=np.float64)

    z = scores / max(temperature, 1e-9)
    z -= z.max()
    p_soft = np.exp(z)
    p_soft /= p_soft.sum()
    p_uniform = np.ones_like(p_soft) / len(p_soft)
    return alpha * p_soft + (1.0 - alpha) * p_uniform


def effective_sample_size(weights: np.ndarray) -> float:
    return 1.0 / float(np.sum(weights ** 2))
