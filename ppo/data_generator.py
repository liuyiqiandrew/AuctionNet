"""Aggregate raw ad-opportunity rows into per-timestep pvalues and bids parquets.

Input:  a DataFrame with one row per (advertiser, impression, timestep) ad opportunity.
Output: three DataFrames written to disk as parquets:
    - pvalues_df:     one row per (period, timestep, advertiser), with lists of pValue/pValueSigma
    - bids_df:        one row per (period, timestep), with list-of-lists bid/cost/isExposed/advertiserNumber
                      over the top-3 bidders per impression, sorted ascending by bid
    - constraints_df: one row per (period, advertiser), with raw (budget, CPAConstraint)
"""

import numpy as np
import pandas as pd


def _reorder_list_of_lists(lst, positions):
    arr = np.array(lst)
    reordered = np.take_along_axis(arr, positions, axis=1)
    return reordered.tolist()


def generate_pvalue_df(data: pd.DataFrame) -> pd.DataFrame:
    out = data.groupby(
        ["deliveryPeriodIndex", "timeStepIndex", "advertiserNumber", "advertiserCategoryIndex"]
    ).agg(
        {"pValue": lambda x: x.tolist(),
         "pValueSigma": lambda x: x.tolist()}
    )
    out.reset_index(inplace=True)
    return out


def generate_bids_df(data: pd.DataFrame) -> pd.DataFrame:
    per_imp = (
        data[data["xi"] == 1]
        .groupby(["deliveryPeriodIndex", "timeStepIndex", "pvIndex"])
        .agg({"bid": list, "isExposed": list, "cost": list, "advertiserNumber": list})
        .reset_index()
    )
    per_ts = per_imp.groupby(["deliveryPeriodIndex", "timeStepIndex"]).agg(
        {"bid": list, "isExposed": list, "cost": list, "advertiserNumber": list}
    ).reset_index()

    # Sort each impression's top bids ascending so [:, -1] is the highest bid.
    per_ts["positions"] = per_ts.apply(lambda x: np.argsort(x.bid), axis=1)
    for col in ("bid", "isExposed", "cost", "advertiserNumber"):
        per_ts[col] = per_ts.apply(lambda x: _reorder_list_of_lists(x[col], x.positions), axis=1)
    per_ts.drop(columns=["positions"], inplace=True)
    return per_ts


def generate_advertiser_constraints_df(data: pd.DataFrame) -> pd.DataFrame:
    """Per-(period, advertiser) raw (budget, CPAConstraint) for bc_range='default' mode."""
    return data.groupby(
        ["deliveryPeriodIndex", "advertiserNumber"], as_index=False
    ).agg({"budget": "first", "CPAConstraint": "first"})
