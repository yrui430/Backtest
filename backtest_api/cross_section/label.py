from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats


def compute_forward_returns(
    df: pd.DataFrame,
    stock_col: str,
    date_col: str,
    price_col: str,
    h: int,
    lag: int,
) -> pd.DataFrame:
    """Compute forward returns per stock: label(t) = price(t+lag+h) / price(t+lag) - 1.

    Groups by stock, shifts prices forward by (lag+h) and (lag).
    """
    result = df.copy()
    labels = []

    for stock, group in result.groupby(stock_col):
        g = group.sort_values(date_col).copy()
        prices = g[price_col].values.astype(np.float64)
        n = len(prices)
        label = np.full(n, np.nan)
        for t in range(n):
            entry = t + lag
            exit_ = t + lag + h
            if exit_ < n and entry < n:
                label[t] = prices[exit_] / prices[entry] - 1.0
        labels.append(pd.Series(label, index=g.index))

    result["label"] = pd.concat(labels)
    return result


def compute_ic_decay(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    stock_col: str,
    date_col: str,
    feature_col: str,
    price_col: str,
    h: int,
    decay_lags: List[int],
) -> pd.DataFrame:
    """Compute IC decay: for each lag, compute mean cross-sectional IC.

    IC_decay(lag) = mean_over_t[ corr_cs( feature(t), ret(t+lag : t+lag+h) ) ]
    """
    merged = feature_df[[stock_col, date_col, feature_col]].merge(
        price_df[[stock_col, date_col, price_col]],
        on=[stock_col, date_col],
        how="inner",
    )

    results = []
    for lag_val in decay_lags:
        ret_df = compute_forward_returns(
            merged[[stock_col, date_col, price_col]],
            stock_col=stock_col,
            date_col=date_col,
            price_col=price_col,
            h=h,
            lag=lag_val,
        )
        merged_with_ret = merged.copy()
        merged_with_ret["_ret"] = ret_df["label"]

        ic_list = []
        rank_ic_list = []
        for date_val, group in merged_with_ret.groupby(date_col):
            feat = group[feature_col].values
            ret = group["_ret"].values
            mask = ~(np.isnan(feat) | np.isnan(ret))
            if mask.sum() < 3:
                continue
            ic = float(np.corrcoef(feat[mask], ret[mask])[0, 1])
            rank_ic, _ = stats.spearmanr(feat[mask], ret[mask])
            ic_list.append(ic)
            rank_ic_list.append(float(rank_ic))

        ic_mean = np.nanmean(ic_list) if ic_list else np.nan
        rank_ic_mean = np.nanmean(rank_ic_list) if rank_ic_list else np.nan

        results.append({
            "lag": lag_val,
            "ic_mean": ic_mean,
            "rank_ic_mean": rank_ic_mean,
        })

    return pd.DataFrame(results)
