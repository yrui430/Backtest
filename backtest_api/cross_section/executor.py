from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest_api.cross_section.config import (
    CrossSectionExecutionSpec,
    CrossSectionLabelSpec,
)


def assign_quantile_groups(signals: pd.Series, n_groups: int) -> pd.Series:
    """Assign stocks to quantile groups based on signal ranking.

    Group 1 = lowest signals, group n = highest signals.
    """
    ranks = signals.rank(method="first")
    n = len(signals)
    groups = pd.Series(
        np.ceil(ranks / n * n_groups).astype(int).clip(1, n_groups),
        index=signals.index,
    )
    return groups


def compute_weights_group(
    signals: pd.Series,
    groups: pd.Series,
    n_groups: int,
    direction: str,
    mode: str,
    weight_func: Optional[Callable] = None,
) -> pd.Series:
    """Compute portfolio weights using group-based method.

    long_short: long group = weight sum 1, short group = weight sum -1, net = 0.
    long_only: long group = weight sum 1, no short.
    """
    weights = pd.Series(0.0, index=signals.index)

    if direction == "momentum":
        long_group = n_groups
        short_group = 1
    else:
        long_group = 1
        short_group = n_groups

    # Long leg
    long_mask = groups == long_group
    if long_mask.any():
        if weight_func is not None:
            long_w = weight_func(signals[long_mask])
        else:
            long_w = pd.Series(1.0 / long_mask.sum(), index=signals[long_mask].index)
        weights[long_mask] = long_w

    # Short leg (only for long_short)
    if mode == "long_short":
        short_mask = groups == short_group
        if short_mask.any():
            if weight_func is not None:
                short_w = -weight_func(signals[short_mask])
            else:
                short_w = pd.Series(
                    -1.0 / short_mask.sum(), index=signals[short_mask].index
                )
            weights[short_mask] = short_w

    return weights


def compute_weights_money(
    signals: pd.Series,
    direction: str,
    mode: str,
) -> pd.Series:
    """Compute portfolio weights proportional to signal magnitude."""
    weights = pd.Series(0.0, index=signals.index)

    if direction == "mean_reversion":
        signals = -signals

    pos_mask = signals > 0
    neg_mask = signals < 0

    if mode == "long_short":
        if pos_mask.any():
            pos_sum = signals[pos_mask].sum()
            if pos_sum > 0:
                weights[pos_mask] = signals[pos_mask] / pos_sum
        if neg_mask.any():
            neg_sum = signals[neg_mask].abs().sum()
            if neg_sum > 0:
                weights[neg_mask] = signals[neg_mask] / neg_sum
    elif mode == "long_only":
        if pos_mask.any():
            pos_sum = signals[pos_mask].sum()
            if pos_sum > 0:
                weights[pos_mask] = signals[pos_mask] / pos_sum

    return weights


def execute_cross_section_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    spec: CrossSectionExecutionSpec,
    label_spec: CrossSectionLabelSpec,
    stock_col: str,
    date_col: str,
    price_col: str,
    display_mode: str = "long_short",
    returns_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Execute the cross-section backtest: signal -> weight -> portfolio PnL.

    Args:
        returns_df: If provided, use pre-computed returns instead of computing
            from price_df. Must have columns [stock_col, date_col, '_return'].
            Used in label mode where labels are the returns.

    Returns dict with:
      - portfolio_gross_pnl: pd.Series (per bar)
      - portfolio_net_pnl: pd.Series (per bar)
      - weights: pd.DataFrame (stock_col, date_col, weight, group)
      - group_returns: pd.DataFrame (per bar, per group return)
      - timestamps: sorted unique dates
      - fees: pd.Series
    """
    lag = label_spec.lag

    if returns_df is not None:
        # Label mode: use pre-computed returns
        merged = signal_df[[stock_col, date_col, "signal"]].merge(
            returns_df[[stock_col, date_col, "_return"]],
            on=[stock_col, date_col],
            how="inner",
        )
    else:
        # Raw mode: compute returns from prices
        merged = signal_df[[stock_col, date_col, "signal"]].merge(
            price_df[[stock_col, date_col, price_col]],
            on=[stock_col, date_col],
            how="inner",
        )
        merged = merged.sort_values([stock_col, date_col])
        merged["_return"] = merged.groupby(stock_col)[price_col].pct_change()

    dates = sorted(merged[date_col].unique())
    n_dates = len(dates)

    # Build return lookup: (date, stock) -> return
    return_lookup = {}
    for _, row in merged.iterrows():
        return_lookup[(row[date_col], row[stock_col])] = row["_return"]

    # Compute weights per cross-section
    all_weights = []
    for date_val, group in merged.groupby(date_col):
        signals = group.set_index(stock_col)["signal"]

        groups = assign_quantile_groups(signals, spec.n_groups)

        if spec.weight_method == "group":
            w = compute_weights_group(
                signals, groups, spec.n_groups,
                direction=spec.signal_direction,
                mode=display_mode,
                weight_func=spec.group_weight_func,
            )
        else:
            w = compute_weights_money(signals, spec.signal_direction, display_mode)

        for s in signals.index:
            all_weights.append({
                stock_col: s,
                date_col: date_val,
                "weight": w[s],
                "group": groups[s],
            })

    weights_df = pd.DataFrame(all_weights)

    # Build weight lookup: date -> {stock: weight}
    weight_by_date = {}
    group_by_date = {}
    for date_val, grp in weights_df.groupby(date_col):
        weight_by_date[date_val] = dict(zip(grp[stock_col], grp["weight"]))
        group_by_date[date_val] = dict(zip(grp[stock_col], grp["group"]))

    # Compute portfolio PnL (with lag)
    portfolio_gross = np.zeros(n_dates)
    portfolio_fees = np.zeros(n_dates)
    group_return_data = {g: np.zeros(n_dates) for g in range(1, spec.n_groups + 1)}

    # Track group stock counts per date for averaging
    group_stock_counts = {g: np.zeros(n_dates) for g in range(1, spec.n_groups + 1)}

    prev_weights = {}

    for i, date_val in enumerate(dates):
        # Portfolio return: apply weights from lag bars ago
        if i >= lag:
            weight_date = dates[i - lag]
            wt_dict = weight_by_date.get(weight_date, {})
            gt_dict = group_by_date.get(weight_date, {})

            stocks_today = merged[merged[date_col] == date_val][stock_col].unique()
            for s in stocks_today:
                ret = return_lookup.get((date_val, s), np.nan)
                if np.isnan(ret):
                    continue
                w = wt_dict.get(s, 0.0)
                portfolio_gross[i] += w * ret

                # Accumulate group returns
                g = gt_dict.get(s, 0)
                if 1 <= g <= spec.n_groups:
                    group_return_data[g][i] += ret
                    group_stock_counts[g][i] += 1

        # Average group returns
        for g in range(1, spec.n_groups + 1):
            if group_stock_counts[g][i] > 0:
                group_return_data[g][i] /= group_stock_counts[g][i]

        # Fee: based on weight changes
        curr_weights = weight_by_date.get(date_val, {})
        fee = 0.0
        all_stocks = set(curr_weights.keys()) | set(prev_weights.keys())
        for s in all_stocks:
            delta = abs(curr_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
            fee += spec.fee_rate * delta
        portfolio_fees[i] = fee
        prev_weights = curr_weights

    portfolio_net = portfolio_gross - portfolio_fees

    group_returns = pd.DataFrame({"date": dates})
    for g in range(1, spec.n_groups + 1):
        group_returns[f"group_{g}"] = group_return_data[g]

    return {
        "portfolio_gross_pnl": pd.Series(portfolio_gross, index=range(n_dates)),
        "portfolio_net_pnl": pd.Series(portfolio_net, index=range(n_dates)),
        "weights": weights_df,
        "group_returns": group_returns,
        "timestamps": pd.Series(dates),
        "fees": pd.Series(portfolio_fees, index=range(n_dates)),
    }
