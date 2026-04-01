from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

plt.rcParams["figure.max_open_warning"] = 0

from backtest_api.metrics import (
    annualized_return,
    total_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_recovery,
    turnover,
    information_ratio,
)


def build_cs_summary_table(
    pnl_before: pd.Series,
    pnl_after: pd.Series,
    positions: pd.Series,
    ic_series: np.ndarray,
    rank_ic_series: np.ndarray,
    bars_per_year: int = 252,
) -> pd.DataFrame:
    """Build performance summary table for cross-section backtest."""
    ic_mean = float(np.nanmean(ic_series))
    rank_ic_mean = float(np.nanmean(rank_ic_series))
    ir_val = information_ratio(pd.Series(ic_series).dropna())
    to = turnover(positions)

    rows = []
    for name, pnl in [("Before Fee", pnl_before), ("After Fee", pnl_after)]:
        rows.append({
            "Annualized Return": annualized_return(pnl, bars_per_year),
            "Total Return": total_return(pnl),
            "Volatility": volatility(pnl, bars_per_year),
            "Sharpe Ratio": sharpe_ratio(pnl, bars_per_year),
            "Sortino Ratio": sortino_ratio(pnl, bars_per_year),
            "Turnover": to,
            "Max Drawdown": max_drawdown(pnl),
            "Max Drawdown Recovery Time": max_drawdown_recovery(pnl),
            "IC": ic_mean,
            "Rank IC": rank_ic_mean,
            "IR": ir_val,
        })

    return pd.DataFrame(rows, index=["Before Fee", "After Fee"])


def plot_quantile_returns(
    group_returns: pd.DataFrame,
    n_groups: int,
    title: str = "Quantile Returns",
) -> matplotlib.figure.Figure:
    """Plot cumulative returns for each quantile group."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for g in range(1, n_groups + 1):
        col = f"group_{g}"
        if col in group_returns.columns:
            cum = (1.0 + group_returns[col]).cumprod()
            ax.plot(group_returns["date"].values, cum.values, label=f"Group {g}", linewidth=1.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_group_ic(
    group_ic: pd.DataFrame,
    title: str = "Group IC",
) -> matplotlib.figure.Figure:
    """Plot bar chart of IC mean per quantile group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(group_ic["group"].astype(str), group_ic["ic_mean"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Group")
    ax.set_ylabel("IC Mean")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_ic_cumsum(
    timestamps: pd.Series,
    ic_series: np.ndarray,
    rank_ic_series: np.ndarray,
    title: str = "IC Cumsum",
) -> matplotlib.figure.Figure:
    """Plot IC and Rank IC cumulative sum with mean annotation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ic_cumsum = np.nancumsum(ic_series)
    rank_ic_cumsum = np.nancumsum(rank_ic_series)
    ic_mean = np.nanmean(ic_series)
    rank_ic_mean = np.nanmean(rank_ic_series)

    ts = timestamps.values if hasattr(timestamps, "values") else np.array(timestamps)

    ax1.plot(ts, ic_cumsum, linewidth=1.0, color="steelblue")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("IC Cumsum")
    ax1.set_title(f"{title} — IC (mean={ic_mean:.4f})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, rank_ic_cumsum, linewidth=1.0, color="darkorange")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rank IC Cumsum")
    ax2.set_title(f"{title} — Rank IC (mean={rank_ic_mean:.4f})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ic_decay(
    decay_lags: List[int],
    ic_means: List[float],
    rank_ic_means: List[float],
    title: str = "IC Decay",
) -> matplotlib.figure.Figure:
    """Plot IC Decay bar chart: x=lag, y=IC mean."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(decay_lags))
    width = 0.35

    ax.bar(x - width / 2, ic_means, width, label="IC", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, rank_ic_means, width, label="Rank IC", color="darkorange", alpha=0.8)

    ax.set_xlabel("Lag")
    ax.set_ylabel("IC Mean")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in decay_lags])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig
