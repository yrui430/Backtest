# backtest_api/report.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

plt.rcParams["figure.max_open_warning"] = 0  # suppress warning in batch mode

from backtest_api.metrics import (
    annualized_return,
    total_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_recovery,
    turnover,
    compute_ic,
    compute_rank_ic,
    information_ratio,
)
from backtest_api.numba_utils import rolling_pearson, rolling_spearman


def build_summary_table(
    pnl_before: pd.Series,
    pnl_after: pd.Series,
    positions: pd.Series,
    feature: np.ndarray,
    label: np.ndarray,
    bars_per_year: int = 252,
) -> pd.DataFrame:
    """Build the performance summary table with before/after fee rows."""
    ic_val = compute_ic(feature, label)
    rank_ic_val = compute_rank_ic(feature, label)

    ic_rolling = rolling_pearson(
        feature.astype(np.float64), label.astype(np.float64), window=20
    )
    ic_series = pd.Series(ic_rolling).dropna()
    ir_val = information_ratio(ic_series)

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
            "Rank IC": rank_ic_val,
            "IC": ic_val,
            "IR": ir_val,
        })

    return pd.DataFrame(rows, index=["Before Fee", "After Fee"])


def plot_pnl_curve(
    timestamps: pd.Series,
    pnl_before: pd.Series,
    pnl_after: pd.Series,
    title: str = "PnL Curve",
) -> matplotlib.figure.Figure:
    """Plot cumulative PnL curve. All labels in English."""
    cum_before = (1.0 + pnl_before).cumprod()
    cum_after = (1.0 + pnl_after).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps.values, cum_before.values, label="Before Fee", linewidth=1.2)
    ax.plot(timestamps.values, cum_after.values, label="After Fee", linewidth=1.2)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_rolling_ic(
    timestamps: pd.Series,
    feature: np.ndarray,
    label: np.ndarray,
    rolling_window: int = 20,
    title: str = "Rolling IC",
) -> matplotlib.figure.Figure:
    """Plot rolling IC and Rank IC. Uses numba, NOT pandas rolling."""
    feat = feature.astype(np.float64)
    lab = label.astype(np.float64)
    ic_arr = rolling_pearson(feat, lab, window=rolling_window)
    rank_ic_arr = rolling_spearman(feat, lab, window=rolling_window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(timestamps.values, ic_arr, linewidth=0.8, color="steelblue")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("IC (Pearson)")
    ax1.set_title(f"{title} — IC Rolling Window = {rolling_window}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(timestamps.values, rank_ic_arr, linewidth=0.8, color="darkorange")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Rank IC (Spearman)")
    ax2.set_title(f"{title} — Rank IC Rolling Window = {rolling_window}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
