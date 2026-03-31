# backtest_api/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def total_return(pnl_series: pd.Series) -> float:
    return float(np.prod(1.0 + pnl_series.values) - 1.0)


def annualized_return(pnl_series: pd.Series, bars_per_year: int = 252) -> float:
    n = len(pnl_series)
    if n == 0:
        return 0.0
    cum = np.prod(1.0 + pnl_series.values)
    return float(cum ** (bars_per_year / n) - 1.0)


def volatility(pnl_series: pd.Series, bars_per_year: int = 252) -> float:
    return float(np.std(pnl_series.values, ddof=1) * np.sqrt(bars_per_year))


def sharpe_ratio(pnl_series: pd.Series, bars_per_year: int = 252) -> float:
    vol = np.std(pnl_series.values, ddof=1)
    if vol == 0:
        return 0.0
    mean_ret = np.mean(pnl_series.values)
    return float(mean_ret / vol * np.sqrt(bars_per_year))


def sortino_ratio(pnl_series: pd.Series, bars_per_year: int = 252) -> float:
    arr = pnl_series.values
    downside = arr[arr < 0]
    if len(downside) == 0:
        return np.inf
    dd = float(np.std(downside, ddof=1))
    if dd == 0:
        return 0.0
    return float(np.mean(arr) / dd * np.sqrt(bars_per_year))


def max_drawdown(pnl_series: pd.Series) -> float:
    cum = np.cumprod(1.0 + pnl_series.values)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def max_drawdown_recovery(pnl_series: pd.Series) -> int:
    cum = np.cumprod(1.0 + pnl_series.values)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    mdd_idx = int(np.argmax(dd))
    peak_val = peak[mdd_idx]
    for i in range(mdd_idx + 1, len(cum)):
        if cum[i] >= peak_val:
            return i - mdd_idx
    return len(cum) - mdd_idx


def turnover(positions: pd.Series) -> float:
    pos = positions.values
    changes = np.abs(np.diff(pos))
    return float(np.mean(changes)) if len(changes) > 0 else 0.0


def compute_ic(feature: np.ndarray, label: np.ndarray) -> float:
    mask = ~(np.isnan(feature) | np.isnan(label))
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(feature[mask], label[mask])[0, 1])


def compute_rank_ic(feature: np.ndarray, label: np.ndarray) -> float:
    mask = ~(np.isnan(feature) | np.isnan(label))
    if mask.sum() < 3:
        return np.nan
    corr, _ = stats.spearmanr(feature[mask], label[mask])
    return float(corr)


def information_ratio(ic_series: pd.Series) -> float:
    std = ic_series.std()
    if std == 0:
        return 0.0
    return float(ic_series.mean() / std)
