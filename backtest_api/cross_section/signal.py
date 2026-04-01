from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest_api.cross_section.config import CrossSectionSignalSpec


def winsorize(
    values: np.ndarray,
    method: str = "mad",
    n_sigma: float = 3.0,
    lower: float = 0.01,
    upper: float = 0.99,
    custom_func: Optional[Callable] = None,
) -> np.ndarray:
    """Clip outliers in a single cross-section snapshot."""
    if method == "mad":
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        if mad == 0:
            return values.copy()
        lower_bound = median - n_sigma * 1.4826 * mad
        upper_bound = median + n_sigma * 1.4826 * mad
        return np.clip(values, lower_bound, upper_bound)
    elif method == "percentile":
        lo = np.nanpercentile(values, lower * 100)
        hi = np.nanpercentile(values, upper * 100)
        return np.clip(values, lo, hi)
    elif method == "std":
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1)
        if std == 0:
            return values.copy()
        lower_bound = mean - n_sigma * std
        upper_bound = mean + n_sigma * std
        return np.clip(values, lower_bound, upper_bound)
    elif method == "custom":
        return custom_func(values)
    else:
        raise ValueError(f"Unknown winsorize method: {method}")


def neutralize(
    values: np.ndarray,
    industry: np.ndarray,
    method: str = "regression",
    custom_func: Optional[Callable] = None,
) -> np.ndarray:
    """Remove industry effects from a single cross-section snapshot."""
    if method == "regression":
        unique_ind = np.unique(industry)
        dummies = np.zeros((len(values), len(unique_ind)), dtype=np.float64)
        for i, ind in enumerate(unique_ind):
            dummies[industry == ind, i] = 1.0
        beta, _, _, _ = np.linalg.lstsq(dummies, values, rcond=None)
        fitted = dummies @ beta
        return values - fitted
    elif method == "demean":
        result = values.copy()
        for ind in np.unique(industry):
            mask = industry == ind
            result[mask] = values[mask] - np.nanmean(values[mask])
        return result
    elif method == "intra_industry":
        result = values.copy()
        for ind in np.unique(industry):
            mask = industry == ind
            subset = values[mask]
            std = np.nanstd(subset, ddof=1)
            if std == 0:
                result[mask] = 0.0
            else:
                result[mask] = (subset - np.nanmean(subset)) / std
        return result
    elif method == "custom":
        return custom_func(values, industry)
    else:
        raise ValueError(f"Unknown neutralize method: {method}")


def normalize(values: np.ndarray, method: str = "min_max") -> np.ndarray:
    """Normalize values to [-1, 1] range."""
    if method == "min_max":
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if vmax == vmin:
            return np.zeros_like(values)
        return 2.0 * (values - vmin) / (vmax - vmin) - 1.0
    elif method == "zscore":
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1)
        if std == 0:
            return np.zeros_like(values)
        z = (values - mean) / std
        return np.clip(z / 3.0, -1.0, 1.0)
    elif method == "rank":
        n = len(values)
        if n <= 1:
            return np.zeros_like(values)
        order = np.argsort(np.argsort(values))
        return 2.0 * order / (n - 1) - 1.0
    else:
        raise ValueError(f"Unknown normalize method: {method}")


def generate_cross_section_signals(
    feature_df: pd.DataFrame,
    spec: CrossSectionSignalSpec,
    stock_col: str,
    date_col: str,
    feature_col: str,
    industry_col: Optional[str] = None,
) -> pd.DataFrame:
    """Apply winsorize -> neutralize -> normalize per cross-section (grouped by date)."""
    result = feature_df.copy()
    signals = np.full(len(result), np.nan)

    for date_val, group in result.groupby(date_col):
        idx = group.index
        values = group[feature_col].values.astype(np.float64)

        if np.all(np.isnan(values)):
            continue

        if spec.winsorize_enabled:
            values = winsorize(
                values,
                method=spec.winsorize_method,
                n_sigma=spec.winsorize_n_sigma,
                lower=spec.winsorize_lower,
                upper=spec.winsorize_upper,
                custom_func=spec.winsorize_func,
            )

        if spec.neutralize_enabled and industry_col is not None:
            ind_values = group[industry_col].values
            values = neutralize(
                values,
                ind_values,
                method=spec.neutralize_method,
                custom_func=spec.neutralize_func,
            )

        values = normalize(values, method=spec.normalize_method)
        signals[idx] = values

    result["signal"] = signals
    return result
