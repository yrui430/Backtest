from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_api.config import SignalSpec
from backtest_api.numba_utils import rolling_quantile


def generate_signals(feature: pd.Series, spec: SignalSpec) -> pd.Series:
    """Generate trading signals from a feature series based on SignalSpec."""
    if spec.method == "quantile":
        signal = _quantile_signal(feature, spec)
    elif spec.method == "threshold":
        signal = _threshold_signal(feature, spec)
    elif spec.method == "custom":
        signal = spec.signal_mapper(feature)
        return signal.reset_index(drop=True)
    else:
        raise ValueError(f"Unknown signal method: {spec.method}")

    if spec.signal_direction == "mean_reversion":
        signal = signal * -1

    return signal


def _quantile_signal(feature: pd.Series, spec: SignalSpec) -> pd.Series:
    arr = feature.values.astype(np.float64)
    upper = rolling_quantile(arr, window=spec.rolling_window, q=spec.upper_quantile)
    lower = rolling_quantile(arr, window=spec.rolling_window, q=spec.lower_quantile)

    signal = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(arr[i]):
            signal[i] = np.nan
        elif arr[i] > upper[i]:
            signal[i] = 1.0
        elif arr[i] < lower[i]:
            signal[i] = -1.0
        else:
            signal[i] = 0.0

    return pd.Series(signal)


def _threshold_signal(feature: pd.Series, spec: SignalSpec) -> pd.Series:
    arr = feature.values.astype(np.float64)
    signal = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            signal[i] = np.nan
        elif arr[i] > spec.upper_threshold:
            signal[i] = 1.0
        elif arr[i] < spec.lower_threshold:
            signal[i] = -1.0
        else:
            signal[i] = 0.0
    return pd.Series(signal)
