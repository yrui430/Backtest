from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_api.config import LabelSpec, ExecutionSpec


def execute_timing_backtest(
    timestamps: pd.Series,
    prices: pd.Series,
    signals: pd.Series,
    label_spec: LabelSpec,
    exec_spec: ExecutionSpec,
    display_mode: str = "long_short",
) -> pd.DataFrame:
    """Execute timing backtest with position stacking and fee calculation."""
    n = len(prices)
    h = label_spec.label_horizon
    lag = label_spec.lag
    weight = 1.0 / h
    fee_rate = exec_spec.fee_rate

    prices_arr = prices.values.astype(np.float64)
    signals_arr = signals.values.astype(np.float64)

    # Filter signals by display_mode
    if display_mode == "long_only":
        signals_arr = np.where(signals_arr > 0, signals_arr, 0.0)
    elif display_mode == "short_only":
        signals_arr = np.where(signals_arr < 0, signals_arr, 0.0)

    # Build position array: each signal at t creates a position of weight*sign
    # from t+lag to t+lag+h-1
    position = np.zeros(n, dtype=np.float64)
    for t in range(n):
        sig = signals_arr[t]
        if np.isnan(sig) or sig == 0.0:
            continue
        start = t + lag
        end = min(t + lag + h, n)
        for k in range(start, end):
            position[k] += weight * sig

    # Apply hurdle filter (only when hurdle_enabled and hurdle_func provided)
    if exec_spec.hurdle_enabled and exec_spec.hurdle_func is not None:
        filtered_pos = np.zeros(n, dtype=np.float64)
        filtered_pos[0] = position[0]
        for i in range(1, n):
            if exec_spec.hurdle_func(position[i], filtered_pos[i - 1]):
                filtered_pos[i] = position[i]
            else:
                filtered_pos[i] = filtered_pos[i - 1]
        position = filtered_pos

    # PnL: per-bar return based on position and price change
    price_ret = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if prices_arr[i - 1] != 0:
            price_ret[i] = (prices_arr[i] - prices_arr[i - 1]) / prices_arr[i - 1]

    pnl_before = position * price_ret

    # Fee: charged on absolute position change
    pos_change = np.zeros(n, dtype=np.float64)
    pos_change[0] = abs(position[0])
    for i in range(1, n):
        pos_change[i] = abs(position[i] - position[i - 1])

    fee_cost = pos_change * fee_rate * 2  # open + close
    pnl_after = pnl_before - fee_cost

    result = pd.DataFrame({
        "timestamp": timestamps.values,
        "position": position,
        "price_return": price_ret,
        "pnl_before_fee": pnl_before,
        "pnl_after_fee": pnl_after,
        "fee_cost": fee_cost,
    })
    return result
