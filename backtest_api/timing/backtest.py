from __future__ import annotations

from typing import List, Literal, Optional, Callable

import numpy as np
import pandas as pd

from backtest_api.base import BaseBacktest, BacktestResult
from backtest_api.config import (
    TimingBacktestConfig,
    LabelSpec,
    SignalSpec,
    ExecutionSpec,
)
from backtest_api.data_loader import load_file, align_timestamps
from backtest_api.timing.label import compute_labels_from_raw
from backtest_api.timing.signal import generate_signals
from backtest_api.timing.executor import execute_timing_backtest
from backtest_api.report import build_summary_table, plot_pnl_curve, plot_rolling_ic


class TimingBacktest(BaseBacktest):
    """Single-asset timing backtest with signal-based entry/exit."""

    def __init__(
        self,
        feature_path: str,
        time_col: str = "timestamp",
        feature_cols: Optional[List[str]] = None,
        # ---- label mode ----
        label_path: Optional[str] = None,
        label_col: Optional[str] = None,
        # ---- raw data mode ----
        raw_data_path: Optional[str] = None,
        price_col: Optional[str] = None,
        use_raw_data: bool = False,
        exec_price_type: Literal["close", "vwap"] = "close",
        # ---- label spec ----
        display_labels: Optional[List[int]] = None,
        lag: int = 1,
        # ---- signal spec ----
        signal_method: Literal["quantile", "threshold", "custom"] = "quantile",
        upper_quantile: float = 0.8,
        lower_quantile: float = 0.2,
        rolling_window: int = 100,
        upper_threshold: Optional[float] = None,
        lower_threshold: Optional[float] = None,
        signal_direction: Literal["momentum", "mean_reversion"] = "momentum",
        signal_mapper: Optional[Callable] = None,
        display_modes: Optional[List[str]] = None,
        # ---- execution spec ----
        fee_rate: float = 0.00005,
        show_before_fee: bool = True,
        show_after_fee: bool = True,
        hurdle_enabled: bool = False,
        hurdle_func: Optional[Callable] = None,
        # ---- report ----
        ic_rolling_window: int = 20,
        bars_per_year: int = 252,
    ) -> None:
        self.config = TimingBacktestConfig(
            feature_path=feature_path,
            label_path=label_path,
            label_col=label_col,
            raw_data_path=raw_data_path,
            price_col=price_col,
            time_col=time_col,
            feature_cols=feature_cols or [],
            use_raw_data=use_raw_data,
            exec_price_type=exec_price_type,
        )
        self.label_spec = LabelSpec(
            label_horizon=1,
            lag=lag,
            exec_price_type=exec_price_type,
            display_labels=display_labels or [1],
        )
        self.signal_spec = SignalSpec(
            method=signal_method,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
            rolling_window=rolling_window,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            signal_direction=signal_direction,
            signal_mapper=signal_mapper,
            display_modes=display_modes or ["long_only", "short_only", "long_short"],
        )
        self.exec_spec = ExecutionSpec(
            fee_rate=fee_rate,
            show_before_fee=show_before_fee,
            show_after_fee=show_after_fee,
            hurdle_enabled=hurdle_enabled,
            hurdle_func=hurdle_func,
        )
        self.ic_rolling_window = ic_rolling_window
        self.bars_per_year = bars_per_year

        self._feature_df: Optional[pd.DataFrame] = None
        self._label_df: Optional[pd.DataFrame] = None
        self._prices: Optional[pd.Series] = None
        self._timestamps: Optional[pd.Series] = None

    def load_data(self) -> bool:
        """Load and align data. Returns False if alignment fails."""
        feat_df = load_file(self.config.feature_path)

        if self.config.use_raw_data:
            raw_df = load_file(self.config.raw_data_path)
            result = align_timestamps(feat_df, raw_df, self.config.time_col)
            if result is None:
                return False
            feat_df, raw_df = result
            self._prices = raw_df[self.config.price_col].reset_index(drop=True)
            self._timestamps = feat_df[self.config.time_col].reset_index(drop=True)
            self._feature_df = feat_df
            self._label_df = None
        else:
            label_df = load_file(self.config.label_path)
            result = align_timestamps(feat_df, label_df, self.config.time_col)
            if result is None:
                return False
            feat_df, label_df = result
            self._timestamps = feat_df[self.config.time_col].reset_index(drop=True)
            self._feature_df = feat_df
            self._label_df = label_df
            self._prices = label_df[self.config.label_col].reset_index(drop=True)

        return True

    def validate(self) -> None:
        self.config.validate()
        if self.signal_spec.method != "quantile":
            self.signal_spec.validate()

    def run(self) -> Optional[BacktestResult]:
        """Run the full timing backtest pipeline."""
        self.validate()
        if not self.load_data():
            return None

        result = BacktestResult()

        for feat_col in self.config.feature_cols:
            feature = self._feature_df[feat_col].values.astype(np.float64)
            signal = generate_signals(pd.Series(feature), self.signal_spec)

            for horizon in self.label_spec.display_labels:
                if self.config.use_raw_data:
                    label_series = compute_labels_from_raw(self._prices, horizon)
                else:
                    label_series = self._label_df[self.config.label_col].reset_index(drop=True)

                label_arr = label_series.values.astype(np.float64)

                if self.config.use_raw_data:
                    exec_prices = self._prices
                else:
                    exec_prices = self._prices

                current_label_spec = LabelSpec(
                    label_horizon=horizon,
                    lag=self.label_spec.lag,
                    exec_price_type=self.label_spec.exec_price_type,
                )

                for mode in self.signal_spec.display_modes:
                    key = f"{feat_col}_label{horizon}_{mode}"

                    exec_result = execute_timing_backtest(
                        timestamps=self._timestamps,
                        prices=exec_prices,
                        signals=signal,
                        label_spec=current_label_spec,
                        exec_spec=self.exec_spec,
                        display_mode=mode,
                    )

                    table = build_summary_table(
                        pnl_before=exec_result["pnl_before_fee"],
                        pnl_after=exec_result["pnl_after_fee"],
                        positions=exec_result["position"],
                        feature=feature,
                        label=label_arr,
                        bars_per_year=self.bars_per_year,
                    )
                    result.summary_tables[key] = table
                    result.raw_data[key] = exec_result

                    pnl_fig = plot_pnl_curve(
                        self._timestamps,
                        exec_result["pnl_before_fee"],
                        exec_result["pnl_after_fee"],
                        title=f"PnL — {feat_col} label{horizon} {mode}",
                    )
                    result.figures[f"pnl_{key}"] = pnl_fig

                ic_key = f"ic_{feat_col}_label{horizon}"
                ic_fig = plot_rolling_ic(
                    self._timestamps,
                    feature,
                    label_arr,
                    rolling_window=self.ic_rolling_window,
                    title=f"IC — {feat_col} label{horizon}",
                )
                result.figures[ic_key] = ic_fig

        return result

    def evaluate(self) -> pd.DataFrame:
        result = self.run()
        if result is None:
            return pd.DataFrame()
        tables = []
        for key, table in result.summary_tables.items():
            t = table.copy()
            t["config"] = key
            tables.append(t)
        return pd.concat(tables, ignore_index=False) if tables else pd.DataFrame()

    def report(self) -> None:
        result = self.run()
        if result is not None:
            result.show()
