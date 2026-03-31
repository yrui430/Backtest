from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

import pandas as pd


@dataclass
class TimingBacktestConfig:
    # ---- required ----
    feature_path: str
    time_col: str = "timestamp"
    feature_cols: List[str] = field(default_factory=list)

    # ---- mode A: has label ----
    label_path: Optional[str] = None
    label_col: Optional[str] = None

    # ---- mode B: raw data ----
    raw_data_path: Optional[str] = None
    price_col: Optional[str] = None

    use_raw_data: bool = False
    exec_price_type: Literal["close", "vwap"] = "close"

    def validate(self) -> None:
        if not self.feature_cols:
            raise ValueError("feature_cols must not be empty.")
        if not self.use_raw_data:
            if not self.label_path:
                raise ValueError("label_path is required when use_raw_data=False.")
            if not self.label_col:
                raise ValueError("label_col is required when use_raw_data=False.")
        else:
            if not self.raw_data_path:
                raise ValueError("raw_data_path is required when use_raw_data=True.")
            if not self.price_col:
                raise ValueError("price_col is required when use_raw_data=True.")


@dataclass
class LabelSpec:
    label_horizon: int = 1
    lag: int = 1
    exec_price_type: Literal["close", "vwap"] = "close"
    display_labels: List[int] = field(default_factory=lambda: [1])


@dataclass
class SignalSpec:
    method: Literal["quantile", "threshold", "custom"] = "quantile"

    upper_quantile: float = 0.8
    lower_quantile: float = 0.2
    rolling_window: int = 100

    upper_threshold: Optional[float] = None
    lower_threshold: Optional[float] = None

    signal_direction: Literal["momentum", "mean_reversion"] = "momentum"

    signal_mapper: Optional[Callable[[pd.Series], pd.Series]] = None

    display_modes: List[Literal["long_only", "short_only", "long_short"]] = field(
        default_factory=lambda: ["long_only", "short_only", "long_short"]
    )

    def validate(self) -> None:
        if self.method == "custom" and self.signal_mapper is None:
            raise ValueError("signal_mapper is required when method='custom'.")
        if self.method == "threshold":
            if self.upper_threshold is None or self.lower_threshold is None:
                raise ValueError(
                    "upper_threshold and lower_threshold are required when method='threshold'."
                )


@dataclass
class ExecutionSpec:
    fee_rate: float = 0.00005
    show_before_fee: bool = True
    show_after_fee: bool = True
    hurdle_enabled: bool = False
    hurdle_func: Optional[Callable] = None
