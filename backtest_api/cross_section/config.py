from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional


@dataclass
class CrossSectionConfig:
    # ---- required ----
    feature_path: str
    feature_cols: List[str] = field(default_factory=list)

    # ---- mode A: raw data (default) ----
    raw_data_path: Optional[str] = None
    price_col: str = "close"

    # ---- mode B: pre-computed label ----
    label_path: Optional[str] = None
    label_col: Optional[str] = None

    use_raw_data: bool = True
    exec_price_type: Literal["close", "vwap", "open"] = "close"

    # ---- column names ----
    stock_col: str = "stock_id"
    date_col: str = "date_id"
    time_col: Optional[str] = "time_id"
    industry_col: Optional[str] = None

    # ---- data format ----
    data_format: Literal["long", "wide", "multi_file"] = "long"

    def validate(self) -> None:
        if not self.feature_cols:
            raise ValueError("feature_cols must not be empty.")
        if self.use_raw_data:
            if not self.raw_data_path:
                raise ValueError("raw_data_path is required when use_raw_data=True.")
        else:
            if not self.label_path:
                raise ValueError("label_path is required when use_raw_data=False.")
            if not self.label_col:
                raise ValueError("label_col is required when use_raw_data=False.")


@dataclass
class CrossSectionSignalSpec:
    # ---- winsorize ----
    winsorize_enabled: bool = True
    winsorize_method: Literal["mad", "percentile", "std", "custom"] = "mad"
    winsorize_n_sigma: float = 3.0
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    winsorize_func: Optional[Callable] = None

    # ---- neutralize ----
    neutralize_enabled: bool = False
    neutralize_method: Literal["regression", "demean", "intra_industry", "custom"] = "regression"
    industry_col: Optional[str] = None
    neutralize_func: Optional[Callable] = None

    # ---- normalize ----
    normalize_method: Literal["min_max", "zscore", "rank"] = "min_max"

    def validate(self) -> None:
        if self.winsorize_method == "custom" and self.winsorize_func is None:
            raise ValueError("winsorize_func is required when winsorize_method='custom'.")
        if self.neutralize_enabled:
            if not self.industry_col:
                raise ValueError("industry_col is required when neutralize_enabled=True.")
            if self.neutralize_method == "custom" and self.neutralize_func is None:
                raise ValueError("neutralize_func is required when neutralize_method='custom'.")


@dataclass
class CrossSectionLabelSpec:
    h: int = 1
    lag: int = 1
    decay_lags: List[int] = field(default_factory=lambda: [1, 2, 5])


@dataclass
class CrossSectionExecutionSpec:
    weight_method: Literal["group", "money"] = "group"
    n_groups: int = 5
    group_weight_func: Optional[Callable] = None
    display_modes: List[Literal["long_only", "long_short"]] = field(
        default_factory=lambda: ["long_only", "long_short"]
    )
    signal_direction: Literal["momentum", "mean_reversion"] = "momentum"
    fee_rate: float = 0.00005
    show_before_fee: bool = True
    show_after_fee: bool = True
    initial_capital: float = 1_000_000

    def validate(self) -> None:
        if self.n_groups < 2:
            raise ValueError("n_groups must be >= 2.")
