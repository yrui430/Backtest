"""Pydantic request/response models for the backtest API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TimingBacktestRequest(BaseModel):
    """Request body for POST /backtest/timing."""

    # ---- data ----
    feature_path: str = Field(..., description="Path to feature file (parquet/csv/h5)")
    time_col: str = Field("timestamp", description="Timestamp column name")
    feature_cols: List[str] = Field(..., description="Feature columns to backtest")

    # ---- label mode ----
    label_path: Optional[str] = Field(None, description="Path to label file")
    label_col: Optional[str] = Field(None, description="Label column name")

    # ---- raw data mode ----
    raw_data_path: Optional[str] = Field(None, description="Path to raw price file")
    price_col: Optional[str] = Field(None, description="Price column name (close/vwap)")
    use_raw_data: bool = Field(False, description="Use raw data mode")
    exec_price_type: Literal["close", "vwap"] = Field("close")

    # ---- label spec ----
    display_labels: List[int] = Field([1], description="Label horizons to evaluate")
    lag: int = Field(1, description="Signal-to-execution lag (bars)")

    # ---- signal spec ----
    signal_method: Literal["quantile", "threshold"] = Field("quantile")
    upper_quantile: float = Field(0.8)
    lower_quantile: float = Field(0.2)
    rolling_window: int = Field(100)
    upper_threshold: Optional[float] = None
    lower_threshold: Optional[float] = None
    signal_direction: Literal["momentum", "mean_reversion"] = Field("momentum")
    display_modes: List[Literal["long_only", "short_only", "long_short"]] = Field(
        ["long_only", "short_only", "long_short"]
    )

    # ---- execution spec ----
    fee_rate: float = Field(0.00005, description="One-sided fee rate")
    show_before_fee: bool = True
    show_after_fee: bool = True
    mode: Literal["diff", "label"] = Field("diff", description="Execution mode: 'diff' (signal-driven) or 'label' (fixed holding)")


    # ---- report ----
    ic_rolling_window: int = Field(20)
    bars_per_year: int = Field(252)

    # ---- response options ----
    include_raw_data: bool = Field(False, description="Include raw PnL data in response")
    include_charts: bool = Field(True, description="Include base64 chart images")


class TimingBacktestResponse(BaseModel):
    """Response body for POST /backtest/timing."""

    status: str = Field(..., description="'success' or 'error'")
    message: Optional[str] = None
    summary_tables: Dict[str, Any] = Field(default_factory=dict)
    charts: Dict[str, str] = Field(default_factory=dict, description="base64 PNG images")
    raw_data: Optional[Dict[str, Any]] = None


class CrossSectionBacktestRequest(BaseModel):
    """Request body for POST /backtest/cross-section."""

    # ---- data ----
    feature_path: str = Field(..., description="Path to feature file")
    raw_data_path: Optional[str] = Field(None, description="Path to raw price file")
    label_path: Optional[str] = Field(None, description="Path to label file")
    use_raw_data: bool = Field(True, description="Use raw data mode (default)")

    # ---- columns ----
    stock_col: str = Field("stock_id")
    date_col: str = Field("date_id")
    time_col: Optional[str] = Field("time_id")
    price_col: str = Field("close")
    exec_price_type: Literal["close", "vwap", "open"] = Field("close")
    feature_cols: List[str] = Field(..., description="Feature columns to backtest")
    label_col: Optional[str] = None
    industry_col: Optional[str] = None

    # ---- data format ----
    data_format: Literal["long", "wide", "multi_file"] = Field("long")

    # ---- winsorize ----
    winsorize_enabled: bool = Field(True)
    winsorize_method: Literal["mad", "percentile", "std"] = Field("mad")
    winsorize_n_sigma: float = Field(3.0)
    winsorize_lower: float = Field(0.01)
    winsorize_upper: float = Field(0.99)

    # ---- neutralize ----
    neutralize_enabled: bool = Field(False)
    neutralize_method: Literal["regression", "demean", "intra_industry"] = Field("regression")

    # ---- normalize ----
    normalize_method: Literal["min_max", "zscore", "rank"] = Field("min_max")

    # ---- label / IC Decay ----
    h: int = Field(1, description="Holding horizon")
    lag: int = Field(1, description="Execution delay")
    decay_lags: List[int] = Field([1, 2, 5])

    # ---- execution ----
    weight_method: Literal["group", "money"] = Field("group")
    n_groups: int = Field(5)
    display_modes: List[Literal["long_only", "long_short"]] = Field(["long_only", "long_short"])
    signal_direction: Literal["momentum", "mean_reversion"] = Field("momentum")
    fee_rate: float = Field(0.00005)
    show_before_fee: bool = True
    show_after_fee: bool = True
    initial_capital: float = Field(1_000_000)

    # ---- report ----
    ic_rolling_window: int = Field(20)
    bars_per_year: int = Field(252)

    # ---- response options ----
    include_raw_data: bool = Field(False)
    include_charts: bool = Field(True)


class CrossSectionBacktestResponse(BaseModel):
    """Response body for POST /backtest/cross-section."""

    status: str = Field(..., description="'success' or 'error'")
    message: Optional[str] = None
    summary_tables: Dict[str, Any] = Field(default_factory=dict)
    charts: Dict[str, str] = Field(default_factory=dict, description="base64 PNG images")
    raw_data: Optional[Dict[str, Any]] = None
