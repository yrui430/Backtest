"""backtest_api — Quantitative strategy backtesting framework.

Provides timing backtest (Phase 1), with cross-sectional and HF planned.

Quick start (Python)::

    from backtest_api import TimingBacktest

    bt = TimingBacktest(
        feature_path="features.parquet",
        label_path="labels.parquet",
        feature_cols=["alpha1"],
        label_col="ret_1",
    )
    result = bt.run()
    result.show()

Quick start (REST API)::

    python main.py          # starts server on http://localhost:8000
    # POST /backtest/timing  with JSON body
"""

from backtest_api.base import BaseBacktest, BacktestResult
from backtest_api.config import (
    ExecutionSpec,
    LabelSpec,
    SignalSpec,
    TimingBacktestConfig,
)
from backtest_api.data_loader import align_timestamps, load_file
from backtest_api.metrics import (
    annualized_return,
    compute_ic,
    compute_rank_ic,
    information_ratio,
    max_drawdown,
    max_drawdown_recovery,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    turnover,
    volatility,
)
from backtest_api.timing import TimingBacktest
from backtest_api.cross_section import CrossSectionBacktest
from backtest_api.cross_section.config import (
    CrossSectionConfig,
    CrossSectionSignalSpec,
    CrossSectionLabelSpec,
    CrossSectionExecutionSpec,
)

__all__ = [
    # Core
    "TimingBacktest",
    "CrossSectionBacktest",
    "BaseBacktest",
    "BacktestResult",
    # Timing Config
    "TimingBacktestConfig",
    "LabelSpec",
    "SignalSpec",
    "ExecutionSpec",
    # Cross-Section Config
    "CrossSectionConfig",
    "CrossSectionSignalSpec",
    "CrossSectionLabelSpec",
    "CrossSectionExecutionSpec",
    # Data
    "load_file",
    "align_timestamps",
    # Metrics
    "total_return",
    "annualized_return",
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "max_drawdown_recovery",
    "turnover",
    "compute_ic",
    "compute_rank_ic",
    "information_ratio",
]
