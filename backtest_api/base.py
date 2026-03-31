from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BacktestResult:
    """Container for backtest output: tables, figures, raw data."""

    def __init__(self) -> None:
        self.summary_tables: dict[str, pd.DataFrame] = {}
        self.figures: dict[str, Any] = {}
        self.raw_data: dict[str, pd.DataFrame] = {}

    def show(self) -> None:
        """Print summary tables and display figures."""
        for name, table in self.summary_tables.items():
            print(f"\n=== {name} ===")
            print(table.to_string())
        for name, fig in self.figures.items():
            fig.suptitle(name) if not fig._suptitle else None
            fig.show()


class BaseBacktest(ABC):

    @abstractmethod
    def load_data(self) -> None:
        """Load data, validate format, check timestamp alignment."""
        ...

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration."""
        ...

    @abstractmethod
    def run(self) -> BacktestResult:
        """Execute the backtest pipeline, return result."""
        ...

    @abstractmethod
    def evaluate(self) -> pd.DataFrame:
        """Compute performance metrics table."""
        ...

    @abstractmethod
    def report(self) -> None:
        """Generate charts + table output."""
        ...
