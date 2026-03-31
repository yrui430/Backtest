# backtest_api/tests/test_metrics.py
import numpy as np
import pandas as pd
import pytest
from backtest_api.metrics import (
    annualized_return,
    total_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_recovery,
    turnover,
    compute_ic,
    compute_rank_ic,
    information_ratio,
)


class TestReturnMetrics:
    def test_total_return(self):
        pnl = pd.Series([0.01, 0.02, -0.005, 0.015])
        result = total_return(pnl)
        expected = (1.01 * 1.02 * 0.995 * 1.015) - 1.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_annualized_return(self):
        pnl = pd.Series([0.001] * 252)
        result = annualized_return(pnl, bars_per_year=252)
        expected = (1.001 ** 252) - 1.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_volatility(self):
        pnl = pd.Series([0.01, -0.01, 0.01, -0.01])
        result = volatility(pnl, bars_per_year=252)
        assert result > 0

    def test_sharpe_ratio(self):
        pnl = pd.Series([0.01] * 100)
        result = sharpe_ratio(pnl, bars_per_year=252)
        assert result > 10

    def test_sortino_ratio(self):
        pnl = pd.Series([0.01, 0.02, -0.005, 0.015, -0.001])
        result = sortino_ratio(pnl, bars_per_year=252)
        assert result > 0


class TestDrawdown:
    def test_max_drawdown(self):
        pnl = pd.Series([0.1, 0.1, -0.3, -0.1, 0.2])
        result = max_drawdown(pnl)
        assert result > 0
        assert result <= 1.0

    def test_max_drawdown_recovery(self):
        pnl = pd.Series([0.1, 0.1, -0.3, -0.1, 0.05, 0.05, 0.1, 0.1, 0.1])
        bars = max_drawdown_recovery(pnl)
        assert isinstance(bars, (int, float))
        assert bars >= 0


class TestIC:
    def test_perfect_ic(self):
        feature = np.arange(100.0)
        label = feature * 2.0 + 1.0
        result = compute_ic(feature, label)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_perfect_rank_ic(self):
        feature = np.arange(100.0)
        label = feature * 2.0
        result = compute_rank_ic(feature, label)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_information_ratio(self):
        ic_series = pd.Series([0.05, 0.03, 0.07, 0.02, 0.06])
        result = information_ratio(ic_series)
        expected = ic_series.mean() / ic_series.std()
        assert result == pytest.approx(expected, rel=1e-6)


class TestTurnover:
    def test_basic(self):
        positions = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0])
        result = turnover(positions)
        expected = np.mean([1.0, 0.0, 2.0, 1.0])
        assert result == pytest.approx(expected)
