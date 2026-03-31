# backtest_api/tests/test_report.py
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from backtest_api.report import build_summary_table, plot_pnl_curve, plot_rolling_ic


class TestBuildSummaryTable:
    def test_output_columns(self):
        pnl_before = pd.Series(np.random.randn(200) * 0.01)
        pnl_after = pd.Series(pnl_before - 0.0001)
        positions = pd.Series(np.random.choice([-1, 0, 1], 200).astype(float))
        feature = np.random.randn(200)
        label = np.random.randn(200)
        table = build_summary_table(
            pnl_before=pnl_before,
            pnl_after=pnl_after,
            positions=positions,
            feature=feature,
            label=label,
            bars_per_year=252,
        )
        expected_cols = [
            "Annualized Return", "Total Return", "Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Turnover",
            "Max Drawdown", "Max Drawdown Recovery Time",
            "Rank IC", "IC", "IR",
        ]
        for col in expected_cols:
            assert col in table.columns, f"Missing column: {col}"

    def test_before_and_after_fee_rows(self):
        pnl_before = pd.Series(np.random.randn(100) * 0.01)
        pnl_after = pd.Series(pnl_before - 0.0001)
        positions = pd.Series(np.random.choice([-1, 0, 1], 100).astype(float))
        feature = np.random.randn(100)
        label = np.random.randn(100)
        table = build_summary_table(
            pnl_before=pnl_before,
            pnl_after=pnl_after,
            positions=positions,
            feature=feature,
            label=label,
        )
        assert len(table) == 2
        assert table.index.tolist() == ["Before Fee", "After Fee"]


class TestPlotPnlCurve:
    @patch("matplotlib.pyplot.show")
    def test_creates_figure(self, mock_show):
        timestamps = pd.date_range("2026-01-01", periods=50, freq="1min")
        pnl_before = pd.Series(np.random.randn(50) * 0.01)
        pnl_after = pd.Series(pnl_before - 0.0001)
        fig = plot_pnl_curve(timestamps, pnl_before, pnl_after, title="Test PnL")
        assert fig is not None


class TestPlotRollingIC:
    @patch("matplotlib.pyplot.show")
    def test_creates_figure(self, mock_show):
        timestamps = pd.date_range("2026-01-01", periods=200, freq="1min")
        feature = np.random.randn(200)
        label = np.random.randn(200)
        fig = plot_rolling_ic(
            timestamps, feature, label,
            rolling_window=20, title="Test IC",
        )
        assert fig is not None
