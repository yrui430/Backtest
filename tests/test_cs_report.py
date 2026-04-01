"""Tests for cross-section report charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

from backtest_api.cross_section.report import (
    plot_quantile_returns,
    plot_group_ic,
    plot_ic_cumsum,
    plot_ic_decay,
    build_cs_summary_table,
)


@pytest.fixture
def group_returns():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(42)
    df = pd.DataFrame({"date": dates})
    for g in range(1, 6):
        df[f"group_{g}"] = np.random.randn(50) * 0.01 + g * 0.001
    return df


@pytest.fixture
def group_ic():
    return pd.DataFrame({
        "group": [1, 2, 3, 4, 5],
        "ic_mean": [0.01, 0.02, 0.03, 0.04, 0.05],
    })


class TestPlotQuantileReturns:
    def test_returns_figure(self, group_returns):
        fig = plot_quantile_returns(group_returns, n_groups=5, title="Test")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_correct_lines(self, group_returns):
        fig = plot_quantile_returns(group_returns, n_groups=5, title="Test")
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 5


class TestPlotGroupIc:
    def test_returns_figure(self, group_ic):
        fig = plot_group_ic(group_ic, title="Test")
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotIcCumsum:
    def test_returns_figure(self):
        np.random.seed(42)
        timestamps = pd.date_range("2024-01-01", periods=50, freq="D")
        ic = np.random.randn(50) * 0.05
        rank_ic = np.random.randn(50) * 0.05
        fig = plot_ic_cumsum(timestamps, ic, rank_ic, title="Test")
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotIcDecay:
    def test_returns_figure(self):
        fig = plot_ic_decay(
            decay_lags=[1, 2, 5],
            ic_means=[0.05, 0.03, 0.01],
            rank_ic_means=[0.04, 0.02, 0.008],
            title="Test",
        )
        assert isinstance(fig, matplotlib.figure.Figure)


class TestBuildCsSummaryTable:
    def test_basic_output(self):
        np.random.seed(42)
        n = 100
        pnl_before = pd.Series(np.random.randn(n) * 0.01)
        pnl_after = pd.Series(np.random.randn(n) * 0.01)
        positions = pd.Series(np.random.randn(n) * 0.5)
        ic_series = np.random.randn(n) * 0.05
        rank_ic_series = np.random.randn(n) * 0.05
        table = build_cs_summary_table(
            pnl_before, pnl_after, positions, ic_series, rank_ic_series,
        )
        assert "Before Fee" in table.index
        assert "After Fee" in table.index
        assert "Sharpe Ratio" in table.columns
        assert "IC" in table.columns
