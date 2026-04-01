"""Tests for cross-section executor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_api.cross_section.executor import (
    assign_quantile_groups,
    compute_weights_group,
    compute_weights_money,
    execute_cross_section_backtest,
)
from backtest_api.cross_section.config import (
    CrossSectionExecutionSpec,
    CrossSectionLabelSpec,
)


class TestAssignQuantileGroups:
    def test_basic_5_groups(self):
        signals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        groups = assign_quantile_groups(signals, n_groups=5)
        assert groups.min() == 1
        assert groups.max() == 5
        assert groups.iloc[0] == 1
        assert groups.iloc[-1] == 5

    def test_3_groups(self):
        signals = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        groups = assign_quantile_groups(signals, n_groups=3)
        assert set(groups.unique()).issubset({1, 2, 3})


class TestComputeWeightsGroup:
    def test_long_short_momentum(self):
        signals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        groups = pd.Series([1, 1, 3, 5, 5])
        weights = compute_weights_group(
            signals, groups, n_groups=5, direction="momentum", mode="long_short"
        )
        long_sum = weights[groups == 5].sum()
        short_sum = weights[groups == 1].sum()
        assert long_sum == pytest.approx(1.0)
        assert short_sum == pytest.approx(-1.0)
        assert weights[groups == 3].sum() == pytest.approx(0.0)

    def test_long_only_momentum(self):
        signals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        groups = pd.Series([1, 2, 3, 4, 5])
        weights = compute_weights_group(
            signals, groups, n_groups=5, direction="momentum", mode="long_only"
        )
        assert weights[groups == 5].sum() == pytest.approx(1.0)
        assert weights[groups != 5].sum() == pytest.approx(0.0)

    def test_long_short_mean_reversion(self):
        signals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        groups = pd.Series([1, 2, 3, 4, 5])
        weights = compute_weights_group(
            signals, groups, n_groups=5, direction="mean_reversion", mode="long_short"
        )
        assert weights[groups == 1].sum() == pytest.approx(1.0)
        assert weights[groups == 5].sum() == pytest.approx(-1.0)

    def test_custom_weight_func(self):
        signals = pd.Series([1.0, 2.0, 3.0])
        groups = pd.Series([1, 1, 1])
        def w_func(sigs):
            return sigs / sigs.sum()
        weights = compute_weights_group(
            signals, groups, n_groups=1, direction="momentum", mode="long_only",
            weight_func=w_func,
        )
        assert weights.sum() == pytest.approx(1.0)
        assert weights.iloc[2] > weights.iloc[0]


class TestComputeWeightsMoney:
    def test_long_short(self):
        signals = pd.Series([0.5, 0.3, -0.4, -0.6])
        weights = compute_weights_money(signals, direction="momentum", mode="long_short")
        positive_sum = weights[weights > 0].sum()
        negative_sum = weights[weights < 0].sum()
        assert positive_sum == pytest.approx(1.0)
        assert negative_sum == pytest.approx(-1.0)

    def test_long_only_momentum(self):
        signals = pd.Series([0.5, 0.3, -0.4, -0.6])
        weights = compute_weights_money(signals, direction="momentum", mode="long_only")
        assert weights.sum() == pytest.approx(1.0)
        assert (weights[signals < 0] == 0).all()

    def test_long_only_mean_reversion(self):
        signals = pd.Series([0.5, 0.3, -0.4, -0.6])
        weights = compute_weights_money(
            signals, direction="mean_reversion", mode="long_only"
        )
        assert weights.sum() == pytest.approx(1.0)
        assert (weights[signals > 0] == 0).all()


class TestExecuteCrossSectionBacktest:
    @pytest.fixture
    def simple_signal_df(self):
        np.random.seed(42)
        stocks = ["A", "B", "C"]
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        rows = []
        for d in dates:
            for s in stocks:
                rows.append({
                    "stock_id": s,
                    "date_id": d,
                    "signal": np.random.randn(),
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def simple_price_df(self):
        np.random.seed(42)
        stocks = ["A", "B", "C"]
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        rows = []
        prices = {"A": 100.0, "B": 50.0, "C": 200.0}
        for d in dates:
            for s in stocks:
                prices[s] *= 1 + np.random.randn() * 0.02
                rows.append({
                    "stock_id": s,
                    "date_id": d,
                    "close": prices[s],
                })
        return pd.DataFrame(rows)

    def test_returns_required_keys(self, simple_signal_df, simple_price_df):
        spec = CrossSectionExecutionSpec()
        label_spec = CrossSectionLabelSpec()
        result = execute_cross_section_backtest(
            signal_df=simple_signal_df,
            price_df=simple_price_df,
            spec=spec,
            label_spec=label_spec,
            stock_col="stock_id",
            date_col="date_id",
            price_col="close",
            display_mode="long_short",
        )
        assert "portfolio_gross_pnl" in result
        assert "portfolio_net_pnl" in result
        assert "weights" in result
        assert "group_returns" in result

    def test_long_short_net_zero(self, simple_signal_df, simple_price_df):
        spec = CrossSectionExecutionSpec(weight_method="group", n_groups=3)
        label_spec = CrossSectionLabelSpec()
        result = execute_cross_section_backtest(
            signal_df=simple_signal_df,
            price_df=simple_price_df,
            spec=spec,
            label_spec=label_spec,
            stock_col="stock_id",
            date_col="date_id",
            price_col="close",
            display_mode="long_short",
        )
        weights = result["weights"]
        for _, group in weights.groupby("date_id"):
            net = group["weight"].sum()
            assert abs(net) < 1e-10

    def test_long_only_positive_weights(self, simple_signal_df, simple_price_df):
        spec = CrossSectionExecutionSpec(weight_method="group", n_groups=3)
        label_spec = CrossSectionLabelSpec()
        result = execute_cross_section_backtest(
            signal_df=simple_signal_df,
            price_df=simple_price_df,
            spec=spec,
            label_spec=label_spec,
            stock_col="stock_id",
            date_col="date_id",
            price_col="close",
            display_mode="long_only",
        )
        weights = result["weights"]
        assert (weights["weight"] >= -1e-10).all()
