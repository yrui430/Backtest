"""Tests for cross-section label computation and IC decay."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_api.cross_section.label import (
    compute_forward_returns,
    compute_ic_decay,
)


@pytest.fixture
def price_df():
    """Simple price data: 3 stocks, 20 dates with known prices."""
    np.random.seed(42)
    stocks = ["A", "B", "C"]
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
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


@pytest.fixture
def feature_df(price_df):
    """Feature data aligned with price_df."""
    np.random.seed(123)
    df = price_df[["stock_id", "date_id"]].copy()
    df["alpha1"] = np.random.randn(len(df))
    return df


class TestComputeForwardReturns:
    def test_basic_h1_lag1(self, price_df):
        result = compute_forward_returns(
            price_df,
            stock_col="stock_id",
            date_col="date_id",
            price_col="close",
            h=1,
            lag=1,
        )
        assert "label" in result.columns
        stock_a = result[result["stock_id"] == "A"]
        assert stock_a["label"].isna().sum() > 0

    def test_h5_lag1(self, price_df):
        result = compute_forward_returns(
            price_df,
            stock_col="stock_id",
            date_col="date_id",
            price_col="close",
            h=5,
            lag=1,
        )
        stock_a = result[result["stock_id"] == "A"]
        assert stock_a["label"].isna().sum() >= 5

    def test_formula_correctness(self, price_df):
        """Verify: label(t) = price(t+lag+h) / price(t+lag) - 1."""
        result = compute_forward_returns(
            price_df,
            stock_col="stock_id",
            date_col="date_id",
            price_col="close",
            h=2,
            lag=1,
        )
        stock_a_prices = price_df[price_df["stock_id"] == "A"]["close"].values
        stock_a_labels = result[result["stock_id"] == "A"]["label"].values
        expected = stock_a_prices[3] / stock_a_prices[1] - 1
        np.testing.assert_almost_equal(stock_a_labels[0], expected)


class TestComputeIcDecay:
    def test_returns_correct_shape(self, feature_df, price_df):
        result = compute_ic_decay(
            feature_df=feature_df,
            price_df=price_df,
            stock_col="stock_id",
            date_col="date_id",
            feature_col="alpha1",
            price_col="close",
            h=1,
            decay_lags=[1, 2, 5],
        )
        assert len(result) == 3
        assert "lag" in result.columns
        assert "ic_mean" in result.columns
        assert "rank_ic_mean" in result.columns

    def test_decay_lags_match(self, feature_df, price_df):
        result = compute_ic_decay(
            feature_df=feature_df,
            price_df=price_df,
            stock_col="stock_id",
            date_col="date_id",
            feature_col="alpha1",
            price_col="close",
            h=1,
            decay_lags=[1, 3],
        )
        assert list(result["lag"]) == [1, 3]
