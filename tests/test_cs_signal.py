"""Tests for cross-section signal pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_api.cross_section.signal import (
    winsorize,
    neutralize,
    normalize,
    generate_cross_section_signals,
)
from backtest_api.cross_section.config import CrossSectionSignalSpec


class TestWinsorize:
    def test_mad_clips_outliers(self):
        values = np.array([1.0, 2.0, 3.0, 100.0, 2.5])
        result = winsorize(values, method="mad", n_sigma=3.0)
        assert result[3] < 100.0
        np.testing.assert_almost_equal(result[0], 1.0)

    def test_percentile_clips(self):
        np.random.seed(42)
        values = np.random.randn(100)
        values[0] = 50.0
        result = winsorize(values, method="percentile", lower=0.01, upper=0.99)
        assert result[0] < 50.0

    def test_std_clips(self):
        values = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 1.5, 100.0])
        result = winsorize(values, method="std", n_sigma=2.0)
        assert result[6] < 100.0

    def test_custom_func(self):
        values = np.array([1.0, 2.0, 3.0])
        custom = lambda v: np.clip(v, 0, 2)
        result = winsorize(values, method="custom", custom_func=custom)
        np.testing.assert_array_equal(result, [1.0, 2.0, 2.0])


class TestNeutralize:
    def test_regression_removes_industry_effect(self):
        np.random.seed(42)
        values = np.array([10.0, 11.0, 12.0, 1.0, 2.0, 3.0])
        industry = np.array(["A", "A", "A", "B", "B", "B"])
        result = neutralize(values, industry, method="regression")
        mean_a = np.mean(result[industry == "A"])
        mean_b = np.mean(result[industry == "B"])
        assert abs(mean_a) < 1.0
        assert abs(mean_b) < 1.0

    def test_demean_subtracts_industry_mean(self):
        values = np.array([10.0, 12.0, 1.0, 3.0])
        industry = np.array(["A", "A", "B", "B"])
        result = neutralize(values, industry, method="demean")
        np.testing.assert_almost_equal(result[0], -1.0)
        np.testing.assert_almost_equal(result[1], 1.0)
        np.testing.assert_almost_equal(result[2], -1.0)
        np.testing.assert_almost_equal(result[3], 1.0)

    def test_intra_industry_standardizes(self):
        values = np.array([10.0, 12.0, 1.0, 3.0])
        industry = np.array(["A", "A", "B", "B"])
        result = neutralize(values, industry, method="intra_industry")
        assert abs(np.mean(result[:2])) < 1e-10
        assert abs(np.mean(result[2:])) < 1e-10

    def test_custom_func(self):
        values = np.array([10.0, 12.0])
        industry = np.array(["A", "A"])
        custom = lambda v, ind: v - v.mean()
        result = neutralize(values, industry, method="custom", custom_func=custom)
        np.testing.assert_almost_equal(result[0], -1.0)
        np.testing.assert_almost_equal(result[1], 1.0)


class TestNormalize:
    def test_min_max_range(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize(values, method="min_max")
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_zscore_truncated(self):
        np.random.seed(42)
        values = np.random.randn(100) * 10
        result = normalize(values, method="zscore")
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_rank_range(self):
        values = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        result = normalize(values, method="rank")
        assert result.min() >= -1.0
        assert result.max() <= 1.0
        assert result[0] == pytest.approx(1.0)   # value 5.0 is highest
        assert result[1] == pytest.approx(-1.0)  # value 1.0 is lowest


class TestGenerateCrossSectionSignals:
    def test_basic_pipeline(self, cs_long_data):
        spec = CrossSectionSignalSpec(
            winsorize_enabled=True,
            winsorize_method="mad",
            neutralize_enabled=False,
            normalize_method="min_max",
        )
        result = generate_cross_section_signals(
            feature_df=cs_long_data,
            spec=spec,
            stock_col="stock_id",
            date_col="date_id",
            feature_col="alpha1",
        )
        assert "signal" in result.columns
        assert result["signal"].min() >= -1.0
        assert result["signal"].max() <= 1.0
        assert len(result) == len(cs_long_data)

    def test_with_neutralization(self, cs_long_data):
        spec = CrossSectionSignalSpec(
            winsorize_enabled=True,
            neutralize_enabled=True,
            neutralize_method="demean",
            industry_col="industry",
            normalize_method="rank",
        )
        result = generate_cross_section_signals(
            feature_df=cs_long_data,
            spec=spec,
            stock_col="stock_id",
            date_col="date_id",
            feature_col="alpha1",
            industry_col="industry",
        )
        assert result["signal"].min() >= -1.0
        assert result["signal"].max() <= 1.0

    def test_winsorize_disabled(self, cs_long_data):
        spec = CrossSectionSignalSpec(
            winsorize_enabled=False,
            neutralize_enabled=False,
            normalize_method="min_max",
        )
        result = generate_cross_section_signals(
            feature_df=cs_long_data,
            spec=spec,
            stock_col="stock_id",
            date_col="date_id",
            feature_col="alpha1",
        )
        assert "signal" in result.columns
