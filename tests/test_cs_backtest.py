"""Integration tests for CrossSectionBacktest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from backtest_api.cross_section import CrossSectionBacktest
from backtest_api.base import BacktestResult


class TestCrossSectionBacktestRawMode:
    def test_run_returns_result(self, cs_feature_parquet, cs_raw_data_parquet):
        bt = CrossSectionBacktest(
            feature_path=cs_feature_parquet,
            raw_data_path=cs_raw_data_parquet,
            use_raw_data=True,
            feature_cols=["alpha1"],
            stock_col="stock_id",
            date_col="date_id",
            time_col=None,
            price_col="close",
            n_groups=3,
            display_modes=["long_short"],
        )
        result = bt.run()
        assert result is not None
        assert isinstance(result, BacktestResult)

    def test_run_has_summary_tables(self, cs_feature_parquet, cs_raw_data_parquet):
        bt = CrossSectionBacktest(
            feature_path=cs_feature_parquet,
            raw_data_path=cs_raw_data_parquet,
            use_raw_data=True,
            feature_cols=["alpha1"],
            stock_col="stock_id",
            date_col="date_id",
            time_col=None,
            price_col="close",
            n_groups=3,
            display_modes=["long_short"],
        )
        result = bt.run()
        assert len(result.summary_tables) > 0

    def test_run_has_figures(self, cs_feature_parquet, cs_raw_data_parquet):
        bt = CrossSectionBacktest(
            feature_path=cs_feature_parquet,
            raw_data_path=cs_raw_data_parquet,
            use_raw_data=True,
            feature_cols=["alpha1"],
            stock_col="stock_id",
            date_col="date_id",
            time_col=None,
            price_col="close",
            n_groups=3,
            display_modes=["long_short"],
        )
        result = bt.run()
        assert len(result.figures) > 0

    def test_multiple_features(self, cs_feature_parquet, cs_raw_data_parquet):
        bt = CrossSectionBacktest(
            feature_path=cs_feature_parquet,
            raw_data_path=cs_raw_data_parquet,
            use_raw_data=True,
            feature_cols=["alpha1", "alpha2"],
            stock_col="stock_id",
            date_col="date_id",
            time_col=None,
            price_col="close",
            n_groups=3,
            display_modes=["long_short"],
        )
        result = bt.run()
        assert len(result.summary_tables) >= 2


class TestCrossSectionBacktestLabelMode:
    def test_run_with_labels(self, cs_feature_parquet, cs_label_parquet):
        bt = CrossSectionBacktest(
            feature_path=cs_feature_parquet,
            use_raw_data=False,
            label_path=cs_label_parquet,
            label_col="ret",
            feature_cols=["alpha1"],
            stock_col="stock_id",
            date_col="date_id",
            time_col=None,
            price_col="close",
            n_groups=3,
            display_modes=["long_only"],
        )
        result = bt.run()
        assert result is not None
        assert len(result.summary_tables) > 0


class TestCrossSectionBacktestValidation:
    def test_missing_raw_data_path(self, cs_feature_parquet):
        bt = CrossSectionBacktest(
            feature_path=cs_feature_parquet,
            use_raw_data=True,
            feature_cols=["alpha1"],
        )
        with pytest.raises(ValueError, match="raw_data_path"):
            bt.run()

    def test_missing_label_path(self, cs_feature_parquet):
        bt = CrossSectionBacktest(
            feature_path=cs_feature_parquet,
            use_raw_data=False,
            feature_cols=["alpha1"],
        )
        with pytest.raises(ValueError, match="label_path"):
            bt.run()
