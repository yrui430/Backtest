"""Tests for data_loader cross-section extensions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_api.data_loader import (
    wide_to_long,
    load_directory,
    load_cross_section_data,
)


class TestWideToLong:
    def test_basic_conversion(self, cs_wide_data):
        result = wide_to_long(
            cs_wide_data,
            time_col="date_id",
            stock_col="stock_id",
            value_col="feature",
        )
        assert "stock_id" in result.columns
        assert "date_id" in result.columns
        assert "feature" in result.columns
        assert len(result) == 30  # 10 dates * 3 stocks

    def test_preserves_values(self, cs_wide_data):
        result = wide_to_long(
            cs_wide_data,
            time_col="date_id",
            stock_col="stock_id",
            value_col="feature",
        )
        stock_a = result[result["stock_id"] == "A"]["feature"].values
        expected = cs_wide_data["A"].values
        np.testing.assert_array_almost_equal(stock_a, expected)


class TestLoadDirectory:
    def test_loads_all_files(self, cs_multi_file_dir):
        result = load_directory(cs_multi_file_dir, stock_col="stock_id")
        assert len(result["stock_id"].unique()) == 3
        assert set(result["stock_id"].unique()) == {"A", "B", "C"}

    def test_has_stock_col(self, cs_multi_file_dir):
        result = load_directory(cs_multi_file_dir, stock_col="stock_id")
        assert "stock_id" in result.columns


class TestLoadCrossSectionData:
    def test_long_format(self, cs_feature_parquet):
        result = load_cross_section_data(
            cs_feature_parquet,
            data_format="long",
        )
        assert "stock_id" in result.columns
        assert "alpha1" in result.columns

    def test_wide_format(self, tmp_path, cs_wide_data):
        path = tmp_path / "wide.parquet"
        cs_wide_data.to_parquet(path, index=False)
        result = load_cross_section_data(
            str(path),
            data_format="wide",
            time_col="date_id",
            stock_col="stock_id",
            value_col="feature",
        )
        assert "stock_id" in result.columns
        assert len(result) == 30

    def test_multi_file_format(self, cs_multi_file_dir):
        result = load_cross_section_data(
            cs_multi_file_dir,
            data_format="multi_file",
            stock_col="stock_id",
        )
        assert len(result["stock_id"].unique()) == 3
