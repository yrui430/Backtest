"""Tests for cross-section config dataclasses."""
from __future__ import annotations

import pytest
from backtest_api.cross_section.config import (
    CrossSectionConfig,
    CrossSectionSignalSpec,
    CrossSectionLabelSpec,
    CrossSectionExecutionSpec,
)


class TestCrossSectionConfig:
    def test_defaults(self):
        cfg = CrossSectionConfig(feature_path="f.parquet")
        assert cfg.feature_path == "f.parquet"
        assert cfg.use_raw_data is True
        assert cfg.stock_col == "stock_id"
        assert cfg.date_col == "date_id"
        assert cfg.time_col == "time_id"
        assert cfg.price_col == "close"
        assert cfg.exec_price_type == "close"
        assert cfg.data_format == "long"

    def test_validate_raw_mode_missing_raw_data_path(self):
        cfg = CrossSectionConfig(feature_path="f.parquet", use_raw_data=True, feature_cols=["a"])
        with pytest.raises(ValueError, match="raw_data_path"):
            cfg.validate()

    def test_validate_label_mode_missing_label_path(self):
        cfg = CrossSectionConfig(
            feature_path="f.parquet", use_raw_data=False, feature_cols=["a"]
        )
        with pytest.raises(ValueError, match="label_path"):
            cfg.validate()

    def test_validate_label_mode_missing_label_col(self):
        cfg = CrossSectionConfig(
            feature_path="f.parquet",
            use_raw_data=False,
            label_path="l.parquet",
            feature_cols=["a"],
        )
        with pytest.raises(ValueError, match="label_col"):
            cfg.validate()

    def test_validate_feature_cols_empty(self):
        cfg = CrossSectionConfig(
            feature_path="f.parquet",
            raw_data_path="r.parquet",
            feature_cols=[],
        )
        with pytest.raises(ValueError, match="feature_cols"):
            cfg.validate()

    def test_validate_raw_mode_ok(self):
        cfg = CrossSectionConfig(
            feature_path="f.parquet",
            raw_data_path="r.parquet",
            feature_cols=["alpha1"],
        )
        cfg.validate()

    def test_validate_label_mode_ok(self):
        cfg = CrossSectionConfig(
            feature_path="f.parquet",
            use_raw_data=False,
            label_path="l.parquet",
            label_col="ret",
            feature_cols=["alpha1"],
        )
        cfg.validate()


class TestCrossSectionSignalSpec:
    def test_defaults(self):
        spec = CrossSectionSignalSpec()
        assert spec.winsorize_enabled is True
        assert spec.winsorize_method == "mad"
        assert spec.winsorize_n_sigma == 3.0
        assert spec.neutralize_enabled is False
        assert spec.neutralize_method == "regression"
        assert spec.normalize_method == "min_max"

    def test_validate_neutralize_without_industry_col(self):
        spec = CrossSectionSignalSpec(neutralize_enabled=True)
        with pytest.raises(ValueError, match="industry_col"):
            spec.validate()

    def test_validate_custom_winsorize_without_func(self):
        spec = CrossSectionSignalSpec(winsorize_method="custom")
        with pytest.raises(ValueError, match="winsorize_func"):
            spec.validate()

    def test_validate_custom_neutralize_without_func(self):
        spec = CrossSectionSignalSpec(
            neutralize_enabled=True,
            neutralize_method="custom",
            industry_col="industry",
        )
        with pytest.raises(ValueError, match="neutralize_func"):
            spec.validate()


class TestCrossSectionLabelSpec:
    def test_defaults(self):
        spec = CrossSectionLabelSpec()
        assert spec.h == 1
        assert spec.lag == 1
        assert spec.decay_lags == [1, 2, 5]


class TestCrossSectionExecutionSpec:
    def test_defaults(self):
        spec = CrossSectionExecutionSpec()
        assert spec.weight_method == "group"
        assert spec.n_groups == 5
        assert spec.display_modes == ["long_only", "long_short"]
        assert spec.signal_direction == "momentum"
        assert spec.fee_rate == 0.00005
        assert spec.initial_capital == 1_000_000

    def test_validate_n_groups_too_small(self):
        spec = CrossSectionExecutionSpec(n_groups=1)
        with pytest.raises(ValueError, match="n_groups"):
            spec.validate()
