import pytest
from backtest_api.config import TimingBacktestConfig, LabelSpec, SignalSpec, ExecutionSpec


class TestTimingBacktestConfig:
    def test_label_mode_requires_label_path(self):
        """use_raw_data=False requires label_path and label_col."""
        with pytest.raises(ValueError, match="label_path"):
            TimingBacktestConfig(
                feature_path="f.parquet",
                time_col="timestamp",
                feature_cols=["alpha"],
                use_raw_data=False,
            ).validate()

    def test_raw_mode_requires_price_col(self):
        """use_raw_data=True requires raw_data_path and price_col."""
        with pytest.raises(ValueError, match="raw_data_path"):
            TimingBacktestConfig(
                feature_path="f.parquet",
                time_col="timestamp",
                feature_cols=["alpha"],
                use_raw_data=True,
            ).validate()

    def test_valid_label_mode(self):
        cfg = TimingBacktestConfig(
            feature_path="f.parquet",
            label_path="l.parquet",
            label_col="ret",
            time_col="timestamp",
            feature_cols=["alpha"],
            use_raw_data=False,
        )
        cfg.validate()  # should not raise

    def test_valid_raw_mode(self):
        cfg = TimingBacktestConfig(
            feature_path="f.parquet",
            raw_data_path="ohlcv.parquet",
            price_col="close",
            time_col="timestamp",
            feature_cols=["alpha"],
            use_raw_data=True,
            exec_price_type="close",
        )
        cfg.validate()  # should not raise


class TestSignalSpec:
    def test_custom_requires_mapper(self):
        with pytest.raises(ValueError, match="signal_mapper"):
            SignalSpec(method="custom", signal_mapper=None).validate()

    def test_threshold_requires_values(self):
        with pytest.raises(ValueError, match="threshold"):
            SignalSpec(method="threshold").validate()

    def test_quantile_defaults_valid(self):
        SignalSpec(method="quantile").validate()  # should not raise


class TestLabelSpec:
    def test_defaults(self):
        spec = LabelSpec()
        assert spec.label_horizon == 1
        assert spec.lag == 1
        assert spec.exec_price_type == "close"

    def test_display_labels_default(self):
        spec = LabelSpec()
        assert spec.display_labels == [1]


class TestExecutionSpec:
    def test_defaults(self):
        spec = ExecutionSpec()
        assert spec.fee_rate == 0.00005
        assert spec.show_before_fee is True
        assert spec.show_after_fee is True
        assert spec.hurdle_enabled is False
