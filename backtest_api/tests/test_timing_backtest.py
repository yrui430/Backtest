import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from backtest_api.timing.backtest import TimingBacktest


@pytest.fixture
def sample_data(tmp_path):
    """Create sample feature + label parquet files."""
    n = 300
    ts = pd.date_range("2026-01-01", periods=n, freq="1min")
    feat_df = pd.DataFrame({
        "timestamp": ts,
        "alpha_001": np.random.RandomState(42).randn(n),
        "alpha_002": np.cumsum(np.random.RandomState(43).randn(n) * 0.1),
    })
    label_df = pd.DataFrame({
        "timestamp": ts,
        "ret_1bar": np.random.RandomState(44).randn(n) * 0.01,
    })
    feat_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.parquet"
    feat_df.to_parquet(feat_path, index=False)
    label_df.to_parquet(label_path, index=False)
    return str(feat_path), str(label_path)


@pytest.fixture
def sample_raw_data(tmp_path):
    """Create sample feature + raw OHLCV parquet files."""
    n = 300
    ts = pd.date_range("2026-01-01", periods=n, freq="1min")
    feat_df = pd.DataFrame({
        "timestamp": ts,
        "alpha_001": np.random.RandomState(42).randn(n),
    })
    prices = 100.0 + np.cumsum(np.random.RandomState(45).randn(n) * 0.5)
    raw_df = pd.DataFrame({
        "timestamp": ts,
        "close": prices,
        "vwap": prices + np.random.RandomState(46).randn(n) * 0.1,
    })
    feat_path = tmp_path / "features.parquet"
    raw_path = tmp_path / "ohlcv.parquet"
    feat_df.to_parquet(feat_path, index=False)
    raw_df.to_parquet(raw_path, index=False)
    return str(feat_path), str(raw_path)


class TestTimingBacktestWithLabel:
    @patch("matplotlib.pyplot.show")
    def test_run_basic(self, mock_show, sample_data):
        feat_path, label_path = sample_data
        bt = TimingBacktest(
            feature_path=feat_path,
            label_path=label_path,
            time_col="timestamp",
            feature_cols=["alpha_001"],
            label_col="ret_1bar",
        )
        result = bt.run()
        assert len(result.summary_tables) > 0
        assert len(result.figures) > 0

    @patch("matplotlib.pyplot.show")
    def test_multi_feature(self, mock_show, sample_data):
        feat_path, label_path = sample_data
        bt = TimingBacktest(
            feature_path=feat_path,
            label_path=label_path,
            time_col="timestamp",
            feature_cols=["alpha_001", "alpha_002"],
            label_col="ret_1bar",
        )
        result = bt.run()
        assert len(result.summary_tables) >= 2

    @patch("matplotlib.pyplot.show")
    def test_multi_display_labels(self, mock_show, sample_data):
        feat_path, label_path = sample_data
        bt = TimingBacktest(
            feature_path=feat_path,
            label_path=label_path,
            time_col="timestamp",
            feature_cols=["alpha_001"],
            label_col="ret_1bar",
            display_labels=[1, 2],
        )
        result = bt.run()
        assert len(result.summary_tables) > 0


class TestTimingBacktestWithRawData:
    @patch("matplotlib.pyplot.show")
    def test_run_raw_close(self, mock_show, sample_raw_data):
        feat_path, raw_path = sample_raw_data
        bt = TimingBacktest(
            feature_path=feat_path,
            raw_data_path=raw_path,
            time_col="timestamp",
            feature_cols=["alpha_001"],
            price_col="close",
            use_raw_data=True,
            exec_price_type="close",
        )
        result = bt.run()
        assert len(result.summary_tables) > 0

    @patch("matplotlib.pyplot.show")
    def test_run_raw_vwap(self, mock_show, sample_raw_data):
        feat_path, raw_path = sample_raw_data
        bt = TimingBacktest(
            feature_path=feat_path,
            raw_data_path=raw_path,
            time_col="timestamp",
            feature_cols=["alpha_001"],
            price_col="vwap",
            use_raw_data=True,
            exec_price_type="vwap",
        )
        result = bt.run()
        assert len(result.summary_tables) > 0


class TestTimingBacktestMisaligned:
    def test_misaligned_aborts(self, tmp_path, capsys):
        ts1 = pd.date_range("2026-01-01", periods=100, freq="1min")
        ts2 = pd.date_range("2026-06-01", periods=100, freq="1min")
        feat_df = pd.DataFrame({"timestamp": ts1, "alpha": np.zeros(100)})
        label_df = pd.DataFrame({"timestamp": ts2, "ret": np.zeros(100)})
        feat_path = tmp_path / "f.parquet"
        label_path = tmp_path / "l.parquet"
        feat_df.to_parquet(feat_path, index=False)
        label_df.to_parquet(label_path, index=False)

        bt = TimingBacktest(
            feature_path=str(feat_path),
            label_path=str(label_path),
            time_col="timestamp",
            feature_cols=["alpha"],
            label_col="ret",
        )
        result = bt.run()
        assert result is None
        captured = capsys.readouterr()
        assert "时间戳无法对齐" in captured.out
