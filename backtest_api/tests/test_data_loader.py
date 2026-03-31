import pandas as pd
import numpy as np
import pytest
import tempfile
from pathlib import Path
from backtest_api.data_loader import load_file, align_timestamps


class TestLoadFile:
    def test_load_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)
        result = load_file(str(path))
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_load_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        result = load_file(str(path))
        assert len(result) == 2

    def test_load_h5(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.h5"
        df.to_hdf(path, key="data", mode="w")
        result = load_file(str(path))
        assert len(result) == 2

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("data")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_file(str(path))


class TestAlignTimestamps:
    def test_aligned(self):
        ts = pd.date_range("2026-01-01", periods=10, freq="1min")
        df1 = pd.DataFrame({"timestamp": ts, "feat": np.arange(10.0)})
        df2 = pd.DataFrame({"timestamp": ts, "label": np.arange(10.0)})
        r1, r2 = align_timestamps(df1, df2, time_col="timestamp")
        assert len(r1) == 10
        assert len(r2) == 10

    def test_partial_overlap(self):
        ts1 = pd.date_range("2026-01-01", periods=10, freq="1min")
        ts2 = pd.date_range("2026-01-01 00:05", periods=10, freq="1min")
        df1 = pd.DataFrame({"timestamp": ts1, "feat": np.arange(10.0)})
        df2 = pd.DataFrame({"timestamp": ts2, "label": np.arange(10.0)})
        r1, r2 = align_timestamps(df1, df2, time_col="timestamp")
        assert len(r1) == 5
        assert len(r2) == 5

    def test_no_overlap_aborts(self, capsys):
        ts1 = pd.date_range("2026-01-01", periods=5, freq="1min")
        ts2 = pd.date_range("2026-02-01", periods=5, freq="1min")
        df1 = pd.DataFrame({"timestamp": ts1, "feat": np.arange(5.0)})
        df2 = pd.DataFrame({"timestamp": ts2, "label": np.arange(5.0)})
        result = align_timestamps(df1, df2, time_col="timestamp")
        assert result is None
        captured = capsys.readouterr()
        assert "时间戳无法对齐" in captured.out

    def test_low_overlap_aborts(self, capsys):
        ts1 = pd.date_range("2026-01-01", periods=100, freq="1min")
        ts2_part = pd.date_range("2026-01-01", periods=10, freq="1min")
        ts2_other = pd.date_range("2026-06-01", periods=90, freq="1min")
        # Use append method compatible with newer pandas
        ts2 = ts2_part.append(ts2_other)
        df1 = pd.DataFrame({"timestamp": ts1, "feat": np.arange(100.0)})
        df2 = pd.DataFrame({"timestamp": ts2, "label": np.arange(100.0)})
        result = align_timestamps(df1, df2, time_col="timestamp")
        assert result is None
        captured = capsys.readouterr()
        assert "时间戳无法对齐" in captured.out
