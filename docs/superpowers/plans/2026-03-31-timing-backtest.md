# Timing Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a timing backtest API that takes features + labels (or OHLCV), generates signals, simulates trades, and outputs performance metrics + charts.

**Architecture:** Strategy pattern with `BaseBacktest` abstract base class. `TimingBacktest` inherits and implements single-asset signal-based entry/exit. All rolling computations use numba, never pandas rolling.

**Tech Stack:** Python 3.10+, pandas, numpy, numba, matplotlib, dataclasses

---

## File Structure

```
backtest_api/
├── __init__.py              # Package init, version
├── base.py                  # BaseBacktest ABC
├── config.py                # All dataclass configs (TimingBacktestConfig, LabelSpec, SignalSpec, ExecutionSpec)
├── data_loader.py           # File reading (h5/csv/parquet), timestamp alignment check
├── numba_utils.py           # @njit rolling_quantile, rolling_pearson, rolling_spearman
├── metrics.py               # Sharpe, Sortino, MDD, turnover, IC, Rank IC, IR
├── report.py                # Table formatting + matplotlib chart generation
├── timing/
│   ├── __init__.py          # Re-export TimingBacktest
│   ├── backtest.py          # TimingBacktest(BaseBacktest) main class
│   ├── label.py             # compute_labels_from_raw() — price diff
│   ├── signal.py            # quantile/threshold/custom signal generation
│   └── executor.py          # Position stacking, fee calc, PnL series
└── tests/
    ├── __init__.py
    ├── test_numba_utils.py
    ├── test_data_loader.py
    ├── test_config.py
    ├── test_metrics.py
    ├── test_label.py
    ├── test_signal.py
    ├── test_executor.py
    └── test_timing_backtest.py
```

---

### Task 1: Project Scaffold & Config Dataclasses

**Files:**
- Create: `backtest_api/__init__.py`
- Create: `backtest_api/config.py`
- Create: `backtest_api/tests/__init__.py`
- Create: `backtest_api/tests/test_config.py`

- [ ] **Step 1: Write failing tests for config validation**

```python
# backtest_api/tests/test_config.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_api'`

- [ ] **Step 3: Implement config dataclasses**

```python
# backtest_api/__init__.py
"""backtest_api — Backtesting framework for timing, cross-sectional, and HF strategies."""
```

```python
# backtest_api/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

import pandas as pd


@dataclass
class TimingBacktestConfig:
    # ---- required ----
    feature_path: str
    time_col: str = "timestamp"
    feature_cols: List[str] = field(default_factory=list)

    # ---- mode A: has label ----
    label_path: Optional[str] = None
    label_col: Optional[str] = None

    # ---- mode B: raw data ----
    raw_data_path: Optional[str] = None
    price_col: Optional[str] = None

    use_raw_data: bool = False
    exec_price_type: Literal["close", "vwap"] = "close"

    def validate(self) -> None:
        if not self.feature_cols:
            raise ValueError("feature_cols must not be empty.")
        if not self.use_raw_data:
            if not self.label_path:
                raise ValueError("label_path is required when use_raw_data=False.")
            if not self.label_col:
                raise ValueError("label_col is required when use_raw_data=False.")
        else:
            if not self.raw_data_path:
                raise ValueError("raw_data_path is required when use_raw_data=True.")
            if not self.price_col:
                raise ValueError("price_col is required when use_raw_data=True.")


@dataclass
class LabelSpec:
    label_horizon: int = 1
    lag: int = 1
    exec_price_type: Literal["close", "vwap"] = "close"
    display_labels: List[int] = field(default_factory=lambda: [1])


@dataclass
class SignalSpec:
    method: Literal["quantile", "threshold", "custom"] = "quantile"

    upper_quantile: float = 0.8
    lower_quantile: float = 0.2
    rolling_window: int = 100

    upper_threshold: Optional[float] = None
    lower_threshold: Optional[float] = None

    signal_direction: Literal["momentum", "mean_reversion"] = "momentum"

    signal_mapper: Optional[Callable[[pd.Series], pd.Series]] = None

    display_modes: List[Literal["long_only", "short_only", "long_short"]] = field(
        default_factory=lambda: ["long_only", "short_only", "long_short"]
    )

    def validate(self) -> None:
        if self.method == "custom" and self.signal_mapper is None:
            raise ValueError("signal_mapper is required when method='custom'.")
        if self.method == "threshold":
            if self.upper_threshold is None or self.lower_threshold is None:
                raise ValueError(
                    "upper_threshold and lower_threshold are required when method='threshold'."
                )


@dataclass
class ExecutionSpec:
    fee_rate: float = 0.00005
    show_before_fee: bool = True
    show_after_fee: bool = True
    hurdle_enabled: bool = False
    hurdle_func: Optional[Callable] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_config.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/__init__.py backtest_api/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add config dataclasses with validation"
```

---

### Task 2: Numba Rolling Utilities

**Files:**
- Create: `backtest_api/numba_utils.py`
- Create: `backtest_api/tests/test_numba_utils.py`

- [ ] **Step 1: Write failing tests for numba functions**

```python
# backtest_api/tests/test_numba_utils.py
import numpy as np
import pytest
from backtest_api.numba_utils import rolling_quantile, rolling_pearson, rolling_spearman


class TestRollingQuantile:
    def test_basic(self):
        arr = np.arange(1.0, 11.0)  # [1,2,...,10]
        result = rolling_quantile(arr, window=5, q=0.5)
        # First 4 values should be NaN (not enough data)
        assert np.isnan(result[:4]).all()
        # At index 4, window is [1,2,3,4,5], median=3.0
        assert result[4] == pytest.approx(3.0, abs=0.5)

    def test_all_same(self):
        arr = np.ones(20)
        result = rolling_quantile(arr, window=10, q=0.8)
        assert np.isnan(result[:9]).all()
        assert result[9] == pytest.approx(1.0)

    def test_nan_handling(self):
        arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        result = rolling_quantile(arr, window=3, q=0.5)
        assert result.shape == arr.shape


class TestRollingPearson:
    def test_perfect_correlation(self):
        x = np.arange(1.0, 101.0)
        y = x * 2.0 + 1.0
        result = rolling_pearson(x, y, window=20)
        assert np.isnan(result[:19]).all()
        # Perfect linear relationship → corr ≈ 1.0
        assert result[19] == pytest.approx(1.0, abs=1e-6)

    def test_no_correlation(self):
        rng = np.random.RandomState(42)
        x = rng.randn(500)
        y = rng.randn(500)
        result = rolling_pearson(x, y, window=100)
        # Random data → corr should be near 0
        valid = result[~np.isnan(result)]
        assert np.abs(valid.mean()) < 0.15


class TestRollingSpearman:
    def test_perfect_rank_correlation(self):
        x = np.arange(1.0, 51.0)
        y = x * 3.0
        result = rolling_spearman(x, y, window=20)
        assert np.isnan(result[:19]).all()
        assert result[19] == pytest.approx(1.0, abs=1e-6)

    def test_inverse_rank_correlation(self):
        x = np.arange(1.0, 51.0)
        y = -x
        result = rolling_spearman(x, y, window=20)
        assert result[19] == pytest.approx(-1.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_numba_utils.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement numba rolling functions**

```python
# backtest_api/numba_utils.py
from __future__ import annotations

import numpy as np
import numba


@numba.njit
def rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """Rolling quantile using insertion sort within each window. No pandas."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        # Collect non-NaN values in window
        buf = np.empty(window, dtype=np.float64)
        count = 0
        for j in range(i - window + 1, i + 1):
            v = arr[j]
            if not np.isnan(v):
                buf[count] = v
                count += 1
        if count < 2:
            continue
        # Insertion sort
        for a in range(1, count):
            key = buf[a]
            b = a - 1
            while b >= 0 and buf[b] > key:
                buf[b + 1] = buf[b]
                b -= 1
            buf[b + 1] = key
        # Linear interpolation for quantile
        pos = q * (count - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, count - 1)
        frac = pos - lo
        out[i] = buf[lo] * (1.0 - frac) + buf[hi] * frac
    return out


@numba.njit
def rolling_pearson(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation. No pandas."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        cnt = 0
        for j in range(i - window + 1, i + 1):
            xv = x[j]
            yv = y[j]
            if np.isnan(xv) or np.isnan(yv):
                continue
            sx += xv
            sy += yv
            sxx += xv * xv
            syy += yv * yv
            sxy += xv * yv
            cnt += 1
        if cnt < 3:
            continue
        mx = sx / cnt
        my = sy / cnt
        cov = sxy / cnt - mx * my
        vx = sxx / cnt - mx * mx
        vy = syy / cnt - my * my
        if vx <= 0.0 or vy <= 0.0:
            continue
        out[i] = cov / np.sqrt(vx * vy)
    return out


@numba.njit
def _rank_array(arr: np.ndarray, n: int) -> np.ndarray:
    """Rank values in arr[0:n], handling ties with average rank."""
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = i
    # Insertion sort by value
    for i in range(1, n):
        key_idx = idx[i]
        key_val = arr[key_idx]
        j = i - 1
        while j >= 0 and arr[idx[j]] > key_val:
            idx[j + 1] = idx[j]
            j -= 1
        idx[j + 1] = key_idx
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[idx[j + 1]] == arr[idx[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


@numba.njit
def rolling_spearman(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Spearman rank correlation. No pandas."""
    n = len(x)
    out = np.full(n, np.nan)
    xbuf = np.empty(window, dtype=np.float64)
    ybuf = np.empty(window, dtype=np.float64)
    for i in range(window - 1, n):
        cnt = 0
        for j in range(i - window + 1, i + 1):
            xv = x[j]
            yv = y[j]
            if np.isnan(xv) or np.isnan(yv):
                continue
            xbuf[cnt] = xv
            ybuf[cnt] = yv
            cnt += 1
        if cnt < 3:
            continue
        xranks = _rank_array(xbuf, cnt)
        yranks = _rank_array(ybuf, cnt)
        # Pearson on ranks
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for k in range(cnt):
            rx = xranks[k]
            ry = yranks[k]
            sx += rx
            sy += ry
            sxx += rx * rx
            syy += ry * ry
            sxy += rx * ry
        mx = sx / cnt
        my = sy / cnt
        cov = sxy / cnt - mx * my
        vx = sxx / cnt - mx * mx
        vy = syy / cnt - my * my
        if vx <= 0.0 or vy <= 0.0:
            continue
        out[i] = cov / np.sqrt(vx * vy)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_numba_utils.py -v`
Expected: All 7 tests PASS (first run may be slow due to numba JIT compilation)

- [ ] **Step 5: Commit**

```bash
git add backtest_api/numba_utils.py tests/test_numba_utils.py
git commit -m "feat: add numba-accelerated rolling quantile, pearson, spearman"
```

---

### Task 3: Data Loader & Timestamp Alignment

**Files:**
- Create: `backtest_api/data_loader.py`
- Create: `backtest_api/tests/test_data_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# backtest_api/tests/test_data_loader.py
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
        # Overlap: 00:05 to 00:09 = 5 rows
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
        ts2 = ts2_part.append(ts2_other)
        df1 = pd.DataFrame({"timestamp": ts1, "feat": np.arange(100.0)})
        df2 = pd.DataFrame({"timestamp": ts2, "label": np.arange(100.0)})
        result = align_timestamps(df1, df2, time_col="timestamp")
        # 10/100 = 10% overlap < 50% → abort
        assert result is None
        captured = capsys.readouterr()
        assert "时间戳无法对齐" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_data_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement data_loader**

```python
# backtest_api/data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_file(path: str) -> pd.DataFrame:
    """Load a DataFrame from parquet, csv, or h5 file."""
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".h5", ".hdf5"):
        return pd.read_hdf(path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def align_timestamps(
    df_feature: pd.DataFrame,
    df_label: pd.DataFrame,
    time_col: str,
    min_overlap_ratio: float = 0.5,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Align two DataFrames on timestamp. Returns None and prints if overlap too low."""
    ts_feat = set(df_feature[time_col])
    ts_label = set(df_label[time_col])
    common = ts_feat & ts_label

    total = max(len(ts_feat), len(ts_label))
    overlap_ratio = len(common) / total if total > 0 else 0.0

    if len(common) == 0 or overlap_ratio < min_overlap_ratio:
        feat_min = df_feature[time_col].min()
        feat_max = df_feature[time_col].max()
        label_min = df_label[time_col].min()
        label_max = df_label[time_col].max()
        print(
            f"时间戳无法对齐，请对齐数据之后再来。"
            f"Feature 时间范围: [{feat_min}, {feat_max}], "
            f"Label 时间范围: [{label_min}, {label_max}], "
            f"交集比例: {overlap_ratio:.1%}"
        )
        return None

    common_sorted = sorted(common)
    df_f = df_feature[df_feature[time_col].isin(common)].sort_values(time_col).reset_index(drop=True)
    df_l = df_label[df_label[time_col].isin(common)].sort_values(time_col).reset_index(drop=True)
    return df_f, df_l
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_data_loader.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/data_loader.py tests/test_data_loader.py
git commit -m "feat: add data loader with timestamp alignment check"
```

---

### Task 4: Label Computation

**Files:**
- Create: `backtest_api/timing/__init__.py`
- Create: `backtest_api/timing/label.py`
- Create: `backtest_api/tests/test_label.py`

- [ ] **Step 1: Write failing tests**

```python
# backtest_api/tests/test_label.py
import numpy as np
import pandas as pd
import pytest
from backtest_api.timing.label import compute_labels_from_raw


class TestComputeLabelsFromRaw:
    def test_horizon_1(self):
        prices = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0])
        result = compute_labels_from_raw(prices, label_horizon=1)
        # label[t] = price[t+1] - price[t]
        expected = pd.Series([1.0, 2.0, -1.0, 3.0, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_horizon_2(self):
        prices = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0])
        result = compute_labels_from_raw(prices, label_horizon=2)
        # label[t] = price[t+2] - price[t]
        expected = pd.Series([3.0, 1.0, 2.0, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_horizon_5(self):
        prices = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0, 110.0])
        result = compute_labels_from_raw(prices, label_horizon=5)
        # label[t] = price[t+5] - price[t]
        expected = pd.Series([10.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_returns_same_length(self):
        prices = pd.Series(np.arange(50.0))
        for h in [1, 2, 5]:
            result = compute_labels_from_raw(prices, label_horizon=h)
            assert len(result) == len(prices)
            # Last h values should be NaN
            assert np.isnan(result.iloc[-h:]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_label.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement label computation**

```python
# backtest_api/timing/__init__.py
"""Timing backtest module."""

# backtest_api/timing/label.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_labels_from_raw(prices: pd.Series, label_horizon: int) -> pd.Series:
    """Compute label as price[t+h] - price[t] (first difference with horizon h)."""
    shifted = prices.shift(-label_horizon)
    label = shifted - prices
    return label.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_label.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/timing/__init__.py backtest_api/timing/label.py tests/test_label.py
git commit -m "feat: add label computation from raw price data"
```

---

### Task 5: Signal Generation

**Files:**
- Create: `backtest_api/timing/signal.py`
- Create: `backtest_api/tests/test_signal.py`

- [ ] **Step 1: Write failing tests**

```python
# backtest_api/tests/test_signal.py
import numpy as np
import pandas as pd
import pytest
from backtest_api.timing.signal import generate_signals
from backtest_api.config import SignalSpec


class TestQuantileSignal:
    def test_basic_momentum(self):
        np.random.seed(42)
        feature = pd.Series(np.random.randn(200))
        spec = SignalSpec(
            method="quantile",
            upper_quantile=0.8,
            lower_quantile=0.2,
            rolling_window=50,
            signal_direction="momentum",
        )
        signal = generate_signals(feature, spec)
        assert len(signal) == 200
        assert set(signal.dropna().unique()).issubset({-1, 0, 1})
        # First 49 values should be NaN (not enough window)
        assert np.isnan(signal.iloc[:49]).all()

    def test_mean_reversion_flips(self):
        np.random.seed(42)
        feature = pd.Series(np.random.randn(200))
        spec_mom = SignalSpec(method="quantile", rolling_window=50, signal_direction="momentum")
        spec_mr = SignalSpec(method="quantile", rolling_window=50, signal_direction="mean_reversion")
        sig_mom = generate_signals(feature, spec_mom)
        sig_mr = generate_signals(feature, spec_mr)
        # Mean reversion = -1 * momentum (where not NaN or 0)
        valid = sig_mom.notna() & (sig_mom != 0)
        np.testing.assert_array_equal(sig_mr[valid].values, -sig_mom[valid].values)


class TestThresholdSignal:
    def test_basic(self):
        feature = pd.Series([0.5, 1.5, -1.5, 0.0, 2.0, -2.0])
        spec = SignalSpec(
            method="threshold",
            upper_threshold=1.0,
            lower_threshold=-1.0,
            signal_direction="momentum",
        )
        signal = generate_signals(feature, spec)
        expected = pd.Series([0, 1, -1, 0, 1, -1])
        pd.testing.assert_series_equal(signal, expected, check_names=False, check_dtype=False)

    def test_mean_reversion(self):
        feature = pd.Series([0.5, 1.5, -1.5])
        spec = SignalSpec(
            method="threshold",
            upper_threshold=1.0,
            lower_threshold=-1.0,
            signal_direction="mean_reversion",
        )
        signal = generate_signals(feature, spec)
        expected = pd.Series([0, -1, 1])
        pd.testing.assert_series_equal(signal, expected, check_names=False, check_dtype=False)


class TestCustomSignal:
    def test_custom_mapper(self):
        def my_mapper(s: pd.Series) -> pd.Series:
            return (s > 0).astype(int) * 2 - 1  # >0 → +1, <=0 → -1

        feature = pd.Series([1.0, -1.0, 0.5, -0.5])
        spec = SignalSpec(method="custom", signal_mapper=my_mapper)
        signal = generate_signals(feature, spec)
        expected = pd.Series([1, -1, 1, -1])
        pd.testing.assert_series_equal(signal, expected, check_names=False, check_dtype=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_signal.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement signal generation**

```python
# backtest_api/timing/signal.py
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_api.config import SignalSpec
from backtest_api.numba_utils import rolling_quantile


def generate_signals(feature: pd.Series, spec: SignalSpec) -> pd.Series:
    """Generate trading signals from a feature series based on SignalSpec."""
    if spec.method == "quantile":
        signal = _quantile_signal(feature, spec)
    elif spec.method == "threshold":
        signal = _threshold_signal(feature, spec)
    elif spec.method == "custom":
        signal = spec.signal_mapper(feature)
        return signal.reset_index(drop=True)
    else:
        raise ValueError(f"Unknown signal method: {spec.method}")

    if spec.signal_direction == "mean_reversion":
        signal = signal * -1

    return signal


def _quantile_signal(feature: pd.Series, spec: SignalSpec) -> pd.Series:
    arr = feature.values.astype(np.float64)
    upper = rolling_quantile(arr, window=spec.rolling_window, q=spec.upper_quantile)
    lower = rolling_quantile(arr, window=spec.rolling_window, q=spec.lower_quantile)

    signal = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(arr[i]):
            signal[i] = np.nan
        elif arr[i] > upper[i]:
            signal[i] = 1.0
        elif arr[i] < lower[i]:
            signal[i] = -1.0
        else:
            signal[i] = 0.0

    return pd.Series(signal)


def _threshold_signal(feature: pd.Series, spec: SignalSpec) -> pd.Series:
    arr = feature.values.astype(np.float64)
    signal = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            signal[i] = np.nan
        elif arr[i] > spec.upper_threshold:
            signal[i] = 1.0
        elif arr[i] < spec.lower_threshold:
            signal[i] = -1.0
        else:
            signal[i] = 0.0
    return pd.Series(signal)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_signal.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/timing/signal.py tests/test_signal.py
git commit -m "feat: add signal generation (quantile, threshold, custom + momentum/mean_reversion)"
```

---

### Task 6: Executor — Position Stacking & PnL

**Files:**
- Create: `backtest_api/timing/executor.py`
- Create: `backtest_api/tests/test_executor.py`

- [ ] **Step 1: Write failing tests**

```python
# backtest_api/tests/test_executor.py
import numpy as np
import pandas as pd
import pytest
from backtest_api.timing.executor import execute_timing_backtest
from backtest_api.config import LabelSpec, ExecutionSpec


class TestExecuteTimingBacktest:
    def _make_data(self):
        n = 20
        ts = pd.date_range("2026-01-01", periods=n, freq="1min")
        prices = pd.Series(100.0 + np.cumsum(np.random.RandomState(42).randn(n) * 0.5))
        signals = pd.Series(np.zeros(n))
        signals.iloc[2] = 1    # long at t=2
        signals.iloc[5] = -1   # short at t=5
        signals.iloc[10] = 1   # long at t=10
        return ts, prices, signals

    def test_horizon_1_basic_shape(self):
        ts, prices, signals = self._make_data()
        label_spec = LabelSpec(label_horizon=1, lag=1, exec_price_type="close")
        exec_spec = ExecutionSpec(fee_rate=0.00005)
        result = execute_timing_backtest(
            timestamps=ts, prices=prices, signals=signals,
            label_spec=label_spec, exec_spec=exec_spec,
        )
        assert "pnl_before_fee" in result.columns
        assert "pnl_after_fee" in result.columns
        assert "position" in result.columns
        assert len(result) == len(ts)

    def test_horizon_2_position_weight(self):
        ts, prices, signals = self._make_data()
        label_spec = LabelSpec(label_horizon=2, lag=1, exec_price_type="close")
        exec_spec = ExecutionSpec(fee_rate=0.0)
        result = execute_timing_backtest(
            timestamps=ts, prices=prices, signals=signals,
            label_spec=label_spec, exec_spec=exec_spec,
        )
        # With horizon=2, each position is 0.5 weight
        # At t=3 (lag=1 after signal at t=2), position should be 0.5
        assert result["position"].iloc[3] == pytest.approx(0.5)

    def test_zero_fee(self):
        ts, prices, signals = self._make_data()
        label_spec = LabelSpec(label_horizon=1, lag=1)
        exec_spec = ExecutionSpec(fee_rate=0.0)
        result = execute_timing_backtest(
            timestamps=ts, prices=prices, signals=signals,
            label_spec=label_spec, exec_spec=exec_spec,
        )
        # With zero fee, before and after should be identical
        np.testing.assert_array_almost_equal(
            result["pnl_before_fee"].values,
            result["pnl_after_fee"].values,
        )

    def test_fee_reduces_pnl(self):
        ts, prices, signals = self._make_data()
        label_spec = LabelSpec(label_horizon=1, lag=1)
        exec_spec = ExecutionSpec(fee_rate=0.001)
        result = execute_timing_backtest(
            timestamps=ts, prices=prices, signals=signals,
            label_spec=label_spec, exec_spec=exec_spec,
        )
        cum_before = result["pnl_before_fee"].cumsum().iloc[-1]
        cum_after = result["pnl_after_fee"].cumsum().iloc[-1]
        # After fee should be <= before fee
        assert cum_after <= cum_before

    def test_hurdle_reduces_turnover(self):
        ts, prices, signals = self._make_data()
        label_spec = LabelSpec(label_horizon=1, lag=1)
        # Hurdle: only change position if abs difference > 0.3
        def hurdle(new_pos, old_pos):
            return abs(new_pos - old_pos) > 0.3

        exec_no_hurdle = ExecutionSpec(fee_rate=0.0)
        exec_hurdle = ExecutionSpec(fee_rate=0.0, hurdle_enabled=True, hurdle_func=hurdle)
        r1 = execute_timing_backtest(ts, prices, signals, label_spec, exec_no_hurdle)
        r2 = execute_timing_backtest(ts, prices, signals, label_spec, exec_hurdle)
        # Hurdle version should have <= position changes
        changes1 = np.sum(np.abs(np.diff(r1["position"].values)))
        changes2 = np.sum(np.abs(np.diff(r2["position"].values)))
        assert changes2 <= changes1

    def test_long_only_filter(self):
        ts, prices, signals = self._make_data()
        label_spec = LabelSpec(label_horizon=1, lag=1)
        exec_spec = ExecutionSpec(fee_rate=0.0)
        result = execute_timing_backtest(
            timestamps=ts, prices=prices, signals=signals,
            label_spec=label_spec, exec_spec=exec_spec,
            display_mode="long_only",
        )
        # No short positions should exist
        assert (result["position"] >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement executor**

```python
# backtest_api/timing/executor.py
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_api.config import LabelSpec, ExecutionSpec


def execute_timing_backtest(
    timestamps: pd.Series,
    prices: pd.Series,
    signals: pd.Series,
    label_spec: LabelSpec,
    exec_spec: ExecutionSpec,
    display_mode: str = "long_short",
) -> pd.DataFrame:
    """Execute timing backtest with position stacking and fee calculation.

    Returns DataFrame with columns: timestamp, position, pnl_before_fee, pnl_after_fee.
    """
    n = len(prices)
    h = label_spec.label_horizon
    lag = label_spec.lag
    weight = 1.0 / h
    fee_rate = exec_spec.fee_rate

    prices_arr = prices.values.astype(np.float64)
    signals_arr = signals.values.astype(np.float64)

    # Filter signals by display_mode
    if display_mode == "long_only":
        signals_arr = np.where(signals_arr > 0, signals_arr, 0.0)
    elif display_mode == "short_only":
        signals_arr = np.where(signals_arr < 0, signals_arr, 0.0)
    # "long_short" keeps all signals

    # Build position array: each signal at t creates a position of weight*sign
    # from t+lag to t+lag+h-1
    position = np.zeros(n, dtype=np.float64)
    for t in range(n):
        sig = signals_arr[t]
        if np.isnan(sig) or sig == 0.0:
            continue
        start = t + lag
        end = min(t + lag + h, n)
        for k in range(start, end):
            position[k] += weight * sig

    # Apply hurdle filter: only change position if hurdle_func allows it
    # Only active when hurdle_enabled=True AND hurdle_func is provided
    if exec_spec.hurdle_enabled and exec_spec.hurdle_func is not None:
        filtered_pos = np.zeros(n, dtype=np.float64)
        filtered_pos[0] = position[0]
        for i in range(1, n):
            if exec_spec.hurdle_func(position[i], filtered_pos[i - 1]):
                filtered_pos[i] = position[i]
            else:
                filtered_pos[i] = filtered_pos[i - 1]
        position = filtered_pos

    # PnL: per-bar return based on position and price change
    price_ret = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if prices_arr[i - 1] != 0:
            price_ret[i] = (prices_arr[i] - prices_arr[i - 1]) / prices_arr[i - 1]

    pnl_before = position * price_ret

    # Fee: charged on absolute position change
    pos_change = np.zeros(n, dtype=np.float64)
    pos_change[0] = abs(position[0])
    for i in range(1, n):
        pos_change[i] = abs(position[i] - position[i - 1])

    fee_cost = pos_change * fee_rate * 2  # open + close
    pnl_after = pnl_before - fee_cost

    result = pd.DataFrame({
        "timestamp": timestamps.values,
        "position": position,
        "price_return": price_ret,
        "pnl_before_fee": pnl_before,
        "pnl_after_fee": pnl_after,
        "fee_cost": fee_cost,
    })
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_executor.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/timing/executor.py tests/test_executor.py
git commit -m "feat: add executor with position stacking, fee calc, display mode filter"
```

---

### Task 7: Metrics Calculation

**Files:**
- Create: `backtest_api/metrics.py`
- Create: `backtest_api/tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# backtest_api/tests/test_metrics.py
import numpy as np
import pandas as pd
import pytest
from backtest_api.metrics import (
    annualized_return,
    total_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_recovery,
    turnover,
    compute_ic,
    compute_rank_ic,
    information_ratio,
)


class TestReturnMetrics:
    def test_total_return(self):
        pnl = pd.Series([0.01, 0.02, -0.005, 0.015])
        result = total_return(pnl)
        # (1.01)*(1.02)*(0.995)*(1.015) - 1
        expected = (1.01 * 1.02 * 0.995 * 1.015) - 1.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_annualized_return(self):
        pnl = pd.Series([0.001] * 252)  # 252 bars
        result = annualized_return(pnl, bars_per_year=252)
        expected = (1.001 ** 252) - 1.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_volatility(self):
        pnl = pd.Series([0.01, -0.01, 0.01, -0.01])
        result = volatility(pnl, bars_per_year=252)
        assert result > 0

    def test_sharpe_ratio(self):
        pnl = pd.Series([0.01] * 100)
        result = sharpe_ratio(pnl, bars_per_year=252)
        # Constant positive returns → very high Sharpe
        assert result > 10

    def test_sortino_ratio(self):
        pnl = pd.Series([0.01, 0.02, -0.005, 0.015, -0.001])
        result = sortino_ratio(pnl, bars_per_year=252)
        assert result > 0


class TestDrawdown:
    def test_max_drawdown(self):
        pnl = pd.Series([0.1, 0.1, -0.3, -0.1, 0.2])
        result = max_drawdown(pnl)
        assert result > 0
        assert result <= 1.0

    def test_max_drawdown_recovery(self):
        pnl = pd.Series([0.1, 0.1, -0.3, -0.1, 0.05, 0.05, 0.1, 0.1, 0.1])
        bars = max_drawdown_recovery(pnl)
        assert isinstance(bars, (int, float))
        assert bars >= 0


class TestIC:
    def test_perfect_ic(self):
        feature = np.arange(100.0)
        label = feature * 2.0 + 1.0
        result = compute_ic(feature, label)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_perfect_rank_ic(self):
        feature = np.arange(100.0)
        label = feature * 2.0
        result = compute_rank_ic(feature, label)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_information_ratio(self):
        ic_series = pd.Series([0.05, 0.03, 0.07, 0.02, 0.06])
        result = information_ratio(ic_series)
        expected = ic_series.mean() / ic_series.std()
        assert result == pytest.approx(expected, rel=1e-6)


class TestTurnover:
    def test_basic(self):
        positions = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0])
        result = turnover(positions)
        # Changes: 0→1=1, 1→1=0, 1→-1=2, -1→0=1 → mean=4/4=1.0
        expected = np.mean([1.0, 0.0, 2.0, 1.0])
        assert result == pytest.approx(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement metrics**

```python
# backtest_api/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def total_return(pnl_series: pd.Series) -> float:
    """Cumulative compounded return."""
    return float(np.prod(1.0 + pnl_series.values) - 1.0)


def annualized_return(pnl_series: pd.Series, bars_per_year: int = 252) -> float:
    """Annualized compounded return."""
    n = len(pnl_series)
    if n == 0:
        return 0.0
    cum = np.prod(1.0 + pnl_series.values)
    return float(cum ** (bars_per_year / n) - 1.0)


def volatility(pnl_series: pd.Series, bars_per_year: int = 252) -> float:
    """Annualized volatility."""
    return float(np.std(pnl_series.values, ddof=1) * np.sqrt(bars_per_year))


def sharpe_ratio(pnl_series: pd.Series, bars_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (assuming rf=0)."""
    vol = np.std(pnl_series.values, ddof=1)
    if vol == 0:
        return 0.0
    mean_ret = np.mean(pnl_series.values)
    return float(mean_ret / vol * np.sqrt(bars_per_year))


def sortino_ratio(pnl_series: pd.Series, bars_per_year: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    arr = pnl_series.values
    downside = arr[arr < 0]
    if len(downside) == 0:
        return np.inf
    dd = float(np.std(downside, ddof=1))
    if dd == 0:
        return 0.0
    return float(np.mean(arr) / dd * np.sqrt(bars_per_year))


def max_drawdown(pnl_series: pd.Series) -> float:
    """Maximum drawdown from cumulative return curve."""
    cum = np.cumprod(1.0 + pnl_series.values)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def max_drawdown_recovery(pnl_series: pd.Series) -> int:
    """Number of bars to recover from max drawdown."""
    cum = np.cumprod(1.0 + pnl_series.values)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    mdd_idx = int(np.argmax(dd))

    # Find recovery point (where cum >= peak at mdd point)
    peak_val = peak[mdd_idx]
    for i in range(mdd_idx + 1, len(cum)):
        if cum[i] >= peak_val:
            return i - mdd_idx
    # Not recovered
    return len(cum) - mdd_idx


def turnover(positions: pd.Series) -> float:
    """Average absolute position change per bar."""
    pos = positions.values
    changes = np.abs(np.diff(pos))
    return float(np.mean(changes)) if len(changes) > 0 else 0.0


def compute_ic(feature: np.ndarray, label: np.ndarray) -> float:
    """Pearson correlation between feature and label."""
    mask = ~(np.isnan(feature) | np.isnan(label))
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(feature[mask], label[mask])[0, 1])


def compute_rank_ic(feature: np.ndarray, label: np.ndarray) -> float:
    """Spearman rank correlation between feature and label."""
    mask = ~(np.isnan(feature) | np.isnan(label))
    if mask.sum() < 3:
        return np.nan
    corr, _ = stats.spearmanr(feature[mask], label[mask])
    return float(corr)


def information_ratio(ic_series: pd.Series) -> float:
    """IR = mean(IC) / std(IC)."""
    std = ic_series.std()
    if std == 0:
        return 0.0
    return float(ic_series.mean() / std)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_metrics.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/metrics.py tests/test_metrics.py
git commit -m "feat: add performance metrics (Sharpe, Sortino, MDD, IC, turnover, IR)"
```

---

### Task 8: Report — Table & Charts

**Files:**
- Create: `backtest_api/report.py`
- Create: `backtest_api/tests/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
# backtest_api/tests/test_report.py
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from backtest_api.report import build_summary_table, plot_pnl_curve, plot_rolling_ic


class TestBuildSummaryTable:
    def test_output_columns(self):
        pnl_before = pd.Series(np.random.randn(200) * 0.01)
        pnl_after = pd.Series(pnl_before - 0.0001)
        positions = pd.Series(np.random.choice([-1, 0, 1], 200).astype(float))
        feature = np.random.randn(200)
        label = np.random.randn(200)
        table = build_summary_table(
            pnl_before=pnl_before,
            pnl_after=pnl_after,
            positions=positions,
            feature=feature,
            label=label,
            bars_per_year=252,
        )
        expected_cols = [
            "Annualized Return", "Total Return", "Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Turnover",
            "Max Drawdown", "Max Drawdown Recovery Time",
            "Rank IC", "IC", "IR",
        ]
        for col in expected_cols:
            assert col in table.columns, f"Missing column: {col}"

    def test_before_and_after_fee_rows(self):
        pnl_before = pd.Series(np.random.randn(100) * 0.01)
        pnl_after = pd.Series(pnl_before - 0.0001)
        positions = pd.Series(np.random.choice([-1, 0, 1], 100).astype(float))
        feature = np.random.randn(100)
        label = np.random.randn(100)
        table = build_summary_table(
            pnl_before=pnl_before,
            pnl_after=pnl_after,
            positions=positions,
            feature=feature,
            label=label,
        )
        assert len(table) == 2  # before_fee row + after_fee row
        assert table.index.tolist() == ["Before Fee", "After Fee"]


class TestPlotPnlCurve:
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_creates_figure(self, mock_show, mock_save):
        timestamps = pd.date_range("2026-01-01", periods=50, freq="1min")
        pnl_before = pd.Series(np.random.randn(50) * 0.01)
        pnl_after = pd.Series(pnl_before - 0.0001)
        fig = plot_pnl_curve(timestamps, pnl_before, pnl_after, title="Test PnL")
        assert fig is not None


class TestPlotRollingIC:
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_creates_figure(self, mock_show, mock_save):
        timestamps = pd.date_range("2026-01-01", periods=200, freq="1min")
        feature = np.random.randn(200)
        label = np.random.randn(200)
        fig = plot_rolling_ic(
            timestamps, feature, label,
            rolling_window=20, title="Test IC",
        )
        assert fig is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement report module**

```python
# backtest_api/report.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

from backtest_api.metrics import (
    annualized_return,
    total_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_recovery,
    turnover,
    compute_ic,
    compute_rank_ic,
    information_ratio,
)
from backtest_api.numba_utils import rolling_pearson, rolling_spearman


def build_summary_table(
    pnl_before: pd.Series,
    pnl_after: pd.Series,
    positions: pd.Series,
    feature: np.ndarray,
    label: np.ndarray,
    bars_per_year: int = 252,
) -> pd.DataFrame:
    """Build the performance summary table with before/after fee rows."""
    ic_val = compute_ic(feature, label)
    rank_ic_val = compute_rank_ic(feature, label)

    # Compute rolling IC series for IR
    ic_rolling = rolling_pearson(
        feature.astype(np.float64), label.astype(np.float64), window=20
    )
    ic_series = pd.Series(ic_rolling).dropna()
    ir_val = information_ratio(ic_series)

    to = turnover(positions)

    rows = []
    for name, pnl in [("Before Fee", pnl_before), ("After Fee", pnl_after)]:
        rows.append({
            "Annualized Return": annualized_return(pnl, bars_per_year),
            "Total Return": total_return(pnl),
            "Volatility": volatility(pnl, bars_per_year),
            "Sharpe Ratio": sharpe_ratio(pnl, bars_per_year),
            "Sortino Ratio": sortino_ratio(pnl, bars_per_year),
            "Turnover": to,
            "Max Drawdown": max_drawdown(pnl),
            "Max Drawdown Recovery Time": max_drawdown_recovery(pnl),
            "Rank IC": rank_ic_val,
            "IC": ic_val,
            "IR": ir_val,
        })

    return pd.DataFrame(rows, index=["Before Fee", "After Fee"])


def plot_pnl_curve(
    timestamps: pd.Series,
    pnl_before: pd.Series,
    pnl_after: pd.Series,
    title: str = "PnL Curve",
) -> matplotlib.figure.Figure:
    """Plot cumulative PnL curve (before and after fee). All labels in English."""
    cum_before = (1.0 + pnl_before).cumprod()
    cum_after = (1.0 + pnl_after).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps.values, cum_before.values, label="Before Fee", linewidth=1.2)
    ax.plot(timestamps.values, cum_after.values, label="After Fee", linewidth=1.2)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_rolling_ic(
    timestamps: pd.Series,
    feature: np.ndarray,
    label: np.ndarray,
    rolling_window: int = 20,
    title: str = "Rolling IC",
) -> matplotlib.figure.Figure:
    """Plot rolling Pearson IC and Spearman Rank IC. All labels in English.
    Uses numba functions, not pandas rolling.
    """
    feat = feature.astype(np.float64)
    lab = label.astype(np.float64)
    ic_arr = rolling_pearson(feat, lab, window=rolling_window)
    rank_ic_arr = rolling_spearman(feat, lab, window=rolling_window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(timestamps.values, ic_arr, linewidth=0.8, color="steelblue")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("IC (Pearson)")
    ax1.set_title(f"{title} — IC Rolling Window = {rolling_window}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(timestamps.values, rank_ic_arr, linewidth=0.8, color="darkorange")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Rank IC (Spearman)")
    ax2.set_title(f"{title} — Rank IC Rolling Window = {rolling_window}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_report.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/report.py tests/test_report.py
git commit -m "feat: add report module with summary table and PnL/IC charts"
```

---

### Task 9: BaseBacktest ABC

**Files:**
- Create: `backtest_api/base.py`

- [ ] **Step 1: Implement BaseBacktest**

```python
# backtest_api/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BacktestResult:
    """Container for backtest output: tables, figures, raw data."""

    def __init__(self) -> None:
        self.summary_tables: dict[str, pd.DataFrame] = {}
        self.figures: dict[str, Any] = {}
        self.raw_data: dict[str, pd.DataFrame] = {}

    def show(self) -> None:
        """Print summary tables and display figures."""
        for name, table in self.summary_tables.items():
            print(f"\n=== {name} ===")
            print(table.to_string())
        for name, fig in self.figures.items():
            fig.suptitle(name) if not fig._suptitle else None
            fig.show()


class BaseBacktest(ABC):

    @abstractmethod
    def load_data(self) -> None:
        """Load data, validate format, check timestamp alignment."""
        ...

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration."""
        ...

    @abstractmethod
    def run(self) -> BacktestResult:
        """Execute the backtest pipeline, return result."""
        ...

    @abstractmethod
    def evaluate(self) -> pd.DataFrame:
        """Compute performance metrics table."""
        ...

    @abstractmethod
    def report(self) -> None:
        """Generate charts + table output."""
        ...
```

- [ ] **Step 2: Commit**

```bash
git add backtest_api/base.py
git commit -m "feat: add BaseBacktest ABC and BacktestResult container"
```

---

### Task 10: TimingBacktest Main Class — Integration

**Files:**
- Create: `backtest_api/timing/backtest.py`
- Update: `backtest_api/timing/__init__.py`
- Create: `backtest_api/tests/test_timing_backtest.py`

- [ ] **Step 1: Write failing integration tests**

```python
# backtest_api/tests/test_timing_backtest.py
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
        # Should have tables for each feature
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_timing_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement TimingBacktest**

```python
# backtest_api/timing/backtest.py
from __future__ import annotations

from typing import List, Literal, Optional, Callable

import numpy as np
import pandas as pd

from backtest_api.base import BaseBacktest, BacktestResult
from backtest_api.config import (
    TimingBacktestConfig,
    LabelSpec,
    SignalSpec,
    ExecutionSpec,
)
from backtest_api.data_loader import load_file, align_timestamps
from backtest_api.timing.label import compute_labels_from_raw
from backtest_api.timing.signal import generate_signals
from backtest_api.timing.executor import execute_timing_backtest
from backtest_api.report import build_summary_table, plot_pnl_curve, plot_rolling_ic


class TimingBacktest(BaseBacktest):
    """Single-asset timing backtest with signal-based entry/exit."""

    def __init__(
        self,
        feature_path: str,
        time_col: str = "timestamp",
        feature_cols: Optional[List[str]] = None,
        # ---- label mode ----
        label_path: Optional[str] = None,
        label_col: Optional[str] = None,
        # ---- raw data mode ----
        raw_data_path: Optional[str] = None,
        price_col: Optional[str] = None,
        use_raw_data: bool = False,
        exec_price_type: Literal["close", "vwap"] = "close",
        # ---- label spec ----
        display_labels: Optional[List[int]] = None,
        lag: int = 1,
        # ---- signal spec ----
        signal_method: Literal["quantile", "threshold", "custom"] = "quantile",
        upper_quantile: float = 0.8,
        lower_quantile: float = 0.2,
        rolling_window: int = 100,
        upper_threshold: Optional[float] = None,
        lower_threshold: Optional[float] = None,
        signal_direction: Literal["momentum", "mean_reversion"] = "momentum",
        signal_mapper: Optional[Callable] = None,
        display_modes: Optional[List[str]] = None,
        # ---- execution spec ----
        fee_rate: float = 0.00005,
        show_before_fee: bool = True,
        show_after_fee: bool = True,
        hurdle_enabled: bool = False,
        hurdle_func: Optional[Callable] = None,
        # ---- report ----
        ic_rolling_window: int = 20,
        bars_per_year: int = 252,
    ) -> None:
        self.config = TimingBacktestConfig(
            feature_path=feature_path,
            label_path=label_path,
            label_col=label_col,
            raw_data_path=raw_data_path,
            price_col=price_col,
            time_col=time_col,
            feature_cols=feature_cols or [],
            use_raw_data=use_raw_data,
            exec_price_type=exec_price_type,
        )
        self.label_spec = LabelSpec(
            label_horizon=1,  # set per display_label in loop
            lag=lag,
            exec_price_type=exec_price_type,
            display_labels=display_labels or [1],
        )
        self.signal_spec = SignalSpec(
            method=signal_method,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
            rolling_window=rolling_window,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            signal_direction=signal_direction,
            signal_mapper=signal_mapper,
            display_modes=display_modes or ["long_only", "short_only", "long_short"],
        )
        self.exec_spec = ExecutionSpec(
            fee_rate=fee_rate,
            show_before_fee=show_before_fee,
            show_after_fee=show_after_fee,
            hurdle_enabled=hurdle_enabled,
            hurdle_func=hurdle_func,
        )
        self.ic_rolling_window = ic_rolling_window
        self.bars_per_year = bars_per_year

        # Populated by load_data
        self._feature_df: Optional[pd.DataFrame] = None
        self._label_df: Optional[pd.DataFrame] = None
        self._prices: Optional[pd.Series] = None
        self._timestamps: Optional[pd.Series] = None

    def load_data(self) -> bool:
        """Load and align data. Returns False if alignment fails."""
        feat_df = load_file(self.config.feature_path)

        if self.config.use_raw_data:
            raw_df = load_file(self.config.raw_data_path)
            result = align_timestamps(feat_df, raw_df, self.config.time_col)
            if result is None:
                return False
            feat_df, raw_df = result
            self._prices = raw_df[self.config.price_col].reset_index(drop=True)
            self._timestamps = feat_df[self.config.time_col].reset_index(drop=True)
            self._feature_df = feat_df
            self._label_df = None  # computed per horizon
        else:
            label_df = load_file(self.config.label_path)
            result = align_timestamps(feat_df, label_df, self.config.time_col)
            if result is None:
                return False
            feat_df, label_df = result
            self._timestamps = feat_df[self.config.time_col].reset_index(drop=True)
            self._feature_df = feat_df
            self._label_df = label_df
            self._prices = label_df[self.config.label_col].reset_index(drop=True)

        return True

    def validate(self) -> None:
        self.config.validate()
        if self.signal_spec.method != "quantile":
            self.signal_spec.validate()

    def run(self) -> Optional[BacktestResult]:
        """Run the full timing backtest pipeline."""
        self.validate()
        if not self.load_data():
            return None

        result = BacktestResult()

        for feat_col in self.config.feature_cols:
            feature = self._feature_df[feat_col].values.astype(np.float64)
            signal = generate_signals(
                pd.Series(feature), self.signal_spec
            )

            for horizon in self.label_spec.display_labels:
                # Compute label for this horizon
                if self.config.use_raw_data:
                    label_series = compute_labels_from_raw(self._prices, horizon)
                else:
                    label_series = self._label_df[self.config.label_col].reset_index(drop=True)

                label_arr = label_series.values.astype(np.float64)

                # Prices for execution
                if self.config.use_raw_data:
                    exec_prices = self._prices
                else:
                    # For label mode, use cumulative label as proxy price for PnL
                    exec_prices = self._prices

                current_label_spec = LabelSpec(
                    label_horizon=horizon,
                    lag=self.label_spec.lag,
                    exec_price_type=self.label_spec.exec_price_type,
                )

                for mode in self.signal_spec.display_modes:
                    key = f"{feat_col}_label{horizon}_{mode}"

                    exec_result = execute_timing_backtest(
                        timestamps=self._timestamps,
                        prices=exec_prices,
                        signals=signal,
                        label_spec=current_label_spec,
                        exec_spec=self.exec_spec,
                        display_mode=mode,
                    )

                    table = build_summary_table(
                        pnl_before=exec_result["pnl_before_fee"],
                        pnl_after=exec_result["pnl_after_fee"],
                        positions=exec_result["position"],
                        feature=feature,
                        label=label_arr,
                        bars_per_year=self.bars_per_year,
                    )
                    result.summary_tables[key] = table
                    result.raw_data[key] = exec_result

                    pnl_fig = plot_pnl_curve(
                        self._timestamps,
                        exec_result["pnl_before_fee"],
                        exec_result["pnl_after_fee"],
                        title=f"PnL — {feat_col} label{horizon} {mode}",
                    )
                    result.figures[f"pnl_{key}"] = pnl_fig

                # IC chart per feature × horizon (not per mode)
                ic_key = f"ic_{feat_col}_label{horizon}"
                ic_fig = plot_rolling_ic(
                    self._timestamps,
                    feature,
                    label_arr,
                    rolling_window=self.ic_rolling_window,
                    title=f"IC — {feat_col} label{horizon}",
                )
                result.figures[ic_key] = ic_fig

        return result

    def evaluate(self) -> pd.DataFrame:
        """Run and return combined summary table."""
        result = self.run()
        if result is None:
            return pd.DataFrame()
        tables = []
        for key, table in result.summary_tables.items():
            t = table.copy()
            t["config"] = key
            tables.append(t)
        return pd.concat(tables, ignore_index=False) if tables else pd.DataFrame()

    def report(self) -> None:
        """Run and display results."""
        result = self.run()
        if result is not None:
            result.show()
```

- [ ] **Step 4: Update timing __init__.py**

```python
# backtest_api/timing/__init__.py
"""Timing backtest module."""
from backtest_api.timing.backtest import TimingBacktest

__all__ = ["TimingBacktest"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/test_timing_backtest.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backtest_api/base.py backtest_api/timing/backtest.py backtest_api/timing/__init__.py tests/test_timing_backtest.py
git commit -m "feat: add TimingBacktest main class with full pipeline integration"
```

---

### Task 11: Run Full Test Suite & Final Cleanup

**Files:**
- All files created above

- [ ] **Step 1: Run full test suite**

Run: `cd D:/quant/backtest/backtest_api && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (approximately 38 tests)

- [ ] **Step 2: Fix any failures**

If any test fails, read the error, fix the specific issue, and re-run.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup, all tests passing"
```
