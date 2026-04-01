# Cross-Section Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cross-sectional factor backtesting module that evaluates multi-asset strategies by constructing portfolios from cross-sectional signal rankings.

**Architecture:** Mirror the timing module structure. New `cross_section/` package with config, signal, label, executor, report, and backtest modules. Extend `data_loader.py` for multi-format loading. Add API endpoint. Reuse `metrics.py` and `base.py` as-is.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, matplotlib, FastAPI, Pydantic

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `backtest_api/cross_section/__init__.py` | Package init, export `CrossSectionBacktest` |
| Create | `backtest_api/cross_section/config.py` | Config dataclasses |
| Create | `backtest_api/cross_section/signal.py` | winsorize → neutralize → normalize |
| Create | `backtest_api/cross_section/label.py` | forward returns + IC decay |
| Create | `backtest_api/cross_section/executor.py` | weight construction + PnL |
| Create | `backtest_api/cross_section/report.py` | cross-section charts |
| Create | `backtest_api/cross_section/backtest.py` | `CrossSectionBacktest` orchestrator |
| Modify | `backtest_api/data_loader.py` | Add `wide_to_long`, `load_directory`, `load_cross_section_data` |
| Modify | `backtest_api/schemas.py` | Add `CrossSectionBacktestRequest/Response` |
| Modify | `backtest_api/api.py` | Add `POST /backtest/cross-section` |
| Modify | `backtest_api/__init__.py` | Export cross-section classes |
| Create | `tests/__init__.py` | Test package |
| Create | `tests/conftest.py` | Shared fixtures for cross-section test data |
| Create | `tests/test_cs_config.py` | Config validation tests |
| Create | `tests/test_cs_data_loader.py` | Data loader extension tests |
| Create | `tests/test_cs_signal.py` | Signal pipeline tests |
| Create | `tests/test_cs_label.py` | Label + IC decay tests |
| Create | `tests/test_cs_executor.py` | Executor tests |
| Create | `tests/test_cs_report.py` | Report chart tests |
| Create | `tests/test_cs_backtest.py` | Integration tests |
| Delete | `backtest_api/cross-section/` | Remove old hyphenated directory |

---

### Task 1: Directory Setup & Config Dataclasses

**Files:**
- Delete: `backtest_api/cross-section/` (old directory with hyphen)
- Create: `backtest_api/cross_section/__init__.py`
- Create: `backtest_api/cross_section/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_cs_config.py`

- [ ] **Step 1: Remove old directory and create package**

```bash
rm -rf backtest_api/cross-section
mkdir -p backtest_api/cross_section
mkdir -p tests
```

- [ ] **Step 2: Write the failing test for config dataclasses**

File: `tests/test_cs_config.py`

```python
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
        cfg = CrossSectionConfig(feature_path="f.parquet", use_raw_data=True)
        with pytest.raises(ValueError, match="raw_data_path"):
            cfg.validate()

    def test_validate_label_mode_missing_label_path(self):
        cfg = CrossSectionConfig(
            feature_path="f.parquet", use_raw_data=False
        )
        with pytest.raises(ValueError, match="label_path"):
            cfg.validate()

    def test_validate_label_mode_missing_label_col(self):
        cfg = CrossSectionConfig(
            feature_path="f.parquet",
            use_raw_data=False,
            label_path="l.parquet",
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
        cfg.validate()  # should not raise

    def test_validate_label_mode_ok(self):
        cfg = CrossSectionConfig(
            feature_path="f.parquet",
            use_raw_data=False,
            label_path="l.parquet",
            label_col="ret",
            feature_cols=["alpha1"],
        )
        cfg.validate()  # should not raise


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

    def test_validate_custom_group_weight_without_func(self):
        spec = CrossSectionExecutionSpec(group_weight_func=None)
        spec.validate()  # should not raise, group_weight_func is optional (defaults to equal weight)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_api.cross_section'`

- [ ] **Step 4: Create empty __init__ files**

File: `backtest_api/cross_section/__init__.py`

```python
"""Cross-section backtest module."""
```

File: `tests/__init__.py`

```python
```

- [ ] **Step 5: Write config implementation**

File: `backtest_api/cross_section/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional


@dataclass
class CrossSectionConfig:
    # ---- required ----
    feature_path: str
    feature_cols: List[str] = field(default_factory=list)

    # ---- mode A: raw data (default) ----
    raw_data_path: Optional[str] = None
    price_col: str = "close"

    # ---- mode B: pre-computed label ----
    label_path: Optional[str] = None
    label_col: Optional[str] = None

    use_raw_data: bool = True
    exec_price_type: Literal["close", "vwap", "open"] = "close"

    # ---- column names ----
    stock_col: str = "stock_id"
    date_col: str = "date_id"
    time_col: Optional[str] = "time_id"
    industry_col: Optional[str] = None

    # ---- data format ----
    data_format: Literal["long", "wide", "multi_file"] = "long"

    def validate(self) -> None:
        if not self.feature_cols:
            raise ValueError("feature_cols must not be empty.")
        if self.use_raw_data:
            if not self.raw_data_path:
                raise ValueError("raw_data_path is required when use_raw_data=True.")
        else:
            if not self.label_path:
                raise ValueError("label_path is required when use_raw_data=False.")
            if not self.label_col:
                raise ValueError("label_col is required when use_raw_data=False.")


@dataclass
class CrossSectionSignalSpec:
    # ---- winsorize ----
    winsorize_enabled: bool = True
    winsorize_method: Literal["mad", "percentile", "std", "custom"] = "mad"
    winsorize_n_sigma: float = 3.0
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    winsorize_func: Optional[Callable] = None

    # ---- neutralize ----
    neutralize_enabled: bool = False
    neutralize_method: Literal["regression", "demean", "intra_industry", "custom"] = "regression"
    industry_col: Optional[str] = None
    neutralize_func: Optional[Callable] = None

    # ---- normalize ----
    normalize_method: Literal["min_max", "zscore", "rank"] = "min_max"

    def validate(self) -> None:
        if self.winsorize_method == "custom" and self.winsorize_func is None:
            raise ValueError("winsorize_func is required when winsorize_method='custom'.")
        if self.neutralize_enabled:
            if not self.industry_col:
                raise ValueError("industry_col is required when neutralize_enabled=True.")
            if self.neutralize_method == "custom" and self.neutralize_func is None:
                raise ValueError("neutralize_func is required when neutralize_method='custom'.")


@dataclass
class CrossSectionLabelSpec:
    h: int = 1
    lag: int = 1
    decay_lags: List[int] = field(default_factory=lambda: [1, 2, 5])


@dataclass
class CrossSectionExecutionSpec:
    weight_method: Literal["group", "money"] = "group"
    n_groups: int = 5
    group_weight_func: Optional[Callable] = None
    display_modes: List[Literal["long_only", "long_short"]] = field(
        default_factory=lambda: ["long_only", "long_short"]
    )
    signal_direction: Literal["momentum", "mean_reversion"] = "momentum"
    fee_rate: float = 0.00005
    show_before_fee: bool = True
    show_after_fee: bool = True
    initial_capital: float = 1_000_000

    def validate(self) -> None:
        if self.n_groups < 2:
            raise ValueError("n_groups must be >= 2.")
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_config.py -v
```

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add backtest_api/cross_section/__init__.py backtest_api/cross_section/config.py tests/__init__.py tests/test_cs_config.py
git commit -m "feat(cross-section): add config dataclasses with validation"
```

---

### Task 2: Data Loader Extensions

**Files:**
- Modify: `backtest_api/data_loader.py`
- Create: `tests/test_cs_data_loader.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write shared test fixtures**

File: `tests/conftest.py`

```python
"""Shared fixtures for cross-section tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import os
import tempfile


@pytest.fixture
def cs_long_data():
    """Cross-section data in long format: 3 stocks, 10 dates."""
    np.random.seed(42)
    stocks = ["A", "B", "C"]
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for d in dates:
        for s in stocks:
            rows.append({
                "stock_id": s,
                "date_id": d,
                "close": 100 + np.random.randn() * 5,
                "vwap": 100 + np.random.randn() * 5,
                "open": 100 + np.random.randn() * 5,
                "alpha1": np.random.randn(),
                "alpha2": np.random.randn(),
                "industry": "tech" if s == "A" else "fin",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def cs_wide_data():
    """Cross-section data in wide format: columns = stock_ids, rows = dates."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date_id": dates,
        "A": np.random.randn(10),
        "B": np.random.randn(10),
        "C": np.random.randn(10),
    })


@pytest.fixture
def cs_multi_file_dir(cs_long_data, tmp_path):
    """Directory with one parquet file per stock."""
    for stock in ["A", "B", "C"]:
        subset = cs_long_data[cs_long_data["stock_id"] == stock].copy()
        subset.to_parquet(tmp_path / f"{stock}.parquet", index=False)
    return str(tmp_path)


@pytest.fixture
def cs_feature_parquet(cs_long_data, tmp_path):
    """Feature file in long format as parquet."""
    path = tmp_path / "features.parquet"
    cs_long_data[["stock_id", "date_id", "alpha1", "alpha2", "industry"]].to_parquet(
        path, index=False
    )
    return str(path)


@pytest.fixture
def cs_raw_data_parquet(cs_long_data, tmp_path):
    """Raw price data file in long format as parquet."""
    path = tmp_path / "raw_data.parquet"
    cs_long_data[["stock_id", "date_id", "close", "vwap", "open"]].to_parquet(
        path, index=False
    )
    return str(path)


@pytest.fixture
def cs_label_parquet(cs_long_data, tmp_path):
    """Pre-computed label file in long format as parquet."""
    df = cs_long_data[["stock_id", "date_id", "close"]].copy()
    # Simulate forward returns as labels
    df["ret"] = df.groupby("stock_id")["close"].transform(
        lambda x: x.shift(-1) / x - 1
    )
    df = df.drop(columns=["close"])
    path = tmp_path / "labels.parquet"
    df.to_parquet(path, index=False)
    return str(path)
```

- [ ] **Step 2: Write failing tests for data loader extensions**

File: `tests/test_cs_data_loader.py`

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_data_loader.py -v
```

Expected: FAIL — `ImportError: cannot import name 'wide_to_long'`

- [ ] **Step 4: Implement data loader extensions**

Append to `backtest_api/data_loader.py` (after existing code):

```python
def wide_to_long(
    df: pd.DataFrame,
    time_col: str,
    stock_col: str = "stock_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """Convert wide-format DataFrame (columns=stocks, rows=time) to long format."""
    id_cols = [c for c in [time_col] if c in df.columns]
    value_cols = [c for c in df.columns if c not in id_cols]
    result = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name=stock_col,
        value_name=value_col,
    )
    return result.sort_values([time_col, stock_col]).reset_index(drop=True)


def load_directory(dir_path: str, stock_col: str = "stock_id") -> pd.DataFrame:
    """Load all parquet/csv/h5 files from a directory, one file per stock.

    The stock identifier is derived from the filename (without extension).
    """
    import os
    dir_p = Path(dir_path)
    frames = []
    for f in sorted(dir_p.iterdir()):
        if f.suffix.lower() in (".parquet", ".csv", ".h5", ".hdf5"):
            df = load_file(str(f))
            stock_name = f.stem
            if stock_col not in df.columns:
                df[stock_col] = stock_name
            frames.append(df)
    if not frames:
        raise ValueError(f"No data files found in {dir_path}")
    return pd.concat(frames, ignore_index=True)


def load_cross_section_data(
    path: str,
    data_format: str = "long",
    stock_col: str = "stock_id",
    time_col: str = "date_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """Unified entry point for cross-section data loading.

    Supports 'long' (default), 'wide', and 'multi_file' formats.
    All formats are converted to long format internally.
    """
    if data_format == "long":
        return load_file(path)
    elif data_format == "wide":
        df = load_file(path)
        return wide_to_long(df, time_col=time_col, stock_col=stock_col, value_col=value_col)
    elif data_format == "multi_file":
        return load_directory(path, stock_col=stock_col)
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_data_loader.py -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add backtest_api/data_loader.py tests/conftest.py tests/test_cs_data_loader.py
git commit -m "feat(cross-section): extend data_loader with wide_to_long, load_directory, load_cross_section_data"
```

---

### Task 3: Signal Pipeline (winsorize → neutralize → normalize)

**Files:**
- Create: `backtest_api/cross_section/signal.py`
- Create: `tests/test_cs_signal.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_cs_signal.py`

```python
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
        assert result[3] < 100.0  # outlier should be clipped
        np.testing.assert_almost_equal(result[0], 1.0)  # non-outlier unchanged

    def test_percentile_clips(self):
        np.random.seed(42)
        values = np.random.randn(100)
        values[0] = 50.0  # extreme outlier
        result = winsorize(values, method="percentile", lower=0.01, upper=0.99)
        assert result[0] < 50.0

    def test_std_clips(self):
        values = np.array([1.0, 2.0, 3.0, 100.0, 2.5])
        result = winsorize(values, method="std", n_sigma=3.0)
        assert result[3] < 100.0

    def test_custom_func(self):
        values = np.array([1.0, 2.0, 3.0])
        custom = lambda v: np.clip(v, 0, 2)
        result = winsorize(values, method="custom", custom_func=custom)
        np.testing.assert_array_equal(result, [1.0, 2.0, 2.0])


class TestNeutralize:
    def test_regression_removes_industry_effect(self):
        np.random.seed(42)
        # industry A has higher mean
        values = np.array([10.0, 11.0, 12.0, 1.0, 2.0, 3.0])
        industry = np.array(["A", "A", "A", "B", "B", "B"])
        result = neutralize(values, industry, method="regression")
        # After regression, means should be closer to zero
        mean_a = np.mean(result[industry == "A"])
        mean_b = np.mean(result[industry == "B"])
        assert abs(mean_a) < 1.0
        assert abs(mean_b) < 1.0

    def test_demean_subtracts_industry_mean(self):
        values = np.array([10.0, 12.0, 1.0, 3.0])
        industry = np.array(["A", "A", "B", "B"])
        result = neutralize(values, industry, method="demean")
        np.testing.assert_almost_equal(result[0], -1.0)  # 10 - 11 = -1
        np.testing.assert_almost_equal(result[1], 1.0)   # 12 - 11 = 1
        np.testing.assert_almost_equal(result[2], -1.0)  # 1 - 2 = -1
        np.testing.assert_almost_equal(result[3], 1.0)   # 3 - 2 = 1

    def test_intra_industry_standardizes(self):
        values = np.array([10.0, 12.0, 1.0, 3.0])
        industry = np.array(["A", "A", "B", "B"])
        result = neutralize(values, industry, method="intra_industry")
        # Within each industry, mean should be ~0
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
        values = np.random.randn(100) * 10  # wide range
        result = normalize(values, method="zscore")
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_rank_range(self):
        values = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        result = normalize(values, method="rank")
        assert result.min() >= -1.0
        assert result.max() <= 1.0
        # Highest value should map to max
        assert result[0] == pytest.approx(1.0)  # value 5.0 is highest
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_signal.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement signal pipeline**

File: `backtest_api/cross_section/signal.py`

```python
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest_api.cross_section.config import CrossSectionSignalSpec


def winsorize(
    values: np.ndarray,
    method: str = "mad",
    n_sigma: float = 3.0,
    lower: float = 0.01,
    upper: float = 0.99,
    custom_func: Optional[Callable] = None,
) -> np.ndarray:
    """Clip outliers in a single cross-section snapshot."""
    if method == "mad":
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        if mad == 0:
            return values.copy()
        lower_bound = median - n_sigma * 1.4826 * mad
        upper_bound = median + n_sigma * 1.4826 * mad
        return np.clip(values, lower_bound, upper_bound)
    elif method == "percentile":
        lo = np.nanpercentile(values, lower * 100)
        hi = np.nanpercentile(values, upper * 100)
        return np.clip(values, lo, hi)
    elif method == "std":
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1)
        if std == 0:
            return values.copy()
        lower_bound = mean - n_sigma * std
        upper_bound = mean + n_sigma * std
        return np.clip(values, lower_bound, upper_bound)
    elif method == "custom":
        return custom_func(values)
    else:
        raise ValueError(f"Unknown winsorize method: {method}")


def neutralize(
    values: np.ndarray,
    industry: np.ndarray,
    method: str = "regression",
    custom_func: Optional[Callable] = None,
) -> np.ndarray:
    """Remove industry effects from a single cross-section snapshot."""
    if method == "regression":
        unique_ind = np.unique(industry)
        # Build dummy matrix and regress
        dummies = np.zeros((len(values), len(unique_ind)), dtype=np.float64)
        for i, ind in enumerate(unique_ind):
            dummies[industry == ind, i] = 1.0
        # OLS: residual = values - dummies @ (dummies^T dummies)^-1 dummies^T values
        beta, _, _, _ = np.linalg.lstsq(dummies, values, rcond=None)
        fitted = dummies @ beta
        return values - fitted
    elif method == "demean":
        result = values.copy()
        for ind in np.unique(industry):
            mask = industry == ind
            result[mask] = values[mask] - np.nanmean(values[mask])
        return result
    elif method == "intra_industry":
        result = values.copy()
        for ind in np.unique(industry):
            mask = industry == ind
            subset = values[mask]
            std = np.nanstd(subset, ddof=1)
            if std == 0:
                result[mask] = 0.0
            else:
                result[mask] = (subset - np.nanmean(subset)) / std
        return result
    elif method == "custom":
        return custom_func(values, industry)
    else:
        raise ValueError(f"Unknown neutralize method: {method}")


def normalize(values: np.ndarray, method: str = "min_max") -> np.ndarray:
    """Normalize values to [-1, 1] range."""
    if method == "min_max":
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if vmax == vmin:
            return np.zeros_like(values)
        return 2.0 * (values - vmin) / (vmax - vmin) - 1.0
    elif method == "zscore":
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1)
        if std == 0:
            return np.zeros_like(values)
        z = (values - mean) / std
        return np.clip(z / 3.0, -1.0, 1.0)  # 3-sigma → [-1, 1]
    elif method == "rank":
        n = len(values)
        if n == 0:
            return values.copy()
        # rank from 0 to n-1, then map to [-1, 1]
        order = np.argsort(np.argsort(values))  # 0-based rank
        return 2.0 * order / (n - 1) - 1.0 if n > 1 else np.zeros_like(values)
    else:
        raise ValueError(f"Unknown normalize method: {method}")


def generate_cross_section_signals(
    feature_df: pd.DataFrame,
    spec: CrossSectionSignalSpec,
    stock_col: str,
    date_col: str,
    feature_col: str,
    industry_col: Optional[str] = None,
) -> pd.DataFrame:
    """Apply winsorize → neutralize → normalize per cross-section (grouped by date)."""
    result = feature_df.copy()
    signals = np.full(len(result), np.nan)

    for date_val, group in result.groupby(date_col):
        idx = group.index
        values = group[feature_col].values.astype(np.float64)

        # Skip if all NaN
        if np.all(np.isnan(values)):
            continue

        # Step 1: Winsorize
        if spec.winsorize_enabled:
            values = winsorize(
                values,
                method=spec.winsorize_method,
                n_sigma=spec.winsorize_n_sigma,
                lower=spec.winsorize_lower,
                upper=spec.winsorize_upper,
                custom_func=spec.winsorize_func,
            )

        # Step 2: Neutralize
        if spec.neutralize_enabled and industry_col is not None:
            ind_values = group[industry_col].values
            values = neutralize(
                values,
                ind_values,
                method=spec.neutralize_method,
                custom_func=spec.neutralize_func,
            )

        # Step 3: Normalize
        values = normalize(values, method=spec.normalize_method)

        signals[idx] = values

    result["signal"] = signals
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_signal.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/cross_section/signal.py tests/test_cs_signal.py
git commit -m "feat(cross-section): add signal pipeline with winsorize, neutralize, normalize"
```

---

### Task 4: Label Computation & IC Decay

**Files:**
- Create: `backtest_api/cross_section/label.py`
- Create: `tests/test_cs_label.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_cs_label.py`

```python
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
        # Last rows per stock should have NaN (no future data)
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
        # More NaNs at the end with h=5
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
        # For t=0: label = price[0+1+2] / price[0+1] - 1 = price[3]/price[1] - 1
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_label.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement label computation**

File: `backtest_api/cross_section/label.py`

```python
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats


def compute_forward_returns(
    df: pd.DataFrame,
    stock_col: str,
    date_col: str,
    price_col: str,
    h: int,
    lag: int,
) -> pd.DataFrame:
    """Compute forward returns per stock: label(t) = price(t+lag+h) / price(t+lag) - 1.

    Groups by stock, shifts prices forward by (lag+h) and (lag).
    """
    result = df.copy()
    labels = []

    for stock, group in result.groupby(stock_col):
        g = group.sort_values(date_col).copy()
        prices = g[price_col].values.astype(np.float64)
        n = len(prices)
        label = np.full(n, np.nan)
        for t in range(n):
            entry = t + lag
            exit_ = t + lag + h
            if exit_ < n and entry < n:
                label[t] = prices[exit_] / prices[entry] - 1.0
        labels.append(pd.Series(label, index=g.index))

    result["label"] = pd.concat(labels)
    return result


def compute_ic_decay(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    stock_col: str,
    date_col: str,
    feature_col: str,
    price_col: str,
    h: int,
    decay_lags: List[int],
) -> pd.DataFrame:
    """Compute IC decay: for each lag, compute mean cross-sectional IC.

    IC_decay(lag) = mean_over_t[ corr_cs( feature(t), ret(t+lag : t+lag+h) ) ]
    """
    # Merge feature and price data
    merged = feature_df[[stock_col, date_col, feature_col]].merge(
        price_df[[stock_col, date_col, price_col]],
        on=[stock_col, date_col],
        how="inner",
    )

    results = []
    for lag_val in decay_lags:
        # Compute forward returns for this lag
        ret_df = compute_forward_returns(
            merged[[stock_col, date_col, price_col]],
            stock_col=stock_col,
            date_col=date_col,
            price_col=price_col,
            h=h,
            lag=lag_val,
        )
        merged_with_ret = merged.copy()
        merged_with_ret["_ret"] = ret_df["label"]

        # Compute cross-sectional IC per date
        ic_list = []
        rank_ic_list = []
        for date_val, group in merged_with_ret.groupby(date_col):
            feat = group[feature_col].values
            ret = group["_ret"].values
            mask = ~(np.isnan(feat) | np.isnan(ret))
            if mask.sum() < 3:
                continue
            ic = float(np.corrcoef(feat[mask], ret[mask])[0, 1])
            rank_ic, _ = stats.spearmanr(feat[mask], ret[mask])
            ic_list.append(ic)
            rank_ic_list.append(float(rank_ic))

        ic_mean = np.nanmean(ic_list) if ic_list else np.nan
        rank_ic_mean = np.nanmean(rank_ic_list) if rank_ic_list else np.nan

        results.append({
            "lag": lag_val,
            "ic_mean": ic_mean,
            "rank_ic_mean": rank_ic_mean,
        })

    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_label.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/cross_section/label.py tests/test_cs_label.py
git commit -m "feat(cross-section): add label computation and IC decay"
```

---

### Task 5: Executor (Weight Construction + PnL)

**Files:**
- Create: `backtest_api/cross_section/executor.py`
- Create: `tests/test_cs_executor.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_cs_executor.py`

```python
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
        # Lowest values should be group 1, highest group 5
        assert groups.iloc[0] == 1  # value 1.0
        assert groups.iloc[-1] == 5  # value 10.0

    def test_3_groups(self):
        signals = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        groups = assign_quantile_groups(signals, n_groups=3)
        assert set(groups.unique()).issubset({1, 2, 3})


class TestComputeWeightsGroup:
    def test_long_short_momentum(self):
        signals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        groups = pd.Series([1, 1, 3, 5, 5])  # group 1 = low, group 5 = high
        weights = compute_weights_group(
            signals, groups, n_groups=5, direction="momentum", mode="long_short"
        )
        # Long group 5, short group 1
        long_sum = weights[groups == 5].sum()
        short_sum = weights[groups == 1].sum()
        assert long_sum == pytest.approx(1.0)
        assert short_sum == pytest.approx(-1.0)
        # Middle groups should be 0
        assert weights[groups == 3].sum() == pytest.approx(0.0)

    def test_long_only_momentum(self):
        signals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        groups = pd.Series([1, 2, 3, 4, 5])
        weights = compute_weights_group(
            signals, groups, n_groups=5, direction="momentum", mode="long_only"
        )
        # Only top group has weight
        assert weights[groups == 5].sum() == pytest.approx(1.0)
        assert weights[groups != 5].sum() == pytest.approx(0.0)

    def test_long_short_mean_reversion(self):
        signals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        groups = pd.Series([1, 2, 3, 4, 5])
        weights = compute_weights_group(
            signals, groups, n_groups=5, direction="mean_reversion", mode="long_short"
        )
        # Long group 1 (lowest), short group 5 (highest)
        assert weights[groups == 1].sum() == pytest.approx(1.0)
        assert weights[groups == 5].sum() == pytest.approx(-1.0)

    def test_custom_weight_func(self):
        signals = pd.Series([1.0, 2.0, 3.0])
        groups = pd.Series([1, 1, 1])
        # Custom: weight proportional to signal
        def w_func(sigs):
            return sigs / sigs.sum()
        weights = compute_weights_group(
            signals, groups, n_groups=1, direction="momentum", mode="long_only",
            weight_func=w_func,
        )
        assert weights.sum() == pytest.approx(1.0)
        assert weights.iloc[2] > weights.iloc[0]  # higher signal → higher weight


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
        """3 stocks, 10 dates, with signals."""
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
        """3 stocks, 10 dates, with prices."""
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
        # Check net exposure is ~0 at each date
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_executor.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement executor**

File: `backtest_api/cross_section/executor.py`

```python
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest_api.cross_section.config import (
    CrossSectionExecutionSpec,
    CrossSectionLabelSpec,
)


def assign_quantile_groups(signals: pd.Series, n_groups: int) -> pd.Series:
    """Assign stocks to quantile groups based on signal ranking.

    Group 1 = lowest signals, group n = highest signals.
    """
    ranks = signals.rank(method="first")
    n = len(signals)
    # Map ranks [1, n] to groups [1, n_groups]
    groups = pd.Series(
        np.ceil(ranks / n * n_groups).astype(int).clip(1, n_groups),
        index=signals.index,
    )
    return groups


def compute_weights_group(
    signals: pd.Series,
    groups: pd.Series,
    n_groups: int,
    direction: str,
    mode: str,
    weight_func: Optional[Callable] = None,
) -> pd.Series:
    """Compute portfolio weights using group-based method.

    Args:
        signals: Signal values for each stock in the cross-section.
        groups: Group assignments (1=lowest, n_groups=highest).
        n_groups: Total number of groups.
        direction: 'momentum' or 'mean_reversion'.
        mode: 'long_short' or 'long_only'.
        weight_func: Optional custom function for intra-group weights.
            Takes a Series of signals, returns a Series of weights (summing to 1).
    """
    weights = pd.Series(0.0, index=signals.index)

    if direction == "momentum":
        long_group = n_groups
        short_group = 1
    else:  # mean_reversion
        long_group = 1
        short_group = n_groups

    # Long leg
    long_mask = groups == long_group
    if long_mask.any():
        if weight_func is not None:
            long_w = weight_func(signals[long_mask])
        else:
            long_w = pd.Series(1.0 / long_mask.sum(), index=signals[long_mask].index)
        weights[long_mask] = long_w

    # Short leg (only for long_short)
    if mode == "long_short":
        short_mask = groups == short_group
        if short_mask.any():
            if weight_func is not None:
                short_w = -weight_func(signals[short_mask])
            else:
                short_w = pd.Series(
                    -1.0 / short_mask.sum(), index=signals[short_mask].index
                )
            weights[short_mask] = short_w

    return weights


def compute_weights_money(
    signals: pd.Series,
    direction: str,
    mode: str,
) -> pd.Series:
    """Compute portfolio weights proportional to signal magnitude.

    Args:
        signals: Signal values for each stock.
        direction: 'momentum' or 'mean_reversion'.
        mode: 'long_short' or 'long_only'.
    """
    weights = pd.Series(0.0, index=signals.index)

    if direction == "mean_reversion":
        signals = -signals

    pos_mask = signals > 0
    neg_mask = signals < 0

    if mode == "long_short":
        # Long: positive signals, normalized to sum=1
        if pos_mask.any():
            pos_sum = signals[pos_mask].sum()
            if pos_sum > 0:
                weights[pos_mask] = signals[pos_mask] / pos_sum
        # Short: negative signals, normalized to sum=-1
        if neg_mask.any():
            neg_sum = signals[neg_mask].abs().sum()
            if neg_sum > 0:
                weights[neg_mask] = signals[neg_mask] / neg_sum  # already negative
    elif mode == "long_only":
        if pos_mask.any():
            pos_sum = signals[pos_mask].sum()
            if pos_sum > 0:
                weights[pos_mask] = signals[pos_mask] / pos_sum

    return weights


def execute_cross_section_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    spec: CrossSectionExecutionSpec,
    label_spec: CrossSectionLabelSpec,
    stock_col: str,
    date_col: str,
    price_col: str,
    display_mode: str = "long_short",
) -> dict:
    """Execute the cross-section backtest: signal → weight → portfolio PnL.

    Returns dict with:
      - portfolio_gross_pnl: pd.Series (per bar)
      - portfolio_net_pnl: pd.Series (per bar)
      - weights: pd.DataFrame (stock_col, date_col, weight)
      - group_returns: pd.DataFrame (per bar, per group cumulative return)
      - timestamps: sorted unique dates
    """
    lag = label_spec.lag

    # Merge signal and price
    merged = signal_df[[stock_col, date_col, "signal"]].merge(
        price_df[[stock_col, date_col, price_col]],
        on=[stock_col, date_col],
        how="inner",
    )

    dates = sorted(merged[date_col].unique())
    n_dates = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Compute per-stock bar returns
    merged = merged.sort_values([stock_col, date_col])
    merged["_return"] = merged.groupby(stock_col)[price_col].pct_change()

    # Compute weights per cross-section
    all_weights = []
    for date_val, group in merged.groupby(date_col):
        signals = group.set_index(stock_col)["signal"]

        if spec.weight_method == "group":
            groups = assign_quantile_groups(signals, spec.n_groups)
            w = compute_weights_group(
                signals, groups, spec.n_groups,
                direction=spec.signal_direction,
                mode=display_mode,
                weight_func=spec.group_weight_func,
            )
            for s in signals.index:
                all_weights.append({
                    stock_col: s,
                    date_col: date_val,
                    "weight": w[s],
                    "group": groups[s],
                })
        else:  # money
            w = compute_weights_money(signals, spec.signal_direction, display_mode)
            groups = assign_quantile_groups(signals, spec.n_groups)
            for s in signals.index:
                all_weights.append({
                    stock_col: s,
                    date_col: date_val,
                    "weight": w[s],
                    "group": groups[s],
                })

    weights_df = pd.DataFrame(all_weights)

    # Compute portfolio PnL (with lag)
    portfolio_gross = np.zeros(n_dates)
    portfolio_fees = np.zeros(n_dates)

    # Group returns for quantile return chart
    group_return_data = {g: np.zeros(n_dates) for g in range(1, spec.n_groups + 1)}

    prev_weights = {}  # stock → weight from previous bar

    for i, date_val in enumerate(dates):
        date_weights = weights_df[weights_df[date_col] == date_val]
        date_merged = merged[merged[date_col] == date_val]

        # Portfolio return: sum(weight_i * return_i(t+lag))
        # We apply weights from (lag) bars ago
        if i >= lag:
            weight_date = dates[i - lag]
            wt = weights_df[weights_df[date_col] == weight_date]
            wt_dict = dict(zip(wt[stock_col], wt["weight"]))
            wt_group = dict(zip(wt[stock_col], wt["group"]))

            for _, row in date_merged.iterrows():
                s = row[stock_col]
                ret = row["_return"]
                if np.isnan(ret):
                    continue
                w = wt_dict.get(s, 0.0)
                portfolio_gross[i] += w * ret

                # Group returns
                g = wt_group.get(s, 0)
                if 1 <= g <= spec.n_groups:
                    # Equal-weight return for each group
                    group_stocks = [k for k, v in wt_group.items() if v == g]
                    n_in_group = len(group_stocks)
                    if n_in_group > 0:
                        group_return_data[g][i] += ret / n_in_group

        # Fee: based on weight changes
        curr_weights = dict(zip(date_weights[stock_col], date_weights["weight"]))
        fee = 0.0
        all_stocks = set(curr_weights.keys()) | set(prev_weights.keys())
        for s in all_stocks:
            delta = abs(curr_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
            fee += spec.fee_rate * delta
        portfolio_fees[i] = fee
        prev_weights = curr_weights

    portfolio_net = portfolio_gross - portfolio_fees

    # Build group_returns DataFrame
    group_returns = pd.DataFrame({"date": dates})
    for g in range(1, spec.n_groups + 1):
        group_returns[f"group_{g}"] = group_return_data[g]

    return {
        "portfolio_gross_pnl": pd.Series(portfolio_gross, index=range(n_dates)),
        "portfolio_net_pnl": pd.Series(portfolio_net, index=range(n_dates)),
        "weights": weights_df,
        "group_returns": group_returns,
        "timestamps": pd.Series(dates),
        "fees": pd.Series(portfolio_fees, index=range(n_dates)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_executor.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/cross_section/executor.py tests/test_cs_executor.py
git commit -m "feat(cross-section): add executor with group-based and money-based weight methods"
```

---

### Task 6: Report — Cross-Section Charts

**Files:**
- Create: `backtest_api/cross_section/report.py`
- Create: `tests/test_cs_report.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_cs_report.py`

```python
"""Tests for cross-section report charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

from backtest_api.cross_section.report import (
    plot_quantile_returns,
    plot_group_ic,
    plot_ic_cumsum,
    plot_ic_decay,
    build_cs_summary_table,
)


@pytest.fixture
def group_returns():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(42)
    df = pd.DataFrame({"date": dates})
    for g in range(1, 6):
        df[f"group_{g}"] = np.random.randn(50) * 0.01 + g * 0.001
    return df


@pytest.fixture
def group_ic():
    return pd.DataFrame({
        "group": [1, 2, 3, 4, 5],
        "ic_mean": [0.01, 0.02, 0.03, 0.04, 0.05],
    })


class TestPlotQuantileReturns:
    def test_returns_figure(self, group_returns):
        fig = plot_quantile_returns(group_returns, n_groups=5, title="Test")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_correct_lines(self, group_returns):
        fig = plot_quantile_returns(group_returns, n_groups=5, title="Test")
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 5


class TestPlotGroupIc:
    def test_returns_figure(self, group_ic):
        fig = plot_group_ic(group_ic, title="Test")
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotIcCumsum:
    def test_returns_figure(self):
        np.random.seed(42)
        timestamps = pd.date_range("2024-01-01", periods=50, freq="D")
        ic = np.random.randn(50) * 0.05
        rank_ic = np.random.randn(50) * 0.05
        fig = plot_ic_cumsum(timestamps, ic, rank_ic, title="Test")
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotIcDecay:
    def test_returns_figure(self):
        fig = plot_ic_decay(
            decay_lags=[1, 2, 5],
            ic_means=[0.05, 0.03, 0.01],
            rank_ic_means=[0.04, 0.02, 0.008],
            title="Test",
        )
        assert isinstance(fig, matplotlib.figure.Figure)


class TestBuildCsSummaryTable:
    def test_basic_output(self):
        np.random.seed(42)
        n = 100
        pnl_before = pd.Series(np.random.randn(n) * 0.01)
        pnl_after = pd.Series(np.random.randn(n) * 0.01)
        positions = pd.Series(np.random.randn(n) * 0.5)
        ic_series = np.random.randn(n) * 0.05
        rank_ic_series = np.random.randn(n) * 0.05
        table = build_cs_summary_table(
            pnl_before, pnl_after, positions, ic_series, rank_ic_series,
        )
        assert "Before Fee" in table.index
        assert "After Fee" in table.index
        assert "Sharpe Ratio" in table.columns
        assert "IC" in table.columns
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_report.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement report module**

File: `backtest_api/cross_section/report.py`

```python
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

plt.rcParams["figure.max_open_warning"] = 0

from backtest_api.metrics import (
    annualized_return,
    total_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_recovery,
    turnover,
    information_ratio,
)


def build_cs_summary_table(
    pnl_before: pd.Series,
    pnl_after: pd.Series,
    positions: pd.Series,
    ic_series: np.ndarray,
    rank_ic_series: np.ndarray,
    bars_per_year: int = 252,
) -> pd.DataFrame:
    """Build performance summary table for cross-section backtest."""
    ic_mean = float(np.nanmean(ic_series))
    rank_ic_mean = float(np.nanmean(rank_ic_series))
    ir_val = information_ratio(pd.Series(ic_series).dropna())
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
            "IC": ic_mean,
            "Rank IC": rank_ic_mean,
            "IR": ir_val,
        })

    return pd.DataFrame(rows, index=["Before Fee", "After Fee"])


def plot_quantile_returns(
    group_returns: pd.DataFrame,
    n_groups: int,
    title: str = "Quantile Returns",
) -> matplotlib.figure.Figure:
    """Plot cumulative returns for each quantile group."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for g in range(1, n_groups + 1):
        col = f"group_{g}"
        if col in group_returns.columns:
            cum = (1.0 + group_returns[col]).cumprod()
            ax.plot(group_returns["date"].values, cum.values, label=f"Group {g}", linewidth=1.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_group_ic(
    group_ic: pd.DataFrame,
    title: str = "Group IC",
) -> matplotlib.figure.Figure:
    """Plot bar chart of IC mean per quantile group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(group_ic["group"].astype(str), group_ic["ic_mean"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Group")
    ax.set_ylabel("IC Mean")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_ic_cumsum(
    timestamps: pd.Series,
    ic_series: np.ndarray,
    rank_ic_series: np.ndarray,
    title: str = "IC Cumsum",
) -> matplotlib.figure.Figure:
    """Plot IC and Rank IC cumulative sum with mean annotation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ic_cumsum = np.nancumsum(ic_series)
    rank_ic_cumsum = np.nancumsum(rank_ic_series)
    ic_mean = np.nanmean(ic_series)
    rank_ic_mean = np.nanmean(rank_ic_series)

    ts = timestamps.values if hasattr(timestamps, 'values') else np.array(timestamps)

    ax1.plot(ts, ic_cumsum, linewidth=1.0, color="steelblue")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("IC Cumsum")
    ax1.set_title(f"{title} — IC (mean={ic_mean:.4f})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, rank_ic_cumsum, linewidth=1.0, color="darkorange")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rank IC Cumsum")
    ax2.set_title(f"{title} — Rank IC (mean={rank_ic_mean:.4f})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ic_decay(
    decay_lags: List[int],
    ic_means: List[float],
    rank_ic_means: List[float],
    title: str = "IC Decay",
) -> matplotlib.figure.Figure:
    """Plot IC Decay bar chart: x=lag, y=IC mean."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(decay_lags))
    width = 0.35

    ax.bar(x - width / 2, ic_means, width, label="IC", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, rank_ic_means, width, label="Rank IC", color="darkorange", alpha=0.8)

    ax.set_xlabel("Lag")
    ax.set_ylabel("IC Mean")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in decay_lags])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_report.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_api/cross_section/report.py tests/test_cs_report.py
git commit -m "feat(cross-section): add report module with quantile return, IC cumsum, IC decay charts"
```

---

### Task 7: CrossSectionBacktest Main Class

**Files:**
- Create: `backtest_api/cross_section/backtest.py`
- Modify: `backtest_api/cross_section/__init__.py`
- Create: `tests/test_cs_backtest.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_cs_backtest.py`

```python
"""Integration tests for CrossSectionBacktest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
        # Should have results for both features
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_backtest.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement CrossSectionBacktest**

File: `backtest_api/cross_section/backtest.py`

```python
from __future__ import annotations

from typing import Callable, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from backtest_api.base import BaseBacktest, BacktestResult
from backtest_api.data_loader import load_cross_section_data
from backtest_api.cross_section.config import (
    CrossSectionConfig,
    CrossSectionSignalSpec,
    CrossSectionLabelSpec,
    CrossSectionExecutionSpec,
)
from backtest_api.cross_section.signal import generate_cross_section_signals
from backtest_api.cross_section.label import compute_forward_returns, compute_ic_decay
from backtest_api.cross_section.executor import execute_cross_section_backtest
from backtest_api.cross_section.report import (
    build_cs_summary_table,
    plot_quantile_returns,
    plot_group_ic,
    plot_ic_cumsum,
    plot_ic_decay,
)
from backtest_api.report import plot_pnl_curve


class CrossSectionBacktest(BaseBacktest):
    """Multi-asset cross-sectional factor backtest."""

    def __init__(
        self,
        # ---- data paths ----
        feature_path: str,
        raw_data_path: Optional[str] = None,
        label_path: Optional[str] = None,
        use_raw_data: bool = True,
        # ---- column names ----
        stock_col: str = "stock_id",
        date_col: str = "date_id",
        time_col: Optional[str] = "time_id",
        price_col: str = "close",
        exec_price_type: Literal["close", "vwap", "open"] = "close",
        feature_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        industry_col: Optional[str] = None,
        # ---- data format ----
        data_format: Literal["long", "wide", "multi_file"] = "long",
        # ---- winsorize ----
        winsorize_enabled: bool = True,
        winsorize_method: Literal["mad", "percentile", "std", "custom"] = "mad",
        winsorize_n_sigma: float = 3.0,
        winsorize_lower: float = 0.01,
        winsorize_upper: float = 0.99,
        winsorize_func: Optional[Callable] = None,
        # ---- neutralize ----
        neutralize_enabled: bool = False,
        neutralize_method: Literal["regression", "demean", "intra_industry", "custom"] = "regression",
        neutralize_func: Optional[Callable] = None,
        # ---- normalize ----
        normalize_method: Literal["min_max", "zscore", "rank"] = "min_max",
        # ---- label / IC Decay ----
        h: int = 1,
        lag: int = 1,
        decay_lags: Optional[List[int]] = None,
        # ---- execution ----
        weight_method: Literal["group", "money"] = "group",
        n_groups: int = 5,
        group_weight_func: Optional[Callable] = None,
        display_modes: Optional[List[str]] = None,
        signal_direction: Literal["momentum", "mean_reversion"] = "momentum",
        fee_rate: float = 0.00005,
        show_before_fee: bool = True,
        show_after_fee: bool = True,
        initial_capital: float = 1_000_000,
        # ---- report ----
        ic_rolling_window: int = 20,
        bars_per_year: int = 252,
    ) -> None:
        self.config = CrossSectionConfig(
            feature_path=feature_path,
            raw_data_path=raw_data_path,
            label_path=label_path,
            use_raw_data=use_raw_data,
            stock_col=stock_col,
            date_col=date_col,
            time_col=time_col,
            price_col=price_col,
            exec_price_type=exec_price_type,
            feature_cols=feature_cols or [],
            label_col=label_col,
            industry_col=industry_col,
            data_format=data_format,
        )
        self.signal_spec = CrossSectionSignalSpec(
            winsorize_enabled=winsorize_enabled,
            winsorize_method=winsorize_method,
            winsorize_n_sigma=winsorize_n_sigma,
            winsorize_lower=winsorize_lower,
            winsorize_upper=winsorize_upper,
            winsorize_func=winsorize_func,
            neutralize_enabled=neutralize_enabled,
            neutralize_method=neutralize_method,
            industry_col=industry_col,
            neutralize_func=neutralize_func,
            normalize_method=normalize_method,
        )
        self.label_spec = CrossSectionLabelSpec(
            h=h,
            lag=lag,
            decay_lags=decay_lags or [1, 2, 5],
        )
        self.exec_spec = CrossSectionExecutionSpec(
            weight_method=weight_method,
            n_groups=n_groups,
            group_weight_func=group_weight_func,
            display_modes=display_modes or ["long_only", "long_short"],
            signal_direction=signal_direction,
            fee_rate=fee_rate,
            show_before_fee=show_before_fee,
            show_after_fee=show_after_fee,
            initial_capital=initial_capital,
        )
        self.ic_rolling_window = ic_rolling_window
        self.bars_per_year = bars_per_year

        self._feature_df: Optional[pd.DataFrame] = None
        self._price_df: Optional[pd.DataFrame] = None
        self._label_df: Optional[pd.DataFrame] = None

    def load_data(self) -> bool:
        """Load and prepare data. Returns False on failure."""
        cfg = self.config
        sc = cfg.stock_col
        dc = cfg.date_col

        self._feature_df = load_cross_section_data(
            cfg.feature_path,
            data_format=cfg.data_format,
            stock_col=sc,
            time_col=dc,
        )

        if cfg.use_raw_data:
            raw_df = load_cross_section_data(
                cfg.raw_data_path,
                data_format=cfg.data_format,
                stock_col=sc,
                time_col=dc,
            )
            self._price_df = raw_df
            self._label_df = None
        else:
            label_df = load_cross_section_data(
                cfg.label_path,
                data_format=cfg.data_format,
                stock_col=sc,
                time_col=dc,
            )
            self._label_df = label_df
            self._price_df = label_df

        return True

    def validate(self) -> None:
        self.config.validate()
        if self.signal_spec.neutralize_enabled:
            self.signal_spec.validate()
        self.exec_spec.validate()

    def run(self) -> Optional[BacktestResult]:
        """Run the full cross-section backtest pipeline."""
        self.validate()
        if not self.load_data():
            return None

        result = BacktestResult()
        cfg = self.config
        sc = cfg.stock_col
        dc = cfg.date_col

        for feat_col in cfg.feature_cols:
            # Generate signals
            signal_df = generate_cross_section_signals(
                feature_df=self._feature_df,
                spec=self.signal_spec,
                stock_col=sc,
                date_col=dc,
                feature_col=feat_col,
                industry_col=cfg.industry_col,
            )

            # Compute labels
            if cfg.use_raw_data:
                label_df = compute_forward_returns(
                    self._price_df,
                    stock_col=sc,
                    date_col=dc,
                    price_col=cfg.price_col,
                    h=self.label_spec.h,
                    lag=self.label_spec.lag,
                )
            else:
                label_df = self._label_df.copy()
                label_df = label_df.rename(columns={cfg.label_col: "label"})

            # Compute cross-sectional IC per date for this feature
            ic_per_date = []
            rank_ic_per_date = []
            dates_for_ic = []
            merged_ic = signal_df[[sc, dc, "signal"]].merge(
                label_df[[sc, dc, "label"]], on=[sc, dc], how="inner"
            )
            for date_val, group in merged_ic.groupby(dc):
                feat = group["signal"].values
                lab = group["label"].values
                mask = ~(np.isnan(feat) | np.isnan(lab))
                if mask.sum() < 3:
                    continue
                ic = float(np.corrcoef(feat[mask], lab[mask])[0, 1])
                rank_ic, _ = stats.spearmanr(feat[mask], lab[mask])
                ic_per_date.append(ic)
                rank_ic_per_date.append(float(rank_ic))
                dates_for_ic.append(date_val)

            ic_arr = np.array(ic_per_date)
            rank_ic_arr = np.array(rank_ic_per_date)
            ic_dates = pd.Series(dates_for_ic)

            for mode in self.exec_spec.display_modes:
                key = f"{feat_col}_{mode}"

                exec_result = execute_cross_section_backtest(
                    signal_df=signal_df,
                    price_df=self._price_df,
                    spec=self.exec_spec,
                    label_spec=self.label_spec,
                    stock_col=sc,
                    date_col=dc,
                    price_col=cfg.exec_price_type,
                    display_mode=mode,
                )

                # Summary table
                positions = exec_result["weights"].groupby(dc)["weight"].apply(
                    lambda w: w.abs().sum()
                )
                table = build_cs_summary_table(
                    pnl_before=exec_result["portfolio_gross_pnl"],
                    pnl_after=exec_result["portfolio_net_pnl"],
                    positions=pd.Series(positions.values),
                    ic_series=ic_arr,
                    rank_ic_series=rank_ic_arr,
                    bars_per_year=self.bars_per_year,
                )
                result.summary_tables[key] = table
                result.raw_data[key] = exec_result

                # PnL curve
                pnl_fig = plot_pnl_curve(
                    exec_result["timestamps"],
                    exec_result["portfolio_gross_pnl"],
                    exec_result["portfolio_net_pnl"],
                    title=f"PnL — {feat_col} {mode}",
                )
                result.figures[f"pnl_{key}"] = pnl_fig

                # Quantile returns
                qr_fig = plot_quantile_returns(
                    exec_result["group_returns"],
                    n_groups=self.exec_spec.n_groups,
                    title=f"Quantile Returns — {feat_col} {mode}",
                )
                result.figures[f"quantile_{key}"] = qr_fig

                # Group IC
                group_ic_data = self._compute_group_ic(
                    signal_df, label_df, exec_result["weights"],
                    sc, dc,
                )
                gic_fig = plot_group_ic(
                    group_ic_data,
                    title=f"Group IC — {feat_col} {mode}",
                )
                result.figures[f"group_ic_{key}"] = gic_fig

            # IC Cumsum (per feature, across all modes)
            if len(ic_arr) > 0:
                ic_fig = plot_ic_cumsum(
                    ic_dates,
                    ic_arr,
                    rank_ic_arr,
                    title=f"IC Cumsum — {feat_col}",
                )
                result.figures[f"ic_cumsum_{feat_col}"] = ic_fig

            # IC Decay (per feature)
            if cfg.use_raw_data:
                decay_df = compute_ic_decay(
                    feature_df=signal_df[[sc, dc, "signal"]].rename(
                        columns={"signal": feat_col}
                    ),
                    price_df=self._price_df,
                    stock_col=sc,
                    date_col=dc,
                    feature_col=feat_col,
                    price_col=cfg.price_col,
                    h=self.label_spec.h,
                    decay_lags=self.label_spec.decay_lags,
                )
                decay_fig = plot_ic_decay(
                    decay_lags=self.label_spec.decay_lags,
                    ic_means=decay_df["ic_mean"].tolist(),
                    rank_ic_means=decay_df["rank_ic_mean"].tolist(),
                    title=f"IC Decay — {feat_col}",
                )
                result.figures[f"ic_decay_{feat_col}"] = decay_fig

        return result

    def _compute_group_ic(
        self,
        signal_df: pd.DataFrame,
        label_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        stock_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """Compute IC mean per quantile group."""
        # Merge signals with labels and group assignments
        merged = signal_df[[stock_col, date_col, "signal"]].merge(
            label_df[[stock_col, date_col, "label"]], on=[stock_col, date_col], how="inner"
        )
        merged = merged.merge(
            weights_df[[stock_col, date_col, "group"]].drop_duplicates(),
            on=[stock_col, date_col],
            how="inner",
        )

        group_ics = []
        for g in sorted(merged["group"].unique()):
            g_data = merged[merged["group"] == g]
            ic_list = []
            for _, group_data in g_data.groupby(date_col):
                feat = group_data["signal"].values
                lab = group_data["label"].values
                mask = ~(np.isnan(feat) | np.isnan(lab))
                if mask.sum() < 3:
                    continue
                ic = float(np.corrcoef(feat[mask], lab[mask])[0, 1])
                if not np.isnan(ic):
                    ic_list.append(ic)
            group_ics.append({
                "group": g,
                "ic_mean": np.nanmean(ic_list) if ic_list else np.nan,
            })

        return pd.DataFrame(group_ics)

    def evaluate(self) -> pd.DataFrame:
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
        result = self.run()
        if result is not None:
            result.show()
```

- [ ] **Step 4: Update __init__.py**

File: `backtest_api/cross_section/__init__.py`

```python
"""Cross-section backtest module."""
from backtest_api.cross_section.backtest import CrossSectionBacktest

__all__ = ["CrossSectionBacktest"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/test_cs_backtest.py -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add backtest_api/cross_section/backtest.py backtest_api/cross_section/__init__.py tests/test_cs_backtest.py
git commit -m "feat(cross-section): add CrossSectionBacktest integration class"
```

---

### Task 8: API Endpoint & Schemas

**Files:**
- Modify: `backtest_api/schemas.py`
- Modify: `backtest_api/api.py`

- [ ] **Step 1: Add Pydantic schemas**

Append to `backtest_api/schemas.py`:

```python
class CrossSectionBacktestRequest(BaseModel):
    """Request body for POST /backtest/cross-section."""

    # ---- data ----
    feature_path: str = Field(..., description="Path to feature file")
    raw_data_path: Optional[str] = Field(None, description="Path to raw price file")
    label_path: Optional[str] = Field(None, description="Path to label file")
    use_raw_data: bool = Field(True, description="Use raw data mode (default)")

    # ---- columns ----
    stock_col: str = Field("stock_id")
    date_col: str = Field("date_id")
    time_col: Optional[str] = Field("time_id")
    price_col: str = Field("close")
    exec_price_type: Literal["close", "vwap", "open"] = Field("close")
    feature_cols: List[str] = Field(..., description="Feature columns to backtest")
    label_col: Optional[str] = None
    industry_col: Optional[str] = None

    # ---- data format ----
    data_format: Literal["long", "wide", "multi_file"] = Field("long")

    # ---- winsorize ----
    winsorize_enabled: bool = Field(True)
    winsorize_method: Literal["mad", "percentile", "std"] = Field("mad")
    winsorize_n_sigma: float = Field(3.0)
    winsorize_lower: float = Field(0.01)
    winsorize_upper: float = Field(0.99)

    # ---- neutralize ----
    neutralize_enabled: bool = Field(False)
    neutralize_method: Literal["regression", "demean", "intra_industry"] = Field("regression")

    # ---- normalize ----
    normalize_method: Literal["min_max", "zscore", "rank"] = Field("min_max")

    # ---- label / IC Decay ----
    h: int = Field(1, description="Holding horizon")
    lag: int = Field(1, description="Execution delay")
    decay_lags: List[int] = Field([1, 2, 5])

    # ---- execution ----
    weight_method: Literal["group", "money"] = Field("group")
    n_groups: int = Field(5)
    display_modes: List[Literal["long_only", "long_short"]] = Field(["long_only", "long_short"])
    signal_direction: Literal["momentum", "mean_reversion"] = Field("momentum")
    fee_rate: float = Field(0.00005)
    show_before_fee: bool = True
    show_after_fee: bool = True
    initial_capital: float = Field(1_000_000)

    # ---- report ----
    ic_rolling_window: int = Field(20)
    bars_per_year: int = Field(252)

    # ---- response options ----
    include_raw_data: bool = Field(False)
    include_charts: bool = Field(True)


class CrossSectionBacktestResponse(BaseModel):
    """Response body for POST /backtest/cross-section."""

    status: str = Field(..., description="'success' or 'error'")
    message: Optional[str] = None
    summary_tables: Dict[str, Any] = Field(default_factory=dict)
    charts: Dict[str, str] = Field(default_factory=dict, description="base64 PNG images")
    raw_data: Optional[Dict[str, Any]] = None
```

- [ ] **Step 2: Add API endpoint**

Append to `backtest_api/api.py` (after the timing endpoint), also update the import at the top:

Add to imports:

```python
from backtest_api.schemas import (
    TimingBacktestRequest,
    TimingBacktestResponse,
    CrossSectionBacktestRequest,
    CrossSectionBacktestResponse,
)
from backtest_api.cross_section import CrossSectionBacktest
```

Add endpoint:

```python
@app.post("/backtest/cross-section", response_model=CrossSectionBacktestResponse)
def run_cross_section_backtest(req: CrossSectionBacktestRequest):
    """Run a cross-sectional factor backtest and return metrics + charts."""
    try:
        bt = CrossSectionBacktest(
            feature_path=req.feature_path,
            raw_data_path=req.raw_data_path,
            label_path=req.label_path,
            use_raw_data=req.use_raw_data,
            stock_col=req.stock_col,
            date_col=req.date_col,
            time_col=req.time_col,
            price_col=req.price_col,
            exec_price_type=req.exec_price_type,
            feature_cols=req.feature_cols,
            label_col=req.label_col,
            industry_col=req.industry_col,
            data_format=req.data_format,
            winsorize_enabled=req.winsorize_enabled,
            winsorize_method=req.winsorize_method,
            winsorize_n_sigma=req.winsorize_n_sigma,
            winsorize_lower=req.winsorize_lower,
            winsorize_upper=req.winsorize_upper,
            neutralize_enabled=req.neutralize_enabled,
            neutralize_method=req.neutralize_method,
            normalize_method=req.normalize_method,
            h=req.h,
            lag=req.lag,
            decay_lags=req.decay_lags,
            weight_method=req.weight_method,
            n_groups=req.n_groups,
            display_modes=req.display_modes,
            signal_direction=req.signal_direction,
            fee_rate=req.fee_rate,
            show_before_fee=req.show_before_fee,
            show_after_fee=req.show_after_fee,
            initial_capital=req.initial_capital,
            ic_rolling_window=req.ic_rolling_window,
            bars_per_year=req.bars_per_year,
        )

        result = bt.run()

        if result is None:
            return CrossSectionBacktestResponse(
                status="error",
                message="Data loading or alignment failed.",
            )

        tables = {}
        for key, df in result.summary_tables.items():
            tables[key] = df.reset_index().rename(
                columns={"index": "fee_type"}
            ).to_dict(orient="records")

        charts = {}
        if req.include_charts:
            for key, fig in result.figures.items():
                charts[key] = _fig_to_base64(fig)

        raw = None
        if req.include_raw_data:
            raw = {}
            for key, data in result.raw_data.items():
                if isinstance(data, dict):
                    serialized = {}
                    for k, v in data.items():
                        if isinstance(v, (pd.DataFrame, pd.Series)):
                            serialized[k] = v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v.tolist()
                        else:
                            serialized[k] = v
                    raw[key] = serialized
                elif isinstance(data, pd.DataFrame):
                    raw[key] = data.to_dict(orient="records")

        return CrossSectionBacktestResponse(
            status="success",
            summary_tables=tables,
            charts=charts,
            raw_data=raw,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: Run tests to verify nothing is broken**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add backtest_api/schemas.py backtest_api/api.py
git commit -m "feat(cross-section): add API endpoint POST /backtest/cross-section"
```

---

### Task 9: Public API Exports & Cleanup

**Files:**
- Modify: `backtest_api/__init__.py`

- [ ] **Step 1: Update public exports**

Add to `backtest_api/__init__.py`:

Add imports:

```python
from backtest_api.cross_section import CrossSectionBacktest
from backtest_api.cross_section.config import (
    CrossSectionConfig,
    CrossSectionSignalSpec,
    CrossSectionLabelSpec,
    CrossSectionExecutionSpec,
)
```

Add to `__all__`:

```python
    "CrossSectionBacktest",
    "CrossSectionConfig",
    "CrossSectionSignalSpec",
    "CrossSectionLabelSpec",
    "CrossSectionExecutionSpec",
```

- [ ] **Step 2: Run full test suite**

```bash
cd D:/quant/standard/backtest/backtest_api && python -m pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 3: Verify import works**

```bash
cd D:/quant/standard/backtest/backtest_api && python -c "from backtest_api import CrossSectionBacktest; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backtest_api/__init__.py
git commit -m "feat(cross-section): export CrossSectionBacktest and config classes from public API"
```
