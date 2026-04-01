"""Shared fixtures for cross-section tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
    df["ret"] = df.groupby("stock_id")["close"].transform(
        lambda x: x.shift(-1) / x - 1
    )
    df = df.drop(columns=["close"])
    path = tmp_path / "labels.parquet"
    df.to_parquet(path, index=False)
    return str(path)
