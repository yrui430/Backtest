from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

### 后续优化:增加对不同文件的兼容问题，另外要对大文件进行识别，如果有秒文件需要求用户确认是否加载全部数据，或者增加按时间范围加载的功能。
### 另外可以增加一个功能，自动识别时间戳列和特征列，减少用户输入错误的可能性。
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

    df_f = df_feature[df_feature[time_col].isin(common)].sort_values(time_col).reset_index(drop=True)
    df_l = df_label[df_label[time_col].isin(common)].sort_values(time_col).reset_index(drop=True)
    return df_f, df_l


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
