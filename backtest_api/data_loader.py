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

    df_f = df_feature[df_feature[time_col].isin(common)].sort_values(time_col).reset_index(drop=True)
    df_l = df_label[df_label[time_col].isin(common)].sort_values(time_col).reset_index(drop=True)
    return df_f, df_l
