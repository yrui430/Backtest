from __future__ import annotations

import numpy as np
import pandas as pd


def compute_labels_from_raw(prices: pd.Series, label_horizon: int) -> pd.Series:
    """Compute label as price[t+h] - price[t] (first difference with horizon h)."""
    shifted = prices.shift(-label_horizon)
    label = shifted - prices
    return label.reset_index(drop=True)
