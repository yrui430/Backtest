import numpy as np
import pandas as pd
import pytest
from backtest_api.timing.label import compute_labels_from_raw


class TestComputeLabelsFromRaw:
    def test_horizon_1(self):
        prices = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0])
        result = compute_labels_from_raw(prices, label_horizon=1)
        expected = pd.Series([1.0, 2.0, -1.0, 3.0, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_horizon_2(self):
        prices = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0])
        result = compute_labels_from_raw(prices, label_horizon=2)
        expected = pd.Series([3.0, 1.0, 2.0, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_horizon_5(self):
        prices = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0, 110.0])
        result = compute_labels_from_raw(prices, label_horizon=5)
        expected = pd.Series([10.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_returns_same_length(self):
        prices = pd.Series(np.arange(50.0))
        for h in [1, 2, 5]:
            result = compute_labels_from_raw(prices, label_horizon=h)
            assert len(result) == len(prices)
            assert np.isnan(result.iloc[-h:]).all()
