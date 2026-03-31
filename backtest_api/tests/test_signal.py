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
            return (s > 0).astype(int) * 2 - 1

        feature = pd.Series([1.0, -1.0, 0.5, -0.5])
        spec = SignalSpec(method="custom", signal_mapper=my_mapper)
        signal = generate_signals(feature, spec)
        expected = pd.Series([1, -1, 1, -1])
        pd.testing.assert_series_equal(signal, expected, check_names=False, check_dtype=False)
