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
