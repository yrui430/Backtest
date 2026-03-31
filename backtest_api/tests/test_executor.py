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
        assert cum_after <= cum_before

    def test_hurdle_reduces_turnover(self):
        ts, prices, signals = self._make_data()
        label_spec = LabelSpec(label_horizon=1, lag=1)
        def hurdle(new_pos, old_pos):
            return abs(new_pos - old_pos) > 0.3
        exec_no_hurdle = ExecutionSpec(fee_rate=0.0)
        exec_hurdle = ExecutionSpec(fee_rate=0.0, hurdle_enabled=True, hurdle_func=hurdle)
        r1 = execute_timing_backtest(ts, prices, signals, label_spec, exec_no_hurdle)
        r2 = execute_timing_backtest(ts, prices, signals, label_spec, exec_hurdle)
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
        assert (result["position"] >= 0).all()
