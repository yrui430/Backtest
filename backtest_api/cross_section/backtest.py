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

            # Compute cross-sectional IC per date
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

            # Prepare returns_df for label mode
            returns_df_for_exec = None
            if not cfg.use_raw_data:
                returns_df_for_exec = label_df[[sc, dc, "label"]].copy()
                returns_df_for_exec = returns_df_for_exec.rename(columns={"label": "_return"})

            for mode in self.exec_spec.display_modes:
                key = f"{feat_col}_{mode}"

                exec_result = execute_cross_section_backtest(
                    signal_df=signal_df,
                    price_df=self._price_df,
                    spec=self.exec_spec,
                    label_spec=self.label_spec,
                    stock_col=sc,
                    date_col=dc,
                    price_col=cfg.exec_price_type if cfg.use_raw_data else cfg.price_col,
                    display_mode=mode,
                    returns_df=returns_df_for_exec,
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

            # IC Cumsum (per feature)
            if len(ic_arr) > 0:
                ic_fig = plot_ic_cumsum(
                    ic_dates,
                    ic_arr,
                    rank_ic_arr,
                    title=f"IC Cumsum — {feat_col}",
                )
                result.figures[f"ic_cumsum_{feat_col}"] = ic_fig

            # IC Decay (per feature, raw mode only)
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
