"""FastAPI application for the backtest API."""
from __future__ import annotations

import io
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backtest_api.schemas import (
    TimingBacktestRequest,
    TimingBacktestResponse,
    CrossSectionBacktestRequest,
    CrossSectionBacktestResponse,
)
from backtest_api.timing import TimingBacktest
from backtest_api.cross_section import CrossSectionBacktest

app = FastAPI(
    title="Backtest API",
    description="Quantitative strategy backtesting service — timing, cross-sectional, HF",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/backtest/timing", response_model=TimingBacktestResponse)
def run_timing_backtest(req: TimingBacktestRequest):
    """Run a single-asset timing backtest and return metrics + charts."""
    try:
        bt = TimingBacktest(
            feature_path=req.feature_path,
            time_col=req.time_col,
            feature_cols=req.feature_cols,
            label_path=req.label_path,
            label_col=req.label_col,
            raw_data_path=req.raw_data_path,
            price_col=req.price_col,
            use_raw_data=req.use_raw_data,
            exec_price_type=req.exec_price_type,
            display_labels=req.display_labels,
            lag=req.lag,
            signal_method=req.signal_method,
            upper_quantile=req.upper_quantile,
            lower_quantile=req.lower_quantile,
            rolling_window=req.rolling_window,
            upper_threshold=req.upper_threshold,
            lower_threshold=req.lower_threshold,
            signal_direction=req.signal_direction,
            display_modes=req.display_modes,
            fee_rate=req.fee_rate,
            show_before_fee=req.show_before_fee,
            show_after_fee=req.show_after_fee,
            mode=req.mode,
            ic_rolling_window=req.ic_rolling_window,
            bars_per_year=req.bars_per_year,
        )

        result = bt.run()

        if result is None:
            return TimingBacktestResponse(
                status="error",
                message="Timestamp alignment failed or data loading error. Check file paths and time columns.",
            )

        # Summary tables → JSON
        tables = {}
        for key, df in result.summary_tables.items():
            tables[key] = df.reset_index().rename(columns={"index": "fee_type"}).to_dict(orient="records")

        # Charts → base64 PNG
        charts = {}
        if req.include_charts:
            for key, fig in result.figures.items():
                charts[key] = _fig_to_base64(fig)

        # Raw data → JSON (optional)
        raw = None
        if req.include_raw_data:
            raw = {}
            for key, df in result.raw_data.items():
                raw[key] = df.to_dict(orient="records")

        return TimingBacktestResponse(
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
