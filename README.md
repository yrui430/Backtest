# Backtest API

A quantitative strategy backtesting framework with both a **Python library** and a **REST API** (FastAPI).
Supports **timing backtest** (single-asset, signal-based) and **cross-section backtest** (multi-asset, factor-ranking-based). HF module is planned.

---

## Features

- **Timing Backtest**: single-asset signal-driven entry/exit with two execution modes
- **Cross-Section Backtest**: multi-asset factor evaluation with portfolio construction
- **Signal pipeline**: rolling quantile, threshold, custom mapper (timing); winsorize → neutralize → normalize (cross-section)
- **Weight methods**: group-based (quantile ranking) or money-based (signal-proportional)
- **Metrics**: annualized return, Sharpe, Sortino, max drawdown, IC, Rank IC, IR, turnover
- **Charts**: PnL curve, rolling IC, quantile return, group IC, IC cumsum, IC decay
- **REST API**: run backtests via HTTP, get JSON metrics + base64 chart images
- **File formats**: Parquet, CSV, HDF5
- **Data formats**: long table (default), wide table, multi-file directory

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Timing Backtest — Quick Start

```python
from backtest_api import TimingBacktest

bt = TimingBacktest(
    feature_path="data/features.parquet",
    label_path="data/labels.parquet",
    feature_cols=["alpha1", "alpha2"],
    label_col="ret_1",
    display_labels=[1, 5],
    display_modes=["long_short"],
    fee_rate=0.0001,
)
result = bt.run()
result.show()  # prints tables + displays charts
```

### Raw data mode (auto-compute labels from prices)

```python
bt = TimingBacktest(
    feature_path="data/features.parquet",
    raw_data_path="data/prices.parquet",
    feature_cols=["alpha1"],
    price_col="close",
    use_raw_data=True,
    display_labels=[1, 3, 5],
)
result = bt.run()
```

---

## Cross-Section Backtest — Quick Start

```python
from backtest_api import CrossSectionBacktest

bt = CrossSectionBacktest(
    feature_path="data/features.parquet",
    raw_data_path="data/prices.parquet",
    use_raw_data=True,
    feature_cols=["alpha1"],
    stock_col="stock_id",
    date_col="date_id",
    price_col="close",
    n_groups=5,
    display_modes=["long_short"],
    fee_rate=0.0001,
)
result = bt.run()
result.show()
```

### Label mode (pre-computed labels)

```python
bt = CrossSectionBacktest(
    feature_path="data/features.parquet",
    use_raw_data=False,
    label_path="data/labels.parquet",
    label_col="ret_1",
    feature_cols=["alpha1"],
    stock_col="stock_id",
    date_col="date_id",
    n_groups=5,
)
result = bt.run()
```

### With industry neutralization

```python
bt = CrossSectionBacktest(
    feature_path="data/features.parquet",
    raw_data_path="data/prices.parquet",
    use_raw_data=True,
    feature_cols=["alpha1"],
    stock_col="stock_id",
    date_col="date_id",
    price_col="close",
    industry_col="industry",
    neutralize_enabled=True,
    neutralize_method="regression",   # "regression" | "demean" | "intra_industry" | "custom"
    normalize_method="rank",
)
result = bt.run()
```

---

## Quick Start — REST API

### Start the server

```bash
python main.py
# Server runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Run a timing backtest

```bash
curl -X POST http://localhost:8000/backtest/timing \
  -H "Content-Type: application/json" \
  -d '{
    "feature_path": "data/features.parquet",
    "label_path": "data/labels.parquet",
    "feature_cols": ["alpha1"],
    "label_col": "ret_1",
    "display_labels": [1],
    "display_modes": ["long_short"],
    "fee_rate": 0.0001,
    "include_charts": true
  }'
```

### Run a cross-section backtest

```bash
curl -X POST http://localhost:8000/backtest/cross-section \
  -H "Content-Type: application/json" \
  -d '{
    "feature_path": "data/features.parquet",
    "raw_data_path": "data/prices.parquet",
    "use_raw_data": true,
    "feature_cols": ["alpha1"],
    "stock_col": "stock_id",
    "date_col": "date_id",
    "price_col": "close",
    "n_groups": 5,
    "display_modes": ["long_short"],
    "fee_rate": 0.0001,
    "include_charts": true
  }'
```

### Response structure

```json
{
  "status": "success",
  "summary_tables": {
    "alpha1_long_short": [
      {
        "fee_type": "Before Fee",
        "Annualized Return": 0.12,
        "Sharpe Ratio": 1.85,
        "Max Drawdown": 0.05,
        "IC": 0.03,
        "Rank IC": 0.04
      }
    ]
  },
  "charts": {
    "pnl_alpha1_long_short": "<base64 PNG>",
    "quantile_alpha1_long_short": "<base64 PNG>",
    "ic_cumsum_alpha1": "<base64 PNG>",
    "ic_decay_alpha1": "<base64 PNG>"
  }
}
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/backtest/timing` | Run timing backtest |
| `POST` | `/backtest/cross-section` | Run cross-section backtest |
| `GET` | `/docs` | Swagger UI (auto-generated) |

---

## Full Parameter Example — Timing Backtest (Python)

```python
from backtest_api import TimingBacktest

bt = TimingBacktest(
    # ========== Data (required) ==========
    feature_path="data/features.parquet",   # Feature file (.parquet/.csv/.h5)
    time_col="timestamp",                   # Timestamp column, default "timestamp"
    feature_cols=["alpha1", "alpha2"],       # Feature columns to backtest (required)

    # ========== Mode A: Label mode (pick A or B) ==========
    label_path="data/labels.parquet",       # Label file path
    label_col="ret_1",                      # Label column name

    # ========== Mode B: Raw data mode (pick A or B) ==========
    # raw_data_path="data/prices.parquet",  # Raw price file path
    # price_col="close",                    # Price column name
    # use_raw_data=True,                    # Enable raw data mode (auto-computes labels)

    exec_price_type="close",                # "close" | "vwap", default "close"

    # ========== Label / holding period ==========
    display_labels=[1, 3, 5],               # Holding horizons to evaluate, default [1]
    lag=1,                                  # Signal-to-trade delay in bars, default 1

    # ========== Signal generation ==========
    signal_method="quantile",               # "quantile" | "threshold" | "custom", default "quantile"
    upper_quantile=0.8,                     # Upper quantile (quantile method), default 0.8
    lower_quantile=0.2,                     # Lower quantile (quantile method), default 0.2
    rolling_window=100,                     # Rolling window for quantile, default 100
    signal_direction="momentum",            # "momentum" | "mean_reversion", default "momentum"
    display_modes=["long_only", "short_only", "long_short"],  # Trading modes, default all three

    # ========== Execution / fees ==========
    fee_rate=0.00005,                       # One-sided fee rate, default 0.5 bps
    show_before_fee=True,                   # Show before-fee results, default True
    show_after_fee=True,                    # Show after-fee results, default True
    mode="diff",                            # "diff" (signal-driven) | "label" (fixed holding), default "diff"

    # ========== Report / charts ==========
    ic_rolling_window=20,                   # IC rolling window, default 20
    bars_per_year=252,                      # Bars per year for annualization, default 252
)

result = bt.run()       # Run backtest -> BacktestResult
result.show()           # Print tables + display charts
```

---

## Full Parameter Example — Cross-Section Backtest (Python)

```python
from backtest_api import CrossSectionBacktest

bt = CrossSectionBacktest(
    # ========== Data (required) ==========
    feature_path="data/features.parquet",   # Feature file (.parquet/.csv/.h5)
    feature_cols=["alpha1", "alpha2"],       # Feature columns to backtest (required)

    # ========== Mode A: Raw data mode (default, pick A or B) ==========
    raw_data_path="data/prices.parquet",    # Raw price file path
    use_raw_data=True,                      # Default True

    # ========== Mode B: Label mode (pick A or B) ==========
    # label_path="data/labels.parquet",     # Pre-computed label file
    # label_col="ret_1",                    # Label column name
    # use_raw_data=False,                   # Switch to label mode

    # ========== Column names ==========
    stock_col="stock_id",                   # Stock identifier column, default "stock_id"
    date_col="date_id",                     # Date column, default "date_id"
    time_col="time_id",                     # Optional time column, default "time_id"
    price_col="close",                      # Price column for label computation, default "close"
    exec_price_type="close",                # Price column for PnL, default "close"
    industry_col="industry",                # Industry column (optional, for neutralization)

    # ========== Data format ==========
    data_format="long",                     # "long" | "wide" | "multi_file", default "long"

    # ========== Winsorize (outlier removal) ==========
    winsorize_enabled=True,                 # Enable winsorization, default True
    winsorize_method="mad",                 # "mad" | "percentile" | "std" | "custom", default "mad"
    winsorize_n_sigma=3.0,                  # MAD/std multiplier, default 3.0
    # winsorize_lower=0.01,                 # Percentile lower bound (percentile method)
    # winsorize_upper=0.99,                 # Percentile upper bound (percentile method)
    # winsorize_func=my_func,              # Custom function (custom method, Python only)

    # ========== Industry neutralization ==========
    neutralize_enabled=False,               # Enable neutralization, default False
    neutralize_method="regression",         # "regression" | "demean" | "intra_industry" | "custom"
    # neutralize_func=my_func,             # Custom function (custom method, Python only)

    # ========== Normalization ==========
    normalize_method="min_max",             # "min_max" | "zscore" | "rank", default "min_max"

    # ========== Label / IC Decay ==========
    h=1,                                    # Holding horizon for forward returns, default 1
    lag=1,                                  # Execution delay (bars), default 1
    decay_lags=[1, 2, 5],                   # IC Decay lag sequence, default [1, 2, 5]

    # ========== Execution / portfolio ==========
    weight_method="group",                  # "group" (quantile ranking) | "money" (signal-proportional)
    n_groups=5,                             # Number of quantile groups, default 5
    # group_weight_func=my_func,           # Custom intra-group weight function (Python only)
    display_modes=["long_only", "long_short"],  # Trading modes, default both
    signal_direction="momentum",            # "momentum" | "mean_reversion", default "momentum"
    fee_rate=0.00005,                       # One-sided fee rate, default 0.5 bps
    show_before_fee=True,
    show_after_fee=True,
    initial_capital=1_000_000,              # Initial capital, default 1,000,000

    # ========== Report / charts ==========
    ic_rolling_window=20,
    bars_per_year=252,
)

result = bt.run()       # Run backtest -> BacktestResult
result.show()           # Print tables + display charts
```

---

## Cross-Section Signal Pipeline

Each cross-section (per time point) is processed independently:

```
feature[t] -> Winsorize -> Neutralize -> Normalize -> signal[t] in [-1, 1]
```

### Step 1: Winsorize (optional, enabled by default)

| Method | Description | Key Parameter |
|--------|-------------|---------------|
| `"mad"` (default) | Median Absolute Deviation clipping | `winsorize_n_sigma=3.0` |
| `"percentile"` | Percentile truncation | `winsorize_lower=0.01, winsorize_upper=0.99` |
| `"std"` | Standard deviation clipping | `winsorize_n_sigma=3.0` |
| `"custom"` | User-provided function | `winsorize_func` (Python only) |

### Step 2: Industry Neutralization (optional, disabled by default)

| Method | Description |
|--------|-------------|
| `"regression"` (default) | Industry dummy regression, take residuals |
| `"demean"` | Subtract industry mean |
| `"intra_industry"` | Standardize within each industry |
| `"custom"` | User-provided function |

### Step 3: Normalize

| Method | Output Range |
|--------|-------------|
| `"min_max"` (default) | [-1, 1] |
| `"zscore"` | Truncated to [-1, 1] |
| `"rank"` | rank/N mapped to [-1, 1] |

---

## Cross-Section Weight Methods

### Group-Based (default)

Rank stocks by signal into `n_groups` quantile groups, then:

| Mode | Momentum | Mean Reversion |
|------|----------|----------------|
| `long_short` | Long top group (sum=1), short bottom group (sum=-1) | Opposite |
| `long_only` | Long top group (sum=1) | Long bottom group (sum=1) |

Group-internal: equal weight by default, or custom via `group_weight_func`.

### Money-Based

Allocate weights directly proportional to signal magnitude:

| Mode | Description |
|------|-------------|
| `long_short` | Positive signals -> long (sum=1), negative -> short (sum=-1) |
| `long_only` | Only positive signals, proportional weights (sum=1) |

---

## Cross-Section Charts

| Chart | Description |
|-------|-------------|
| **PnL Curve** | Before/after fee cumulative return |
| **Quantile Return** | One line per group, good factor = monotonic |
| **Group IC Bar** | IC mean per quantile group |
| **IC / Rank IC Cumsum** | Cumulative IC curves with time-series mean annotation |
| **IC Decay Bar** | x=lag, y=IC mean. Fixed `h`, varying `lag`, shows factor timeliness |

---

## Cross-Section Symbols

| Symbol | Meaning | Default |
|--------|---------|---------|
| `t` | Factor generation time | - |
| `h` | Holding horizon (forward return window) | 1 |
| `lag` | Execution delay (bars from signal to trade) | 1 |
| `decay_lags` | Lag sequence for IC Decay | [1, 2, 5] |
| `n_groups` | Number of quantile groups | 5 |

**IC Decay formula**: `IC_d(lag) = mean_over_t[ corr_cs( feature(t), ret(t+lag : t+lag+h) ) ]`

---

## Metrics Explained

| Metric | Description |
|--------|-------------|
| Annualized Return | Compound annual growth rate |
| Total Return | Cumulative return over the period |
| Volatility | Annualized standard deviation of returns |
| Sharpe Ratio | Risk-adjusted return (mean / vol) |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Max DD Recovery | Bars to recover from max drawdown |
| Turnover | Average position change per bar |
| IC | Pearson correlation (feature vs label) |
| Rank IC | Spearman rank correlation |
| IR | Information ratio (mean IC / std IC) |

---

## Timing Backtest Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **`feature_path`** | `str` | *required* | Path to feature data file |
| **`feature_cols`** | `List[str]` | *required* | Feature column names to backtest |
| `time_col` | `str` | `"timestamp"` | Timestamp column name |
| `label_path` | `str` | `None` | Path to label file (label mode) |
| `label_col` | `str` | `None` | Label column name |
| `raw_data_path` | `str` | `None` | Path to raw price file (raw mode) |
| `price_col` | `str` | `None` | Price column (close/vwap) |
| `use_raw_data` | `bool` | `False` | Enable raw data mode |
| `exec_price_type` | `str` | `"close"` | `"close"` or `"vwap"` |
| `display_labels` | `List[int]` | `[1]` | Label horizons to evaluate |
| `lag` | `int` | `1` | Signal-to-trade delay (bars) |
| `signal_method` | `str` | `"quantile"` | `"quantile"`, `"threshold"`, or `"custom"` |
| `upper_quantile` | `float` | `0.8` | Upper quantile (quantile method) |
| `lower_quantile` | `float` | `0.2` | Lower quantile (quantile method) |
| `rolling_window` | `int` | `100` | Rolling window for quantile |
| `signal_direction` | `str` | `"momentum"` | `"momentum"` or `"mean_reversion"` |
| `display_modes` | `List[str]` | `["long_only","short_only","long_short"]` | Trading modes |
| `fee_rate` | `float` | `0.00005` | One-sided fee rate (0.5 bps) |
| `mode` | `str` | `"diff"` | `"diff"` (signal-driven) or `"label"` (fixed holding) |
| `ic_rolling_window` | `int` | `20` | Rolling window for IC chart |
| `bars_per_year` | `int` | `252` | Bars per year (annualization) |

---

## Cross-Section Backtest Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **`feature_path`** | `str` | *required* | Path to feature data file |
| **`feature_cols`** | `List[str]` | *required* | Feature column names to backtest |
| `raw_data_path` | `str` | `None` | Path to raw price file (raw mode) |
| `label_path` | `str` | `None` | Path to label file (label mode) |
| `use_raw_data` | `bool` | `True` | Use raw data mode (default) |
| `stock_col` | `str` | `"stock_id"` | Stock identifier column |
| `date_col` | `str` | `"date_id"` | Date column |
| `time_col` | `str` | `"time_id"` | Optional time column |
| `price_col` | `str` | `"close"` | Price column for label computation |
| `exec_price_type` | `str` | `"close"` | Price column for PnL execution |
| `industry_col` | `str` | `None` | Industry column (for neutralization) |
| `data_format` | `str` | `"long"` | `"long"`, `"wide"`, or `"multi_file"` |
| `winsorize_enabled` | `bool` | `True` | Enable outlier removal |
| `winsorize_method` | `str` | `"mad"` | `"mad"`, `"percentile"`, `"std"`, `"custom"` |
| `winsorize_n_sigma` | `float` | `3.0` | MAD/std multiplier |
| `neutralize_enabled` | `bool` | `False` | Enable industry neutralization |
| `neutralize_method` | `str` | `"regression"` | `"regression"`, `"demean"`, `"intra_industry"`, `"custom"` |
| `normalize_method` | `str` | `"min_max"` | `"min_max"`, `"zscore"`, `"rank"` |
| `h` | `int` | `1` | Holding horizon (forward return window) |
| `lag` | `int` | `1` | Execution delay (bars) |
| `decay_lags` | `List[int]` | `[1, 2, 5]` | IC Decay lag sequence |
| `weight_method` | `str` | `"group"` | `"group"` (quantile) or `"money"` (signal-proportional) |
| `n_groups` | `int` | `5` | Number of quantile groups |
| `display_modes` | `List[str]` | `["long_only","long_short"]` | Trading modes |
| `signal_direction` | `str` | `"momentum"` | `"momentum"` or `"mean_reversion"` |
| `fee_rate` | `float` | `0.00005` | One-sided fee rate (0.5 bps) |
| `initial_capital` | `float` | `1,000,000` | Initial capital |
| `ic_rolling_window` | `int` | `20` | IC rolling window |
| `bars_per_year` | `int` | `252` | Bars per year (annualization) |

---

## Project Structure

```
backtest_api/
├── main.py                      # API server entry point
├── requirements.txt
├── README.md
└── backtest_api/
    ├── __init__.py              # Public API exports
    ├── api.py                   # FastAPI routes
    ├── schemas.py               # Pydantic request/response models
    ├── base.py                  # BaseBacktest ABC + BacktestResult
    ├── config.py                # Timing config dataclasses
    ├── data_loader.py           # File I/O + format conversion
    ├── metrics.py               # Performance metrics
    ├── numba_utils.py           # Numba JIT rolling calculations
    ├── report.py                # Timing summary tables + charts
    ├── timing/
    │   ├── __init__.py
    │   ├── backtest.py          # TimingBacktest (main entry)
    │   ├── label.py             # Label computation from raw prices
    │   ├── signal.py            # Signal generation
    │   └── executor.py          # Timing execution engine
    └── cross_section/
        ├── __init__.py
        ├── config.py            # Cross-section config dataclasses
        ├── signal.py            # winsorize -> neutralize -> normalize
        ├── label.py             # Forward returns + IC decay
        ├── executor.py          # Weight construction + portfolio PnL
        ├── report.py            # Cross-section charts
        └── backtest.py          # CrossSectionBacktest (main entry)
```

---

---

# Backtest API (中文文档)

量化策略回测框架，同时提供 **Python 库** 和 **REST API**（FastAPI）。
支持 **择时回测**（单资产、信号驱动）和 **截面回测**（多资产、因子排名），高频回测模块后续开发。

---

## 功能特性

- **择时回测**：单资产信号驱动，支持 label / diff 两种执行模式
- **截面回测**：多资产因子评估，支持 group-based / money-based 两种权重构建方式
- **信号处理**：择时（滚动分位数/阈值/自定义）；截面（去极值 → 行业中性化 → 归一化）
- **指标体系**：年化收益率、夏普、Sortino、最大回撤、IC、Rank IC、IR、换手率
- **图表输出**：净值曲线、分组收益、组内 IC、IC cumsum、IC Decay 柱形图
- **REST API**：HTTP 运行回测，返回 JSON 指标 + base64 图片
- **文件格式**：Parquet、CSV、HDF5
- **数据格式**：长表（默认）、宽表、多文件目录

---

## 安装

```bash
pip install -r requirements.txt
```

---

## 择时回测 — 快速开始

```python
from backtest_api import TimingBacktest

bt = TimingBacktest(
    feature_path="data/features.parquet",
    label_path="data/labels.parquet",
    feature_cols=["alpha1", "alpha2"],
    label_col="ret_1",
    display_labels=[1, 5],           # 评估 label1 和 label5
    display_modes=["long_short"],     # 仅展示多空模式
    fee_rate=0.0001,                  # 单边万一
)
result = bt.run()
result.show()  # 打印表格 + 显示图表
```

### 原始数据模式（从价格自动计算标签）

```python
bt = TimingBacktest(
    feature_path="data/features.parquet",
    raw_data_path="data/prices.parquet",
    feature_cols=["alpha1"],
    price_col="close",
    use_raw_data=True,
    display_labels=[1, 3, 5],
)
result = bt.run()
```

---

## 截面回测 — 快速开始

```python
from backtest_api import CrossSectionBacktest

bt = CrossSectionBacktest(
    feature_path="data/features.parquet",
    raw_data_path="data/prices.parquet",
    use_raw_data=True,
    feature_cols=["alpha1"],
    stock_col="stock_id",
    date_col="date_id",
    price_col="close",
    n_groups=5,                       # 分5组
    display_modes=["long_short"],     # 多空
    fee_rate=0.0001,
)
result = bt.run()
result.show()
```

### 标签模式（预计算标签）

```python
bt = CrossSectionBacktest(
    feature_path="data/features.parquet",
    use_raw_data=False,
    label_path="data/labels.parquet",
    label_col="ret_1",
    feature_cols=["alpha1"],
    stock_col="stock_id",
    date_col="date_id",
    n_groups=5,
)
result = bt.run()
```

### 带行业中性化

```python
bt = CrossSectionBacktest(
    feature_path="data/features.parquet",
    raw_data_path="data/prices.parquet",
    use_raw_data=True,
    feature_cols=["alpha1"],
    stock_col="stock_id",
    date_col="date_id",
    price_col="close",
    industry_col="industry",
    neutralize_enabled=True,
    neutralize_method="regression",   # "regression" | "demean" | "intra_industry" | "custom"
    normalize_method="rank",
)
result = bt.run()
```

---

## 快速开始 — REST API

### 启动服务

```bash
python main.py
# 服务运行在 http://localhost:8000
# Swagger 文档: http://localhost:8000/docs
```

### 截面回测请求

```bash
curl -X POST http://localhost:8000/backtest/cross-section \
  -H "Content-Type: application/json" \
  -d '{
    "feature_path": "data/features.parquet",
    "raw_data_path": "data/prices.parquet",
    "use_raw_data": true,
    "feature_cols": ["alpha1"],
    "stock_col": "stock_id",
    "date_col": "date_id",
    "price_col": "close",
    "n_groups": 5,
    "display_modes": ["long_short"],
    "fee_rate": 0.0001,
    "include_charts": true
  }'
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/backtest/timing` | 运行择时回测 |
| `POST` | `/backtest/cross-section` | 运行截面回测 |
| `GET` | `/docs` | Swagger 文档（自动生成） |

---

## 完整参数调用示例 — 截面回测 (Python)

```python
from backtest_api import CrossSectionBacktest

bt = CrossSectionBacktest(
    # ========== 数据配置（必填） ==========
    feature_path="data/features.parquet",   # 特征文件路径（支持 .parquet/.csv/.h5）
    feature_cols=["alpha1", "alpha2"],       # 要回测的特征列名（必填）

    # ========== 模式 A：原始数据模式（默认，A/B 二选一） ==========
    raw_data_path="data/prices.parquet",    # 原始价格文件路径
    use_raw_data=True,                      # 默认 True

    # ========== 模式 B：标签模式（A/B 二选一） ==========
    # label_path="data/labels.parquet",     # 预计算标签文件
    # label_col="ret_1",                    # 标签列名
    # use_raw_data=False,                   # 切换到标签模式

    # ========== 列名配置 ==========
    stock_col="stock_id",                   # 股票标识列，默认 "stock_id"
    date_col="date_id",                     # 日期列，默认 "date_id"
    time_col="time_id",                     # 时间列（可选），默认 "time_id"
    price_col="close",                      # 价格列（label 计算用），默认 "close"
    exec_price_type="close",                # 执行价格列（PnL 计算用），默认 "close"
    industry_col="industry",                # 行业列（可选，中性化用）

    # ========== 数据格式 ==========
    data_format="long",                     # "long" | "wide" | "multi_file"，默认 "long"

    # ========== 去极值 ==========
    winsorize_enabled=True,                 # 启用去极值，默认 True
    winsorize_method="mad",                 # "mad" | "percentile" | "std" | "custom"，默认 "mad"
    winsorize_n_sigma=3.0,                  # MAD/标准差倍数，默认 3.0

    # ========== 行业中性化 ==========
    neutralize_enabled=False,               # 启用中性化，默认 False
    neutralize_method="regression",         # "regression" | "demean" | "intra_industry" | "custom"

    # ========== 归一化 ==========
    normalize_method="min_max",             # "min_max" | "zscore" | "rank"，默认 "min_max"

    # ========== Label / IC Decay ==========
    h=1,                                    # 持有期（前向收益窗口），默认 1
    lag=1,                                  # 执行延迟（bar 数），默认 1
    decay_lags=[1, 2, 5],                   # IC Decay 展示的 lag 序列，默认 [1, 2, 5]

    # ========== 执行 / 组合 ==========
    weight_method="group",                  # "group"（分组排名）| "money"（信号比例），默认 "group"
    n_groups=5,                             # 分位数组数，默认 5
    # group_weight_func=my_func,           # 自定义组内权重函数（仅 Python）
    display_modes=["long_only", "long_short"],  # 展示模式，默认两种
    signal_direction="momentum",            # "momentum" | "mean_reversion"，默认 "momentum"
    fee_rate=0.00005,                       # 单边费率，默认万分之 0.5
    show_before_fee=True,
    show_after_fee=True,
    initial_capital=1_000_000,              # 起始资金，默认 100 万

    # ========== 报告 / 图表 ==========
    ic_rolling_window=20,
    bars_per_year=252,
)

result = bt.run()       # 运行回测，返回 BacktestResult
result.show()           # 打印表格 + 显示图表

# 或者：
df = bt.evaluate()      # 只拿汇总 DataFrame
bt.report()             # 直接打印+画图
```

---

## 截面信号处理流程

对每个截面（每个时间点独立处理）：

```
feature[t] -> 去极值 -> 行业中性化 -> 归一化 -> signal[t] ∈ [-1, 1]
```

### 第一步：去极值（默认开启）

| 方法 | 说明 | 关键参数 |
|------|------|----------|
| `"mad"`（默认） | MAD 中位数绝对偏差法 | `winsorize_n_sigma=3.0` |
| `"percentile"` | 百分位截断 | `winsorize_lower=0.01, winsorize_upper=0.99` |
| `"std"` | 标准差截断 | `winsorize_n_sigma=3.0` |
| `"custom"` | 自定义函数 | `winsorize_func`（仅 Python） |

### 第二步：行业中性化（默认关闭）

| 方法 | 说明 |
|------|------|
| `"regression"`（默认） | 行业哑变量回归取残差 |
| `"demean"` | 减去行业均值 |
| `"intra_industry"` | 行业内标准化 |
| `"custom"` | 自定义函数 |

### 第三步：归一化

| 方法 | 输出范围 |
|------|----------|
| `"min_max"`（默认） | [-1, 1] |
| `"zscore"` | 截断到 [-1, 1] |
| `"rank"` | rank/N 映射到 [-1, 1] |

---

## 截面权重构建

### Group-Based（默认）

按信号排序分成 `n_groups` 组：

| 模式 | Momentum | Mean Reversion |
|------|----------|----------------|
| `long_short` | 多最高组（权重和=1），空最低组（权重和=-1） | 相反 |
| `long_only` | 只多最高组（权重和=1） | 只多最低组（权重和=1） |

组内默认等权，可通过 `group_weight_func` 自定义。

### Money-Based

直接按信号数值大小分配仓位：

| 模式 | 说明 |
|------|------|
| `long_short` | 正信号→多头（和=1），负信号→空头（和=-1） |
| `long_only` | 只取正信号，按比例分配（和=1） |

---

## 截面图表

| 图表 | 说明 |
|------|------|
| **净值曲线** | 费前/费后累计收益 |
| **分组收益曲线** | 每组一条线，好因子应呈单调性 |
| **分组 IC 柱形图** | 每组的截面 IC 均值 |
| **IC / Rank IC Cumsum** | IC 累计和曲线，标注时序均值 |
| **IC Decay 柱形图** | 横轴 lag，纵轴 IC 均值，展示因子时效性衰减 |

---

## 截面符号体系

| 符号 | 含义 | 默认 |
|------|------|------|
| `t` | 因子产生时间 | — |
| `h` | 持有期（前向收益窗口） | 1 |
| `lag` | 执行延迟（信号到成交的 bar 数） | 1 |
| `decay_lags` | IC Decay 柱形图的 lag 序列 | [1, 2, 5] |
| `n_groups` | 分位数组数 | 5 |

**IC Decay 公式**：`IC_d(lag) = mean_over_t[ corr_截面( feature(t), ret(t+lag : t+lag+h) ) ]`

---

## 指标说明

| 指标 | 说明 |
|------|------|
| Annualized Return | 年化收益率（复合年化增长率） |
| Total Return | 区间总收益率 |
| Volatility | 年化波动率 |
| Sharpe Ratio | 夏普比率（均值/波动率） |
| Sortino Ratio | Sortino 比率（仅考虑下行风险） |
| Max Drawdown | 最大回撤 |
| Max DD Recovery | 最大回撤修复时间（bar 数） |
| Turnover | 换手率（每 bar 平均仓位变化） |
| IC | 信息系数（Pearson 相关性：特征 vs 标签） |
| Rank IC | 排序信息系数（Spearman 秩相关） |
| IR | 信息比率（IC 均值 / IC 标准差） |

---

## 项目结构

```
backtest_api/
├── main.py                      # API 服务入口
├── requirements.txt
├── README.md
└── backtest_api/
    ├── __init__.py              # 公开 API 导出
    ├── api.py                   # FastAPI 路由
    ├── schemas.py               # Pydantic 请求/响应模型
    ├── base.py                  # 基类 BaseBacktest + BacktestResult
    ├── config.py                # 择时配置 dataclass
    ├── data_loader.py           # 文件加载 + 格式转换
    ├── metrics.py               # 绩效指标计算
    ├── numba_utils.py           # Numba JIT 滚动计算
    ├── report.py                # 择时汇总表格 + 图表
    ├── timing/
    │   ├── __init__.py
    │   ├── backtest.py          # TimingBacktest 主类
    │   ├── label.py             # 从原始价格计算标签
    │   ├── signal.py            # 信号生成
    │   └── executor.py          # 择时执行引擎
    └── cross_section/
        ├── __init__.py
        ├── config.py            # 截面配置 dataclass
        ├── signal.py            # 去极值 → 中性化 → 归一化
        ├── label.py             # 前向收益 + IC Decay
        ├── executor.py          # 权重构建 + 组合 PnL
        ├── report.py            # 截面图表
        └── backtest.py          # CrossSectionBacktest 主类
```

---

## License

MIT
