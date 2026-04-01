# Cross-Section Backtest Design Spec

## Overview

Phase 2 of the backtest framework: multi-asset cross-sectional factor backtesting. Evaluates factor quality by constructing portfolios based on cross-sectional signal rankings across a stock universe.

Inherits from `BaseBacktest`, mirrors the timing module structure (Approach A), shares `metrics.py` and `data_loader.py` (extended).

## Symbols & Conventions

| Symbol | Meaning | Default |
|--------|---------|---------|
| `t` | Factor/signal generation time | - |
| `h` | Holding horizon (forward return window) | 1 |
| `lag` | Execution delay (bars from signal to trade) | 1 |
| `decay_lags` | Lag sequence for IC Decay chart | [1, 2, 5] |
| `n_groups` | Number of quantile groups | 5 |

IC Decay formula: `IC_d(lag) = mean_over_t[ corr_cs( feature(t), ret(t+lag : t+lag+h) ) ]`

## Module Structure

```
backtest_api/
├── cross_section/
│   ├── __init__.py
│   ├── config.py          # CrossSectionConfig, CrossSectionSignalSpec,
│   │                      #   CrossSectionLabelSpec, CrossSectionExecutionSpec
│   ├── signal.py          # winsorize → neutralize → normalize
│   ├── label.py           # forward return + IC Decay computation
│   ├── executor.py        # signal → weight → portfolio PnL
│   ├── backtest.py        # CrossSectionBacktest orchestrator
│   └── report.py          # cross-section-specific charts
├── data_loader.py         # extended: wide_to_long, load_directory, load_cross_section_data
├── metrics.py             # reused as-is
├── api.py                 # extended: POST /backtest/cross-section
└── schemas.py             # extended: CrossSectionBacktestRequest/Response
```

---

## Section 1: Data Input & Loading

### Two data modes (same as timing)

- **Raw mode** (`use_raw_data=True`, default): `feature_path` + `raw_data_path` (price data). Labels computed automatically from prices.
- **Label mode** (`use_raw_data=False`): `feature_path` + `label_path` (pre-computed labels).

### Default data format: Long table

Standard input is a single large file with columns:
- Identifiers: `stock_id`, `date_id`, `time_id` (optional)
- Prices: `close`, `vwap`, `open`, etc.
- Features: one or more feature columns

Supported file formats: HDF5, CSV, Parquet.

### Price column semantics

- `price_col` (default `"close"`): used for **label computation** (forward returns for IC calculation)
- `exec_price_type` (default `"close"`): used for **execution returns** (PnL calculation)

These can differ: e.g., compute IC against close-to-close returns, but execute PnL at vwap. Same pattern as timing module.

### Directory rename

Existing `cross-section/` directory (with hyphen) must be renamed to `cross_section/` (with underscore) for Python import compatibility. The old `cross-section/idea.md` can be removed after the spec is finalized.

### Three input formats supported

| Format | `data_format` | Description |
|--------|--------------|-------------|
| Long (default) | `"long"` | Standard long table with stock_id, date_id, ... |
| Wide | `"wide"` | Columns = stock_ids, rows = timestamps, values = feature/price |
| Multi-file | `"multi_file"` | Directory path, one file per stock |

All formats are converted to long format internally.

### data_loader.py extensions

New functions (existing `load_file()` and `align_timestamps()` untouched):

```python
def wide_to_long(df, stock_col, time_cols) -> pd.DataFrame
def load_directory(dir_path, stock_col) -> pd.DataFrame
def load_cross_section_data(path, data_format, ...) -> pd.DataFrame
```

Processing requires grouping by time for all cross-sectional operations.

---

## Section 2: Signal Pipeline (Feature -> Signal)

Per cross-section (each time point independently):

```
feature[t] → Step 1: Winsorize → Step 2: Neutralize → Step 3: Normalize → signal[t] ∈ [-1, 1]
```

### Step 1: Winsorize (optional, enabled by default)

| Method | `winsorize_method` | Parameters | Default |
|--------|--------------------|------------|---------|
| MAD | `"mad"` | `winsorize_n_sigma=3.0` | Yes (default method) |
| Percentile | `"percentile"` | `winsorize_lower=0.01, winsorize_upper=0.99` | - |
| Std | `"std"` | `winsorize_n_sigma=3.0` | - |
| Custom | `"custom"` | `winsorize_func` | - |

### Step 2: Industry Neutralization (optional, disabled by default)

Requires `industry_col` to be specified.

| Method | `neutralize_method` | Description | Default |
|--------|---------------------|-------------|---------|
| Regression | `"regression"` | Industry dummy regression, take residuals | Yes (default method) |
| Demean | `"demean"` | Subtract industry mean | - |
| Intra-industry | `"intra_industry"` | Standardize within each industry, then merge | - |
| Custom | `"custom"` | `neutralize_func` | - |

### Step 3: Normalize

| Method | `normalize_method` | Output range |
|--------|-------------------|-------------|
| Min-Max | `"min_max"` (default) | [-1, 1] |
| Z-score | `"zscore"` | Truncated to [-1, 1] |
| Rank | `"rank"` | rank/N mapped to [-1, 1] |

### Config

```python
@dataclass
class CrossSectionSignalSpec:
    winsorize_enabled: bool = True
    winsorize_method: Literal["mad", "percentile", "std", "custom"] = "mad"
    winsorize_n_sigma: float = 3.0
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    winsorize_func: Optional[Callable] = None

    neutralize_enabled: bool = False
    neutralize_method: Literal["regression", "demean", "intra_industry", "custom"] = "regression"
    industry_col: Optional[str] = None
    neutralize_func: Optional[Callable] = None

    normalize_method: Literal["min_max", "zscore", "rank"] = "min_max"
```

### Functions

```python
# cross_section/signal.py

def winsorize(values: np.ndarray, method: str, **kwargs) -> np.ndarray
def neutralize(values: np.ndarray, industry: np.ndarray, method: str, **kwargs) -> np.ndarray
def normalize(values: np.ndarray, method: str) -> np.ndarray
def generate_cross_section_signals(
    feature_df: pd.DataFrame,
    spec: CrossSectionSignalSpec,
    stock_col: str,
    date_col: str,
    feature_col: str,
    industry_col: Optional[str] = None,
) -> pd.DataFrame
```

---

## Section 3: Label Computation & IC Decay

### Label computation

**Raw mode**: For each stock s at time t:
```
label(s, t, h) = price(s, t+h) / price(s, t) - 1
```

**Label mode**: User-provided pre-computed labels.

### IC Decay

Fixed `h`, varying `lag`, measures factor timeliness decay:

```
IC_decay(lag) = mean_over_t[ corr_cross_section( feature(t), ret(t+lag : t+lag+h) ) ]

where ret(t+lag : t+lag+h) = price(s, t+lag+h) / price(s, t+lag) - 1
```

### Config

```python
@dataclass
class CrossSectionLabelSpec:
    h: int = 1                                                    # holding horizon
    lag: int = 1                                                   # execution delay
    decay_lags: List[int] = field(default_factory=lambda: [1, 2, 5])
```

### Functions

```python
# cross_section/label.py

def compute_forward_returns(
    df: pd.DataFrame,
    stock_col: str,
    date_col: str,
    price_col: str,
    h: int,
    lag: int,
) -> pd.DataFrame

def compute_ic_decay(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    stock_col: str,
    date_col: str,
    feature_col: str,
    price_col: str,
    h: int,
    decay_lags: List[int],
) -> pd.DataFrame  # columns: lag, ic_mean, rank_ic_mean
```

---

## Section 4: Executor (Signal -> Weight -> PnL)

### Two weight methods

#### Method 1: Group-Based (default)

Rank stocks by signal into `n_groups` quantile groups per cross-section.

**long_short + momentum** (default):
- Long: group n (highest signals), weight sum = 1
- Short: group 1 (lowest signals), weight sum = -1
- Net exposure = 0

**long_short + mean_reversion**: opposite (long group 1, short group n)

**long_only + momentum**: only hold group n, weight sum = 1

**long_only + mean_reversion**: only hold group 1, weight sum = 1

Group-internal weights: equal weight (default) or custom function via `group_weight_func`.

#### Method 2: Money-Based

Allocate weights directly proportional to signal magnitude. No grouping.

**long_short**:
- Positive signals → long, `weight_i = signal_i / sum(positive signals)`, total = 1
- Negative signals → short, `weight_i = signal_i / sum|negative signals|`, total = -1

**long_only + momentum**: only positive signal stocks, weights proportional to signal, sum = 1

**long_only + mean_reversion**: only negative signal stocks, weights proportional to |signal|, sum = 1

### PnL calculation

```
For each bar t:
  portfolio_return(t) = sum( weight_i(t) * return_i(t+lag) )
  fee(t) = fee_rate * sum|weight_i(t) - weight_i(t-1)|
  net_return(t) = portfolio_return(t) - fee(t)
  position_i(t) = initial_capital * weight_i(t)
```

### Weight constraints

- **long_short**: long total = 1, short total = -1, net = 0
- **long_only**: long total = 1, no short positions

### Config

```python
@dataclass
class CrossSectionExecutionSpec:
    weight_method: Literal["group", "money"] = "group"
    n_groups: int = 5
    group_weight_func: Optional[Callable] = None
    display_modes: List[Literal["long_only", "long_short"]] = field(
        default_factory=lambda: ["long_only", "long_short"]
    )
    signal_direction: Literal["momentum", "mean_reversion"] = "momentum"
    fee_rate: float = 0.00005
    show_before_fee: bool = True
    show_after_fee: bool = True
    initial_capital: float = 1_000_000
```

### Functions

```python
# cross_section/executor.py

def assign_quantile_groups(signals: pd.Series, n_groups: int) -> pd.Series
def compute_weights_group(
    signals: pd.Series, groups: pd.Series, n_groups: int,
    direction: str, mode: str, weight_func: Optional[Callable] = None,
) -> pd.Series
def compute_weights_money(signals: pd.Series, direction: str, mode: str) -> pd.Series
def execute_cross_section_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    spec: CrossSectionExecutionSpec,
    label_spec: CrossSectionLabelSpec,
    stock_col: str,
    date_col: str,
    price_col: str,
) -> dict
    # Returns: portfolio_gross_pnl, portfolio_net_pnl, weights,
    #          group_returns, group_ic
```

---

## Section 5: Report — Cross-Section Charts

### Charts

| # | Chart | X-axis | Y-axis | Description |
|---|-------|--------|--------|-------------|
| 1 | PnL Curve | Time | Cumulative return | Before/after fee portfolio equity curve. Reuse `plot_pnl_curve` logic. |
| 2 | Quantile Return | Time | Cumulative return | One line per group. Good factor = monotonic across groups. |
| 3 | Group IC Bar | Group (1~n) | IC mean | Per-group: mean of time-series IC within group. Evaluates factor power distribution. |
| 4 | IC/Rank IC Cumsum | Time | Cumulative IC | IC and Rank IC cumsum curves + time-series mean annotation. |
| 5 | IC Decay Bar | Lag (decay_lags) | IC mean | Fixed h, varying lag. Shows factor timeliness decay. |

### Summary Table

Reuses `metrics.py`: Annualized Return, Total Return, Volatility, Sharpe, Sortino, Max Drawdown, Max Drawdown Recovery, Turnover, IC, Rank IC, IR.

### Functions

```python
# cross_section/report.py

def plot_quantile_returns(group_returns: pd.DataFrame, n_groups: int, title: str) -> Figure
def plot_group_ic(group_ic: pd.DataFrame, title: str) -> Figure
def plot_ic_cumsum(timestamps, ic_series, rank_ic_series, title: str) -> Figure
def plot_ic_decay(decay_lags, ic_means, rank_ic_means, title: str) -> Figure
```

---

## Section 6: CrossSectionBacktest Main Class

### Full parameter list

```python
class CrossSectionBacktest(BaseBacktest):
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
        decay_lags: List[int] = [1, 2, 5],
        # ---- execution ----
        weight_method: Literal["group", "money"] = "group",
        n_groups: int = 5,
        group_weight_func: Optional[Callable] = None,
        display_modes: List[Literal["long_only", "long_short"]] = ["long_only", "long_short"],
        signal_direction: Literal["momentum", "mean_reversion"] = "momentum",
        fee_rate: float = 0.00005,
        show_before_fee: bool = True,
        show_after_fee: bool = True,
        initial_capital: float = 1_000_000,
        # ---- report ----
        ic_rolling_window: int = 20,
        bars_per_year: int = 252,
    )
```

### run() pipeline

```
load_data()
  → load feature + (raw_data or label)
  → convert to long format (internal)
validate()

for each feature_col:
  ├── generate_cross_section_signals()
  │     group by time → winsorize → neutralize → normalize
  ├── compute_forward_returns() [raw mode] or use label_col
  ├── for each display_mode:
  │     ├── execute_cross_section_backtest()
  │     ├── build_summary_table()
  │     ├── plot_pnl_curve()
  │     ├── plot_quantile_returns()
  │     └── plot_group_ic()
  ├── plot_ic_cumsum()
  └── compute_ic_decay() → plot_ic_decay()

return BacktestResult
```

### API endpoint

- `POST /backtest/cross-section`
- Request: `CrossSectionBacktestRequest` (all params except Callable types)
- Response: `CrossSectionBacktestResponse` (status, summary_tables, charts, raw_data)
- Added to `api.py` and `schemas.py`
