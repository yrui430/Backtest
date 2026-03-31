# 时序回测 API 设计规格书

> 本文档定义 `backtest_api` 项目中 **时序回测（Timing Backtest）** 模块的完整设计。
> 基于 `read_me.md` 的需求严格实现。

---

## 1. 项目总览

### 1.1 定位

功能性 API，输入数据后回测 feature 的表现，可对比 benchmark。项目分三期：

| 阶段 | 模块 | 状态 |
|------|------|------|
| Phase 1 | 时序回测 `TimingBacktest` | **本期实现** |
| Phase 2 | 截面回测 `CrossSectionalBacktest` | 待定 |
| Phase 3 | 高频回测 `HFBacktest` | 待定 |

### 1.2 架构模式

Strategy Pattern — `BaseBacktest` 抽象基类，三种回测类型继承同一接口。

### 1.3 模块结构

```
backtest_api/
├── base.py                # BaseBacktest 抽象基类
├── config.py              # DataSpec, LabelSpec, SignalSpec, ExecutionSpec 等 dataclass
├── metrics.py             # 共享指标计算 (Sharpe, MDD, IC 等)
├── report.py              # 共享报告生成 (表格 + 图表)
├── numba_utils.py         # numba 加速的 rolling 计算函数
├── timing/
│   ├── __init__.py
│   ├── backtest.py        # TimingBacktest(BaseBacktest) 主类
│   ├── label.py           # Label 计算 (close/vwap 一阶差分)
│   ├── signal.py          # Feature → Signal 映射 (分位数/阈值/自定义)
│   └── executor.py        # 信号执行 & 仓位叠加逻辑
├── cross_sectional/       # Phase 2 占位
├── hf/                    # Phase 3 占位
└── read_me.md
```

---

## 2. 数据输入层

### 2.1 支持的文件格式

| 格式 | 读取方式 |
|------|----------|
| `.parquet` | `pd.read_parquet()` |
| `.csv` | `pd.read_csv()` |
| `.h5` | `pd.read_hdf()` |

根据文件后缀名自动选择读取方式。

### 2.2 输入模式

```python
@dataclass
class TimingBacktestConfig:
    # ---- 数据路径 ----
    feature_path: str                        # feature 文件路径（必填）

    # 模式 A（有 label）: 提供 label_path + label_col
    label_path: Optional[str] = None         # label 文件路径
    label_col: Optional[str] = None          # label 列名

    # 模式 B（无 label，用原始数据）: 提供 raw_data_path + price_col
    raw_data_path: Optional[str] = None      # 原始 OHLCV 数据路径
    price_col: Optional[str] = None          # close 或 vwap 列名

    # ---- 列名定义（用户必须指定） ----
    time_col: str = "timestamp"              # timestamp 列名
    feature_cols: List[str] = field(default_factory=list)  # feature 列名列表

    # ---- 数据模式 ----
    use_raw_data: bool = False               # False=有label, True=用原始数据算label
    exec_price_type: Literal["close", "vwap"] = "close"
    # 仅 use_raw_data=True 时生效，决定 label 计算方式和成交价
```

**互斥校验**：
- `use_raw_data=False` 时，`label_path` 和 `label_col` 必填
- `use_raw_data=True` 时，`raw_data_path` 和 `price_col` 必填

当 `use_raw_data=True` 时，内部通过一阶差分计算 label：
```
label_1 = price[t+1] - price[t]          # label_horizon=1
label_2 = price[t+2] - price[t]          # label_horizon=2
label_5 = price[t+5] - price[t]          # label_horizon=5
```

### 2.3 Timestamp 对齐校验

在 `load_data()` 阶段：

1. 用户已指定列名后，检查 feature 和 label（或原始数据）的 timestamp 列
2. 对两个 DataFrame 做 timestamp 交集
3. **如果交集为空或交集比例过低（< 50%），print 提示信息并中断**：
   ```
   "时间戳无法对齐，请对齐数据之后再来。Feature 时间范围: [start, end], Label 时间范围: [start, end], 交集比例: X%"
   ```
4. 中断后续全部流程，不继续执行

---

## 3. Label 定义

### 3.1 LabelSpec

```python
@dataclass
class LabelSpec:
    label_horizon: int = 1
    # label_horizon=1: 信号发出后下一个bar开仓, 持有1个bar
    # label_horizon=2: 持有2个bar
    # label_horizon=5: 持有5个bar
    # 用户可在展示时多选: [1, 2, 5]

    lag: int = 1
    # lag=1: 出信号后下一个bar交易（默认）
    # lag=2: 出信号后第二个bar交易

    exec_price_type: Literal["close", "vwap"] = "close"
    # 决定成交价: close → 收盘价成交, vwap → VWAP 价成交
```

### 3.2 多 Label 展示

用户可同时选择多个 `label_horizon` 展示：

```python
display_labels: List[int] = [1]  # 默认 label1, 可选 [1, 2, 5] 等
```

每个 label_horizon 独立计算绩效指标，输出到同一张表格的不同行。

### 3.3 仓位叠加机制

当 `label_horizon = N` 时：
- 每个 bar 发出的信号占 `1/N` 仓位
- 同时持有最多 N 个仓位（来自不同 bar 的信号）
- 最终收益率 = 各仓位收益之和

示例（`label_horizon=2`）:
```
t=1: signal=long → 开 0.5 仓位, 持有到 t=3
t=2: signal=short → 开 0.5 仓位, 持有到 t=4
t=3: 总收益 = 0.5*(price[3]-price[1]) + 0.5*(price[4]-price[2]) 的逐 bar 累加
```

---

## 4. 信号生成

### 4.1 SignalSpec

```python
@dataclass
class SignalSpec:
    method: Literal["quantile", "threshold", "custom"] = "quantile"

    # 分位数法参数
    upper_quantile: float = 0.8
    lower_quantile: float = 0.2
    rolling_window: int = 100

    # 阈值法参数
    upper_threshold: Optional[float] = None
    lower_threshold: Optional[float] = None

    # 信号方向
    signal_direction: Literal["momentum", "mean_reversion"] = "momentum"
    # momentum (默认): feature > upper → long, feature < lower → short
    # mean_reversion: feature > upper → short, feature < lower → long (信号取反)

    # 自定义映射
    signal_mapper: Optional[Callable[[pd.Series], pd.Series]] = None
    # 用户 import 的映射函数, 默认 None

    # 展示模式（可多选，list 形式）
    display_modes: List[Literal["long_only", "short_only", "long_short"]] = field(
        default_factory=lambda: ["long_only", "short_only", "long_short"]
    )
    # 默认展示全部三组; 用户可选其中任意子集, 如 ["long_only", "long_short"]
```

### 4.2 信号逻辑

#### 分位数法（默认）

对每个 feature 列，用 **numba 加速的 rolling window** 计算历史分位数：
```
upper_band[t] = quantile(feature[t-window:t], upper_quantile)  # numba 实现
lower_band[t] = quantile(feature[t-window:t], lower_quantile)  # numba 实现

# signal_direction = "momentum" (默认):
signal[t] = +1 (long)   if feature[t] > upper_band[t]
signal[t] = -1 (short)  if feature[t] < lower_band[t]
signal[t] =  0 (flat)   otherwise

# signal_direction = "mean_reversion" (信号取反):
signal[t] = -1 (short)  if feature[t] > upper_band[t]
signal[t] = +1 (long)   if feature[t] < lower_band[t]
signal[t] =  0 (flat)   otherwise
```

#### 阈值法

```
# signal_direction = "momentum" (默认):
signal[t] = +1 (long)   if feature[t] > upper_threshold
signal[t] = -1 (short)  if feature[t] < lower_threshold
signal[t] =  0 (flat)   otherwise

# signal_direction = "mean_reversion" (信号取反):
signal[t] = -1 (short)  if feature[t] > upper_threshold
signal[t] = +1 (long)   if feature[t] < lower_threshold
signal[t] =  0 (flat)   otherwise
```

**实现方式**：momentum 为默认逻辑，mean_reversion 只需对最终 signal 乘以 -1。

#### 自定义映射

```python
# 用户传入:
def my_mapper(feature_series: pd.Series) -> pd.Series:
    """返回 signal series, 值域 {-1, 0, +1}"""
    ...

signal_spec = SignalSpec(method="custom", signal_mapper=my_mapper)
```

### 4.3 展示模式过滤

根据 `display_modes` 列表，对信号做过滤后分别计算绩效：

| display_mode | 行为 |
|---|---|
| `long_only` | 仅保留 signal=+1 的交易，short 时段 position=0 |
| `short_only` | 仅保留 signal=-1 的交易，long 时段 position=0 |
| `long_short` | 保留 signal=+1 和 signal=-1，完整多空 |

用户通过 `display_modes` 选择展示其中任意子集，默认展示全部三组。

### 4.4 多 Feature 处理

当 `feature_cols` 包含多个 feature 时：
- **信号生成**：对每个 feature 独立生成信号，独立计算绩效
- **IC / Rank IC**：对每个 feature 独立计算与 label 的相关性
- **输出表格**：每个 feature × label_horizon × display_mode 为一行

---

## 5. 执行与费率

### 5.1 ExecutionSpec

```python
@dataclass
class ExecutionSpec:
    fee_rate: float = 0.00005      # 万五单边
    show_before_fee: bool = True
    show_after_fee: bool = True

    # 换仓 hurdle（仅自定义映射时可用）
    hurdle_enabled: bool = False
    hurdle_func: Optional[Callable] = None
    # 默认 Equal Weighted Portfolio Construction
    # 如果 signal_mapper 为 None → hurdle 模块整体跳过
```

### 5.2 执行逻辑

1. 信号产生于 bar `t`
2. 在 bar `t + lag` 执行交易
3. 成交价 = `exec_price_type` 对应的价格列（close 或 vwap）
4. 费用 = `|position_change| * exec_price * fee_rate * 2`（开仓+平仓双边）
5. 持有 `label_horizon` 个 bar 后平仓

### 5.3 Hurdle 模块

仅当 `signal_mapper is not None` 时激活：

- `hurdle_enabled=False`（默认）: Equal Weighted，不做换仓过滤
- `hurdle_enabled=True`: 调用 `hurdle_func` 判断是否执行换仓
- 用户可自定义 hurdle 条件函数

---

## 6. 输出

### 6.1 绩效表格

每个 `(label_horizon, display_mode, fee_type)` 组合输出一行：

| 字段 | 说明 |
|------|------|
| Annualized Return | 区间内年化收益率 |
| Total Return | 区间总收益率 |
| Volatility | 收益率波动率 |
| Sharpe Ratio | 年化夏普比率 |
| Sortino Ratio | 下行风险调整收益 |
| Turnover | 换手率 |
| Max Drawdown | 区间内最大回撤 |
| Max Drawdown Recovery Time | 最大回撤修复时间（bar 数） |
| Rank IC | Spearman rank 相关系数 (feature vs label) |
| IC | Pearson 相关系数 (feature vs label) |
| IR | Information Ratio = IC_mean / IC_std |

### 6.2 图表

所有图表标注使用**英文**。

#### 图 1: PnL Curve
- Before Fee 和 After Fee 两条线
- X 轴: timestamp, Y 轴: Cumulative Return
- 每个 display_mode 一张图

#### 图 2: IC Rolling Window
- IC = Pearson correlation(feature, label) 在 rolling window 内的值
- 用户需指定 `ic_rolling_window` 参数
- **必须用 numba 实现，禁止 pandas rolling**

#### 图 3: Rank IC Rolling Window
- Rank IC = Spearman rank correlation(feature, label) 在 rolling window 内的值
- 同样使用 numba 加速
- **必须用 numba 实现，禁止 pandas rolling**

---

## 7. numba 加速约束

以下计算**禁止**使用 `pd.Series.rolling()`、`pd.DataFrame.rolling()` 等 pandas rolling 函数：

| 计算项 | numba 函数签名 |
|--------|---------------|
| Rolling 分位数 (信号生成) | `rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray` |
| Rolling IC | `rolling_pearson(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray` |
| Rolling Rank IC | `rolling_spearman(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray` |

所有 numba 函数集中在 `numba_utils.py` 中，使用 `@numba.njit` 装饰器。

---

## 8. API 调用接口

### 8.1 最简调用（有 label）

```python
from backtest_api.timing import TimingBacktest

bt = TimingBacktest(
    feature_path="features.parquet",
    label_path="labels.parquet",
    time_col="timestamp",
    feature_cols=["alpha_001", "alpha_002"],
    label_col="ret_1bar",
)
result = bt.run()
result.show()          # 输出表格 + 图表
```

### 8.2 原始数据调用（无 label）

```python
bt = TimingBacktest(
    feature_path="features.parquet",
    raw_data_path="ohlcv.parquet",
    time_col="timestamp",
    feature_cols=["alpha_001"],
    price_col="close",
    use_raw_data=True,
    exec_price_type="vwap",
    display_labels=[1, 2, 5],
    lag=1,
    fee_rate=0.00005,
    display_modes=["long_only", "short_only", "long_short"],
)
result = bt.run()
result.show()
```

### 8.3 自定义映射调用

```python
from my_signals import custom_signal_mapper

bt = TimingBacktest(
    feature_path="features.parquet",
    label_path="labels.parquet",
    time_col="timestamp",
    feature_cols=["alpha_001"],
    label_col="ret_1bar",
    signal_method="custom",
    signal_mapper=custom_signal_mapper,
    hurdle_enabled=True,
    hurdle_func=my_hurdle_func,
)
result = bt.run()
result.show()
```

---

## 9. BaseBacktest 接口

```python
from abc import ABC, abstractmethod

class BaseBacktest(ABC):

    @abstractmethod
    def load_data(self) -> None:
        """加载数据, 校验格式, 检查 timestamp 对齐"""
        ...

    @abstractmethod
    def validate(self) -> None:
        """校验配置合法性"""
        ...

    @abstractmethod
    def run(self) -> "BacktestResult":
        """执行回测主流程, 返回结果对象"""
        ...

    @abstractmethod
    def evaluate(self) -> pd.DataFrame:
        """计算绩效指标表格"""
        ...

    @abstractmethod
    def report(self) -> None:
        """生成图表 + 表格输出"""
        ...
```

`TimingBacktest`、`CrossSectionalBacktest`、`HFBacktest` 均继承此接口。

---

## 10. 错误处理

| 场景 | 行为 |
|------|------|
| Timestamp 不对齐 | `print("时间戳无法对齐，请对齐数据之后再来。")` → 中断 |
| 文件格式不支持 | `raise ValueError("不支持的文件格式: {ext}")` |
| Feature 列不存在 | `raise KeyError("Feature 列 '{col}' 不存在于数据中")` |
| Label 列不存在 | `raise KeyError("Label 列 '{col}' 不存在于数据中")` |
| use_raw_data=True 但未指定 price_col | `raise ValueError("使用原始数据时必须指定 price_col")` |
| signal_mapper=None 但 method="custom" | `raise ValueError("自定义映射模式需要提供 signal_mapper")` |
| hurdle_enabled=True 但 signal_mapper=None | 忽略 hurdle，按默认 Equal Weighted 处理 |
