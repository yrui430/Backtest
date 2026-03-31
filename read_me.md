### For backtesting of two main types of data, bar data and tick data, it is also necessary to determine whether the bar data is time series data or cross-sectional data.
### 对于两大类数据的回测，bar数据和tick数据，对于bar数据，需要确定是时序数据还是截面数据
### 对于另类数据，如新闻类，基本面类数据，后续维护
### tick数据的回测待更新和维护，时序和截面数据可以很好的调用

功能性api仅适用于输入数据之后，回测feature的表现，还可对比benchmark数据

一、定义时序回测类
1.确定可输入数据集类型，可接受的数据类型 h5,csv，parquet
2.需要输入需要包含feature和label，如果没有label，可输入原始数据，如果输入原始数据，在原数据选择true，在label选择false，如果输入原始数据，需要用户定义输入数据的列名，需要包含time和close或者vwap，用户需要自定义是用vwap做回测还是用close做回测，无论是定义label还是定义数据都需要用户选择字段 label = int 1 ,默认是1，即信号发出来signal之后的下一个bar开仓进去，持有1个bar，（做close和label的一阶差分）如果选择是2的话，需要持有2个bar，即预测2个bar之后的收益率，但是我们后续做回测的时候我们的position默认是0.5，也就是近似认为是0.5个仓位去做，而下一个bar的signal，我们认为是另一个0.5个仓位去做，而最终的收益率是两个仓位加和起来。你需要注意label和feature的时间能不能对齐（当然需要先让用户选择输入的列的名称了），如果不能够对齐，请你print一个请对其数据之后再来，中断后续流程，我们从feature生成signal的范围有两个，一个分位数法，分位数默认是0.8和0.2，rolling_window的方式，默认是100，signal默认和position是signal超过分位数上端为long，低于分位数下端为short，而这里用户也可以定义自己的选择，是展示long only，还是short only，还是long short，还是任两者，或者all，如果选用阈值法，需要定义你选择的阈值，可以是绝对的的数字，另外如果可以定义feature -> signal的映射，请输入映射函数import，但默认为None
3.展示的部分，需要用户选择是和label1，还是label2，还是label5，也可以多选，默认统一是label1，默认的lag是1，也就是出信号在下一个bar交易，也可以是lag2，需要用户定义，需要选择费率，默认是万五做单边，而输出的结果，第一你需要做到数形结合，用户需要定义费前，费后，（以及如果用户选择自己的feature-> signal映射条件用户需要选择是否添加换仓hurdle 的条件，默认是Equal weighted portfolio construction，如果用户没有选择映射，就不会调用这一个板块，同时用户也可以自定义hurdle条件），表格需要包含的字段是，区间内年化收益率，区间总收益率，波动率，夏普比率，以及Sortino Ratio，换手率，区间内最大回撤，最大回撤修复时间，rank ic，ic ，ir 。第二部分你需要做出来净值曲线pnl（费前和费后，记住作图的时候都要用英文），rank ic 和 ic rolling window，这里也需要rolling window的窗口，以及ic本身是和label的相关性不是吗，这里你需要避免用任何的rolling window的函数，你需要用numba做加速

---

## API 使用说明

### 安装依赖

```bash
pip install pandas numpy numba matplotlib scipy tables
```

### 快速开始

#### 1. 有 Label 数据

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
result.show()  # 输出表格 + 图表
```

#### 2. 无 Label，用原始价格数据

```python
bt = TimingBacktest(
    feature_path="features.parquet",
    raw_data_path="ohlcv.parquet",
    time_col="timestamp",
    feature_cols=["alpha_001"],
    price_col="close",
    use_raw_data=True,
    exec_price_type="close",    # 用 close 价成交，也可选 "vwap"
    display_labels=[1, 2, 5],   # 同时展示 label1, label2, label5
    lag=1,                       # 出信号后下一个bar交易
    fee_rate=0.00005,            # 万五单边
    display_modes=["long_only", "short_only", "long_short"],
)
result = bt.run()
result.show()
```

#### 3. 自定义 Signal 映射 + Hurdle

```python
from my_signals import custom_signal_mapper, my_hurdle_func

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

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `feature_path` | str | 必填 | Feature 文件路径 |
| `time_col` | str | "timestamp" | 时间戳列名 |
| `feature_cols` | List[str] | 必填 | Feature 列名列表 |
| `label_path` | str | None | Label 文件路径 (模式A) |
| `label_col` | str | None | Label 列名 (模式A) |
| `raw_data_path` | str | None | 原始数据路径 (模式B) |
| `price_col` | str | None | 价格列名 (模式B) |
| `use_raw_data` | bool | False | True=用原始数据算label |
| `exec_price_type` | "close"/"vwap" | "close" | 成交价类型 |
| `display_labels` | List[int] | [1] | 展示哪些 label horizon |
| `lag` | int | 1 | 信号延迟几个bar交易 |
| `signal_method` | "quantile"/"threshold"/"custom" | "quantile" | 信号生成方法 |
| `upper_quantile` | float | 0.8 | 分位数上界 |
| `lower_quantile` | float | 0.2 | 分位数下界 |
| `rolling_window` | int | 100 | 分位数 rolling 窗口 |
| `upper_threshold` | float | None | 阈值法上界 |
| `lower_threshold` | float | None | 阈值法下界 |
| `signal_direction` | "momentum"/"mean_reversion" | "momentum" | 信号方向 |
| `signal_mapper` | Callable | None | 自定义映射函数 |
| `display_modes` | List[str] | ["long_only","short_only","long_short"] | 展示模式 |
| `fee_rate` | float | 0.00005 | 单边费率(万五) |
| `hurdle_enabled` | bool | False | 是否启用换仓hurdle |
| `hurdle_func` | Callable | None | 自定义hurdle函数 |
| `ic_rolling_window` | int | 20 | IC rolling 窗口 |
| `bars_per_year` | int | 252 | 年化参数 |

### 输出内容

**表格** (每个 feature × label_horizon × display_mode 一组):
- Annualized Return, Total Return, Volatility, Sharpe Ratio, Sortino Ratio
- Turnover, Max Drawdown, Max Drawdown Recovery Time
- Rank IC, IC, IR

**图表** (英文标注):
- PnL Curve (Before Fee / After Fee)
- Rolling IC (Pearson) + Rolling Rank IC (Spearman)

### 项目结构

```
backtest_api/
├── __init__.py          # Package init
├── base.py              # BaseBacktest 抽象基类
├── config.py            # 配置 dataclass
├── data_loader.py       # 数据加载 (h5/csv/parquet)
├── metrics.py           # 绩效指标计算
├── numba_utils.py       # numba 加速 rolling 函数
├── report.py            # 表格 + 图表生成
├── timing/
│   ├── __init__.py
│   ├── backtest.py      # TimingBacktest 主类
│   ├── label.py         # Label 计算
│   ├── signal.py        # Signal 生成
│   └── executor.py      # 执行 & PnL 计算
└── tests/               # 61 个测试
```
