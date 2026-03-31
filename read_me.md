### For backtesting of two main types of data, bar data and tick data, it is also necessary to determine whether the bar data is time series data or cross-sectional data.
### 对于两大类数据的回测，bar数据和tick数据，对于bar数据，需要确定是时序数据还是截面数据
### 对于另类数据，如新闻类，基本面类数据，后续维护
### tick数据的回测待更新和维护，时序和截面数据可以很好的调用

功能性api仅适用于输入数据之后，回测feature的表现，还可对比benchmark数据

一、定义时序回测类
1.确定可输入数据集类型，可接受的数据类型 h5,csv，parquet
2.需要输入需要包含feature和label，如果没有label，可输入原始数据，如果输入原始数据，在原数据选择true，在label选择false，如果输入原始数据，需要用户定义输入数据的列名，需要包含time和close或者vwap，用户需要自定义是用vwap做回测还是用close做回测，无论是定义label还是定义数据都需要用户选择字段 label = int 1 ,默认是1，即信号发出来signal之后的下一个bar开仓进去，持有1个bar，（做close和label的一阶差分）如果选择是2的话，需要持有2个bar，即预测2个bar之后的收益率，但是我们后续做回测的时候我们的position默认是0.5，也就是近似认为是0.5个仓位去做，而下一个bar的signal，我们认为是另一个0.5个仓位去做，而最终的收益率是两个仓位加和起来。你需要注意label和feature的时间能不能对齐（当然需要先让用户选择输入的列的名称了），如果不能够对齐，请你print一个请对其数据之后再来，中断后续流程，我们从feature生成signal的范围有两个，一个分位数法，分位数默认是0.8和0.2，rolling_window的方式，默认是100，signal默认和position是signal超过分位数上端为long，低于分位数下端为short，而这里用户也可以定义自己的选择，是展示long only，还是short only，还是long short，还是任两者，或者all，如果选用阈值法，需要定义你选择的阈值，可以是绝对的的数字，另外如果可以定义feature -> signal的映射，请输入映射函数import，但默认为None
3.展示的部分，需要用户选择是和label1，还是label2，还是label5，也可以多选，默认统一是label1，默认的lag是1，也就是出信号在下一个bar交易，也可以是lag2，需要用户定义，需要选择费率，默认是万五做单边，而输出的结果，第一你需要做到数形结合，用户需要定义费前，费后，（以及如果用户选择自己的feature-> signal映射条件用户需要选择是否添加换仓hurdle 的条件，默认是Equal weighted portfolio construction，如果用户没有选择映射，就不会调用这一个板块，同时用户也可以自定义hurdle条件），表格需要包含的字段是，区间内年化收益率，区间总收益率，波动率，夏普比率，以及Sortino Ratio，换手率，区间内最大回撤，最大回撤修复时间，rank ic，ic ，ir 。第二部分你需要做出来净值曲线pnl（费前和费后，记住作图的时候都要用英文），rank ic 和 ic rolling window，这里也需要rolling window的窗口，以及ic本身是和label的相关性不是吗，这里你需要避免用任何的rolling window的函数，你需要用numba做加速