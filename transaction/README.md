# A股交易代理系统

这是一个基于Python的A股交易代理系统，可以帮助您获取股票数据、进行技术分析并生成交易建议。

## 功能特点

- 自动获取A股市场数据
- 计算常用技术指标（MA、RSI、MACD等）
- 生成交易信号和建议
- 股票筛选功能
- 每日市场报告生成

## 安装依赖

1. 确保您已安装Python 3.7+
2. 安装所需的依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行示例脚本

```bash
python example.py
```

这将生成一个每日市场报告，包含：
- 大盘指数信息
- 买入推荐股票
- 卖出推荐股票

### 2. 使用交易代理API

```python
from agents.stock_agent import StockAgent

# 初始化代理
agent = StockAgent()

# 分析单个股票
analysis = agent.analyze_stock("000001")  # 分析平安银行

# 获取每日报告
daily_report = agent.get_daily_report()

# 使用自定义条件筛选股票
criteria = {
    'min_price': 10,
    'max_price': 50,
    'min_volume': 2000000
}
screened_stocks = agent.screen_stocks(criteria)
```

## 项目结构

```
transaction/
├── agents/             # 交易代理实现
├── data/              # 数据存储目录
├── strategies/        # 交易策略实现
├── utils/            # 工具函数
├── example.py        # 示例脚本
└── requirements.txt  # 项目依赖
```

## 注意事项

1. 本系统仅供学习和参考使用，不构成投资建议
2. 使用前请确保您有稳定的网络连接
3. 建议在A股交易时间内使用，以获取最新数据
4. 技术指标和交易信号仅供参考，请结合其他因素做出投资决策

## 数据来源

本系统使用akshare库获取A股市场数据，数据源包括：
- 股票日线数据
- 指数数据
- 股票列表

## TODO
使用LLM 进行选股

## 贡献指南

欢迎提交问题和改进建议！

## 许可证

MIT License 