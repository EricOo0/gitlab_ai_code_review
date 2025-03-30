import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import StockDataFetcher
from strategies.base_strategy import BaseStrategy
import pandas as pd
from typing import List, Dict, Union
from datetime import datetime, timedelta

class StockAgent:
    def __init__(self):
        """初始化股票交易代理"""
        self.data_fetcher = StockDataFetcher()
        self.strategy = BaseStrategy()
        
    def analyze_stock(self, stock_code: str, days: int = 90) -> Dict[str, Union[str, float]]:
        """分析单个股票并生成交易建议
        
        Args:
            stock_code (str): 股票代码
            days (int): 分析的历史数据天数
        
        Returns:
            Dict: 包含分析结果和建议的字典
        """
        # 获取股票数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        df = self.data_fetcher.fetch_stock_daily(stock_code, start_date, end_date)
        if df is None:
            return {"error": f"无法获取股票{stock_code}的数据"}
        
        # 计算技术指标
        df = self.strategy.calculate_ma(df)
        df = self.strategy.calculate_rsi(df)
        df = self.strategy.calculate_macd(df)
        
        # 生成交易信号
        signals = self.strategy.generate_signals(df)
        
        # 确保日期被格式化为字符串
        if isinstance(signals['date'], pd.Timestamp):
            signals['date'] = signals['date'].strftime('%Y-%m-%d')
        
        # 添加基本面数据
        signals['stock_code'] = stock_code
        return signals
    
    def screen_stocks(self, criteria: Dict = None) -> List[Dict]:
        """根据给定条件筛选股票
        
        Args:
            criteria (Dict): 筛选条件，默认为None使用基本筛选条件
        
        Returns:
            List[Dict]: 符合条件的股票列表及其分析结果
        """
        if criteria is None:
            criteria = {
                'min_price': 5,    # 最低价格
                'max_price': 100,  # 最高价格
                'min_volume': 1000000  # 最小成交量
            }
        
        # 获取股票列表
        stock_list = self.data_fetcher.fetch_stock_list()
        if stock_list is None:
            return []
        
        results = []
        for _, row in stock_list.iterrows():
            stock_code = row['code']
            analysis = self.analyze_stock(stock_code)
            
            if 'error' in analysis:
                continue
                
            # 应用筛选条件
            if (criteria['min_price'] <= analysis['price'] <= criteria['max_price']):
                results.append(analysis)
        
        # 按信号强度的绝对值排序，取前20个最强信号
        results.sort(key=lambda x: abs(x['strength']), reverse=True)
        return results[:20]
    
    def get_daily_report(self) -> Dict:
        """生成每日市场报告
        
        Returns:
            Dict: 包含市场概况和推荐股票的字典
        """
        # 获取大盘指数数据
        index_data = self.data_fetcher.fetch_index_data()
        
        # 筛选股票
        recommended_stocks = self.screen_stocks()
        
        # 准备市场概况数据
        market_summary = {}
        if index_data is not None and not index_data.empty:
            latest_index = index_data.iloc[-1]
            market_summary['index'] = {
                'close': float(latest_index['close']),
                'change_pct': float(latest_index['change_pct']),
                'volume': float(latest_index['volume']),
                'date': latest_index.name.strftime('%Y-%m-%d') if isinstance(latest_index.name, pd.Timestamp) else str(latest_index.name)
            }
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'market_summary': market_summary,
            'recommendations': {
                'buy': [stock for stock in recommended_stocks if stock['signal'] == 'BUY'],
                'sell': [stock for stock in recommended_stocks if stock['signal'] == 'SELL']
            }
        }
        
        return report 