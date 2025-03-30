import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

class StockDataFetcher:
    def __init__(self):
        """初始化数据获取器"""
        load_dotenv()
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_stock_daily(self, stock_code: str, start_date: str = None, end_date: str = None):
        """获取单个股票的每日行情数据
        
        Args:
            stock_code (str): 股票代码（如：000001）
            start_date (str): 开始日期，格式：YYYYMMDD
            end_date (str): 结束日期，格式：YYYYMMDD
        """
        try:
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            # 使用akshare获取A股历史行情数据
            df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                  start_date=start_date, end_date=end_date, adjust="qfq")
            
            # 重命名列以匹配我们的代码
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change_amount',
                '换手率': 'turnover'
            }
            df = df.rename(columns=column_mapping)
            
            # 确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 确保数值列是float类型
            numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 
                             'amplitude', 'change_pct', 'change_amount', 'turnover']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 保存数据到本地
            file_path = os.path.join(self.data_dir, f'{stock_code}_daily.csv')
            df.to_csv(file_path)
            return df
        except Exception as e:
            print(f"获取股票{stock_code}数据时发生错误: {str(e)}")
            return None

    def fetch_index_data(self, index_code: str = 'sh000001'):
        """获取指数数据
        
        Args:
            index_code (str): 指数代码，默认为上证指数（sh000001）
        """
        try:
            df = ak.stock_zh_index_daily(symbol=index_code)
            
            # 计算涨跌幅
            df['change_pct'] = df['close'].pct_change() * 100
            df['change_amount'] = df['close'] - df['close'].shift(1)
            
            # 确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 确保数值列是float类型
            numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'change_pct', 'change_amount']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            file_path = os.path.join(self.data_dir, f'index_{index_code}.csv')
            df.to_csv(file_path)
            return df
        except Exception as e:
            print(f"获取指数{index_code}数据时发生错误: {str(e)}")
            return None

    def fetch_stock_list(self):
        """获取A股所有股票列表"""
        try:
            stock_list = ak.stock_info_a_code_name()
            file_path = os.path.join(self.data_dir, 'stock_list.csv')
            stock_list.to_csv(file_path, index=False)
            return stock_list
        except Exception as e:
            print(f"获取股票列表时发生错误: {str(e)}")
            return None 