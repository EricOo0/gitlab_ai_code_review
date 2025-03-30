import pandas as pd
import numpy as np
from typing import Dict, List, Union

class BaseStrategy:
    def __init__(self):
        """初始化基础策略类"""
        pass

    def calculate_ma(self, data: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """计算移动平均线
        
        Args:
            data (pd.DataFrame): 股票数据，必须包含'close'列
            periods (List[int]): MA周期列表
        """
        df = data.copy()
        for period in periods:
            df[f'MA{period}'] = df['close'].rolling(window=period).mean()
        return df

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算RSI指标
        
        Args:
            data (pd.DataFrame): 股票数据，必须包含'close'列
            period (int): RSI计算周期
        """
        df = data.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI{period}'] = 100 - (100 / (1 + rs))
        return df

    def calculate_macd(self, data: pd.DataFrame, 
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            data (pd.DataFrame): 股票数据，必须包含'close'列
            fast_period (int): 快线周期
            slow_period (int): 慢线周期
            signal_period (int): 信号线周期
        """
        df = data.copy()
        # 计算快线和慢线的EMA
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        df['MACD'] = ema_fast - ema_slow
        # 计算信号线
        df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        # 计算MACD柱状图
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        return df

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """生成交易信号
        
        Args:
            data (pd.DataFrame): 包含技术指标的股票数据
        
        Returns:
            Dict: 包含交易信号和建议的字典
        """
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        
        signals = {
            'date': latest.name,
            'price': latest['close'],
            'signal': 'HOLD',
            'strength': 0,
            'reason': []
        }

        # MA趋势强度（考虑价格偏离MA的程度）
        if 'close' in data.columns and 'MA20' in data.columns:
            ma_deviation = (latest['close'] - latest['MA20']) / latest['MA20'] * 100
            ma_score = ma_deviation * 2  # 每1%的偏离给予2分的权重
            signals['strength'] += ma_score
            if ma_score > 0:
                signals['reason'].append(f'价格高于20日均线 {abs(ma_deviation):.1f}%')
            else:
                signals['reason'].append(f'价格低于20日均线 {abs(ma_deviation):.1f}%')

        # RSI动量（考虑RSI的具体数值）
        if 'RSI14' in data.columns:
            rsi = latest['RSI14']
            if rsi < 30:
                rsi_score = (30 - rsi) * 0.5  # 每低于30一个点给予0.5分
                signals['strength'] += rsi_score
                signals['reason'].append(f'RSI超卖 ({rsi:.1f})')
            elif rsi > 70:
                rsi_score = (rsi - 70) * -0.5  # 每高于70一个点扣除0.5分
                signals['strength'] += rsi_score
                signals['reason'].append(f'RSI超买 ({rsi:.1f})')

        # MACD趋势强度（考虑MACD的变化率和柱状图高度）
        if all(x in data.columns for x in ['MACD', 'Signal', 'MACD_Hist']):
            hist = latest['MACD_Hist']
            hist_change = hist - prev['MACD_Hist']
            hist_score = hist * 5  # MACD柱状图的高度权重
            hist_momentum = hist_change * 3  # MACD柱状图变化的权重
            signals['strength'] += (hist_score + hist_momentum)
            
            if hist > 0:
                if hist_change > 0:
                    signals['reason'].append(f'MACD金叉增强 (柱高:{hist:.3f}, 动量:{hist_change:.3f})')
                else:
                    signals['reason'].append(f'MACD金叉减弱 (柱高:{hist:.3f}, 动量:{hist_change:.3f})')
            else:
                if hist_change < 0:
                    signals['reason'].append(f'MACD死叉增强 (柱高:{hist:.3f}, 动量:{hist_change:.3f})')
                else:
                    signals['reason'].append(f'MACD死叉减弱 (柱高:{hist:.3f}, 动量:{hist_change:.3f})')

        # 成交量分析
        if 'volume' in data.columns:
            # 计算20日平均成交量
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            vol_ratio = latest['volume'] / avg_volume
            vol_score = (vol_ratio - 1) * 2  # 高于平均成交量每100%给予2分
            signals['strength'] += vol_score
            if vol_ratio > 1:
                signals['reason'].append(f'成交量放大 {(vol_ratio-1)*100:.1f}%')
            else:
                signals['reason'].append(f'成交量萎缩 {(1-vol_ratio)*100:.1f}%')

        # 根据信号强度确定最终信号
        if signals['strength'] >= 5:
            signals['signal'] = 'BUY'
        elif signals['strength'] <= -5:
            signals['signal'] = 'SELL'

        signals['reason'] = '; '.join(signals['reason'])
        return signals 