from agents.stock_agent import StockAgent
import json
from datetime import datetime

def main():
    # 初始化股票交易代理
    agent = StockAgent()
    analysis = agent.analyze_stock("002230")
    print(analysis)
    analysis = agent.analyze_stock("001696")
    print(analysis)
    analysis = agent.analyze_stock("600030")
    print(analysis)
    analysis = agent.analyze_stock("510630")
    print(analysis) 
    # 获取每日报告
    print("正在生成每日市场报告...")
    daily_report = agent.get_daily_report()
    
    # 保存报告到文件
    filename = f"market_report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(f"data/{filename}", 'w', encoding='utf-8') as f:
        json.dump(daily_report, f, ensure_ascii=False, indent=2)
    
    # 打印报告摘要
    print(f"\n=== 市场报告 ({daily_report['date']}) ===")
    
    # 打印大盘信息
    if daily_report.get('market_summary', {}).get('index'):
        index_data = daily_report['market_summary']['index']
        print("\n大盘信息:")
        print(f"上证指数: {index_data.get('close', 'N/A')}")
        print(f"涨跌幅: {index_data.get('change_pct', 'N/A')}%")
    else:
        print("\n无法获取大盘信息")
    
    # 打印推荐股票
    buy_recommendations = daily_report['recommendations']['buy']
    if buy_recommendations:
        print("\n买入推荐:")
        for stock in buy_recommendations[:5]:  # 只显示前5个推荐
            print(f"股票代码: {stock['stock_code']}")
            print(f"当前价格: {stock['price']:.2f}")
            print(f"信号强度: {stock['strength']:.2f}")
            print(f"原因: {stock['reason']}")
            print("---")
    else:
        print("\n当前无买入推荐")
    
    sell_recommendations = daily_report['recommendations']['sell']
    if sell_recommendations:
        print("\n卖出推荐:")
        for stock in sell_recommendations[:5]:  # 只显示前5个推荐
            print(f"股票代码: {stock['stock_code']}")
            print(f"当前价格: {stock['price']:.2f}")
            print(f"信号强度: {stock['strength']:.2f}")
            print(f"原因: {stock['reason']}")
            print("---")
    else:
        print("\n当前无卖出推荐")
    
    print(f"\n完整报告已保存到: data/{filename}")

if __name__ == "__main__":
    main() 