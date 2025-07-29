# -*- coding: utf-8 -*-
"""
Enhanced Market Analysis Script
- Find growing stocks with comprehensive metrics
- Top N stocks by recent growth
- Identify volatile stocks with risk metrics
- Better error handling and data validation
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Expanded stock list
STOCK_LIST = [
 # Technology (15)
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
    'META', 'NFLX', 'NVDA', 'AMD', 'INTC',
    'ADBE', 'CSCO', 'ORCL', 'CRM', 'IBM',

    # Finance (8)
    'JPM', 'BAC', 'WFC', 'GS', 'MS',
    'V', 'MA', 'AXP',

    # Healthcare (7)
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK',

    # Consumer (9)
    'WMT', 'HD', 'PG', 'KO', 'PEP',
    'NKE', 'MCD', 'COST', 'DIS',

    # Industrial (6)
    'BA', 'CAT', 'GE', 'MMM', 'UPS', 'FDX',

    # Energy (5)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG'
]

def get_stock_data(symbol, period='6mo'):
    """Fetch historical data with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            print(f"⚠️  No data found for {symbol}")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"❌ Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()

def get_stock_info(symbol):
    """Get basic stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'current_price': info.get('currentPrice', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'volume': info.get('volume', 'N/A')
        }
    except:
        return {'current_price': 'N/A', 'market_cap': 'N/A', 'volume': 'N/A'}

def get_growth_percentage(data):
    """Calculate percentage growth over the period"""
    if data.empty or len(data) < 2:
        return None
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    growth = ((end_price - start_price) / start_price) * 100
    return round(growth, 2)

def get_volatility(data):
    """Calculate standard deviation of daily returns (annualized)"""
    if data.empty or len(data) < 2:
        return None
    data = data.copy()
    data['Daily Return'] = data['Close'].pct_change()
    daily_volatility = np.std(data['Daily Return'].dropna())
    # Annualize volatility (assuming 252 trading days)
    annualized_volatility = daily_volatility * np.sqrt(252) * 100
    return round(annualized_volatility, 2)

def get_sharpe_ratio(data, risk_free_rate=0.02):
    """Calculate Sharpe ratio (risk-adjusted return)"""
    if data.empty or len(data) < 2:
        return None
    data = data.copy()
    data['Daily Return'] = data['Close'].pct_change()
    
    # Calculate annualized return
    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
    days = len(data)
    annualized_return = (1 + total_return) ** (252 / days) - 1
    
    # Calculate annualized volatility
    daily_volatility = np.std(data['Daily Return'].dropna())
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    if annualized_volatility == 0:
        return None
    
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    return round(sharpe, 2)

def analyze_market(stocks, top_n=10, period='6mo'):
    """Comprehensive market analysis"""
    results = []
    
    print(f"🔍 Analyzing {len(stocks)} stocks over {period} period...")
    print("=" * 60)
    
    for i, symbol in enumerate(stocks, 1):
        print(f"Processing {symbol} ({i}/{len(stocks)})...", end=' ')
        
        try:
            data = get_stock_data(symbol, period)
            if data.empty:
                print("❌")
                continue
                
            info = get_stock_info(symbol)
            growth = get_growth_percentage(data)
            volatility = get_volatility(data)
            sharpe = get_sharpe_ratio(data)
            
            if all(x is not None for x in [growth, volatility, sharpe]):
                results.append({
                    'symbol': symbol,
                    'growth': growth,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'current_price': info['current_price']
                })
                print("✅")
            else:
                print("⚠️")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not results:
        print("No valid data found for analysis!")
        return [], [], []
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Sort by different metrics
    top_growing = df.nlargest(top_n, 'growth')
    top_volatile = df.nlargest(top_n, 'volatility')
    best_risk_adjusted = df.nlargest(top_n, 'sharpe_ratio')
    
    # Display results
    print(f"\n📈 TOP {top_n} GROWING STOCKS:")
    print("-" * 50)
    print(f"{'Symbol':<8} {'Growth':<10} {'Price':<12} {'Sharpe':<8}")
    print("-" * 50)
    for _, row in top_growing.iterrows():
        price = f"${row['current_price']:.2f}" if isinstance(row['current_price'], (int, float)) else "N/A"
        print(f"{row['symbol']:<8} {row['growth']:>6.2f}%   {price:<12} {row['sharpe_ratio']:>6.2f}")
    
    print(f"\n⚡ TOP {top_n} VOLATILE STOCKS:")
    print("-" * 50)
    print(f"{'Symbol':<8} {'Volatility':<12} {'Growth':<10} {'Sharpe':<8}")
    print("-" * 50)
    for _, row in top_volatile.iterrows():
        print(f"{row['symbol']:<8} {row['volatility']:>8.2f}%   {row['growth']:>6.2f}%   {row['sharpe_ratio']:>6.2f}")
    
    print(f"\n🎯 TOP {top_n} RISK-ADJUSTED PERFORMERS:")
    print("-" * 50)
    print(f"{'Symbol':<8} {'Sharpe':<8} {'Growth':<10} {'Volatility':<12}")
    print("-" * 50)
    for _, row in best_risk_adjusted.iterrows():
        print(f"{row['symbol']:<8} {row['sharpe_ratio']:>6.2f}   {row['growth']:>6.2f}%   {row['volatility']:>8.2f}%")
    
    return top_growing, top_volatile, best_risk_adjusted

def main():
    """Main execution function"""
    print("🚀 Enhanced Stock Market Analysis")
    print("=" * 40)
    
    # You can customize these parameters
    TOP_N = 10
    PERIOD = '6mo'  # Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    try:
        top_growing, top_volatile, best_risk_adjusted = analyze_market(
            STOCK_LIST, top_n=TOP_N, period=PERIOD
        )
        
        print(f"\n✅ Analysis complete! Processed {len(STOCK_LIST)} stocks.")
        print(f"📊 Period: {PERIOD} | Top N: {TOP_N}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")

if __name__ == '__main__':
    main()