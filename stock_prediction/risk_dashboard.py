import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .bearish-card {
        background-color: #ffe6e6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #ff4444;
    }
    .bullish-card {
        background-color: #e6ffe6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #44ff44;
    }
</style>
""", unsafe_allow_html=True)

# Stock lists
GROWING_STOCKS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
    'META', 'NFLX', 'NVDA', 'AMD', 'INTC',
    'BA', 'KO', 'PEP', 'WMT', 'JPM', 'V', 'MA',
    'UNH', 'HD', 'PG', 'JNJ', 'BAC', 'XOM', 'CVX'
]

BEARISH_STOCKS = [
    'PLTR', 'SNOW', 'ZM', 'PTON', 'ROKU', 'SNAP', 
    'SPOT', 'SQ', 'PYPL', 'ZG', 'BBY', 'TGT', 
    'COST', 'NKE', 'SBUX', 'WFC', 'C', 'GS',
    'SLB', 'HAL', 'MRO', 'PLD', 'EXR', 'AAL',
    'UAL', 'DAL', 'F', 'GM', 'DIS'
]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period='6mo'):
    """Fetch historical data with caching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(symbol):
    """Get basic stock information with caching"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'symbol': symbol,
            'current_price': info.get('currentPrice', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'short_name': info.get('shortName', symbol)
        }
    except:
        return {
            'symbol': symbol, 'current_price': 'N/A', 'market_cap': 'N/A', 
            'volume': 'N/A', 'pe_ratio': 'N/A', 'forward_pe': 'N/A', 
            'price_to_sales': 'N/A', 'debt_to_equity': 'N/A', 
            'beta': 'N/A', 'short_name': symbol
        }

def calculate_metrics(data):
    """Calculate all trading metrics"""
    if data.empty or len(data) < 2:
        return {}
    
    # Growth percentage
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    growth = ((end_price - start_price) / start_price) * 100
    
    # Volatility (annualized)
    data_copy = data.copy()
    data_copy['Daily Return'] = data_copy['Close'].pct_change()
    daily_volatility = np.std(data_copy['Daily Return'].dropna())
    annualized_volatility = daily_volatility * np.sqrt(252) * 100
    
    # Sharpe ratio
    total_return = (end_price / start_price) - 1
    days = len(data)
    annualized_return = (1 + total_return) ** (252 / days) - 1
    sharpe = (annualized_return - 0.02) / (daily_volatility * np.sqrt(252)) if daily_volatility > 0 else 0
    
    # Max drawdown
    data_copy['Cumulative Max'] = data_copy['Close'].expanding().max()
    data_copy['Drawdown'] = (data_copy['Close'] - data_copy['Cumulative Max']) / data_copy['Cumulative Max']
    max_drawdown = data_copy['Drawdown'].min() * 100
    
    # RSI
    delta = data_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
    
    return {
        'growth': round(growth, 2),
        'volatility': round(annualized_volatility, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown': round(max_drawdown, 2),
        'rsi': round(current_rsi, 2)
    }

def analyze_bearish_signals(info, metrics):
    """Analyze bearish signals"""
    signals = []
    signal_count = 0
    
    # High valuation signals
    if isinstance(info.get('pe_ratio'), (int, float)) and info['pe_ratio'] > 50:
        signals.append(f"High P/E: {info['pe_ratio']:.1f}")
        signal_count += 1
    
    if isinstance(info.get('price_to_sales'), (int, float)) and info['price_to_sales'] > 10:
        signals.append(f"High P/S: {info['price_to_sales']:.1f}")
        signal_count += 1
    
    # Technical signals
    if metrics.get('rsi', 50) > 70:
        signals.append(f"Overbought RSI: {metrics['rsi']:.1f}")
        signal_count += 1
    
    # Performance signals
    if metrics.get('growth', 0) < -20:
        signals.append(f"Declining: {metrics['growth']:.1f}%")
        signal_count += 1
    
    if metrics.get('max_drawdown', 0) < -30:
        signals.append(f"High Drawdown: {metrics['max_drawdown']:.1f}%")
        signal_count += 1
    
    return signals, signal_count

def create_price_chart(symbol, data, period):
    """Create interactive price chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA20'],
        mode='lines',
        name='MA20',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA50'],
        mode='lines',
        name='MA50',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{symbol} Price Chart ({period})',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        showlegend=True
    )
    
    return fig

def create_performance_chart(df, chart_type='growth'):
    """Create performance comparison chart"""
    if chart_type == 'growth':
        fig = px.bar(
            df.head(15), 
            x='symbol', 
            y='growth',
            title='Top Performing Stocks by Growth',
            color='growth',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
    else:
        fig = px.scatter(
            df, 
            x='volatility', 
            y='growth',
            size='market_cap',
            color='sharpe_ratio',
            hover_name='symbol',
            title='Risk-Return Analysis',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<p class="main-header">📈 Market Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Analysis Settings")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Growth Stocks", "Bearish Stocks", "Compare Both", "Individual Stock"]
    )
    
    period = st.sidebar.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )
    
    top_n = st.sidebar.slider("Top N Results", 5, 25, 15)
    
    # Main content based on selection
    if analysis_type == "Individual Stock":
        st.header("🔍 Individual Stock Analysis")
        
        # Stock symbol input
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL").upper()
        
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                data = get_stock_data(symbol, period)
                
                if not data.empty:
                    info = get_stock_info(symbol)
                    metrics = calculate_metrics(data)
                    signals, signal_count = analyze_bearish_signals(info, metrics)
                    
                    # Display metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Current Price", 
                            f"${info['current_price']:.2f}" if isinstance(info['current_price'], (int, float)) else "N/A"
                        )
                        st.metric("Growth", f"{metrics.get('growth', 0):.2f}%")
                    
                    with col2:
                        st.metric("Volatility", f"{metrics.get('volatility', 0):.2f}%")
                        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                        st.metric("RSI", f"{metrics.get('rsi', 50):.1f}")
                    
                    with col4:
                        st.metric("P/E Ratio", f"{info['pe_ratio']:.1f}" if isinstance(info['pe_ratio'], (int, float)) else "N/A")
                        st.metric("Beta", f"{info['beta']:.2f}" if isinstance(info['beta'], (int, float)) else "N/A")
                    
                    # Price chart
                    st.plotly_chart(create_price_chart(symbol, data, period), use_container_width=True)
                    
                    # Bearish signals if any
                    if signals:
                        st.warning(f"⚠️ Bearish Signals Detected ({signal_count})")
                        for signal in signals:
                            st.write(f"• {signal}")
                else:
                    st.error(f"No data found for {symbol}")
    
    elif analysis_type == "Growth Stocks":
        st.header("📈 Growth Stock Analysis")
        
        # Initialize session state for data
        if 'growth_data' not in st.session_state:
            st.session_state.growth_data = None
        
        if st.button("🔄 Analyze Growth Stocks") or st.session_state.growth_data is None:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(GROWING_STOCKS):
                status_text.text(f"Analyzing {symbol}... ({i+1}/{len(GROWING_STOCKS)})")
                progress_bar.progress((i + 1) / len(GROWING_STOCKS))
                
                data = get_stock_data(symbol, period)
                if not data.empty:
                    info = get_stock_info(symbol)
                    metrics = calculate_metrics(data)
                    
                    if metrics.get('growth') is not None:
                        result = {**info, **metrics}
                        results.append(result)
            
            status_text.text("Analysis complete!")
            progress_bar.empty()
            
            if results:
                st.session_state.growth_data = pd.DataFrame(results)
        
        if st.session_state.growth_data is not None and not st.session_state.growth_data.empty:
            df = st.session_state.growth_data.copy()
            df_sorted = df.sort_values('growth', ascending=False)
            
            # Display top performers
            st.subheader(f"🏆 Top {top_n} Growth Stocks")
            
            # Performance chart
            st.plotly_chart(create_performance_chart(df_sorted, 'growth'), use_container_width=True)
            
            # Detailed table
            display_cols = ['symbol', 'short_name', 'growth', 'current_price', 'volatility', 'sharpe_ratio', 'rsi']
            st.dataframe(
                df_sorted[display_cols].head(top_n),
                use_container_width=True,
                hide_index=True
            )
    
    elif analysis_type == "Bearish Stocks":
        st.header("📉 Bearish Stock Analysis")
        
        if 'bearish_data' not in st.session_state:
            st.session_state.bearish_data = None
        
        if st.button("🔄 Analyze Bearish Stocks") or st.session_state.bearish_data is None:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(BEARISH_STOCKS):
                status_text.text(f"Analyzing {symbol}... ({i+1}/{len(BEARISH_STOCKS)})")
                progress_bar.progress((i + 1) / len(BEARISH_STOCKS))
                
                data = get_stock_data(symbol, period)
                if not data.empty:
                    info = get_stock_info(symbol)
                    metrics = calculate_metrics(data)
                    signals, signal_count = analyze_bearish_signals(info, metrics)
                    
                    if metrics.get('growth') is not None:
                        result = {**info, **metrics, 'bearish_signals': signal_count, 'signal_details': ', '.join(signals)}
                        results.append(result)
            
            status_text.text("Analysis complete!")
            progress_bar.empty()
            
            if results:
                st.session_state.bearish_data = pd.DataFrame(results)
        
        if st.session_state.bearish_data is not None and not st.session_state.bearish_data.empty:
            df = st.session_state.bearish_data.copy()
            df_sorted = df.sort_values(['bearish_signals', 'growth'], ascending=[False, True])
            
            # Display most bearish stocks
            st.subheader(f"⚠️ Top {top_n} Potentially Declining Stocks")
            
            for _, row in df_sorted.head(top_n).iterrows():
                with st.expander(f"{row['symbol']} - {row['short_name']} ({row['bearish_signals']} signals)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Growth", f"{row['growth']:.2f}%")
                        st.metric("Current Price", f"${row['current_price']:.2f}" if isinstance(row['current_price'], (int, float)) else "N/A")
                    
                    with col2:
                        st.metric("RSI", f"{row['rsi']:.1f}")
                        st.metric("Max Drawdown", f"{row['max_drawdown']:.2f}%")
                    
                    with col3:
                        st.metric("P/E Ratio", f"{row['pe_ratio']:.1f}" if isinstance(row['pe_ratio'], (int, float)) else "N/A")
                        st.metric("Bearish Signals", row['bearish_signals'])
                    
                    if row['signal_details']:
                        st.write("**Signals:**", row['signal_details'])
    
    else:  # Compare Both
        st.header("⚖️ Growth vs Bearish Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Top Growth Opportunities")
            if st.button("Analyze Growth"):
                # Quick growth analysis
                growth_results = []
                for symbol in GROWING_STOCKS[:10]:  # Limit for comparison
                    data = get_stock_data(symbol, period)
                    if not data.empty:
                        metrics = calculate_metrics(data)
                        if metrics.get('growth'):
                            growth_results.append({'Symbol': symbol, 'Growth': metrics['growth']})
                
                if growth_results:
                    growth_df = pd.DataFrame(growth_results).sort_values('Growth', ascending=False)
                    st.dataframe(growth_df.head(10), hide_index=True)
        
        with col2:
            st.subheader("📉 Top Decline Risks")
            if st.button("Analyze Bearish"):
                # Quick bearish analysis
                bearish_results = []
                for symbol in BEARISH_STOCKS[:10]:  # Limit for comparison
                    data = get_stock_data(symbol, period)
                    if not data.empty:
                        info = get_stock_info(symbol)
                        metrics = calculate_metrics(data)
                        signals, signal_count = analyze_bearish_signals(info, metrics)
                        
                        if metrics.get('growth') is not None:
                            bearish_results.append({
                                'Symbol': symbol, 
                                'Growth': metrics['growth'], 
                                'Signals': signal_count
                            })
                
                if bearish_results:
                    bearish_df = pd.DataFrame(bearish_results).sort_values(['Signals', 'Growth'], ascending=[False, True])
                    st.dataframe(bearish_df.head(10), hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This dashboard is for educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()