# -*- coding: utf-8 -*-
"""
Interactive Stock Market Analysis Dashboard
Run with: streamlit run stock_dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
# At the top of stock_dashboard.py
from stock_analysis import get_stock_data, get_growth_percentage, get_volatility
# Then use your existing functions in the dashboard
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > label {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Default stock lists
DEFAULT_STOCKS = [
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



SECTOR_STOCKS = {
    "Technology": [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
        'META', 'NFLX', 'NVDA', 'AMD', 'INTC',
        'ADBE', 'CSCO', 'ORCL', 'CRM', 'IBM'
    ],
    "Finance": [
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        'V', 'MA', 'AXP'
    ],
    "Healthcare": [
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK'
    ],
    "Consumer": [
        'WMT', 'HD', 'PG', 'KO', 'PEP',
        'NKE', 'MCD', 'COST', 'DIS'
    ],
    "Industrial": [
        'BA', 'CAT', 'GE', 'MMM', 'UPS', 'FDX'
    ],
    "Energy": [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG'
    ]
}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period='6mo'):
    """Fetch historical data with caching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

@st.cache_data(ttl=300)
def get_stock_info(symbol):
    """Get basic stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('longName', symbol),
            'current_price': info.get('currentPrice', 0),
            'market_cap': info.get('marketCap', 0),
            'volume': info.get('volume', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'sector': info.get('sector', 'Unknown')
        }
    except:
        return {
            'name': symbol,
            'current_price': 0,
            'market_cap': 0,
            'volume': 0,
            'pe_ratio': 0,
            'sector': 'Unknown'
        }

def calculate_metrics(data):
    """Calculate comprehensive stock metrics"""
    if data is None or data.empty or len(data) < 2:
        return None
    
    # Growth calculation
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    growth = ((end_price - start_price) / start_price) * 100
    
    # Daily returns
    data = data.copy()
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Volatility (annualized)
    daily_volatility = np.std(data['Daily_Return'].dropna())
    volatility = daily_volatility * np.sqrt(252) * 100
    
    # Sharpe ratio
    total_return = (end_price / start_price) - 1
    days = len(data)
    annualized_return = (1 + total_return) ** (252 / days) - 1
    risk_free_rate = 0.02
    sharpe = (annualized_return - risk_free_rate) / (daily_volatility * np.sqrt(252)) if daily_volatility > 0 else 0
    
    # Maximum drawdown
    rolling_max = data['Close'].expanding().max()
    drawdown = (data['Close'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Moving averages
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    return {
        'growth': round(growth, 2),
        'volatility': round(volatility, 2),
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown': round(max_drawdown, 2),
        'current_price': round(end_price, 2),
        'data_with_ma': data
    }

def create_price_chart(data, symbol):
    """Create interactive price chart with moving averages"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Price Chart', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'MA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MA_20'],
                mode='lines',
                name='MA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'MA_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MA_50'],
                mode='lines',
                name='MA 50',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} Stock Analysis',
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    st.title("📈 Interactive Stock Market Dashboard")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Dashboard Settings")
    
    # Analysis period
    period_options = {
        '1 Month': '1mo',
        '3 Months': '3mo', 
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y'
    }
    selected_period = st.sidebar.selectbox(
        "📅 Analysis Period",
        options=list(period_options.keys()),
        index=2
    )
    period = period_options[selected_period]
    
    # Stock selection method
    selection_method = st.sidebar.radio(
        "🎯 Stock Selection",
        ["Default Portfolio", "By Sector", "Custom Symbols"]
    )
    
    if selection_method == "Default Portfolio":
        selected_stocks = DEFAULT_STOCKS
    elif selection_method == "By Sector":
        selected_sector = st.sidebar.selectbox(
            "Choose Sector",
            options=list(SECTOR_STOCKS.keys())
        )
        selected_stocks = SECTOR_STOCKS[selected_sector]
    else:
        custom_input = st.sidebar.text_area(
            "Enter stock symbols (comma-separated)",
            value="AAPL,GOOGL,MSFT,TSLA,NVDA",
            help="Example: AAPL,GOOGL,MSFT"
        )
        selected_stocks = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
    
    # Number of top stocks to display
    top_n = st.sidebar.slider("📊 Top N Stocks", 5, 20, 10)
    
    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Store results
        results = []
        
        # Analyze each stock
        for i, symbol in enumerate(selected_stocks):
            status_text.text(f'Analyzing {symbol}... ({i+1}/{len(selected_stocks)})')
            progress_bar.progress((i + 1) / len(selected_stocks))
            
            # Get data and calculate metrics
            data = get_stock_data(symbol, period)
            if data is not None:
                metrics = calculate_metrics(data)
                if metrics:
                    info = get_stock_info(symbol)
                    results.append({
                        'Symbol': symbol,
                        'Name': info['name'][:30] + '...' if len(info['name']) > 30 else info['name'],
                        'Current Price': metrics['current_price'],
                        'Growth (%)': metrics['growth'],
                        'Volatility (%)': metrics['volatility'],
                        'Sharpe Ratio': metrics['sharpe_ratio'],
                        'Max Drawdown (%)': metrics['max_drawdown'],
                        'Market Cap': info['market_cap'],
                        'Sector': info['sector']
                    })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if not results:
            st.error("❌ No valid data found for the selected stocks!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Display summary metrics
        st.header("📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks Analyzed", len(df))
        with col2:
            avg_growth = df['Growth (%)'].mean()
            st.metric("Average Growth", f"{avg_growth:.2f}%")
        with col3:
            avg_volatility = df['Volatility (%)'].mean()
            st.metric("Average Volatility", f"{avg_volatility:.2f}%")
        with col4:
            avg_sharpe = df['Sharpe Ratio'].mean()
            st.metric("Average Sharpe Ratio", f"{avg_sharpe:.2f}")
        
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Top Performers", "⚡ Most Volatile", "🎯 Risk-Adjusted", "📋 Full Data", "📈 Individual Analysis"
        ])
        
        with tab1:
            st.subheader(f"🏆 Top {min(top_n, len(df))} Growing Stocks")
            top_growth = df.nlargest(top_n, 'Growth (%)')[['Symbol', 'Name', 'Current Price', 'Growth (%)', 'Sharpe Ratio']]
            st.dataframe(top_growth, use_container_width=True)
            
            # Growth chart
            fig = px.bar(
                top_growth, 
                x='Symbol', 
                y='Growth (%)',
                title=f'Top {min(top_n, len(df))} Stock Growth',
                color='Growth (%)',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader(f"⚡ Top {min(top_n, len(df))} Volatile Stocks")
            top_volatile = df.nlargest(top_n, 'Volatility (%)')[['Symbol', 'Name', 'Volatility (%)', 'Growth (%)', 'Max Drawdown (%)']]
            st.dataframe(top_volatile, use_container_width=True)
            
            # Volatility vs Growth scatter
            fig = px.scatter(
                df, 
                x='Volatility (%)', 
                y='Growth (%)',
                size='Market Cap',
                hover_data=['Symbol', 'Name'],
                title='Growth vs Volatility Analysis',
                color='Sharpe Ratio',
                color_continuous_scale='RdYlBu'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader(f"🎯 Top {min(top_n, len(df))} Risk-Adjusted Performers")
            top_sharpe = df.nlargest(top_n, 'Sharpe Ratio')[['Symbol', 'Name', 'Sharpe Ratio', 'Growth (%)', 'Volatility (%)']]
            st.dataframe(top_sharpe, use_container_width=True)
            
            # Sharpe ratio chart
            fig = px.bar(
                top_sharpe, 
                x='Symbol', 
                y='Sharpe Ratio',
                title=f'Top {min(top_n, len(df))} Sharpe Ratios',
                color='Sharpe Ratio',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("📋 Complete Analysis Results")
            # Format the dataframe for better display
            display_df = df.copy()
            for col in ['Growth (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']:
                display_df[col] = display_df[col].round(2)
            
            st.dataframe(
                display_df.sort_values('Growth (%)', ascending=False),
                use_container_width=True
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f'stock_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with tab5:
            st.subheader("📈 Individual Stock Analysis")
            
            # Stock selector
            stock_to_analyze = st.selectbox(
                "Choose a stock for detailed analysis:",
                options=df['Symbol'].tolist()
            )
            
            if stock_to_analyze:
                # Get detailed data
                detailed_data = get_stock_data(stock_to_analyze, period)
                if detailed_data is not None:
                    detailed_metrics = calculate_metrics(detailed_data)
                    if detailed_metrics and 'data_with_ma' in detailed_metrics:
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${detailed_metrics['current_price']}")
                        with col2:
                            st.metric("Growth", f"{detailed_metrics['growth']}%")
                        with col3:
                            st.metric("Volatility", f"{detailed_metrics['volatility']}%")
                        with col4:
                            st.metric("Sharpe Ratio", f"{detailed_metrics['sharpe_ratio']}")
                        
                        # Price chart
                        chart = create_price_chart(detailed_metrics['data_with_ma'], stock_to_analyze)
                        st.plotly_chart(chart, use_container_width=True)
    
    else:
        st.info("👆 Configure your settings in the sidebar and click 'Run Analysis' to get started!")
        
        # Show sample of available stocks
        st.subheader("📋 Available Stock Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Default Portfolio:**")
            st.write(", ".join(DEFAULT_STOCKS[:12]))
            st.write("*...and more*")
        
        with col2:
            st.write("**Available Sectors:**")
            for sector, stocks in SECTOR_STOCKS.items():
                st.write(f"• **{sector}**: {len(stocks)} stocks")

if __name__ == "__main__":
    main()