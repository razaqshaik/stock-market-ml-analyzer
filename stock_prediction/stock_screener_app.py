import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Universe Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockUniverseScreener:
    def __init__(self, universe_type='sp100'):
        self.universe_type = universe_type
        self.stock_universe = self._get_stock_universe()
        self.results_df = pd.DataFrame()
        
    def _get_stock_universe(self) -> List[str]:
        """Get stock universe based on type"""
        if self.universe_type == 'sp100':
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
                'JPM', 'JNJ', 'V', 'PG', 'HD', 'MA', 'NFLX', 'DIS', 'PYPL', 'ADBE',
                'PFE', 'KO', 'ORCL', 'CRM', 'XOM', 'TMO', 'ABT', 'NKE',
                'INTC', 'AMD', 'CSCO', 'ACN', 'TXN', 'QCOM', 'DHR', 'UNP', 'NEE',
                'PM', 'RTX', 'INTU', 'IBM', 'AMGN', 'WMT', 'HON', 'BA', 'SPGI',
                'CAT', 'GS', 'LOW', 'SBUX', 'AXP', 'BLK', 'GILD', 'MDT', 'ISRG',
                'CVX', 'BKNG', 'MU', 'NOW', 'ZTS', 'DE', 'SYK', 'MMM', 'TJX',
                'PLD', 'C', 'SCHW', 'CB', 'USB', 'TMUS', 'FIS', 'CME', 'MO',
                'SO', 'DUK', 'CL', 'MDLZ', 'CI', 'NSC', 'WM', 'EQIX', 'AON',
                'ITW', 'COP', 'CSX', 'MMC', 'PNC', 'GM', 'F', 'ATVI', 'EMR',
                'GD', 'APD', 'SHW', 'CMI', 'NOC', 'FCX', 'ECL', 'ADI', 'EOG'
            ]
        elif self.universe_type == 'nasdaq100':
            return [
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
                'NFLX', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC', 'AMD', 'QCOM',
                'TXN', 'MU', 'AVGO', 'INTU', 'ISRG', 'COST', 'SBUX', 'GILD',
                'AMGN', 'BKNG', 'MDLZ', 'ATVI', 'EA', 'WBA', 'REGN', 'ILMN',
                'ADI', 'LRCX', 'KLAC', 'AMAT', 'SNPS', 'CDNS',
                'CHTR', 'CMCSA', 'NXPI', 'MCHP', 'PAYX', 'FAST', 'VRTX'
            ]
        elif self.universe_type == 'custom':
            return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        else:
            return self._get_stock_universe()

    @st.cache_data
    def get_stock_data(_self, symbol: str, period: str = '6mo') -> pd.DataFrame:
        """Fetch stock data with caching"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                return pd.DataFrame()
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    @st.cache_data
    def get_stock_info(_self, symbol: str) -> Dict:
        """Fetch stock info with caching"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            return {}

    def calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate key technical and fundamental metrics"""
        if data.empty:
            return {}
        
        try:
            current_price = data['Close'].iloc[-1]
            start_price = data['Close'].iloc[0]
            growth = ((current_price - start_price) / start_price) * 100
            
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            risk_free_rate = 0.02
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            rsi = self._calculate_rsi(data['Close'])
            
            ma_20 = data['Close'].rolling(20).mean().iloc[-1]
            ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else ma_20
            
            avg_volume = data['Volume'].mean()
            recent_volume = data['Volume'].tail(5).mean()
            volume_trend = ((recent_volume - avg_volume) / avg_volume) * 100
            
            return {
                'current_price': current_price,
                'growth': growth,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'rsi': rsi,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'volume_trend': volume_trend,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            return {}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50

    def analyze_bearish_signals(self, info: Dict, metrics: Dict) -> Tuple[List[str], int]:
        """Analyze bearish signals"""
        signals = []
        
        if not metrics:
            return signals, 0
        
        if metrics.get('rsi', 50) > 70:
            signals.append('RSI Overbought')
        
        if metrics.get('rsi', 50) < 30 and metrics.get('growth', 0) < -10:
            signals.append('RSI Oversold + Downtrend')
        
        if metrics.get('volatility', 0) > 40:
            signals.append('High Volatility')
        
        if metrics.get('growth', 0) < -15:
            signals.append('Significant Decline')
        
        current_price = metrics.get('current_price', 0)
        ma_20 = metrics.get('ma_20', 0)
        ma_50 = metrics.get('ma_50', 0)
        
        if current_price < ma_20 and ma_20 < ma_50:
            signals.append('Below Moving Averages')
        
        if metrics.get('sharpe_ratio', 0) < -0.5:
            signals.append('Poor Risk-Adjusted Returns')
        
        pe_ratio = info.get('trailingPE', 0)
        if pe_ratio and pe_ratio > 50:
            signals.append('High P/E Ratio')
        
        return signals, len(signals)

    def classify_stock(self, metrics: Dict, signal_count: int, info: Dict) -> str:
        """Classify stock based on metrics"""
        if not metrics:
            return 'No Data'
        
        growth = metrics.get('growth', 0)
        volatility = metrics.get('volatility', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        rsi = metrics.get('rsi', 50)
        
        if (growth > 10 and signal_count <= 1 and sharpe_ratio > 0.5 and volatility < 30):
            return 'Strong Buy'
        elif (growth > 5 and signal_count == 0 and 30 <= rsi <= 70):
            return 'Buy'
        elif (growth > 0 and signal_count <= 1 and sharpe_ratio > 0):
            return 'Hold'
        elif (growth < -15 or signal_count >= 3 or sharpe_ratio < -0.5):
            return 'Strong Sell'
        elif (growth < -5 or signal_count >= 2):
            return 'Sell'
        else:
            return 'Neutral'

    def screen_universe(self, period: str = '6mo', progress_bar=None) -> pd.DataFrame:
        """Screen universe with progress tracking"""
        results = []
        total_stocks = len(self.stock_universe)
        
        for i, symbol in enumerate(self.stock_universe):
            if progress_bar:
                progress_bar.progress((i + 1) / total_stocks, text=f"Processing {symbol}...")
            
            try:
                data = self.get_stock_data(symbol, period)
                if data.empty:
                    continue
                
                info = self.get_stock_info(symbol)
                metrics = self.calculate_metrics(data)
                signals, signal_count = self.analyze_bearish_signals(info, metrics)
                classification = self.classify_stock(metrics, signal_count, info)
                
                result = {
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'sector': info.get('sector', 'Unknown'),
                    'current_price': metrics.get('current_price', 0),
                    'growth': metrics.get('growth', 0),
                    'volatility': metrics.get('volatility', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'rsi': metrics.get('rsi', 50),
                    'volume_trend': metrics.get('volume_trend', 0),
                    'signals': signal_count,
                    'bearish_signals': ', '.join(signals) if signals else 'None',
                    'classification': classification,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', None)
                }
                
                results.append(result)
                
            except Exception as e:
                continue
        
        self.results_df = pd.DataFrame(results)
        return self.results_df

# Initialize session state
if 'screener' not in st.session_state:
    st.session_state.screener = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# Main app
def main():
    st.title("📈 Stock Universe Screener Dashboard")
    st.markdown("Analyze stock universes with technical and fundamental indicators")

    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Configuration")
        
        universe_type = st.selectbox(
            "Select Universe",
            ['sp100', 'nasdaq100', 'custom'],
            format_func=lambda x: {
                'sp100': 'S&P 100',
                'nasdaq100': 'NASDAQ 100',
                'custom': 'Custom Watchlist'
            }[x]
        )
        
        period = st.selectbox(
            "Analysis Period",
            ['1mo', '3mo', '6mo', '1y', '2y'],
            index=2
        )
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Initializing screener..."):
                st.session_state.screener = StockUniverseScreener(universe_type)
            
            progress_bar = st.progress(0, text="Starting analysis...")
            
            with st.spinner("Screening stocks..."):
                results_df = st.session_state.screener.screen_universe(period, progress_bar)
                st.session_state.results_df = results_df
            
            progress_bar.empty()
            st.success(f"Analysis complete! Processed {len(results_df)} stocks.")
    
    # Main content
    if not st.session_state.results_df.empty:
        df = st.session_state.results_df
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Stocks", len(df))
        with col2:
            bullish_count = len(df[df['classification'].isin(['Strong Buy', 'Buy'])])
            st.metric("Bullish", bullish_count, f"{bullish_count/len(df)*100:.1f}%")
        with col3:
            bearish_count = len(df[df['classification'].isin(['Strong Sell', 'Sell'])])
            st.metric("Bearish", bearish_count, f"{bearish_count/len(df)*100:.1f}%")
        with col4:
            avg_growth = df['growth'].mean()
            st.metric("Avg Growth", f"{avg_growth:.1f}%")
        with col5:
            avg_volatility = df['volatility'].mean()
            st.metric("Avg Volatility", f"{avg_volatility:.1f}%")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🟢 Bullish", "🔴 Bearish", "📈 Charts", "📋 Data"])
        
        with tab1:
            st.subheader("Portfolio Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Classification pie chart
                classification_counts = df['classification'].value_counts()
                fig_pie = px.pie(
                    values=classification_counts.values,
                    names=classification_counts.index,
                    title="Stock Classifications Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Sector performance
                sector_performance = df.groupby('sector')['growth'].mean().sort_values(ascending=False)
                fig_bar = px.bar(
                    x=sector_performance.values,
                    y=sector_performance.index,
                    orientation='h',
                    title="Average Growth by Sector",
                    labels={'x': 'Average Growth (%)', 'y': 'Sector'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Risk-Return scatter
            fig_scatter = px.scatter(
                df, x='volatility', y='growth',
                color='classification',
                size='market_cap',
                hover_data=['symbol', 'company_name'],
                title="Risk-Return Profile",
                labels={'volatility': 'Volatility (%)', 'growth': 'Growth (%)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab2:
            st.subheader("🟢 Bullish Opportunities")
            bullish_df = df[df['classification'].isin(['Strong Buy', 'Buy'])].sort_values('growth', ascending=False)
            
            if not bullish_df.empty:
                st.dataframe(
                    bullish_df[['symbol', 'company_name', 'current_price', 'growth', 'volatility', 'rsi', 'classification']],
                    use_container_width=True
                )
                
                # Top performers chart
                top_10 = bullish_df.head(10)
                fig_bullish = px.bar(
                    top_10, x='growth', y='symbol',
                    orientation='h',
                    color='classification',
                    title="Top 10 Bullish Stocks by Growth"
                )
                st.plotly_chart(fig_bullish, use_container_width=True)
            else:
                st.info("No bullish stocks found in current analysis.")
        
        with tab3:
            st.subheader("🔴 Bearish Risks")
            bearish_df = df[df['classification'].isin(['Strong Sell', 'Sell'])].sort_values('signals', ascending=False)
            
            if not bearish_df.empty:
                st.dataframe(
                    bearish_df[['symbol', 'company_name', 'current_price', 'growth', 'signals', 'bearish_signals', 'classification']],
                    use_container_width=True
                )
                
                # Worst performers chart
                worst_10 = bearish_df.head(10)
                fig_bearish = px.bar(
                    worst_10, x='growth', y='symbol',
                    orientation='h',
                    color='signals',
                    title="Top 10 Bearish Stocks by Signal Count"
                )
                st.plotly_chart(fig_bearish, use_container_width=True)
            else:
                st.info("No bearish stocks found in current analysis.")
        
        with tab4:
            st.subheader("📈 Technical Analysis Charts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI distribution
                fig_rsi = px.histogram(
                    df, x='rsi', nbins=20,
                    title="RSI Distribution"
                )
                fig_rsi.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="Oversold")
                fig_rsi.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # Sharpe ratio distribution
                fig_sharpe = px.histogram(
                    df, x='sharpe_ratio', nbins=20,
                    title="Sharpe Ratio Distribution"
                )
                fig_sharpe.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")
                st.plotly_chart(fig_sharpe, use_container_width=True)
            
            # Signal analysis
            signal_analysis = df.groupby('signals').size()
            fig_signals = px.bar(
                x=signal_analysis.index, y=signal_analysis.values,
                title="Distribution of Bearish Signals",
                labels={'x': 'Number of Signals', 'y': 'Number of Stocks'}
            )
            st.plotly_chart(fig_signals, use_container_width=True)
        
        with tab5:
            st.subheader("📋 Complete Dataset")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                classification_filter = st.multiselect(
                    "Filter by Classification",
                    options=df['classification'].unique(),
                    default=df['classification'].unique()
                )
            
            with col2:
                sector_filter = st.multiselect(
                    "Filter by Sector",
                    options=df['sector'].unique(),
                    default=df['sector'].unique()
                )
            
            with col3:
                growth_range = st.slider(
                    "Growth Range (%)",
                    float(df['growth'].min()),
                    float(df['growth'].max()),
                    (float(df['growth'].min()), float(df['growth'].max()))
                )
            
            # Apply filters
            filtered_df = df[
                (df['classification'].isin(classification_filter)) &
                (df['sector'].isin(sector_filter)) &
                (df['growth'] >= growth_range[0]) &
                (df['growth'] <= growth_range[1])
            ]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f'stock_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
    
    else:
        st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis' to get started!")

if __name__ == "__main__":
    main()