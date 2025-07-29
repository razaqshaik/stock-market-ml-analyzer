# 📈 Stock Market ML Analyzer

A comprehensive stock market analysis and prediction system that combines interactive dashboards with machine learning models to provide data-driven insights for traders and investors.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Flask](https://img.shields.io/badge/flask-2.3+-green.svg)

## 🌟 Features

### 📊 Interactive Stock Dashboard
- **Real-time analysis** of 50+ popular stocks across 6 major sectors
- **Portfolio-wide metrics**: Growth %, Volatility, Sharpe Ratio, Max Drawdown
- **Multiple analysis modes**: Default portfolio, sector-wise, custom selection
- **Interactive visualizations**: Candlestick charts, moving averages, volume analysis
- **Risk assessment**: Top performers vs. high-risk stock identification

### 🔍 Bearish Stock Detection
- **Automated screening** for potentially declining stocks
- **Multi-signal analysis**: Valuation metrics, technical indicators, performance metrics
- **Risk scoring system** with detailed signal breakdowns
- **Early warning system** for portfolio risk management

### 🤖 ML-Powered Price Prediction
- **Four prediction models**: Linear Regression, Random Forest, ARIMA, LSTM
- **30+ technical features**: RSI, MACD, Bollinger Bands, Volume analysis
- **Smart recommendation system**: BUY/SELL/HOLD with confidence levels
- **Model consensus scoring** for prediction reliability

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Flask API      │    │   Yahoo Finance │
│   Dashboard     │    │  (ML Predictions)│    │      API        │
│                 │    │                  │    │                 │
│ • Portfolio     │◄──►│ • Linear Reg     │◄──►│ • Real-time     │
│ • Risk Analysis │    │ • Random Forest  │    │ • Historical    │
│ • Visualizations│    │ • ARIMA          │    │ • Stock Info    │
│                 │    │ • LSTM           │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip
Git
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-market-ml-analyzer.git
cd stock-market-ml-analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Applications

#### 🔥 Interactive Dashboard (Streamlit)
```bash
streamlit run stock_dashboard.py
```
Access at: `http://localhost:8501`


#### 🔥 ML Prediction API (Flask)
```bash
python app.py
```
Access at: `http://localhost:5000`

## 📋 Requirements

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
flask>=2.3.0
flask-cors>=4.0.0
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
warnings
datetime
math
io
base64
os

# Optional dependencies for advanced features
statsmodels>=0.14.0  # For ARIMA predictions
tensorflow>=2.13.0   # For LSTM predictions
```

## 🎯 Usage Examples

### Dashboard Analysis
1. **Launch the Streamlit dashboard**
2. **Select analysis type**: Default Portfolio, By Sector, or Custom Symbols
3. **Choose time period**: 1 month to 2 years
4. **Run analysis** to get comprehensive metrics
5. **Explore tabs**: Top Performers, Most Volatile, Risk-Adjusted, Individual Analysis

### ML Predictions
```python
import requests

# API endpoint
url = "http://localhost:5000/api/predict"

# Request prediction
data = {"symbol": "AAPL"}
response = requests.post(url, json=data)
result = response.json()

print(f"Current Price: ${result['current_data']['close']}")
print(f"Predicted Price: ${result['predictions']['random_forest']}")
print(f"Recommendation: {result['recommendation']['decision']}")
```



## 🔧 Technical Features

### Data Processing Pipeline
```
Yahoo Finance → Feature Engineering → Technical Indicators → ML Models → Predictions
```

### Feature Engineering (30+ Indicators)
- **Price Patterns**: Moving averages (5, 10, 20, 50-day), Bollinger Bands
- **Technical Indicators**: RSI, MACD, Williams %R, Stochastic Oscillator  
- **Volume Analysis**: Volume ratios, volume moving averages
- **Momentum Indicators**: 5, 10, 20-day price momentum
- **Volatility Measures**: Historical volatility, ATR

### Machine Learning Models
1. **Enhanced Linear Regression**: Multi-feature analysis with standardization
2. **Random Forest**: Ensemble method with feature importance ranking
3. **ARIMA**: Time series analysis for trend prediction
4. **LSTM Neural Network**: Deep learning for complex pattern recognition

## 📊 Stock Categories

### Default Portfolio (50 stocks)
- **Technology**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NFLX, NVDA, AMD, INTC, ADBE, CSCO, ORCL, CRM, IBM
- **Finance**: JPM, BAC, WFC, GS, MS, V, MA, AXP
- **Healthcare**: JNJ, UNH, PFE, ABBV, TMO, ABT, MRK
- **Consumer**: WMT, HD, PG, KO, PEP, NKE, MCD, COST, DIS
- **Industrial**: BA, CAT, GE, MMM, UPS, FDX
- **Energy**: XOM, CVX, COP, SLB, EOG

## 🎨 Screenshots

### Dashboard Overview
![Dashboard](assets/dashboard_screenshot.png)

### ML Predictions Interface
![Predictions](assets/prediction_screenshot.png)

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

## 📈 Future Enhancements

- [ ] Real-time WebSocket connections
- [ ] Portfolio optimization algorithms
- [ ] Sentiment analysis integration
- [ ] Mobile app development
- [ ] Advanced risk metrics (VaR, CVaR)
- [ ] Cryptocurrency support
- [ ] News sentiment correlation
- [ ] Options pricing models

## ⚠️ Disclaimer

**This software is for educational and research purposes only. It is not financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance API** for providing reliable financial data
- **Streamlit** team for the amazing dashboard framework
- **Scikit-learn** and **TensorFlow** communities for ML libraries
- **Plotly** for interactive visualization capabilities

## 📞 Support

If you encounter any issues or have questions:

1. **Check existing issues**: [GitHub Issues](https://github.com/yourusername/stock-market-ml-analyzer/issues)
2. **Create new issue**: Provide detailed description and error logs
3. **Discussion forum**: Join our community discussions

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/stock-market-ml-analyzer&type=Date)](https://star-history.com/#yourusername/stock-market-ml-analyzer&Date)

---

**Made with ❤️**
