# -*- coding: utf-8 -*-
"""
Enhanced Stock Price Prediction with Multiple Features
Flask Backend with Machine Learning Models
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import os
import io
import base64
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Try to import optional packages
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("ARIMA not available. Install statsmodels for ARIMA predictions.")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("LSTM not available. Install tensorflow for LSTM predictions.")

app = Flask(__name__)
CORS(app)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs('static', exist_ok=True)

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

def create_technical_features(df):
    """Create technical indicators and additional features"""
    try:
        df_features = df.copy()
        
        # Price-based features
        df_features['Price_Range'] = df_features['High'] - df_features['Low']
        df_features['Price_Change'] = df_features['Close'] - df_features['Open']
        df_features['Price_Change_Pct'] = (df_features['Close'] - df_features['Open']) / df_features['Open'] * 100
        
        # Moving averages
        df_features['MA_5'] = df_features['Close'].rolling(window=5).mean()
        df_features['MA_10'] = df_features['Close'].rolling(window=10).mean()
        df_features['MA_20'] = df_features['Close'].rolling(window=20).mean()
        df_features['MA_50'] = df_features['Close'].rolling(window=50).mean()
        
        # Moving average ratios
        df_features['MA_5_20_Ratio'] = df_features['MA_5'] / df_features['MA_20']
        df_features['MA_10_50_Ratio'] = df_features['MA_10'] / df_features['MA_50']
        df_features['Price_MA20_Ratio'] = df_features['Close'] / df_features['MA_20']
        
        # Volatility features
        df_features['Volatility_5'] = df_features['Close'].rolling(window=5).std()
        df_features['Volatility_20'] = df_features['Close'].rolling(window=20).std()
        
        # Volume features
        df_features['Volume_MA_5'] = df_features['Volume'].rolling(window=5).mean()
        df_features['Volume_MA_20'] = df_features['Volume'].rolling(window=20).mean()
        df_features['Volume_Ratio'] = df_features['Volume'] / df_features['Volume_MA_20']
        
        # Price momentum
        df_features['Momentum_5'] = df_features['Close'] / df_features['Close'].shift(5) - 1
        df_features['Momentum_10'] = df_features['Close'] / df_features['Close'].shift(10) - 1
        df_features['Momentum_20'] = df_features['Close'] / df_features['Close'].shift(20) - 1
        
        # Manual technical indicators (without TA-Lib dependency)
        try:
            # RSI calculation
            df_features['RSI'] = calculate_rsi(df_features['Close'], window=14)
            
            # MACD calculation
            macd_line, macd_signal, macd_histogram = calculate_macd(df_features['Close'])
            df_features['MACD'] = macd_line
            df_features['MACD_Signal'] = macd_signal
            df_features['MACD_Hist'] = macd_histogram
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(df_features['Close'], window=20)
            df_features['BB_Upper'] = bb_upper
            df_features['BB_Lower'] = bb_lower
            df_features['BB_Middle'] = bb_middle
            df_features['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            df_features['BB_Position'] = (df_features['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Williams %R
            df_features['Williams_R'] = calculate_williams_r(df_features['High'], 
                                                           df_features['Low'], 
                                                           df_features['Close'], window=14)
            
            # Stochastic Oscillator
            stoch_k, stoch_d = calculate_stochastic(df_features['High'], 
                                                   df_features['Low'], 
                                                   df_features['Close'])
            df_features['Stoch_K'] = stoch_k
            df_features['Stoch_D'] = stoch_d
            
            # Additional indicators
            df_features['CCI'] = calculate_cci(df_features['High'], df_features['Low'], 
                                             df_features['Close'], window=20)
            df_features['ATR'] = calculate_atr(df_features['High'], df_features['Low'], 
                                             df_features['Close'], window=14)
            
        except Exception as e:
            print(f"Technical indicators error (continuing without them): {e}")
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df_features[f'Close_Lag_{lag}'] = df_features['Close'].shift(lag)
            df_features[f'Volume_Lag_{lag}'] = df_features['Volume'].shift(lag)
        
        # Time-based features
        df_features['DayOfWeek'] = pd.to_datetime(df_features['Date']).dt.dayofweek
        df_features['Month'] = pd.to_datetime(df_features['Date']).dt.month
        df_features['Quarter'] = pd.to_datetime(df_features['Date']).dt.quarter
        
        return df_features
    
    except Exception as e:
        print(f"Error creating features: {e}")
        return df

def get_feature_columns():
    """Define which columns to use as features"""
    base_features = [
        'Open', 'High', 'Low', 'Volume',
        'Price_Range', 'Price_Change', 'Price_Change_Pct',
        'MA_5', 'MA_10', 'MA_20', 'MA_50',
        'MA_5_20_Ratio', 'MA_10_50_Ratio', 'Price_MA20_Ratio',
        'Volatility_5', 'Volatility_20',
        'Volume_MA_5', 'Volume_MA_20', 'Volume_Ratio',
        'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
        'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3', 'Volume_Lag_5',
        'DayOfWeek', 'Month', 'Quarter'
    ]
    
    # Technical indicators (manually calculated)
    technical_features = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'BB_Position',
        'Williams_R', 'Stoch_K', 'Stoch_D', 'CCI', 'ATR'
    ]
    
    return base_features, technical_features

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band, rolling_mean

def calculate_williams_r(high, low, close, window=14):
    """Calculate Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_cci(high, low, close, window=20):
    """Calculate Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

@app.route('/api/predict', methods=['POST'])
def predict_stock():
    try:    
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        # Get historical data
        stock_data = get_historical_data(symbol)
        if stock_data is None:
            return jsonify({'error': f'Could not fetch data for symbol {symbol}'}), 400
        
        # Create features
        stock_data_features = create_technical_features(stock_data)
        
        # Get current stock info
        current_data = get_current_stock_data(stock_data_features)
        
        # Run predictions
        predictions = {}
        errors = {}
        feature_importance = {}
        
        # Enhanced Linear Regression
        try:
            lr_pred, lr_error, lr_importance = enhanced_linear_regression_prediction(stock_data_features)
            predictions['linear_regression'] = round(lr_pred, 2)
            errors['linear_regression'] = round(lr_error, 2)
            feature_importance['linear_regression'] = lr_importance
        except Exception as e:
            predictions['linear_regression'] = None
            errors['linear_regression'] = None
            print(f"Enhanced Linear Regression error: {e}")
        
        # Random Forest
        try:
            rf_pred, rf_error, rf_importance = random_forest_prediction(stock_data_features)
            predictions['random_forest'] = round(rf_pred, 2)
            errors['random_forest'] = round(rf_error, 2)
            feature_importance['random_forest'] = rf_importance
        except Exception as e:
            predictions['random_forest'] = None
            errors['random_forest'] = None
            print(f"Random Forest error: {e}")
        
        # ARIMA (if available)
        if ARIMA_AVAILABLE:
            try:
                arima_pred, arima_error = arima_prediction(stock_data)
                predictions['arima'] = round(arima_pred, 2)
                errors['arima'] = round(arima_error, 2)
            except Exception as e:
                predictions['arima'] = None
                errors['arima'] = None
                print(f"ARIMA error: {e}")
        else:
            predictions['arima'] = None
            errors['arima'] = None
        
        # Enhanced LSTM (if available)
        if LSTM_AVAILABLE:
            try:
                lstm_pred, lstm_error = enhanced_lstm_prediction(stock_data_features)
                predictions['lstm'] = round(lstm_pred, 2)
                errors['lstm'] = round(lstm_error, 2)
            except Exception as e:
                predictions['lstm'] = None
                errors['lstm'] = None
                print(f"Enhanced LSTM error: {e}")
        else:
            predictions['lstm'] = None
            errors['lstm'] = None
        
        # Generate recommendation
        recommendation = generate_recommendation(stock_data_features, predictions, current_data)
        
        # Create response
        response = {
            'symbol': symbol,
            'current_data': current_data,
            'predictions': predictions,
            'errors': errors,
            'feature_importance': feature_importance,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict_stock: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def get_historical_data(symbol):
    """Fetch historical stock data using yfinance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years of data
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
        
        # Reset index and clean data
        data = data.reset_index()
        data = data.dropna()
        
        return data
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_current_stock_data(df):
    """Extract current stock data from the dataframe"""
    try:
        latest = df.iloc[-1]
        return {
            'open': round(latest['Open'], 2),
            'close': round(latest['Close'], 2),
            'high': round(latest['High'], 2),
            'low': round(latest['Low'], 2),
            'volume': int(latest['Volume']),
            'date': latest['Date'].strftime('%Y-%m-%d') if hasattr(latest['Date'], 'strftime') else str(latest['Date'])
        }
    except Exception as e:
        print(f"Error getting current data: {e}")
        return {}

def enhanced_linear_regression_prediction(df):
    """Enhanced Linear Regression with multiple features"""
    try:
        # Prepare features
        base_features, technical_features = get_feature_columns()
        available_features = []
        
        # Check which features are available
        for feature in base_features + technical_features:
            if feature in df.columns:
                available_features.append(feature)
        
        # Prepare data
        forecast_days = 1
        df_lr = df.copy()
        df_lr['Close_shifted'] = df_lr['Close'].shift(-forecast_days)
        df_lr = df_lr.dropna()
        
        # Features and target
        X = df_lr[available_features].values
        y = df_lr['Close_shifted'].values
        
        # Train-test split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate error
        error = math.sqrt(mean_squared_error(y_test, y_pred))
        
        # Feature importance (absolute coefficients)
        feature_importance = dict(zip(available_features, np.abs(model.coef_)))
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:10])
        
        # Future prediction
        last_features = df[available_features].iloc[-1:].values
        future_pred = model.predict(scaler_X.transform(last_features))
        
        return float(future_pred[0]), error, feature_importance
    
    except Exception as e:
        print(f"Enhanced linear regression error: {e}")
        raise e

def random_forest_prediction(df):
    """Random Forest prediction with multiple features"""
    try:
        # Prepare features
        base_features, technical_features = get_feature_columns()
        available_features = []
        
        # Check which features are available
        for feature in base_features + technical_features:
            if feature in df.columns:
                available_features.append(feature)
        
        # Prepare data
        forecast_days = 1
        df_rf = df.copy()
        df_rf['Close_shifted'] = df_rf['Close'].shift(-forecast_days)
        df_rf = df_rf.dropna()
        
        # Features and target
        X = df_rf[available_features].values
        y = df_rf['Close_shifted'].values
        
        # Train-test split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate error
        error = math.sqrt(mean_squared_error(y_test, y_pred))
        
        # Feature importance
        feature_importance = dict(zip(available_features, model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:10])
        
        # Future prediction
        last_features = df[available_features].iloc[-1:].values
        future_pred = model.predict(last_features)
        
        return float(future_pred[0]), error, feature_importance
    
    except Exception as e:
        print(f"Random forest error: {e}")
        raise e

def enhanced_lstm_prediction(df):
    """Enhanced LSTM prediction with multiple features"""
    try:
        # Prepare features
        base_features, technical_features = get_feature_columns()
        available_features = []
        
        # Check which features are available (limit to most important ones for LSTM)
        important_features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20', 
                            'Volatility_5', 'RSI', 'MACD', 'Close_Lag_1']
        for feature in important_features:
            if feature in df.columns:
                available_features.append(feature)
        
        # If no additional features, use basic OHLCV
        if len(available_features) < 3:
            available_features = ['Open', 'High', 'Low', 'Volume']
        
        # Prepare data
        feature_data = df[available_features + ['Close']].dropna().values
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create sequences
        sequence_length = 10
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, :-1])  # All features except Close
            y.append(scaled_data[i, -1])  # Close price
        
        X, y = np.array(X), np.array(y)
        
        # Train-test split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Inverse transform predictions
        # Create dummy array with same shape as original for inverse transform
        dummy_features = np.zeros((len(predictions), len(available_features)))
        predictions_full = np.column_stack([dummy_features, predictions])
        predictions_inv = scaler.inverse_transform(predictions_full)[:, -1]
        
        dummy_features_test = np.zeros((len(y_test), len(available_features)))
        y_test_full = np.column_stack([dummy_features_test, y_test.reshape(-1, 1)])
        y_test_inv = scaler.inverse_transform(y_test_full)[:, -1]
        
        # Calculate error
        error = math.sqrt(mean_squared_error(y_test_inv, predictions_inv))
        
        # Future prediction
        last_sequence = scaled_data[-sequence_length:, :-1].reshape(1, sequence_length, -1)
        future_pred_scaled = model.predict(last_sequence)
        
        dummy_future = np.zeros((1, len(available_features)))
        future_pred_full = np.column_stack([dummy_future, future_pred_scaled])
        future_pred = scaler.inverse_transform(future_pred_full)[0, -1]
        
        return float(future_pred), error
    
    except Exception as e:
        print(f"Enhanced LSTM error: {e}")
        raise e

def arima_prediction(df):
    """ARIMA prediction (only if statsmodels is available)"""
    try:
        # Prepare data
        data = df['Close'].values
        
        # Train-test split
        split = int(0.8 * len(data))
        train, test = data[:split], data[split:]
        
        # ARIMA model
        model = ARIMA(train, order=(5, 1, 0))
        fitted_model = model.fit()
        
        # Predictions
        predictions = []
        history = list(train)
        
        for i in range(len(test)):
            model_temp = ARIMA(history, order=(5, 1, 0))
            fitted_temp = model_temp.fit()
            pred = fitted_temp.forecast(steps=1)[0]
            predictions.append(pred)
            history.append(test[i])
        
        # Calculate error
        error = math.sqrt(mean_squared_error(test, predictions))
        
        # Future prediction
        final_model = ARIMA(data, order=(5, 1, 0))
        final_fitted = final_model.fit()
        future_pred = final_fitted.forecast(steps=1)[0]
        
        return float(future_pred), error
    
    except Exception as e:
        print(f"ARIMA error: {e}")
        raise e

def generate_recommendation(df, predictions, current_data):
    """Generate buy/sell recommendation based on predictions"""
    try:
        current_price = current_data.get('close', 0)
        valid_predictions = [p for p in predictions.values() if p is not None]
        
        if not valid_predictions:
            return {
                'decision': 'HOLD',
                'confidence': 'LOW',
                'reason': 'Insufficient prediction data'
            }
        
        avg_prediction = np.mean(valid_predictions)
        price_change = ((avg_prediction - current_price) / current_price) * 100
        
        # Calculate prediction consensus
        prediction_std = np.std(valid_predictions) if len(valid_predictions) > 1 else 0
        consensus_score = 1 - (prediction_std / current_price) if current_price > 0 else 0
        
        # Simple decision logic with consensus consideration
        if price_change > 2:
            decision = 'BUY'
            confidence = 'HIGH' if price_change > 5 and consensus_score > 0.8 else 'MEDIUM'
        elif price_change < -2:
            decision = 'SELL'
            confidence = 'HIGH' if price_change < -5 and consensus_score > 0.8 else 'MEDIUM'
        else:
            decision = 'HOLD'
            confidence = 'MEDIUM' if consensus_score > 0.7 else 'LOW'
        
        return {
            'decision': decision,
            'confidence': confidence,
            'expected_change': round(price_change, 2),
            'avg_prediction': round(avg_prediction, 2),
            'consensus_score': round(consensus_score, 2),
            'reason': f'Expected price change: {price_change:.1f}%, Consensus: {consensus_score:.1f}'
        }
    
    except Exception as e:
        print(f"Recommendation error: {e}")
        return {
            'decision': 'HOLD',
            'confidence': 'LOW',
            'reason': 'Error generating recommendation'
        }

if __name__ == '__main__':
    print("Starting Enhanced Stock Prediction Server...")
    print("Available models:")
    print(f"- Enhanced Linear Regression: Always available")
    print(f"- Random Forest: Always available") 
    print(f"- ARIMA: {'Available' if ARIMA_AVAILABLE else 'Not available (install statsmodels)'}")
    print(f"- Enhanced LSTM: {'Available' if LSTM_AVAILABLE else 'Not available (install tensorflow)'}")
    print("Features include: Technical indicators, Moving averages, Volatility, Volume analysis, Price momentum")
    
    app.run(debug=True, host='0.0.0.0', port=5000)