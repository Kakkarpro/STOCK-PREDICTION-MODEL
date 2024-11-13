# data_manager.py
import os
import pandas as pd
from datetime import datetime

class DataManager:
    def __init__(self):
        # Create directory structure
        self.data_dir = 'stock_data'
        self.price_dir = os.path.join(self.data_dir, 'historical_prices')
        self.info_dir = os.path.join(self.data_dir, 'company_info')
        self.initialize_storage()

    def initialize_storage(self):
        """Initialize storage directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.price_dir, exist_ok=True)
        os.makedirs(self.info_dir, exist_ok=True)

    def save_stock_data(self, ticker, price_data, info):
        """Save stock data and info to CSV files"""
        # Save historical price data
        price_path = os.path.join(self.price_dir, f"{ticker}_prices.csv")
        price_data.to_csv(price_path)

        # Save company info
        info_df = pd.DataFrame([{
            'ticker': ticker,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        info_path = os.path.join(self.info_dir, f"{ticker}_info.csv")
        info_df.to_csv(info_path, index=False)

    def load_stock_data(self, ticker, use_cached=True):
        """Load stock data from CSV if available"""
        if not use_cached:
            return None, None

        try:
            # Try loading price data
            price_path = os.path.join(self.price_dir, f"{ticker}_prices.csv")
            info_path = os.path.join(self.info_dir, f"{ticker}_info.csv")

            if os.path.exists(price_path) and os.path.exists(info_path):
                price_data = pd.read_csv(price_path, index_col=0, parse_dates=True)
                info_data = pd.read_csv(info_path)
                return price_data, info_data.iloc[0].to_dict()

        except Exception as e:
            print(f"Error loading cached data: {e}")
        
        return None, None

# train_model.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import datetime
from data_manager import DataManager

def fetch_sp500_top50():
    """Fetch top 50 S&P 500 stocks by market cap"""
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500.nlargest(50, 'Market Cap')['Symbol'].tolist()

def fetch_stock_data(ticker, years=14, data_manager=None):
    """Fetch historical stock data with caching"""
    # Try loading from cache first
    if data_manager:
        cached_data, cached_info = data_manager.load_stock_data(ticker)
        if cached_data is not None:
            return cached_data, cached_info

    # Fetch from yfinance if not cached
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=years*365)
    data = yf.download(ticker, start=start_date, end=end_date)
    info = yf.Ticker(ticker).info

    # Save the data if data manager is provided
    if data_manager:
        data_manager.save_stock_data(ticker, data, info)

    return data, info

def prepare_data(data, sequence_length=60):
    """Prepare data for LSTM model"""
    df = data[['Adj Close']].copy()
    
    # Create features
    df['Returns'] = df['Adj Close'].pct_change()
    df['MA5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - 30):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length:i + sequence_length + 30, 0])
    
    return np.array(X), np.array(y), scaler

def create_model(input_shape):
    """Create LSTM model"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(30)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_models():
    """Train models on top 50 S&P 500 stocks"""
    data_manager = DataManager()
    tickers = fetch_sp500_top50()
    sequence_length = 60
    
    X_combined = []
    y_combined = []
    
    # Save list of processed tickers
    with open(os.path.join(data_manager.data_dir, 'processed_tickers.txt'), 'w') as f:
        f.write('\n'.join(tickers))
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        data, info = fetch_stock_data(ticker, data_manager=data_manager)
        X, y, scaler = prepare_data(data, sequence_length)
        X_combined.extend(X)
        y_combined.extend(y)
    
    X_combined = np.array(X_combined)
    y_combined = np.array(y_combined)
    
    model = create_model((sequence_length, X_combined.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Save training metadata
    training_metadata = {
        'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sequence_length': sequence_length,
        'input_shape': X_combined.shape,
        'feature_names': ['Adj Close', 'Returns', 'MA5', 'MA20', 'Volatility']
    }
    
    with open(os.path.join(data_manager.data_dir, 'training_metadata.pkl'), 'wb') as f:
        pickle.dump(training_metadata, f)
    
    # Train the model
    history = model.fit(
        X_combined, y_combined,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(data_manager.data_dir, 'training_history.csv'))
    
    # Save model and scaler
    model.save(os.path.join(data_manager.data_dir, 'stock_prediction_model.h5'))
    with open(os.path.join(data_manager.data_dir, 'stock_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

# predict.py
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import datetime
from data_manager import DataManager

def load_prediction_resources(data_dir):
    """Load the trained model and scaler"""
    model = load_model(Path(data_dir) / 'stock_prediction_model.h5')
    with open(Path(data_dir) / 'stock_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(Path(data_dir) / 'training_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return model, scaler, metadata

def predict_prices(ticker):
    """Predict stock prices for next 5 days and month"""
    data_manager = DataManager()
    model, scaler, metadata = load_prediction_resources(data_manager.data_dir)
    
    # Get stock data and information
    data, stock_info = fetch_stock_data(ticker, data_manager=data_manager)
    
    df = prepare_prediction_data(data)
    scaled_data = scaler.transform(df)
    
    sequence_length = metadata['sequence_length']
    X_pred = scaled_data[-sequence_length:].reshape(1, sequence_length, scaled_data.shape[1])
    
    predicted_scaled = model.predict(X_pred)[0]
    predicted_prices = scaler.inverse_transform(
        np.column_stack((predicted_scaled, np.zeros((len(predicted_scaled), scaled_data.shape[1]-1))))
    )[:, 0]
    
    return {
        'stock_info': stock_info,
        'next_5_days': predicted_prices[:5],
        'next_month': predicted_prices,
        'last_price': data['Adj Close'][-1],
        'last_date': data.index[-1]
    }

def prepare_prediction_data(data):
    """Prepare the most recent data for prediction"""
    df = data[['Adj Close']].copy()
    df['Returns'] = df['Adj Close'].pct_change()
    df['MA5'] = df['Adj Close'].rolling(window=5).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    return df.dropna()

if __name__ == "__main__":
    while True:
        ticker = input("\nEnter stock ticker (or 'quit' to exit): ").upper()
        if ticker == 'QUIT':
            break
            
        try:
            results = predict_prices(ticker)
            
            print("\nStock Information:")
            print(f"Last Updated: {results['last_date'].strftime('%Y-%m-%d')}")
            print(f"Current Price: ${results['last_price']:.2f}")
            for key, value in results['stock_info'].items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:,.2f}")
                else:
                    print(f"{key}: {value}")
            
            print("\nPrice Predictions:")
            print("\nNext 5 Days:")
            for i, price in enumerate(results['next_5_days'], 1):
                change = ((price - results['last_price']) / results['last_price']) * 100
                print(f"Day {i}: ${price:.2f} ({change:+.2f}%)")
            
            print("\nMonthly Forecast (5-day intervals):")
            for i, price in enumerate(results['next_month'][5:], 6):
                if i % 5 == 0:
                    change = ((price - results['last_price']) / results['last_price']) * 100
                    print(f"Day {i}: ${price:.2f} ({change:+.2f}%)")
                    
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")