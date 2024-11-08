import pandas as pd
import numpy as np

def calculate_technical_features(df):
    """Calculate technical indicators using pandas"""
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Basic price features
    data['returns'] = data['Adj Close'].pct_change()
    data['log_returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    
    # Moving averages
    data['sma_5'] = data['Adj Close'].rolling(window=5).mean()
    data['sma_20'] = data['Adj Close'].rolling(window=20).mean()
    data['sma_50'] = data['Adj Close'].rolling(window=50).mean()
    
    # Volatility
    data['volatility'] = data['returns'].rolling(window=20).std()
    
    # Price momentum
    data['momentum_5'] = data['Adj Close'] / data['Adj Close'].shift(5) - 1
    data['momentum_20'] = data['Adj Close'] / data['Adj Close'].shift(20) - 1
    
    # Volume features
    data['volume_ma_5'] = data['Volume'].rolling(window=5).mean()
    data['volume_ma_20'] = data['Volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['Volume'] / data['volume_ma_20']
    
    # Price relative to moving averages
    data['price_sma5_ratio'] = data['Adj Close'] / data['sma_5']
    data['price_sma20_ratio'] = data['Adj Close'] / data['sma_20']
    
    # Bollinger Bands
    data['bb_middle'] = data['sma_20']
    data['bb_std'] = data['Adj Close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
    data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    return data

def prepare_features(data):
    """Prepare features for model input"""
    # Calculate technical features
    df = calculate_technical_features(data)
    
    # Select features for model
    feature_columns = [
        'Adj Close', 'returns', 'log_returns',
        'sma_5', 'sma_20', 'sma_50',
        'volatility', 'momentum_5', 'momentum_20',
        'volume_ratio', 'price_sma5_ratio', 'price_sma20_ratio',
        'bb_width'
    ]
    
    # Drop rows with NaN values
    df = df[feature_columns].dropna()
    
    return df
