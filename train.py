import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

TICKER_LIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'JPM', 'JNJ', 'NVDA',
    'V', 'PG', 'UNH', 'HD', 'DIS', 'MA', 'PYPL', 'ADBE', 'NFLX', 'INTC',
    'KO', 'PEP', 'CSCO', 'CMCSA', 'VZ', 'MRK', 'ABT', 'T', 'CRM', 'WMT',
    'NKE', 'ORCL', 'XOM', 'MCD', 'IBM', 'GS', 'BA', 'MMM', 'CAT', 'GE',
    'TXN', 'QCOM', 'BMY', 'AMGN', 'MDT', 'HON', 'CVX', 'SCHW', 'LMT', 'AXP'
]

def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]

# Step 2: Preprocess Data
def preprocess_data(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)
    
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Step 3: Build and Train LSTM Model
def build_improved_lstm_model():
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    # Using Adam optimizer with a lower learning rate to aid convergence
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
# Step 4: Train and Save the Model for Each Ticker
def train_and_save_model(ticker, start_date, end_date):
    print(f"Training model for {ticker}...")
    data = load_stock_data(ticker, start_date, end_date)
    X_train, y_train, scaler = preprocess_data(data.values)
    
    model = build_improved_lstm_model()
    model.fit(X_train, y_train, batch_size=1, epochs=5)
    
    # Save the model and scaler for the ticker
    model_path = f"{ticker}_lstm_stock_model.h5"
    scaler_path = f"{ticker}_scaler.npy"
    model.save(model_path)
    np.save(scaler_path, [scaler.min_, scaler.scale_])
    print(f"Model for {ticker} saved as '{model_path}' and scaler as '{scaler_path}'.")

# Define the main function to train models for all tickers
def main():
    start_date = "2010-01-01"
    end_date = "2023-01-01"

    # Train and save the model for each ticker in the list
    for ticker in TICKER_LIST:
        train_and_save_model(ticker, start_date, end_date)

if __name__ == "__main__":
    main()
