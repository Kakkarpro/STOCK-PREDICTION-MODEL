import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def load_ticker_resources(ticker):
    try:
        model = load_model(f"{ticker}_lstm_stock_model.h5")
        
        scaler_params = np.load(f"{ticker}_scaler.npy")
        scaler = MinMaxScaler()
        scaler.min_, scaler.scale_ = scaler_params
        
        return model, scaler
    except Exception as e:
        raise Exception(f"Error loading resources for {ticker}: {e}")

def fetch_stock_data(ticker, lookback_days=100):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)
        
        info = {}
        try:
            raw_info = stock.info
            info['market_cap'] = raw_info.get('marketCap', 'N/A')
            info['pe_ratio'] = raw_info.get('forwardPE', 'N/A')
            info['dividend_yield'] = raw_info.get('dividendYield', 'N/A')
            if info['dividend_yield'] not in ['N/A', None]:
                info['dividend_yield'] *= 100 
            info['beta'] = raw_info.get('beta', 'N/A')
            info['52w_high'] = raw_info.get('fiftyTwoWeekHigh', 'N/A')
            info['52w_low'] = raw_info.get('fiftyTwoWeekLow', 'N/A')
            info['avg_volume'] = raw_info.get('averageVolume', 'N/A')
            
        except Exception as e:
            print(f"Warning: Could not fetch some fundamental data: {e}")
        
        return history, info
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {e}")

def prepare_prediction_data(data, scaler):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    X_pred = scaled_data[-60:].reshape(1, 60, 1)
    return X_pred, close_prices[-1][0]

def inverse_transform_predictions(predictions, scaler):
    predictions_2d = predictions.reshape(-1, 1)
    return scaler.inverse_transform(predictions_2d).flatten()

def main():
    while True:
        ticker = input("\nEnter stock ticker (or 'quit' to exit): ").upper()
        if ticker == 'QUIT':
            break
        
        try:
            print(f"\nLoading model for {ticker}...")
            model, scaler = load_ticker_resources(ticker)
            
            print(f"Fetching recent data for {ticker}...")
            data, stock_info = fetch_stock_data(ticker)
            
            X_pred, current_price = prepare_prediction_data(data, scaler)
            
            raw_predictions = model.predict(X_pred)[0]
            predictions = inverse_transform_predictions(raw_predictions, scaler)
            
            print("\nStock Information:")
            print(f"Ticker: {ticker}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Last Updated: {data.index[-1].strftime('%Y-%m-%d')}")
            
            print("\nFundamental Data:")
            for key, value in stock_info.items():
                if isinstance(value, (int, float)) and value != 'N/A':
                    if key == 'market_cap':
                        print(f"{key}: ${value:,.0f}")
                    elif key in ['pe_ratio', 'dividend_yield', 'beta']:
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
            
            print("\nPrice Predictions:")
            print("\nForcast for next week:")
            for i, price in enumerate(predictions[:5], 1):
                change = ((price - current_price) / current_price) * 100
                print(f" ${price:.2f}")
            
            # print("\nMonthly Forecast (5-day intervals):")
            # for i, price in enumerate(predictions[5:], 6):
            #     if i % 5 == 0:
            #         change = ((price - current_price) / current_price) * 100
            #         print(f"Day {i}: ${price:.2f} ({change:+.2f}%)")
                    
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

if __name__ == "__main__":
    main()
