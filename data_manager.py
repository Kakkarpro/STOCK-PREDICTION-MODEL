import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the top 50 stock symbols
top_50_stocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'BRK-B', 'JPM', 'JNJ', 'NVDA',
    'V', 'PG', 'UNH', 'HD', 'DIS', 'MA', 'PYPL', 'ADBE', 'NFLX', 'INTC',
    'KO', 'PEP', 'CSCO', 'CMCSA', 'VZ', 'MRK', 'ABT', 'T', 'CRM', 'WMT',
    'NKE', 'ORCL', 'XOM', 'MCD', 'IBM', 'GS', 'BA', 'MMM', 'CAT', 'GE',
    'TXN', 'QCOM', 'BMY', 'AMGN', 'MDT', 'HON', 'CVX', 'SCHW', 'LMT', 'AXP'
]

# Set the date range for the past 14 years
end_date = datetime.today()
start_date = end_date - timedelta(days=14 * 365)  # Approximate 14 years

all_stocks_data = pd.DataFrame()

# Download and save data for each stock
for symbol in top_50_stocks:
    print(f"Downloading data for {symbol}...")
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Add a column for the stock symbol to identify it in the combined CSV
    stock_data['Symbol'] = symbol
    
    # Append to the main DataFrame
    all_stocks_data = pd.concat([all_stocks_data, stock_data])

# Save to a CSV file
all_stocks_data.to_csv('top_50_stocks_14_years.csv')
print("Data for the top 50 stocks has been saved to 'top_50_stocks_14_years.csv'")
