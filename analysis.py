import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from scipy import stats

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def get_data(tk, st='2010-01-01'):
    s = yf.Ticker(tk)
    d = s.history(start=st)
    return d, s

def get_stock_info(tk):
    s = yf.Ticker(tk)
    i = s.info
    return {
        'name': i.get('longName', 'N/A'),
        'sector': i.get('sector', 'N/A'),
        'industry': i.get('industry', 'N/A'),
        'market_cap': i.get('marketCap', 'N/A'),
        'pe_ratio': i.get('forwardPE', 'N/A'),
        'dividend_yield': i.get('dividendYield', 'N/A'),
        '52w_high': i.get('fiftyTwoWeekHigh', 'N/A'),
        '52w_low': i.get('fiftyTwoWeekLow', 'N/A'),
        'avg_volume': i.get('averageVolume', 'N/A'),
        'beta': i.get('beta', 'N/A'),
        'recommendation': i.get('recommendationKey', 'N/A'),
        'target_price': i.get('targetMeanPrice', 'N/A'),
        'earnings_date': i.get('earningsDate', ['N/A'])[0] if isinstance(i.get('earningsDate', []), list) else 'N/A',
        'peg_ratio': i.get('pegRatio', 'N/A'),
        'profit_margins': i.get('profitMargins', 'N/A')
    }

def calculate_technical_indicators(df):
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    # Moving Averages
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()
    
    # Additional Technical Indicators
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Rate of Change
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Average True Range (ATR)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    return df

def format_large_num(num):
    if not isinstance(num, (int, float)) or pd.isna(num):
        return 'N/A'
    if num >= 1e9:
        return f'${num/1e9:.2f}B'
    if num >= 1e6:
        return f'${num/1e6:.2f}M'
    return f'${num:,.2f}'

def calculate_trend_strength(df, window=20):
    price_change = df['Close'].pct_change()
    trend = price_change.rolling(window=window).mean()
    volatility = price_change.rolling(window=window).std()
    trend_strength = abs(trend) / volatility
    return trend_strength.iloc[-1]

def get_support_resistance(df, window=20):
    highs = df['High'].rolling(window=window).max()
    lows = df['Low'].rolling(window=window).min()
    return lows.iloc[-1], highs.iloc[-1]

def calculate_risk_metrics(df):
    returns = df['Close'].pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    var_95 = np.percentile(returns, 5)
    return {
        'daily_volatility': returns.std(),
        'annual_volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': sharpe,
        'var_95': var_95
    }

def analyze_price_action(df):
    last_close = df['Close'].iloc[-1]
    sma_50 = df['50_MA'].iloc[-1]
    sma_200 = df['200_MA'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    signals = []
    if last_close > sma_50:
        signals.append("Above 50-day MA (Bullish)")
    else:
        signals.append("Below 50-day MA (Bearish)")
        
    if last_close > sma_200:
        signals.append("Above 200-day MA (Bullish)")
    else:
        signals.append("Below 200-day MA (Bearish)")
        
    if rsi > 70:
        signals.append("RSI indicates overbought")
    elif rsi < 30:
        signals.append("RSI indicates oversold")
        
    return signals

def plot_analysis(df, ticker):
    fig = plt.figure(figsize=(15, 12))
    
    # Price and Moving Averages
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['Close'], label='Price', linewidth=2)
    ax1.plot(df.index, df['50_MA'], label='50-day MA', linewidth=1.5)
    ax1.plot(df.index, df['200_MA'], label='200-day MA', linewidth=1.5)
    ax1.plot(df.index, df['BB_Upper'], 'g--', label='BB Upper', alpha=0.5)
    ax1.plot(df.index, df['BB_Lower'], 'r--', label='BB Lower', alpha=0.5)
    ax1.set_title(f'{ticker} Price Analysis')
    ax1.legend()
    ax1.grid(True)
    
    # Volume
    ax2 = plt.subplot(3, 1, 2)
    ax2.bar(df.index, df['Volume'], alpha=0.8)
    ax2.set_title('Trading Volume')
    ax2.grid(True)
    
    # RSI
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax3.set_title('RSI Indicator')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_technical_analysis(df):
    last_row = df.iloc[-1]
    signals = analyze_price_action(df)
    trend_strength = calculate_trend_strength(df)
    
    print("\nTECHNICAL ANALYSIS:")
    print("-" * 50)
    print(f"RSI (14): {last_row['RSI']:.2f}")
    print(f"MACD: {last_row['MACD']:.2f}")
    print(f"MACD Signal: {last_row['MACD_Signal']:.2f}")
    print(f"Momentum: {last_row['Momentum']:.2f}")
    print(f"Rate of Change: {last_row['ROC']:.2f}%")
    print(f"ATR: {last_row['ATR']:.2f}")
    print(f"\nTrend Strength: {trend_strength:.2f}")
    print("\nPrice Action Signals:")
    for signal in signals:
        print(f"- {signal}")

def analysis_main(ticker):
    # tk = input("Enter stock ticker: ").upper()
    tk = ticker.upper()
    try:
        print(f"\nFetching data for {tk}...")
        d, s = get_data(tk)
        info = get_stock_info(tk)
        
        print("Calculating technical indicators...")
        d = calculate_technical_indicators(d)
        
        print("Generating analysis...")
        risk_metrics = calculate_risk_metrics(d)
        support, resistance = get_support_resistance(d)
        
        print("\n" + "="*70)
        print(f"COMPREHENSIVE STOCK ANALYSIS: {info['name']}")
        print("="*70)
        
        # Print all analyses
        print(f"\nCurrent Price: ${d['Close'].iloc[-1]:.2f}")
        print(f"Market Cap: {format_large_num(info['market_cap'])}")
        print(f"Beta: {info['beta']}")
        
        print_technical_analysis(d)
        
        print("\nRISK METRICS:")
        print(f"Annual Volatility: {risk_metrics['annual_volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print(f"Value at Risk (95%): {risk_metrics['var_95']*100:.2f}%")
        
        print("\nSUPPORT/RESISTANCE LEVELS:")
        print(f"Support: ${support:.2f}")
        print(f"Resistance: ${resistance:.2f}")
        
        print("\nGenerating charts...")
        plot_analysis(d, tk)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Check ticker and try again.")
