import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_sma_signal(symbol, start_date='2000-01-01', end_date='2020-12-31'):
   data = yf.download(symbol, start=start_date, end=end_date)
   data = data[['Close']]
   data.dropna(inplace=True)

   # Calculate SMAs
   short_window = 50
   long_window = 200
   data['SMA_Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
   data['SMA_Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

   # Generate signals
   data['Signal'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, -1)
   
   # Shift signals to avoid look-ahead bias
   sma_signal_series = data['Signal'].shift(1).fillna(0)
   sma_signal_series.index = data.index
   
   return sma_signal_series

if __name__ == "__main__":
   symbol = '^GSPC'
   data = yf.download(symbol, start='2000-01-01', end='2020-12-31')
   
   # Calculate SMAs
   data['SMA_Short'] = data['Close'].rolling(window=50, min_periods=1).mean()
   data['SMA_Long'] = data['Close'].rolling(window=200, min_periods=1).mean()
   
   # Get signals
   signals = get_sma_signal(symbol)
   
   # Plot results
   plt.figure(figsize=(14, 7))
   plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
   plt.plot(data.index, data['SMA_Short'], label='50-Day SMA', alpha=0.9)
   plt.plot(data.index, data['SMA_Long'], label='200-Day SMA', alpha=0.9)

   # Mark buy and sell signals
   buy_signals = data[signals == 1]
   sell_signals = data[signals == -1]
   
   plt.plot(buy_signals.index, buy_signals['SMA_Short'],
           '^', markersize=10, color='g', label='Buy Signal')
   plt.plot(sell_signals.index, sell_signals['SMA_Short'],
           'v', markersize=10, color='r', label='Sell Signal')

   plt.title('SMA Crossover Strategy')
   plt.xlabel('Date')
   plt.ylabel('Price')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()