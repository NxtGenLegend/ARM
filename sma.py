# Import necessary libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download historical data for a stock (e.g., Apple Inc.)
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'

data = yf.download(symbol, start=start_date, end=end_date)
data = data[['Close']]
data.dropna(inplace=True)

# Calculate the Simple Moving Averages
short_window = 50
long_window = 200

data['SMA_Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
data['SMA_Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

# Generate signals
data['Signal'] = 0
data['Signal'][short_window:] = \
    np.where(data['SMA_Short'][short_window:] > data['SMA_Long'][short_window:], 1, 0)

# Generate trading orders
data['Position'] = data['Signal'].diff()

# Plot the closing price along with the SMAs and signals
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['SMA_Short'], label=f'{short_window}-Day SMA', alpha=0.9)
plt.plot(data['SMA_Long'], label=f'{long_window}-Day SMA', alpha=0.9)

# Mark buy and sell signals
plt.plot(data[data['Position'] == 1].index,
         data['SMA_Short'][data['Position'] == 1],
         '^', markersize=10, color='g', label='Buy Signal')
plt.plot(data[data['Position'] == -1].index,
         data['SMA_Short'][data['Position'] == -1],
         'v', markersize=10, color='r', label='Sell Signal')

plt.title('SMA Crossover Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
