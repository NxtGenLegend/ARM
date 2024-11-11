# sma.py

# Import necessary libraries
import pandas as pd
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_sma_signal():
    # Data Collection
    symbol = '^GSPC'  # Changed to S&P 500 index for consistency
    start_date = '2000-01-01'  # Extended start date
    end_date = '2023-12-31'  # Adjust the end date as needed

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
    data['Signal'][long_window:] = \
        np.where(data['SMA_Short'][long_window:] > data['SMA_Long'][long_window:], 1, -1)

    # Shift signals to avoid look-ahead bias
    sma_signal_series = data['Signal'].shift(1).fillna(0)
    sma_signal_series.index = data.index

    return sma_signal_series

if __name__ == "__main__":
    # Call the function to generate signals
    sma_signal_series = get_sma_signal()

    # Plot the results
    import matplotlib.pyplot as plt

    # Reconstruct the data used in get_sma_signal()
    symbol = '^GSPC'
    start_date = '2000-01-01'
    end_date = '2023-12-31'

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
    data['Signal'][long_window:] = \
        np.where(data['SMA_Short'][long_window:] > data['SMA_Long'][long_window:], 1, -1)

    data['Position'] = data['Signal']

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