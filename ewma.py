# Import necessary libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download historical data for a stock 
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

data = yf.download(symbol, start=start_date, end=end_date)
data = data[['Close']]
data.dropna(inplace=True)

# Calculate EWMA
ewma_window = 20
data['EWMA'] = data['Close'].ewm(span=ewma_window, adjust=False).mean()

# Calculate Bollinger Bands
std_dev = data['Close'].ewm(span=ewma_window, adjust=False).std()
data['Upper Band'] = data['EWMA'] + (std_dev * 2)
data['Lower Band'] = data['EWMA'] - (std_dev * 2)

# Plot the closing price, EWMA, and Bollinger Bands
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['EWMA'], label='EWMA', color='blue')
plt.plot(data['Upper Band'], label='Upper Band', color='green')
plt.plot(data['Lower Band'], label='Lower Band', color='red')
plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.1)
plt.title('EWMA with Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
