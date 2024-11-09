# Import necessary libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# List of stocks
symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB']
start_date = '2018-01-01'
end_date = '2023-01-01'

# Download data
data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

# Calculate momentum scores (e.g., 6-month returns)
momentum_window = 126  # Approx 6 months (21 trading days per month)

# Calculate returns over the momentum window
momentum_scores = data.pct_change(momentum_window)

# Rank the assets
rankings = momentum_scores.rank(axis=1, method='min', ascending=False)

# Define quantiles
num_quantiles = 5
quantile_labels = range(1, num_quantiles + 1)
quantiles = rankings.apply(lambda x: pd.qcut(x, q=num_quantiles, labels=quantile_labels), axis=1)

# Assign positions based on quantiles
positions = quantiles.applymap(lambda x: 1 if x == 1 else 0)  # Overweight top quantile

# Shift positions to avoid look-ahead bias
positions = positions.shift(1)
positions.fillna(0, inplace=True)

# Calculate daily returns
daily_returns = data.pct_change()

# Calculate strategy returns
strategy_returns = (positions * daily_returns).mean(axis=1)

# Calculate cumulative returns
cumulative_returns = (1 + strategy_returns).cumprod()

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(cumulative_returns.index, cumulative_returns.values, label='Momentum Strategy')
plt.title('Momentum Strategy Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
