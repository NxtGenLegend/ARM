from lstm import get_lstm_signal
from hmm import get_hmm_signal
from sma import get_sma_signal
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Getting Signals from Each Model
lstm_signal_series = get_lstm_signal()
hmm_signal_series = get_hmm_signal()
sma_signal_series = get_sma_signal()

# Aligning Signals into a dataframe
signals = pd.DataFrame({
    'LSTM_Signal': lstm_signal_series,
    'HMM_Signal': hmm_signal_series,
    'SMA_Signal': sma_signal_series
})

# Ensure all signals are aligned on the same dates
signals.dropna(inplace=True)

# Combining Signals Using Majority Voting
signals['Combined_Signal'] = signals[['LSTM_Signal', 'HMM_Signal', 'SMA_Signal']].sum(axis=1)

# Define final signal
def get_final_signal(x):
    if x >= 2:
        return 1  # Strong buy
    elif x <= -2:
        return -1  # Strong sell
    else:
        return 0  # Hold 

signals['Final_Signal'] = signals['Combined_Signal'].apply(get_final_signal)

# Implementing Portfolio Rebalancing

# Shifting signals to avoid look-ahead bias
signals['Position'] = signals['Final_Signal'].shift(1).fillna(0)

# Define asset weights based on position
signals['Stock_Weight'] = np.where(signals['Position'] == 1, 0.7,
                          np.where(signals['Position'] == -1, 0.4, 0.5))
signals['Bond_Weight'] = 1 - signals['Stock_Weight']

# Download stock and bond prices
symbol = '^GSPC'  # S&P 500 index
bond_symbol = 'IEF'  # iShares 7-10 Year Treasury Bond ETF

start_date = signals.index.min().strftime('%Y-%m-%d')
end_date = signals.index.max().strftime('%Y-%m-%d')

stock_data = yf.download(symbol, start=start_date, end=end_date)['Close']
bond_data = yf.download(bond_symbol, start=start_date, end=end_date)['Close']

# Combine price data
price_data = pd.DataFrame({
    'Stock_Close': stock_data,
    'Bond_Close': bond_data
})

price_data.dropna(inplace=True)

# Calculating Returns
price_data['Stock_Return'] = price_data['Stock_Close'].pct_change()
price_data['Bond_Return'] = price_data['Bond_Close'].pct_change()
price_data.dropna(inplace=True)

# Merge signals and returns
portfolio = signals.join(price_data[['Stock_Return', 'Bond_Return']], how='inner')
portfolio.dropna(inplace=True)

# Calculating Portfolio Returns
portfolio['Portfolio_Return'] = (portfolio['Stock_Weight'] * portfolio['Stock_Return'] +
                                 portfolio['Bond_Weight'] * portfolio['Bond_Return'])

# Calculating cumulative returns
portfolio['Cumulative_Return'] = (1 + portfolio['Portfolio_Return']).cumprod()

# Evaluating Performance
cumulative_return = portfolio['Cumulative_Return'][-1] - 1
years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
annualized_return = portfolio['Cumulative_Return'][-1] ** (1 / years) - 1
annualized_volatility = portfolio['Portfolio_Return'].std() * np.sqrt(252)
sharpe_ratio = (annualized_return) / annualized_volatility

print(f'Cumulative Return: {cumulative_return * 100:.2f}%')
print(f'Annualized Return: {annualized_return * 100:.2f}%')
print(f'Annualized Volatility: {annualized_volatility * 100:.2f}%')
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

# Plot Results
plt.figure(figsize=(14, 7))
plt.plot(portfolio.index, portfolio['Cumulative_Return'], label='Combined Strategy')
plt.title('Portfolio Performance with Combined Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()