# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Step 1: Data Collection
symbol = '^GSPC'  # S&P 500 index 
start_date = '2010-01-01'
end_date = '2023-01-01'

data = yf.download(symbol, start=start_date, end=end_date)
data = data[['Close']]
data.dropna(inplace=True)

# Step 2: Calculate Log Returns
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# Step 3: Prepare Data for HMM
returns = data['Log_Returns'].values.reshape(-1, 1)

# Step 4: Fit the HMM Model
# We'll use 2 components for bull and bear markets
model = GaussianHMM(n_components=2, covariance_type='full', n_iter=1000)
model.fit(returns)

# Step 5: Predict the Regimes
hidden_states = model.predict(returns)
data['Regime'] = hidden_states

# Step 6: Analyze the Regimes
# Extract means and variances of each hidden state
means = np.array([model.means_[i][0] for i in range(model.n_components)])
variances = np.array([np.diag(model.covars_[i])[0] for i in range(model.n_components)])

print("Means of each hidden state:", means)
print("Variances of each hidden state:", variances)

# Identify bull and bear markets based on mean returns
if means[0] > means[1]:
    bull_state = 0
    bear_state = 1
else:
    bull_state = 1
    bear_state = 0

# Step 7: Plot the Regimes
plt.figure(figsize=(14, 7))
for i in range(model.n_components):
    state = (hidden_states == i)
    plt.plot(data.index[state], data['Close'][state], '.', label=f'State {i}')
plt.title('Market Regimes Identified by HMM')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Step 8: Define Rebalancing Strategy
def rebalance_portfolio(state):
    if state == bull_state:
        # Bull market: Increase stock exposure
        stock_weight = 0.7
        bond_weight = 0.3
    else:
        # Bear market: Decrease stock exposure
        stock_weight = 0.4
        bond_weight = 0.6
    return stock_weight, bond_weight

# Step 9: Backtesting the Strategy
# Get bond data (e.g., iShares 7-10 Year Treasury Bond ETF)
bond_symbol = 'IEF'
bond_data = yf.download(bond_symbol, start=start_date, end=end_date)
bond_data = bond_data[['Close']]
bond_data.rename(columns={'Close': 'Bond_Close'}, inplace=True)
bond_data.dropna(inplace=True)

# Calculate bond returns
bond_data['Bond_Log_Returns'] = np.log(bond_data['Bond_Close'] / bond_data['Bond_Close'].shift(1))
bond_data.dropna(inplace=True)

# Align data
data = data.join(bond_data['Bond_Log_Returns'], how='inner')

# Initialize portfolio
portfolio = pd.DataFrame(index=data.index)
portfolio['Total'] = 100000  # Starting with $100,000

# Backtest
for i in range(1, len(data)):
    # Get the current state
    state = data['Regime'].iloc[i - 1]
    # Get the asset weights
    stock_weight, bond_weight = rebalance_portfolio(state)
    # Calculate returns
    stock_return = data['Log_Returns'].iloc[i]
    bond_return = data['Bond_Log_Returns'].iloc[i]
    # Total portfolio return
    total_return = np.exp(stock_weight * stock_return + bond_weight * bond_return)
    # Update portfolio value
    portfolio['Total'].iloc[i] = portfolio['Total'].iloc[i - 1] * total_return

# Step 10: Plot Portfolio Performance
plt.figure(figsize=(14, 7))
plt.plot(portfolio.index, portfolio['Total'], label='HMM Strategy Portfolio')
plt.title('Portfolio Performance Using HMM Regime Switching')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

# Step 11: Calculate Performance Metrics
# Calculate cumulative returns
cumulative_return = (portfolio['Total'][-1] / portfolio['Total'][0]) - 1
print(f'Cumulative Return: {cumulative_return * 100:.2f}%')

# Calculate annualized return
years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
annualized_return = (portfolio['Total'][-1] / portfolio['Total'][0]) ** (1 / years) - 1
print(f'Annualized Return: {annualized_return * 100:.2f}%')

# Calculate annualized volatility
portfolio['Daily_Return'] = portfolio['Total'].pct_change()
annualized_volatility = portfolio['Daily_Return'].std() * np.sqrt(252)
print(f'Annualized Volatility: {annualized_volatility * 100:.2f}%')

# Calculate Sharpe Ratio (Assuming risk-free rate is 0)
sharpe_ratio = (annualized_return) / annualized_volatility
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

def get_hmm_signal():
    # Generate HMM signals
    # Bull regime: signal = 1 (buy); Bear regime: signal = -1 (sell)
    hmm_signal_series = np.where(data['Regime'] == bull_state, 1, -1)
    hmm_signal_series = pd.Series(hmm_signal_series, index=data.index)
    
    return hmm_signal_series