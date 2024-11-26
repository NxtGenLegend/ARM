import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def get_hmm_signal():
    # Data Collection
    symbol = '^GSPC'  # S&P 500 index
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data.dropna(inplace=True)

    # Calculate Log Returns
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # Prepare Data for HMM
    returns = data['Log_Returns'].values.reshape(-1, 1)

    # Fit the HMM Model
    model = GaussianHMM(n_components=2, covariance_type='full', n_iter=1000)
    model.fit(returns)

    # Predict the Regimes
    hidden_states = model.predict(returns)
    data['Regime'] = hidden_states

    # Analyze the Regimes
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

    return data, hidden_states, bull_state, bear_state

def rebalance_portfolio(state, bull_state, bear_state):
    if state == bull_state:
        # Bull market: Increase stock exposure
        stock_weight = 0.7
        bond_weight = 0.3
    else:
        # Bear market: Decrease stock exposure
        stock_weight = 0.4
        bond_weight = 0.6
    return stock_weight, bond_weight

def backtest_strategy(data, hidden_states, bull_state, bear_state):
    # Get bond data
    bond_symbol = 'IEF'
    bond_data = yf.download(bond_symbol, start=data.index.min(), end=data.index.max())
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
        state = data['Regime'].iloc[i - 1]
        stock_weight, bond_weight = rebalance_portfolio(state, bull_state, bear_state)
        stock_return = data['Log_Returns'].iloc[i]
        bond_return = data['Bond_Log_Returns'].iloc[i]
        total_return = np.exp(stock_weight * stock_return + bond_weight * bond_return)
        portfolio['Total'].iloc[i] = portfolio['Total'].iloc[i - 1] * total_return

    return portfolio

def calculate_performance(portfolio):
    # Calculate cumulative returns
    cumulative_return = (portfolio['Total'][-1] / portfolio['Total'][0]) - 1

    # Calculate annualized return
    years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    annualized_return = (portfolio['Total'][-1] / portfolio['Total'][0]) ** (1 / years) - 1

    # Calculate annualized volatility
    portfolio['Daily_Return'] = portfolio['Total'].pct_change()
    annualized_volatility = portfolio['Daily_Return'].std() * np.sqrt(252)

    # Calculate Sharpe Ratio
    sharpe_ratio = (annualized_return) / annualized_volatility

    return cumulative_return, annualized_return, annualized_volatility, sharpe_ratio

if __name__ == "__main__":
    # Get HMM signals and regimes
    data, hidden_states, bull_state, bear_state = get_hmm_signal()

    # Plot the regimes
    plt.figure(figsize=(14, 7))
    for i in range(2):  # Assuming two hidden states (bull and bear)
        state = (hidden_states == i)
        plt.plot(data.index[state], data['Close'][state], '.', label=f'State {i}')
    plt.title('Market Regimes Identified by HMM')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    # Backtest the strategy
    portfolio = backtest_strategy(data, hidden_states, bull_state, bear_state)

    # Plot portfolio performance
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio.index, portfolio['Total'], label='HMM Strategy Portfolio')
    plt.title('Portfolio Performance Using HMM Regime Switching')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

    # Calculate and print performance metrics
    cumulative_return, annualized_return, annualized_volatility, sharpe_ratio = calculate_performance(portfolio)
    print(f'Cumulative Return: {cumulative_return * 100:.2f}%')
    print(f'Annualized Return: {annualized_return * 100:.2f}%')
    print(f'Annualized Volatility: {annualized_volatility * 100:.2f}%')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
