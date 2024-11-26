import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hmm import get_hmm_signal, backtest_strategy as backtest_hmm, calculate_performance as hmm_performance
from sma import get_sma_signal, backtest_strategy_sma, calculate_performance_sma
from lstm import get_lstm_signal
import warnings
import yfinance as yf
warnings.filterwarnings("ignore")

def align_signals(signals):
    """
    Align signals to the same date range by interpolating smaller signals to the largest signal's index.
    """
    largest_index = max(signals, key=lambda x: len(x.index)).index  # Find the largest date range
    aligned_signals = {}
    for name, signal in signals.items():
        # Interpolate smaller signals to align with the largest index
        f = interp1d(signal.index.astype(int), signal.values, kind="nearest", fill_value="extrapolate")
        aligned_signal = pd.Series(f(largest_index.astype(int)), index=largest_index)
        aligned_signals[name] = aligned_signal
    return aligned_signals

def combine_signals(hmm_signal, sma_signal, lstm_signal, weights=(1, 1, 1)):
    """
    Combine signals using weighted voting.
    Signals:
    - 1: Strong Buy
    - -1: Strong Sell
    - 0: Hold
    """
    # Combine with weights
    combined_signal = (
        weights[0] * hmm_signal +
        weights[1] * sma_signal +
        weights[2] * lstm_signal
    )
    # Map to buy/sell/hold with varying strengths
    final_signal = combined_signal.apply(lambda x: 
        1 if x > 1.5 else -1 if x < -1.5 else 0
    )
    return final_signal

def backtest_combined_strategy(combined_signal, stock_data, bond_data):
    """
    Backtest the combined strategy using combined signals to rebalance portfolio.
    """
    stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    bond_data['Log_Returns'] = np.log(bond_data['Close'] / bond_data['Close'].shift(1))
    
    # Align bond data to stock data
    bond_data = bond_data.reindex(stock_data.index).fillna(method="ffill").dropna()

    # Initialize portfolio
    portfolio = pd.DataFrame(index=stock_data.index)
    portfolio['Total'] = 100000  # Starting value

    # Backtest
    for i in range(1, len(portfolio)):
        signal = combined_signal.iloc[i - 1]
        # Determine weights based on signal
        stock_weight = 0.7 if signal == 1 else 0.4 if signal == -1 else 0.5
        bond_weight = 1 - stock_weight

        # Calculate returns
        stock_return = stock_data['Log_Returns'].iloc[i]
        bond_return = bond_data['Log_Returns'].iloc[i]
        total_return = np.exp(stock_weight * stock_return + bond_weight * bond_return)

        # Update portfolio value
        portfolio['Total'].iloc[i] = portfolio['Total'].iloc[i - 1] * total_return

    return portfolio

if __name__ == "__main__":
    # Generate signals
    hmm_signal, hmm_states, hmm_bull, hmm_bear = get_hmm_signal()
    sma_signal = get_sma_signal()
    lstm_signal, lstm_predicted_returns, lstm_actual_returns = get_lstm_signal()

    # Align signals
    signals = {"HMM": hmm_signal['Regime'], "SMA": sma_signal, "LSTM": lstm_signal}
    aligned_signals = align_signals(signals)

    # Combine signals
    weights = (1, 1, 1)  # Equal weights for now; can be adjusted
    combined_signal = combine_signals(
        aligned_signals['HMM'], aligned_signals['SMA'], aligned_signals['LSTM'], weights
    )

    # Download stock and bond data
    stock_symbol = '^GSPC'
    bond_symbol = 'IEF'
    stock_data = yf.download(stock_symbol, start="2000-01-01", end="2023-12-31")
    bond_data = yf.download(bond_symbol, start="2000-01-01", end="2023-12-31")

    # Backtest the combined strategy
    portfolio = backtest_combined_strategy(combined_signal, stock_data, bond_data)

    # Plot combined portfolio performance
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio.index, portfolio['Total'], label='Combined Strategy Portfolio')
    plt.title('Portfolio Performance Using Combined Strategy')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

    # Print performance metrics
    cumulative_return = (portfolio['Total'][-1] / portfolio['Total'][0]) - 1
    annualized_return = (portfolio['Total'][-1] / portfolio['Total'][0]) ** (1 / ((portfolio.index[-1] - portfolio.index[0]).days / 365.25)) - 1
    portfolio['Daily_Return'] = portfolio['Total'].pct_change()
    annualized_volatility = portfolio['Daily_Return'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    print(f'Cumulative Return: {cumulative_return * 100:.2f}%')
    print(f'Annualized Return: {annualized_return * 100:.2f}%')
    print(f'Annualized Volatility: {annualized_volatility * 100:.2f}%')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
