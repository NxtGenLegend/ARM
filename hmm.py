# hmm.py

# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import warnings

from sklearn.preprocessing import StandardScaler

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
    model = GaussianHMM(n_components=2, covariance_type='diag', n_iter=3000, init_params='st')
    model.fit(returns)

    # Predict the Regimes
    hidden_states = model.predict(returns)
    data['Regime'] = hidden_states

    # Analyze the Regimes
    means = np.array([model.means_[i][0] for i in range(model.n_components)])

    # Identify bull and bear markets based on mean returns
    if means[0] > means[1]:
        bull_state = 0
        bear_state = 1
    else:
        bull_state = 1
        bear_state = 0

    # Generate HMM signals
    # Bull regime: signal = 1 (buy); Bear regime: signal = -1 (sell)
    hmm_signals = np.where(data['Regime'] == bull_state, 1,
                           np.where(data['Regime'] == bear_state, -1, 0))
    hmm_signal_series = pd.Series(hmm_signals, index=data.index)

    return hmm_signal_series


if __name__ == "__main__":
    # Call the function to generate signals
    hmm_signal_series = get_hmm_signal()

    # Plot the results
    import matplotlib.pyplot as plt

    # Reconstruct the data used in get_hmm_signal()
    symbol = '^GSPC'  # S&P 500 index
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data.dropna(inplace=True)

    # Calculate Log Returns
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # Fit the HMM Model (as done in get_hmm_signal())
    # returns = data['Log_Returns'].values.reshape(-1, 1)

    # Scale the data
    scaler = StandardScaler()
    returns = scaler.fit_transform(data['Log_Returns'].values.reshape(-1, 1))

    model = GaussianHMM(n_components=2, covariance_type='diag', n_iter=3000, init_params='st')
    model.fit(returns)
    hidden_states = model.predict(returns)
    data['Regime'] = hidden_states

    # Identify bull and bear states
    means = np.array([model.means_[i][0] for i in range(model.n_components)])
    if means[0] > means[1]:
        bull_state = 0
        bear_state = 1
    else:
        bull_state = 1
        bear_state = 0

    # Plot the Regimes
    plt.figure(figsize=(14, 7))
    for i in range(model.n_components):
        state = (hidden_states == i)
        plt.plot(data.index[state], data['Close'][state], '.', label=f'Regime {i}')
    plt.title('Market Regimes Identified by HMM')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

