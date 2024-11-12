import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import warnings

from sklearn.preprocessing import StandardScaler

# Enable the filter for warnings in general
warnings.simplefilter("always")

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


    # Try initializing and fitting the HMM model, if it doesn't converge, try with inverted returns
    try:
        model = GaussianHMM(n_components=2, covariance_type='full', n_iter=3000,
                            init_params='random')  # Changed covariance and init_params

        # Catch warnings during model fitting
        with warnings.catch_warnings(record=True) as caught_warnings:  # Catch warnings
            model.fit(returns)

            # Check for convergence manually based on log likelihood change
            prev_score = -np.inf  # Initialize previous score to a very low value
            for i in range(3000):  # Limit the number of iterations manually
                model.fit(returns)
                current_score = model.score(returns)  # Get the log likelihood score
                score_diff = current_score - prev_score
                prev_score = current_score

                if abs(score_diff) < 1e-4:  # Arbitrary threshold to decide if converged
                    print(f"Model converged after {i + 1} iterations.")
                    break
            else:
                # If no convergence after all iterations, try inverting returns
                print("Model did not converge, inverting returns and retrying...")
                data['Log_Returns'] = -data['Log_Returns']  # Invert the log returns
                returns = data['Log_Returns'].values.reshape(-1, 1)
                model.fit(returns)
                print("Model fit after inverting returns.")

    except Exception as e:  # Catch any general exceptions
        print(f"Model did not converge, error: {e}")

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

    # Fetch the same data used in get_hmm_signal() for plotting
    symbol = '^GSPC'  # S&P 500 index
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data.dropna(inplace=True)

    # Align the signals with the data
    data = data.join(hmm_signal_series.rename('Signal'), how='inner')

    # Map signals to regimes for plotting
    data['Regime'] = np.where(data['Signal'] == 1, 'Bull', 'Bear')

    # Plot the Regimes
    plt.figure(figsize=(14, 7))
    for regime in ['Bull', 'Bear']:
        regime_data = data[data['Regime'] == regime]
        plt.plot(regime_data.index, regime_data['Close'], '.', label=f'{regime} Regime')
    plt.title('Market Regimes Identified by HMM')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()