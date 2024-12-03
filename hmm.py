import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def get_hmm_signal(
    target_symbol,
    start_date='2010-01-01',
    end_date='2023-12-31'
):
    """
    Train HMM on multiple stocks and apply it to a specific stock.
    """
    # Get training data with 1 year holdout
    train_end = pd.Timestamp(end_date) - pd.Timedelta(days=252)  
    train_data = yf.download(target_symbol, start=start_date, end=train_end)[['Close']]
    
    # Calculate features for single stock
    train_data['Log_Returns'] = np.log(train_data['Close'] / train_data['Close'].shift(1))
    train_data['Volatility'] = train_data['Log_Returns'].rolling(window=20).std()
    train_data['Momentum'] = train_data['Close'] / train_data['Close'].shift(10) - 1
    
    # Drop any NaN values
    train_data.dropna(inplace=True)

    # Scale features for HMM
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train_data[['Log_Returns', 'Volatility', 'Momentum']])

    # Train the HMM model
    hmm_model = GaussianHMM(n_components=2, covariance_type='full', n_iter=1000, random_state=42)
    hmm_model.fit(scaled_features)

    # Load the target stock data
    target_data = yf.download(target_symbol, start=start_date, end=end_date)[['Close']]
    target_data['Log_Returns'] = np.log(target_data['Close'] / target_data['Close'].shift(1))
    target_data['Volatility'] = target_data['Log_Returns'].rolling(window=20).std()
    target_data['Momentum'] = target_data['Close'] / target_data['Close'].shift(10) - 1
    target_data.dropna(inplace=True)

    # Scale the target stock features
    scaled_target_features = scaler.transform(target_data[['Log_Returns', 'Volatility', 'Momentum']])

    # Predict regimes for the target stock
    target_data['Regime'] = hmm_model.predict(scaled_target_features)

    # Analyze the regimes
    means = np.array([hmm_model.means_[i][0] for i in range(hmm_model.n_components)])
    variances = np.array([np.diag(hmm_model.covars_[i])[0] for i in range(hmm_model.n_components)])
    print("Means of each hidden state:", means)
    print("Variances of each hidden state:", variances)
    print("Transition Matrix:\n", hmm_model.transmat_)

    # Identify bull and bear markets based on mean returns
    if means[0] > means[1]:
        bull_state = 0
        bear_state = 1
    else:
        bull_state = 1
        bear_state = 0

    return target_data, hmm_model, bull_state, bear_state

def main():
    # Define target stock symbol
    target_symbol = '^GSPC'

    # Get HMM signal and regimes
    data, hmm_model, bull_state, bear_state = get_hmm_signal(target_symbol)

    # Plot the regimes for the target stock
    plt.figure(figsize=(14, 7))
    for i in range(hmm_model.n_components):
        state = (data['Regime'] == i)
        plt.plot(data.index[state], data['Close'][state], '.', label=f'State {i}')
    plt.title(f'Market Regimes Identified by HMM for {target_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()