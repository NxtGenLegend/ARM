import matplotlib.pyplot as plt
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
    data['Signal'][long_window:] = np.where(data['SMA_Short'][long_window:] > data['SMA_Long'][long_window:], 1, -1)

    # Shift signals to avoid look-ahead bias
    sma_signal_series = data['Signal'].shift(1).fillna(0)
    sma_signal_series.index = data.index

    return sma_signal_series

# Function to calculate portfolio weights based on signals
def rebalance_portfolio_sma(signal):
    if signal == 1:  # Buy signal
        stock_weight = 0.7
        bond_weight = 0.3
    elif signal == -1:  # Sell signal
        stock_weight = 0.4
        bond_weight = 0.6
    else:  # Neutral
        stock_weight = 0.5
        bond_weight = 0.5
    return stock_weight, bond_weight

# Function to backtest the SMA strategy
def backtest_strategy_sma(data, signal_series):
    # Download bond data (e.g., iShares 7-10 Year Treasury Bond ETF)
    bond_symbol = 'IEF'
    bond_data = yf.download(bond_symbol, start=data.index.min(), end=data.index.max())
    bond_data = bond_data[['Close']]
    bond_data.rename(columns={'Close': 'Bond_Close'}, inplace=True)
    bond_data.dropna(inplace=True)

    # Calculate bond returns
    bond_data['Bond_Log_Returns'] = np.log(bond_data['Bond_Close'] / bond_data['Bond_Close'].shift(1))
    bond_data.dropna(inplace=True)

    # Align bond data with the main data
    data = data.join(bond_data['Bond_Log_Returns'], how='inner')

    # Calculate stock returns
    data['Stock_Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # Initialize portfolio
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Total'] = 100000  # Starting with $100,000

    # Backtest portfolio based on SMA signals
    for i in range(1, len(data)):
        signal = signal_series.iloc[i - 1]  # Use signal from the previous day
        stock_weight, bond_weight = rebalance_portfolio_sma(signal)

        stock_return = data['Stock_Log_Returns'].iloc[i]
        bond_return = data['Bond_Log_Returns'].iloc[i]
        total_return = np.exp(stock_weight * stock_return + bond_weight * bond_return)

        portfolio['Total'].iloc[i] = portfolio['Total'].iloc[i - 1] * total_return

    return portfolio

# Function to calculate performance metrics
def calculate_performance_sma(portfolio):
    # Calculate cumulative returns
    cumulative_return = (portfolio['Total'][-1] / portfolio['Total'][0]) - 1

    # Calculate annualized return
    years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    annualized_return = (portfolio['Total'][-1] / portfolio['Total'][0]) ** (1 / years) - 1

    # Calculate annualized volatility
    portfolio['Daily_Return'] = portfolio['Total'].pct_change()
    annualized_volatility = portfolio['Daily_Return'].std() * np.sqrt(252)

    # Calculate Sharpe Ratio (Assuming risk-free rate is 0)
    sharpe_ratio = annualized_return / annualized_volatility

    return cumulative_return, annualized_return, annualized_volatility, sharpe_ratio

if __name__ == "__main__":
    # Call the function to generate signals
    sma_signal_series = get_sma_signal()

    # Reconstruct data used in get_sma_signal()
    symbol = '^GSPC'
    start_date = '2000-01-01'
    end_date = '2023-12-31'

    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data.dropna(inplace=True)

    # Calculate the Simple Moving Averages
    short_window = 20
    long_window = 100

    data['SMA_Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    # Generate signals
    # Uncomments
    # MAKE ARRAY OF LEN, INITIALIZE WITH ZEROS
    data['Signal'] = np.array(data['SMA_Short'])
    for i in range(len(data['Signal'])):
        if (data['SMA_Short'][i] > data['SMA_Long'][i]):
            data['Signal'][i] = 1
        elif (data['SMA_Short'][i] < data['SMA_Long'][i]):
            data['Signal'][i] = -1
        #data['Signal'][i] = data['SMA_Short'][i] > data['SMA_Long'][i]
    #print(np.where(data['SMA_Short'][long_window:] > data['SMA_Long'][long_window:], 1, -1))

    data['Position'] = data['Signal']
    #for i in range(len(data['Signal'])):
    #    print(data['Signal'][i])
    #print(data['Signal'][long_window:])

    # Make portfolio
    portfolio = backtest_strategy_sma(data, sma_signal_series)

    # Backtest and evaluate the strategy

    # Plot portfolio performance
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio.index, portfolio['Total'], label='SMA Strategy Portfolio')
    plt.title('Portfolio Performance Using SMA Strategy')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

    # Calculate performance metrics
    cumulative_return, annualized_return, annualized_volatility, sharpe_ratio = calculate_performance_sma(portfolio)
    print(f'Cumulative Return: {cumulative_return * 100:.2f}%')
    print(f'Annualized Return: {annualized_return * 100:.2f}%')
    print(f'Annualized Volatility: {annualized_volatility * 100:.2f}%')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')


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