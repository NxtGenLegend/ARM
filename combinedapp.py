import numpy as np
import pandas as pd
import yfinance as yf
from lstm import get_lstm_signal
from hmm import get_hmm_signal
from sma import get_sma_signal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CombinedStrategy:
    #'^GSPC'
    def __init__(self, symbol: str, start_date: str = '2010-01-01', 
                 end_date: str = '2023-12-31'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.signals = None
        self.load_data()
        
    def load_data(self):
        """Load market data"""
        data = yf.download(self.symbol, self.start_date, self.end_date)
        if data.empty or 'Close' not in data.columns:
            raise ValueError("Data download failed or 'Close' column is missing.")
        self.data = data[['Close']]
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data.dropna(inplace=True)
        print(f"Data loaded from {self.data.index[0]} to {self.data.index[-1]}")

    def generate_signals(self):
        """Combine signals from all models"""
        print("Generating signals...")
        hmm_data, hidden_states, bull_state, bear_state = get_hmm_signal(self.symbol)
        lstm_signals, predictions, actuals = get_lstm_signal(self.symbol)
        sma_signals = get_sma_signal(self.symbol)

        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['HMM'] = pd.Series(hidden_states, index=hmm_data.index).reindex(self.signals.index).fillna(0)
        self.signals['LSTM'] = pd.Series(lstm_signals, index=lstm_signals.index).reindex(self.signals.index).fillna(0)
        self.signals['SMA'] = pd.Series(sma_signals, index=sma_signals.index).reindex(self.signals.index).fillna(0)

        def calculate_allocation(row):
            hmm_weight = 0.7 if row['HMM'] == bull_state else 0.3
            lstm_weight = 0.1 if row['LSTM'] == 1 else -0.1
            sma_weight = 0.1 if row['SMA'] == 1 else -0.1
            return np.clip(hmm_weight + lstm_weight + sma_weight, 0.1, 0.9)
        
        self.signals['Stock_Allocation'] = self.signals.apply(calculate_allocation, axis=1)
        self.signals['Bond_Allocation'] = 1 - self.signals['Stock_Allocation']
        return self.signals

    def backtest_strategy(self):
        """Backtest the combined strategy"""
        try:
            # Download bond data
            bond_data = yf.download('IEF', self.start_date, self.end_date)
            if bond_data.empty or 'Close' not in bond_data.columns:
                raise ValueError("Bond data is empty or lacks a 'Close' column.")
            bond_data['Bond_Returns'] = bond_data['Close'].pct_change().dropna()

            # Align bond data with the portfolio index
            bond_returns = bond_data['Bond_Returns'].reindex(self.data.index).fillna(0)

            # Initialize portfolio DataFrame
            portfolio = pd.DataFrame(index=self.signals.index)
            portfolio['Stock_Alloc'] = self.signals['Stock_Allocation'].shift(1).fillna(0.5)
            portfolio['Bond_Alloc'] = self.signals['Bond_Allocation'].shift(1).fillna(0.5)

            # Calculate portfolio returns
            portfolio['Returns'] = (
                portfolio['Stock_Alloc'] * self.data['Returns'] + 
                portfolio['Bond_Alloc'] * bond_returns
            )

            # Calculate portfolio value over time
            portfolio['Value'] = (1 + portfolio['Returns']).cumprod() * 100000

            return portfolio
        except Exception as e:
            print(f"Error during backtest: {e}")
            raise

    def calculate_metrics(self, portfolio):
        """Calculate performance metrics"""
        returns = portfolio['Returns'].dropna()
        total_return = portfolio['Value'].iloc[-1] / 100000 - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility
        drawdowns = (portfolio['Value'] / portfolio['Value'].cummax() - 1)
        max_drawdown = drawdowns.min()
        return {
            'Total Return': total_return * 100,
            'Annual Return': annual_return * 100,
            'Annual Volatility': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown * 100,
        }

    def plot_results(self, portfolio):
        """Plot strategy results"""
        plt.figure(figsize=(15, 7))
        plt.plot(portfolio['Value'], label='Combined Strategy Portfolio', linewidth=2)
        plt.plot((1 + self.data['Returns']).cumprod() * 100000, label='Benchmark', alpha=0.7)
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    strategy = CombinedStrategy('JNJ')
    signals = strategy.generate_signals()
    portfolio = strategy.backtest_strategy()
    metrics = strategy.calculate_metrics(portfolio)
    print("\nStrategy Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
    strategy.plot_results(portfolio)

if __name__ == "__main__":
    main()
