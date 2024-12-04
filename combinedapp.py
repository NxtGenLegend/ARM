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
    def __init__(self, symbol: str, start_date: str = '1999-01-01', 
                 end_date: str = '2020-12-31'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.signals = None
        self.load_data()
        
    def load_data(self):
        """Load market data with proper lag handling"""
        # Add warmup period for signal calculation
        warmup_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=252)
        data = yf.download(self.symbol, warmup_start, self.end_date)
        
        if data.empty or 'Close' not in data.columns:
            raise ValueError("Data download failed or 'Close' column is missing.")
            
        # Calculate returns with proper shift to avoid lookahead bias
        data['Returns'] = data['Close'].pct_change().shift(1)
        data.dropna(inplace=True)
        
        # Trim to requested period after calculating initial returns
        self.data = data[self.start_date:][['Close', 'Returns']]
        print(f"Data loaded from {self.data.index[0]} to {self.data.index[-1]}")

    def generate_signals(self):
        """Combine signals from all models with proper lag handling"""
        print("Generating signals...")
        
        # Get signals from individual models
        hmm_data, hidden_states, bull_state, bear_state = get_hmm_signal(self.symbol)
        lstm_signals, predictions, actuals = get_lstm_signal(self.symbol)
        sma_signals = get_sma_signal(self.symbol)

        # Create signals DataFrame aligned with price data
        self.signals = pd.DataFrame(index=self.data.index)
        
        # Align and shift all signals by 1 day to avoid lookahead bias
        self.signals['HMM'] = pd.Series(hidden_states, index=hmm_data.index).shift(1).reindex(self.signals.index).fillna(0)
        self.signals['LSTM'] = pd.Series(lstm_signals, index=lstm_signals.index).shift(1).reindex(self.signals.index).fillna(0)
        self.signals['SMA'] = pd.Series(sma_signals, index=sma_signals.index).shift(1).reindex(self.signals.index).fillna(0)

        def calculate_allocation(row):
            # Calculate weights based on previous day's signals
            
            # MODE 1 PARAMS (Low) - Defensive 
            # hmm_weight = 0.4 if row['HMM'] == bull_state else -0.55
            # lstm_weight = 0.4 if row['LSTM'] == 1 else -0.4
            # sma_weight = 0.4 if row['SMA'] == 1 else -0.4
            
            # MODE 2 Params (MID) - Stable Market
            # hmm_weight = 0.6 if row['HMM'] == bull_state else -0.8
            # lstm_weight = 0.4 if row['LSTM'] == 1 else -0.8
            # sma_weight = -0.4 if row['SMA'] == 1 else 0.4
            
            #MODE 3 PARAMS (MID) - Volatile
            # hmm_weight = 0.3 if row['HMM'] == bull_state else 0.6
            # lstm_weight = 0.225 if row['LSTM'] == 1 else -0.225
            # sma_weight = 0.225 if row['SMA'] == 1 else -0.225
            
            hmm_weight = 0.6 if row['HMM'] == bull_state else -0.8
            lstm_weight = 0.4 if row['LSTM'] == 1 else -0.8
            sma_weight = -0.4 if row['SMA'] == 1 else 0.4

            
            # Combine signals with bounds
            return np.clip(hmm_weight + lstm_weight + sma_weight, 0.1, 0.9)
        
        # Calculate allocations
        self.signals['Stock_Allocation'] = self.signals.apply(calculate_allocation, axis=1)
        self.signals['Bond_Allocation'] = 1 - self.signals['Stock_Allocation']
        
        return self.signals

    def backtest_strategy(self):
        """Backtest the strategy with proper return alignment"""
        try:
            # Download and prepare bond data
            bond_data = yf.download('IEF', self.start_date, self.end_date)
            if bond_data.empty or 'Close' not in bond_data.columns:
                raise ValueError("Bond data is empty or lacks a 'Close' column.")
            
            # Calculate bond returns with proper shift
            bond_data['Bond_Returns'] = bond_data['Close'].pct_change().shift(1)
            bond_returns = bond_data['Bond_Returns'].reindex(self.data.index).fillna(0)

            # Initialize portfolio
            portfolio = pd.DataFrame(index=self.signals.index)
            portfolio['Stock_Alloc'] = self.signals['Stock_Allocation']
            portfolio['Bond_Alloc'] = self.signals['Bond_Allocation']
            
            # Calculate portfolio returns using shifted data
            portfolio['Returns'] = (
                portfolio['Stock_Alloc'] * self.data['Returns'] + 
                portfolio['Bond_Alloc'] * bond_returns
            )

            # Calculate cumulative portfolio value
            portfolio['Value'] = (1 + portfolio['Returns']).cumprod() * 100000

            return portfolio
            
        except Exception as e:
            print(f"Error during backtest: {e}")
            raise

    def calculate_metrics(self, portfolio):
        """Calculate performance metrics using properly lagged returns"""
        returns = portfolio['Returns'].dropna()
        
        # Calculate metrics using geometric returns
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
            'Max Drawdown': max_drawdown * 100
        }

    def plot_results(self, portfolio):
        """Plot strategy results with benchmark comparison"""
        plt.figure(figsize=(15, 7))
        
        # Plot strategy and benchmark performance
        plt.plot(portfolio['Value'], label='Combined Strategy Portfolio', linewidth=2)
        plt.plot((1 + self.data['Returns']).cumprod() * 100000, 
                 label='Benchmark', alpha=0.7)
        
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

def main():
    try:
        # Initialize and run strategy
        strategy = CombinedStrategy('PG')
        signals = strategy.generate_signals()
        portfolio = strategy.backtest_strategy()
        metrics = strategy.calculate_metrics(portfolio)
        
        # Print metrics
        print("\nStrategy Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")
        
        # Plot results
        strategy.plot_results(portfolio)
        
    except Exception as e:
        print(f"Error in strategy execution: {e}")

if __name__ == "__main__":
    main()