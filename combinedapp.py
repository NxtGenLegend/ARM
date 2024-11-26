# combined_strategy.py
# Import individual models (assuming they are in the same directory)
from lstm import get_lstm_signal
from hmm import get_hmm_signal
from sma import get_sma_signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

class StrategyVisualizer:
    def __init__(self, signals_df: pd.DataFrame, data: pd.DataFrame, metrics: Dict[str, float]):
        self.signals = signals_df
        self.data = data
        self.metrics = metrics
        
    def plot_signals_performance(self):
        """Plot test period signals and performance."""
        # Get last 252 trading days (approximately 1 year) for clearer visualization
        test_period = self.signals.index[-252:]
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot normalized price movement (dark blue)
        price_returns = (1 + self.data.loc[test_period, 'Returns']).cumprod()
        norm_price = price_returns / price_returns.iloc[0]
        ax.plot(test_period, norm_price, 
                color='#000080', label='S&P 500', linewidth=2, alpha=0.7)
        
        # Plot signals as background colors
        for idx in test_period:
            if self.signals.loc[idx, 'Combined_Signal'] > 0:
                # Light green for buy signals
                ax.axvline(idx, color='#90EE90', alpha=0.2)
            elif self.signals.loc[idx, 'Combined_Signal'] < 0:
                # Light red for sell signals
                ax.axvline(idx, color='#FFB6C1', alpha=0.2)
        
        # Plot strategy performance (dark green)
        strategy_returns = (1 + self.signals.loc[test_period, 'Strategy_Returns']).cumprod()
        norm_strategy = strategy_returns / strategy_returns.iloc[0]
        ax.plot(test_period, norm_strategy, 
                color='#006400', label='Strategy', linewidth=2)
        
        # Add legend and labels
        ax.set_title('Strategy Signals vs Market Performance (Last Year)', fontsize=12)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Normalized Price', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics text box
        metrics_text = (f"Annual Return: {self.metrics['Annual Return']:.1f}%\n"
                       f"Sharpe Ratio: {self.metrics['Sharpe Ratio']:.2f}")
        ax.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

class CombinedTradingStrategy:
    def __init__(self, 
                 symbol: str = '^GSPC',
                 start_date: str = '2000-01-01',
                 end_date: str = '2023-12-31',
                 weights: Dict[str, float] = {'lstm': 0.4, 'hmm': 0.3, 'sma': 0.3}):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.weights = weights
        self.data = None
        self.signals = None
        
    def get_market_data(self) -> pd.DataFrame:
        """Fetch and prepare market data."""
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        self.data = data[['Close']].copy()
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data.dropna(inplace=True)
        return self.data

    def align_signals(self, signals_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Align signals from different models to the same timeframe."""
        signals_df = pd.DataFrame(index=self.data.index)
        
        # Add signals from each model
        for name, signal in signals_dict.items():
            # Ensure signal is a Series with the correct index
            if isinstance(signal, pd.Series):
                signals_df[f'{name}_Signal'] = signal
            else:
                signals_df[f'{name}_Signal'] = pd.Series(signal, index=self.data.index)
        
        # Forward fill any missing values and handle NaN
        signals_df = signals_df.fillna(method='ffill').fillna(0)
        
        return signals_df

    def normalize_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all signals to [-1, 1] range."""
        for col in signals_df.columns:
            max_abs = max(abs(signals_df[col].min()), abs(signals_df[col].max()))
            if max_abs > 0:
                signals_df[col] = signals_df[col] / max_abs
        return signals_df

    def generate_combined_signal(self) -> pd.DataFrame:
        """Generate and combine trading signals from all models."""
        print("Generating signals from individual models...")
        
        try:
            # Get LSTM signals
            lstm_signals, lstm_predictions, _ = get_lstm_signal()
            print("LSTM signals generated successfully")
        except Exception as e:
            print(f"Error generating LSTM signals: {e}")
            lstm_signals = pd.Series(0, index=self.data.index)
        
        try:
            # Get HMM signals
            hmm_data, hidden_states, bull_state, _ = get_hmm_signal()
            hmm_signals = pd.Series(
                np.where(hidden_states == bull_state, 1, -1),
                index=hmm_data.index
            )
            print("HMM signals generated successfully")
        except Exception as e:
            print(f"Error generating HMM signals: {e}")
            hmm_signals = pd.Series(0, index=self.data.index)
        
        try:
            # Get SMA signals
            sma_signals = get_sma_signal()
            print("SMA signals generated successfully")
        except Exception as e:
            print(f"Error generating SMA signals: {e}")
            sma_signals = pd.Series(0, index=self.data.index)
        
        # Combine all signals
        signals_dict = {
            'LSTM': lstm_signals,
            'HMM': hmm_signals,
            'SMA': sma_signals
        }
        
        # Align and normalize signals
        signals_df = self.align_signals(signals_dict)
        signals_df = self.normalize_signals(signals_df)
        
        # Calculate weighted combined signal
        signals_df['Combined_Signal'] = (
            self.weights['lstm'] * signals_df['LSTM_Signal'] +
            self.weights['hmm'] * signals_df['HMM_Signal'] +
            self.weights['sma'] * signals_df['SMA_Signal']
        )
        
        # Generate trading decisions
        signals_df['Trading_Decision'] = pd.cut(
            signals_df['Combined_Signal'],
            bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
            labels=['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        )
        
        # Calculate strategy returns
        signals_df['Strategy_Returns'] = self.data['Returns'] * signals_df['Combined_Signal'].shift(1)
        
        self.signals = signals_df
        return signals_df

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        if self.signals is None:
            raise ValueError("Signals not generated yet")
            
        strategy_returns = self.signals['Strategy_Returns'].dropna()
        
        # Calculate annualized metrics
        annual_return = strategy_returns.mean() * 252 * 100  # Convert to percentage
        annual_vol = strategy_returns.std() * np.sqrt(252) * 100  # Convert to percentage
        sharpe_ratio = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
        
        # Calculate drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100  # Convert to percentage
        
        # Calculate hit ratio
        hits = np.sum(np.sign(strategy_returns) == np.sign(self.signals['Combined_Signal'].shift(1).dropna()))
        hit_ratio = hits / len(strategy_returns)
        
        return {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Hit Ratio': hit_ratio
        }
    
    def run_strategy(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Run the complete trading strategy."""
        print("Starting combined trading strategy...")
        self.get_market_data()
        signals = self.generate_combined_signal()
        metrics = self.calculate_metrics()
        
        # Create visualizer and generate simplified plot
        visualizer = StrategyVisualizer(signals, self.data, metrics)
        visualizer.plot_signals_performance()
        
        return signals, metrics

def main():
    # Initialize strategy
    strategy = CombinedTradingStrategy(
        symbol='^GSPC',
        start_date='2000-01-01',
        end_date='2023-12-31',
        weights={'lstm': 0.4, 'hmm': 0.3, 'sma': 0.3}
    )
    
    # Run strategy
    signals, metrics = strategy.run_strategy()
    
    # Print performance metrics
    print("\nStrategy Performance:")
    for metric, value in metrics.items():
        if metric in ['Annual Return', 'Annual Volatility', 'Max Drawdown']:
            print(f"{metric}: {value:.2f}%")
        elif metric == 'Hit Ratio':
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()