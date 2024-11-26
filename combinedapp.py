import numpy as np
from lstm import get_lstm_signal
from hmm import get_hmm_signal
from sma import get_sma_signal
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# Use a basic style that's guaranteed to work
plt.style.use('default')

class StrategyVisualizer:
    def __init__(self, signals_df: pd.DataFrame, data: pd.DataFrame, metrics: Dict[str, float]):
        self.signals = signals_df
        self.data = data
        self.metrics = metrics
        
    def plot_strategy_performance(self, save_path: Optional[str] = None):
        """Plot cumulative returns of the strategy vs. buy & hold."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
        
        # Calculate cumulative returns
        strategy_cum_returns = (1 + self.signals['Strategy_Returns'].fillna(0)).cumprod()
        market_cum_returns = (1 + self.data['Returns'].fillna(0)).cumprod()
        
        # Plot cumulative returns with enhanced styling
        ax1.plot(strategy_cum_returns.index, strategy_cum_returns, 
                label='Strategy', linewidth=2, color='#2ecc71')
        ax1.plot(market_cum_returns.index, market_cum_returns, 
                label='Buy & Hold', linewidth=2, color='#3498db', alpha=0.7)
        
        # Add markers for trades
        trade_signals = self.signals['Combined_Signal'].shift(1)
        entry_points = self.signals[trade_signals != trade_signals.shift(1)]
        
        for idx, signal in entry_points['Combined_Signal'].items():
            if signal > 0:
                ax1.scatter(idx, strategy_cum_returns[idx], 
                          marker='^', color='#27ae60', s=100, label='_nolegend_')
            else:
                ax1.scatter(idx, strategy_cum_returns[idx], 
                          marker='v', color='#c0392b', s=100, label='_nolegend_')
        
        ax1.set_title('Strategy Performance: Cumulative Returns Comparison', 
                     fontsize=14, pad=20)
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        metrics_text = (f"Annual Return: {self.metrics['Annual Return']:.1%}\n"
                       f"Sharpe Ratio: {self.metrics['Sharpe Ratio']:.2f}\n"
                       f"Max Drawdown: {self.metrics['Max Drawdown']:.1%}")
        ax1.text(0.02, 0.98, metrics_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot combined signal strength with enhanced styling
        ax2.fill_between(self.signals.index, self.signals['Combined_Signal'], 
                        where=self.signals['Combined_Signal'] >= 0,
                        alpha=0.3, color='#2ecc71', label='Positive Signal')
        ax2.fill_between(self.signals.index, self.signals['Combined_Signal'], 
                        where=self.signals['Combined_Signal'] < 0,
                        alpha=0.3, color='#e74c3c', label='Negative Signal')
        ax2.plot(self.signals.index, self.signals['Combined_Signal'], 
                color='#34495e', label='Signal Strength', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_title('Combined Signal Strength Over Time', fontsize=14, pad=20)
        ax2.set_ylabel('Signal Strength', fontsize=12)
        ax2.legend(fontsize=10, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_contributions(self, save_path: Optional[str] = None):
        """Plot individual model signals and their contributions."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Define colors for each model
        colors = {'LSTM_Signal': '#2ecc71', 
                 'HMM_Signal': '#e74c3c', 
                 'SMA_Signal': '#3498db'}
        
        # Plot individual model signals
        for col, color in colors.items():
            ax1.plot(self.signals.index, self.signals[col], 
                    label=col.replace('_Signal', ''), 
                    color=color, alpha=0.7)
                    
        ax1.set_title('Individual Model Signals Over Time', 
                     fontsize=14, pad=20)
        ax1.set_ylabel('Signal Strength', fontsize=12)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot rolling correlations between models
        window = 60  # 60-day rolling correlation
        correlations = self.signals[['LSTM_Signal', 'HMM_Signal', 'SMA_Signal']].rolling(window).corr()
        
        # Extract pairwise correlations
        lstm_hmm = correlations.xs('LSTM_Signal').HMM_Signal
        lstm_sma = correlations.xs('LSTM_Signal').SMA_Signal
        hmm_sma = correlations.xs('HMM_Signal').SMA_Signal
        
        ax2.plot(lstm_hmm.index, lstm_hmm, label='LSTM-HMM', color='#9b59b6')
        ax2.plot(lstm_sma.index, lstm_sma, label='LSTM-SMA', color='#f1c40f')
        ax2.plot(hmm_sma.index, hmm_sma, label='HMM-SMA', color='#1abc9c')
        
        ax2.set_title(f'{window}-Day Rolling Correlations Between Models', 
                     fontsize=14, pad=20)
        ax2.set_ylabel('Correlation', fontsize=12)
        ax2.legend(fontsize=10, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1, 1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_performance_metrics(self, save_path: Optional[str] = None):
        """Plot key performance metrics over time."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Rolling Sharpe Ratio (252-day)
        rolling_returns = self.signals['Strategy_Returns'].fillna(0)
        rolling_std = rolling_returns.rolling(252).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns.rolling(252).mean() * 252) / rolling_std
        
        ax1.plot(rolling_sharpe.index, rolling_sharpe, color='#2ecc71')
        ax1.set_title('Rolling Sharpe Ratio (252-day)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Volatility (252-day)
        rolling_vol = rolling_std * 100  # Convert to percentage
        ax2.plot(rolling_vol.index, rolling_vol, color='#e74c3c')
        ax2.set_title('Rolling Volatility (252-day)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Returns (252-day)
        rolling_annual_returns = (rolling_returns.rolling(252).mean() * 252 * 100)
        ax3.plot(rolling_annual_returns.index, rolling_annual_returns, color='#3498db')
        ax3.set_title('Rolling Annual Returns', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling Beta (252-day)
        market_returns = self.data['Returns'].fillna(0)
        rolling_cov = rolling_returns.rolling(252).cov(market_returns)
        rolling_market_var = market_returns.rolling(252).var()
        rolling_beta = rolling_cov / rolling_market_var
        
        ax4.plot(rolling_beta.index, rolling_beta, color='#9b59b6')
        ax4.set_title('Rolling Beta (252-day)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        self.data = data[['Close']].copy()
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data.dropna(inplace=True)
        return self.data
    
    def align_signals(self, 
                     lstm_signals: pd.Series, 
                     hmm_signals: pd.Series, 
                     sma_signals: pd.Series) -> pd.DataFrame:
        signals_df = pd.DataFrame(index=self.data.index)
        
        # Ensure all signals are Series with the correct index
        if isinstance(lstm_signals, pd.Series):
            signals_df['LSTM_Signal'] = lstm_signals
        else:
            signals_df['LSTM_Signal'] = pd.Series(lstm_signals, index=self.data.index)
            
        if isinstance(hmm_signals, pd.Series):
            signals_df['HMM_Signal'] = hmm_signals
        else:
            signals_df['HMM_Signal'] = pd.Series(hmm_signals, index=self.data.index)
            
        if isinstance(sma_signals, pd.Series):
            signals_df['SMA_Signal'] = sma_signals
        else:
            signals_df['SMA_Signal'] = pd.Series(sma_signals, index=self.data.index)
        
        # Forward fill any missing values
        signals_df = signals_df.fillna(method='ffill')
        
        return signals_df
    
    def normalize_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        for col in signals_df.columns:
            max_abs = max(abs(signals_df[col].min()), abs(signals_df[col].max()))
            if max_abs > 0:
                signals_df[col] = signals_df[col] / max_abs
        return signals_df
    
    def generate_combined_signal(self) -> pd.DataFrame:
        print("Generating signals from individual models...")
        
        # Get signals from each model
        try:
            lstm_signals, lstm_predictions, lstm_actuals = get_lstm_signal()
            # Ensure lstm_predictions is 1-dimensional
            if hasattr(lstm_predictions, 'flatten'):
                lstm_predictions = lstm_predictions.flatten()
            print("LSTM signals generated successfully")
        except Exception as e:
            print(f"Error generating LSTM signals: {e}")
            lstm_signals = pd.Series(0, index=self.data.index)
            lstm_predictions = np.zeros(len(self.data.index))
            
        try:
            hmm_data, hidden_states, bull_state, _ = get_hmm_signal()
            hmm_signals = pd.Series(np.where(hidden_states == bull_state, 1, -1), index=hmm_data.index)
            print("HMM signals generated successfully")
        except Exception as e:
            print(f"Error generating HMM signals: {e}")
            hmm_signals = pd.Series(0, index=self.data.index)
            
        try:
            sma_signals = get_sma_signal()
            print("SMA signals generated successfully")
        except Exception as e:
            print(f"Error generating SMA signals: {e}")
            sma_signals = pd.Series(0, index=self.data.index)
        
        # Align and normalize signals
        signals_df = self.align_signals(lstm_signals, hmm_signals, sma_signals)
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
            bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
            labels=['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        )
        
        # Calculate position sizes
        signals_df['Position_Size'] = signals_df['Combined_Signal'].abs()
        
        # Add expected returns - ensure proper alignment
        try:
            expected_returns_index = signals_df.index[-len(lstm_predictions):]
            signals_df['Expected_Return'] = np.nan
            signals_df.loc[expected_returns_index, 'Expected_Return'] = lstm_predictions
        except Exception as e:
            print(f"Warning: Could not add expected returns: {e}")
            signals_df['Expected_Return'] = np.nan
        
        self.signals = signals_df
        return signals_df
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        if self.signals is None:
            raise ValueError("Signals not generated yet. Run generate_combined_signal() first.")
            
        # Calculate strategy returns using the combined signal
        self.signals['Strategy_Returns'] = self.data['Returns'] * self.signals['Combined_Signal'].shift(1)
        
        # Calculate metrics
        total_return = (1 + self.signals['Strategy_Returns']).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(self.signals)) - 1
        volatility = self.signals['Strategy_Returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        max_drawdown = (self.signals['Strategy_Returns'].cumsum().expanding().max() - 
                       self.signals['Strategy_Returns'].cumsum()).max()
        
        # Calculate hit ratio
        correct_predictions = np.sum(np.sign(self.signals['Strategy_Returns']) == 
                                   np.sign(self.signals['Combined_Signal'].shift(1)))
        hit_ratio = correct_predictions / len(self.signals['Strategy_Returns'].dropna())
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Hit Ratio': hit_ratio
        }
    
    def run_strategy(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        print("Starting combined trading strategy...")
        self.get_market_data()
        signals = self.generate_combined_signal()
        metrics = self.calculate_performance_metrics()
        
        # Create visualizer and generate plots
        visualizer = StrategyVisualizer(signals, self.data, metrics)
        
        print("\nGenerating performance visualizations...")
        visualizer.plot_strategy_performance()
        visualizer.plot_model_contributions()
        visualizer.plot_performance_metrics()
        
        return signals, metrics

def main():
    # Initialize strategy
    strategy = CombinedTradingStrategy(
        symbol='^GSPC',
        start_date='2000-01-01',
        end_date='2023-12-31',
        weights={'lstm': 0.4, 'hmm': 0.3, 'sma': 0.3}
    )
    
    # Run strategy and generate visualizations
    signals, metrics = strategy.run_strategy()
    
    # Print results
    print("\nLatest Trading Signals:")
    print(signals[['Combined_Signal', 'Trading_Decision', 'Position_Size', 'Expected_Return']].tail())
    
    print("\nStrategy Performance:")
    for metric, value in metrics.items():
        if metric in ['Total Return', 'Annual Return', 'Annual Volatility', 'Max Drawdown', 'Hit Ratio']:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()