#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

logger = logging.getLogger(__name__)

class MomentumPortfolioSimulator:
    def __init__(self, initial_capital: float = 10000, position_size: float = 100, 
                 slope_window: int = 30, sigma_multiplier: float = 2.0, stiffness_threshold: float = None):
        """
        Initialize momentum portfolio simulator
        
        Args:
            initial_capital: Starting capital in USD
            position_size: Size of each position in USD
            slope_window: Rolling window size for slope distribution calculation
            sigma_multiplier: Multiplier for standard deviation threshold (e.g., 2.0 for mean + 2σ)
            stiffness_threshold: Threshold for doubling position size (e.g., 1.5σ)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.slope_window = slope_window
        self.sigma_multiplier = sigma_multiplier
        # Load stiffness_threshold from main.py if not provided
        if stiffness_threshold is None:
            try:
                import main
                self.stiffness_threshold = main.STIFFNESS_THRESHOLD
            except ImportError:
                from config import config
                self.stiffness_threshold = config.STIFFNESS_THRESHOLD
        else:
            self.stiffness_threshold = stiffness_threshold
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_stiffness(self, ema_slope: float, slope_threshold: float, 
                                   rolling_mean: float, rolling_std: float) -> float:
        """
        Calculate position stiffness - how many standard deviations above the threshold
        
        Args:
            ema_slope: Current EMA slope value
            slope_threshold: Current slope threshold (mean + sigma_multiplier * std)
            rolling_mean: Rolling mean of EMA slope
            rolling_std: Rolling standard deviation of EMA slope
            
        Returns:
            Stiffness value (how many std devs above threshold)
        """
        if rolling_std == 0:
            return 0.0
            
        # Calculate how many standard deviations above the threshold this slope is
        # threshold = mean + sigma_multiplier * std
        # stiffness = (ema_slope - threshold) / std
        stiffness = (ema_slope - slope_threshold) / rolling_std
        
        return stiffness
        
    def calculate_ema_difference_slope(self, data: pd.DataFrame, token_name: str, 
                                     short_period: int = 5, long_period: int = 20) -> pd.DataFrame:
        """
        Calculate EMA difference and its slope with adaptive thresholds
        
        Args:
            data: DataFrame with price and EMA data
            token_name: Name of the token
            short_period: Short EMA period in days
            long_period: Long EMA period in days
            
        Returns:
            DataFrame with EMA difference, slope, and adaptive thresholds
        """
        price_col = f'{token_name}_price'
        ema_short_col = f'{token_name}_ema_{short_period}d'
        ema_long_col = f'{token_name}_ema_{long_period}d'
        volume_ema_short_col = f'{token_name}_volume_ema_{short_period}d'
        volume_ema_long_col = f'{token_name}_volume_ema_{long_period}d'
        
        # Check if we have the required columns
        required_cols = [price_col, ema_short_col, ema_long_col]
        if not all(col in data.columns for col in required_cols):
            self.logger.warning(f"Missing required columns for {token_name}. Need: {required_cols}")
            return pd.DataFrame()
        
        # Calculate EMA difference (short - long)
        data[f'{token_name}_ema_difference'] = data[ema_short_col] - data[ema_long_col]
        
        # Calculate slope of EMA difference (first derivative)
        data[f'{token_name}_ema_slope'] = data[f'{token_name}_ema_difference'].diff()
        
        # Calculate rolling statistics for adaptive threshold
        rolling_mean = data[f'{token_name}_ema_slope'].rolling(window=self.slope_window, min_periods=1).mean()
        rolling_std = data[f'{token_name}_ema_slope'].rolling(window=self.slope_window, min_periods=1).std()
        
        # Calculate adaptive threshold: mean + σ_multiplier * std
        data[f'{token_name}_slope_threshold'] = rolling_mean + (self.sigma_multiplier * rolling_std)
        
        # Calculate volume EMA slope if available
        if volume_ema_short_col in data.columns and volume_ema_long_col in data.columns:
            data[f'{token_name}_volume_ema_difference'] = data[volume_ema_short_col] - data[volume_ema_long_col]
            data[f'{token_name}_volume_ema_slope'] = data[f'{token_name}_volume_ema_difference'].diff()
        else:
            data[f'{token_name}_volume_ema_slope'] = 0
        
        return data
    
    def calculate_momentum_signals(self, data: pd.DataFrame, token_name: str, 
                                 short_period: int = 5, long_period: int = 20) -> pd.DataFrame:
        """
        Calculate trading signals based on momentum (EMA difference slope) with adaptive thresholds
        
        Args:
            data: DataFrame with price, EMA, and slope data
            token_name: Name of the token
            short_period: Short EMA period in days
            long_period: Long EMA period in days
            
        Returns:
            DataFrame with trading signals
        """
        # Calculate EMA difference and slope
        data = self.calculate_ema_difference_slope(data, token_name, short_period, long_period)
        
        if data.empty:
            return pd.DataFrame()
        
        # Generate trading signals
        signals = []
        position_open = False
        entry_price = 0
        entry_date = None
        
        for idx, (i, row) in enumerate(data.iterrows()):
            ema_slope = row[f'{token_name}_ema_slope']
            slope_threshold = row[f'{token_name}_slope_threshold']
            volume_ema_slope = row[f'{token_name}_volume_ema_slope']
            current_price = row[f'{token_name}_price']
            
            # Get rolling statistics for stiffness calculation using integer position
            rolling_mean = data[f'{token_name}_ema_slope'].rolling(window=self.slope_window, min_periods=1).mean().iloc[idx]
            rolling_std = data[f'{token_name}_ema_slope'].rolling(window=self.slope_window, min_periods=1).std().iloc[idx]
            
            # Calculate position stiffness (how many std devs above threshold)
            stiffness = self.calculate_position_stiffness(
                ema_slope, slope_threshold, rolling_mean, rolling_std
            )
            
            # Determine position size multiplier based on stiffness
            position_multiplier = 2.0 if stiffness > self.stiffness_threshold else 1.0
            
            # Calculate position size with leverage
            from simple_trader import calculate_leverage_position_size
            leverage_info = calculate_leverage_position_size(
                token_name, 
                self.position_size, 
                position_multiplier
            )
            
            position_size_usd = leverage_info['position_size_usd']
            actual_leverage = leverage_info['leverage']
            
            signal = 'HOLD'
            pnl = 0
            
            # Entry signal: EMA Slope > Adaptive Threshold AND Volume EMA Slope > 0
            if not position_open and ema_slope > slope_threshold and volume_ema_slope > 0:
                signal = 'BUY'
                position_open = True
                entry_price = current_price
                entry_date = i
                
                # Log with stiffness and leverage information
                stiffness_info = f"Stiffness: {stiffness:.2f}σ above threshold"
                leverage_display = f"Leverage: {actual_leverage:.1f}x (Max: {leverage_info['max_leverage']}x)"
                multiplier_info = f"Position Size: ${position_size_usd:.2f} ({position_multiplier}x USD, {leverage_info['leverage']:.1f}x leverage)"
                
                self.logger.info(f"{token_name} BUY signal at {i}: Price=${current_price:.2f}, Slope={ema_slope:.6f}, Threshold={slope_threshold:.6f}, Vol Slope={volume_ema_slope:.6f}, {stiffness_info}, {leverage_display}, {multiplier_info}")
                
            
            
            # Exit signals
            elif position_open:
                # Exit signal 1: EMA Slope < 0 (momentum turning negative)
                if ema_slope < 0:
                    signal = 'SELL'
                    pnl = (current_price - entry_price) / entry_price * 100
                    position_open = False
                    self.logger.info(f"{token_name} SELL signal at {i}: Price=${current_price:.2f}, Slope={ema_slope:.6f}, PnL={pnl:.2f}%")
                    
                
                # Exit signal 2: Volume EMA Slope < 0 AND Volume EMA Difference < 10% (volume declining and low)
                elif volume_ema_slope < 0:
                    # Calculate volume EMA percentage
                    volume_ema_short_col = f'{token_name}_volume_ema_{short_period}d'
                    volume_ema_long_col = f'{token_name}_volume_ema_{long_period}d'
                    
                    if volume_ema_short_col in data.columns and volume_ema_long_col in data.columns:
                        volume_ema_percentage = (data.loc[i, volume_ema_short_col] / data.loc[i, volume_ema_long_col] - 1) * 100
                        
                        if volume_ema_percentage < 10.0:
                            signal = 'SELL_VOLUME'
                            pnl = (current_price - entry_price) / entry_price * 100
                            position_open = False
                            self.logger.info(f"{token_name} SELL_VOLUME signal at {i}: Price=${current_price:.2f}, Vol Slope={volume_ema_slope:.6f}, Vol EMA%={volume_ema_percentage:.2f}%, PnL={pnl:.2f}%")
                            
            
            signals.append({
                'date': i,
                'price': current_price,
                'ema_slope': ema_slope,
                'slope_threshold': slope_threshold,
                'volume_ema_slope': volume_ema_slope,
                'signal': signal,
                'position_open': position_open,
                'entry_price': entry_price if position_open else 0,
                'pnl': pnl,
                'stiffness': stiffness,
                'position_multiplier': position_multiplier,
                'position_size_usd': position_size_usd,
                'leverage': actual_leverage,
                'max_leverage': leverage_info['max_leverage']
            })
        
        return pd.DataFrame(signals)
    
    def simulate_momentum_portfolio(self, data: pd.DataFrame, token_names: List[str], 
                                  short_period: int = 5, long_period: int = 20) -> Dict:
        """
        Simulate momentum portfolio trading for all tokens
        
        Args:
            data: DataFrame with price and EMA data for all tokens
            token_names: List of token names to simulate
            short_period: Short EMA period in days
            long_period: Long EMA period in days
            
        Returns:
            Dictionary with simulation results
        """
        self.logger.info(f"Starting momentum portfolio simulation for {len(token_names)} tokens")
        self.logger.info(f"Short EMA period: {short_period}d, Long EMA period: {long_period}d")
        self.logger.info(f"Slope window: {self.slope_window}d, Sigma multiplier: {self.sigma_multiplier}")
        
        portfolio_results = {}
        all_trades = []
        total_pnl = 0
        total_trades = 0
        
        for token_name in token_names:
            self.logger.info(f"Simulating {token_name}...")
            
            # Calculate signals for this token
            signals_df = self.calculate_momentum_signals(data, token_name, short_period, long_period)
            
            if signals_df.empty:
                self.logger.warning(f"No signals calculated for {token_name}")
                continue
            
            # Extract trades
            trades = signals_df[signals_df['signal'].isin(['BUY', 'SELL', 'SELL_VOLUME'])]
            
            # Calculate token-specific metrics
            token_pnl = trades['pnl'].sum()
            token_trades = len(trades[trades['signal'].isin(['SELL', 'SELL_VOLUME'])])
            
            portfolio_results[token_name] = {
                'signals': signals_df,
                'trades': trades,
                'total_pnl': token_pnl,
                'total_trades': token_trades,
                'win_rate': len(trades[trades['pnl'] > 0]) / max(1, token_trades) * 100
            }
            
            total_pnl += token_pnl
            total_trades += token_trades
            
            # Add to all trades list
            for _, trade in trades.iterrows():
                all_trades.append({
                    'token': token_name,
                    'date': trade['date'],
                    'signal': trade['signal'],
                    'price': trade['price'],
                    'pnl': trade['pnl'],
                    'ema_slope': trade['ema_slope'],
                    'slope_threshold': trade['slope_threshold'],
                    'volume_ema_slope': trade['volume_ema_slope'],
                    'stiffness': trade.get('stiffness', 0.0),
                    'position_multiplier': trade.get('position_multiplier', 1.0),
                    'position_size_usd': trade.get('position_size_usd', 100),
                    'leverage': trade.get('leverage', 1.0)
                })
            
            self.logger.info(f"{token_name}: {token_trades} trades, PnL: {token_pnl:.2f}%")
        
        # Create summary
        summary = {
                'total_pnl': total_pnl,
                'total_trades': total_trades,
            'overall_win_rate': len([t for t in all_trades if t['pnl'] > 0]) / max(1, total_trades) * 100,
            'tokens_analyzed': len(portfolio_results),
            'all_trades': pd.DataFrame(all_trades),
            'stiffness_stats': {
                'normal_positions': len([t for t in all_trades if t.get('position_multiplier', 1.0) == 1.0 and t.get('signal') == 'BUY']),
                'double_positions': len([t for t in all_trades if t.get('position_multiplier', 1.0) == 2.0 and t.get('signal') == 'BUY']),
                'total_position_value': sum([t.get('position_size_usd', 100) for t in all_trades if t.get('signal') == 'BUY']),
                'avg_stiffness': np.mean([t.get('stiffness', 0.0) for t in all_trades if t.get('signal') == 'BUY']),
                'max_stiffness': max([t.get('stiffness', 0.0) for t in all_trades if t.get('signal') == 'BUY'], default=0.0),
                'stiffness_threshold': self.stiffness_threshold
            }
        }
        
        self.logger.info(f"Momentum portfolio simulation completed: {total_trades} total trades, {total_pnl:.2f}% total PnL")
        self.logger.info(f"Stiffness-based position sizing: {summary['stiffness_stats']['normal_positions']} normal, {summary['stiffness_stats']['double_positions']} double-size positions")
        self.logger.info(f"Average stiffness: {summary['stiffness_stats']['avg_stiffness']:.2f}σ, Max stiffness: {summary['stiffness_stats']['max_stiffness']:.2f}σ")
        self.logger.info(f"Total position value: ${summary['stiffness_stats']['total_position_value']:.2f}")
        self.logger.info(f"Stiffness threshold for double positions: >{summary['stiffness_stats']['stiffness_threshold']}σ")
        
        return {
            'portfolio_results': portfolio_results,
            'summary': summary
        }
    
    def plot_momentum_portfolio_results(self, simulation_results: Dict, save_dir: str,
                            short_period: int = 5, long_period: int = 20) -> None:
        """
        Plot momentum portfolio simulation results
        
        Args:
            simulation_results: Results from momentum portfolio simulation
            save_dir: Directory to save plots
            short_period: Short EMA period used
            long_period: Long EMA period used
        """
        self.logger.info("Creating momentum portfolio simulation plots...")
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        portfolio_results = simulation_results['portfolio_results']
        summary = simulation_results['summary']
        
        # Create subplots for each token
        num_tokens = len(portfolio_results)
        if num_tokens == 0:
            self.logger.warning("No portfolio results to plot")
            return
        
        cols = 4
        rows = (num_tokens + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'cyan', 'magenta']
        
        for i, (token_name, results) in enumerate(portfolio_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            signals_df = results['signals']
            
            # Debug: Log the exact type and structure
            self.logger.info(f"DEBUG: {token_name} signals type: {type(signals_df)}")
            if hasattr(signals_df, 'shape'):
                self.logger.info(f"DEBUG: {token_name} signals shape: {signals_df.shape}")
            if hasattr(signals_df, 'columns'):
                self.logger.info(f"DEBUG: {token_name} signals columns: {list(signals_df.columns)}")
            
            # Ensure signals_df is a DataFrame
            if not isinstance(signals_df, pd.DataFrame):
                if isinstance(signals_df, np.ndarray):
                    # Convert numpy array to DataFrame if needed
                    # The numpy array should have the same structure as the original DataFrame
                    self.logger.info(f"DEBUG: Converting numpy array for {token_name}, array shape: {signals_df.shape}")
                    signals_df = pd.DataFrame(signals_df, columns=['date', 'price', 'ema_slope', 'slope_threshold', 'volume_ema_slope', 'signal', 'position_open', 'entry_price', 'pnl', 'stiffness', 'position_multiplier', 'position_size_usd', 'leverage', 'max_leverage'])
                    self.logger.info(f"Converted numpy array to DataFrame for {token_name}")
                else:
                    self.logger.warning(f"Unexpected signals type for {token_name}: {type(signals_df)}")
                continue
            
            # Debug: Check the structure
            self.logger.info(f"Signals DataFrame for {token_name}: shape={signals_df.shape}, columns={list(signals_df.columns)}")
            
            # Debug: Check type right before plotting
            self.logger.info(f"DEBUG: Right before plotting - signals_df type: {type(signals_df)}")
            if hasattr(signals_df, 'plot'):
                self.logger.info("DEBUG: signals_df has .plot method")
            else:
                self.logger.info("DEBUG: signals_df does NOT have .plot method")
            
            # Debug: Check column types
            self.logger.info(f"DEBUG: date column type: {type(signals_df['date'])}")
            self.logger.info(f"DEBUG: price column type: {type(signals_df['price'])}")
            
            # Plot price
            ax.plot(signals_df['date'], signals_df['price'], label='Price', 
                   linewidth=1, alpha=0.7, color='black')
            
            # Plot buy/sell points
            buy_signals = signals_df[signals_df['signal'] == 'BUY']
            sell_signals = signals_df[signals_df['signal'].isin(['SELL', 'SELL_VOLUME'])]
            
            # Debug: Check if filtering converted to numpy array
            self.logger.info(f"DEBUG: buy_signals type: {type(buy_signals)}, sell_signals type: {type(sell_signals)}")
            
            if not buy_signals.empty:
                ax.scatter(buy_signals['date'], buy_signals['price'], 
                          color='green', marker='^', s=100, label='BUY', zorder=5)
            
            if not sell_signals.empty:
                ax.scatter(sell_signals['date'], sell_signals['price'], 
                          color='red', marker='v', s=100, label='SELL', zorder=5)
            
            # Add PnL information
            total_pnl = results['total_pnl']
            total_trades = results['total_trades']
            win_rate = results['win_rate']
            
            ax.set_title(f'{token_name}\nPnL: {total_pnl:.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Price (USD)', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
        
        # Hide empty subplots
        for i in range(len(portfolio_results), len(axes)):
            axes[i].set_visible(False)
        
        # Add global summary
        plt.suptitle(f'Momentum Portfolio Simulation Results - Last 3 Months\nShort EMA: {short_period}d, Long EMA: {long_period}d\nSlope Window: {self.slope_window}d, Sigma Multiplier: {self.sigma_multiplier}\nTotal PnL: {summary["total_pnl"]:.2f}% | Total Trades: {summary["total_trades"]} | Overall Win Rate: {summary["overall_win_rate"]:.1f}%\nStiffness-based Sizing: {summary["stiffness_stats"]["normal_positions"]} normal, {summary["stiffness_stats"]["double_positions"]} double-size positions\nAvg Stiffness: {summary["stiffness_stats"]["avg_stiffness"]:.2f}σ | Threshold: >{summary["stiffness_stats"]["stiffness_threshold"]}σ', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        portfolio_filename = f"{save_dir}/momentum_portfolio_simulation_latest.png"
        plt.savefig(portfolio_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Momentum portfolio simulation plot saved to: {portfolio_filename}")
        
        # Check if plots should be displayed
        try:
            from main import DISPLAY_PLOTS
            if DISPLAY_PLOTS:
                plt.show()
            else:
                plt.close()  # Close the plot to free memory
                self.logger.info("Portfolio simulation plot saved but not displayed (DISPLAY_PLOTS = False)")
        except ImportError:
            # If main.py is not available, default to displaying plots
            plt.show()
    
    def save_momentum_portfolio_results(self, simulation_results: Dict, save_path: str, 
                            short_period: int = 5, long_period: int = 20) -> None:
        """
        Save momentum portfolio simulation results to Excel
        
        Args:
            simulation_results: Results from momentum portfolio simulation
            save_path: Path to save Excel file
            short_period: Short EMA period used
            long_period: Long EMA period used
        """
        self.logger.info(f"Saving momentum portfolio results to: {save_path}")
        
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        try:
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                # Save summary
                summary = simulation_results['summary']
                summary_df = pd.DataFrame([{
                    'Metric': 'Data Period',
                    'Value': 'Last 3 Months (90 days)'
                }, {
                    'Metric': 'Short EMA Period',
                    'Value': f'{short_period} days'
                }, {
                    'Metric': 'Long EMA Period',
                    'Value': f'{long_period} days'
                }, {
                    'Metric': 'Slope Window',
                    'Value': f'{self.slope_window} days'
                }, {
                    'Metric': 'Sigma Multiplier',
                    'Value': self.sigma_multiplier
                }, {
                    'Metric': 'Total PnL (%)',
                    'Value': summary['total_pnl']
                }, {
                    'Metric': 'Total Trades',
                    'Value': summary['total_trades']
                }, {
                    'Metric': 'Overall Win Rate (%)',
                    'Value': summary['overall_win_rate']
                }, {
                    'Metric': 'Tokens Analyzed',
                    'Value': summary['tokens_analyzed']
                }])
                
                # Ensure summary data is not empty
                if not summary_df.empty:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    self.logger.info("Summary sheet saved successfully")
                else:
                    self.logger.warning("Summary data is empty, skipping Summary sheet")
                
                # Save all trades if available and not empty
                if 'all_trades' in summary and not summary['all_trades'].empty:
                    summary['all_trades'].to_excel(writer, sheet_name='All_Trades', index=False)
                    self.logger.info("All_Trades sheet saved successfully")
                else:
                    self.logger.warning("All trades data is empty or missing, skipping All_Trades sheet")
                
                # Save individual token results
                portfolio_results = simulation_results['portfolio_results']
                sheets_created = 0
                
                for token_name, results in portfolio_results.items():
                    # Save signals if available and not empty
                    if 'signals' in results and not results['signals'].empty:
                        sheet_name = f'{token_name}_Signals'
                        # Ensure sheet name is valid (Excel has restrictions)
                        if len(sheet_name) <= 31:  # Excel sheet name limit
                            results['signals'].to_excel(writer, sheet_name=sheet_name, index=False)
                            sheets_created += 1
                            self.logger.info(f"Signals sheet for {token_name} saved successfully")
                        else:
                            self.logger.warning(f"Sheet name too long for {token_name}_Signals, skipping")
                    else:
                        self.logger.warning(f"Signals data for {token_name} is empty or missing, skipping")
                    
                    # Save trades if available and not empty
                    if 'trades' in results and not results['trades'].empty:
                        sheet_name = f'{token_name}_Trades'
                        # Ensure sheet name is valid (Excel has restrictions)
                        if len(sheet_name) <= 31:  # Excel sheet name limit
                            results['trades'].to_excel(writer, sheet_name=sheet_name, index=False)
                            sheets_created += 1
                            self.logger.info(f"Trades sheet for {token_name} saved successfully")
                        else:
                            self.logger.warning(f"Sheet name too long for {token_name}_Trades, skipping")
                    else:
                        self.logger.warning(f"Trades data for {token_name} is empty or missing, skipping")
                
                # Ensure at least one sheet was created
                if sheets_created == 0:
                    self.logger.error("No valid sheets were created. Creating a minimal Summary sheet.")
                    # Create a minimal summary sheet to avoid Excel error
                    minimal_summary = pd.DataFrame([{
                        'Metric': 'Status',
                        'Value': 'Momentum portfolio simulation completed but no detailed data available'
                    }])
                    minimal_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                self.logger.info(f"Momentum portfolio results saved successfully to: {save_path}")
                self.logger.info(f"Total sheets created: {sheets_created + 1}")  # +1 for Summary sheet
                
        except Exception as e:
            self.logger.error(f"Error saving momentum portfolio results to Excel: {str(e)}")
            # Try to save a minimal file as backup
            try:
                backup_path = save_path.replace('.xlsx', '_backup.xlsx')
                minimal_data = pd.DataFrame([{
                    'Metric': 'Error',
                    'Value': f'Failed to save detailed results: {str(e)}'
                }])
                minimal_data.to_excel(backup_path, sheet_name='Error_Info', index=False)
                self.logger.info(f"Backup file saved to: {backup_path}")
            except Exception as backup_error:
                self.logger.error(f"Failed to create backup file: {str(backup_error)}")
            raise

def run_momentum_portfolio_simulation(db_path: str, save_dir: str = 'results', 
                                   short_period: int = 5, long_period: int = 20,
                                   slope_window: int = 30, sigma_multiplier: float = 2.0,
                                   position_size: float = 100, stiffness_threshold: float = None) -> Dict:
    """
    Run momentum portfolio simulation with adaptive slope strategy
    Data is filtered to only the last 3 months for simulation
    
    Args:
        db_path: Path to the database file
        save_dir: Directory to save results
        short_period: Short EMA period in days
        long_period: Long EMA period in days
        slope_window: Rolling window size for slope distribution calculation
        sigma_multiplier: Multiplier for standard deviation threshold (e.g., 2.0 for mean + 2σ)
        position_size: Base position size in USD for each trade
        stiffness_threshold: Threshold for doubling position size (e.g., 1.5σ)
        
    Returns:
        Dictionary with simulation results
    """
    logger.info("Starting Momentum Portfolio Simulation...")
    logger.info(f"Short EMA period: {short_period}d, Long EMA period: {long_period}d")
    logger.info(f"Slope window: {slope_window} days, Sigma multiplier: {sigma_multiplier}")
    logger.info("Filtering data to last 3 months for simulation...")
    
    try:
        # Load data from database
        from database_manager import CryptoDatabase
        db = CryptoDatabase(db_path)
        
        # Get all available tokens
        cursor = db.conn.cursor()
        cursor.execute('SELECT DISTINCT token_symbol FROM crypto_prices')
        tokens = [row[0] for row in cursor.fetchall()]
        
        if not tokens:
            logger.error("No tokens found in database")
            return None
        
        logger.info(f"Found {len(tokens)} tokens in database: {', '.join(tokens)}")
        
        # Load data for all tokens
        data = db.get_latest_data(tokens, days=365)
        
        if data.empty:
            logger.error("No data loaded from database")
            return None
        
        logger.info(f"Loaded {len(data)} data points from database")
        logger.info(f"Data range: {data.index.min()} to {data.index.max()}")
        
        # Calculate EMAs for all tokens since the EMA data table is empty
        logger.info("Calculating EMAs for momentum portfolio simulation...")
        try:
            from ema_analysis import EMAAnalyzer
            
            # Calculate EMAs for short period
            analyzer_short = EMAAnalyzer(short_period=short_period, long_period=long_period)
            ema_data = analyzer_short.calculate_emas_for_all_tokens(data)
            
            if ema_data.empty:
                logger.error("Failed to calculate EMAs - returned empty DataFrame")
                return None
            
            logger.info(f"EMAs calculated successfully. Data shape: {ema_data.shape}")
            logger.info(f"EMA columns: {list(ema_data.columns)}")
            
        except Exception as e:
            logger.error(f"Error calculating EMAs: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Filter data to only the last 3 months (90 days)
        # Use the maximum date from the data to get the most recent data
        max_date = ema_data.index.max()
        three_months_ago = max_date - pd.Timedelta(days=90)
        filtered_data = ema_data[ema_data.index >= three_months_ago]
        
        # Ensure all tokens have data in the filtered range
        # Remove tokens that don't have enough data points
        min_data_points = 30  # At least 30 data points for meaningful analysis
        valid_tokens = []
        for col in [col for col in filtered_data.columns if col.endswith('_price')]:
            token_name = col.replace('_price', '')
            token_data = filtered_data[col].dropna()
            if len(token_data) >= min_data_points:
                valid_tokens.append(token_name)
            else:
                logger.warning(f"Token {token_name} has only {len(token_data)} data points, skipping")
        
        if not valid_tokens:
            logger.error("No tokens have sufficient data for analysis")
            return None
        
        logger.info(f"Filtered to last 3 months: {len(filtered_data)} data points")
        logger.info(f"Filtered data range: {filtered_data.index.min()} to {filtered_data.index.max()}")
        
        # Use only valid tokens that have sufficient data
        token_names = valid_tokens
        
        if not token_names:
            logger.error("No valid tokens found for analysis")
            return None
        
        logger.info(f"Found {len(token_names)} valid tokens: {', '.join(token_names)}")
        
        # Validate that required EMA columns exist for each token
        for token_name in token_names:
            required_cols = [
                f'{token_name}_price',
                f'{token_name}_ema_{short_period}d',
                f'{token_name}_ema_{long_period}d'
            ]
            missing_cols = [col for col in required_cols if col not in filtered_data.columns]
            if missing_cols:
                logger.error(f"Missing required columns for {token_name}: {missing_cols}")
                return None
        
        logger.info("All required EMA columns validated successfully")
        
        # Load stiffness_threshold from main.py if not provided
        if stiffness_threshold is None:
            try:
                import main
                stiffness_threshold = main.STIFFNESS_THRESHOLD
                logger.info(f"Loaded stiffness_threshold from main.py: {stiffness_threshold}")
            except ImportError:
                from config import config
                stiffness_threshold = config.STIFFNESS_THRESHOLD
                logger.info(f"Loaded stiffness_threshold from config: {stiffness_threshold}")
        
        # Initialize simulator
        simulator = MomentumPortfolioSimulator(
            initial_capital=10000, 
            position_size=position_size,
            slope_window=slope_window,
            sigma_multiplier=sigma_multiplier,
            stiffness_threshold=stiffness_threshold
        )
        
        # Run simulation with filtered data
        simulation_results = simulator.simulate_momentum_portfolio(filtered_data, token_names, short_period, long_period)
        
        # Create plots
        simulator.plot_momentum_portfolio_results(simulation_results, save_dir, short_period, long_period)
        
        # Save results
        portfolio_filename = f"{save_dir}/momentum_portfolio_simulation_latest.xlsx"
        simulator.save_momentum_portfolio_results(simulation_results, portfolio_filename, short_period, long_period)
        
        logger.info("Momentum portfolio simulation completed successfully!")
        
        return simulation_results
        
    except Exception as e:
        logger.error(f"Error during momentum portfolio simulation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None 