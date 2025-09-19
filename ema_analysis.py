#!/usr/bin/env python3
"""
EMA Analysis module for Crypto Data
Calculates and plots Exponential Moving Averages for crypto tokens
"""

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from database_manager import CryptoDatabase

logger = logging.getLogger(__name__)

class EMAAnalyzer:
    """Class for calculating and analyzing Exponential Moving Averages"""
    
    def __init__(self, short_period: int = 10, long_period: int = 60):
        """
        Initialize EMA Analyzer
        
        Args:
            short_period: Short EMA period (default: 10 days)
            long_period: Long EMA period (default: 60 days)
        """
        self.short_period = short_period
        self.long_period = long_period
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_emas_for_token(self, price_data: pd.Series, volume_data: pd.Series, token_name: str) -> pd.DataFrame:
        """
        Calculate both short and long EMAs for price and volume of a token
        
        Args:
            price_data: Price series for the token
            volume_data: Volume series for the token
            token_name: Name of the token
            
        Returns:
            DataFrame with original price, volume, and EMAs for both
        """
        # Calculate price EMAs
        price_ema_short = self.calculate_ema(price_data, self.short_period)
        price_ema_long = self.calculate_ema(price_data, self.long_period)
        
        # Calculate volume EMAs
        volume_ema_short = self.calculate_ema(volume_data, self.short_period)
        volume_ema_long = self.calculate_ema(volume_data, self.long_period)
        
        # Create DataFrame
        result_df = pd.DataFrame({
            f'{token_name}_price': price_data,
            f'{token_name}_volume': volume_data,
            f'{token_name}_ema_{self.short_period}d': price_ema_short,
            f'{token_name}_ema_{self.long_period}d': price_ema_long,
            f'{token_name}_volume_ema_{self.short_period}d': volume_ema_short,
            f'{token_name}_volume_ema_{self.long_period}d': volume_ema_long
        })
        
        return result_df
    
    def calculate_ema_percentage(self, ema_short: pd.Series, ema_long: pd.Series) -> pd.Series:
        """
        Calculate the percentage of EMA_short over EMA_long
        
        Args:
            ema_short: Short EMA series
            ema_long: Long EMA series
            
        Returns:
            Series with percentage values
        """
        # Avoid division by zero
        ema_long_safe = ema_long.replace(0, np.nan)
        percentage = (ema_short / ema_long_safe - 1) * 100
        return percentage
    
    def calculate_emas_for_all_tokens(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMAs for all tokens in the dataset
        
        Args:
            data: DataFrame with price and volume data for all tokens
            
        Returns:
            DataFrame with original prices, volumes, and EMAs for all tokens
        """
        self.logger.info(f"Calculating EMAs for all tokens...")
        self.logger.info(f"Short period: {self.short_period} days, Long period: {self.long_period} days")
        
        # Get price columns
        price_columns = [col for col in data.columns if col.endswith('_price')]
        token_names = [col.replace('_price', '') for col in price_columns]
        
        self.logger.info(f"Found {len(token_names)} tokens: {', '.join(token_names)}")
        
        # Calculate EMAs for each token
        ema_dataframes = []
        
        for token_name in token_names:
            price_col = f'{token_name}_price'
            volume_col = f'{token_name}_volume'
            
            if price_col in data.columns and volume_col in data.columns:
                token_data = self.calculate_emas_for_token(data[price_col], data[volume_col], token_name)
                
                # Calculate price EMA percentage
                ema_short_col = f'{token_name}_ema_{self.short_period}d'
                ema_long_col = f'{token_name}_ema_{self.long_period}d'
                ema_percentage = self.calculate_ema_percentage(token_data[ema_short_col], token_data[ema_long_col])
                token_data[f'{token_name}_ema_percentage'] = ema_percentage
                
                # Calculate volume EMA percentage
                volume_ema_short_col = f'{token_name}_volume_ema_{self.short_period}d'
                volume_ema_long_col = f'{token_name}_volume_ema_{self.long_period}d'
                volume_ema_percentage = self.calculate_ema_percentage(token_data[volume_ema_short_col], token_data[volume_ema_long_col])
                token_data[f'{token_name}_volume_ema_percentage'] = volume_ema_percentage
                
                ema_dataframes.append(token_data)
                self.logger.info(f"Calculated price and volume EMAs for {token_name}")
            elif price_col in data.columns:
                # Fallback: only price data available
                token_data = self.calculate_emas_for_token(data[price_col], pd.Series([0]*len(data), index=data.index), token_name)
                
                # Calculate price EMA percentage
                ema_short_col = f'{token_name}_ema_{self.short_period}d'
                ema_long_col = f'{token_name}_ema_{self.long_period}d'
                ema_percentage = self.calculate_ema_percentage(token_data[ema_short_col], token_data[ema_long_col])
                token_data[f'{token_name}_ema_percentage'] = ema_percentage
                
                ema_dataframes.append(token_data)
                self.logger.info(f"Calculated price EMAs only for {token_name}")
            else:
                self.logger.warning(f"Price column not found for {token_name}")
        
        # Combine all data
        if ema_dataframes:
            combined_df = pd.concat(ema_dataframes, axis=1)
            combined_df = combined_df.sort_index()
            
            self.logger.info(f"EMA calculation completed. Data shape: {combined_df.shape}")
            return combined_df
        else:
            self.logger.error("No EMA data calculated")
            return pd.DataFrame()
    
    def plot_emas_for_token(self, data: pd.DataFrame, token_name: str, 
                           save_path: Optional[str] = None, include_volume: bool = False) -> None:
        """
        Plot EMAs for a specific token
        
        Args:
            data: DataFrame with price and EMA data
            token_name: Name of the token to plot
            save_path: Optional path to save the plot
            include_volume: If True, include volume data with dual y-axes
        """
        price_col = f'{token_name}_price'
        ema_short_col = f'{token_name}_ema_{self.short_period}d'
        ema_long_col = f'{token_name}_ema_{self.long_period}d'
        
        if not all(col in data.columns for col in [price_col, ema_short_col, ema_long_col]):
            self.logger.warning(f"Missing data columns for {token_name}")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price and EMAs on primary y-axis
        ax.plot(data.index, data[price_col], label=f'{token_name} Price', 
                linewidth=1, alpha=0.8, color='black')
        ax.plot(data.index, data[ema_short_col], 
                label=f'{token_name} EMA {self.short_period}d', 
                linewidth=2, color='blue')
        ax.plot(data.index, data[ema_long_col], 
                label=f'{token_name} EMA {self.long_period}d', 
                linewidth=2, color='red')
        
        # Add volume data if requested and available
        if include_volume:
            volume_col = f'{token_name}_volume'
            volume_ema_short_col = f'{token_name}_volume_ema_{self.short_period}d'
            volume_ema_long_col = f'{token_name}_volume_ema_{self.long_period}d'
            
            if all(col in data.columns for col in [volume_col, volume_ema_short_col, volume_ema_long_col]):
                # Create secondary y-axis for volume
                ax2 = ax.twinx()
                
                # Convert volume to billions for better readability
                volume_billions = data[volume_col] / 1e9
                ax2.bar(data.index, volume_billions, alpha=0.4, color='orange', 
                       label='Volume (B)', width=1)
                
                # Plot volume EMAs
                volume_ema_short_billions = data[volume_ema_short_col] / 1e9
                volume_ema_long_billions = data[volume_ema_long_col] / 1e9
                
                ax2.plot(data.index, volume_ema_short_billions, 
                       label=f'Vol EMA {self.short_period}d', 
                       linewidth=2, color='darkorange', linestyle='--')
                ax2.plot(data.index, volume_ema_long_billions, 
                       label=f'Vol EMA {self.long_period}d', 
                       linewidth=2, color='red', linestyle='--')
                
                # Set secondary y-axis properties
                ax2.set_ylabel('Volume (Billions USD)', fontsize=12, color='orange')
                ax2.tick_params(axis='y', labelsize=10, colors='orange')
                ax2.spines['right'].set_color('orange')
        
        # Get the last data point timestamp and values
        last_timestamp = data.index[-1]
        last_price = data[price_col].iloc[-1]
        
        # Add last data point annotation
        annotation_text = f'Last: {last_timestamp.strftime("%Y-%m-%d %H:%M")}\n${last_price:.2f}'
        
        if include_volume and volume_col in data.columns:
            last_volume = data[volume_col].iloc[-1]
            if last_volume >= 1e9:
                volume_text = f"Vol: {last_volume/1e9:.1f}B"
            elif last_volume >= 1e6:
                volume_text = f"Vol: {last_volume/1e6:.1f}M"
            else:
                volume_text = f"Vol: {last_volume/1e3:.1f}K"
            annotation_text += f'\n{volume_text}'
        
        ax.annotate(annotation_text, 
                    xy=(last_timestamp, last_price),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=10, ha='left', va='bottom')
        
        # Customize the plot
        ax.set_title(f'{token_name} Price and Exponential Moving Averages', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12, color='black')
        ax.tick_params(axis='y', labelsize=11, colors='black')
        ax.spines['left'].set_color('black')
        
        # Combine legends if volume is included
        if include_volume and volume_col in data.columns:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
        else:
            ax.legend(fontsize=11, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to: {save_path}")
        
        # Check if plots should be displayed
        try:
            from main import DISPLAY_PLOTS
            if DISPLAY_PLOTS:
                plt.show()
            else:
                plt.close()  # Close the plot to free memory
                self.logger.info("Plot saved but not displayed (DISPLAY_PLOTS = False)")
        except ImportError:
            # If main.py is not available, default to displaying plots
            plt.show()
    
    def plot_all_tokens_emas(self, data: pd.DataFrame, 
                            save_dir: str = 'results', use_fixed_filename: bool = False) -> None:
        """
        Plot EMAs for all tokens in a single combined subplot
        
        Args:
            data: DataFrame with price and EMA data for all tokens
            save_dir: Directory to save the plot
            use_fixed_filename: If True, use fixed filename instead of timestamp
        """
        self.logger.info("Creating combined EMA plot for all tokens...")
        
        # Get token names
        price_columns = [col for col in data.columns if col.endswith('_price')]
        token_names = [col.replace('_price', '') for col in price_columns]
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create combined plot
        self.create_combined_plot(data, token_names, save_dir, use_fixed_filename)
        
        # Create percentage difference plots
        self.create_percentage_plots(data, token_names, save_dir, use_fixed_filename)
        
        self.logger.info(f"Combined EMA plot saved to: {save_dir}")
    
    def create_combined_plot(self, data: pd.DataFrame, token_names: List[str], 
                           save_dir: str, use_fixed_filename: bool = False) -> None:
        """
        Create a combined plot showing all tokens' EMAs (price only) for the last two months
        
        Args:
            data: DataFrame with price, EMA, and volume data
            token_names: List of token names
            save_dir: Directory to save the plot
            use_fixed_filename: If True, use fixed filename instead of timestamp
        """
        # Filter data to only the last two months
        two_months_ago = data.index.max() - pd.Timedelta(days=60)
        filtered_data = data[data.index >= two_months_ago]
        
        self.logger.info(f"Creating combined plot with data from {filtered_data.index.min()} to {filtered_data.index.max()}")
        
        # Create subplots for all tokens (3 rows, 4 columns for 12 tokens)
        num_tokens = len(token_names)
        cols = 4
        rows = (num_tokens + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'cyan', 'magenta', 
                 'lime', 'navy', 'olive', 'teal']
        
        for i, token_name in enumerate(token_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            price_col = f'{token_name}_price'
            ema_short_col = f'{token_name}_ema_{self.short_period}d'
            ema_long_col = f'{token_name}_ema_{self.long_period}d'
            
            if all(col in filtered_data.columns for col in [price_col, ema_short_col, ema_long_col]):
                # Plot price and EMAs on primary y-axis only
                ax.plot(filtered_data.index, filtered_data[price_col], label='Price', 
                       linewidth=1, alpha=0.7, color='black')
                ax.plot(filtered_data.index, filtered_data[ema_short_col], 
                       label=f'EMA {self.short_period}d', 
                       linewidth=2, color=colors[i])
                ax.plot(filtered_data.index, filtered_data[ema_long_col], 
                       label=f'EMA {self.long_period}d', 
                       linewidth=2, color='darkred')
                
                # Get the last data point timestamp and values
                last_timestamp = filtered_data.index[-1]
                last_price = filtered_data[price_col].iloc[-1]
                last_ema_short = filtered_data[ema_short_col].iloc[-1]
                last_ema_long = filtered_data[ema_long_col].iloc[-1]
                
                # Calculate percentage of EMA_short over EMA_long
                if last_ema_long != 0:
                    ema_percentage = (last_ema_short / last_ema_long - 1) * 100
                    percentage_text = f"EMA%: {ema_percentage:+.2f}%"
                else:
                    percentage_text = "EMA%: N/A"
                
                # Add last data point annotation with price EMA percentage only
                ax.annotate(f'Last: {last_timestamp.strftime("%Y-%m-%d %H:%M")}\n${last_price:.2f}\n{percentage_text}', 
                           xy=(last_timestamp, last_price),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           fontsize=8, ha='left', va='bottom')
                
                ax.set_title(f'{token_name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Price (USD)', fontsize=10)
                
                # Simple legend for price and EMAs only
                ax.legend(fontsize=8, loc='upper left')
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.tick_params(axis='y', labelsize=9)
        
        # Hide empty subplots
        for i in range(len(token_names), len(axes)):
            axes[i].set_visible(False)
        
        # Add global timestamp annotation
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.suptitle(f'Crypto Token Price and EMA Analysis - Last 2 Months ({self.short_period}d and {self.long_period}d)\nAnalysis performed: {analysis_time}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save combined plot
        if use_fixed_filename:
            combined_filename = f"{save_dir}/combined_ema_analysis_latest.png"
        else:
            combined_filename = f"{save_dir}/combined_ema_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Combined plot saved to: {combined_filename}")
        
        # Check if plots should be displayed
        try:
            from main import DISPLAY_PLOTS
            if DISPLAY_PLOTS:
                plt.show()
            else:
                plt.close()  # Close the plot to free memory
                self.logger.info("Combined plot saved but not displayed (DISPLAY_PLOTS = False)")
        except ImportError:
            # If main.py is not available, default to displaying plots
            plt.show()
    
    def create_percentage_plots(self, data: pd.DataFrame, token_names: List[str], 
                              save_dir: str, use_fixed_filename: bool = False) -> None:
        """
        Create plots showing percentage differences between EMA short and long for price and volume
        (Last 30 days with trend arrows)
        
        Args:
            data: DataFrame with price, EMA, and volume data
            token_names: List of token names
            save_dir: Directory to save the plot
            use_fixed_filename: If True, use fixed filename instead of timestamp
        """
        self.logger.info("Creating percentage difference plots (last 30 days)...")
        
        # Get last 30 days of data
        last_30_days = data.tail(30)
        
        # Create subplots for all tokens (3 rows, 4 columns for 12 tokens)
        num_tokens = len(token_names)
        cols = 4
        rows = (num_tokens + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'red', 'cyan', 'magenta', 
                 'lime', 'navy', 'olive', 'teal']
        
        for i, token_name in enumerate(token_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            price_col = f'{token_name}_price'
            ema_short_col = f'{token_name}_ema_{self.short_period}d'
            ema_long_col = f'{token_name}_ema_{self.long_period}d'
            volume_ema_short_col = f'{token_name}_volume_ema_{self.short_period}d'
            volume_ema_long_col = f'{token_name}_volume_ema_{self.long_period}d'
            
            if all(col in data.columns for col in [price_col, ema_short_col, ema_long_col]):
                # Calculate percentage differences for last 30 days
                price_percentage = (last_30_days[ema_short_col] / last_30_days[ema_long_col] - 1) * 100
                
                # Plot price EMA percentage on primary y-axis
                ax.plot(last_30_days.index, price_percentage, 
                       label=f'Price EMA% ({self.short_period}d vs {self.long_period}d)', 
                       linewidth=2, color=colors[i])
                
                # Add volume EMA percentage if available on secondary y-axis
                if volume_ema_short_col in data.columns and volume_ema_long_col in data.columns:
                    # Create secondary y-axis for volume
                    ax2 = ax.twinx()
                    
                    volume_percentage = (last_30_days[volume_ema_short_col] / last_30_days[volume_ema_long_col] - 1) * 100
                    ax2.plot(last_30_days.index, volume_percentage, 
                           label=f'Volume EMA% ({self.short_period}d vs {self.long_period}d)', 
                           linewidth=2, color='orange', linestyle='--')
                    
                    # Set secondary y-axis properties
                    ax2.set_ylabel('Volume EMA % Difference', fontsize=9, color='orange')
                    ax2.tick_params(axis='y', labelsize=8, colors='orange')
                    ax2.spines['right'].set_color('orange')
                
                # Add horizontal lines for trading signals
                # Price signals on primary axis
                ax.axhline(y=1.0, color='green', linestyle='-', alpha=0.5, label='Price BUY Signal (1%)')
                ax.axhline(y=-1.0, color='red', linestyle='-', alpha=0.5, label='Price SELL Signal (-1%)')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Price Neutral (0%)')
                
                # Volume signals on secondary axis if available
                if volume_ema_short_col in data.columns and volume_ema_long_col in data.columns:
                    ax2.axhline(y=20.0, color='darkgreen', linestyle=':', alpha=0.5, label='Volume BUY (20%)')
                    ax2.axhline(y=-60.0, color='darkred', linestyle=':', alpha=0.5, label='Volume SELL (-60%)')
                    ax2.axhline(y=0, color='orange', linestyle='-', alpha=0.3, label='Volume Neutral (0%)')
                
                # Get current values for annotation
                current_price_percentage = price_percentage.iloc[-1]
                current_volume_percentage = volume_percentage.iloc[-1] if volume_ema_short_col in data.columns and volume_ema_long_col in data.columns else None
                
                # Calculate trend direction (last 5 days vs previous 5 days)
                if len(price_percentage) >= 10:
                    recent_trend = price_percentage.iloc[-5:].mean() - price_percentage.iloc[-10:-5].mean()
                    trend_arrow = "UP" if recent_trend > 0 else "DOWN" if recent_trend < 0 else "FLAT"
                    trend_text = "UP" if recent_trend > 0 else "DOWN" if recent_trend < 0 else "SIDE"
                else:
                    trend_arrow = "FLAT"
                    trend_text = "SIDE"
                
                # Get last data point timestamp
                last_timestamp = last_30_days.index[-1]
                
                # Add current values annotation with trend and timestamp
                annotation_text = f'Last: {last_timestamp.strftime("%Y-%m-%d %H:%M")}\nPrice EMA%: {current_price_percentage:+.2f}%\nTrend: {trend_text} {trend_arrow}'
                if current_volume_percentage is not None:
                    annotation_text += f'\nVol EMA%: {current_volume_percentage:+.2f}%'
                
                # Position annotation based on which axis has the larger range
                if current_volume_percentage is not None:
                    # Use volume percentage for y-position if available
                    ax.annotate(annotation_text, 
                               xy=(last_30_days.index[-1], current_volume_percentage),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               fontsize=9, ha='left', va='bottom')
                else:
                    # Use price percentage if no volume data
                    ax.annotate(annotation_text, 
                               xy=(last_30_days.index[-1], current_price_percentage),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               fontsize=9, ha='left', va='bottom')
                
                ax.set_title(f'{token_name} - EMA Percentage Differences (Last 30 Days)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Price EMA % Difference', fontsize=10, color=colors[i])
                ax.tick_params(axis='y', labelsize=9, colors=colors[i])
                ax.spines['left'].set_color(colors[i])
                
                # Combine legends from both axes
                lines1, labels1 = ax.get_legend_handles_labels()
                if volume_ema_short_col in data.columns and volume_ema_long_col in data.columns:
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
                else:
                    ax.legend(fontsize=8, loc='upper left')
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        # Hide empty subplots
        for i in range(len(token_names), len(axes)):
            axes[i].set_visible(False)
        
        # Add global title
        analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.suptitle(f'EMA Percentage Differences Analysis ({self.short_period}d vs {self.long_period}d)\nAnalysis performed: {analysis_time}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save percentage plot
        if use_fixed_filename:
            percentage_filename = f"{save_dir}/ema_percentage_analysis_latest.png"
        else:
            percentage_filename = f"{save_dir}/ema_percentage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(percentage_filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Percentage analysis plot saved to: {percentage_filename}")
        
        # Check if plots should be displayed
        try:
            from main import DISPLAY_PLOTS
            if DISPLAY_PLOTS:
                plt.show()
            else:
                plt.close()  # Close the plot to free memory
                self.logger.info("Percentage analysis plot saved but not displayed (DISPLAY_PLOTS = False)")
        except ImportError:
            # If main.py is not available, default to displaying plots
            plt.show()
    
    def save_ema_data(self, data: pd.DataFrame, save_path: str) -> None:
        """
        Save EMA data to Excel file with multiple sheets
        
        Args:
            data: DataFrame with price and EMA data
            save_path: Path to save the Excel file
        """
        self.logger.info(f"Saving EMA data to Excel file: {save_path}")
        
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # Save all data
            data.to_excel(writer, sheet_name='All_Data')
            self.logger.info(f"All data saved to sheet: All_Data")
            
            # Create separate sheets for prices and EMAs
            price_columns = [col for col in data.columns if col.endswith('_price')]
            ema_short_columns = [col for col in data.columns if f'_ema_{self.short_period}d' in col]
            ema_long_columns = [col for col in data.columns if f'_ema_{self.long_period}d' in col]
            ema_percentage_columns = [col for col in data.columns if '_ema_percentage' in col]
            
            if price_columns:
                price_df = data[price_columns]
                price_df.to_excel(writer, sheet_name='Prices')
                self.logger.info(f"Prices saved to sheet: Prices")
            
            if ema_short_columns:
                ema_short_df = data[ema_short_columns]
                ema_short_df.to_excel(writer, sheet_name=f'EMA_{self.short_period}d')
                self.logger.info(f"EMA {self.short_period}d saved to sheet: EMA_{self.short_period}d")
            
            if ema_long_columns:
                ema_long_df = data[ema_long_columns]
                ema_long_df.to_excel(writer, sheet_name=f'EMA_{self.long_period}d')
                self.logger.info(f"EMA {self.long_period}d saved to sheet: EMA_{self.long_period}d")
            
            if ema_percentage_columns:
                ema_percentage_df = data[ema_percentage_columns]
                ema_percentage_df.to_excel(writer, sheet_name='EMA_Percentages')
                self.logger.info(f"EMA Percentages saved to sheet: EMA_Percentages")
            
            # Create summary sheet
            summary_data = []
            for col in price_columns:
                token_name = col.replace('_price', '')
                ema_short_col = f'{token_name}_ema_{self.short_period}d'
                ema_long_col = f'{token_name}_ema_{self.long_period}d'
                
                if all(col_name in data.columns for col_name in [col, ema_short_col, ema_long_col]):
                    current_price = data[col].iloc[-1]
                    current_ema_short = data[ema_short_col].iloc[-1]
                    current_ema_long = data[ema_long_col].iloc[-1]
                    
                    # Get current EMA percentage
                    ema_percentage_col = f'{token_name}_ema_percentage'
                    current_ema_percentage = data[ema_percentage_col].iloc[-1] if ema_percentage_col in data.columns else None
                    
                    # Calculate crossover signals
                    ema_short_series = data[ema_short_col]
                    ema_long_series = data[ema_long_col]
                    
                    # Bullish crossover (short EMA crosses above long EMA)
                    bullish_crossovers = ((ema_short_series > ema_long_series) & 
                                        (ema_short_series.shift(1) <= ema_long_series.shift(1))).sum()
                    
                    # Bearish crossover (short EMA crosses below long EMA)
                    bearish_crossovers = ((ema_short_series < ema_long_series) & 
                                        (ema_short_series.shift(1) >= ema_long_series.shift(1))).sum()
                    
                    summary_data.append({
                        'Token': token_name,
                        'Current_Price': current_price,
                        f'Current_EMA_{self.short_period}d': current_ema_short,
                        f'Current_EMA_{self.long_period}d': current_ema_long,
                        'Current_EMA_Percentage': f"{current_ema_percentage:+.2f}%" if current_ema_percentage is not None else "N/A",
                        'Price_vs_EMA_Short': 'Above' if current_price > current_ema_short else 'Below',
                        'Price_vs_EMA_Long': 'Above' if current_price > current_ema_long else 'Below',
                        'EMA_Short_vs_Long': 'Above' if current_ema_short > current_ema_long else 'Below',
                        'Bullish_Crossovers': bullish_crossovers,
                        'Bearish_Crossovers': bearish_crossovers,
                        'Data_Points': len(data)
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            self.logger.info(f"Summary saved to sheet: Summary")
        
        self.logger.info(f"All EMA data saved successfully to Excel file: {save_path}")

def run_ema_analysis(data_source: str, results_dir: str = 'results', 
                    short_period: int = 10, long_period: int = 60, 
                    db_path: str = "crypto_data.db") -> Dict:
    """
    Main function to run EMA analysis
    
    Args:
        data_source: Path to the Excel file with crypto data or 'database' for SQLite
        results_dir: Directory to save results
        short_period: Short EMA period (default: 10 days)
        long_period: Long EMA period (default: 60 days)
        db_path: Path to SQLite database (if using database)
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting EMA Analysis...")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Load data from source
        if data_source == 'database':
            logger.info(f"Loading data from database: {db_path}")
            db = CryptoDatabase(db_path)
            # Get all available tokens from database
            cursor = db.conn.cursor()
            cursor.execute('SELECT DISTINCT token_symbol FROM crypto_prices ORDER BY token_symbol')
            token_symbols = [row[0] for row in cursor.fetchall()]
            
            if not token_symbols:
                logger.error("No tokens found in database")
                return {}
            
            data = db.get_latest_data(token_symbols, days=365)
            db.close_connection()
            
            if data.empty:
                logger.error("No data found in database")
                return {}
                
            logger.info(f"Data loaded successfully from database. Shape: {data.shape}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"Tokens found: {', '.join(token_symbols)}")
            
        else:
            # Load from Excel file (legacy support)
            logger.info(f"Loading data from Excel file: {data_source}")
            data = pd.read_excel(data_source, sheet_name='Daily_Data', index_col=0)
            data.index = pd.to_datetime(data.index)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Initialize EMA analyzer
        analyzer = EMAAnalyzer(short_period=short_period, long_period=long_period)
        
        # Calculate EMAs
        ema_data = analyzer.calculate_emas_for_all_tokens(data)
        
        if ema_data.empty:
            logger.error("No EMA data calculated")
            return {}
        
        # Save EMA data with fixed filename
        ema_filename = f"{results_dir}/ema_analysis_latest.xlsx"
        analyzer.save_ema_data(ema_data, ema_filename)
        
        # Store EMA data in database if using database
        if data_source == 'database':
            logger.info("Storing EMA data in database...")
            db = CryptoDatabase(db_path)
            
            # Get token symbols from column names
            token_symbols = []
            for col in ema_data.columns:
                if col.endswith('_price'):
                    token_symbol = col.replace('_price', '')
                    token_symbols.append(token_symbol)
            
            # Store EMA data for each token
            for token_symbol in token_symbols:
                ema_short_col = f'{token_symbol}_ema_{short_period}d'
                ema_long_col = f'{token_symbol}_ema_{long_period}d'
                ema_percentage_col = f'{token_symbol}_ema_percentage'
                
                if all(col in ema_data.columns for col in [ema_short_col, ema_long_col, ema_percentage_col]):
                    # Create EMA dataframe for this token
                    ema_df = pd.DataFrame({
                        ema_short_col: ema_data[ema_short_col],
                        ema_long_col: ema_data[ema_long_col],
                        ema_percentage_col: ema_data[ema_percentage_col]
                    })
                    
                    db.update_ema_data(token_symbol, ema_df)
                    
                    # Calculate and store summary data
                    current_price = ema_data[f'{token_symbol}_price'].iloc[-1]
                    current_ema_short = ema_data[ema_short_col].iloc[-1]
                    current_ema_long = ema_data[ema_long_col].iloc[-1]
                    current_ema_percentage = ema_data[ema_percentage_col].iloc[-1]
                    
                    # Calculate crossovers
                    ema_short_series = ema_data[ema_short_col]
                    ema_long_series = ema_data[ema_long_col]
                    
                    bullish_crossovers = ((ema_short_series > ema_long_series) & 
                                        (ema_short_series.shift(1) <= ema_long_series.shift(1))).sum()
                    
                    bearish_crossovers = ((ema_short_series < ema_long_series) & 
                                        (ema_short_series.shift(1) >= ema_long_series.shift(1))).sum()
                    
                    summary_data = {
                        'current_price': current_price,
                        'current_ema_short': current_ema_short,
                        'current_ema_long': current_ema_long,
                        'current_ema_percentage': current_ema_percentage,
                        'bullish_crossovers': bullish_crossovers,
                        'bearish_crossovers': bearish_crossovers,
                        'data_points': len(ema_data)
                    }
                    
                    db.update_analysis_summary(token_symbol, summary_data)
            
            db.close_connection()
            logger.info("EMA data stored in database successfully")
        
        # Create plots with fixed filename
        analyzer.plot_all_tokens_emas(ema_data, results_dir, use_fixed_filename=True)
        
        logger.info("EMA Analysis completed successfully!")
        
        return {
            'ema_data': ema_data,
            'excel_file': ema_filename,
            'results_dir': results_dir,
            'short_period': short_period,
            'long_period': long_period,
            'data_shape': ema_data.shape,
            'tokens_analyzed': len([col for col in ema_data.columns if col.endswith('_price')]),
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'short_ema_period': short_period,
                'long_ema_period': long_period,
                'data_source': data_source
            }
        }
        
    except Exception as e:
        logger.error(f"Error during EMA analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the EMA analysis
    logging.basicConfig(level=logging.INFO)
    
    # You can test with a sample data file
    # results = run_ema_analysis("data/specific_crypto_data_YYYYMMDD_HHMMSS.xlsx")
    print("EMA Analysis module loaded successfully!")
    print("Use run_ema_analysis() function to analyze your crypto data.") 