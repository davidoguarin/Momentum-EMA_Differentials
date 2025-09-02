#!/usr/bin/env python3
"""
Data Manager for handling Excel files and data persistence
Handles the GitHub Actions workflow issue where files are committed after code runs
"""

import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data files and handles GitHub Actions workflow timing"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def get_latest_excel_file(self, pattern="*.xlsx"):
        """
        Get the latest Excel file matching the pattern
        
        Args:
            pattern: File pattern to search for (e.g., "momentum_portfolio_simulation_*.xlsx")
        
        Returns:
            str: Path to the latest file, or None if not found
        """
        try:
            # Search in results directory
            search_pattern = str(self.results_dir / pattern)
            files = glob.glob(search_pattern)
            
            if not files:
                logger.warning(f"No files found matching pattern: {pattern}")
                return None
            
            # Sort by modification time (newest first)
            latest_file = max(files, key=os.path.getmtime)
            logger.info(f"📁 Using latest file: {latest_file}")
            return latest_file
            
        except Exception as e:
            logger.error(f"Error finding latest Excel file: {e}")
            return None
    
    def get_portfolio_results_file(self):
        """Get the latest portfolio simulation results file"""
        return self.get_latest_excel_file("momentum_portfolio_simulation_*.xlsx")
    
    def get_trading_orders_file(self):
        """Get the latest trading orders file"""
        orders_file = "trading_orders.xlsx"
        if os.path.exists(orders_file):
            return orders_file
        
        # Look for timestamped versions
        return self.get_latest_excel_file("trading_orders_*.xlsx")
    
    def load_portfolio_results(self):
        """
        Load portfolio simulation results from Excel file
        
        Returns:
            dict: Portfolio results data
        """
        file_path = self.get_portfolio_results_file()
        
        if not file_path:
            logger.warning("No portfolio results file found, will generate new data")
            return None
        
        try:
            # Load the Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            # Extract relevant data
            results = {
                'summary': excel_data.get('Summary', pd.DataFrame()),
                'all_trades': excel_data.get('All_Trades', pd.DataFrame()),
                'file_path': file_path,
                'file_timestamp': os.path.getmtime(file_path)
            }
            
            logger.info(f"✅ Loaded portfolio results from {file_path}")
            logger.info(f"📊 Found {len(results['all_trades'])} trades in historical data")
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading portfolio results: {e}")
            return None
    
    def load_historical_data_in_memory(self):
        """
        Load historical data and keep it in memory for calculations
        
        Returns:
            dict: Historical data in memory format
        """
        historical_data = {
            'portfolio_results': None,
            'trading_orders': None,
            'crypto_data': None,
            'parameters': {}
        }
        
        # Load portfolio results
        portfolio_file = self.get_portfolio_results_file()
        if portfolio_file:
            try:
                excel_data = pd.read_excel(portfolio_file, sheet_name=None)
                historical_data['portfolio_results'] = {
                    'summary': excel_data.get('Summary', pd.DataFrame()),
                    'all_trades': excel_data.get('All_Trades', pd.DataFrame()),
                    'signals': {}  # Will store signals by token
                }
                
                # Extract signals for each token
                for sheet_name in excel_data.keys():
                    if sheet_name.startswith('Signals_'):
                        token = sheet_name.replace('Signals_', '')
                        historical_data['portfolio_results']['signals'][token] = excel_data[sheet_name]
                
                logger.info(f"✅ Loaded historical portfolio data: {len(historical_data['portfolio_results']['all_trades'])} trades")
            except Exception as e:
                logger.error(f"Error loading historical portfolio data: {e}")
        
        # Load trading orders
        orders_file = self.get_trading_orders_file()
        if orders_file:
            try:
                historical_data['trading_orders'] = pd.read_excel(orders_file)
                logger.info(f"✅ Loaded historical trading orders: {len(historical_data['trading_orders'])} orders")
            except Exception as e:
                logger.error(f"Error loading historical trading orders: {e}")
        
        # Load crypto data
        crypto_file = self.get_latest_excel_file("specific_crypto_data_*.xlsx")
        if crypto_file:
            try:
                historical_data['crypto_data'] = pd.read_excel(crypto_file)
                logger.info(f"✅ Loaded historical crypto data: {len(historical_data['crypto_data'])} records")
            except Exception as e:
                logger.error(f"Error loading historical crypto data: {e}")
        
        # Extract parameters from historical data
        if historical_data['portfolio_results'] and not historical_data['portfolio_results']['summary'].empty:
            summary = historical_data['portfolio_results']['summary']
            if 'Parameter' in summary.columns and 'Value' in summary.columns:
                for _, row in summary.iterrows():
                    param_name = row['Parameter']
                    param_value = row['Value']
                    historical_data['parameters'][param_name] = param_value
                logger.info(f"✅ Extracted {len(historical_data['parameters'])} parameters from historical data")
        
        return historical_data
    
    def load_trading_orders(self):
        """
        Load trading orders from Excel file
        
        Returns:
            pd.DataFrame: Trading orders data
        """
        file_path = self.get_trading_orders_file()
        
        if not file_path:
            logger.info("No trading orders file found, starting fresh")
            return pd.DataFrame()
        
        try:
            orders_df = pd.read_excel(file_path)
            logger.info(f"✅ Loaded {len(orders_df)} trading orders from {file_path}")
            return orders_df
            
        except Exception as e:
            logger.error(f"Error loading trading orders: {e}")
            return pd.DataFrame()
    
    def should_use_historical_data(self):
        """
        Determine if we should use historical data or generate new data
        
        Returns:
            bool: True if historical data should be used
        """
        # Always use historical data as base, then fetch new data
        logger.info("📊 Using historical data as base, fetching new data for updates")
        return True
    
    def get_data_strategy(self):
        """
        Get the data strategy based on environment and available files
        
        Returns:
            dict: Data strategy configuration
        """
        strategy = {
            'use_historical': self.should_use_historical_data(),
            'extract_data': True,  # Always extract new data
            'run_analysis': True,  # Always run analysis with combined data
            'run_simulation': True  # Always run simulation with combined data
        }
        
        logger.info(f"📋 Data strategy: {strategy}")
        return strategy

def get_data_manager():
    """Get a DataManager instance"""
    return DataManager()

# Convenience functions
def should_use_historical_data():
    """Check if historical data should be used"""
    dm = DataManager()
    return dm.should_use_historical_data()

def get_latest_portfolio_file():
    """Get the latest portfolio results file"""
    dm = DataManager()
    return dm.get_portfolio_results_file()

def load_historical_portfolio_results():
    """Load historical portfolio results"""
    dm = DataManager()
    return dm.load_portfolio_results()

if __name__ == "__main__":
    # Test the data manager
    dm = DataManager()
    
    print("🧪 Testing Data Manager...")
    print(f"📁 Results directory: {dm.results_dir}")
    
    # Test file detection
    portfolio_file = dm.get_portfolio_results_file()
    print(f"📊 Portfolio file: {portfolio_file}")
    
    orders_file = dm.get_trading_orders_file()
    print(f"📋 Orders file: {orders_file}")
    
    # Test data strategy
    strategy = dm.get_data_strategy()
    print(f"📋 Data strategy: {strategy}")
    
    # Test historical data loading
    if portfolio_file:
        results = dm.load_portfolio_results()
        if results:
            print(f"✅ Loaded {len(results['all_trades'])} historical trades")
        else:
            print("❌ Failed to load historical data")
    else:
        print("ℹ️ No historical data available")