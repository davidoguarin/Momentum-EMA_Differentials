#!/usr/bin/env python3
"""
Main script for Crypto EMA Analysis
Downloads crypto tokens and performs EMA analysis with momentum strategy
"""

import sys
import logging
import os
import glob
from datetime import datetime

# Import our custom modules
from data_acquisition import fetch_specific_crypto_data
from ema_analysis import run_ema_analysis
from portfolio_simulation import run_momentum_portfolio_simulation
from config import config
from google_cloud_storage import upload_trading_results_to_gcs
from data_manager import DataManager

# Configuration flags
EXTRACT_DATA = False  # Set to True to fetch new data from API
RUN_EMA_ANALYSIS = True  # Set to True to run EMA analysis
RUN_PORTFOLIO_SIMULATION = True  # Set to True to run portfolio simulation
DISPLAY_PLOTS = False  # Set to True to display plots, False to just save them

# Strategy parameters
MOMENTUM_SHORT_PERIOD = 5  # Short EMA period for momentum strategy
MOMENTUM_LONG_PERIOD = 15  # Long EMA period for momentum strategy
MOMENTUM_SLOPE_WINDOW = 30  # Rolling window for slope distribution
MOMENTUM_SIGMA_MULTIPLIER = 0.5  # Sigma multiplier for adaptive threshold

# Trading parameters
BASE_POSITION_SIZE = 50  # Base position size in USD
STIFFNESS_THRESHOLD = 3  # Threshold for double position size
TRADING_ENABLED = True # Set to True to execute real trades (not just simulation)
# Leverage configuration
LEVERAGE_MULTIPLIER = 1  # Leverage multiplier (0.0 = no leverage, 1.0 = max leverage)

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/crypto_ema.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    logger = logging.getLogger(__name__)
    
    required_packages = ['requests', 'pandas', 'numpy', 'openpyxl', 'tqdm', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please run 'pip install -r requirements.txt' to install dependencies")
        return False
    
    return True

def find_latest_data_file():
    """Find the most recent crypto data file"""
    data_pattern = "data/specific_crypto_data_*.xlsx"
    data_files = glob.glob(data_pattern)
    
    if not data_files:
        return None
    
    # Sort by modification time and return the most recent
    latest_file = max(data_files, key=os.path.getmtime)
    return latest_file

def main():
    """Main function to orchestrate the crypto EMA analysis"""
    logger = setup_logging()
    
    logger.info("Starting Crypto EMA Analysis with Momentum Strategy")
    logger.info(f"Analysis started at: {datetime.now()}")
    
    # Initialize data manager
    data_manager = DataManager()
    data_strategy = data_manager.get_data_strategy()
    
    logger.info(f"Configuration:")
    logger.info(f"  - EXTRACT_DATA: {data_strategy['extract_data']}")
    logger.info(f"  - RUN_EMA_ANALYSIS: {data_strategy['run_analysis']}")
    logger.info(f"  - RUN_PORTFOLIO_SIMULATION: {data_strategy['run_simulation']}")
    logger.info(f"  - DISPLAY_PLOTS: {DISPLAY_PLOTS}")
    logger.info(f"  - TRADING_ENABLED: {TRADING_ENABLED}")
    logger.info(f"  - LEVERAGE_MULTIPLIER: {LEVERAGE_MULTIPLIER:.1f}x (0.0 = no leverage, 1.0 = max leverage)")
    logger.info(f"  - Using Historical Data: {data_strategy['use_historical']}")
    
    logger.info("EMA and Momentum Strategy Parameters:")
    logger.info(f"  - Short EMA: {MOMENTUM_SHORT_PERIOD}d, Long EMA: {MOMENTUM_LONG_PERIOD}d")
    logger.info(f"  - Slope Window: {MOMENTUM_SLOPE_WINDOW}d, Sigma Multiplier: {MOMENTUM_SIGMA_MULTIPLIER}")
    logger.info(f"  - Note: Same EMA periods used for both analysis and momentum strategy")
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("Dependencies check failed. Please run setup first.")
        return
    
    try:
        data_file = None
        portfolio_results = None
        historical_data = None
        
        # Step 0: Load Historical Data in Memory
        logger.info("Step 0: Loading historical data in memory...")
        historical_data = data_manager.load_historical_data_in_memory()
        
        # Step 1: Data Acquisition
        if data_strategy['extract_data']:
            logger.info("Step 1: Starting data acquisition...")
            
            # Get API credentials from environment variables
            api_key = config.get_api_credentials()
            
            # Fetch specific crypto data and store in database
            data = fetch_specific_crypto_data(api_key=api_key)
            
            if data is None:
                logger.error("Failed to fetch crypto data")
                return
            
            logger.info(f"Data acquired successfully. Found {data['metadata']['tokens_count']} tokens")
            logger.info(f"Tokens: {', '.join(data['metadata']['tokens_fetched'])}")
            logger.info(f"Database: {data['database_path']}")
        else:
            logger.info("Step 1: Data acquisition SKIPPED (extract_data = False)")
            logger.info("Using existing database data...")
        
        # Step 2: EMA Analysis
        if data_strategy['run_analysis']:
            logger.info("Step 2: Starting EMA Analysis...")
            logger.info(f"Short EMA period: {MOMENTUM_SHORT_PERIOD} days, Long EMA period: {MOMENTUM_LONG_PERIOD} days")
            
            ema_results = run_ema_analysis(
                data_source='database',
                short_period=MOMENTUM_SHORT_PERIOD,
                long_period=MOMENTUM_LONG_PERIOD,
                display_plots=DISPLAY_PLOTS
            )
            
            if ema_results:
                logger.info("EMA Analysis completed successfully!")
                logger.info(f"Tokens analyzed: {ema_results['tokens_analyzed']}")
                logger.info(f"Results directory: {ema_results['results_dir']}")
            else:
                logger.error("Error during analysis")
                return
        else:
            logger.info("Step 2: EMA Analysis SKIPPED (run_analysis = False)")
        
        # Step 3: Portfolio Simulation (Momentum Strategy Only)
        if data_strategy['run_simulation']:
            logger.info("Step 3: Starting Momentum Portfolio Simulation...")
            
            logger.info("Using MOMENTUM Portfolio Strategy:")
            logger.info("Trading Rules:")
            logger.info(f"  - BUY: EMA Slope > Adaptive Threshold (mean + {MOMENTUM_SIGMA_MULTIPLIER}σ) AND Volume EMA Slope > 0")
            logger.info(f"  - SELL: EMA Slope < 0 (momentum turning negative)")
            logger.info(f"  - SELL_VOLUME: Volume EMA Slope < 0 AND Volume EMA Difference < 10% (volume declining and low)")
            logger.info(f"  - Adaptive Threshold: Rolling {MOMENTUM_SLOPE_WINDOW}d window, {MOMENTUM_SIGMA_MULTIPLIER}σ above mean")
            logger.info("Position Sizing (Stiffness-based with Leverage):")
            logger.info(f"  - Normal positions: Base position size (${BASE_POSITION_SIZE}) × {LEVERAGE_MULTIPLIER:.1f}x leverage")
            logger.info(f"  - Strong signals: Double USD amount when stiffness > {STIFFNESS_THRESHOLD}σ above threshold")
            logger.info(f"  - Leverage stays constant, only USD amount increases")
            logger.info(f"  - Stiffness = (EMA Slope - Threshold) / Standard Deviation")
            
            # Use historical data if available, otherwise use database
            if historical_data and historical_data.get('crypto_data') is not None:
                logger.info("📊 Using historical crypto data in memory for simulation")
                # TODO: Modify run_momentum_portfolio_simulation to accept DataFrame instead of db_path
                portfolio_results = run_momentum_portfolio_simulation(
                    db_path="crypto_data.db",  # Will be updated to use historical_data
                    short_period=MOMENTUM_SHORT_PERIOD,
                    long_period=MOMENTUM_LONG_PERIOD,
                    slope_window=MOMENTUM_SLOPE_WINDOW,
                    sigma_multiplier=MOMENTUM_SIGMA_MULTIPLIER,
                    position_size=BASE_POSITION_SIZE,
                    stiffness_threshold=STIFFNESS_THRESHOLD
                )
            else:
                logger.info("�� Using database for simulation (no historical data available)")
                portfolio_results = run_momentum_portfolio_simulation(
                    db_path="crypto_data.db",
                    short_period=MOMENTUM_SHORT_PERIOD,
                    long_period=MOMENTUM_LONG_PERIOD,
                    slope_window=MOMENTUM_SLOPE_WINDOW,
                    sigma_multiplier=MOMENTUM_SIGMA_MULTIPLIER,
                    position_size=BASE_POSITION_SIZE,
                    stiffness_threshold=STIFFNESS_THRESHOLD
                )
            
            if portfolio_results:
                logger.info("Portfolio Simulation completed successfully!")
                logger.info(f"Total PnL: {portfolio_results['summary']['total_pnl']:.2f}%")
                logger.info(f"Total Trades: {portfolio_results['summary']['total_trades']}")
                logger.info(f"Overall Win Rate: {portfolio_results['summary']['overall_win_rate']:.1f}%")
                
                # Display stiffness-based position sizing statistics
                if 'stiffness_stats' in portfolio_results['summary']:
                    stiffness_stats = portfolio_results['summary']['stiffness_stats']
                    logger.info("Stiffness-based Position Sizing Results (with Leverage):")
                    logger.info(f"  - Normal positions: {stiffness_stats['normal_positions']}")
                    logger.info(f"  - Double-size positions: {stiffness_stats['double_positions']}")
                    logger.info(f"  - Average stiffness: {stiffness_stats['avg_stiffness']:.2f}σ")
                    logger.info(f"  - Maximum stiffness: {stiffness_stats['max_stiffness']:.2f}σ")
                    logger.info(f"  - Total position value: ${stiffness_stats['total_position_value']:.2f}")
                    logger.info(f"  - Stiffness threshold for double positions: >{stiffness_stats['stiffness_threshold']}σ")
                    logger.info(f"  - Leverage multiplier: {LEVERAGE_MULTIPLIER:.1f}x")
            else:
                logger.error("Error during portfolio simulation")
                return
        else:
            logger.info("Step 3: Portfolio Simulation SKIPPED (run_simulation = False)")
        
        # Step 4: Trading Execution (if requested)
        if TRADING_ENABLED:
            logger.info("Step 4: Starting Trading Execution...")
            
            try:
                # Import trading functions
                from simple_trader import setup_hyperliquid, get_market_data, calculate_signals, find_trading_opportunities, execute_trade
                
                logger.info("Trading Configuration:")
                logger.info(f"  - Base Position Size: ${BASE_POSITION_SIZE}")
                logger.info(f"  - Stiffness Threshold: {STIFFNESS_THRESHOLD}σ")
                logger.info(f"  - Leverage Multiplier: {LEVERAGE_MULTIPLIER:.1f}x")
                logger.info(f"  - Real Trading: {'✅ ENABLED' if TRADING_ENABLED else '❌ DISABLED (Simulation)'}")
                
                # Setup Hyperliquid connection
                exchange = setup_hyperliquid()
                if not exchange:
                    logger.error("❌ Cannot execute trades without Hyperliquid connection")
                    return
                
                # Use portfolio simulation results if available, otherwise calculate new signals
                if data_strategy['run_simulation'] and portfolio_results and TRADING_ENABLED:
                    logger.info("📊 Using portfolio simulation results for trading execution")
                    # Extract latest signals from portfolio simulation
                    signals = {}
                    portfolio_data = portfolio_results.get('portfolio_results', {})
                    for token_name, token_results in portfolio_data.items():
                        if 'signals' in token_results and not token_results['signals'].empty:
                            signals[token_name] = token_results['signals']
                    
                    if not signals:
                        logger.warning("⚠️ No signals available from portfolio simulation")
                        return
                else:
                    # Get latest market data and calculate signals
                    logger.info("📊 Calculating new signals for trading execution")
                    data = get_market_data()
                    if data is None:
                        logger.error("❌ Cannot execute trades without market data")
                        return
                    
                    # Calculate latest signals
                    signals = calculate_signals(data)
                    if not signals:
                        logger.warning("⚠️ No signals calculated")
                        return
                
                # Find trading opportunities
                opportunities = find_trading_opportunities(signals)
                if not opportunities:
                    logger.info("📊 No trading opportunities found at this time")
                    return
                
                logger.info(f"📊 Found {len(opportunities)} trading opportunities")
                
                # Execute trades
                successful_trades = 0
                failed_trades = 0
                
                for opportunity in opportunities:
                    try:
                        result = execute_trade(opportunity, trading_enabled=TRADING_ENABLED)
                        if result and result.get('success', False):
                            successful_trades += 1
                            logger.info(f"✅ Trade executed successfully: {opportunity['token_name']} {opportunity['signal_type']}")
                        else:
                            failed_trades += 1
                            logger.error(f"❌ Trade failed: {opportunity['token_name']} {opportunity['signal_type']}")
                    except Exception as e:
                        failed_trades += 1
                        logger.error(f"❌ Trade execution error: {str(e)}")
                
                logger.info(f"�� Trading Summary: {successful_trades} successful, {failed_trades} failed")
                
                # Display recent errors from trading_errors.log
                if os.path.exists('trading_errors.log'):
                    logger.info("📝 Recent trading errors:")
                    try:
                        with open('trading_errors.log', 'r') as f:
                            lines = f.readlines()
                            for line in lines[-5:]:  # Show last 5 lines
                                if line.strip() and not line.startswith('='):
                                    logger.info(f"    {line}")
                    except FileNotFoundError:
                        logger.info("📝 No error log file found")
                    
                    logger.info("🎯 Trading execution completed!")
                
            except Exception as e:
                logger.error(f"❌ Error during trading execution: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return
        else:
            logger.info("Step 4: Trading Execution SKIPPED (TRADING_ENABLED = False)")
        
        # Step 5: Upload results to Google Cloud Storage
        logger.info("Step 5: Uploading results to Google Cloud Storage...")
        try:
            upload_result = upload_trading_results_to_gcs("results")
            if upload_result.get("success", False):
                logger.info(f"✅ Successfully uploaded {len(upload_result.get('uploaded_files', []))} files to Google Cloud Storage")
                if upload_result.get("failed_files"):
                    logger.warning(f"⚠️ Failed to upload {len(upload_result['failed_files'])} files")
            else:
                logger.warning(f"⚠️ Google Cloud Storage upload failed: {upload_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"⚠️ Google Cloud Storage upload error: {str(e)}")
        
        # Step 6: Analysis completed
        logger.info("Step 6: Analysis completed successfully!")
        logger.info(f"Analysis finished at: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
