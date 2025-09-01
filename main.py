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
from config import config, get_api_credentials, get_trading_credentials, get_wallet_config

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
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
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
    
    logger.info(f"Configuration:")
    logger.info(f"  - EXTRACT_DATA: {EXTRACT_DATA}")
    logger.info(f"  - RUN_EMA_ANALYSIS: {RUN_EMA_ANALYSIS}")
    logger.info(f"  - RUN_PORTFOLIO_SIMULATION: {RUN_PORTFOLIO_SIMULATION}")
    logger.info(f"  - DISPLAY_PLOTS: {DISPLAY_PLOTS}")
    logger.info(f"  - TRADING_ENABLED: {TRADING_ENABLED}")
    logger.info(f"  - LEVERAGE_MULTIPLIER: {LEVERAGE_MULTIPLIER:.1f}x (0.0 = no leverage, 1.0 = max leverage)")
    
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
        
        # Step 1: Data Acquisition
        if EXTRACT_DATA:
            logger.info("Step 1: Starting data acquisition...")
            
            # Get API credentials from environment variables
            api_key = get_api_credentials()
            
            # Fetch specific crypto data and store in database
            data = fetch_specific_crypto_data(api_key=api_key, db_path="crypto_data.db")
            
            if not data:
                logger.error("No data acquired. Check API connectivity.")
                return
            
            logger.info(f"Data acquired successfully. Found {data['metadata']['tokens_count']} tokens")
            logger.info(f"Tokens: {', '.join(data['metadata']['tokens_fetched'])}")
            logger.info(f"Database: {data['database_path']}")
        else:
            logger.info("Step 1: Data acquisition SKIPPED (EXTRACT_DATA = False)")
            logger.info("Using existing database data...")
        
        # Step 2: EMA Analysis
        if RUN_EMA_ANALYSIS:
            logger.info("Step 2: Starting EMA Analysis...")
            logger.info(f"Short EMA period: {MOMENTUM_SHORT_PERIOD} days, Long EMA period: {MOMENTUM_LONG_PERIOD} days")
            
            ema_results = run_ema_analysis(
                data_source='database',
                results_dir='results',
                short_period=MOMENTUM_SHORT_PERIOD,
                long_period=MOMENTUM_LONG_PERIOD,
                db_path="crypto_data.db"
            )
            
            if ema_results:
                logger.info("EMA Analysis completed successfully!")
                logger.info(f"Results saved to: {ema_results['excel_file']}")
                logger.info(f"Data shape: {ema_results['data_shape']}")
                logger.info(f"Tokens analyzed: {ema_results['tokens_analyzed']}")
                logger.info(f"Results directory: {ema_results['results_dir']}")
            else:
                logger.error("Error during analysis")
                return
        else:
            logger.info("Step 2: EMA Analysis SKIPPED (RUN_EMA_ANALYSIS = False)")
        
        # Step 3: Portfolio Simulation (Momentum Strategy Only)
        if RUN_PORTFOLIO_SIMULATION:
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
            logger.info("Step 3: Portfolio Simulation SKIPPED (RUN_PORTFOLIO_SIMULATION = False)")
        
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
                if RUN_PORTFOLIO_SIMULATION and portfolio_results and TRADING_ENABLED:
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
                        logger.warning("⚠️ No signals calculated, no trading opportunities")
                        return
                
                # Find current trading opportunities
                opportunities = find_trading_opportunities(signals)
                
                if not opportunities:
                    logger.info("📊 No trading opportunities found at this time")
                else:
                    logger.info(f"🎯 Found {len(opportunities)} trading opportunities:")
                    
                    # Execute trades
                    successful_trades = 0
                    failed_trades = 0
                    
                    for i, opportunity in enumerate(opportunities, 1):
                        logger.info(f"Trade {i}/{len(opportunities)}:")
                        success = execute_trade(exchange, opportunity, TRADING_ENABLED)
                        
                        if success:
                            logger.info(f"✅ Trade {i} completed successfully")
                            successful_trades += 1
                        else:
                            logger.error(f"❌ Trade {i} failed")
                            failed_trades += 1
                        
                        logger.info("-" * 30)
                    
                    # Display trading session summary
                    logger.info("📊 TRADING SESSION SUMMARY:")
                    logger.info(f"  - Total Opportunities: {len(opportunities)}")
                    
                    logger.info(f"  - Successful Trades: {successful_trades}")
                    logger.info(f"  - Failed Trades: {failed_trades}")
                    
                    # Check for error log file
                    try:
                        with open('trading_errors.log', 'r') as f:
                            error_content = f.read()
                            if error_content.strip():
                                logger.info("📝 Recent Errors (see trading_errors.log for details):")
                                # Show last few error lines
                                error_lines = error_content.strip().split('\n')[-20:]
                                for line in error_lines:
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
        
        # Step 5: Analysis completed
        logger.info("Step 5: Analysis completed successfully!")
        logger.info(f"Analysis finished at: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 