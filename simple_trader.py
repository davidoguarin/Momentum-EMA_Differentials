#!/usr/bin/env python3
"""
Simple Trader - Manual execution of momentum strategy trades
Run this script when you want to check for trading opportunities and execute trades
"""

import ccxt
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from eth_account import Account
import os

# Import our working modules
from portfolio_simulation import MomentumPortfolioSimulator
from database_manager import CryptoDatabase
from config import config

# Enable HD wallet features
Account.enable_unaudited_hdwallet_features()

# Configuration - These will be overridden when imported from main.py or environment variables
WALLET_ADDRESS = None  # Will be loaded from environment variables
SEED_PHRASE = None     # Will be loaded from environment variables
BASE_POSITION_SIZE = 50  # Base position size in USD
STIFFNESS_THRESHOLD = 1.5  # Threshold for double position size

# Strategy parameters
SHORT_EMA_PERIOD = 5
LONG_EMA_PERIOD = 15
SLOPE_WINDOW = 30
SIGMA_MULTIPLIER = 0.5

# Leverage configuration
MAX_LEVERAGE_CONFIG = {
    'XRP': 20,
    'BTC': 40,
    'ETH': 25,
    'SOL': 20,
    'MATIC': 20,
    'AVAX': 10,
    'LINK': 10,
    'DOT': 10,
}

# Leverage multiplier will be set from main.py
LEVERAGE_MULTIPLIER = None

# Order tracking
ORDERS_FILE = "trading_orders.xlsx"
open_positions = {}  # Track open positions for PnL calculation

def update_config_from_main():
    """Update configuration from main.py if available and load environment variables"""
    global BASE_POSITION_SIZE, STIFFNESS_THRESHOLD, TRADING_ENABLED
    global SHORT_EMA_PERIOD, LONG_EMA_PERIOD, SLOPE_WINDOW, SIGMA_MULTIPLIER
    global LEVERAGE_MULTIPLIER, WALLET_ADDRESS, SEED_PHRASE
    
    # First, try to load from environment variables
    wallet_addr, seed_phrase = config.get_wallet_config()
    if wallet_addr and seed_phrase:
        WALLET_ADDRESS = wallet_addr
        SEED_PHRASE = seed_phrase
        print(f"✅ Wallet configuration loaded from environment variables")
    
    # Then try to update from main.py if available
    try:
        import main
        BASE_POSITION_SIZE = main.BASE_POSITION_SIZE
        STIFFNESS_THRESHOLD = main.STIFFNESS_THRESHOLD
        TRADING_ENABLED = main.TRADING_ENABLED
        SHORT_EMA_PERIOD = main.MOMENTUM_SHORT_PERIOD
        LONG_EMA_PERIOD = main.MOMENTUM_LONG_PERIOD
        SLOPE_WINDOW = main.MOMENTUM_SLOPE_WINDOW
        SIGMA_MULTIPLIER = main.MOMENTUM_SIGMA_MULTIPLIER
        
        # Update leverage multiplier if available
        if hasattr(main, 'LEVERAGE_MULTIPLIER'):
            LEVERAGE_MULTIPLIER = main.LEVERAGE_MULTIPLIER
        
        print(f"✅ Configuration updated from main.py")
        print(f"   Base Position Size: ${BASE_POSITION_SIZE}")
        print(f"   Stiffness Threshold: {STIFFNESS_THRESHOLD}σ")
        print(f"   Trading Enabled: {'✅ YES' if TRADING_ENABLED else '❌ NO'}")
        print(f"   Leverage Multiplier: {LEVERAGE_MULTIPLIER:.1f}x")
        
    except ImportError:
        print("ℹ️ Using default configuration (main.py not available)")
    except Exception as e:
        print(f"⚠️ Error updating config from main.py: {str(e)}")
        print("ℹ️ Using default configuration")
    
    # Load existing orders after updating config
    load_existing_orders()

def load_existing_orders():
    """Load existing orders from Excel file"""
    global open_positions
    
    if os.path.exists(ORDERS_FILE):
        try:
            # Load orders sheet
            orders_df = pd.read_excel(ORDERS_FILE, sheet_name='Orders')
            
            # Load open positions sheet
            if 'Open_Positions' in pd.ExcelFile(ORDERS_FILE).sheet_names:
                positions_df = pd.read_excel(ORDERS_FILE, sheet_name='Open_Positions')
                
                # Reconstruct open_positions dictionary
                for _, row in positions_df.iterrows():
                    position_id = f"{row['Token']}_{row['Entry_Date']}"
                    open_positions[position_id] = {
                        'token': row['Token'],
                        'entry_price': row['Entry_Price'],
                        'entry_date': row['Entry_Date'],
                        'units': row['Units'],
                        'position_size': row['Position_Size_USD'], # Changed to Position_Size_USD
                        'stiffness': row['Stiffness'],
                        'position_multiplier': row['Position_Multiplier'],
                        'executed': False # Added executed field
                    }
                
                print(f"✅ Loaded {len(open_positions)} existing open positions")
            else:
                print("ℹ️ No existing open positions found")
                
            print(f"✅ Loaded existing orders from {ORDERS_FILE}")
            return orders_df
            
        except Exception as e:
            print(f"⚠️ Error loading existing orders: {str(e)}")
            return None
    else:
        print("ℹ️ No existing orders file found, starting fresh")
        return None

def log_error_to_file(error_data):
    """
    Log detailed error information to a dedicated error log file
    """
    try:
        error_log_file = 'trading_errors.log'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR TIMESTAMP: {timestamp}\n")
            f.write(f"ERROR TYPE: {error_data.get('error_type', 'UNKNOWN')}\n")
            f.write(f"TOKEN: {error_data.get('token', 'UNKNOWN')}\n")
            f.write(f"POSITION SIZE: ${error_data.get('position_size_usd', 0):.2f}\n")
            f.write(f"UNITS: {error_data.get('units', 0):.4f}\n")
            f.write(f"PRICE: ${error_data.get('price', 0):.4f}\n")
            f.write(f"MARKET: {error_data.get('market', 'UNKNOWN')}\n")
            f.write(f"ERROR MESSAGE: {error_data.get('error_message', 'UNKNOWN')}\n")
            f.write(f"USDC BALANCE: {error_data.get('usdc_balance', 'UNKNOWN')}\n")
            f.write(f"{'='*80}\n")
        
        print(f"📝 Error logged to {error_log_file}")
    except Exception as e:
        print(f"⚠️ Failed to log error to file: {e}")

def save_order_to_excel(order_data):
    """Save order data to Excel file"""
    try:
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Check if file exists and load existing data
        if os.path.exists(ORDERS_FILE):
            try:
                # Load existing orders
                with pd.ExcelWriter(ORDERS_FILE, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    # Load existing orders sheet
                    try:
                        existing_orders = pd.read_excel(ORDERS_FILE, sheet_name='Orders')
                        # Find the next empty row
                        next_row = len(existing_orders) + 2  # +2 because Excel is 1-indexed and we want to start after header
                    except:
                        existing_orders = pd.DataFrame()
                        next_row = 2
                    
                    # Create new order row
                    new_order = pd.DataFrame([order_data])
                    
                    # Write to specific row
                    new_order.to_excel(writer, sheet_name='Orders', startrow=next_row-1, index=False, header=False)
                    
            except Exception as e:
                print(f"⚠️ Error appending to existing file: {e}")
                # If append fails, create new file
                create_new_excel_file(order_data)
        else:
            # Create new file
            create_new_excel_file(order_data)
            
        print(f"✅ Order saved to Excel: {ORDERS_FILE}")
        
    except Exception as e:
        print(f"❌ Error saving order to Excel: {e}")

def create_new_excel_file(order_data):
    """Create new Excel file with order data"""
    try:
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Create orders DataFrame
        orders_df = pd.DataFrame([order_data])
        
        # Create open positions DataFrame
        positions_df = pd.DataFrame(columns=[
            'Token', 'Entry_Date', 'Entry_Price', 'Units', 'Position_Size_USD', 
            'Position_Type', 'Stiffness', 'Position_Multiplier'
        ])
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(ORDERS_FILE, engine='openpyxl') as writer:
            orders_df.to_excel(writer, sheet_name='Orders', index=False)
            positions_df.to_excel(writer, sheet_name='Open_Positions', index=False)
            
    except Exception as e:
        print(f"❌ Error creating new Excel file: {e}")

def update_open_positions_sheet():
    """Update the Open_Positions sheet in Excel"""
    try:
        if not os.path.exists(ORDERS_FILE):
            return
            
        # Create positions DataFrame
        positions_data = []
        for position_id, position in open_positions.items():
            positions_data.append({
                'Token': position['token'],
                'Entry_Date': position['entry_date'],
                'Entry_Price': position['entry_price'],
                'Units': position['units'],
                'Position_Size_USD': position['position_size'],
                'Position_Type': position['position_type'],
                'Stiffness': position['stiffness'],
                'Position_Multiplier': position['position_multiplier']
            })
        
        positions_df = pd.DataFrame(positions_data)
        
        # Update the file
        with pd.ExcelWriter(ORDERS_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            positions_df.to_excel(writer, sheet_name='Open_Positions', index=False)
            
    except Exception as e:
        print(f"⚠️ Error updating open positions sheet: {e}")

def _find_market_symbol(token_name):
    """Find the correct market symbol for a token"""
    try:
        # Map token names to their Hyperliquid market symbols
        market_mapping = {
            'SOL': 'SOL/USDC:USDC',
            'BTC': 'BTC/USDC:USDC',
            'ETH': 'ETH/USDC:USDC',
            'ARB': 'ARB/USDC:USDC',
            'NEAR': 'NEAR/USDC:USDC',
            'SUI': 'SUI/USDC:USDC',
            'TRX': 'TRX/USDC:USDC',
            'XRP': 'XRP/USDC:USDC'
        }
        
        if token_name in market_mapping:
            return market_mapping[token_name]
        else:
            print(f"⚠️ Unknown token: {token_name}")
            return None
            
    except Exception as e:
        print(f"❌ Error finding market symbol for {token_name}: {str(e)}")
        return None

def check_account_balance_and_margin(exchange, required_position_size_usd, leverage=10.0):
    """
    Check if account has sufficient balance and margin for a position
    
    Returns:
        dict with balance info and whether trade is possible
    """
    try:
        balance = exchange.fetch_balance()
        usdc_balance = balance.get('USDC', {}).get('free', 0)
        
        # For futures trading, we need to check margin requirements
        # The required margin is the position size / leverage
        # This is the actual capital you need to invest
        
        required_margin = required_position_size_usd / leverage
        
        can_trade = usdc_balance >= required_margin
        
        return {
            'can_trade': can_trade,
            'usdc_balance': usdc_balance,
            'required_margin': required_margin,
            'margin_ratio': usdc_balance / required_margin if required_margin > 0 else float('inf'),
            'insufficient_amount': max(0, required_margin - usdc_balance)
        }
    except Exception as e:
        print(f"⚠️ Could not check account balance: {e}")
        return {
            'can_trade': False,
            'usdc_balance': 0,
            'required_margin': required_position_size_usd * 0.5,
            'margin_ratio': 0,
            'insufficient_amount': required_position_size_usd * 0.5,
            'error': str(e)
        }

def calculate_leverage_position_size(token, base_position_size_usd, stiffness_multiplier=1.0, custom_leverage_multiplier=None):
    """Calculate position size with leverage and stiffness multiplier
    
    Returns:
        dict with:
        - position_size_usd: The effective position size (base × stiffness × leverage)
        - effective_position_size: Same as position_size_usd
        - leverage: Actual leverage used
        - max_leverage: Maximum allowed leverage for this token
        - leverage_multiplier: The multiplier applied to max leverage
    """
    # Get max leverage for this token
    max_leverage = MAX_LEVERAGE_CONFIG.get(token, 10)  # Default to 10x if not found
    
    # Use custom leverage multiplier if provided, otherwise use global setting
    leverage_multiplier = custom_leverage_multiplier if custom_leverage_multiplier is not None else LEVERAGE_MULTIPLIER
    
    # If leverage_multiplier is still None, default to 1.0 (100% of max leverage)
    if leverage_multiplier is None:
        leverage_multiplier = 1.0
    
    # Calculate actual leverage based on multiplier
    actual_leverage = max_leverage * leverage_multiplier
    
    # Apply stiffness multiplier (doubles USD amount when stiffness threshold is met)
    adjusted_position_size_usd = base_position_size_usd * stiffness_multiplier
    
    # Calculate units needed (considering leverage)
    # With leverage, we can control more units with the same USD
    effective_position_size = adjusted_position_size_usd * actual_leverage
    
    return {
        'position_size_usd': effective_position_size,  # This should be the leverage-adjusted amount
        'effective_position_size': effective_position_size,
        'leverage': actual_leverage,
        'max_leverage': max_leverage,
        'leverage_multiplier': leverage_multiplier
    }

def calculate_pnl(entry_price, exit_price, units, position_size):
    """Calculate PnL for a closed position"""
    try:
        # Calculate PnL in USD
        pnl_usd = (exit_price - entry_price) * units
        
        # Calculate PnL percentage
        pnl_percent = (pnl_usd / position_size) * 100
        
        return pnl_usd, pnl_percent
        
    except Exception as e:
        print(f"❌ Error calculating PnL: {str(e)}")
        return 0, 0

def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def setup_hyperliquid():
    """Setup Hyperliquid connection"""
    try:
        # Derive private key from seed phrase
        private_key = Account.from_mnemonic(SEED_PHRASE).key.hex()
        
        # Initialize exchange
        exchange = ccxt.hyperliquid({
            'privateKey': private_key,
            'walletAddress': WALLET_ADDRESS,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Load markets
        exchange.load_markets()
        print(f"✅ Hyperliquid connected. Available markets: {len(exchange.markets)}")
        
        return exchange
        
    except Exception as e:
        print(f"❌ Failed to setup Hyperliquid: {str(e)}")
        return None

def get_market_data():
    """Get latest market data from database and calculate EMAs if needed"""
    try:
        db = CryptoDatabase("crypto_data.db")
        
        # Get all available tokens
        token_symbols = ['SOL', 'BTC', 'ETH', 'ARB', 'NEAR', 'SUI', 'TRX', 'XRP']
        data = db.get_latest_data(token_symbols)
        db.close_connection()
        
        if data is not None and not data.empty:
            print(f"✅ Market data loaded: {len(data)} data points")
            
            # Check if we have EMA data
            ema_columns = [col for col in data.columns if 'ema_' in col]
            if not ema_columns:
                print("⚠️ No EMA data found in database. Calculating EMAs on the fly...")
                
                # Calculate EMAs for all tokens
                for token_name in token_symbols:
                    price_col = f'{token_name}_price'
                    volume_col = f'{token_name}_volume'
                    
                    if price_col in data.columns and volume_col in data.columns:
                        # Calculate price EMAs
                        data[f'{token_name}_ema_5d'] = data[price_col].ewm(span=5, adjust=False).mean()
                        data[f'{token_name}_ema_15d'] = data[price_col].ewm(span=15, adjust=False).mean()
                        
                        # Calculate volume EMAs
                        data[f'{token_name}_volume_ema_5d'] = data[volume_col].ewm(span=5, adjust=False).mean()
                        data[f'{token_name}_volume_ema_15d'] = data[volume_col].ewm(span=15, adjust=False).mean()
                        
                        # Calculate EMA differences and slopes
                        data[f'{token_name}_ema_difference'] = data[f'{token_name}_ema_5d'] - data[f'{token_name}_ema_15d']
                        data[f'{token_name}_ema_slope'] = data[f'{token_name}_ema_difference'].diff()
                        
                        data[f'{token_name}_volume_ema_difference'] = data[f'{token_name}_volume_ema_5d'] - data[f'{token_name}_volume_ema_15d']
                        data[f'{token_name}_volume_ema_slope'] = data[f'{token_name}_volume_ema_difference'].diff()
                
                print(f"✅ EMAs calculated for {len(token_symbols)} tokens")
            
            # Check if we have the latest data point with signals
            if len(data) > 0:
                latest_row = data.iloc[-1]
                print(f"📅 Latest data point: {latest_row.name}")
                
                # Check for any BUY signals in the latest data
                for token in token_symbols:
                    if f'{token}_ema_slope' in data.columns:
                        slope = latest_row[f'{token}_ema_slope']
                        if pd.notna(slope):
                            print(f"   {token}: EMA Slope = {slope:.6f}")
            
            return data
        else:
            print("❌ No market data available")
            return None
            
    except Exception as e:
        print(f"❌ Error getting market data: {str(e)}")
        return None

def calculate_signals(data):
    """Calculate trading signals using momentum strategy"""
    try:
        # Initialize strategy
        strategy = MomentumPortfolioSimulator(
            initial_capital=10000,
            position_size=BASE_POSITION_SIZE,
            slope_window=SLOPE_WINDOW,
            sigma_multiplier=SIGMA_MULTIPLIER
        )
        
        # Get token names from price columns
        price_columns = [col for col in data.columns if col.endswith('_price')]
        token_names = [col.replace('_price', '') for col in price_columns]
        
        print(f"📊 Calculating signals for {len(token_names)} tokens...")
        
        signals = {}
        for token_name in token_names:
            try:
                # Calculate signals for this token
                token_signals = strategy.calculate_momentum_signals(
                    data, token_name, SHORT_EMA_PERIOD, LONG_EMA_PERIOD
                )
                
                if not token_signals.empty:
                    signals[token_name] = token_signals
                    
            except Exception as e:
                print(f"⚠️ Error calculating signals for {token_name}: {str(e)}")
                continue
        
        print(f"✅ Signals calculated for {len(signals)} tokens")
        return signals
        
    except Exception as e:
        print(f"❌ Error calculating signals: {str(e)}")
        return {}

def find_trading_opportunities(signals):
    """Find current trading opportunities from signals"""
    opportunities = []
    
    try:
        # Get latest data point from the first token's signals
        if not signals:
            print("⚠️ No signals available")
            return []
        
        # Get the first token's signals to find the latest data
        first_token = list(signals.keys())[0]
        if first_token not in signals or signals[first_token].empty:
            print("⚠️ No signal data available")
            return []
        
        latest_data = signals[first_token].iloc[-1]
        
        for token_name in ['SOL', 'BTC', 'ETH', 'ARB', 'NEAR', 'SUI', 'TRX', 'XRP']:
            # Check if we have signals for this token
            if token_name not in signals or signals[token_name].empty:
                continue
            
            # Get latest signal for this token
            token_signals = signals[token_name]
            latest_signal = token_signals.iloc[-1]
            
            # Check for BUY signals (opening positions)
            if 'signal' in latest_signal and latest_signal['signal'] == 'BUY':
                # Check if we already have an open position for this token
                has_open_position = any(pos_data['token'] == token_name for pos_data in open_positions.values())
                
                if not has_open_position:
                    # Get signal details
                    current_price = latest_signal['price']
                    stiffness = latest_signal.get('stiffness', 0.0)
                    position_multiplier = latest_signal.get('position_multiplier', 1.0)
                    position_size = latest_signal.get('position_size_usd', BASE_POSITION_SIZE)
                    leverage = latest_signal.get('leverage', 10.0)  # Default to 10x if not found
                    
                    opportunities.append({
                        'token': token_name,
                        'signal_type': 'BUY',
                        'price': current_price,
                        'stiffness': stiffness,
                        'position_multiplier': position_multiplier,
                        'position_size': position_size,
                        'leverage': leverage,
                        'action': 'BUY'
                    })
            
            # Check for SELL signals (closing positions)
            elif 'signal' in latest_signal and latest_signal['signal'] in ['SELL', 'SELL_VOLUME']:
                # Check if we have an open position for this token
                has_open_position = any(pos_data['token'] == token_name for pos_data in open_positions.values())
                
                if has_open_position:
                    # Find the open position
                    open_position = None
                    for pos_data in open_positions.values():
                        if pos_data['token'] == token_name:
                            open_position = pos_data
                            break
                    
                    if open_position:
                        current_price = latest_signal['price']
                        opportunities.append({
                            'token': token_name,
                            'signal_type': 'SELL',
                            'price': current_price,
                            'stiffness': 0.0,  # Not relevant for SELL
                            'position_multiplier': 1.0,  # Not relevant for SELL
                            'position_size': open_position['position_size'],
                            'action': latest_signal['signal']
                        })
        
        if opportunities:
            print(f"🎯 Found {len(opportunities)} trading opportunities:")
            for opp in opportunities:
                if opp['signal_type'] == 'BUY':
                    print(f"   📈 {opp['action']} opportunity: {opp['token']}")
                    print(f"      Price: ${opp['price']:.4f}")
                    print(f"      Stiffness: {opp['stiffness']:.2f}σ")
                    print(f"      Position Size: ${opp['position_size']:.2f} ({opp['position_multiplier']:.1f}x)")
                else:
                    print(f"   📉 {opp['action']} opportunity: {opp['token']}")
                    print(f"      Price: ${opp['price']:.4f}")
                    print(f"      Position Size: ${opp['position_size']:.2f}")
                print()
        
        return opportunities
        
    except Exception as e:
        print(f"❌ Error finding trading opportunities: {str(e)}")
        return []

def execute_trade(exchange, opportunity, trading_enabled=None):
    """Execute a trade based on opportunity"""
    try:
        # Use passed trading_enabled parameter or fall back to global
        if trading_enabled is None:
            trading_enabled = TRADING_ENABLED
            
        token = opportunity['token']
        signal_type = opportunity['signal_type']
        price = opportunity['price']
        stiffness = opportunity['stiffness']
        position_multiplier = opportunity['position_multiplier']
        position_size = opportunity['position_size']
        leverage = opportunity.get('leverage', 10.0)  # Default to 10x if not found
        
        # Find market symbol
        market = _find_market_symbol(token)
        if not market:
            print(f"❌ Market not found for {token}")
            return False
        
        # Check account balance and margin before attempting trade
        balance_check = check_account_balance_and_margin(exchange, position_size, leverage)
        
        if not balance_check['can_trade']:
            print(f"❌ INSUFFICIENT MARGIN - Trade cannot be executed:")
            print(f"   - Required Margin: ${balance_check['required_margin']:.2f}")
            print(f"   - Available USDC: ${balance_check['usdc_balance']:.2f}")
            print(f"   - Insufficient Amount: ${balance_check['insufficient_amount']:.2f}")
            print(f"   - Margin Ratio: {balance_check['margin_ratio']:.2f}")
            
            # Log failed trade attempt
            try:
                save_order_to_excel({
                    'token': token,
                    'signal_type': signal_type,
                    'price': price,
                    'units': 0,
                    'position_size': position_size,
                    'status': 'FAILED_INSUFFICIENT_MARGIN',
                    'error': f"Insufficient margin. Required: ${balance_check['required_margin']:.2f}, Available: ${balance_check['usdc_balance']:.2f}",
                    'timestamp': datetime.now(),
                    'executed': False
                })
                print(f"   - Error logged to Excel")
            except Exception as excel_error:
                print(f"   - Failed to log error to Excel: {excel_error}")
            
            return False
        
        print(f"✅ Margin check passed:")
        print(f"   - Required Margin: ${balance_check['required_margin']:.2f}")
        print(f"   - Available USDC: ${balance_check['usdc_balance']:.2f}")
        print(f"   - Margin Ratio: {balance_check['margin_ratio']:.2f}")
        
        if signal_type == "BUY":
            # Opening position - position_size should already be leverage-adjusted from portfolio simulation
            # But we need to ensure it's properly calculated
            units = position_size / price
            
            print(f"🚀 Executing BUY order for {token}")
            print(f"   Market: {market}")
            print(f"   Price: ${price:.4f}")
            print(f"   Units: {units:.4f}")
            print(f"   Position Size: ${position_size:.2f}")
            print(f"   Stiffness: {stiffness:.2f}σ")
            print(f"   Multiplier: {position_multiplier}x")
            
            if trading_enabled:
                # Real trading
                try:
                    # Create order for futures trading
                    order = exchange.create_order(
                        symbol=market,
                        type='market',  # Use market order for immediate execution
                        side='buy',
                        amount=units,
                        price=price,  # Required for slippage calculation
                        params={
                            'reduceOnly': False,  # This is for opening a new position
                            'positionSide': 'LONG'  # Specify long position
                        }
                    )
                    
                    print(f"✅ Real BUY order executed: {order['id']}")
                    
                    # Save order to Excel
                    order_data = {
                        'Order_ID': f"{token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'Date': datetime.now(),
                        'Token': token,
                        'Order_Type': 'BUY',
                        'Price': price,
                        'Units': units,
                        'Position_Size_USD': position_size,
                        'Stiffness': stiffness,
                        'Position_Multiplier': position_multiplier,
                        'Market': market,
                        'Status': 'OPEN',
                        'Entry_Date': datetime.now(),
                        'Entry_Price': price,
                        'Exit_Date': None,
                        'Exit_Price': None,
                        'PnL': None,
                        'PnL_Percent': None,
                        'Executed': True
                    }
                    
                    save_order_to_excel(order_data)
                    
                    # Add to open positions
                    position_id = f"{token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    open_positions[position_id] = {
                        'token': token,
                        'entry_price': price,
                        'entry_date': datetime.now(),
                        'units': units,
                        'position_size': position_size,
                        'position_type': 'LONG',
                        'stiffness': stiffness,
                        'position_multiplier': position_multiplier
                    }
                    
                    # Update open positions sheet
                    update_open_positions_sheet()
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Real order failed: {e}")
                    return False
            else:
                # Simulation mode
                print("📊 SIMULATION MODE - Order not executed")
                
                # Save simulated order to Excel
                order_data = {
                    'Order_ID': f"{token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'Date': datetime.now(),
                    'Token': token,
                    'Order_Type': 'BUY',
                    'Price': price,
                    'Units': units,
                    'Position_Size_USD': position_size,
                    'Stiffness': stiffness,
                    'Position_Multiplier': position_multiplier,
                    'Market': market,
                    'Status': 'OPEN',
                    'Entry_Date': datetime.now(),
                    'Entry_Price': price,
                    'Exit_Date': None,
                    'Exit_Price': None,
                    'PnL': None,
                    'PnL_Percent': None,
                    'Executed': False
                }
                
                save_order_to_excel(order_data)
                
                # Add to open positions for simulation
                position_id = f"{token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                open_positions[position_id] = {
                    'token': token,
                    'entry_price': price,
                    'entry_date': datetime.now(),
                    'units': units,
                    'position_size': position_size,
                    'position_type': 'LONG',
                    'stiffness': stiffness,
                    'position_multiplier': position_multiplier
                }
                
                # Update open positions sheet
                update_open_positions_sheet()
                
                return True
                
        elif signal_type == "SELL":
            # Closing position
            position_id = opportunity['position_id']
            position = open_positions[position_id]
            entry_price = position['entry_price']
            units = position['units']
            
            # Calculate PnL
            pnl_usd, pnl_percent = calculate_pnl(entry_price, price, units, position['position_size'])
            
            print(f"🔴 Executing SELL order for {token}")
            print(f"   Market: {market}")
            print(f"   Entry Price: ${entry_price:.4f}")
            print(f"   Exit Price: ${price:.4f}")
            print(f"   Units: {units:.4f}")
            print(f"   PnL: ${pnl_usd:.2f} ({pnl_percent:.2f}%)")
            
            if trading_enabled:
                # Real trading
                try:
                    # Create order for futures trading (closing position)
                    order = exchange.create_order(
                        symbol=market,
                        type='market',  # Use market order for immediate execution
                        side='sell',
                        amount=units,
                        price=price,  # Required for slippage calculation
                        params={
                            'reduceOnly': True,  # This is for closing an existing position
                            'positionSide': 'LONG'  # Specify long position to close
                        }
                    )
                    
                    print(f"✅ Real SELL order executed: {order['id']}")
                    
                    # Save order to Excel
                    order_data = {
                        'Order_ID': f"{token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'Date': datetime.now(),
                        'Token': token,
                        'Order_Type': 'SELL',
                        'Price': price,
                        'Units': units,
                        'Position_Size_USD': position['position_size'],
                        'Stiffness': position['stiffness'],
                        'Position_Multiplier': position['position_multiplier'],
                        'Market': market,
                        'Status': 'CLOSED',
                        'Entry_Date': position['entry_date'],
                        'Entry_Price': entry_price,
                        'Exit_Date': datetime.now(),
                        'Exit_Price': price,
                        'PnL': pnl_usd,
                        'PnL_Percent': pnl_percent,
                        'Executed': True
                    }
                    
                    save_order_to_excel(order_data)
                    
                    # Remove from open positions
                    del open_positions[position_id]
                    
                    # Update open positions sheet
                    update_open_positions_sheet()
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Real order failed: {e}")
                    return False
            else:
                # Simulation mode
                print("📊 SIMULATION MODE - Order not executed")
                
                # Save simulated order to Excel
                order_data = {
                    'Order_ID': f"{token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'Date': datetime.now(),
                    'Token': token,
                    'Order_Type': 'SELL',
                    'Price': price,
                    'Units': units,
                    'Position_Size_USD': position['position_size'],
                    'Stiffness': position['stiffness'],
                    'Position_Multiplier': position['position_multiplier'],
                    'Market': market,
                    'Status': 'CLOSED',
                    'Entry_Date': position['entry_date'],
                    'Entry_Price': entry_price,
                    'Exit_Date': datetime.now(),
                    'Exit_Price': price,
                    'PnL': pnl_usd,
                    'PnL_Percent': pnl_percent,
                    'Executed': False
                }
                
                save_order_to_excel(order_data)
                
                # Remove from open positions for simulation
                del open_positions[position_id]
                
                # Update open positions sheet
                update_open_positions_sheet()
                
                return True
        
        return False
        
    except Exception as e:
            error_msg = str(e)
            print(f"❌ Error executing trade: {error_msg}")
            
            # Enhanced error logging for margin/collateral issues
            if "insufficient margin" in error_msg.lower() or "insufficient funds" in error_msg.lower():
                print(f"💰 MARGIN ERROR DETAILS:")
                print(f"   - Token: {token}")
                print(f"   - Position Size: ${position_size:.2f}")
                print(f"   - Units: {units:.4f}")
                print(f"   - Price: ${price:.4f}")
                print(f"   - Market: {market}")
                
                # Try to get account balance for debugging
                try:
                    balance = exchange.fetch_balance()
                    usdc_balance = balance.get('USDC', {}).get('free', 0)
                    print(f"   - Available USDC: ${usdc_balance:.2f}")
                except:
                    print(f"   - Could not fetch balance")
                
                # Log detailed error to file
                error_log = {
                    'timestamp': datetime.now().isoformat(),
                    'error_type': 'INSUFFICIENT_MARGIN',
                    'token': token,
                    'position_size_usd': position_size,
                    'units': units,
                    'price': price,
                    'market': market,
                    'error_message': error_msg,
                    'usdc_balance': usdc_balance if 'usdc_balance' in locals() else 'Unknown'
                }
                
                # Log error to dedicated error log file
                log_error_to_file(error_log)
                
                # Save error to Excel for tracking
                try:
                    save_order_to_excel({
                        'token': token,
                        'signal_type': signal_type,
                        'price': price,
                        'units': units,
                        'position_size': position_size,
                        'status': 'FAILED',
                        'error': error_msg,
                        'timestamp': datetime.now(),
                        'executed': False
                    })
                    print(f"   - Error logged to Excel")
                except Exception as excel_error:
                    print(f"   - Failed to log error to Excel: {excel_error}")
                    
            elif "asset=" in error_msg:
                print(f"🔧 ASSET ID ERROR:")
                print(f"   - Error: {error_msg}")
                print(f"   - This might be a Hyperliquid-specific asset ID issue")
                print(f"   - Token: {token}")
                print(f"   - Market: {market}")
                
            return False

def main():
    """Main function"""
    print("🚀 Simple Trader - Momentum Strategy")
    print("=" * 50)
    
    # Update configuration from main.py if available
    update_config_from_main()
    
    # Load existing orders and positions
    load_existing_orders()
    
    print(f"Trading Enabled: {'✅ YES' if TRADING_ENABLED else '❌ NO (Simulation)'}")
    print(f"Base Position Size: ${BASE_POSITION_SIZE}")
    print(f"Stiffness Threshold: {STIFFNESS_THRESHOLD}σ")
    print(f"Open Positions: {len(open_positions)}")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging()
    
    # Setup Hyperliquid
    exchange = setup_hyperliquid()
    if not exchange:
        print("❌ Cannot continue without Hyperliquid connection")
        return
    
    # Get market data
    data = get_market_data()
    if data is None:
        print("❌ Cannot continue without market data")
        return
    
    # Calculate signals
    signals = calculate_signals(data)
    if not signals:
        print("❌ No signals calculated")
        return
    
    # Find opportunities
    opportunities = find_trading_opportunities(signals)
    
    if not opportunities:
        print("📊 No trading opportunities found at this time")
        return
    
    print(f"🎯 Found {len(opportunities)} trading opportunities:")
    print()
    
    # Execute trades
    for i, opportunity in enumerate(opportunities, 1):
        print(f"Trade {i}/{len(opportunities)}:")
        success = execute_trade(exchange, opportunity)
        
        if success:
            print(f"✅ Trade {i} completed successfully")
        else:
            print(f"❌ Trade {i} failed")
        
        print("-" * 30)
    
    print("🎯 Trading session completed!")

# Initialize configuration when module is imported
update_config_from_main()

if __name__ == "__main__":
    main() 
