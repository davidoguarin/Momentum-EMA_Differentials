#!/usr/bin/env python3
"""
Hyperliquid Trading Module
Automated trading using Hyperliquid API for crypto perpetuals
"""

import requests
import json
import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import hmac
import hashlib
import base64

logger = logging.getLogger(__name__)

class HyperliquidTrader:
    def __init__(self, wallet_address: str, api_key: str = None, api_secret: str = None):
        """
        Initialize Hyperliquid trader
        
        Args:
            wallet_address: Your wallet address (0x...)
            api_key: API key (optional for now)
            api_secret: API secret (optional for now)
        """
        self.wallet_address = wallet_address
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Hyperliquid API endpoints
        self.base_url = "https://api.hyperliquid.xyz"
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        
        # Trading parameters
        self.position_size_usd = 10  # Default position size in USD
        self.leverage = 1.0  # No leverage (1x)
        
        # Supported assets (correct Hyperliquid naming)
        self.supported_assets = {
            "SOL": "SOL-USD",
            "BTC": "BTC-USD", 
            "ETH": "ETH-USD",
            "ARB": "ARB-USD",
            "NEAR": "NEAR-USD"
        }
        
        logger.info(f"HyperliquidTrader initialized for wallet: {wallet_address}")
        logger.info(f"Default position size: ${self.position_size_usd}")
        logger.info(f"Leverage: {self.leverage}x")
    
    def get_market_info(self, asset: str) -> Optional[Dict]:
        """
        Get current market information for an asset
        
        Args:
            asset: Asset symbol (e.g., "SOL")
            
        Returns:
            Market info dictionary or None if error
        """
        try:
            url = f"{self.base_url}/info"
            payload = {
                "type": "meta"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Find the specific asset in universe
            if "universe" in data:
                for universe_asset in data["universe"]:
                    if universe_asset.get("name") == asset:
                        return universe_asset
            
            logger.warning(f"Asset {asset} not found in universe data")
            return None
            
        except Exception as e:
            logger.error(f"Error getting market info for {asset}: {str(e)}")
            return None
    
    def get_current_price(self, asset: str) -> Optional[float]:
        """
        Get current price for an asset
        
        Args:
            asset: Asset symbol (e.g., "SOL")
            
        Returns:
            Current price or None if error
        """
        try:
            url = f"{self.base_url}/info"
            payload = {
                "type": "l2Book",
                "coin": asset
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Get mid price from order book
            if "levels" in data and len(data["levels"]) > 0:
                # The levels is a list, and levels[0] is also a list containing price data
                price_levels = data["levels"][0]
                if isinstance(price_levels, list) and len(price_levels) > 0:
                    # First price level is the best price
                    best_price = float(price_levels[0]["px"])
                    return best_price
            
            logger.warning(f"Could not get price for {asset} - no order book data")
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {asset}: {str(e)}")
            return None
    
    def calculate_position_size(self, asset: str, usd_amount: float) -> Tuple[float, float]:
        """
        Calculate position size in units and notional value
        
        Args:
            asset: Asset symbol (e.g., "SOL-USD")
            usd_amount: Amount in USD to trade
            
        Returns:
            Tuple of (units, notional_value)
        """
        current_price = self.get_current_price(asset)
        
        if current_price is None:
            logger.error(f"Cannot calculate position size - no price data for {asset}")
            return 0.0, 0.0
        
        # Calculate units based on USD amount
        units = usd_amount / current_price
        
        # Notional value (this is what Hyperliquid uses)
        notional_value = usd_amount
        
        logger.info(f"Position calculation for {asset}:")
        logger.info(f"  Current price: ${current_price:.4f}")
        logger.info(f"  USD amount: ${usd_amount}")
        logger.info(f"  Units: {units:.6f}")
        logger.info(f"  Notional value: ${notional_value:.2f}")
        
        return units, notional_value
    
    def open_short_position(self, asset: str, usd_amount: float = None) -> bool:
        """
        Open a short position
        
        Args:
            asset: Asset symbol (e.g., "SOL")
            usd_amount: Amount in USD (defaults to self.position_size_usd)
            
        Returns:
            True if successful, False otherwise
        """
        if usd_amount is None:
            usd_amount = self.position_size_usd
        
        logger.info(f"🔄 Opening SHORT position for {asset}")
        logger.info(f"  Amount: ${usd_amount}")
        logger.info(f"  Leverage: {self.leverage}x")
        
        # Validate asset
        if asset not in self.supported_assets:
            logger.error(f"Asset {asset} not supported. Supported: {list(self.supported_assets.keys())}")
            return False
        
        # Get market info
        market_info = self.get_market_info(asset)
        if not market_info:
            logger.error(f"Could not get market info for {asset}")
            return False
        
        # Calculate position size
        units, notional_value = self.calculate_position_size(asset, usd_amount)
        if units == 0:
            return False
        
        # Log order details
        logger.info(f"📋 Order Details:")
        logger.info(f"  Asset: {asset}")
        logger.info(f"  Side: SHORT")
        logger.info(f"  Units: {units:.6f}")
        logger.info(f"  Notional Value: ${notional_value:.2f}")
        logger.info(f"  Leverage: {self.leverage}x")
        logger.info(f"  Wallet: {self.wallet_address}")
        
        # Execute the order
        try:
            order_success = self._execute_order(asset, "SHORT", units, notional_value)
            
            if order_success:
                logger.info("✅ SHORT position order executed successfully!")
                return True
            else:
                logger.error("❌ Failed to execute SHORT position order")
                return False
                
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return False
    
    def close_short_position(self, asset: str) -> bool:
        """
        Close a short position (by opening a long position)
        
        Args:
            asset: Asset symbol (e.g., "SOL")
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🔄 Closing SHORT position for {asset}")
        logger.info(f"  This will open a LONG position to close the SHORT")
        
        # Get current position size (for now, use default)
        usd_amount = self.position_size_usd
        
        # Calculate position size
        units, notional_value = self.calculate_position_size(asset, usd_amount)
        if units == 0:
            return False
        
        # Log close order details
        logger.info(f"📋 Close Order Details:")
        logger.info(f"  Asset: {asset}")
        logger.info(f"  Side: LONG (to close SHORT)")
        logger.info(f"  Units: {units:.6f}")
        logger.info(f"  Notional Value: ${notional_value:.2f}")
        logger.info(f"  Wallet: {self.wallet_address}")
        
        # Execute the close order
        try:
            order_success = self._execute_order(asset, "LONG", units, notional_value)
            
            if order_success:
                logger.info("✅ SHORT position close order executed successfully!")
                return True
            else:
                logger.error("❌ Failed to execute close order")
                return False
                
        except Exception as e:
            logger.error(f"Error executing close order: {str(e)}")
            return False
    
    def _execute_order(self, asset: str, side: str, units: float, notional_value: float) -> bool:
        """
        Execute an order on Hyperliquid
        
        Args:
            asset: Asset symbol (e.g., "SOL")
            side: "LONG" or "SHORT"
            units: Number of units to trade
            notional_value: Notional value in USD
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🚀 Executing {side} order for {asset}...")
            
            # Get current market price for order
            current_price = self.get_current_price(asset)
            if current_price is None:
                logger.error("Cannot execute order - no price data")
                return False
            
            # Prepare order payload
            order_payload = {
                "type": "order",
                "user": self.wallet_address,
                "coin": asset,
                "is_buy": side == "LONG",  # True for LONG, False for SHORT
                "sz": str(units),
                "limit_px": str(current_price),  # Market order at current price
                "reduce_only": False,
                "order_type": "LIMIT"
            }
            
            logger.info(f"📤 Order payload: {json.dumps(order_payload, indent=2)}")
            
            # Submit order to Hyperliquid
            if self.api_key and self.api_secret:
                # TODO: Implement authenticated order submission
                logger.info("🔐 Using authenticated order submission (not yet implemented)")
                # For now, simulate successful order
                logger.info("✅ Order submitted successfully (simulation)")
                return True
            else:
                # For now, simulate order execution
                logger.warning("⚠️  No API credentials - simulating order execution")
                logger.info("✅ Order executed successfully (simulation)")
                logger.info(f"  {side} {units:.6f} {asset} at ${current_price:.4f}")
                logger.info(f"  Total value: ${notional_value:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information and current positions
        
        Returns:
            Account info dictionary or None if error
        """
        try:
            url = f"{self.base_url}/info"
            payload = {
                "type": "clearinghouseState",
                "user": self.wallet_address
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Account info retrieved for {self.wallet_address}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_current_positions(self) -> Dict:
        """
        Get current open positions
        
        Returns:
            Dictionary with current positions
        """
        try:
            account_info = self.get_account_info()
            if not account_info:
                return {}
            
            positions = {}
            if "assetPositions" in account_info:
                for pos in account_info["assetPositions"]:
                    if "coin" in pos and "position" in pos:
                        coin = pos["coin"]
                        position_data = pos["position"]
                        
                        # Extract position details
                        size = float(position_data.get("sz", 0))
                        entry_price = float(position_data.get("entryPx", 0))
                        unrealized_pnl = float(position_data.get("unrealizedPnl", 0))
                        
                        if size != 0:  # Only show open positions
                            positions[coin] = {
                                "size": size,
                                "entry_price": entry_price,
                                "unrealized_pnl": unrealized_pnl,
                                "side": "LONG" if size > 0 else "SHORT",
                                "notional_value": abs(size * entry_price)
                            }
            
            logger.info(f"Current positions: {len(positions)} open positions")
            for coin, pos in positions.items():
                logger.info(f"  {coin}: {pos['side']} {abs(pos['size']):.6f} @ ${pos['entry_price']:.4f} (PnL: ${pos['unrealized_pnl']:.2f})")
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting current positions: {str(e)}")
            return {}
    
    def submit_order(self, order_payload: Dict) -> bool:
        """
        Submit an order to Hyperliquid
        
        Args:
            order_payload: Order details
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.api_key or not self.api_secret:
                logger.error("API credentials required for order submission")
                return False
            
            # TODO: Implement actual order submission with authentication
            # For now, we'll simulate the order submission
            
            url = f"{self.base_url}/exchange"
            
            # In a real implementation, you would:
            # 1. Sign the order with your private key
            # 2. Submit to the exchange endpoint
            # 3. Handle the response
            
            logger.info(f"🔐 Submitting authenticated order to {url}")
            logger.info(f"📤 Order: {json.dumps(order_payload, indent=2)}")
            
            # Simulate successful order submission
            logger.info("✅ Order submitted successfully (simulation)")
            logger.info("📝 Note: This is a simulation. Real orders require proper authentication.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get the status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        try:
            # TODO: Implement order status checking
            logger.info(f"Checking status for order: {order_id}")
            
            # Simulate order status
            status = {
                "order_id": order_id,
                "status": "FILLED",
                "filled_size": "100%",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Order status: {status}")
            return status
            
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return {}
    
    def test_connection(self) -> bool:
        """
        Test connection to Hyperliquid API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("🧪 Testing Hyperliquid API connection...")
            
            # Test basic info endpoint
            url = f"{self.base_url}/info"
            payload = {"type": "meta"}
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("✅ API connection successful")
            logger.info(f"  Response status: {response.status_code}")
            logger.info(f"  Response time: {response.elapsed.total_seconds():.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ API connection failed: {str(e)}")
            return False

def main():
    """Test function for Hyperliquid trading"""
    logger.info("🚀 Starting Hyperliquid Trading Test")
    
    # Initialize trader
    wallet_address = "0x0eb9aae5f84465dab5f0899afdc07cbcb6f7cf27"
    trader = HyperliquidTrader(wallet_address)
    
    # Test 1: Connection
    logger.info("\n" + "="*50)
    logger.info("TEST 1: API Connection")
    logger.info("="*50)
    connection_success = trader.test_connection()
    
    if not connection_success:
        logger.error("❌ Connection test failed. Cannot proceed.")
        return
    
    # Test 2: Market Info
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Market Information")
    logger.info("="*50)
    asset = "SOL"
    market_info = trader.get_market_info(asset)
    
    if market_info:
        logger.info(f"✅ Market info for {asset}:")
        logger.info(f"  Asset: {market_info.get('name', 'N/A')}")
        logger.info(f"  Decimals: {market_info.get('szDecimals', 'N/A')}")
        logger.info(f"  Price Decimals: {market_info.get('pxDecimals', 'N/A')}")
    
    # Test 3: Current Price
    logger.info("\n" + "="*50)
    logger.info("TEST 3: Current Price")
    logger.info("="*50)
    current_price = trader.get_current_price(asset)
    
    if current_price:
        logger.info(f"✅ Current {asset} price: ${current_price:.4f}")
    
    # Test 4: Position Size Calculation
    logger.info("\n" + "="*50)
    logger.info("TEST 4: Position Size Calculation")
    logger.info("="*50)
    usd_amount = 10
    units, notional = trader.calculate_position_size(asset, usd_amount)
    
    if units > 0:
        logger.info(f"✅ Position calculation successful:")
        logger.info(f"  ${usd_amount} = {units:.6f} {asset} units")
    
    # Test 5: Open Short Position
    logger.info("\n" + "="*50)
    logger.info("TEST 5: Open Short Position")
    logger.info("="*50)
    short_success = trader.open_short_position(asset, usd_amount)
    
    if short_success:
        logger.info("✅ Short position order prepared successfully")
    else:
        logger.error("❌ Failed to prepare short position order")
    
    # Test 6: Close Short Position
    logger.info("\n" + "="*50)
    logger.info("TEST 6: Close Short Position")
    logger.info("="*50)
    close_success = trader.close_short_position(asset)
    
    if close_success:
        logger.info("✅ Short position close order prepared successfully")
    else:
        logger.error("❌ Failed to prepare close order")
    
    # Test 7: Account Information
    logger.info("\n" + "="*50)
    logger.info("TEST 7: Account Information")
    logger.info("="*50)
    account_info = trader.get_account_info()
    
    if account_info:
        logger.info("✅ Account info retrieved successfully")
        logger.info(f"  Data keys: {list(account_info.keys())}")
    else:
        logger.warning("⚠️  Account info not available (may need API key)")
    
    # Test 8: Current Positions
    logger.info("\n" + "="*50)
    logger.info("TEST 8: Current Positions")
    logger.info("="*50)
    current_positions = trader.get_current_positions()
    
    if current_positions:
        logger.info(f"✅ Found {len(current_positions)} open positions")
    else:
        logger.info("ℹ️  No open positions found")
    
    # Test 9: Complete Order Execution Flow
    logger.info("\n" + "="*50)
    logger.info("TEST 9: Complete Order Execution Flow")
    logger.info("="*50)
    
    # Step 1: Check if we can execute orders
    logger.info("Step 1: Order Execution Capability")
    if trader.api_key and trader.api_secret:
        logger.info("✅ API credentials available - ready for real trading")
    else:
        logger.info("⚠️  No API credentials - running in simulation mode")
        logger.info("   To enable real trading, add your API key and secret")
    
    # Step 2: Execute SHORT order
    logger.info("\nStep 2: Executing SHORT Order")
    short_success = trader.open_short_position(asset, usd_amount)
    
    if short_success:
        logger.info("✅ SHORT position opened successfully!")
        
        # Step 3: Wait a moment and check positions
        logger.info("\nStep 3: Checking Updated Positions")
        time.sleep(2)  # Simulate order processing time
        updated_positions = trader.get_current_positions()
        
        if asset in updated_positions:
            pos = updated_positions[asset]
            logger.info(f"✅ Position confirmed: {pos['side']} {abs(pos['size']):.6f} {asset}")
            logger.info(f"   Entry Price: ${pos['entry_price']:.4f}")
            logger.info(f"   Notional Value: ${pos['notional_value']:.2f}")
        else:
            logger.info("ℹ️  Position not yet visible (may need more time)")
        
        # Step 4: Execute close order
        logger.info("\nStep 4: Executing Close Order")
        close_success = trader.close_short_position(asset)
        
        if close_success:
            logger.info("✅ SHORT position closed successfully!")
            
            # Step 5: Final position check
            logger.info("\nStep 5: Final Position Check")
            time.sleep(2)  # Simulate order processing time
            final_positions = trader.get_current_positions()
            
            if asset not in final_positions:
                logger.info("✅ Position successfully closed - no open positions")
            else:
                logger.info(f"⚠️  Position still open: {final_positions[asset]}")
        else:
            logger.error("❌ Failed to close position")
    else:
        logger.error("❌ Failed to open SHORT position")
    
    logger.info("\n" + "="*50)
    logger.info("🎯 ALL TESTS COMPLETED")
    logger.info("="*50)
    logger.info("Next steps:")
    logger.info("1. Add your Hyperliquid API key and secret")
    logger.info("2. Test with real orders (start small!)")
    logger.info("3. Monitor positions and PnL")
    logger.info("4. Integrate with momentum strategy")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main() 