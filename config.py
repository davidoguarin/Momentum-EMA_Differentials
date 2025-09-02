#!/usr/bin/env python3
"""
Configuration module for the Crypto Trading System
Loads environment variables and provides secure access to API keys and credentials
"""

import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for managing environment variables and settings"""
    
    # CoinGecko API Configuration
    COINGECKO_API_KEY = os.environ.get('COINGECKO_API_KEY')
    
    # Hyperliquid Trading Configuration (GitHub Actions compatible)
    HYPERLIQUID_API_KEY = os.environ.get('API_KEY') or os.environ.get('HYPERLIQUID_API_KEY')
    HYPERLIQUID_SECRET = os.environ.get('API_SECRET') or os.environ.get('HYPERLIQUID_SECRET')
    
    # Wallet Configuration
    WALLET_ADDRESS = os.environ.get('WALLET_ADDRESS')
    SEED_PHRASE = os.environ.get('SEED_PHRASE')
    

    
    # Trading Configuration
    BASE_POSITION_SIZE = int(os.environ.get('BASE_POSITION_SIZE', '50'))
    STIFFNESS_THRESHOLD = float(os.environ.get('STIFFNESS_THRESHOLD', '1.5'))
    LEVERAGE_MULTIPLIER = float(os.environ.get('LEVERAGE_MULTIPLIER', '1.0'))
    
    @classmethod
    def validate_config(cls):
        """Validate that all required environment variables are set"""
        required_vars = [
            'COINGECKO_API_KEY',
            'API_KEY', 
            'API_SECRET',
            'WALLET_ADDRESS',
            'SEED_PHRASE'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("   Some features may not work properly")
            return False
        
        logger.info("✅ All required environment variables are set")
        return True
    
    @classmethod
    def get_api_credentials(cls):
        """Get API credentials for CoinGecko"""
        if not cls.COINGECKO_API_KEY:
            logger.warning("⚠️ COINGECKO_API_KEY not set - using public API with rate limits")
            return None
        return cls.COINGECKO_API_KEY
    
    @classmethod
    def get_trading_credentials(cls):
        """Get trading credentials for Hyperliquid"""
        if not cls.HYPERLIQUID_API_KEY or not cls.HYPERLIQUID_SECRET:
            logger.warning("⚠️ Hyperliquid credentials not set - trading will be disabled")
            return None, None
        return cls.HYPERLIQUID_API_KEY, cls.HYPERLIQUID_SECRET
    
    @classmethod
    def get_wallet_config(cls):
        """Get wallet configuration"""
        if not cls.WALLET_ADDRESS or not cls.SEED_PHRASE:
            logger.warning("⚠️ Wallet credentials not set - trading will be disabled")
            return None, None
        return cls.WALLET_ADDRESS, cls.SEED_PHRASE

# Create global config instance
config = Config()

# Export commonly used values
COINGECKO_API_KEY = config.COINGECKO_API_KEY
HYPERLIQUID_API_KEY = config.HYPERLIQUID_API_KEY
HYPERLIQUID_SECRET = config.HYPERLIQUID_SECRET
WALLET_ADDRESS = config.WALLET_ADDRESS
SEED_PHRASE = config.SEED_PHRASE
BASE_POSITION_SIZE = config.BASE_POSITION_SIZE
STIFFNESS_THRESHOLD = config.STIFFNESS_THRESHOLD
LEVERAGE_MULTIPLIER = config.LEVERAGE_MULTIPLIER 
