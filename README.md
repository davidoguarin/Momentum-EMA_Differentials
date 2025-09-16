# Crypto Momentum Trading System

## Strategy Overview

This cryptocurrency trading system employs a multi-layered momentum framework designed to capture the acceleration of price trends by analyzing the dynamics of Exponential Moving Average (EMA) differentials rather than relying solely on standard crossover signals. The core mechanism measures the rate of change in the EMA spread and integrates trading volume patterns as a confirmation filter. Signals are generated when the EMA differential momentum surpasses an adaptive threshold derived from statistical modeling and recent volatility levels, provided that the move is corroborated by rising volume. This methodology enables earlier and more robust entries compared to traditional crossover strategies. Position sizing is dynamically adjusted according to the strength of the signal (measured as momentum stiffness), while risk is tightly managed through volume-driven exit conditions and token-specific leverage constraints.

## Table of Contents

- [System Overview](#system-overview)
- [Trading Strategy](#trading-strategy)
- [Position Management](#position-management)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [API Integration](#api-integration)
- [Results & Analysis](#results--analysis)
- [Troubleshooting](#troubleshooting)

## System Overview

This system combines **EMA Analysis**, **Momentum Portfolio Simulation**, and **Real Trading Execution** to create a comprehensive crypto trading solution. It analyzes 12 major cryptocurrencies (ARB, BTC, ETH, NEAR, SOL, SUI, TRX, XRP, AAVE, ADA, ENA, DOGE) and executes trades based on sophisticated momentum signals.

### Key Features
- **Multi-token Analysis**: Simultaneously analyzes 12 cryptocurrencies
- **Momentum-based Signals**: Uses EMA slope and volume analysis
- **Stiffness-based Position Sizing**: Adjusts position size based on signal strength
- **Leverage Management**: Dynamic leverage allocation per token
- **Real-time Trading**: Live execution via Hyperliquid API
- **Comprehensive Logging**: Detailed trade tracking and performance analysis

## Trading Strategy

### Signal Generation Criteria

#### BUY Signals (Open Long Position)
**Primary Conditions:**
1. **EMA Slope > Adaptive Threshold**
   - Short EMA (5-day) slope > Long EMA (15-day) slope
   - Slope must exceed: `Mean + (0.5 × Standard Deviation)` over 30-day window
   
2. **Volume Confirmation**
   - Volume EMA slope must be positive (> 0)
   - Ensures momentum is supported by trading volume

**Formula:**
```
BUY Signal = (EMA_5d_slope > Threshold) AND (Volume_EMA_slope > 0)
Threshold = Rolling_30d_Mean + (0.5 × Rolling_30d_StdDev)
```

#### SELL Signals (Close Long Position)
**Primary Conditions:**
1. **Momentum Reversal**
   - EMA slope turns negative (< 0)
   - Indicates momentum is losing strength

2. **Volume-based Exit (SELL_VOLUME)**
   - Volume EMA slope < 0 (declining volume)
   - Volume EMA difference < 10% (low volume)
   - Protects against low-liquidity exits

**Formula:**
```
SELL Signal = (EMA_slope < 0)
SELL_VOLUME = (Volume_EMA_slope < 0) AND (Volume_EMA_diff < 10%)
```

### Position Sizing Logic

#### Base Position Size
- **Default**: $50 USD per trade
- **Configurable**: Adjustable via `BASE_POSITION_SIZE` parameter

#### Stiffness-based Multiplier
**Stiffness Calculation:**
```
Stiffness = (EMA_Slope - Threshold) / Standard_Deviation
```

**Position Multiplier:**
- **Normal Signal**: 1.0x (Base position size)
- **Strong Signal**: 2.0x (Double position size)
- **Threshold**: Configurable (default: 3.0σ above mean)

**Formula:**
```
Position_Size = Base_Size × Stiffness_Multiplier × Leverage
```

#### Leverage Management
**Per-Token Maximum Leverage:**
- **ARB**: 10x
- **BTC**: 40x  
- **ETH**: 25x
- **NEAR**: 10x
- **SOL**: 20x
- **SUI**: 10x
- **TRX**: 10x
- **XRP**: 20x

**Leverage Multiplier:**
- **Configurable**: 0.0 to 1.0 (0% to 100% of max leverage)
- **Default**: 1.0 (100% of maximum allowed leverage)

**Example:**
```
XRP Base Position: $50
Max Leverage: 20x
Leverage Multiplier: 1.0
Final Position: $50 × 20x = $1,000 USD
```

## Position Management

### Entry Conditions
1. **Signal Confirmation**: BUY signal generated
2. **Margin Check**: Sufficient USDC balance
3. **Risk Validation**: Position size within limits

### Exit Conditions
1. **Signal-based**: SELL or SELL_VOLUME signal
2. **Manual Override**: Force close via API
3. **Risk Management**: Stop-loss (configurable)

### Position Tracking
- **Real-time Monitoring**: Live position status
- **PnL Calculation**: USD and percentage returns
- **Trade History**: Complete execution log
- **Performance Metrics**: Win rate, total returns

## Installation & Setup

### Prerequisites
- Python 3.8+
- SQLite3
- Hyperliquid API access

### Installation
```bash
# Clone repository
git clone <repository-url>
cd Momentum-CrossingEMAs

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_template.txt .env
# Edit .env with your actual API credentials
```

### Environment Variables
```bash
# Required
COINGECKO_API_KEY=your_coingecko_api_key
WALLET_ADDRESS=your_wallet_address
SEED_PHRASE=your_seed_phrase

# Optional
BASE_POSITION_SIZE=50
STIFFNESS_THRESHOLD=1.5
LEVERAGE_MULTIPLIER=1.0
HYPERLIQUID_TESTNET=true  # Use testnet for development
```

## Configuration

### Main Configuration (`main.py`)
```python
# Core Settings
EXTRACT_DATA = False              # Data acquisition
RUN_EMA_ANALYSIS = True          # EMA calculation
RUN_PORTFOLIO_SIMULATION = True  # Portfolio simulation
TRADING_ENABLED = True           # Live trading
DISPLAY_PLOTS = False            # Plot display

# Trading Parameters
BASE_POSITION_SIZE = 50          # USD per trade
STIFFNESS_THRESHOLD = 3.0        # σ threshold for doubling
LEVERAGE_MULTIPLIER = 1.0        # 0.0 to 1.0

# Analysis Parameters
MOMENTUM_SHORT_PERIOD = 5        # Short EMA days
MOMENTUM_LONG_PERIOD = 15        # Long EMA days
MOMENTUM_SLOPE_WINDOW = 30       # Slope calculation window
MOMENTUM_SIGMA_MULTIPLIER = 0.5  # Threshold multiplier
```

### Trading Configuration (`simple_trader.py`)
```python
# Risk Management
MAX_POSITION_SIZE_USD = 10000    # Maximum position size
MIN_MARGIN_RATIO = 0.5           # Minimum margin requirement

# Order Parameters
ORDER_TIMEOUT = 30               # Order timeout (seconds)
SLIPPAGE_TOLERANCE = 0.02        # 2% slippage tolerance
```

## Usage

### Quick Start
```bash
# Run complete analysis and trading
python main.py

# Run only EMA analysis
python ema_analysis.py

# Run only portfolio simulation
python portfolio_simulation.py
```

## GitHub Deployment

### Security Best Practices
**Repository Secrets**: All API keys and credentials are stored in GitHub repository secrets  
**Git Ignore**: Sensitive files are automatically excluded from commits  
**GitHub Actions**: Automated deployment with secure credential management  

### Deployment Options

#### **Option 1: GitHub Actions (Recommended)**
1. **Fork/Clone** the repository to your GitHub account
2. **Set Repository Secrets** in GitHub:
   - Go to `Settings` → `Secrets and variables` → `Actions`
   - Add the following secrets:
     - `COINGECKO_API_KEY`
     - `WALLET_ADDRESS`
     - `SEED_PHRASE`
3. **Enable GitHub Actions** - The workflow will run automatically
4. **Monitor** execution in the Actions tab

#### **Option 2: Local Development**
1. **Clone** the repository to your local machine
2. **Copy Environment Template**:
   ```bash
   cp env_template.txt .env
   ```
3. **Edit `.env`** with your actual credentials
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the System**:
   ```bash
   python main.py
   ```

### Environment Variables Reference
| Variable | Description | Required | GitHub Secret | Example |
|----------|-------------|----------|---------------|---------|
| `COINGECKO_API_KEY` | CoinGecko API key for data | Yes | `COINGECKO_API_KEY` | `CG-abc123...` |
| `WALLET_ADDRESS` | Your wallet address | Yes | `WALLET_ADDRESS` | `0x1234...` |
| `SEED_PHRASE` | Your wallet seed phrase | Yes | `SEED_PHRASE` | `word1 word2...` |
| `BASE_POSITION_SIZE` | Base position size in USD | No | `BASE_POSITION_SIZE` | `50` |
| `STIFFNESS_THRESHOLD` | Threshold for doubling positions | No | `STIFFNESS_THRESHOLD` | `1.5` |
| `LEVERAGE_MULTIPLIER` | Leverage multiplier (0.0-1.0) | No | `LEVERAGE_MULTIPLIER` | `1.0` |

### Data Acquisition
```python
# Extract fresh data from APIs
EXTRACT_DATA = True
python main.py
```

### Live Trading
```python
# Enable real trading
TRADING_ENABLED = True
python main.py
```

### Simulation Only
```python
# Run simulation without trading
TRADING_ENABLED = False
python main.py
```

## System Architecture

### Core Modules

#### 1. **Data Acquisition** (`data_acquisition.py`)
- Fetches price and volume data from multiple sources
- Supports real-time and historical data
- Data validation and cleaning

#### 2. **EMA Analysis** (`ema_analysis.py`)
- Calculates 5-day and 15-day EMAs
- Computes slope and threshold values
- Generates initial trading signals

#### 3. **Portfolio Simulation** (`portfolio_simulation.py`)
- Runs momentum-based simulation
- Calculates stiffness and position sizing
- Generates final trading opportunities

#### 4. **Trading Execution** (`simple_trader.py`)
- Executes trades via Hyperliquid API
- Manages position lifecycle
- Tracks performance and PnL

#### 5. **Database Management** (`database_manager.py`)
- SQLite database operations
- Data persistence and retrieval
- Performance metrics storage

### Data Flow
```
Data Sources → Data Acquisition → EMA Analysis → Portfolio Simulation → Trading Execution
     ↓              ↓                ↓              ↓                    ↓
  Price/Volume → Clean Data → Signal Generation → Opportunity ID → Order Execution
```

## API Integration

### Hyperliquid Integration
- **Futures Trading**: Long positions only
- **Market Orders**: Immediate execution
- **Position Management**: Real-time tracking
- **Risk Controls**: Margin validation

### API Endpoints Used
- **Market Data**: Price, volume, order book
- **Trading**: Market buy/sell orders
- **Account**: Balance, positions, margin
- **Risk**: Position limits, margin requirements

## Results & Analysis

### Output Files
- **Excel Reports**: `results/momentum_portfolio_simulation_latest.xlsx`
- **Plots**: `results/momentum_portfolio_simulation_latest.png`
- **Logs**: `logs/crypto_ema.log`

### Performance Metrics
- **Total PnL**: Overall portfolio return
- **Win Rate**: Percentage of profitable trades
- **Trade Count**: Number of executed trades
- **Position Sizing**: Stiffness-based allocation

### Analysis Sheets
1. **Summary**: Overall performance metrics
2. **All_Trades**: Complete trade history
3. **Signals**: Per-token signal analysis
4. **Trades**: Per-token trade execution

## Troubleshooting

### Common Issues

#### 1. **Insufficient Margin**
```
Error: "Insufficient margin to place order"
Solution: Check USDC balance and leverage settings
```

#### 2. **API Connection Issues**
```
Error: "Failed to connect to Hyperliquid"
Solution: Verify API credentials and network connection
```

#### 3. **Data Quality Issues**
```
Error: "Insufficient data for analysis"
Solution: Enable EXTRACT_DATA and run data acquisition
```

#### 4. **Position Tracking Errors**
```
Error: "Position not found"
Solution: Check database connection and refresh positions
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization
- **Database Indexing**: Optimize query performance
- **Memory Management**: Monitor memory usage during simulation
- **API Rate Limiting**: Respect exchange rate limits

## Advanced Features

### Custom Strategies
- **Signal Modification**: Adjust BUY/SELL criteria
- **Position Sizing**: Custom stiffness calculations
- **Risk Management**: Dynamic stop-loss and take-profit

### Backtesting
- **Historical Simulation**: Test strategies on past data
- **Performance Analysis**: Risk-adjusted returns
- **Strategy Comparison**: Multiple approach evaluation

### Portfolio Management
- **Risk Allocation**: Per-token position limits
- **Correlation Analysis**: Diversification optimization
- **Rebalancing**: Dynamic portfolio adjustment

## Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 standards
2. **Testing**: Include unit tests for new features
3. **Documentation**: Update README for changes
4. **Error Handling**: Implement robust error handling

### Testing
```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

**Risk Warning**: Cryptocurrency trading involves substantial risk of loss. This system is for educational and research purposes. Always:
- Test thoroughly on testnet before live trading
- Start with small position sizes
- Monitor positions continuously
- Understand the risks involved

**Not Financial Advice**: This documentation does not constitute financial advice. Always consult with qualified financial professionals before making investment decisions.

---

## Quick Reference

### Trading Criteria Summary
- **BUY**: EMA slope > threshold + volume confirmation
- **SELL**: EMA slope < 0 or volume decline
- **Position Size**: Base × Stiffness × Leverage
- **Leverage**: Token-specific maximum × multiplier

### Key Commands
```bash
python main.py                    # Full system
python ema_analysis.py           # EMA analysis only
python portfolio_simulation.py    # Simulation only
```

### Configuration Files
- `main.py` - Main configuration
- `config.py` - System settings
- `api_config.py` - API credentials
- `.env` - Environment variables

---

**Last Updated**: August 29, 2025  
**Status**: Production Ready 
