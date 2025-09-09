#!/usr/bin/env python3
"""
Data acquisition module for Crypto PCA Analysis
Handles fetching data from CoinGecko API and storing in Excel
"""

import logging
import os
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd
from tqdm import tqdm
from database_manager import CryptoDatabase

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

logger = logging.getLogger(__name__)

class CoinGeckoAPI:
    """Wrapper for CoinGecko API calls"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set headers
        if api_key:
            self.session.headers.update({
                'X-CG-API-Key': api_key
            })
        
        # Rate limiting - be more conservative
        self.calls_per_minute = 20 if not api_key else 1000  # Reduced from 30 to 20
        self.last_call_time = 0
        self.call_count = 0
    
    def _rate_limit(self):
        """Implement rate limiting with more conservative approach"""
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - self.last_call_time >= 60:
            self.call_count = 0
            self.last_call_time = current_time
        
        # If we've hit the limit, wait
        if self.call_count >= self.calls_per_minute:
            wait_time = 60 - (current_time - self.last_call_time) + 5  # Add 5 seconds buffer
            logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            self.call_count = 0
            self.last_call_time = time.time()
        
        self.call_count += 1
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting and retry logic"""
        max_retries = 5  # Increased from 3 to 5
        retry_delay = 15  # Wait 15 seconds on rate limit (increased from 10)
        
        for attempt in range(max_retries):
            self._rate_limit()
            
            url = f"{self.base_url}/{endpoint}"
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 429:  # Rate limit hit
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if response.status_code == 429:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logger.error(f"API request failed: {e}")
                    if attempt == max_retries - 1:
                        return {}
        
        logger.error(f"Failed to make request after {max_retries} attempts")
        return {}
    
    def get_top_coins_by_volume(self, limit: int = 200) -> List[Dict]:
        """
        Fetch top coins by trading volume from CoinGecko API
        
        Args:
            limit: Number of coins to fetch (default: 200)
            
        Returns:
            List of coin data dictionaries
        """
        logger.info(f"Fetching top {limit} coins by trading volume...")
        
        # Use multiple pages to get all 200 coins
        all_coins = []
        page = 1
        per_page = 250  # Maximum per page
        
        while len(all_coins) < limit:
            endpoint = "coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'volume_desc',  # Order by volume
                'per_page': per_page,
                'page': page,
                'sparkline': False,
                'price_change_percentage': '24h'
            }
            
            data = self._make_request(endpoint, params)
            
            if not data:
                logger.error(f"Failed to fetch data for page {page}")
                break
            
            all_coins.extend(data)
            
            # If we got less than per_page, we've reached the end
            if len(data) < per_page:
                break
            
            page += 1
        
        # Limit to requested number
        all_coins = all_coins[:limit]
        
        logger.info(f"Successfully fetched {len(all_coins)} coins")
        return all_coins
    
    def get_coin_historical_data(self, coin_id: str, days: int = 365) -> Dict:
        """
        Fetch historical price data for a specific coin
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data
            
        Returns:
            Dictionary containing historical price data
        """
        logger.info(f"Fetching {days} days of historical data for {coin_id}...")
        
        endpoint = f"coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days
        }
        
        data = self._make_request(endpoint, params)
        
        if not data:
            logger.warning(f"No historical data found for {coin_id}")
            return {}
        
        return data
    
    def get_last_hour_prices_batch(self, coin_ids: List[str]) -> Dict[str, float]:
        """
        Fetch the price from the last hour for multiple coins in a single request
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            
        Returns:
            Dictionary mapping coin_id to last hour's price
        """
        logger.info(f"Fetching current hour prices for {len(coin_ids)} coins...")
        
        # Use the simple price endpoint which supports multiple coins
        endpoint = "simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd'
        }
        
        data = self._make_request(endpoint, params)
        
        if data:
            last_hour_prices = {}
            for coin_id in coin_ids:
                if coin_id in data and 'usd' in data[coin_id]:
                    price = data[coin_id]['usd']
                    last_hour_prices[coin_id] = price
                    logger.info(f"Current hour price for {coin_id}: ${price}")
                else:
                    logger.warning(f"Could not fetch current hour price for {coin_id}")
                    last_hour_prices[coin_id] = None
            
            return last_hour_prices
        else:
            logger.warning(f"Could not fetch current hour prices for any coins")
            return {coin_id: None for coin_id in coin_ids}

    def get_coin_historical_data_batch(self, coin_ids: List[str], days: int = 365) -> Dict:
        """
        Fetch historical data for multiple coins in batch (if supported)
        This reduces API calls significantly
        
        Args:
            coin_ids: List of coin IDs
            days: Number of days of historical data
            
        Returns:
            Dictionary with coin_id as key and historical data as value
        """
        logger.info(f"Fetching historical data for {len(coin_ids)} coins in batch...")
        
        batch_data = {}
        
        # Process in smaller batches to avoid overwhelming the API
        batch_size = 5  # Reduced from 10 to 5 for more conservative approach
        for i in range(0, len(coin_ids), batch_size):
            batch = coin_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: coins {i+1}-{min(i+batch_size, len(coin_ids))}")
            
            for coin_id in batch:
                historical_data = self.get_coin_historical_data(coin_id, days)
                if historical_data:
                    batch_data[coin_id] = historical_data
                else:
                    logger.warning(f"Failed to get data for {coin_id}")
            
            # Add delay between batches to be more conservative
            if i + batch_size < len(coin_ids):
                logger.info("Waiting 2 seconds between batches...")
                time.sleep(2)
        
        return batch_data

    def get_current_hour_price(self, coin_id: str) -> Optional[float]:
        """
        Fetch the current hour price for a specific coin
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            Current hour's price or None if failed
        """
        logger.info(f"Fetching current hour price for {coin_id}...")
        
        # Use the simple price endpoint for current price
        endpoint = "simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd'
        }
        
        data = self._make_request(endpoint, params)
        
        if data and coin_id in data and 'usd' in data[coin_id]:
            price = data[coin_id]['usd']
            logger.info(f"Current hour price for {coin_id}: ${price}")
            return price
        else:
            logger.warning(f"Could not fetch current hour price for {coin_id}")
            return None

def process_historical_data(historical_data: Dict, coin_symbol: str) -> pd.DataFrame:
    """
    Process historical data into a pandas DataFrame
    
    Args:
        historical_data: Raw historical data from API
        coin_symbol: Symbol of the coin
        
    Returns:
        DataFrame with daily prices (including the last available API data point)
    """
    if not historical_data or 'prices' not in historical_data:
        return pd.DataFrame()
    
    # Extract prices data
    prices_data = historical_data['prices']
    
    # Convert to DataFrame
    df = pd.DataFrame(prices_data, columns=['timestamp', 'price'])
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Get today's date at midnight
    today_midnight = pd.Timestamp.now().normalize()
    
    # Filter to include data up to today at 00:00 (not beyond)
    df = df[df['date'] <= today_midnight]
    
    # If no data left after filtering, return empty DataFrame
    if df.empty:
        return pd.DataFrame()
    
    # Create daily time series directly from the raw data
    result_df = extract_daily_data_from_raw_simple(df, coin_symbol)
    
    # Add the last available data point from the API as the final point
    if not result_df.empty and len(prices_data) > 0:
        # Get the last timestamp and price from the API data
        last_api_timestamp = prices_data[-1][0]  # Last timestamp from API
        last_api_price = prices_data[-1][1]      # Last price from API
        
        # Convert timestamp to datetime
        last_api_datetime = pd.to_datetime(last_api_timestamp, unit='ms')
        
        # Only add if this timestamp is different from the last one in our result
        if last_api_datetime != result_df.index[-1]:
            # Add the last API data point
            last_api_data = pd.DataFrame({
                coin_symbol: [last_api_price]
            }, index=[last_api_datetime])
            
            # Append the last API data point
            result_df = pd.concat([result_df, last_api_data])
            logger.info(f"Added last API data point for {coin_symbol}: ${last_api_price} at {last_api_datetime}")
        else:
            logger.info(f"Last API data point already exists in result for {coin_symbol}")
    
    return result_df

def process_historical_data_with_volume(historical_data: Dict, coin_symbol: str, last_hour_price: Optional[float] = None) -> pd.DataFrame:
    """
    Process historical data into a pandas DataFrame with both price and volume
    
    Args:
        historical_data: Raw historical data from API
        coin_symbol: Symbol of the coin
        last_hour_price: Last hour's price to add as last point (optional, will use last API data if not provided)
        
    Returns:
        DataFrame with daily prices and volumes
    """
    if not historical_data or 'prices' not in historical_data or 'total_volumes' not in historical_data:
        return pd.DataFrame()
    
    # Extract prices and volumes data
    prices_data = historical_data['prices']
    volumes_data = historical_data['total_volumes']
    
    # Convert to DataFrame
    prices_df = pd.DataFrame(prices_data, columns=['timestamp', 'price'])
    volumes_df = pd.DataFrame(volumes_data, columns=['timestamp', 'volume'])
    
    # Convert timestamp to datetime
    prices_df['date'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
    volumes_df['date'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
    
    # Get today's date at midnight
    today_midnight = pd.Timestamp.now().normalize()
    
    # Filter to include data up to today at 00:00 (not beyond)
    prices_df = prices_df[prices_df['date'] <= today_midnight]
    volumes_df = volumes_df[volumes_df['date'] <= today_midnight]
    
    # If no data left after filtering, return empty DataFrame
    if prices_df.empty or volumes_df.empty:
        return pd.DataFrame()
    
    # Create daily time series directly from the raw data
    result_df = extract_daily_data_from_raw(prices_df, volumes_df, coin_symbol)
    
    # Add the last available data point from the API as the final point
    if not result_df.empty and len(prices_data) > 0:
        # Get the last timestamp and price from the API data
        last_api_timestamp = prices_data[-1][0]  # Last timestamp from API
        last_api_price = prices_data[-1][1]      # Last price from API
        
        # Convert timestamp to datetime
        last_api_datetime = pd.to_datetime(last_api_timestamp, unit='ms')
        
        # Get the last volume from the API data
        last_api_volume = volumes_data[-1][1] if len(volumes_data) > 0 else 0
        
        # IMPORTANT: Round the last API timestamp to the beginning of the current hour
        # This ensures we have exactly one point at the current hour, not multiple random minutes
        current_hour_beginning = last_api_datetime.replace(minute=0, second=0, microsecond=0)
        
        # CRITICAL FIX: Check if we already have ANY data point at the current hour
        # This prevents adding multiple points when crossing day boundaries
        existing_hour_points = result_df[result_df.index.hour == current_hour_beginning.hour]
        
        if existing_hour_points.empty:
            # No data point exists for this hour, add it
            last_api_data = pd.DataFrame({
                f'{coin_symbol}_price': [last_api_price],
                f'{coin_symbol}_volume': [last_api_volume]
            }, index=[current_hour_beginning])
            
            # Append the last API data point
            result_df = pd.concat([result_df, last_api_data])
            logger.info(f"Added last API data point for {coin_symbol}: ${last_api_price} at {current_hour_beginning} (rounded to hour beginning)")
        else:
            logger.info(f"Data point already exists for hour {current_hour_beginning.hour}:00 for {coin_symbol}, skipping duplicate")
    
    return result_df

def extract_daily_data_from_raw(prices_df: pd.DataFrame, volumes_df: pd.DataFrame, coin_symbol: str) -> pd.DataFrame:
    """
    Extract exactly 00:00 and 12:00 data points from raw API data for exactly 365 days
    
    Args:
        prices_df: DataFrame with timestamp and price data
        volumes_df: DataFrame with timestamp and volume data
        coin_symbol: Symbol of the coin
        
    Returns:
        DataFrame with exactly 00:00 and 12:00 points for each day (730 points total)
    """
    if prices_df.empty or volumes_df.empty:
        return pd.DataFrame()
    
    # Get the date range from the data
    start_date = prices_df['date'].min().normalize()
    end_date = prices_df['date'].max().normalize()
    
    # Calculate how many days we have
    days_diff = (end_date - start_date).days
    
    # We need exactly 365 days, so we'll use the most recent 365 days
    if days_diff >= 364:  # We have at least 365 days
        # Use the last 365 days (most recent data)
        start_date = end_date - pd.Timedelta(days=364)
        logger.info(f"Using the most recent 365 days for {coin_symbol} (from {start_date} to {end_date})")
    else:
        # We don't have enough days, adjust to use what we have
        logger.warning(f"Only {days_diff + 1} days of data available for {coin_symbol}, need 365 days")
        # Use all available days and pad with the last available data
        end_date = start_date + pd.Timedelta(days=days_diff)
    
    # Create daily time points at 00:00 and 12:00 for exactly 365 days
    daily_times = []
    current_date = start_date
    
    for day in range(365):  # Exactly 365 days
        # Add 00:00 point
        daily_times.append(current_date)
        # Add 12:00 point
        daily_times.append(current_date + pd.Timedelta(hours=12))
        current_date += pd.Timedelta(days=1)
    
    # Create DataFrame for daily time points
    daily_points = []
    
    for time_point in daily_times:
        # Find the closest existing data point for interpolation
        if time_point in prices_df['date'].values:
            # Point already exists, find the exact match
            price_row = prices_df[prices_df['date'] == time_point].iloc[0]
            volume_row = volumes_df[volumes_df['date'] == time_point].iloc[0]
            
            daily_points.append({
                'timestamp': time_point,
                f'{coin_symbol}_price': price_row['price'],
                f'{coin_symbol}_volume': volume_row['volume']
            })
        else:
            # Find the closest point before and after for interpolation
            before_mask = prices_df['date'] < time_point
            after_mask = prices_df['date'] > time_point
            
            if before_mask.any() and after_mask.any():
                # Interpolate between two points
                before_idx = prices_df[before_mask]['date'].iloc[-1]
                after_idx = prices_df[after_mask]['date'].iloc[0]
                
                # Get the data for interpolation
                before_price = prices_df[prices_df['date'] == before_idx]['price'].iloc[0]
                after_price = prices_df[prices_df['date'] == after_idx]['price'].iloc[0]
                before_volume = volumes_df[volumes_df['date'] == before_idx]['volume'].iloc[0]
                after_volume = volumes_df[volumes_df['date'] == after_idx]['volume'].iloc[0]
                
                # Linear interpolation
                time_diff = (after_idx - before_idx).total_seconds()
                target_diff = (time_point - before_idx).total_seconds()
                ratio = target_diff / time_diff
                
                interpolated_price = before_price + (after_price - before_price) * ratio
                interpolated_volume = before_volume + (after_volume - before_volume) * ratio
                
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': interpolated_price,
                    f'{coin_symbol}_volume': interpolated_volume
                })
            elif before_mask.any():
                # Use the last available point
                last_idx = prices_df[before_mask]['date'].iloc[-1]
                last_price = prices_df[prices_df['date'] == last_idx]['price'].iloc[0]
                last_volume = volumes_df[volumes_df['date'] == last_idx]['volume'].iloc[0]
                
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': last_price,
                    f'{coin_symbol}_volume': last_volume
                })
            elif after_mask.any():
                # Use the first available point
                first_idx = prices_df[after_mask]['date'].iloc[0]
                first_price = prices_df[prices_df['date'] == first_idx]['price'].iloc[0]
                first_volume = volumes_df[volumes_df['date'] == first_idx]['volume'].iloc[0]
                
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': first_price,
                    f'{coin_symbol}_volume': first_volume
                })
    
    # Create DataFrame from daily points
    daily_df = pd.DataFrame(daily_points)
    if not daily_df.empty:
        daily_df = daily_df.set_index('timestamp')
        
        # Sort by timestamp
        daily_df = daily_df.sort_index()
        
        # Verify we have exactly 730 data points (2 per day for 365 days)
        expected_points = 365 * 2
        actual_points = len(daily_df)
        
        if actual_points == expected_points:
            logger.info(f"✅ Extracted daily time series with {actual_points} points (00:00 and 12:00) for {coin_symbol} - exactly 365 days")
            logger.info(f"   Date range: {daily_df.index.min()} to {daily_df.index.max()}")
        else:
            logger.warning(f"⚠️  Expected {expected_points} points but got {actual_points} for {coin_symbol}")
        
        return daily_df
    
    return pd.DataFrame()

def extract_daily_data_from_raw_simple(prices_df: pd.DataFrame, coin_symbol: str) -> pd.DataFrame:
    """
    Extract exactly 00:00 and 12:00 data points from raw API data for exactly 365 days (price only)
    
    Args:
        prices_df: DataFrame with timestamp and price data
        coin_symbol: Symbol of the coin
        
    Returns:
        DataFrame with exactly 00:00 and 12:00 points for each day (730 points total)
    """
    if prices_df.empty:
        return pd.DataFrame()
    
    # Get the date range from the data
    start_date = prices_df['date'].min().normalize()
    end_date = prices_df['date'].max().normalize()
    
    # Calculate how many days we have
    days_diff = (end_date - start_date).days
    
    # We need exactly 365 days, so we'll use the most recent 365 days
    if days_diff >= 364:  # We have at least 365 days
        # Use the last 365 days (most recent data)
        start_date = end_date - pd.Timedelta(days=364)
        logger.info(f"Using the most recent 365 days for {coin_symbol} (from {start_date} to {end_date})")
    else:
        # We don't have enough days, adjust to use what we have
        logger.warning(f"Only {days_diff + 1} days of data available for {coin_symbol}, need 365 days")
        # Use all available days and pad with the last available data
        end_date = start_date + pd.Timedelta(days=days_diff)
    
    # Create daily time points at 00:00 and 12:00 for exactly 365 days
    daily_times = []
    current_date = start_date
    
    for day in range(365):  # Exactly 365 days
        # Add 00:00 point
        daily_times.append(current_date)
        # Add 12:00 point
        daily_times.append(current_date + pd.Timedelta(hours=12))
        current_date += pd.Timedelta(days=1)
    
    # Create DataFrame for daily time points
    daily_points = []
    
    for time_point in daily_times:
        # Find the closest existing data point for interpolation
        if time_point in prices_df['date'].values:
            # Point already exists, find the exact match
            price_row = prices_df[prices_df['date'] == time_point].iloc[0]
            
            daily_points.append({
                'timestamp': time_point,
                coin_symbol: price_row['price']
            })
        else:
            # Find the closest point before and after for interpolation
            before_mask = prices_df['date'] < time_point
            after_mask = prices_df['date'] > time_point
            
            if before_mask.any() and after_mask.any():
                # Interpolate between two points
                before_idx = prices_df[before_mask]['date'].iloc[-1]
                after_idx = prices_df[after_mask]['date'].iloc[0]
                
                # Get the data for interpolation
                before_price = prices_df[prices_df['date'] == before_idx]['price'].iloc[0]
                after_price = prices_df[prices_df['date'] == after_idx]['price'].iloc[0]
                
                # Linear interpolation
                time_diff = (after_idx - before_idx).total_seconds()
                target_diff = (time_point - before_idx).total_seconds()
                ratio = target_diff / time_diff
                
                interpolated_price = before_price + (after_price - before_price) * ratio
                
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: interpolated_price
                })
            elif before_mask.any():
                # Use the last available point
                last_idx = prices_df[before_mask]['date'].iloc[-1]
                last_price = prices_df[prices_df['date'] == last_idx]['price'].iloc[0]
                
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: last_price
                })
            elif after_mask.any():
                # Use the first available point
                first_idx = prices_df[after_mask]['date'].iloc[0]
                first_price = prices_df[prices_df['date'] == first_idx]['price'].iloc[0]
                
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: first_price
                })
    
    # Create DataFrame from daily points
    daily_df = pd.DataFrame(daily_points)
    if not daily_df.empty:
        daily_df = daily_df.set_index('timestamp')
        
        # Sort by timestamp
        daily_df = daily_df.sort_index()
        
        # Verify we have exactly 730 data points (2 per day for 365 days)
        expected_points = 365 * 2
        actual_points = len(daily_df)
        
        if actual_points == expected_points:
            logger.info(f"✅ Extracted daily time series with {actual_points} points (00:00 and 12:00) for {coin_symbol} - exactly 365 days")
            logger.info(f"   Date range: {daily_df.index.min()} to {daily_df.index.max()}")
        else:
            logger.warning(f"⚠️  Expected {expected_points} points but got {actual_points} for {coin_symbol}")
        
        return daily_df
    
    return pd.DataFrame()

def create_daily_time_series(df: pd.DataFrame, coin_symbol: str) -> pd.DataFrame:
    """
    Create a DataFrame with exactly 00:00 and 12:00 points for each day for exactly 365 days
    
    Args:
        df: DataFrame with price and volume data
        coin_symbol: Symbol of the coin
        
    Returns:
        DataFrame with exactly 00:00 and 12:00 points for each day (730 points total)
    """
    if df.empty:
        return df
    
    # Get the date range from the data
    start_date = df.index.min().normalize()
    end_date = df.index.max().normalize()
    
    # Calculate how many days we have
    days_diff = (end_date - start_date).days
    
    # We need exactly 365 days, so we'll use the most recent 365 days
    if days_diff >= 364:  # We have at least 365 days
        # Use the last 365 days (most recent data)
        start_date = end_date - pd.Timedelta(days=364)
        logger.info(f"Using the most recent 365 days for {coin_symbol} (from {start_date} to {end_date})")
    else:
        # We don't have enough days, adjust to use what we have
        logger.warning(f"Only {days_diff + 1} days of data available for {coin_symbol}, need 365 days")
        # Use all available days and pad with the last available data
        end_date = start_date + pd.Timedelta(days=days_diff)
    
    # Create daily time points at 00:00 and 12:00 for exactly 365 days
    daily_times = []
    current_date = start_date
    
    for day in range(365):  # Exactly 365 days
        # Add 00:00 point
        daily_times.append(current_date)
        # Add 12:00 point
        daily_times.append(current_date + pd.Timedelta(hours=12))
        current_date += pd.Timedelta(days=1)
    
    # Create DataFrame for daily time points
    daily_points = []
    
    for time_point in daily_times:
        # Find the closest existing data point for interpolation
        if time_point in df.index:
            # Point already exists, use it
            daily_points.append({
                'timestamp': time_point,
                f'{coin_symbol}_price': df.loc[time_point, f'{coin_symbol}_price'],
                f'{coin_symbol}_volume': df.loc[time_point, f'{coin_symbol}_volume']
            })
        else:
            # Find the closest point before and after for interpolation
            before_mask = df.index < time_point
            after_mask = df.index > time_point
            
            if before_mask.any() and after_mask.any():
                # Interpolate between two points
                before_idx = df.index[before_mask][-1]
                after_idx = df.index[after_mask][0]
                
                # Linear interpolation
                time_diff = (after_idx - before_idx).total_seconds()
                target_diff = (time_point - before_idx).total_seconds()
                ratio = target_diff / time_diff
                
                price_before = df.loc[before_idx, f'{coin_symbol}_price']
                price_after = df.loc[after_idx, f'{coin_symbol}_price']
                volume_before = df.loc[before_idx, f'{coin_symbol}_volume']
                volume_after = df.loc[after_idx, f'{coin_symbol}_volume']
                
                interpolated_price = price_before + (price_after - price_before) * ratio
                interpolated_volume = volume_before + (volume_after - volume_before) * ratio
                
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': interpolated_price,
                    f'{coin_symbol}_volume': interpolated_volume
                })
            elif before_mask.any():
                # Use the last available point
                last_idx = df.index[before_mask][-1]
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': df.loc[last_idx, f'{coin_symbol}_price'],
                    f'{coin_symbol}_volume': df.loc[last_idx, f'{coin_symbol}_volume']
                })
            elif after_mask.any():
                # Use the first available point
                first_idx = df.index[after_mask][0]
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': df.loc[first_idx, f'{coin_symbol}_price'],
                    f'{coin_symbol}_volume': df.loc[first_idx, f'{coin_symbol}_volume']
                })
    
    # Create DataFrame from daily points
    daily_df = pd.DataFrame(daily_points)
    if not daily_df.empty:
        daily_df = daily_df.set_index('timestamp')
        
        # Sort by timestamp
        daily_df = daily_df.sort_index()
        
        # Verify we have exactly 730 data points (2 per day for 365 days)
        expected_points = 365 * 2
        actual_points = len(daily_df)
        
        if actual_points == expected_points:
            logger.info(f"✅ Created daily time series with {actual_points} points (00:00 and 12:00) for {coin_symbol} - exactly 365 days")
            logger.info(f"   Date range: {daily_df.index.min()} to {daily_df.index.max()}")
        else:
            logger.warning(f"⚠️  Expected {expected_points} points but got {actual_points} for {coin_symbol}")
        
        return daily_df
    
    return df

def create_daily_time_series_simple(df: pd.DataFrame, coin_symbol: str) -> pd.DataFrame:
    """
    Create a DataFrame with exactly 00:00 and 12:00 points for each day for exactly 365 days (price only)
    
    Args:
        df: DataFrame with price data only
        coin_symbol: Symbol of the coin
        
    Returns:
        DataFrame with exactly 00:00 and 12:00 points for each day (730 points total)
    """
    if df.empty:
        return df
    
    # Get the date range from the data
    start_date = df.index.min().normalize()
    end_date = df.index.max().normalize()
    
    # Calculate how many days we have
    days_diff = (end_date - start_date).days
    
    # We need exactly 365 days, so we'll use the most recent 365 days
    if days_diff >= 364:  # We have at least 365 days
        # Use the last 365 days (most recent data)
        start_date = end_date - pd.Timedelta(days=364)
        logger.info(f"Using the most recent 365 days for {coin_symbol} (from {start_date} to {end_date})")
    else:
        # We don't have enough days, adjust to use what we have
        logger.warning(f"Only {days_diff + 1} days of data available for {coin_symbol}, need 365 days")
        # Use all available days and pad with the last available data
        end_date = start_date + pd.Timedelta(days=days_diff)
    
    # Create daily time points at 00:00 and 12:00 for exactly 365 days
    daily_times = []
    current_date = start_date
    
    for day in range(365):  # Exactly 365 days
        # Add 00:00 point
        daily_times.append(current_date)
        # Add 12:00 point
        daily_times.append(current_date + pd.Timedelta(hours=12))
        current_date += pd.Timedelta(days=1)
    
    # Create DataFrame for daily time points
    daily_points = []
    
    for time_point in daily_times:
        # Find the closest existing data point for interpolation
        if time_point in df.index:
            # Point already exists, use it
            daily_points.append({
                'timestamp': time_point,
                coin_symbol: df.loc[time_point, coin_symbol]
            })
        else:
            # Find the closest point before and after for interpolation
            before_mask = df.index < time_point
            after_mask = df.index > time_point
            
            if before_mask.any() and after_mask.any():
                # Interpolate between two points
                before_idx = df.index[before_mask][-1]
                after_idx = df.index[after_mask][0]
                
                # Linear interpolation
                time_diff = (after_idx - before_idx).total_seconds()
                target_diff = (time_point - before_idx).total_seconds()
                ratio = target_diff / time_diff
                
                price_before = df.loc[before_idx, coin_symbol]
                price_after = df.loc[after_idx, coin_symbol]
                
                interpolated_price = price_before + (price_after - price_before) * ratio
                
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: interpolated_price
                })
            elif before_mask.any():
                # Use the last available point
                last_idx = df.index[before_mask][-1]
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: df.loc[last_idx, coin_symbol]
                })
            elif after_mask.any():
                # Use the first available point
                first_idx = df.index[after_mask][0]
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: df.loc[first_idx, coin_symbol]
                })
    
    # Create DataFrame from daily points
    daily_df = pd.DataFrame(daily_points)
    if not daily_df.empty:
        daily_df = daily_df.set_index('timestamp')
        
        # Sort by timestamp
        daily_df = daily_df.sort_index()
        
        # Verify we have exactly 730 data points (2 per day for 365 days)
        expected_points = 365 * 2
        actual_points = len(daily_df)
        
        if actual_points == expected_points:
            logger.info(f"✅ Created daily time series with {actual_points} points (00:00 and 12:00) for {coin_symbol} - exactly 365 days")
            logger.info(f"   Date range: {daily_df.index.min()} to {daily_df.index.max()}")
        else:
            logger.warning(f"⚠️  Expected {expected_points} points but got {actual_points} for {coin_symbol}")
        
        return daily_df
    
    return df

def add_daily_time_points_simple(df: pd.DataFrame, coin_symbol: str) -> pd.DataFrame:
    """
    Add daily points at 00:00 and 12:00 for each day for exactly 365 days in the data (price only)
    
    Args:
        df: DataFrame with price data only
        coin_symbol: Symbol of the coin
        
    Returns:
        DataFrame with additional daily time points
    """
    if df.empty:
        return df
    
    # Get the date range from the data
    start_date = df.index.min().normalize()
    end_date = df.index.max().normalize()
    
    # Calculate how many days we have
    days_diff = (end_date - start_date).days
    
    # We need exactly 365 days, so we'll use the most recent 365 days
    if days_diff >= 364:  # We have at least 365 days
        # Use the last 365 days (most recent data)
        start_date = end_date - pd.Timedelta(days=364)
        logger.info(f"Using the most recent 365 days for {coin_symbol} (from {start_date} to {end_date})")
    else:
        # We don't have enough days, adjust to use what we have
        logger.warning(f"Only {days_diff + 1} days of data available for {coin_symbol}, need 365 days")
        # Use all available days and pad with the last available data
        end_date = start_date + pd.Timedelta(days=days_diff)
    
    # Create daily time points at 00:00 and 12:00 for exactly 365 days
    daily_times = []
    current_date = start_date
    
    for day in range(365):  # Exactly 365 days
        # Add 00:00 point
        daily_times.append(current_date)
        # Add 12:00 point
        daily_times.append(current_date + pd.Timedelta(hours=12))
        current_date += pd.Timedelta(days=1)
    
    # Create DataFrame for daily time points
    daily_points = []
    
    for time_point in daily_times:
        # Find the closest existing data point for interpolation
        if time_point in df.index:
            # Point already exists, use it
            daily_points.append({
                'timestamp': time_point,
                coin_symbol: df.loc[time_point, coin_symbol]
            })
        else:
            # Find the closest point before and after for interpolation
            before_mask = df.index < time_point
            after_mask = df.index > time_point
            
            if before_mask.any() and after_mask.any():
                # Interpolate between two points
                before_idx = df.index[before_mask][-1]
                after_idx = df.index[after_mask][0]
                
                # Linear interpolation
                time_diff = (after_idx - before_idx).total_seconds()
                target_diff = (time_point - before_idx).total_seconds()
                ratio = target_diff / time_diff
                
                price_before = df.loc[before_idx, coin_symbol]
                price_after = df.loc[after_idx, coin_symbol]
                
                interpolated_price = price_before + (price_after - price_before) * ratio
                
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: interpolated_price
                })
            elif before_mask.any():
                # Use the last available point
                last_idx = df.index[before_mask][-1]
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: df.loc[last_idx, coin_symbol]
                })
            elif after_mask.any():
                # Use the first available point
                first_idx = df.index[after_mask][0]
                daily_points.append({
                    'timestamp': time_point,
                    coin_symbol: df.loc[first_idx, coin_symbol]
                })
    
    # Create DataFrame from daily points
    daily_df = pd.DataFrame(daily_points)
    if not daily_df.empty:
        daily_df = daily_df.set_index('timestamp')
        
        # Combine with original data and remove duplicates
        combined_df = pd.concat([df, daily_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()
        
        # Verify we have exactly 730 data points (2 per day for 365 days)
        expected_points = 365 * 2
        actual_points = len(combined_df)
        
        if actual_points == expected_points:
            logger.info(f"✅ Added daily time points with {actual_points} total points (00:00 and 12:00) for {coin_symbol} - exactly 365 days")
            logger.info(f"   Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        else:
            logger.warning(f"⚠️  Expected {expected_points} points but got {actual_points} for {coin_symbol}")
        
        return combined_df
    
    return df

def add_daily_time_points(df: pd.DataFrame, coin_symbol: str) -> pd.DataFrame:
    """
    Add daily points at 00:00 and 12:00 for each day for exactly 365 days in the data
    
    Args:
        df: DataFrame with price and volume data
        coin_symbol: Symbol of the coin
        
    Returns:
        DataFrame with additional daily time points
    """
    if df.empty:
        return df
    
    # Get the date range from the data
    start_date = df.index.min().normalize()
    end_date = df.index.max().normalize()
    
    # Calculate how many days we have
    days_diff = (end_date - start_date).days
    
    # We need exactly 365 days, so we'll use the most recent 365 days
    if days_diff >= 364:  # We have at least 365 days
        # Use the last 365 days (most recent data)
        start_date = end_date - pd.Timedelta(days=364)
        logger.info(f"Using the most recent 365 days for {coin_symbol} (from {start_date} to {end_date})")
    else:
        # We don't have enough days, adjust to use what we have
        logger.warning(f"Only {days_diff + 1} days of data available for {coin_symbol}, need 365 days")
        # Use all available days and pad with the last available data
        end_date = start_date + pd.Timedelta(days=days_diff)
    
    # Create daily time points at 00:00 and 12:00 for exactly 365 days
    daily_times = []
    current_date = start_date
    
    for day in range(365):  # Exactly 365 days
        # Add 00:00 point
        daily_times.append(current_date)
        # Add 12:00 point
        daily_times.append(current_date + pd.Timedelta(hours=12))
        current_date += pd.Timedelta(days=1)
    
    # Create DataFrame for daily time points
    daily_points = []
    
    for time_point in daily_times:
        # Find the closest existing data point for interpolation
        if time_point in df.index:
            # Point already exists, use it
            daily_points.append({
                'timestamp': time_point,
                f'{coin_symbol}_price': df.loc[time_point, f'{coin_symbol}_price'],
                f'{coin_symbol}_volume': df.loc[time_point, f'{coin_symbol}_volume']
            })
        else:
            # Find the closest point before and after for interpolation
            before_mask = df.index < time_point
            after_mask = df.index > time_point
            
            if before_mask.any() and after_mask.any():
                # Interpolate between two points
                before_idx = df.index[before_mask][-1]
                after_idx = df.index[after_mask][0]
                
                # Linear interpolation
                time_diff = (after_idx - before_idx).total_seconds()
                target_diff = (time_point - before_idx).total_seconds()
                ratio = target_diff / time_diff
                
                price_before = df.loc[before_idx, f'{coin_symbol}_price']
                price_after = df.loc[after_idx, f'{coin_symbol}_price']
                volume_before = df.loc[before_idx, f'{coin_symbol}_volume']
                volume_after = df.loc[after_idx, f'{coin_symbol}_volume']
                
                interpolated_price = price_before + (price_after - price_before) * ratio
                interpolated_volume = volume_before + (volume_after - volume_before) * ratio
                
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': interpolated_price,
                    f'{coin_symbol}_volume': interpolated_volume
                })
            elif before_mask.any():
                # Use the last available point
                last_idx = df.index[before_mask][-1]
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': df.loc[last_idx, f'{coin_symbol}_price'],
                    f'{coin_symbol}_volume': df.loc[last_idx, f'{coin_symbol}_volume']
                })
            elif after_mask.any():
                # Use the first available point
                first_idx = df.index[after_mask][0]
                daily_points.append({
                    'timestamp': time_point,
                    f'{coin_symbol}_price': df.loc[first_idx, f'{coin_symbol}_price'],
                    f'{coin_symbol}_volume': df.loc[first_idx, f'{coin_symbol}_volume']
                })
    
    # Create DataFrame from daily points
    daily_df = pd.DataFrame(daily_points)
    if not daily_df.empty:
        daily_df = daily_df.set_index('timestamp')
        
        # Combine with original data and remove duplicates
        combined_df = pd.concat([df, daily_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()
        
        # Verify we have exactly 730 data points (2 per day for 365 days)
        expected_points = 365 * 2
        actual_points = len(combined_df)
        
        if actual_points == expected_points:
            logger.info(f"✅ Added daily time points with {actual_points} total points (00:00 and 12:00) for {coin_symbol} - exactly 365 days")
            logger.info(f"   Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        else:
            logger.warning(f"⚠️  Expected {expected_points} points but got {actual_points} for {coin_symbol}")
        
        return combined_df
    
    return df

def fetch_crypto_data(limit: int = 200, days: int = 365, api_key: Optional[str] = None) -> Dict:
    """
    Main function to fetch crypto data for PCA analysis
    
    Args:
        limit: Number of top coins to fetch (default: 200)
        days: Number of days of historical data (default: 365)
        api_key: Optional CoinGecko API key for higher rate limits
        
    Returns:
        Dictionary containing all fetched crypto data
    """
    logger.info("Starting crypto data acquisition...")
    
    api = CoinGeckoAPI(api_key)
    
    # Step 1: Get top coins by volume
    top_coins = api.get_top_coins_by_volume(limit)
    
    if not top_coins:
        logger.error("No coins fetched from API")
        return {}
    
    logger.info(f"Found {len(top_coins)} coins. Fetching historical data...")
    
    # Step 2: Extract coin IDs for batch processing
    coin_ids = []
    coin_info_map = {}
    
    for coin in top_coins:
        coin_id = coin.get('id')
        coin_symbol = coin.get('symbol', '').upper()
        
        if coin_id and coin_symbol:
            coin_ids.append(coin_id)
            coin_info_map[coin_id] = {
                'symbol': coin_symbol,
                'info': coin
            }
    
    logger.info(f"Processing {len(coin_ids)} valid coins...")
    
    # Step 3: Get historical data in batches
    batch_data = api.get_coin_historical_data_batch(coin_ids, days)
    
    # Step 4: Process the data
    all_data = {}
    price_dataframes = []
    
    for coin_id, historical_data in batch_data.items():
        coin_symbol = coin_info_map[coin_id]['symbol']
        
        # Process the data
        df = process_historical_data(historical_data, coin_symbol)
        if not df.empty:
            price_dataframes.append(df)
            all_data[coin_id] = {
                'info': coin_info_map[coin_id]['info'],
                'historical': historical_data,
                'processed_data': df
            }
            logger.debug(f"Successfully processed {coin_symbol} ({coin_id})")
        else:
            logger.warning(f"Empty data for {coin_symbol} ({coin_id})")
    
    # Step 5: Combine all price data
    if price_dataframes:
        logger.info(f"Combining data from {len(price_dataframes)} coins...")
        combined_df = pd.concat(price_dataframes, axis=1)
        combined_df = combined_df.sort_index()  # Sort by date
        
        # Save to Excel
        excel_filename = f"data/crypto_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        os.makedirs('data', exist_ok=True)
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='Daily_Prices')
            
            # Create summary sheet
            summary_data = []
            for coin_id, data in all_data.items():
                coin_info = data['info']
                summary_data.append({
                    'Coin_ID': coin_id,
                    'Symbol': coin_info.get('symbol', '').upper(),
                    'Name': coin_info.get('name', ''),
                    'Current_Price': coin_info.get('current_price', 0),
                    'Market_Cap': coin_info.get('market_cap', 0),
                    'Volume_24h': coin_info.get('total_volume', 0),
                    'Price_Change_24h': coin_info.get('price_change_percentage_24h', 0),
                    'Data_Points': len(data['processed_data'])
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Data saved to Excel file: {excel_filename}")
        logger.info(f"Successfully processed {len(price_dataframes)} out of {len(top_coins)} coins")
        
        return {
            'top_coins': top_coins,
            'historical_data': all_data,
            'combined_prices': combined_df,
            'excel_file': excel_filename,
            'metadata': {
                'fetch_date': datetime.now().isoformat(),
                'coins_count': len(all_data),
                'total_coins_fetched': len(top_coins),
                'days_of_data': days,
                'api_key_used': api_key is not None
            }
        }
    else:
        logger.error("No valid historical data found")
        return {}

def fetch_specific_crypto_data(api_key: Optional[str] = None, db_path: str = "crypto_data.db") -> Dict:
    """
    Fetch daily price and volume data for specific crypto tokens and store in SQLite database
    
    Args:
        api_key: Optional CoinGecko API key for higher rate limits
        db_path: Path to SQLite database file
        
    Returns:
        Dictionary containing all fetched crypto data
    """
    logger.info("Starting specific crypto data acquisition for Bitcoin, Ethereum, NEAR, Solana, XRP, SUI, TRON, and Arbitrum...")
    
    api = CoinGeckoAPI(api_key)
    
    # Initialize database
    db = CryptoDatabase(db_path)
    
    # Log API key status
    if api_key:
        logger.info(f"✅ Using API key for higher rate limits (1000 calls/min)")
    else:
        logger.warning(f"⚠️  No API key provided - using free tier (20 calls/min)")
        logger.warning(f"   Consider adding an API key to environment variables for better reliability")
    
    # Define the specific tokens we want to fetch
    target_tokens = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH', 
        'near': 'NEAR',
        'solana': 'SOL',
        'ripple': 'XRP',
        'sui': 'SUI',
        'tron': 'TRX',
        'arbitrum': 'ARB',
        'aave': 'AAVE',
        'cardano': 'ADA',
        'ethena': 'ENA',
        'dogecoin': 'DOGE'
    }
    
    # Initialize data storage
    all_data = {}
    price_volume_dataframes = []
    
    # Check which tokens need updating
    tokens_to_update = []
    for coin_id, symbol in target_tokens.items():
        if db.should_update_token(symbol):
            tokens_to_update.append((coin_id, symbol))
            logger.info(f"🔄 {symbol} needs update (last update: {db.get_last_update_time(symbol)})")
        else:
            logger.info(f"✅ {symbol} is up to date (last update: {db.get_last_update_time(symbol)})")
    
    if not tokens_to_update:
        logger.info("🎉 All tokens are up to date! No new data needed.")
        # Get existing data from database
        existing_symbols = [symbol for _, symbol in target_tokens.items()]
        combined_df = db.get_latest_data(existing_symbols, days=365)
        
        if not combined_df.empty:
            return {
                'target_tokens': target_tokens,
                'historical_data': {},  # Not needed when using database
                'combined_data': combined_df,
                'database_path': db_path,
                'metadata': {
                    'fetch_date': datetime.now().isoformat(),
                    'tokens_count': len(existing_symbols),
                    'total_tokens_requested': len(target_tokens),
                    'days_of_data': 365,
                    'api_key_used': api_key is not None,
                    'tokens_fetched': existing_symbols,
                    'data_source': 'database',
                    'update_type': 'no_update_needed'
                }
            }
        else:
            logger.error("No data found in database")
            return {}
    
    # Fetch data for tokens that need updating
    for coin_id, symbol in tokens_to_update:
        logger.info(f"Fetching historical data for {symbol} ({coin_id})...")
        
        # Add longer delay between tokens to avoid rate limiting
        if (coin_id, symbol) != tokens_to_update[0]:  # Skip delay for first token
            delay = 5  # Increased from 3 to 5 seconds
            logger.info(f"Waiting {delay} seconds before processing {symbol}...")
            time.sleep(delay)
        
        # Get historical data (365 days)
        historical_data = api.get_coin_historical_data(coin_id, days=365)
        
        if historical_data:
            all_data[coin_id] = {
                'symbol': symbol,
                'historical': historical_data,
                'processed_data': None  # Will be processed later
            }
            logger.info(f"Successfully fetched historical data for {symbol} ({coin_id})")
        else:
            logger.error(f"❌ FAILED: No historical data found for {symbol} ({coin_id})")
            logger.error(f"   This token will be skipped in the analysis")
    
    # Step 2: Process all data using the last historical data point as "current hour price"
    logger.info("Processing all data with last historical point as current price...")
    for coin_id, symbol in target_tokens.items():
        if coin_id in all_data:
            historical_data = all_data[coin_id]['historical']
            
            # Extract the last price from historical data as the "current hour price"
            if historical_data and 'prices' in historical_data and len(historical_data['prices']) > 0:
                current_hour_price = historical_data['prices'][-1][1]  # Last price from historical data
                logger.info(f"Using last historical price for {symbol}: ${current_hour_price}")
            else:
                current_hour_price = None
                logger.warning(f"No historical price data for {symbol}")
            
            # Process the data with both price and volume, including current hour's price
            df = process_historical_data_with_volume(historical_data, symbol, current_hour_price)
            if not df.empty:
                # Store data in database
                db.update_crypto_data(symbol, df)
                
                price_volume_dataframes.append(df)
                all_data[coin_id]['processed_data'] = df
                all_data[coin_id]['current_hour_price'] = current_hour_price
                logger.info(f"Successfully processed and stored {symbol} ({coin_id}) - {len(df)} data points")
                if current_hour_price:
                    logger.info(f"Current price for {symbol}: ${current_hour_price}")
            else:
                logger.error(f"❌ FAILED: Empty processed data for {symbol} ({coin_id})")
                logger.error(f"   This token will be skipped in the analysis")
    
    
    # Step 2: Combine all price and volume data
    if price_volume_dataframes:
        logger.info(f"Combining data from {len(price_volume_dataframes)} tokens...")
        combined_df = pd.concat(price_volume_dataframes, axis=1)
        combined_df = combined_df.sort_index()  # Sort by date
    
    # Log which tokens were successfully processed
    successful_tokens = [data['symbol'] for data in all_data.values() if data.get('processed_data') is not None]
    failed_tokens = [symbol for coin_id, symbol in target_tokens.items() 
                    if coin_id not in all_data or all_data[coin_id].get('processed_data') is None]
    
    logger.info(f"✅ SUCCESSFULLY PROCESSED ({len(successful_tokens)} tokens): {', '.join(successful_tokens)}")
    
    if failed_tokens:
        logger.error(f"❌ FAILED TO PROCESS ({len(failed_tokens)} tokens): {', '.join(failed_tokens)}")
        logger.error("This may be due to rate limiting or API issues. Consider running again later.")
        
        # Retry failed tokens with longer delays
        logger.info("🔄 Retrying failed tokens with longer delays...")
        time.sleep(30)  # Wait 30 seconds before retrying
        
        for failed_token in failed_tokens[:]:  # Copy list to avoid modification during iteration
            coin_id = None
            for cid, symbol in target_tokens.items():
                if symbol == failed_token:
                    coin_id = cid
                    break
            
            if coin_id:
                logger.info(f"🔄 Retrying {failed_token} ({coin_id})...")
                time.sleep(10)  # Wait 10 seconds before each retry
                
                historical_data = api.get_coin_historical_data(coin_id, days=365)
                if historical_data:
                    logger.info(f"✅ SUCCESS: Retry worked for {failed_token}")
                    
                    # Process the retried data
                    if 'prices' in historical_data and len(historical_data['prices']) > 0:
                        current_hour_price = historical_data['prices'][-1][1]
                        df = process_historical_data_with_volume(historical_data, failed_token, current_hour_price)
                        if not df.empty:
                            price_volume_dataframes.append(df)
                            all_data[coin_id] = {
                                'symbol': failed_token,
                                'historical': historical_data,
                                'processed_data': df,
                                'current_hour_price': current_hour_price
                            }
                            failed_tokens.remove(failed_token)
                            logger.info(f"✅ Successfully processed retry for {failed_token}")
                else:
                    logger.error(f"❌ Retry failed for {failed_token}")
    
    # Recreate combined dataframe with any retried tokens
    if price_volume_dataframes:
        logger.info(f"Creating final combined dataframe with {len(price_volume_dataframes)} tokens...")
        combined_df = pd.concat(price_volume_dataframes, axis=1)
        combined_df = combined_df.sort_index()  # Sort by date
    
    # Get all data from database (including existing data for tokens that didn't need updating)
    all_symbols = [symbol for _, symbol in target_tokens.items()]
    combined_df = db.get_latest_data(all_symbols, days=365)
    
    if combined_df.empty:
        logger.error("No data available in database")
        db.close_connection()
        return {}
    
    logger.info(f"📊 Retrieved {len(combined_df)} data points for {len(all_symbols)} tokens from database")
    
    # Clean up old data (keep last 400 days)
    db.cleanup_old_data(days_to_keep=400)
    
    # Close database connection
    db.close_connection()
    
    # Final summary
    final_successful_tokens = [data['symbol'] for data in all_data.values() if data.get('processed_data') is not None]
    final_failed_tokens = [symbol for coin_id, symbol in target_tokens.items() 
                          if coin_id not in all_data or all_data[coin_id].get('processed_data') is None]
    
    logger.info(f"🎯 FINAL RESULT: {len(final_successful_tokens)} out of {len(target_tokens)} tokens processed successfully")
    logger.info(f"✅ SUCCESSFUL: {', '.join(final_successful_tokens)}")
    if final_failed_tokens:
        logger.error(f"❌ FAILED: {', '.join(final_failed_tokens)}")
    
    return {
        'target_tokens': target_tokens,
        'historical_data': all_data,
        'combined_data': combined_df,
        'database_path': db_path,
        'metadata': {
            'fetch_date': datetime.now().isoformat(),
            'tokens_count': len(final_successful_tokens),
            'total_tokens_requested': len(target_tokens),
            'days_of_data': 365,
            'api_key_used': api_key is not None,
            'tokens_fetched': final_successful_tokens,
            'data_source': 'database',
            'update_type': 'incremental_update'
        }
    }
    
    if not all_data:
        logger.error("No valid historical data found")
        return {}

def update_current_hour_data(api_key: Optional[str] = None, db_path: str = "crypto_data.db") -> Dict:
    """
    Update only the current hour data for all tokens without re-fetching historical data
    
    Args:
        api_key: Optional CoinGecko API key for higher rate limits
        db_path: Path to SQLite database file
        
    Returns:
        Dictionary containing update results
    """
    logger.info("Starting incremental update of current hour data...")
    
    api = CoinGeckoAPI(api_key)
    db = CryptoDatabase(db_path)
    
    # Define the specific tokens we want to update
    target_tokens = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH', 
        'near': 'NEAR',
        'solana': 'SOL',
        'ripple': 'XRP',
        'sui': 'SUI',
        'tron': 'TRX',
        'arbitrum': 'ARB',
        'aave': 'AAVE',
        'cardano': 'ADA',
        'ethena': 'ENA',
        'dogecoin': 'DOGE'
    }
    
    updated_tokens = []
    failed_tokens = []
    
    for coin_id, symbol in target_tokens.items():
        if db.needs_current_hour_update(symbol):
            logger.info(f"🔄 Updating current hour data for {symbol}...")
            
            # Fetch current hour price
            current_price = api.get_current_hour_price(coin_id)
            
            if current_price is not None:
                # Get current hour timestamp
                current_time = datetime.now()
                current_hour_time = current_time.replace(minute=0, second=0, microsecond=0)
                
                # Get the last volume from existing data
                last_volume = 0
                existing_data = db.get_latest_data([symbol], days=1)
                if not existing_data.empty:
                    volume_col = f'{symbol}_volume'
                    if volume_col in existing_data.columns:
                        last_volume = existing_data[volume_col].iloc[-1]
                
                # Create current hour data point
                current_hour_data = pd.DataFrame({
                    f'{symbol}_price': [current_price],
                    f'{symbol}_volume': [last_volume]
                }, index=[current_hour_time])
                
                # Update database with current hour data
                db.update_crypto_data(symbol, current_hour_data)
                updated_tokens.append(symbol)
                logger.info(f"✅ Updated current hour data for {symbol}: ${current_price}")
            else:
                failed_tokens.append(symbol)
                logger.error(f"❌ Failed to update current hour data for {symbol}")
        else:
            logger.info(f"✅ {symbol} current hour data is up to date")
    
    # Close database connection
    db.close_connection()
    
    logger.info(f"🎯 Incremental update completed: {len(updated_tokens)} tokens updated, {len(failed_tokens)} failed")
    
    return {
        'updated_tokens': updated_tokens,
        'failed_tokens': failed_tokens,
        'update_type': 'current_hour_only',
        'timestamp': datetime.now().isoformat()
    }

def get_api_credentials() -> Optional[str]:
    """
    Get API credentials from user input or environment variable
    
    Returns:
        API key if provided, None otherwise
    """
    # Check environment variable first
    api_key = os.getenv('COINGECKO_API_KEY')
    
    if api_key:
        logger.info("Using API key from environment variable")
        return api_key
    
    # Check .env file (already loaded by dotenv)
    api_key = os.getenv('COINGECKO_API_KEY')
    if api_key:
        logger.info("Using API key from .env file")
        return api_key
    
    # Check simple config file
    try:
        from config import COINGECKO_API_KEY as config_api_key
        if config_api_key and config_api_key != "your_api_key_here":
            logger.info("Using API key from environment variables")
            return config_api_key
    except ImportError:
        pass
    
    # Ask user for API key
    print("\nCoinGecko API Setup:")
    print("Note: API key is optional but provides higher rate limits")
    print("You can get a free API key from: https://www.coingecko.com/en/api")
    print("You can also set COINGECKO_API_KEY environment variable to set your API key permanently")
    
    use_api = input("Do you have a CoinGecko API key? (y/n): ").lower().strip()
    
    if use_api == 'y':
        api_key = input("Enter your CoinGecko API key: ").strip()
        if api_key:
            logger.info("API key provided")
            return api_key
    
    logger.info("No API key provided - using free tier (50 calls/minute)")
    return None

if __name__ == "__main__":
    # Test the data acquisition
    logging.basicConfig(level=logging.INFO)
    
    # Get API credentials
    api_key = get_api_credentials()
    
    print("Choose data acquisition mode:")
    print("1. Fetch specific tokens (Bitcoin, Ethereum, NEAR, Solana)")
    print("2. Fetch top 200 coins by volume")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Fetch specific crypto data
        data = fetch_specific_crypto_data(api_key=api_key)
        
        if data:
            print(f"\n✅ Specific crypto data acquisition completed successfully!")
            print(f"📊 Fetched data for {data['metadata']['tokens_count']} tokens")
            print(f"🪙 Tokens: {', '.join(data['metadata']['tokens_fetched'])}")
            print(f"📅 {data['metadata']['days_of_data']} days of historical data")
            print(f"📁 Excel file saved: {data['excel_file']}")
        else:
            print("\n❌ Specific crypto data acquisition failed")
    else:
        # Fetch general crypto data
        data = fetch_crypto_data(limit=200, days=365, api_key=api_key)
        
        if data:
            print(f"\n✅ Data acquisition completed successfully!")
            print(f"📊 Fetched data for {data['metadata']['coins_count']} coins")
            print(f"📅 {data['metadata']['days_of_data']} days of historical data")
            print(f"📁 Excel file saved: {data['excel_file']}")
        else:
            print("\n❌ Data acquisition failed") 
