#!/usr/bin/env python3
"""
Database Manager for Crypto EMA Analysis
Handles SQLite database operations for storing and updating crypto data
"""

import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class CryptoDatabase:
    """SQLite database manager for crypto data"""
    
    def __init__(self, db_path: str = "crypto_data.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.setup_database()
    
    def setup_database(self):
        """Create database and tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create crypto_prices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crypto_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(token_symbol, timestamp)
                )
            ''')
            
            # Create ema_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ema_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    ema_short REAL NOT NULL,
                    ema_long REAL NOT NULL,
                    ema_percentage REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(token_symbol, timestamp)
                )
            ''')
            
            # Create analysis_summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date DATE NOT NULL,
                    token_symbol TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    current_ema_short REAL NOT NULL,
                    current_ema_long REAL NOT NULL,
                    current_ema_percentage REAL NOT NULL,
                    bullish_crossovers INTEGER DEFAULT 0,
                    bearish_crossovers INTEGER DEFAULT 0,
                    data_points INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(analysis_date, token_symbol)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_crypto_prices_token_timestamp ON crypto_prices(token_symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ema_data_token_timestamp ON ema_data(token_symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_summary_date_token ON analysis_summary(analysis_date, token_symbol)')
            
            self.conn.commit()
            logger.info(f"Database setup completed: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def get_last_update_time(self, token_symbol: str) -> Optional[datetime]:
        """
        Get the last update time for a specific token
        
        Args:
            token_symbol: Token symbol (e.g., 'BTC')
            
        Returns:
            Last update timestamp or None if no data exists
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT MAX(timestamp) FROM crypto_prices 
                WHERE token_symbol = ?
            ''', (token_symbol,))
            
            result = cursor.fetchone()
            if result and result[0]:
                return datetime.fromisoformat(result[0])
            return None
            
        except Exception as e:
            logger.error(f"Error getting last update time for {token_symbol}: {str(e)}")
            return None
    
    def should_update_token(self, token_symbol: str) -> bool:
        """
        Check if a token should be updated based on last update time and current time
        
        Args:
            token_symbol: Token symbol
            
        Returns:
            True if token should be updated, False otherwise
        """
        last_update = self.get_last_update_time(token_symbol)
        if not last_update:
            return True  # No data exists, should update
        
        current_time = datetime.now()
        time_diff = current_time - last_update
        
        # Check if it's been more than 12 hours since last update
        # This ensures we get both 00:00 and 12:00 points
        if time_diff.total_seconds() > 43200:  # 12 hours = 43200 seconds
            return True
        
        # Check if we need to update the current hour data point
        if self.needs_current_hour_update(token_symbol):
            return True
        
        # Check if we're at 00:00 or 12:00 and haven't updated recently
        current_hour = current_time.hour
        if current_hour in [0, 12]:
            # Check if we've updated in the last hour for these key times
            if time_diff.total_seconds() > 3600:  # 1 hour
                return True
        
        return False
    
    def get_last_required_update_time(self, token_symbol: str) -> Optional[datetime]:
        """
        Get the last time we need data from to ensure we have complete coverage
        
        Args:
            token_symbol: Token symbol
            
        Returns:
            Last required update time or None if no data exists
        """
        last_update = self.get_last_update_time(token_symbol)
        if not last_update:
            return None
        
        # We want to ensure we have data up to the last 00:00 or 12:00 point
        # Find the last 00:00 or 12:00 point before the last update
        last_update_hour = last_update.hour
        
        if last_update_hour == 0:  # Last update was at 00:00
            # We need data from the previous 12:00
            required_time = last_update - timedelta(hours=12)
        elif last_update_hour == 12:  # Last update was at 12:00
            # We need data from the previous 00:00
            required_time = last_update - timedelta(hours=12)
        else:  # Last update was at current hour
            # We need data from the last 00:00 or 12:00
            if last_update_hour < 12:
                # We need data from 00:00 of the same day
                required_time = last_update.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # We need data from 12:00 of the same day
                required_time = last_update.replace(hour=12, minute=0, second=0, microsecond=0)
        
        return required_time
    
    def update_crypto_data(self, token_symbol: str, data_df: pd.DataFrame):
        """
        Update crypto data for a token
        
        Args:
            token_symbol: Token symbol
            data_df: DataFrame with timestamp, price, and volume data
        """
        try:
            # Prepare data for insertion
            price_col = f'{token_symbol}_price'
            volume_col = f'{token_symbol}_volume'
            
            if price_col not in data_df.columns or volume_col not in data_df.columns:
                logger.warning(f"Missing price or volume columns for {token_symbol}")
                return
            
            # IMPORTANT: We need to filter ALL data to only include 00:00, 12:00, and the last available data point
            # This ensures the database only contains the data points we want, regardless of what was there before
            
            # First, remove all existing data for this token to ensure clean data
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM crypto_prices WHERE token_symbol = ?', (token_symbol,))
            logger.info(f"Cleared existing data for {token_symbol} to ensure clean formatting")
            
            # Now filter the incoming data to only include 00:00, 12:00, and the last available data point
            filtered_data = []
            
            # First, add all data points that are at 00:00 or 12:00
            for timestamp, row in data_df.iterrows():
                if timestamp.hour in [0, 12]:
                    filtered_data.append((timestamp, row))
            
            # Then, add the last available data point if it's not already included
            if not data_df.empty:
                last_timestamp = data_df.index[-1]
                last_row = data_df.iloc[-1]
                
                # Only add if it's not already in filtered_data (i.e., not at 00:00 or 12:00)
                if last_timestamp.hour not in [0, 12]:
                    filtered_data.append((last_timestamp, last_row))
                    logger.info(f"Adding last available data point for {token_symbol} at {last_timestamp}")
            
            if not filtered_data:
                logger.info(f"No data points at 00:00, 12:00, or last available time found for {token_symbol}")
                return
            
            # Sort filtered data by timestamp to ensure proper order
            filtered_data.sort(key=lambda x: x[0])
            
            # Prepare data for database insertion
            data_to_insert = []
            for timestamp, row in filtered_data:
                price = row[price_col]
                volume = row[volume_col]
                
                # Convert timestamp to string format
                if isinstance(timestamp, pd.Timestamp):
                    timestamp_str = timestamp.isoformat()
                else:
                    timestamp_str = str(timestamp)
                
                data_to_insert.append((token_symbol, timestamp_str, price, volume))
            
            # Insert the filtered data
            cursor.executemany('''
                INSERT INTO crypto_prices 
                (token_symbol, timestamp, price, volume) 
                VALUES (?, ?, ?, ?)
            ''', data_to_insert)
            
            self.conn.commit()
            logger.info(f"Updated {len(data_to_insert)} data points for {token_symbol} (filtered to 00:00, 12:00, and last available data point)")
            
        except Exception as e:
            logger.error(f"Error updating crypto data for {token_symbol}: {str(e)}")
            self.conn.rollback()
    
    def update_ema_data(self, token_symbol: str, ema_df: pd.DataFrame):
        """
        Update EMA data for a token
        
        Args:
            token_symbol: Token symbol
            ema_df: DataFrame with EMA calculations
        """
        try:
            ema_short_col = f'{token_symbol}_ema_5d'
            ema_long_col = f'{token_symbol}_ema_15d'
            ema_percentage_col = f'{token_symbol}_ema_percentage'
            
            if not all(col in ema_df.columns for col in [ema_short_col, ema_long_col, ema_percentage_col]):
                logger.warning(f"Missing EMA columns for {token_symbol}")
                return
            
            # Prepare data for insertion
            data_to_insert = []
            for timestamp, row in ema_df.iterrows():
                ema_short = row[ema_short_col]
                ema_long = row[ema_long_col]
                ema_percentage = row[ema_percentage_col]
                
                # Convert timestamp to string format
                if isinstance(timestamp, pd.Timestamp):
                    timestamp_str = timestamp.isoformat()
                else:
                    timestamp_str = str(timestamp)
                
                data_to_insert.append((token_symbol, timestamp_str, ema_short, ema_long, ema_percentage))
            
            # Insert or replace data
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO ema_data 
                (token_symbol, timestamp, ema_short, ema_long, ema_percentage) 
                VALUES (?, ?, ?, ?, ?)
            ''', data_to_insert)
            
            self.conn.commit()
            logger.info(f"Updated {len(data_to_insert)} EMA data points for {token_symbol}")
            
        except Exception as e:
            logger.error(f"Error updating EMA data for {token_symbol}: {str(e)}")
            self.conn.rollback()
    
    def update_analysis_summary(self, token_symbol: str, summary_data: Dict):
        """
        Update analysis summary for a token
        
        Args:
            token_symbol: Token symbol
            summary_data: Dictionary with summary statistics
        """
        try:
            analysis_date = datetime.now().date().isoformat()
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_summary 
                (analysis_date, token_symbol, current_price, current_ema_short, 
                 current_ema_long, current_ema_percentage, bullish_crossovers, 
                 bearish_crossovers, data_points) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_date,
                token_symbol,
                summary_data.get('current_price', 0),
                summary_data.get('current_ema_short', 0),
                summary_data.get('current_ema_long', 0),
                summary_data.get('current_ema_percentage', 0),
                summary_data.get('bullish_crossovers', 0),
                summary_data.get('bearish_crossovers', 0),
                summary_data.get('data_points', 0)
            ))
            
            self.conn.commit()
            logger.info(f"Updated analysis summary for {token_symbol}")
            
        except Exception as e:
            logger.error(f"Error updating analysis summary for {token_symbol}: {str(e)}")
            self.conn.rollback()
    
    def get_latest_data(self, token_symbols: List[str], days: int = 365) -> pd.DataFrame:
        """
        Get the latest data for specified tokens with 12-hour pattern filtering
        
        Args:
            token_symbols: List of token symbols
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame with price and volume data (only 00:00, 12:00, and current hour)
        """
        try:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            start_date_str = start_date.isoformat()
            
            # Build query for all tokens
            placeholders = ','.join(['?' for _ in token_symbols])
            query = f'''
                SELECT token_symbol, timestamp, price, volume 
                FROM crypto_prices 
                WHERE token_symbol IN ({placeholders}) 
                AND timestamp >= ?
                ORDER BY token_symbol, timestamp
            '''
            
            cursor = self.conn.cursor()
            cursor.execute(query, token_symbols + [start_date_str])
            
            # Convert to DataFrame
            data = cursor.fetchall()
            if not data:
                logger.warning("No data found in database")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['token_symbol', 'timestamp', 'price', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Pivot to wide format
            price_df = df.pivot(index='timestamp', columns='token_symbol', values='price')
            volume_df = df.pivot(index='timestamp', columns='token_symbol', values='volume')
            
            # Rename columns
            price_df.columns = [f'{col}_price' for col in price_df.columns]
            volume_df.columns = [f'{col}_volume' for col in volume_df.columns]
            
            # Combine price and volume data
            combined_df = pd.concat([price_df, volume_df], axis=1)
            combined_df = combined_df.sort_index()
            
            # Apply 12-hour pattern filtering
            combined_df = self._filter_to_12_hour_pattern(combined_df)
            
            logger.info(f"Retrieved {len(combined_df)} data points for {len(token_symbols)} tokens")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error retrieving data from database: {str(e)}")
            return pd.DataFrame()
    
    def _filter_to_12_hour_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to only include 00:00, 12:00, and current hour data points
        
        Args:
            df: DataFrame with timestamp index
            
        Returns:
            Filtered DataFrame with only valid time points
        """
        if df.empty:
            return df
        
        # Get current hour
        current_time = datetime.now()
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        
        # Filter to keep only:
        # 1. Points at 00:00 (midnight)
        # 2. Points at 12:00 (noon) 
        # 3. Current hour point (must be the last)
        valid_timestamps = []
        
        for timestamp in df.index:
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Keep if it's exactly 00:00 or 12:00 AND not after current hour
            if ((hour == 0 and minute == 0) or (hour == 12 and minute == 0)) and timestamp <= current_hour:
                valid_timestamps.append(timestamp)
            # Keep if it's the current hour
            elif timestamp == current_hour:
                valid_timestamps.append(timestamp)
        
        # Return only valid timestamps
        filtered_df = df.loc[valid_timestamps].copy()
        
        logger.info(f"Filtered data: kept {len(filtered_df)} out of {len(df)} data points")
        
        return filtered_df
    
    def get_latest_ema_data(self, token_symbols: List[str], days: int = 365) -> pd.DataFrame:
        """
        Get the latest EMA data for specified tokens
        
        Args:
            token_symbols: List of token symbols
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame with EMA data
        """
        try:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            start_date_str = start_date.isoformat()
            
            # Build query for all tokens
            placeholders = ','.join(['?' for _ in token_symbols])
            query = f'''
                SELECT token_symbol, timestamp, ema_short, ema_long, ema_percentage 
                FROM ema_data 
                WHERE token_symbol IN ({placeholders}) 
                AND timestamp >= ?
                ORDER BY token_symbol, timestamp
            '''
            
            cursor = self.conn.cursor()
            cursor.execute(query, token_symbols + [start_date_str])
            
            # Convert to DataFrame
            data = cursor.fetchall()
            if not data:
                logger.warning("No EMA data found in database")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['token_symbol', 'timestamp', 'ema_short', 'ema_long', 'ema_percentage'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Pivot to wide format
            ema_short_df = df.pivot(index='timestamp', columns='token_symbol', values='ema_short')
            ema_long_df = df.pivot(index='timestamp', columns='token_symbol', values='ema_long')
            ema_percentage_df = df.pivot(index='timestamp', columns='token_symbol', values='ema_percentage')
            
            # Rename columns
            ema_short_df.columns = [f'{col}_ema_5d' for col in ema_short_df.columns]
            ema_long_df.columns = [f'{col}_ema_15d' for col in ema_long_df.columns]
            ema_percentage_df.columns = [f'{col}_ema_percentage' for col in ema_percentage_df.columns]
            
            # Combine EMA data
            combined_df = pd.concat([ema_short_df, ema_long_df, ema_percentage_df], axis=1)
            combined_df = combined_df.sort_index()
            
            logger.info(f"Retrieved {len(combined_df)} EMA data points for {len(token_symbols)} tokens")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error retrieving EMA data from database: {str(e)}")
            return pd.DataFrame()
    
    def get_analysis_summary(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get analysis summary for a specific date or latest date
        
        Args:
            date: Date string (YYYY-MM-DD) or None for latest date
            
        Returns:
            DataFrame with analysis summary
        """
        try:
            cursor = self.conn.cursor()
            
            if date:
                cursor.execute('''
                    SELECT * FROM analysis_summary 
                    WHERE analysis_date = ?
                    ORDER BY token_symbol
                ''', (date,))
            else:
                cursor.execute('''
                    SELECT * FROM analysis_summary 
                    WHERE analysis_date = (SELECT MAX(analysis_date) FROM analysis_summary)
                    ORDER BY token_symbol
                ''')
            
            data = cursor.fetchall()
            if not data:
                logger.warning("No analysis summary found")
                return pd.DataFrame()
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            df = pd.DataFrame(data, columns=columns)
            
            logger.info(f"Retrieved analysis summary for {len(df)} tokens")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving analysis summary: {str(e)}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 400):
        """
        Clean up old data to keep database size manageable
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_date_str = cutoff_date.isoformat()
            
            cursor = self.conn.cursor()
            
            # Delete old price data
            cursor.execute('''
                DELETE FROM crypto_prices 
                WHERE timestamp < ?
            ''', (cutoff_date_str,))
            
            # Delete old EMA data
            cursor.execute('''
                DELETE FROM ema_data 
                WHERE timestamp < ?
            ''', (cutoff_date_str,))
            
            # Delete old summary data
            cursor.execute('''
                DELETE FROM analysis_summary 
                WHERE analysis_date < ?
            ''', (cutoff_date.date().isoformat(),))
            
            self.conn.commit()
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            self.conn.rollback() 

    def get_current_hour_data(self, token_symbol: str) -> Optional[Tuple[float, float]]:
        """
        Get the current hour data for a token
        
        Args:
            token_symbol: Token symbol
            
        Returns:
            Tuple of (price, volume) for current hour, or None if not found
        """
        try:
            current_time = datetime.now()
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            current_hour_str = current_hour.isoformat()
            
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT price, volume FROM crypto_prices 
                WHERE token_symbol = ? AND timestamp = ?
            ''', (token_symbol, current_hour_str))
            
            result = cursor.fetchone()
            if result:
                return (result[0], result[1])
            return None
            
        except Exception as e:
            logger.error(f"Error getting current hour data for {token_symbol}: {str(e)}")
            return None
    
    def needs_current_hour_update(self, token_symbol: str) -> bool:
        """
        Check if the current hour data needs to be updated
        
        Args:
            token_symbol: Token symbol
            
        Returns:
            True if current hour data needs update, False otherwise
        """
        current_hour_data = self.get_current_hour_data(token_symbol)
        if not current_hour_data:
            return True  # No current hour data exists
        
        # Check if the current hour data is from the current hour
        current_time = datetime.now()
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        
        # If we're in a new hour, we need to update
        if current_time.hour != current_hour.hour:
            return True
        
        return False 

    def cleanup_existing_data_format(self):
        """
        Clean up existing database data to ensure only 00:00, 12:00, and last available data points are kept
        This removes all the random hour data points that were incorrectly stored
        """
        try:
            cursor = self.conn.cursor()
            
            # Get all unique tokens
            cursor.execute('SELECT DISTINCT token_symbol FROM crypto_prices')
            tokens = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Cleaning up data format for {len(tokens)} tokens...")
            
            for token_symbol in tokens:
                logger.info(f"Cleaning up {token_symbol}...")
                
                # Get all data for this token
                cursor.execute('''
                    SELECT timestamp, price, volume 
                    FROM crypto_prices 
                    WHERE token_symbol = ? 
                    ORDER BY timestamp
                ''', (token_symbol,))
                
                all_data = cursor.fetchall()
                
                if not all_data:
                    continue
                
                # Filter to only keep 00:00, 12:00, and the last data point
                filtered_data = []
                
                for timestamp_str, price, volume in all_data:
                    # Parse timestamp
                    try:
                        timestamp = pd.to_datetime(timestamp_str)
                    except:
                        logger.warning(f"Invalid timestamp format for {token_symbol}: {timestamp_str}")
                        continue
                    
                    # Keep 00:00 and 12:00 points
                    if timestamp.hour in [0, 12]:
                        filtered_data.append((timestamp_str, price, volume))
                
                # Add the last available data point if it's not already included
                if all_data:
                    last_timestamp_str, last_price, last_volume = all_data[-1]
                    last_timestamp = pd.to_datetime(last_timestamp_str)
                    
                    # Only add if it's not already in filtered_data (i.e., not at 00:00 or 12:00)
                    if last_timestamp.hour not in [0, 12]:
                        # IMPORTANT: Round the last timestamp to the beginning of the current hour
                        # This ensures we have exactly one point at the current hour, not multiple random minutes
                        current_hour_beginning = last_timestamp.replace(minute=0, second=0, microsecond=0)
                        
                        # CRITICAL FIX: Check if we already have ANY data point at this hour
                        # This prevents adding multiple points when crossing day boundaries
                        existing_hour_points = [ts for ts, _, _ in filtered_data if pd.to_datetime(ts).hour == current_hour_beginning.hour]
                        
                        if not existing_hour_points:
                            # Convert back to string format for database insertion
                            current_hour_beginning_str = current_hour_beginning.isoformat()
                            
                            filtered_data.append((current_hour_beginning_str, last_price, last_volume))
                            logger.info(f"Keeping last available data point for {token_symbol} at {current_hour_beginning} (rounded to hour beginning)")
                        else:
                            logger.info(f"Data point already exists for hour {current_hour_beginning.hour}:00 for {token_symbol}, skipping duplicate")
                
                # Remove all existing data for this token
                cursor.execute('DELETE FROM crypto_prices WHERE token_symbol = ?', (token_symbol,))
                
                # Insert only the filtered data
                if filtered_data:
                    cursor.executemany('''
                        INSERT INTO crypto_prices 
                        (token_symbol, timestamp, price, volume) 
                        VALUES (?, ?, ?, ?)
                    ''', [(token_symbol, ts, p, v) for ts, p, v in filtered_data])
                    
                    logger.info(f"Cleaned up {token_symbol}: kept {len(filtered_data)} data points (removed {len(all_data) - len(filtered_data)} random hour points)")
                else:
                    logger.warning(f"No valid data points found for {token_symbol} after filtering")
            
            self.conn.commit()
            logger.info("Database cleanup completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {str(e)}")
            self.conn.rollback()
            raise 