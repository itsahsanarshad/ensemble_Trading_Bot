"""
Binance Data Collector

Fetches OHLCV data from Binance API for all monitored coins.
Supports both REST API for historical data and real-time updates.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 4)[0])

from config import settings, WATCHLIST, TIMEFRAMES, CANDLE_LIMITS
from src.utils import logger
from src.data.database import db, PriceData


class BinanceCollector:
    """
    Collects OHLCV data from Binance API.
    
    Supports:
    - Historical data backfill
    - Real-time data updates
    - Multiple timeframes (1m, 15m, 1h, 4h, 1d)
    - Rate limiting and error handling
    """
    
    # Binance API rate limits
    WEIGHT_LIMIT = 1200  # per minute
    current_weight = 0
    weight_reset_time = time.time()
    
    def __init__(self):
        """Initialize Binance client."""
        self.client = self._create_client()
        self.public_client = self._create_public_client()
        self.data_cache: Dict[str, pd.DataFrame] = {}  # In-memory cache
        
    def _create_client(self) -> Client:
        """Create Binance API client for trading (authenticated)."""
        api_key = settings.exchange.binance_api_key
        api_secret = settings.exchange.binance_secret_key
        
        # Add 10s timeout to all requests to prevent bot stalling
        requests_params = {'timeout': 10}
        
        if settings.exchange.use_testnet:
            client = Client(api_key, api_secret, testnet=True, requests_params=requests_params)
            logger.info("Using Binance Testnet for trading (Timeout: 10s)")
        else:
            client = Client(api_key, api_secret, requests_params=requests_params)
            logger.info("Using Binance Production API for trading (Timeout: 10s)")
        
        return client
    
    def _create_public_client(self) -> Client:
        """Create public Binance client for data collection (no auth needed)."""
        # Add 10s timeout to public requests as well
        requests_params = {'timeout': 10}
        client = Client("", "", requests_params=requests_params)
        logger.info("Using Binance Public API for data collection (Timeout: 10s)")
        return client
    
    def _check_rate_limit(self, weight: int = 1) -> None:
        """Check and handle API rate limiting."""
        current_time = time.time()
        
        # Reset weight counter every minute
        if current_time - self.weight_reset_time >= 60:
            self.current_weight = 0
            self.weight_reset_time = current_time
        
        # If approaching limit, wait
        if self.current_weight + weight > self.WEIGHT_LIMIT:
            wait_time = 60 - (current_time - self.weight_reset_time)
            logger.warning(f"Rate limit approaching, waiting {wait_time:.1f}s")
            time.sleep(wait_time + 1)
            self.current_weight = 0
            self.weight_reset_time = time.time()
        
        self.current_weight += weight
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert our timeframe format to Binance format."""
        mapping = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }
        return mapping.get(timeframe, Client.KLINE_INTERVAL_15MINUTE)
    
    def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 500,
        start_time: int = None,
        use_public: bool = True
    ) -> List[Dict]:
        """
        Fetch candlestick data from Binance.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            timeframe: Candlestick interval
            limit: Number of candles to fetch (max 1000)
            start_time: Start timestamp in milliseconds
            use_public: Use public API (no auth, mainnet data)
        
        Returns:
            List of OHLCV dictionaries
        """
        self._check_rate_limit(weight=1)
        
        try:
            interval = self._convert_timeframe(timeframe)
            
            kwargs = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000),
            }
            
            if start_time:
                kwargs["startTime"] = start_time
            
            # Use public client for historical data, authenticated for live
            client = self.public_client if use_public else self.client
            klines = client.get_klines(**kwargs)
            
            return [
                {
                    "timestamp": kline[0],  # Open time
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "close_time": kline[6],
                    "quote_volume": float(kline[7]),
                    "num_trades": int(kline[8]),
                    "taker_buy_base": float(kline[9]),
                    "taker_buy_quote": float(kline[10]),
                }
                for kline in klines
            ]
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching {symbol} klines: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        self._check_rate_limit(weight=1)
        
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_all_prices(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get current prices for multiple symbols."""
        self._check_rate_limit(weight=2)
        
        symbols = symbols or WATCHLIST
        
        try:
            tickers = self.client.get_all_tickers()
            prices = {t["symbol"]: float(t["price"]) for t in tickers}
            return {s: prices.get(s, 0.0) for s in symbols}
        except Exception as e:
            logger.error(f"Error getting all prices: {e}")
            return {}
    
    def get_24h_stats(self, symbol: str) -> Optional[Dict]:
        """Get 24-hour statistics for a symbol."""
        self._check_rate_limit(weight=1)
        
        try:
            stats = self.client.get_ticker(symbol=symbol)
            return {
                "price_change": float(stats["priceChange"]),
                "price_change_percent": float(stats["priceChangePercent"]),
                "high": float(stats["highPrice"]),
                "low": float(stats["lowPrice"]),
                "volume": float(stats["volume"]),
                "quote_volume": float(stats["quoteVolume"]),
                "weighted_avg_price": float(stats["weightedAvgPrice"]),
                "last_price": float(stats["lastPrice"]),
                "bid_price": float(stats["bidPrice"]),
                "ask_price": float(stats["askPrice"]),
                "open_price": float(stats["openPrice"]),
                "count": int(stats["count"]),  # Number of trades
            }
        except Exception as e:
            logger.error(f"Error getting 24h stats for {symbol}: {e}")
            return None
    
    def backfill_historical_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 30
    ) -> int:
        """
        Backfill historical data for a symbol using public API.
        
        Args:
            symbol: Trading pair
            timeframe: Candlestick interval
            days: Number of days to backfill
        
        Returns:
            Number of records saved
        """
        logger.info(f"Backfilling {days} days of {timeframe} data for {symbol}")
        
        total_saved = 0
        
        # Calculate timestamps
        end_time = int(time.time() * 1000)
        
        # Calculate start time based on days
        start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        
        # Fetch in batches of 1000
        current_start = start_time
        batch_count = 0
        
        while current_start < end_time:
            klines = self.fetch_klines(
                symbol=symbol,
                timeframe=timeframe,
                limit=1000,
                start_time=current_start,
                use_public=True  # Use public API for historical data
            )
            
            if not klines:
                break
            
            # Save to database
            saved = db.save_price_data(klines, symbol, timeframe)
            total_saved += saved
            batch_count += 1
            
            # Progress update every 5 batches
            if batch_count % 5 == 0:
                logger.info(f"  Progress: {total_saved} records saved for {symbol} {timeframe}")
            
            # Update start time for next batch
            current_start = klines[-1]["timestamp"] + 1
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        logger.info(f"✅ Backfilled {total_saved} records for {symbol} {timeframe}")
        return total_saved
    
    def update_all_coins(self, timeframe: str = "15m", limit: int = 100) -> Dict[str, int]:
        """
        Update data for all monitored coins.
        
        Args:
            timeframe: Candlestick interval
            limit: Number of candles per coin
        
        Returns:
            Dict of symbol -> records saved
        """
        results = {}
        
        for symbol in WATCHLIST:
            try:
                # Use authenticated client for live updates
                klines = self.fetch_klines(symbol, timeframe, limit, use_public=False)
                saved = db.save_price_data(klines, symbol, timeframe)
                results[symbol] = saved
                
                # Update cache
                if klines:
                    df = pd.DataFrame(klines)
                    cache_key = f"{symbol}_{timeframe}"
                    self.data_cache[cache_key] = df
                
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                results[symbol] = 0
        
        logger.info(f"Updated {len(results)} coins, saved {sum(results.values())} records")
        return results
    
    def get_dataframe(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get price data as a pandas DataFrame.
        
        Args:
            symbol: Trading pair
            timeframe: Candlestick interval
            limit: Number of candles
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache first
        if cache_key in self.data_cache:
            cached = self.data_cache[cache_key]
            if len(cached) >= limit:
                return cached.tail(limit).reset_index(drop=True)
        
        # Fetch fresh data
        klines = self.fetch_klines(symbol, timeframe, limit)
        
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        
        # Cache it
        self.data_cache[cache_key] = df
        
        return df
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """
        Get order book depth for a symbol.
        
        Args:
            symbol: Trading pair
            limit: Depth limit (max 5000)
        
        Returns:
            Order book with bids and asks
        """
        self._check_rate_limit(weight=5 if limit > 100 else 1)
        
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            
            bids = [(float(p), float(q)) for p, q in depth["bids"]]
            asks = [(float(p), float(q)) for p, q in depth["asks"]]
            
            # Calculate metrics
            total_bid_volume = sum(q for _, q in bids)
            total_ask_volume = sum(q for _, q in asks)
            
            # Best bid/ask
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            
            # Calculate depth at different levels (1%, 2%, 5%)
            bid_depth_1pct = sum(q for p, q in bids if p >= best_bid * 0.99)
            bid_depth_2pct = sum(q for p, q in bids if p >= best_bid * 0.98)
            bid_depth_5pct = sum(q for p, q in bids if p >= best_bid * 0.95)
            
            ask_depth_1pct = sum(q for p, q in asks if p <= best_ask * 1.01)
            ask_depth_2pct = sum(q for p, q in asks if p <= best_ask * 1.02)
            ask_depth_5pct = sum(q for p, q in asks if p <= best_ask * 1.05)
            
            # Imbalance score (-1 to 1, positive = more buying pressure)
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
            
            # Large order detection (walls)
            avg_bid_size = total_bid_volume / len(bids) if bids else 0
            avg_ask_size = total_ask_volume / len(asks) if asks else 0
            
            bid_walls = [(p, q) for p, q in bids if q > avg_bid_size * 5]
            ask_walls = [(p, q) for p, q in asks if q > avg_ask_size * 5]
            
            return {
                "bids": bids[:20],  # Top 20 only
                "asks": asks[:20],
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,
                "spread": (best_ask - best_bid) / best_bid if best_bid > 0 else 0,
                "spread_pct": ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0,
                "bid_volume": total_bid_volume,
                "ask_volume": total_ask_volume,
                "bid_ask_ratio": total_bid_volume / total_ask_volume if total_ask_volume > 0 else 0,
                "imbalance": imbalance,
                "bid_depth_1pct": bid_depth_1pct,
                "bid_depth_2pct": bid_depth_2pct,
                "ask_depth_1pct": ask_depth_1pct,
                "ask_depth_2pct": ask_depth_2pct,
                "bid_walls": bid_walls[:3],  # Top 3 walls
                "ask_walls": ask_walls[:3],
                "liquidity_score": min((total_bid_volume + total_ask_volume) / 1000, 100),  # 0-100 scale
            }
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return None
    
    def check_api_connection(self) -> bool:
        """Test API connection."""
        try:
            self.client.ping()
            server_time = self.client.get_server_time()
            logger.info(f"Binance API connected. Server time: {server_time}")
            return True
        except Exception as e:
            logger.error(f"Binance API connection failed: {e}")
            return False


# Global collector instance
collector = BinanceCollector()
