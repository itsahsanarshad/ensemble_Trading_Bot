"""
Feature Engineering Module

Creates 80+ features for machine learning models from raw OHLCV data.
Features include price, volume, technical indicators, volatility, patterns, and market context.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Try to import ta-lib alternative
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 4)[0])

from src.utils import logger


class FeatureEngineer:
    """
    Feature engineering for crypto trading.
    
    Creates 80+ features grouped into:
    1. Price features (15)
    2. Volume features (12)
    3. Technical indicators (25)
    4. Volatility features (8)
    5. Pattern features (10)
    6. Market context (10)
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names: List[str] = []
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with all features added
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required):
            logger.error("Missing required OHLCV columns")
            return df
        
        # Engineer each feature category
        df = self._price_features(df)
        df = self._volume_features(df)
        df = self._technical_indicators(df)
        df = self._volatility_features(df)
        df = self._pattern_features(df)
        df = self._time_features(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in required + ["timestamp", "datetime", "close_time", "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote"]]
        
        return df
    
    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features (15 features)."""
        
        # Returns at different intervals
        df["return_1"] = df["close"].pct_change(1)
        df["return_4"] = df["close"].pct_change(4)    # 1h with 15m candles
        df["return_16"] = df["close"].pct_change(16)  # 4h with 15m candles
        df["return_96"] = df["close"].pct_change(96)  # 24h with 15m candles
        
        # Price relative to moving averages
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["sma_200"] = df["close"].rolling(window=200, min_periods=1).mean()
        
        df["price_vs_ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
        df["price_vs_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
        df["price_vs_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
        
        # High-low spread
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
        
        # Price momentum (converted to percentage)
        df["momentum_10"] = df["close"].pct_change(10)
        df["momentum_20"] = df["close"].pct_change(20)
        
        # Rate of change
        df["roc_12"] = ((df["close"] - df["close"].shift(12)) / df["close"].shift(12)) * 100
        df["roc_24"] = ((df["close"] - df["close"].shift(24)) / df["close"].shift(24)) * 100
        
        return df
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features (12 features)."""
        
        # Volume moving averages
        df["volume_sma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_sma_50"] = df["volume"].rolling(window=50, min_periods=1).mean()
        
        # Volume ratio to average
        df["volume_ratio_20"] = df["volume"] / df["volume_sma_20"]
        df["volume_ratio_50"] = df["volume"] / df["volume_sma_50"]
        
        # Volume change
        df["volume_change"] = df["volume"].pct_change(1)
        df["volume_change_4"] = df["volume"].pct_change(4)
        
        # Volume trend (slope of volume SMA)
        df["volume_trend"] = df["volume_sma_20"].diff(5) / df["volume_sma_20"].shift(5)
        
        # Volume-price correlation (24 periods)
        df["vol_price_corr"] = df["close"].rolling(24).corr(df["volume"])
        
        # On-Balance Volume (OBV) - FIXED: use rolling instead of cumsum to avoid unbounded growth
        df["price_direction"] = np.where(df["close"] > df["close"].shift(1), 1, 
                                          np.where(df["close"] < df["close"].shift(1), -1, 0))
        # Use rolling sum of signed volume instead of cumulative (avoids data leakage)
        df["obv_rolling"] = (df["volume"] * df["price_direction"]).rolling(window=20, min_periods=1).sum()
        df["obv_sma"] = df["obv_rolling"].rolling(window=20, min_periods=1).mean()
        df["obv_vs_sma"] = (df["obv_rolling"] - df["obv_sma"]) / (df["obv_sma"].abs().replace(0, 1))
        
        # For backward compatibility, also create 'obv' as normalized rolling value
        df["obv"] = df["obv_rolling"]
        
        # Volume weighted average price (VWAP) - FIXED: use rolling instead of cumulative
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).rolling(window=20, min_periods=1).sum() / df["volume"].rolling(window=20, min_periods=1).sum()
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
        
        return df
    
    def _technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators (25 features)."""
        
        # RSI at multiple periods
        df["rsi_7"] = self._calculate_rsi(df["close"], 7)
        df["rsi_14"] = self._calculate_rsi(df["close"], 14)
        df["rsi_21"] = self._calculate_rsi(df["close"], 21)
        
        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Stochastic Oscillator
        low_14 = df["low"].rolling(window=14, min_periods=1).min()
        high_14 = df["high"].rolling(window=14, min_periods=1).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(window=3, min_periods=1).mean()
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20, min_periods=1).mean()
        bb_std = df["close"].rolling(window=20, min_periods=1).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        
        # ADX (Average Directional Index)
        df["adx"] = self._calculate_adx(df, 14)
        
        # Williams %R
        df["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14 + 1e-10)
        
        # CCI (Commodity Channel Index)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(window=20, min_periods=1).mean()
        mean_dev = typical_price.rolling(window=20, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci"] = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-10)
        
        # CMF (Chaikin Money Flow)
        mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-10) * df["volume"]
        df["cmf"] = mfv.rolling(window=20, min_periods=1).sum() / df["volume"].rolling(window=20, min_periods=1).sum()
        
        # Moving average crossovers
        df["ema_20_50_cross"] = np.where(df["ema_20"] > df["ema_50"], 1, -1)
        df["macd_cross"] = np.where(df["macd"] > df["macd_signal"], 1, -1)
        
        return df
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features (8 features)."""
        
        # ATR (Average True Range)
        df["atr_14"] = self._calculate_atr(df, 14)
        df["atr_7"] = self._calculate_atr(df, 7)
        
        # Standard deviation
        df["std_24"] = df["close"].rolling(window=24, min_periods=1).std()
        df["std_normalized"] = df["std_24"] / df["close"]
        
        # Historical volatility
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_24h"] = df["log_return"].rolling(window=96, min_periods=1).std() * np.sqrt(96)
        
        # Bollinger Band width already calculated above
        
        # True Range
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                np.abs(df["high"] - df["close"].shift(1)),
                np.abs(df["low"] - df["close"].shift(1))
            )
        )
        
        # Garman-Klass volatility estimator
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2
        df["gk_volatility"] = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
        
        return df
    
    def _pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pattern recognition features (10 features)."""
        
        # Higher highs and higher lows
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["higher_low"] = (df["low"] > df["low"].shift(1)).astype(int)
        df["lower_high"] = (df["high"] < df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
        
        # Consecutive higher highs/lows
        df["consec_hh"] = df["higher_high"].rolling(window=5, min_periods=1).sum()
        df["consec_hl"] = df["higher_low"].rolling(window=5, min_periods=1).sum()
        
        # Uptrend/Downtrend indicator
        df["uptrend"] = ((df["consec_hh"] >= 3) & (df["consec_hl"] >= 3)).astype(int)
        df["downtrend"] = ((df["lower_high"].rolling(window=5, min_periods=1).sum() >= 3) & 
                           (df["lower_low"].rolling(window=5, min_periods=1).sum() >= 3)).astype(int)
        
        # Consolidation detection (low volatility)
        df["consolidation"] = (df["bb_width"] < df["bb_width"].rolling(window=50, min_periods=1).quantile(0.25)).astype(int)
        
        # Breakout magnitude from 20-period range
        high_20 = df["high"].rolling(window=20, min_periods=1).max()
        low_20 = df["low"].rolling(window=20, min_periods=1).min()
        df["breakout_magnitude"] = (df["close"] - high_20.shift(1)) / (high_20.shift(1) - low_20.shift(1) + 1e-10)
        
        # Distance from recent high/low
        df["dist_from_high_20"] = (high_20 - df["close"]) / df["close"]
        df["dist_from_low_20"] = (df["close"] - low_20) / df["close"]
        
        return df
    
    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features (6 features)."""
        
        if "datetime" in df.columns:
            dt = df["datetime"]
        elif "timestamp" in df.columns:
            dt = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            # If no time info, skip
            return df
        
        # Extract time components
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        
        # Market session (rough approximation)
        # Asian: 0-8 UTC, European: 8-16 UTC, American: 16-24 UTC
        df["asian_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
        df["european_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
        df["american_session"] = (df["hour"] >= 16).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range
        tr = pd.concat([
            high - low,
            np.abs(high - close.shift(1)),
            np.abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR and DM
        atr = pd.Series(tr).rolling(window=period, min_periods=1).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=1).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=1).mean() / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(window=period, min_periods=1).mean()
        
        return adx
    
    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_pct: float = 0.10,
        lookahead_periods: int = 16,
        use_adaptive_atr: bool = False
    ) -> pd.Series:
        """
        Create binary target variable for ML training.
        
        Args:
            df: DataFrame with close prices (and atr_14 if using adaptive)
            target_pct: Target percentage increase (default 10%, ignored if adaptive)
            lookahead_periods: Number of periods to look ahead (16 for 4h with 15m candles)
            use_adaptive_atr: If True, use 1.5x ATR (capped 1-4%) instead of fixed target
        
        Returns:
            Binary series (1 if price increases by target, 0 otherwise)
        """
        if use_adaptive_atr:
            # Adaptive ATR-based target (same as TCN)
            ATR_MULTIPLIER = 1.5
            MIN_TARGET = 0.01  # 1% minimum
            MAX_TARGET = 0.04  # 4% maximum
            
            # Calculate ATR if not present
            if 'atr_14' not in df.columns:
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift())
                tr3 = abs(df['low'] - df['close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
            else:
                atr = df['atr_14']
            
            # Convert ATR to percentage of price
            atr_pct = atr / df['close']
            
            # Adaptive target: 1.5x ATR, capped between 1% and 4%
            adaptive_target = (atr_pct * ATR_MULTIPLIER).clip(MIN_TARGET, MAX_TARGET)
            
            # Maximum price in next N periods
            future_max = df["high"].rolling(window=lookahead_periods, min_periods=1).max().shift(-lookahead_periods)
            
            # Did price increase by adaptive target?
            actual_gain = future_max / df["close"] - 1
            target = (actual_gain > adaptive_target).astype(int)
        else:
            # Fixed target (original behavior)
            # Maximum price in next N periods
            future_max = df["high"].rolling(window=lookahead_periods, min_periods=1).max().shift(-lookahead_periods)
            
            # Did price increase by target_pct?
            target = (future_max / df["close"] > (1 + target_pct)).astype(int)
        
        return target
    
    def prepare_lstm_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 48,
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare sequences for LSTM model.
        
        Args:
            df: DataFrame with features
            sequence_length: Number of timesteps per sequence
            feature_cols: List of feature columns to use
        
        Returns:
            Tuple of (sequences array, feature names)
        """
        if feature_cols is None:
            # Default features for LSTM (30 features)
            feature_cols = [
                "open", "high", "low", "close", "volume",
                "price_vs_ema20", "price_vs_ema50",
                "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "atr_14",
                "volume_ratio_20", "volume_change",
                "return_1", "return_4", "volatility_24h",
                "roc_12", "momentum_10", "adx",
                "higher_high", "higher_low", "uptrend",
                "hour", "day_of_week",
                "bb_position", "stoch_k", "cci", "cmf"
            ]
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        
        # Normalize features
        data = df[available_cols].copy()
        
        # Handle infinities and NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.ffill().bfill().fillna(0)
        
        # Z-score normalization
        means = data.mean()
        stds = data.std()
        stds = stds.replace(0, 1)  # Avoid division by zero
        data = (data - means) / stds
        
        values = data.values
        
        # Create sequences
        sequences = []
        for i in range(len(values) - sequence_length + 1):
            seq = values[i:i + sequence_length]
            sequences.append(seq)
        
        return np.array(sequences), available_cols
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category for analysis."""
        return {
            "price": [
                "return_1", "return_4", "return_16", "return_96",
                "price_vs_ema20", "price_vs_ema50", "price_vs_sma200",
                "hl_spread", "momentum_10", "momentum_20", "roc_12", "roc_24"
            ],
            "volume": [
                "volume_ratio_20", "volume_ratio_50", "volume_change",
                "volume_change_4", "volume_trend", "vol_price_corr",
                "obv_vs_sma", "price_vs_vwap"
            ],
            "technical": [
                "rsi_7", "rsi_14", "rsi_21",
                "macd", "macd_signal", "macd_histogram",
                "stoch_k", "stoch_d", "bb_position", "bb_width",
                "adx", "williams_r", "cci", "cmf"
            ],
            "volatility": [
                "atr_14", "atr_7", "std_24", "std_normalized",
                "volatility_24h", "true_range", "gk_volatility"
            ],
            "pattern": [
                "higher_high", "higher_low", "consec_hh", "consec_hl",
                "uptrend", "downtrend", "consolidation",
                "breakout_magnitude", "dist_from_high_20", "dist_from_low_20"
            ],
            "time": [
                "hour", "day_of_week", "is_weekend",
                "asian_session", "european_session", "american_session"
            ]
        }
    def engineer_multi_timeframe_features_training(
        self,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_4h: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer time-aligned multi-timeframe features for TRAINING.
        
        For each 1h row, looks up the corresponding 15m and 4h values
        from the same time period to create proper training features.
        
        Args:
            df_1h: Primary 1h timeframe data with features
            df_15m: 15-minute data with features  
            df_4h: 4-hour data with features
        
        Returns:
            df_1h with MTF features properly aligned by timestamp
        """
        df = df_1h.copy()
        
        # Ensure timestamp columns exist
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column in 1h data, skipping MTF")
            return df
        
        # Convert timestamps to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Engineer features for 15m and 4h if not already done
        if df_15m is not None and not df_15m.empty:
            if 'rsi_14' not in df_15m.columns:
                df_15m = self.engineer_all_features(df_15m)
            if 'timestamp' in df_15m.columns and not pd.api.types.is_datetime64_any_dtype(df_15m['timestamp']):
                df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
        
        if df_4h is not None and not df_4h.empty:
            if 'rsi_14' not in df_4h.columns:
                df_4h = self.engineer_all_features(df_4h)
            if 'timestamp' in df_4h.columns and not pd.api.types.is_datetime64_any_dtype(df_4h['timestamp']):
                df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
        
        # Create MTF features aligned by timestamp
        logger.info("Creating time-aligned 15m features...")
        
        # Initialize MTF columns with defaults
        mtf_cols_15m = ['rsi_14_15m', 'macd_15m', 'macd_histogram_15m', 
                        'volume_ratio_15m', 'trend_15m', 'momentum_15m', 'bullish_15m']
        mtf_cols_4h = ['rsi_14_4h', 'macd_4h', 'macd_histogram_4h', 'adx_4h',
                       'trend_4h', 'price_vs_ema20_4h', 'bullish_4h']
        
        for col in mtf_cols_15m + mtf_cols_4h:
            df[col] = 0.0
        
        # Process 15m features - for each 1h row, get the 15m values from that hour
        if df_15m is not None and not df_15m.empty and 'timestamp' in df_15m.columns:
            # Create hour bucket for 15m data
            df_15m['hour_bucket'] = df_15m['timestamp'].dt.floor('h')
            
            # Group 15m data by hour and get aggregated values
            agg_15m = df_15m.groupby('hour_bucket').agg({
                'rsi_14': 'mean',
                'macd': 'mean', 
                'macd_histogram': 'mean',
                'volume_ratio_20': 'mean',
                'close': ['first', 'last']
            }).reset_index()
            
            # Flatten column names
            agg_15m.columns = ['hour_bucket', 'rsi_14_15m', 'macd_15m', 
                              'macd_histogram_15m', 'volume_ratio_15m',
                              'close_first', 'close_last']
            
            # Calculate trend and momentum
            agg_15m['trend_15m'] = (agg_15m['close_last'] > agg_15m['close_first']).astype(int) * 2 - 1
            agg_15m['momentum_15m'] = (agg_15m['close_last'] - agg_15m['close_first']) / agg_15m['close_first']
            agg_15m['bullish_15m'] = ((agg_15m['rsi_14_15m'] > 50) & (agg_15m['trend_15m'] > 0)).astype(int)
            
            # Drop helper columns
            agg_15m = agg_15m.drop(columns=['close_first', 'close_last'])
            
            # Create hour bucket for 1h data to merge
            df['hour_bucket'] = df['timestamp'].dt.floor('h')
            
            # Merge 15m features
            df = df.merge(agg_15m, on='hour_bucket', how='left', suffixes=('', '_new'))
            
            # Update columns (handle case where merge creates _new columns)
            for col in mtf_cols_15m:
                if f'{col}_new' in df.columns:
                    df[col] = df[f'{col}_new'].fillna(df[col])
                    df = df.drop(columns=[f'{col}_new'])
                elif col in df.columns:
                    df[col] = df[col].fillna(0)
            
            df = df.drop(columns=['hour_bucket'], errors='ignore')
        
        # Process 4h features - for each 1h row, get the 4h values from containing period
        logger.info("Creating time-aligned 4h features...")
        if df_4h is not None and not df_4h.empty and 'timestamp' in df_4h.columns:
            # Create 4h bucket for 4h data
            df_4h['h4_bucket'] = df_4h['timestamp'].dt.floor('4h')
            
            # Get 4h features per bucket
            features_4h = df_4h.groupby('h4_bucket').agg({
                'rsi_14': 'last',
                'macd': 'last',
                'macd_histogram': 'last',
                'adx': 'last',
                'price_vs_ema20': 'last',
                'ema_20': 'last',
                'ema_50': 'last'
            }).reset_index()
            
            features_4h.columns = ['h4_bucket', 'rsi_14_4h', 'macd_4h', 
                                   'macd_histogram_4h', 'adx_4h', 
                                   'price_vs_ema20_4h', 'ema_20_4h', 'ema_50_4h']
            
            # Calculate trend and bullish
            features_4h['trend_4h'] = (features_4h['ema_20_4h'] > features_4h['ema_50_4h']).astype(int) * 2 - 1
            features_4h['bullish_4h'] = ((features_4h['rsi_14_4h'] > 50) & (features_4h['trend_4h'] > 0)).astype(int)
            
            # Drop helper columns
            features_4h = features_4h.drop(columns=['ema_20_4h', 'ema_50_4h'])
            
            # CRITICAL FIX: Shift 4h aggregated features forward by one 4h bucket
            # This ensures that our 1h candles only see the PREVIOUS fully closed 4h candle
            features_4h['h4_bucket'] = features_4h['h4_bucket'] + pd.Timedelta(hours=4)
            
            # Create 4h bucket for 1h data to merge
            df['h4_bucket'] = df['timestamp'].dt.floor('4h')
            
            # Merge 4h features
            df = df.merge(features_4h, on='h4_bucket', how='left', suffixes=('', '_new'))
            
            # Update columns
            for col in mtf_cols_4h:
                if f'{col}_new' in df.columns:
                    df[col] = df[f'{col}_new'].fillna(df[col])
                    df = df.drop(columns=[f'{col}_new'])
                elif col in df.columns:
                    df[col] = df[col].fillna(0)
            
            df = df.drop(columns=['h4_bucket'], errors='ignore')
        
        # Calculate MTF alignment
        df['mtf_alignment'] = (
            df['bullish_15m'].astype(int) + 
            ((df['rsi_14'] > 50).astype(int) if 'rsi_14' in df.columns else 0) +
            df['bullish_4h'].astype(int)
        )
        df['all_timeframes_bullish'] = (df['mtf_alignment'] == 3).astype(int)
        
        # Fill any remaining NaN
        for col in mtf_cols_15m + mtf_cols_4h + ['mtf_alignment', 'all_timeframes_bullish']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        logger.info(f"Added {len(mtf_cols_15m + mtf_cols_4h) + 2} time-aligned MTF features")
        
        return df
    
    def engineer_multi_timeframe_features(
        self,
        df_primary: pd.DataFrame,
        df_15m: pd.DataFrame = None,
        df_4h: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Engineer MTF features for PREDICTION (uses latest values).
        
        For real-time prediction, we just need the latest 15m/4h values.
        """
        df = df_primary.copy()
        
        # Add 15m confirmation features (short-term momentum)
        if df_15m is not None and not df_15m.empty:
            df_15m_feat = self.engineer_all_features(df_15m)
            
            if 'timestamp' in df_primary.columns and 'timestamp' in df_15m_feat.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_15m_feat['timestamp']):
                    df_15m_feat['timestamp'] = pd.to_datetime(df_15m_feat['timestamp'])
                target_ts = pd.to_datetime(df_primary['timestamp'].iloc[-1])
                mask = df_15m_feat['timestamp'] < (target_ts + pd.Timedelta(hours=1))
                latest_15m = df_15m_feat[mask].tail(4)
            else:
                latest_15m = df_15m_feat.tail(4)
                
            if len(latest_15m) >= 1:
                
                df["rsi_14_15m"] = latest_15m["rsi_14"].iloc[-1] if "rsi_14" in latest_15m else 50
                df["macd_15m"] = latest_15m["macd"].iloc[-1] if "macd" in latest_15m else 0
                df["macd_histogram_15m"] = latest_15m["macd_histogram"].iloc[-1] if "macd_histogram" in latest_15m else 0
                df["volume_ratio_15m"] = latest_15m["volume_ratio_20"].iloc[-1] if "volume_ratio_20" in latest_15m else 1
                
                if "close" in latest_15m:
                    df["trend_15m"] = 1 if latest_15m["close"].iloc[-1] > latest_15m["close"].iloc[0] else -1
                    df["momentum_15m"] = (latest_15m["close"].iloc[-1] - latest_15m["close"].iloc[0]) / latest_15m["close"].iloc[0]
                else:
                    df["trend_15m"] = 0
                    df["momentum_15m"] = 0
                
                rsi_15m = df["rsi_14_15m"].iloc[-1] if len(df) > 0 else 50
                df["bullish_15m"] = 1 if rsi_15m > 50 and df["trend_15m"].iloc[-1] > 0 else 0
        
        # Add 4h confirmation features
        if df_4h is not None and not df_4h.empty:
            df_4h_feat = self.engineer_all_features(df_4h)
            
            if 'timestamp' in df_primary.columns and 'timestamp' in df_4h_feat.columns:
                if not pd.api.types.is_datetime64_any_dtype(df_4h_feat['timestamp']):
                    df_4h_feat['timestamp'] = pd.to_datetime(df_4h_feat['timestamp'])
                target_ts = pd.to_datetime(df_primary['timestamp'].iloc[-1])
                target_4h_ts = target_ts.floor('4h') - pd.Timedelta(hours=4)
                mask = df_4h_feat['timestamp'] <= target_4h_ts
                latest_4h = df_4h_feat[mask].tail(1)
            else:
                latest_4h = df_4h_feat.tail(1)
                
            if len(latest_4h) >= 1:
                
                df["rsi_14_4h"] = latest_4h["rsi_14"].iloc[-1] if "rsi_14" in latest_4h else 50
                df["macd_4h"] = latest_4h["macd"].iloc[-1] if "macd" in latest_4h else 0
                df["macd_histogram_4h"] = latest_4h["macd_histogram"].iloc[-1] if "macd_histogram" in latest_4h else 0
                df["adx_4h"] = latest_4h["adx"].iloc[-1] if "adx" in latest_4h else 25
                
                if "ema_20" in latest_4h and "ema_50" in latest_4h:
                    df["trend_4h"] = 1 if latest_4h["ema_20"].iloc[-1] > latest_4h["ema_50"].iloc[-1] else -1
                else:
                    df["trend_4h"] = 0
                
                if "price_vs_ema20" in latest_4h:
                    df["price_vs_ema20_4h"] = latest_4h["price_vs_ema20"].iloc[-1]
                else:
                    df["price_vs_ema20_4h"] = 0
                
                rsi_4h = df["rsi_14_4h"].iloc[-1] if len(df) > 0 else 50
                trend_4h = df["trend_4h"].iloc[-1] if len(df) > 0 else 0
                df["bullish_4h"] = 1 if rsi_4h > 50 and trend_4h > 0 else 0
        
        # MTF alignment
        if "bullish_15m" in df.columns and "bullish_4h" in df.columns:
            bullish_1h = 1 if df["rsi_14"].iloc[-1] > 50 else 0 if "rsi_14" in df else 0
            df["mtf_alignment"] = df["bullish_15m"] + bullish_1h + df["bullish_4h"]
            df["all_timeframes_bullish"] = (df["mtf_alignment"] == 3).astype(int)
        
        return df


# Global feature engineer instance
feature_engineer = FeatureEngineer()
