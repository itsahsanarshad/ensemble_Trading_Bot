"""
Coin Watchlist Configuration

Top 30 coins by volume on Binance for monitoring.
All pairs are against USDT.
"""

from typing import List, Dict

# Primary watchlist - Top 30 by volume
WATCHLIST: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "POLUSDT",  # Formerly MATICUSDT
    "DOTUSDT",
    "TONUSDT",
    "NEARUSDT",
    "ARBUSDT",
    "OPUSDT",
    "SUIUSDT",
    "APTUSDT",
    "GRTUSDT",
    "UNIUSDT",
    "FILUSDT",
    "VETUSDT",
    "TAOUSDT",
    "ATOMUSDT",
    "ALGOUSDT",
    "HBARUSDT",
    "SEIUSDT",
]

# Coin metadata for position sizing adjustments
COIN_CONFIG: Dict[str, Dict] = {
    "BTCUSDT": {
        "min_trade_size": 0.0001,
        "volatility_factor": 0.8,  # Lower volatility, can increase position
        "category": "large_cap"
    },
    "ETHUSDT": {
        "min_trade_size": 0.001,
        "volatility_factor": 0.85,
        "category": "large_cap"
    },
    "SOLUSDT": {
        "min_trade_size": 0.01,
        "volatility_factor": 1.2,  # Higher volatility
        "category": "large_cap"
    },
}

# Default config for coins not in COIN_CONFIG
DEFAULT_COIN_CONFIG = {
    "min_trade_size": 1.0,
    "volatility_factor": 1.0,
    "category": "mid_cap"
}


def get_coin_config(symbol: str) -> Dict:
    """Get configuration for a specific coin."""
    return COIN_CONFIG.get(symbol, DEFAULT_COIN_CONFIG)


def get_active_coins() -> List[str]:
    """Get list of actively monitored coins."""
    return WATCHLIST.copy()


# Timeframes for data collection
TIMEFRAMES = {
    "1m": 1,      # 1 minute - for real-time monitoring
    "5m": 5,      # 5 minute
    "15m": 15,    # 15 minute - primary for signals
    "1h": 60,     # 1 hour - for LSTM sequences
    "4h": 240,    # 4 hour - for trend analysis
    "1d": 1440,   # 1 day - for long-term context
}

# Number of candles to fetch for each timeframe
CANDLE_LIMITS = {
    "1m": 1440,   # 1 day of data
    "5m": 576,    # 2 days of data
    "15m": 384,   # 4 days of data
    "1h": 720,    # 30 days of data (for LSTM 48h sequences)
    "4h": 360,    # 60 days of data
    "1d": 365,    # 1 year of data
}
