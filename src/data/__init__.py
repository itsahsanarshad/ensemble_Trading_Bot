"""Data package initialization."""

from .database import (
    db,
    DatabaseManager,
    PriceData,
    Indicator,
    Trade,
    ModelPerformance,
    DailyPerformance,
    TradeStatus,
    ExitReason,
)

# Lazy load collector to avoid API calls at import time
try:
    from .collector import collector, BinanceCollector
except Exception as e:
    # API may be temporarily unavailable
    collector = None
    BinanceCollector = None
    import logging
    logging.warning(f"Could not initialize collector: {e}")

from .features import feature_engineer, FeatureEngineer

__all__ = [
    "db",
    "DatabaseManager",
    "PriceData",
    "Indicator",
    "Trade",
    "ModelPerformance",
    "DailyPerformance",
    "TradeStatus",
    "ExitReason",
    "collector",
    "BinanceCollector",
    "feature_engineer",
    "FeatureEngineer",
]
