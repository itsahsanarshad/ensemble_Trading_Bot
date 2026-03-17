"""Utils package initialization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .logger import (
    setup_logging,
    log_trade,
    log_signal,
    log_position_update,
    log_risk_event,
    log_model_prediction,
    logger,
)

# Auto-configure logging using LOG_LEVEL from .env
try:
    from config import settings
    setup_logging(level=settings.log_level.upper())
except Exception:
    setup_logging(level="INFO")

__all__ = [
    "setup_logging",
    "log_trade",
    "log_signal",
    "log_position_update",
    "log_risk_event",
    "log_model_prediction",
    "logger",
]
