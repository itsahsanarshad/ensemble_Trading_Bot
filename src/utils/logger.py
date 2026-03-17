"""
Logging System for Crypto Trading Bot

Uses loguru for structured, colored logging with file rotation.
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

_LOGGING_INITIALIZED = False  # Guard against duplicate handler registration


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the logging system.
    Safe to call multiple times — only initializes once.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    _LOGGING_INITIALIZED = True
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # General log file (rotates daily, keeps 30 days)
    logger.add(
        LOGS_DIR / "bot_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
        compression="zip",
    )
    
    # Trade-specific log (never delete trades)
    logger.add(
        LOGS_DIR / "trades_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        rotation="00:00",
        retention="365 days",  # Keep for 1 year
        filter=lambda record: "TRADE" in record["extra"].get("tags", []),
    )
    
    # Error log (for debugging)
    logger.add(
        LOGS_DIR / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        backtrace=True,
        diagnose=True,
    )
    
    logger.info("Logging system initialized")


def log_trade(
    action: str,
    coin: str,
    price: float,
    size: float,
    reason: str,
    tier: int = 0,
    pnl: float = None,
    **kwargs
) -> None:
    """
    Log a trade with structured data.
    
    Args:
        action: BUY, SELL, STOP_LOSS, TAKE_PROFIT, etc.
        coin: Trading pair (e.g., BTCUSDT)
        price: Execution price
        size: Position size in USDT
        reason: Reason for the trade
        tier: Consensus tier (1-4)
        pnl: Profit/loss if closing position
    """
    trade_log = logger.bind(tags=["TRADE"])
    
    if pnl is not None:
        pnl_str = f"+{pnl:.2f}%" if pnl > 0 else f"{pnl:.2f}%"
        trade_log.info(
            f"[{action}] {coin} @ ${price:.4f} | Size: ${size:.2f} | "
            f"P&L: {pnl_str} | Reason: {reason}"
        )
    else:
        trade_log.info(
            f"[{action}] {coin} @ ${price:.4f} | Size: ${size:.2f} | "
            f"Tier: {tier} | Reason: {reason}"
        )


def log_signal(
    coin: str,
    ta_conf: float,
    ml_conf: float,
    tcn_conf: float,
    decision: str,
    tier: int,
    ta_signal: str = None,
    ml_signal: str = None,
    tcn_signal: str = None,
    reasons: list = None
) -> None:
    """Log a signal decision with full details."""
    # Build signal summary
    ta_sig = f"TA:{ta_signal}({ta_conf:.0%})" if ta_signal else f"TA:{ta_conf:.0%}"
    ml_sig = f"ML:{ml_signal}({ml_conf:.0%})" if ml_signal else f"ML:{ml_conf:.0%}"
    tcn_sig = f"TCN:{tcn_signal}({tcn_conf:.0%})" if tcn_signal else f"TCN:{tcn_conf:.0%}"
    
    # Decision emoji
    emoji = "🟢" if decision == "buy" else ("🔴" if decision == "sell" else "⚪")
    
    logger.info(
        f"📊 SIGNAL | {coin} | {ta_sig} | {ml_sig} | {tcn_sig} | "
        f"{emoji} {decision.upper()} | Tier: {tier}"
    )
    
    # Log reasons if provided
    if reasons:
        reason_str = " | ".join(reasons[:3])
        logger.info(f"   └─ Reasons: {reason_str}")


def log_position_update(
    coin: str,
    entry_price: float,
    current_price: float,
    pnl_percent: float,
    status: str,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    take_profit_1: float = 0.0
) -> None:
    """Log position status update."""
    pnl_str = f"+{pnl_percent:.2f}%" if pnl_percent > 0 else f"{pnl_percent:.2f}%"
    emoji = "📈" if pnl_percent > 0 else "📉"
    
    tp1_str = f" | TP1: ${take_profit_1:.4f}" if take_profit_1 > 0 else ""
    tp2_str = f" | TP2: ${take_profit:.4f}" if take_profit > 0 else ""
    sl_str = f" | SL: ${stop_loss:.4f}" if stop_loss > 0 else ""
    
    logger.info(
        f"{emoji} POSITION | {coin} | Entry: ${entry_price:.4f} | "
        f"Current: ${current_price:.4f} | P&L: {pnl_str}{tp1_str}{tp2_str}{sl_str} | {status}"
    )


def log_risk_event(event: str, details: str) -> None:
    """Log risk management events."""
    logger.warning(f"⚠️ RISK | {event} | {details}")


def log_model_prediction(model: str, coin: str, signal: str, confidence: float, reasons: list = None) -> None:
    """Log individual model predictions."""
    reason_str = ", ".join(reasons[:3]) if reasons else "N/A"
    logger.debug(
        f"🔮 {model.upper()} | {coin} | Signal: {signal} | "
        f"Confidence: {confidence:.1%} | Reasons: {reason_str}"
    )


# NOTE: setup_logging() is called once on startup by src/utils/__init__.py
# which reads LOG_LEVEL from settings. Do NOT call it here — loguru adds
# new sinks on every call, doubling all log output.
