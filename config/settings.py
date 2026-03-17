"""
Configuration Settings for Crypto Trading Bot

All configurable parameters are loaded from environment variables.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def get_env_float(key: str, default: float) -> float:
    """Get environment variable as float."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_int(key: str, default: int) -> int:
    """Get environment variable as int."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Get environment variable as bool."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


@dataclass
class TradingSettings:
    """Trading-related configuration."""
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.02
    max_positions: int = 5
    max_portfolio_risk: float = 0.10
    daily_loss_limit: float = 0.08
    stop_loss_percent: float = 0.03
    take_profit_standard: float = 0.06
    take_profit_high_conviction: float = 0.08
    trailing_stop_activation: float = 0.05
    trailing_stop_distance: float = 0.02
    time_stop_hours: int = 6
    partial_exit_percent: float = 0.40
    partial_exit_target: float = 0.05


@dataclass
class ModelSettings:
    """Model confidence thresholds."""
    tier1_confidence: float = 0.85
    tier2_confidence: float = 0.60
    tier3_confidence: float = 0.60
    tier1_multiplier: float = 1.0
    tier2_multiplier: float = 1.25
    tier3_multiplier: float = 1.75
    tier4_multiplier: float = 0.75


@dataclass
class ExchangeSettings:
    """Exchange API configuration."""
    binance_api_key: str = ""
    binance_secret_key: str = ""
    use_testnet: bool = True
    binance_base_url: str = "https://api.binance.com"
    binance_testnet_url: str = "https://testnet.binance.vision"


@dataclass
class DatabaseSettings:
    """Database configuration."""
    database_url: str = "sqlite:///./data/trading_bot.db"


@dataclass
class NotificationSettings:
    """Telegram and Discord notification settings."""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    
    @property
    def enable_notifications(self) -> bool:
        return bool(
            (self.telegram_bot_token and self.telegram_chat_id) or 
            self.discord_webhook_url
        )


class Settings:
    """Main settings class combining all configurations."""
    
    def __init__(self):
        # Load from environment
        self.trading = TradingSettings(
            initial_capital=get_env_float("INITIAL_CAPITAL", 10000.0),
            risk_per_trade=get_env_float("RISK_PER_TRADE", 0.02),
            max_positions=get_env_int("MAX_POSITIONS", 5),
            max_portfolio_risk=get_env_float("MAX_PORTFOLIO_RISK", 0.10),
            daily_loss_limit=get_env_float("DAILY_LOSS_LIMIT", 0.08),
            stop_loss_percent=get_env_float("STOP_LOSS_PERCENT", 0.03),
            take_profit_standard=get_env_float("TAKE_PROFIT_STANDARD", 0.06),
            take_profit_high_conviction=get_env_float("TAKE_PROFIT_HIGH_CONVICTION", 0.08),
            trailing_stop_activation=get_env_float("TRAILING_STOP_ACTIVATION", 0.05),
            trailing_stop_distance=get_env_float("TRAILING_STOP_DISTANCE", 0.02),
            time_stop_hours=get_env_int("TIME_STOP_HOURS", 6),
            partial_exit_percent=get_env_float("PARTIAL_EXIT_PERCENT", 0.40),
            partial_exit_target=get_env_float("PARTIAL_EXIT_TARGET", 0.05),
        )
        
        self.models = ModelSettings(
            tier1_confidence=get_env_float("TIER1_CONFIDENCE", 0.85),
            tier2_confidence=get_env_float("TIER2_CONFIDENCE", 0.60),
            tier3_confidence=get_env_float("TIER3_CONFIDENCE", 0.60),
        )
        
        self.exchange = ExchangeSettings(
            binance_api_key=get_env("BINANCE_API_KEY", ""),
            binance_secret_key=get_env("BINANCE_SECRET_KEY", ""),
            use_testnet=get_env_bool("USE_TESTNET", True),
        )
        
        self.database = DatabaseSettings(
            database_url=get_env("DATABASE_URL", "sqlite:///./data/trading_bot.db"),
        )
        
        self.notifications = NotificationSettings(
            telegram_bot_token=get_env("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=get_env("TELEGRAM_CHAT_ID", ""),
            discord_webhook_url=get_env("DISCORD_WEBHOOK_URL", ""),
        )
        
        self.log_level = get_env("LOG_LEVEL", "INFO")


# Global settings instance
settings = Settings()


def get_position_size(tier: int, base_capital: float) -> float:
    """
    Calculate position size based on consensus tier.
    
    Args:
        tier: Consensus tier (1-4)
        base_capital: Available capital
    
    Returns:
        Position size in USDT
    """
    base_risk = settings.trading.risk_per_trade
    
    multipliers = {
        1: settings.models.tier1_multiplier,
        2: settings.models.tier2_multiplier,
        3: settings.models.tier3_multiplier,
        4: settings.models.tier4_multiplier,
    }
    
    multiplier = multipliers.get(tier, 1.0)
    return base_capital * base_risk * multiplier


def get_take_profit(tier: int) -> float:
    """Get take profit percentage based on tier."""
    if tier == 3:
        return settings.trading.take_profit_high_conviction
    return settings.trading.take_profit_standard
