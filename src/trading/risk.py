"""
Risk Management Module

Implements all safety controls and risk limits:
- Maximum positions
- Portfolio risk limits
- Daily loss limits
- Per-trade risk management
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 4)[0])

from config import settings
from src.utils import logger, log_risk_event
from src.data import db, Trade


@dataclass
class RiskCheck:
    """Result of a risk check."""
    can_trade: bool
    reason: str = ""
    details: Dict = None


class RiskManager:
    """
    Risk management for the trading bot.
    
    Controls:
    - Max 5 concurrent positions
    - Max 10% portfolio at risk
    - Daily -8% loss limit stops trading
    - Per-coin single position limit
    - Position sizing based on confidence
    """
    
    def __init__(self):
        """Initialize risk manager."""
        self.daily_start_capital = settings.trading.initial_capital
        self.current_capital = settings.trading.initial_capital
        self.daily_pnl = 0.0
        self.trading_paused = False
        self.pause_reason = ""
        self.pause_until = None
        self.consecutive_losses = 0
        
        # Load actual state from database
        self._sync_with_database()
    
    def _sync_with_database(self) -> None:
        """Sync state with database."""
        try:
            # Get today's stats
            today_stats = db.get_daily_stats()
            self.daily_pnl = today_stats.get("pnl", 0)
            
            # Get open positions value
            open_trades = db.get_open_trades()
            self._open_positions = len(open_trades)
            self._open_coins = {t.coin for t in open_trades}
            self._total_risk = sum(t.position_size for t in open_trades)
            
        except Exception as e:
            logger.warning(f"Could not sync with database: {e}")
            self._open_positions = 0
            self._open_coins = set()
            self._total_risk = 0
    
    def set_capital(self, capital: float) -> None:
        """
        Set current capital (for paper trading with custom balance).
        Only updates daily_start_capital if it hasn't been set yet today.
        
        Args:
            capital: Current capital amount
        """
        # Always update current capital
        self.current_capital = capital
        
        # Only set daily_start_capital if it's still the default value
        # This preserves the original balance for the day
        if self.daily_start_capital == settings.trading.initial_capital:
            self.daily_start_capital = capital
            logger.info(f"Risk manager daily start capital set to ${capital:.2f}")
        else:
            logger.debug(f"Risk manager current capital updated to ${capital:.2f}")
    
    def can_open_position(
        self,
        coin: str,
        position_size: float,
        current_price: float
    ) -> RiskCheck:
        """
        Check if a new position can be opened.
        
        Args:
            coin: Trading pair
            position_size: Proposed position size in USDT
            current_price: Current price
        
        Returns:
            RiskCheck with result and reason
        """
        self._sync_with_database()
        
        # Check if trading is paused
        if self.trading_paused:
            if self.pause_until and datetime.utcnow() < self.pause_until:
                return RiskCheck(False, f"Trading paused: {self.pause_reason}")
            else:
                self.trading_paused = False
                self.pause_reason = ""
        
        # Check 1: Daily loss limit
        daily_pnl_pct = self.daily_pnl / self.daily_start_capital
        if daily_pnl_pct <= -settings.trading.daily_loss_limit:
            self._pause_trading("Daily loss limit reached", hours=24)
            return RiskCheck(False, f"Daily loss limit (-8%) reached: {daily_pnl_pct:.1%}")
        
        # Check 2: Maximum positions (adaptive for low balance)
        max_positions = self._get_max_positions_for_balance()
        if self._open_positions >= max_positions:
            return RiskCheck(False, f"Max positions ({max_positions}) reached for current balance")
        
        # Check 3: Duplicate coin
        if coin in self._open_coins:
            return RiskCheck(False, f"Already have position in {coin}")
        
        # Check 4: Portfolio risk limit (adaptive for low balance)
        max_portfolio_risk = self._get_max_portfolio_risk()
        new_total_risk = self._total_risk + position_size
        risk_pct = new_total_risk / self.current_capital
        if risk_pct > max_portfolio_risk:
            return RiskCheck(
                False, 
                f"Would exceed max portfolio risk ({max_portfolio_risk:.0%})"
            )
        
        # Check 5: Minimum position size (Binance minimum is $6 USDT)
        min_size = max(6.0, self.current_capital * 0.01)  # $6 minimum or 1% of capital
        if position_size < min_size:
            return RiskCheck(False, f"Position size ${position_size:.2f} below minimum ${min_size:.2f}")
        
        # Check 6: Consecutive losses reduction
        if self.consecutive_losses >= 5:
            # Reduce position size by 50%
            log_risk_event("Consecutive Losses", f"{self.consecutive_losses} losses, reducing size")
            # Still allow trade but caller should reduce size
            return RiskCheck(
                True, 
                "Trade allowed with reduced size",
                {"reduce_size": True, "factor": 0.5}
            )
        
        return RiskCheck(True, "All risk checks passed")
    
    def _get_max_positions_for_balance(self) -> int:
        """
        Get maximum positions based on current balance.
        Low balance accounts get fewer max positions.
        Uses daily_start_capital to avoid reducing limits after trades.
        
        Returns:
            Maximum number of concurrent positions
        """
        # Use daily_start_capital so limits don't decrease after opening trades
        capital = self.daily_start_capital
        
        if capital < 100:
            return 1  # Only 1 position for very low balance
        elif capital < 200:
            return 2  # 2 positions for $100-200
        elif capital < 300:
            return 3  # 3 positions for $200-300
        else:
            return settings.trading.max_positions  # Full 5 positions for $300+
    
    def _get_max_portfolio_risk(self) -> float:
        """
        Get maximum portfolio risk based on current balance.
        Low balance accounts get higher risk % to allow multiple small trades.
        Uses daily_start_capital to avoid changing limits mid-day.
        
        Returns:
            Maximum portfolio risk as decimal (e.g., 0.30 = 30%)
        """
        # Use daily_start_capital so limits don't change after trades
        capital = self.daily_start_capital
        
        if capital < 200:
            return 0.30  # 30% for very low balance (<$200)
        elif capital < 300:
            return 0.20  # 20% for $200-300
        else:
            return settings.trading.max_portfolio_risk  # 10% for $300+
    
    def calculate_position_size(
        self,
        tier: int,
        confidence: float,
        coin: str
    ) -> float:
        """
        Calculate position size based on tier and confidence.
        Adaptive for low balance accounts.
        
        Args:
            tier: Consensus tier (1-4)
            confidence: Model confidence
            coin: Trading pair
        
        Returns:
            Position size in USDT
        """
        # For low balance (<$300), use fixed $6 minimum per trade
        if self.current_capital < 300:
            # Fixed $6 per trade for low balance accounts
            position_size = 6.0
            
            # Tier adjustments for low balance
            tier_adjustments = {
                1: 1.0,    # $6
                2: 1.17,   # $7
                3: 1.33,   # $8
                4: 0.83,   # $5 (but will be raised to $6 minimum)
            }
            
            position_size *= tier_adjustments.get(tier, 1.0)
            
            # Ensure minimum $6
            position_size = max(6.0, position_size)
            
            # Reduce for consecutive losses
            if self.consecutive_losses >= 3:
                position_size = 6.0  # Back to minimum
            
            return round(position_size, 2)
        
        # Standard calculation for normal balance ($300+)
        base_risk = settings.trading.risk_per_trade
        
        # Tier multipliers
        multipliers = {
            1: 1.0,    # 2%
            2: 1.25,   # 2.5%
            3: 1.75,   # 3.5%
            4: 0.75,   # 1.5%
        }
        
        multiplier = multipliers.get(tier, 1.0)
        
        # Adjust for consecutive losses
        if self.consecutive_losses >= 5:
            multiplier *= 0.5
        elif self.consecutive_losses >= 3:
            multiplier *= 0.75
        
        # Calculate size
        position_size = self.current_capital * base_risk * multiplier
        
        # Apply confidence adjustment (reduce size if confidence is borderline)
        if confidence < 0.65:
            position_size *= 0.8
        elif confidence > 0.85:
            position_size *= 1.1
        
        # Ensure minimum $6
        position_size = max(6.0, position_size)
        
        return round(position_size, 2)
    
    def record_trade_result(self, pnl_usd: float, pnl_pct: float) -> None:
        """
        Record a trade result and update risk state.
        
        Args:
            pnl_usd: Profit/loss in USD
            pnl_pct: Profit/loss percentage
        """
        self.daily_pnl += pnl_usd
        self.current_capital += pnl_usd
        
        if pnl_pct < 0:
            self.consecutive_losses += 1
            
            if self.consecutive_losses >= 5:
                log_risk_event(
                    "Consecutive Losses",
                    f"{self.consecutive_losses} consecutive losses, reducing position sizes"
                )
        else:
            self.consecutive_losses = 0
        
        # Check if daily loss limit hit
        daily_pnl_pct = self.daily_pnl / self.daily_start_capital
        if daily_pnl_pct <= -settings.trading.daily_loss_limit:
            self._pause_trading("Daily loss limit reached", hours=24)
    
    def _pause_trading(self, reason: str, hours: int = 24) -> None:
        """Pause trading for specified duration."""
        self.trading_paused = True
        self.pause_reason = reason
        self.pause_until = datetime.utcnow() + timedelta(hours=hours)
        log_risk_event("Trading Paused", f"{reason} for {hours}h")
    
    def get_stop_loss_price(self, entry_price: float, side: str = "long") -> float:
        """Calculate stop loss price."""
        stop_pct = settings.trading.stop_loss_percent
        if side == "long":
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)
    
    def get_take_profit_price(
        self, 
        entry_price: float, 
        tier: int, 
        side: str = "long"
    ) -> float:
        """Calculate take profit price (TP2 - Fib 2.618 equivalent)."""
        tp_pct = 0.07  # +7.0% for TP2
        
        if side == "long":
            return entry_price * (1 + tp_pct)
        else:
            return entry_price * (1 - tp_pct)
    
    def get_partial_exit_price(self, entry_price: float, side: str = "long") -> float:
        """Calculate partial exit price TP1 (Fib 1.618 equivalent)."""
        partial_pct = 0.035  # +3.5% for TP1
        if side == "long":
            return entry_price * (1 + partial_pct)
        else:
            return entry_price * (1 - partial_pct)
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        side: str = "long"
    ) -> Tuple[bool, float]:
        """
        Calculate trailing stop.
        
        Returns:
            Tuple of (should_activate, stop_price)
        """
        # Check if trailing stop should activate (+5% profit)
        pnl_pct = (current_price - entry_price) / entry_price
        
        if pnl_pct < 0.02:  # Lowered activation from +5% to +2%
            return (False, 0)
        
        # Calculate trailing stop from highest price
        trail_pct = 0.02 # Trail distance fixed to 2% to protect profits
        if side == "long":
            stop_price = highest_price * (1 - trail_pct)
        else:
            stop_price = highest_price * (1 + trail_pct)
        
        return (True, stop_price)
    
    def should_time_stop(self, entry_time: datetime) -> bool:
        """Check if position should be closed due to time stop."""
        max_hours = settings.trading.time_stop_hours
        elapsed = datetime.utcnow() - entry_time
        return elapsed.total_seconds() / 3600 > max_hours
    
    def get_status(self) -> Dict:
        """Get current risk status."""
        self._sync_with_database()
        
        return {
            "current_capital": self.current_capital,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.daily_start_capital,
            "open_positions": self._open_positions,
            "open_coins": list(self._open_coins),
            "total_risk": self._total_risk,
            "risk_pct": self._total_risk / self.current_capital,
            "trading_paused": self.trading_paused,
            "pause_reason": self.pause_reason,
            "consecutive_losses": self.consecutive_losses
        }
    
    def reset_daily(self) -> None:
        """Reset daily counters (call at midnight)."""
        self.daily_start_capital = self.current_capital
        self.daily_pnl = 0.0
        self.trading_paused = False
        self.pause_reason = ""
        logger.info("Daily risk counters reset")


# Global risk manager instance
risk_manager = RiskManager()
