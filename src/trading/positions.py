"""
Position Manager

Tracks and manages open positions:
- Entry/exit tracking
- Stop loss monitoring
- Take profit monitoring
- Trailing stop updates
- Partial exits
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import uuid

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 4)[0])

from config import settings
from src.utils.logger import log_trade, log_position_update, logger
from src.utils.notifiers import discord_notifier
from src.data.database import db, Trade, TradeStatus, ExitReason
from src.trading.risk import risk_manager


@dataclass
class Position:
    """Active trading position."""
    trade_id: str
    coin: str
    entry_price: float
    position_size: float  # in USDT
    position_coins: float  # in coin units
    entry_time: datetime
    stop_loss: float
    take_profit: float
    tier: int
    
    # Tracking
    highest_price: float = 0
    trailing_stop: Optional[float] = None
    partial_exit_done: bool = False
    remaining_size: float = 0
    
    # Model confidences at entry
    ta_confidence: float = 0
    ml_confidence: float = 0
    tcn_confidence: float = 0
    entry_reason: str = ""


class PositionManager:
    """
    Manages all open positions.
    
    Responsibilities:
    - Track all active positions
    - Monitor stop losses
    - Monitor take profits
    - Update and trigger trailing stops
    - Handle partial exits
    - Check time-based exits
    """
    
    def __init__(self):
        """Initialize position manager."""
        self.positions: Dict[str, Position] = {}
        self._load_open_positions()
    
    def _load_open_positions(self) -> None:
        """Load open positions from database with deduplication by coin."""
        try:
            open_trades = db.get_open_trades()
            
            # Deduplicate: keep only the latest trade per coin
            coin_to_trade = {}
            for trade in open_trades:
                if trade.coin not in coin_to_trade:
                    coin_to_trade[trade.coin] = trade
                else:
                    existing = coin_to_trade[trade.coin]
                    if trade.entry_time > existing.entry_time:
                        logger.warning(f"Duplicate position found for {trade.coin}, cancelling older trade {existing.trade_id}")
                        db.update_trade(existing.trade_id, status=TradeStatus.CANCELLED)
                        coin_to_trade[trade.coin] = trade
                    else:
                        logger.warning(f"Duplicate position found for {trade.coin}, cancelling older trade {trade.trade_id}")
                        db.update_trade(trade.trade_id, status=TradeStatus.CANCELLED)
            
            for trade in coin_to_trade.values():
                self.positions[trade.trade_id] = Position(
                    trade_id=trade.trade_id,
                    coin=trade.coin,
                    entry_price=trade.entry_price,
                    position_size=trade.position_size,
                    position_coins=trade.position_size_coins or 0,
                    entry_time=trade.entry_time,
                    stop_loss=trade.stop_loss_price,
                    take_profit=trade.take_profit_price,
                    tier=trade.consensus_tier or 1,
                    highest_price=trade.highest_price or trade.entry_price,
                    trailing_stop=trade.trailing_stop_price,
                    partial_exit_done=bool(trade.partial_exit_done),
                    remaining_size=trade.remaining_size or trade.position_size,
                    ta_confidence=trade.ta_confidence or 0,
                    ml_confidence=trade.ml_confidence or 0,
                    tcn_confidence=trade.tcn_confidence or 0,
                    entry_reason=trade.entry_reason or ""
                )
            logger.info(f"Loaded {len(self.positions)} open positions")
            
            # Migrate stale TP values: if TP2 <= TP1, recalculate using current formula
            # This fixes positions opened before the executor.py TA-key_levels bug was patched.
            for trade_id, pos in self.positions.items():
                tp1 = risk_manager.get_partial_exit_price(pos.entry_price)   # +3.5%
                tp2_correct = risk_manager.get_take_profit_price(pos.entry_price, pos.tier)  # +7.0%
                if pos.take_profit <= tp1:
                    pos.take_profit = tp2_correct
                    db.update_trade(trade_id, take_profit_price=tp2_correct)
                    logger.warning(
                        f"[{pos.coin}] Migrated stale TP2 from ${pos.take_profit:.4f} → "
                        f"${tp2_correct:.4f} (+7%). TP1=${tp1:.4f}, TP2=${tp2_correct:.4f}"
                    )
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    def open_position(
        self,
        coin: str,
        entry_price: float,
        position_size: float,
        tier: int,
        ta_confidence: float = 0,
        ml_confidence: float = 0,
        tcn_confidence: float = 0,
        entry_reason: str = ""
    ) -> Optional[Position]:
        """
        Open a new position.
        
        Args:
            coin: Trading pair
            entry_price: Entry price
            position_size: Position size in USDT
            tier: Consensus tier
            ta_confidence: TA model confidence
            ml_confidence: ML model confidence
            tcn_confidence: TCN model confidence
            entry_reason: Reason for entry
        
        Returns:
            Position object if successful
        """
        trade_id = str(uuid.uuid4())[:12]
        position_coins = position_size / entry_price
        
        # Calculate levels
        stop_loss = risk_manager.get_stop_loss_price(entry_price)
        take_profit = risk_manager.get_take_profit_price(entry_price, tier)
        
        position = Position(
            trade_id=trade_id,
            coin=coin,
            entry_price=entry_price,
            position_size=position_size,
            position_coins=position_coins,
            entry_time=datetime.utcnow(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            tier=tier,
            highest_price=entry_price,
            remaining_size=position_size,
            ta_confidence=ta_confidence,
            ml_confidence=ml_confidence,
            tcn_confidence=tcn_confidence,
            entry_reason=entry_reason
        )
        
        # Save to database
        try:
            trade = Trade(
                trade_id=trade_id,
                coin=coin,
                entry_time=position.entry_time,
                entry_price=entry_price,
                position_size=position_size,
                position_size_coins=position_coins,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                highest_price=entry_price,
                remaining_size=position_size,
                consensus_tier=tier,
                ta_confidence=ta_confidence,
                ml_confidence=ml_confidence,
                tcn_confidence=tcn_confidence,
                entry_reason=entry_reason,
                status=TradeStatus.OPEN
            )
            db.save_trade(trade)
            
            self.positions[trade_id] = position
            
            log_trade(
                "BUY", coin, entry_price, position_size,
                entry_reason, tier=tier
            )
            
            logger.info(
                f"Opened position: {coin} @ ${entry_price:.4f} | "
                f"Size: ${position_size:.2f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}"
            )
            
            return position
            
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            return None
    
    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: ExitReason,
        partial: bool = False
    ) -> Optional[Dict]:
        """
        Close a position (fully or partially).
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for exit
            partial: Whether this is a partial exit
        
        Returns:
            Dictionary with trade results
        """
        if trade_id not in self.positions:
            logger.warning(f"Position {trade_id} not found")
            return None
        
        position = self.positions[trade_id]
        
        if partial and not position.partial_exit_done:
            # Partial exit: sell 50% at TP1
            exit_size = position.remaining_size * 0.50
            position.remaining_size -= exit_size
            position.partial_exit_done = True
            
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
            pnl_usd = exit_size * pnl_pct
            
            # Update database
            db.update_trade(
                trade_id,
                partial_exit_done=1,
                remaining_size=position.remaining_size
            )
            
            log_trade(
                "PARTIAL_SELL", position.coin, exit_price, exit_size,
                exit_reason, pnl=pnl_pct * 100
            )
            
            # Discord TP1 partial exit alert
            discord_notifier.send_partial_exit_alert(
                symbol=position.coin,
                price=exit_price,
                pnl_pct=pnl_pct * 100,
                pnl_usd=pnl_usd,
                remaining_size=position.remaining_size
            )
            
            return {
                "trade_id": trade_id,
                "type": "partial",
                "exit_size": exit_size,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "remaining": position.remaining_size
            }
        else:
            # Full exit
            exit_size = position.remaining_size
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
            pnl_usd = exit_size * pnl_pct
            
            # Update database
            db.update_trade(
                trade_id,
                exit_time=datetime.utcnow(),
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl_percent=pnl_pct * 100,
                pnl_usd=pnl_usd,
                status=TradeStatus.CLOSED
            )
            
            # Record outcome for performance tracking
            from src.models.ensemble import ensemble
            outcome = "WIN" if pnl_usd > 0 else "LOSS"
            ensemble.record_trade_outcome(position.coin, outcome, pnl_usd)
            
            # Build duration string for Discord
            duration = datetime.utcnow() - position.entry_time
            total_secs = int(duration.total_seconds())
            hours, remainder = divmod(total_secs, 3600)
            mins = remainder // 60
            duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
            
            # Send Discord Alert with full trade summary
            discord_notifier.send_sell_alert(
                symbol=position.coin,
                price=exit_price,
                pnl_pct=pnl_pct * 100,
                pnl_usd=pnl_usd,
                reason=exit_reason,
                entry_price=position.entry_price,
                duration_str=duration_str
            )
            
            # Backfill prediction outcomes in DB (makes model accuracy stats real)
            try:
                outcome = "WIN" if pnl_usd > 0 else "LOSS"
                db.backfill_prediction_outcomes(
                    symbol=position.coin,
                    trade_id=trade_id,
                    outcome=outcome,
                    pnl_pct=pnl_pct * 100,
                    entry_time=position.entry_time
                )
            except Exception:
                pass  # Never let this block trade close logic
            
            # Update risk manager
            risk_manager.record_trade_result(pnl_usd, pnl_pct)
            
            # Remove from active positions
            del self.positions[trade_id]
            
            log_trade(
                "SELL", position.coin, exit_price, exit_size,
                exit_reason, pnl=pnl_pct * 100
            )
            
            return {
                "trade_id": trade_id,
                "type": "full",
                "coin": position.coin,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "exit_size": exit_size,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "duration": str(datetime.utcnow() - position.entry_time)
            }
    
    def check_position(self, trade_id: str, current_price: float) -> Optional[Tuple[str, str]]:
        """
        Check a position for exit conditions.
        
        Args:
            trade_id: Trade ID
            current_price: Current price
        
        Returns:
            Tuple of (action, reason) or None if no action needed
        """
        if trade_id not in self.positions:
            return None
        
        position = self.positions[trade_id]
        
        # Update highest price
        if current_price > position.highest_price:
            position.highest_price = current_price
            db.update_trade(trade_id, highest_price=current_price)
        
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # Check stop loss
        if current_price <= position.stop_loss:
            return ("stop_loss", f"Stop loss hit at ${current_price:.4f}")
        
        # ✅ CHECK TP1 FIRST (partial exit + breakeven SL), THEN TP2 (full exit)
        # This is critical: if TP2 is checked first and its value was ever incorrect
        # (Bug 1), the full position closes before TP1 logic runs.
        
        # Check partial exit (TP1 = +3.5%)
        partial_target = risk_manager.get_partial_exit_price(position.entry_price)
        if current_price >= partial_target and not position.partial_exit_done:
            # Move SL to breakeven on TP1 hit
            position.stop_loss = position.entry_price
            db.update_trade(trade_id, stop_loss_price=position.entry_price)
            logger.info(f"[{trade_id}] TP1 Hit @ ${current_price:.4f}! SL moved to Breakeven (${position.entry_price:.4f})")
            return ("partial_exit", f"TP1 hit at ${current_price:.4f}. SL moved to Entry.")
        
        # Check full take profit (TP2 = +7.0%)
        if current_price >= position.take_profit:
            return ("take_profit", f"Take profit TP2 hit at ${current_price:.4f}")
        
        # Check trailing stop
        should_trail, trail_price = risk_manager.calculate_trailing_stop(
            position.entry_price,
            current_price,
            position.highest_price
        )
        
        if should_trail:
            position.trailing_stop = trail_price
            db.update_trade(trade_id, trailing_stop_price=trail_price)
            
            if current_price <= trail_price:
                return ("trailing_stop", f"Trailing stop hit at ${current_price:.4f}")
        
        # Check time stop — uses TIME_STOP_HOURS from settings (.env)
        if risk_manager.should_time_stop(position.entry_time):
            if pnl_pct < 0.015:  # Cut if profit is less than 1.5% after time limit
                return ("time_stop", f"Time stop after {settings.trading.time_stop_hours}h")
        
        # Log position status — throttled to once every 5 minutes to avoid spam
        # (monitor_positions runs every 30s for SL/TP checks, but we don't need to log that often)
        now = datetime.utcnow()
        last_log_attr = f"_last_logged_{trade_id}"
        last_log = getattr(self, last_log_attr, None)
        if last_log is None or (now - last_log).total_seconds() >= 300:
            setattr(self, last_log_attr, now)
            log_position_update(
                position.coin,
                position.entry_price,
                current_price,
                pnl_pct * 100,
                "holding",
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                take_profit_1=partial_target
            )
        
        return None
    
    def check_all_positions(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Check all positions against current prices.
        
        Args:
            prices: Dictionary of coin -> current price
        
        Returns:
            List of exit actions triggered
        """
        exits = []
        
        for trade_id, position in list(self.positions.items()):
            current_price = prices.get(position.coin)
            if not current_price:
                continue
            
            try:
                action = self.check_position(trade_id, current_price)
                if action:
                    action_type, reason = action
                    
                    if action_type == "partial_exit":
                        result = self.close_position(trade_id, current_price, reason, partial=True)
                    else:
                        result = self.close_position(trade_id, current_price, reason)
                    
                    if result:
                        exits.append(result)
            except Exception as e:
                logger.error(f"Error checking position {trade_id} ({position.coin}): {e}")
        
        return exits
    
    def get_open_positions(self) -> List[Dict]:
        """Get list of all open positions."""
        return [
            {
                "trade_id": p.trade_id,
                "coin": p.coin,
                "entry_price": p.entry_price,
                "position_size": p.position_size,
                "remaining_size": p.remaining_size,
                "entry_time": p.entry_time.isoformat(),
                "stop_loss": p.stop_loss,
                "take_profit_1": risk_manager.get_partial_exit_price(p.entry_price),  # TP1 (+3.5%)
                "take_profit": p.take_profit,  # TP2 (+7.0%)
                "trailing_stop": p.trailing_stop,
                "tier": p.tier,
                "partial_exit_done": p.partial_exit_done
            }
            for p in self.positions.values()
        ]
    
    def has_position(self, coin: str) -> bool:
        """Check if there's an open position for a coin."""
        return any(p.coin == coin for p in self.positions.values())
    
    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)


# Global position manager instance
position_manager = PositionManager()
