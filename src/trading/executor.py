"""
Trading Executor

Executes trades based on model signals:
- Order placement (paper trading / live)
- Pre-trade risk checks
- Order monitoring
"""

from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 4)[0])

from config import settings
from src.utils import logger
from src.data import collector
from src.data.state import save_state, load_state
from src.models import ensemble, ConsensusSignal
from src.trading.risk import risk_manager, RiskCheck
from src.trading.positions import position_manager
from src.utils.notifiers import discord_notifier


class ExecutionMode(str, Enum):
    """Execution mode."""
    PAPER = "paper"
    LIVE = "live"


@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: str = ""
    filled_price: float = 0
    filled_size: float = 0
    message: str = ""
    

class TradingExecutor:
    """
    Executes trading signals.
    
    Supports:
    - Paper trading (simulation)
    - Live trading (Binance)
    - Pre-trade risk checks
    - Position opening/closing
    """
    
    # Base position sizes per tier (must match ConsensusEnsemble.TIER*_SIZE)
    TIER_BASE_SIZE = {1: 0.02, 2: 0.025, 3: 0.035, 4: 0.015}
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.PAPER):
        """Initialize executor."""
        self.mode = mode
        
        # Load saved state for paper trading
        if mode == ExecutionMode.PAPER:
            state = load_state()
            self.paper_balance = state["paper_balance"]
            logger.info(f"Loaded paper balance: ${self.paper_balance:.2f}")
            
            # Sync risk manager with paper balance
            from src.trading.risk import risk_manager
            risk_manager.set_capital(self.paper_balance)
        else:
            self.paper_balance = settings.trading.initial_capital
        
        self.paper_trades: Dict[str, Dict] = {}
        
        logger.info(f"Trading Executor initialized in {mode.value} mode")

    def sync_state(self) -> None:
        """Sync in-memory balance and positions with DB to handle resets/out-of-process changes."""
        if self.mode != ExecutionMode.PAPER:
            return
        try:
            state = load_state()
            if state:
                db_balance = state["paper_balance"]
                if abs(self.paper_balance - db_balance) > 0.01:
                    logger.info(f"🔄 State sync: updating in-memory paper balance from ${self.paper_balance:.2f} to ${db_balance:.2f}")
                    self.paper_balance = db_balance
                    
                    # Reload positions from database
                    position_manager._load_open_positions()
                    
                    # Sync risk manager capital and parameters —
                    # do NOT manually zero daily_pnl or _total_risk here.
                    # _sync_with_database() recomputes them accurately from DB.
                    risk_manager.set_capital(self.paper_balance)
                    risk_manager._sync_with_database()
        except Exception as e:
            logger.warning(f"Failed to sync executor state with DB: {e}")
    
    def execute_signal(self, symbol: str, signal: ConsensusSignal) -> Optional[OrderResult]:
        """
        Execute a trading signal.
        
        Args:
            symbol: Trading pair
            signal: Consensus signal from ensemble
        
        Returns:
            OrderResult if trade executed
        """
        self.sync_state()
        if signal.signal != "buy":
            return None
        
        if signal.tier == 0:
            return None
        
        # Get current price
        current_price = collector.get_current_price(symbol)
        if not current_price:
            return OrderResult(False, message="Could not get current price")
        
        # --- Entry Price ---
        entry_price = current_price
        
        # --- SL / TP: ALWAYS from RiskManager. Never from TA key_levels.
        # TA key_levels are only for signal confirmation, NOT exit management.
        stop_loss = risk_manager.get_stop_loss_price(entry_price)    # -3%
        take_profit = risk_manager.get_take_profit_price(entry_price, signal.tier)  # +7% (TP2)
        # TP1 (+3.5%) is computed dynamically in check_position() — no storage needed here
        
        # --- Position Sizing ---
        position_size = risk_manager.calculate_position_size(
            tier=signal.tier,
            confidence=signal.confidence,
            coin=symbol
        )
        
        # Apply regime position multiplier from the ensemble signal
        # (e.g. BEAR = 0.5x, BULL = 1.2x, SIDEWAYS = 0.8x)
        regime_multiplier = signal.position_size_pct / max(self.TIER_BASE_SIZE.get(signal.tier, 0.02), 0.001)
        if 0.5 <= regime_multiplier <= 1.5:  # sanity bounds — ignore extreme values
            position_size *= regime_multiplier
        
        # Determine capital and the exchange's absolute minimum order value.
        # Binance USDT spot pairs enforce a $5–10 USDT minimum notional;
        # we use $6.00 as the safe floor (with a small buffer over $5).
        capital = self.paper_balance if self.mode == ExecutionMode.PAPER else risk_manager.current_capital
        min_size = max(6.0, capital * 0.01)  # never go below $6
        
        # Low-balance fix 1: Prevent the regime / Kelly multiplier from pushing the
        # calculated size below the exchange minimum.  On a $50–$100 account the BEAR
        # 0.6× dampener would otherwise compress $6 → $3.60, causing every trade to be
        # vetoed.  We simply upscale back to the floor when this happens.
        if position_size < min_size:
            logger.debug(
                f"[LowBalance] {symbol}: calculated size ${position_size:.2f} < floor "
                f"${min_size:.2f} — upscaling to floor."
            )
            position_size = min_size
        
        # Hard safety cap: never risk more than 10% of portfolio in a single trade.
        # Low-balance fix 2: For micro accounts the 10% cap can itself be less than the
        # exchange minimum (e.g. 10% of $50 = $5.00 < $6.00).  We ensure the cap is
        # never smaller than the exchange floor so the position survives Check 5 in
        # risk_manager.can_open_position().
        max_position_cap = max(capital * settings.trading.max_portfolio_risk, min_size)
        if position_size > max_position_cap:
            position_size = max_position_cap
        
        # --- Risk Check ---
        risk_check = risk_manager.can_open_position(symbol, position_size, current_price)
        
        if not risk_check.can_trade:
            logger.warning(f"Risk check failed for {symbol}: {risk_check.reason}")
            return OrderResult(False, message=risk_check.reason)
        
        # Adjust size if consecutive losses require reduction
        if risk_check.details and risk_check.details.get("reduce_size"):
            position_size *= risk_check.details["factor"]
        
        # --- Execute Order ---
        if self.mode == ExecutionMode.PAPER:
            result = self._paper_buy(symbol, current_price, position_size)
        else:
            result = self._live_buy(symbol, current_price, position_size)
        
        if result.success:
            position = position_manager.open_position(
                coin=symbol,
                entry_price=result.filled_price,
                position_size=result.filled_size,
                tier=signal.tier,
                ta_confidence=signal.ta_signal.confidence if signal.ta_signal else 0,
                ml_confidence=signal.ml_signal.confidence if signal.ml_signal else 0,
                tcn_confidence=signal.tcn_signal.confidence if signal.tcn_signal else 0,
                entry_reason="; ".join(signal.reasons[:3])
            )
            
            if position:
                # positions.py already saved the correct SL/TP (from risk_manager).
                # No overwrite needed here. Just store for paper trade tracking.
                
                # Bug 3 fix: populate paper_trades so paper_sell can compute P&L correctly
                if self.mode == ExecutionMode.PAPER:
                    self.paper_trades[position.trade_id] = {
                        "symbol": symbol,
                        "entry_price": result.filled_price,
                        "size": result.filled_size,
                        "entry_time": datetime.utcnow().isoformat()
                    }
                    save_state(self.paper_balance, len(self.paper_trades))
                
                # Discord Alert — use risk_manager's correct levels
                discord_notifier.send_buy_alert(
                    symbol=symbol,
                    price=result.filled_price,
                    size=result.filled_size,
                    sl=position.stop_loss,
                    tp=position.take_profit,
                    tier=signal.tier,
                    confidence=signal.confidence
                )
                
                result.message = f"Position opened: {position.trade_id}"

        
        return result
    
    def _paper_buy(
        self,
        symbol: str,
        price: float,
        size: float
    ) -> OrderResult:
        """Execute paper buy order.

        C-4 FIX: Removed the redundant _sync_with_database() call here.
        can_open_position() already synced moments before this is called.
        We optimistically subtract `size` from the balance calculation so
        concurrent signals for different coins cannot both over-spend.
        """
        # Available = cash minus what's already committed to open positions
        available_balance = self.paper_balance - risk_manager._total_risk

        if size > available_balance:
            return OrderResult(
                False,
                message=f"Insufficient available balance (${available_balance:.2f} available, ${size:.2f} needed)"
            )

        # Simulate small slippage (0.1%)
        fill_price = price * 1.001

        self.paper_balance -= size
        # Immediately reflect this commitment so a concurrent scan cannot
        # re-use the same funds (belt-and-suspenders guard alongside risk_manager).
        risk_manager._total_risk += size

        order_id = f"PAPER_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{symbol[:4]}"

        logger.info(
            f"📝 PAPER BUY | {symbol} @ ${fill_price:.4f} | "
            f"Size: ${size:.2f} | Balance: ${self.paper_balance:.2f}"
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            filled_price=fill_price,
            filled_size=size,
            message="Paper order executed"
        )
    
    def _paper_sell(
        self,
        symbol: str,
        price: float,
        size: float
    ) -> OrderResult:
        """Execute paper sell order.

        `size` is the USDT exit value to credit back to the balance.

        C-2 FIX: Slippage is applied once to the USDT amount.
        Old code: size * (fill_price / price) = size * 0.999 — double-applied
        slippage because fill_price was already price * 0.999.
        """
        # 0.1% slippage on the USDT exit value
        credited = size * 0.999

        self.paper_balance += credited

        order_id = f"PAPER_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{symbol[:4]}_SELL"

        logger.info(
            f"📝 PAPER SELL | {symbol} @ ${price:.4f} | "
            f"Value: ${size:.2f} (credited: ${credited:.2f}) | Balance: ${self.paper_balance:.2f}"
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            filled_price=price,
            filled_size=size,
            message="Paper sell executed"
        )
    
    def _live_buy(
        self,
        symbol: str,
        price: float,
        size: float
    ) -> OrderResult:
        """Execute live buy order on Binance.

        L-4 FIX: Removed dead `quantity = size / price` computation block.
        We use quoteOrderQty=size (USDT amount) so Binance calculates the
        exact coin quantity internally. The precision rounding code was
        computing a value that was never used.
        """
        try:
            from binance.client import Client

            client = Client(
                settings.exchange.binance_api_key,
                settings.exchange.binance_secret_key,
                testnet=settings.exchange.use_testnet,
                requests_params={'timeout': 10}
            )

            # Place market order using USDT quote quantity (Binance converts internally)
            order = client.create_order(
                symbol=symbol,
                side="BUY",
                type="MARKET",
                quoteOrderQty=size
            )

            # Get fill details
            fills = order.get("fills", [])
            if fills:
                avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                total_qty = sum(float(f["qty"]) for f in fills)
                avg_price = avg_price / total_qty if total_qty > 0 else price
            else:
                avg_price = price

            logger.info(
                f"🔴 LIVE BUY | {symbol} @ ${avg_price:.4f} | "
                f"Size: ${size:.2f} | OrderId: {order['orderId']}"
            )

            return OrderResult(
                success=True,
                order_id=str(order["orderId"]),
                filled_price=avg_price,
                filled_size=float(order.get("cummulativeQuoteQty", size)),
                message="Live order executed"
            )

        except Exception as e:
            logger.error(f"Live buy error: {e}")
            return OrderResult(False, message=str(e))
    
    def _live_sell(
        self,
        symbol: str,
        quantity: float
    ) -> OrderResult:
        """Execute live sell order on Binance."""
        try:
            import math
            from binance.client import Client
            
            client = Client(
                settings.exchange.binance_api_key,
                settings.exchange.binance_secret_key,
                testnet=settings.exchange.use_testnet,
                requests_params={'timeout': 10}
            )
            
            # Apply LOT_SIZE step_size restrictions dynamically per altcoin
            try:
                symbol_info = client.get_symbol_info(symbol)
                step_size = None
                
                if symbol_info:
                    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    if lot_size_filter:
                        step_size = float(lot_size_filter['stepSize'])
                
                if step_size:
                    # e.g., if qty=100.1235 and step_size=0.1, output=100.1 (truncate perfectly to step)
                    precision_int = int(round(-math.log10(step_size), 0))
                    # Prevent floating precision issues via Decimal or careful math mapping
                    quantity = math.floor(quantity / step_size) * step_size
                    # Round safely to the decimals of the step_size
                    quantity = round(quantity, max(0, precision_int))
                else:
                    quantity = round(quantity, 6)
            except Exception as e:
                logger.warning(f"Could not enforce proper LOT_SIZE for {symbol}. Will attempt fallback logic: {e}")
                quantity = round(quantity, 6)
            
            # Place market sell
            order = client.create_order(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=quantity
            )
            
            # Get fill details
            fills = order.get("fills", [])
            if fills:
                avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                total_qty = sum(float(f["qty"]) for f in fills)
                avg_price = avg_price / total_qty if total_qty > 0 else 0
            else:
                avg_price = 0
            
            logger.info(
                f"🟢 LIVE SELL | {symbol} @ ${avg_price:.4f} | "
                f"Qty: {quantity} | OrderId: {order['orderId']}"
            )
            
            return OrderResult(
                success=True,
                order_id=str(order["orderId"]),
                filled_price=avg_price,
                filled_size=float(order.get("cummulativeQuoteQty", 0)),
                message="Live sell executed"
            )
            
        except Exception as e:
            logger.error(f"Live sell error: {e}")
            return OrderResult(False, message=str(e))
    
    def scan_and_execute(self, symbols: list = None) -> Dict:
        """
        Scan all coins and execute any qualifying signals.
        
        Args:
            symbols: List of symbols to scan
        
        Returns:
            Dictionary with scan results
        """
        self.sync_state()
        from config import WATCHLIST
        symbols = symbols or WATCHLIST
        
        results = {
            "scanned": 0,
            "signals": 0,
            "executed": 0,
            "skipped": 0,
            "errors": 0,
            "trades": []
        }
        
        for symbol in symbols:
            results["scanned"] += 1
            
            try:
                # Skip if already have position
                if position_manager.has_position(symbol):
                    results["skipped"] += 1
                    continue
                
                # Get consensus signal
                signal = ensemble.analyze(symbol)
                
                if signal.signal == "buy" and signal.tier > 0:
                    results["signals"] += 1
                    
                    # Execute the signal
                    result = self.execute_signal(symbol, signal)
                    
                    if result and result.success:
                        results["executed"] += 1
                        results["trades"].append({
                            "symbol": symbol,
                            "tier": signal.tier,
                            "confidence": signal.confidence,
                            "price": result.filled_price,
                            "size": result.filled_size
                        })
                    else:
                        results["skipped"] += 1
                        
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                results["errors"] += 1
        
        return results
    
    def monitor_positions(self) -> Dict:
        """
        Monitor all open positions and handle exits.

        For each triggered exit (SL / TP1 / TP2 / trailing / time-stop):
          - Paper mode → call _paper_sell  → credit exit_value to paper_balance
          - Live mode  → call _live_sell   → place market sell on Binance

        Returns:
            Dictionary with monitoring results
        """
        self.sync_state()

        # Get current prices for all positions
        positions = position_manager.get_open_positions()

        if not positions:
            return {"monitored": 0, "exits": []}

        coins  = [p["coin"] for p in positions]
        prices = collector.get_all_prices(coins)

        # Check all positions — close_position() now returns exit_value / exit_coins
        exits = position_manager.check_all_positions(prices)

        processed_exits = []
        for exit_info in exits:
            coin        = exit_info.get("coin")
            exit_price  = exit_info.get("exit_price", 0)
            exit_value  = exit_info.get("exit_value", 0)   # USDT to return to balance
            exit_coins  = exit_info.get("exit_coins", 0)   # coin qty for live sell
            trade_id    = exit_info.get("trade_id")
            exit_type   = exit_info.get("type")            # "partial" or "full"

            if not coin or exit_value <= 0:
                # Shouldn't happen; skip rather than corrupt the balance
                logger.warning(f"Skipping exit with missing data: {exit_info}")
                continue

            logger.info(
                f"Executing exit | {coin} ({exit_type}) | "
                f"price=${exit_price:.4f} | value=${exit_value:.2f} | coins={exit_coins:.6f}"
            )

            if self.mode == ExecutionMode.PAPER:
                # _paper_sell adds (size × fill_price/price) to paper_balance.
                # We pass exit_value as `size` and price==fill_price so the
                # full exit_value is credited (slippage is already simulated inside).
                sell_result = self._paper_sell(coin, exit_price, exit_value)

                if sell_result.success:
                    # Update paper_trades tracking
                    if exit_type == "full":
                        if trade_id in self.paper_trades:
                            del self.paper_trades[trade_id]
                    else:
                        # Partial: halve the tracked size
                        if trade_id in self.paper_trades:
                            self.paper_trades[trade_id]["size"] *= 0.50

                    # Persist new balance to DB + JSON so dashboard shows it immediately
                    save_state(self.paper_balance, len(self.paper_trades))

                    # Keep risk_manager capital in sync
                    risk_manager.set_capital(self.paper_balance)

                    processed_exits.append(exit_info)
                    logger.info(
                        f"Paper balance after {exit_type} exit of {coin}: "
                        f"${self.paper_balance:.2f}"
                    )
                else:
                    logger.error(
                        f"Paper sell failed for {coin} ({exit_type}): {sell_result.message}"
                    )

            else:
                # Live mode — place market sell on Binance
                sell_result = self._live_sell(coin, exit_coins)

                if sell_result.success:
                    processed_exits.append(exit_info)
                    logger.info(
                        f"Live sell executed for {coin} ({exit_type}): "
                        f"qty={exit_coins:.6f}, orderId={sell_result.order_id}"
                    )
                else:
                    logger.error(
                        f"Live sell FAILED for {coin} ({exit_type}): {sell_result.message}"
                    )

        return {
            "monitored": len(positions),
            "exits":     processed_exits
        }
    
    def get_status(self) -> Dict:
        """Get executor status."""
        return {
            "mode": self.mode.value,
            "paper_balance": self.paper_balance if self.mode == ExecutionMode.PAPER else None,
            "open_positions": position_manager.get_position_count(),
            "risk_status": risk_manager.get_status()
        }


# Global executor instance (defaulting to paper trading)
executor = TradingExecutor(ExecutionMode.PAPER)
