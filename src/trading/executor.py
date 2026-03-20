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

    
    def execute_signal(self, symbol: str, signal: ConsensusSignal) -> Optional[OrderResult]:
        """
        Execute a trading signal.
        
        Args:
            symbol: Trading pair
            signal: Consensus signal from ensemble
        
        Returns:
            OrderResult if trade executed
        """
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
        
        # Hard safety cap: never risk more than 10% of portfolio in a single trade
        capital = self.paper_balance if self.mode == ExecutionMode.PAPER else risk_manager.current_capital
        max_position_cap = capital * settings.trading.max_portfolio_risk
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
        """Execute paper buy order."""
        # Calculate available balance (total - positions at risk)
        from src.trading.risk import risk_manager
        risk_manager._sync_with_database()
        available_balance = self.paper_balance - risk_manager._total_risk
        
        if size > available_balance:
            return OrderResult(
                False, 
                message=f"Insufficient available balance (${available_balance:.2f} available, ${size:.2f} needed)"
            )
        
        # Simulate small slippage (0.1%)
        fill_price = price * 1.001
        
        self.paper_balance -= size
        
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
        """Execute paper sell order."""
        # Simulate small slippage (0.1%)
        fill_price = price * 0.999
        
        self.paper_balance += size * (fill_price / price)
        
        order_id = f"PAPER_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{symbol[:4]}_SELL"
        
        logger.info(
            f"📝 PAPER SELL | {symbol} @ ${fill_price:.4f} | "
            f"Size: ${size:.2f} | Balance: ${self.paper_balance:.2f}"
        )
        
        return OrderResult(
            success=True,
            order_id=order_id,
            filled_price=fill_price,
            filled_size=size,
            message="Paper sell executed"
        )
    
    def _live_buy(
        self,
        symbol: str,
        price: float,
        size: float
    ) -> OrderResult:
        """Execute live buy order on Binance."""
        try:
            from binance.client import Client
            
            client = Client(
                settings.exchange.binance_api_key,
                settings.exchange.binance_secret_key,
                testnet=settings.exchange.use_testnet,
                requests_params={'timeout': 10}
            )
            
            # Calculate quantity with proper precision
            try:
                # Get symbol info for precision
                symbol_info = client.get_symbol_info(symbol)
                precision = 6  # Default
                
                if symbol_info:
                    # Get LOT_SIZE filter for step size
                    lot_size = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    if lot_size:
                        step_size = float(lot_size['stepSize'])
                        # Calculate precision from step size
                        precision = len(str(step_size).rstrip('0').split('.')[-1]) if '.' in str(step_size) else 0
                
                quantity = size / price
                quantity = round(quantity, precision)
            except Exception as e:
                logger.warning(f"Could not get precision for {symbol}, using default: {e}")
                quantity = round(size / price, 6)
            
            # Place market order
            order = client.create_order(
                symbol=symbol,
                side="BUY",
                type="MARKET",
                quoteOrderQty=size  # Buy with USDT amount
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
        
        Returns:
            Dictionary with monitoring results
        """
        # Get current prices for all positions
        positions = position_manager.get_open_positions()
        
        if not positions:
            return {"monitored": 0, "exits": []}
        
        coins = [p["coin"] for p in positions]
        prices = collector.get_all_prices(coins)
        
        # Check all positions
        exits = position_manager.check_all_positions(prices)
        
        return {
            "monitored": len(positions),
            "exits": exits
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
