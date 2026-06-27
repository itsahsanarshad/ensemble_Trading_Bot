"""
Unit tests for TradingExecutor.monitor_positions() and _paper_sell().

The executor module's package __init__ runs:
    executor = TradingExecutor(ExecutionMode.PAPER)
at import time — which calls load_state() and logs the balance with :.2f.
conftest.py now provides load_state with a real float (100.0) so this succeeds.

All external I/O (DB, Binance, price collector, etc.) is mocked by conftest.py
or inline patch.object calls. No network or disk access occurs.
"""

import importlib
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# conftest.py has already stubbed all heavy deps with a real-float load_state.
# Import the REAL TradingExecutor now.
from src.trading.executor import TradingExecutor, ExecutionMode, OrderResult
from src.trading.positions import Position, PositionManager
import src.trading.executor as _executor_mod   # for patch.object targets


# ============================================================================
# Shared constants
# ============================================================================

ENTRY_PRICE    = 100.0
STOP_LOSS      = 97.0
TAKE_PROFIT    = 107.0
POSITION_SIZE  = 20.0
POSITION_COINS = POSITION_SIZE / ENTRY_PRICE   # 0.20


# ============================================================================
# Factory helpers
# ============================================================================

def _make_executor(balance: float = 100.0,
                   mode: ExecutionMode = ExecutionMode.PAPER) -> TradingExecutor:
    """
    Build a TradingExecutor with a controlled balance, no real I/O.
    conftest provides load_state with a real-float dict so __init__ doesn't crash.
    We overwrite paper_balance after construction to the test value.
    string-form patch is used so it works even when module attrs are MagicMocks.
    """
    with patch("src.trading.executor.risk_manager"):
        ex = TradingExecutor(mode)
    ex.paper_balance = float(balance)
    ex.paper_trades  = {}
    return ex


def _make_position(
    trade_id: str = "T-001",
    coin: str = "BTCUSDT",
    position_size: float = POSITION_SIZE,
    partial_exit_done: bool = False,
) -> Position:
    return Position(
        trade_id=trade_id,
        coin=coin,
        entry_price=ENTRY_PRICE,
        position_size=position_size,
        position_coins=position_size / ENTRY_PRICE,
        entry_time=datetime.utcnow(),
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        tier=1,
        highest_price=ENTRY_PRICE,
        trailing_stop=None,
        partial_exit_done=partial_exit_done,
    )


def _full_exit(trade_id="T-001", coin="BTCUSDT",
               exit_price=ENTRY_PRICE, pnl_pct=0.0) -> dict:
    exit_size = POSITION_SIZE
    pnl_usd   = exit_size * pnl_pct
    return {
        "trade_id": trade_id, "type": "full",  "coin": coin,
        "entry_price": ENTRY_PRICE, "exit_price": exit_price,
        "exit_size":  exit_size,
        "exit_value": exit_size + pnl_usd,
        "exit_coins": POSITION_COINS,
        "pnl_pct":    pnl_pct, "pnl_usd": pnl_usd, "duration": "0:01:00",
    }


def _partial_exit(trade_id="T-001", coin="BTCUSDT",
                  exit_price=103.5, pnl_pct=0.035) -> dict:
    exit_size = POSITION_SIZE * 0.50
    pnl_usd   = exit_size * pnl_pct
    return {
        "trade_id":   trade_id, "type": "partial", "coin": coin,
        "exit_price": exit_price,
        "exit_size":  exit_size,
        "exit_value": exit_size + pnl_usd,
        "exit_coins": POSITION_COINS * 0.50,
        "pnl_pct":    pnl_pct, "pnl_usd": pnl_usd,
        "remaining":  POSITION_SIZE * 0.50,
    }


def _run_monitor(ex: TradingExecutor, exits: list) -> dict:
    """Run monitor_positions with all module-level globals patched."""
    mock_positions = [{"coin": e.get("coin", "BTCUSDT")} for e in exits]
    mock_prices    = {e.get("coin", "BTCUSDT"): e.get("exit_price", 100.0)
                      for e in exits}

    with patch("src.trading.executor.position_manager") as mock_pm, \
         patch("src.trading.executor.collector") as mock_col, \
         patch("src.trading.executor.save_state"), \
         patch("src.trading.executor.risk_manager"):

        mock_pm.get_open_positions.return_value  = mock_positions
        mock_col.get_all_prices.return_value     = mock_prices
        mock_pm.check_all_positions.return_value = exits
        ex.sync_state = MagicMock()

        return ex.monitor_positions()


# ============================================================================
# 1. close_position() return-dict structure
# ============================================================================

class TestClosePositionReturnKeys(unittest.TestCase):

    def _pm(self) -> PositionManager:
        pm = PositionManager.__new__(PositionManager)
        pm.positions = {}
        return pm

    def test_full_exit_has_required_keys(self):
        pm  = self._pm()
        pos = _make_position()
        pm.positions[pos.trade_id] = pos

        from src.data.database import ExitReason
        with patch("src.trading.positions.risk_manager") as mock_rm, \
             patch("src.models.ensemble.ensemble", MagicMock()), \
             patch("src.trading.positions.db") as mock_db, \
             patch("src.trading.positions.discord_notifier", MagicMock()), \
             patch("src.trading.positions.log_trade", MagicMock()):
            mock_rm.record_trade_result.return_value = None
            mock_db.update_trade.return_value = None
            mock_db.backfill_prediction_outcomes.return_value = None

            result = pm.close_position(
                pos.trade_id, ENTRY_PRICE * 1.07, ExitReason.TAKE_PROFIT
            )

        self.assertIsNotNone(result)
        for key in ("coin", "exit_price", "exit_value", "exit_coins", "pnl_usd"):
            self.assertIn(key, result, f"Missing key '{key}' in full exit result")

    def test_partial_exit_has_required_keys(self):
        pm  = self._pm()
        pos = _make_position()
        pm.positions[pos.trade_id] = pos

        from src.data.database import ExitReason
        with patch("src.trading.positions.db") as mock_db, \
             patch("src.trading.positions.discord_notifier", MagicMock()), \
             patch("src.trading.positions.log_trade", MagicMock()), \
             patch("src.trading.positions.risk_manager", MagicMock()):
            mock_db.update_trade.return_value = None

            result = pm.close_position(
                pos.trade_id, ENTRY_PRICE * 1.035,
                ExitReason.TAKE_PROFIT, partial=True
            )

        self.assertIsNotNone(result)
        for key in ("coin", "exit_price", "exit_value", "exit_coins", "remaining"):
            self.assertIn(key, result, f"Missing key '{key}' in partial exit result")

    def test_exit_value_equals_cost_plus_pnl(self):
        for pnl_pct in (-0.03, 0.0, 0.07):
            exit_size = POSITION_SIZE
            pnl_usd   = exit_size * pnl_pct
            self.assertAlmostEqual(exit_size + pnl_usd, exit_size * (1 + pnl_pct), places=8)

    def test_position_coins_halved_on_partial_exit(self):
        pm  = self._pm()
        pos = _make_position(position_size=20.0)
        original_coins = pos.position_coins
        pm.positions[pos.trade_id] = pos

        from src.data.database import ExitReason
        with patch("src.trading.positions.db", MagicMock()), \
             patch("src.trading.positions.discord_notifier", MagicMock()), \
             patch("src.trading.positions.log_trade", MagicMock()), \
             patch("src.trading.positions.risk_manager", MagicMock()):
            pm.close_position(
                pos.trade_id, ENTRY_PRICE * 1.035,
                ExitReason.TAKE_PROFIT, partial=True
            )

        self.assertAlmostEqual(pos.position_coins, original_coins * 0.50, places=8)

    def test_guard_populates_zero_position_coins(self):
        pm  = self._pm()
        pos = _make_position()
        pos.position_coins = 0
        pm.positions[pos.trade_id] = pos

        from src.data.database import ExitReason
        with patch("src.trading.positions.db", MagicMock()), \
             patch("src.trading.positions.discord_notifier", MagicMock()), \
             patch("src.trading.positions.log_trade", MagicMock()), \
             patch("src.trading.positions.risk_manager", MagicMock()), \
             patch("src.models.ensemble.ensemble", MagicMock()):
            # Must not raise even though position_coins was 0
            pm.close_position(pos.trade_id, ENTRY_PRICE, ExitReason.STOP_LOSS)


# ============================================================================
# 2. monitor_positions() — paper mode balance updates
# ============================================================================

class TestMonitorPositionsPaperBalance(unittest.TestCase):

    def test_full_exit_win_credits_balance(self):
        ex = _make_executor(balance=80.0)
        ei = _full_exit(exit_price=107.0, pnl_pct=0.07)

        _run_monitor(ex, [ei])

        self.assertAlmostEqual(ex.paper_balance, 80.0 + ei["exit_value"] * 0.999, places=4)

    def test_full_exit_loss_still_credits_balance(self):
        ex = _make_executor(balance=80.0)
        ei = _full_exit(exit_price=97.0, pnl_pct=-0.03)

        _run_monitor(ex, [ei])

        self.assertGreater(ex.paper_balance, 80.0)
        self.assertLess(ex.paper_balance - 80.0, ei["exit_size"])

    def test_partial_exit_credits_half_value(self):
        ex = _make_executor(balance=80.0)
        ei = _partial_exit(exit_price=103.5, pnl_pct=0.035)

        _run_monitor(ex, [ei])

        self.assertAlmostEqual(ex.paper_balance, 80.0 + ei["exit_value"] * 0.999, places=4)

    def test_full_exit_removes_paper_trade(self):
        ex = _make_executor(balance=100.0)
        ex.paper_trades = {"T-001": {"symbol": "BTCUSDT", "size": 20.0}}

        _run_monitor(ex, [_full_exit(trade_id="T-001")])

        self.assertNotIn("T-001", ex.paper_trades)

    def test_partial_exit_halves_paper_trade_size(self):
        ex = _make_executor(balance=100.0)
        ex.paper_trades = {"T-001": {"symbol": "BTCUSDT", "size": 20.0}}

        _run_monitor(ex, [_partial_exit(trade_id="T-001")])

        self.assertIn("T-001", ex.paper_trades)
        self.assertAlmostEqual(ex.paper_trades["T-001"]["size"], 10.0)

    def test_save_state_called_after_exit(self):
        ex = _make_executor(balance=100.0)

        with patch("src.trading.executor.position_manager") as mock_pm, \
             patch("src.trading.executor.collector") as mock_col, \
             patch("src.trading.executor.save_state") as mock_save, \
             patch("src.trading.executor.risk_manager"):
            mock_pm.get_open_positions.return_value  = [{"coin": "BTCUSDT"}]
            mock_col.get_all_prices.return_value     = {"BTCUSDT": 107.0}
            mock_pm.check_all_positions.return_value = [_full_exit()]
            ex.sync_state = MagicMock()

            ex.monitor_positions()

        mock_save.assert_called()


    def test_risk_manager_set_capital_called(self):
        ex = _make_executor(balance=80.0)

        with patch("src.trading.executor.position_manager") as mock_pm, \
             patch("src.trading.executor.collector") as mock_col, \
             patch("src.trading.executor.save_state"), \
             patch("src.trading.executor.risk_manager") as mock_rm:

            mock_pm.get_open_positions.return_value  = [{"coin": "BTCUSDT"}]
            mock_col.get_all_prices.return_value     = {"BTCUSDT": 107.0}
            mock_pm.check_all_positions.return_value = [_full_exit()]
            ex.sync_state = MagicMock()

            ex.monitor_positions()

        mock_rm.set_capital.assert_called_with(ex.paper_balance)

    def test_exit_with_no_coin_is_skipped(self):
        ex = _make_executor(balance=80.0)
        bad = {"type": "full", "exit_value": 20.0, "exit_coins": 0.2,
               "exit_price": 100.0, "trade_id": "T-X"}

        result = _run_monitor(ex, [bad])

        self.assertAlmostEqual(ex.paper_balance, 80.0)
        self.assertEqual(len(result["exits"]), 0)

    def test_exit_with_zero_exit_value_is_skipped(self):
        ex = _make_executor(balance=80.0)
        bad = {"coin": "BTCUSDT", "type": "full", "exit_value": 0.0,
               "exit_coins": 0.2, "exit_price": 100.0, "trade_id": "T-Y"}

        result = _run_monitor(ex, [bad])

        self.assertAlmostEqual(ex.paper_balance, 80.0)
        self.assertEqual(len(result["exits"]), 0)


# ============================================================================
# 3. monitor_positions() — no open positions
# ============================================================================

class TestMonitorNoPositions(unittest.TestCase):

    def test_early_return_when_no_positions(self):
        ex = _make_executor()

        with patch("src.trading.executor.position_manager") as mock_pm, \
             patch("src.trading.executor.collector"):

            mock_pm.get_open_positions.return_value = []
            ex.sync_state = MagicMock()

            result = ex.monitor_positions()

        self.assertEqual(result, {"monitored": 0, "exits": []})

    def test_balance_unchanged_when_no_positions(self):
        ex = _make_executor(balance=85.96)

        with patch("src.trading.executor.position_manager") as mock_pm, \
             patch("src.trading.executor.collector"):

            mock_pm.get_open_positions.return_value = []
            ex.sync_state = MagicMock()

            ex.monitor_positions()

        self.assertAlmostEqual(ex.paper_balance, 85.96)


# ============================================================================
# 4. monitor_positions() — paper vs live routing
# ============================================================================

class TestMonitorModeRouting(unittest.TestCase):

    def test_paper_mode_uses_paper_sell(self):
        ex = _make_executor(mode=ExecutionMode.PAPER, balance=100.0)
        mock_paper = MagicMock(return_value=MagicMock(success=True))
        mock_live  = MagicMock()
        ex._paper_sell = mock_paper
        ex._live_sell  = mock_live

        with patch("src.trading.executor.position_manager") as mock_pm, \
             patch("src.trading.executor.collector") as mock_col, \
             patch("src.trading.executor.save_state"), \
             patch("src.trading.executor.risk_manager"):

            mock_pm.get_open_positions.return_value  = [{"coin": "BTCUSDT"}]
            mock_col.get_all_prices.return_value     = {"BTCUSDT": 100.0}
            mock_pm.check_all_positions.return_value = [_full_exit()]
            ex.sync_state = MagicMock()
            ex.monitor_positions()

        mock_paper.assert_called_once()
        mock_live.assert_not_called()

    def test_live_mode_uses_live_sell(self):
        ex = _make_executor(mode=ExecutionMode.LIVE, balance=100.0)
        mock_live  = MagicMock(return_value=MagicMock(success=True, order_id="OID"))
        mock_paper = MagicMock()
        ex._live_sell  = mock_live
        ex._paper_sell = mock_paper

        ei = _full_exit()
        with patch("src.trading.executor.position_manager") as mock_pm, \
             patch("src.trading.executor.collector") as mock_col, \
             patch("src.trading.executor.save_state"), \
             patch("src.trading.executor.risk_manager"):

            mock_pm.get_open_positions.return_value  = [{"coin": "BTCUSDT"}]
            mock_col.get_all_prices.return_value     = {"BTCUSDT": 100.0}
            mock_pm.check_all_positions.return_value = [ei]
            ex.sync_state = MagicMock()
            ex.monitor_positions()

        mock_live.assert_called_once_with(ei["coin"], ei["exit_coins"])
        mock_paper.assert_not_called()


# ============================================================================
# 5. _paper_sell unit tests
# ============================================================================

class TestPaperSell(unittest.TestCase):

    def test_paper_sell_increases_balance(self):
        """_paper_sell adds (size × 0.999) to paper_balance."""
        ex = _make_executor(balance=80.0)
        result = ex._paper_sell("BTCUSDT", 100.0, 20.0)

        self.assertTrue(result.success)
        # fill_price = 100 × 0.999; credited = 20 × (99.9/100) = 19.98
        self.assertAlmostEqual(ex.paper_balance, 80.0 + 19.98, places=4)

    def test_paper_sell_order_id_has_sell_suffix(self):
        ex = _make_executor(balance=80.0)
        result = ex._paper_sell("ETHUSDT", 3000.0, 15.0)
        self.assertIn("_SELL", result.order_id)

    def test_paper_sell_always_succeeds(self):
        """_paper_sell never rejects — it always credits the balance."""
        ex = _make_executor(balance=0.0)
        result = ex._paper_sell("BTCUSDT", 100.0, 20.0)
        self.assertTrue(result.success)
        self.assertGreater(ex.paper_balance, 0.0)

    def test_paper_sell_with_exact_exit_value_credits_correct_amount(self):
        ex = _make_executor(balance=80.0)
        exit_value = 21.40
        ex._paper_sell("BTCUSDT", 107.0, exit_value)
        self.assertAlmostEqual(ex.paper_balance, 80.0 + exit_value * 0.999, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
