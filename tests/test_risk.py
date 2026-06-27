"""
Unit tests for RiskManager — covers all risk checks, position sizing,
trailing stop calculation, stop/take-profit pricing, and time-stop logic.
All tests are fully isolated (no DB, no network, no file I/O).

Module-level mock setup is handled by tests/conftest.py.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# conftest.py (auto-loaded by pytest) stubs all heavy deps before these imports.
from src.trading.risk import RiskManager, RiskCheck


# ============================================================================
# Helper — build a RiskManager with controlled state (no DB I/O)
# ============================================================================

def _make_rm(capital: float = 1000.0, open_positions: int = 0,
             open_coins: set = None, total_risk: float = 0.0,
             daily_pnl: float = 0.0, consecutive_losses: int = 0) -> RiskManager:
    with patch.object(RiskManager, "_sync_with_database"):
        rm = RiskManager.__new__(RiskManager)
    from config import settings
    rm.daily_start_capital = capital
    rm.current_capital     = capital
    rm.daily_pnl           = daily_pnl
    rm.trading_paused      = False
    rm.pause_reason        = ""
    rm.pause_until         = None
    rm.consecutive_losses  = consecutive_losses
    rm._open_positions     = open_positions
    rm._open_coins         = open_coins if open_coins is not None else set()
    rm._total_risk         = total_risk
    rm._last_sync_time     = None
    rm._sync_ttl_seconds   = 30
    return rm


# ============================================================================
# 1. Price Level Calculations
# ============================================================================

class TestPriceLevels(unittest.TestCase):

    def setUp(self):
        self.rm = _make_rm()

    def test_stop_loss_long_3pct(self):
        sl = self.rm.get_stop_loss_price(100.0, side="long")
        self.assertAlmostEqual(sl, 97.0, places=6)

    def test_stop_loss_short_3pct(self):
        sl = self.rm.get_stop_loss_price(100.0, side="short")
        self.assertAlmostEqual(sl, 103.0, places=6)

    def test_take_profit_long_7pct(self):
        tp = self.rm.get_take_profit_price(100.0, tier=1)
        self.assertAlmostEqual(tp, 107.0, places=6)

    def test_take_profit_short_7pct(self):
        tp = self.rm.get_take_profit_price(100.0, tier=2, side="short")
        self.assertAlmostEqual(tp, 93.0, places=6)

    def test_partial_exit_long_35pct(self):
        tp1 = self.rm.get_partial_exit_price(100.0, side="long")
        self.assertAlmostEqual(tp1, 103.5, places=6)

    def test_partial_exit_short_35pct(self):
        tp1 = self.rm.get_partial_exit_price(100.0, side="short")
        self.assertAlmostEqual(tp1, 96.5, places=6)

    def test_tp1_below_tp2(self):
        tp1 = self.rm.get_partial_exit_price(100.0)
        tp2 = self.rm.get_take_profit_price(100.0, tier=1)
        self.assertLess(tp1, tp2)


# ============================================================================
# 2. Trailing Stop Calculation
# ============================================================================

class TestTrailingStop(unittest.TestCase):

    def setUp(self):
        self.rm = _make_rm()

    def test_not_active_below_2pct_profit(self):
        active, _ = self.rm.calculate_trailing_stop(100.0, 101.5, 101.5)
        self.assertFalse(active)

    def test_active_at_or_above_2pct_profit(self):
        # pnl = exactly 2% → not less than 0.02 → trail activates
        active, price = self.rm.calculate_trailing_stop(100.0, 102.0, 102.0)
        self.assertTrue(active)

    def test_active_above_2pct_profit(self):
        active, price = self.rm.calculate_trailing_stop(100.0, 105.0, 105.0)
        self.assertTrue(active)
        self.assertAlmostEqual(price, 105.0 * 0.98, places=6)

    def test_trail_anchored_to_highest(self):
        highest = 110.0
        current = 107.0
        active, price = self.rm.calculate_trailing_stop(100.0, current, highest)
        self.assertTrue(active)
        self.assertAlmostEqual(price, highest * 0.98, places=6)

    def test_trail_price_less_than_highest(self):
        highest = 110.0
        current = 107.0
        _, trail = self.rm.calculate_trailing_stop(100.0, current, highest)
        self.assertLess(trail, highest)


# ============================================================================
# 3. Time Stop
# ============================================================================

class TestTimeStop(unittest.TestCase):

    def setUp(self):
        self.rm = _make_rm()

    def test_time_stop_not_triggered_fresh_position(self):
        entry = datetime.utcnow() - timedelta(hours=2)
        self.assertFalse(self.rm.should_time_stop(entry))

    def test_time_stop_triggered_after_limit(self):
        from config import settings
        hours = settings.trading.time_stop_hours
        entry = datetime.utcnow() - timedelta(hours=hours + 0.1)
        self.assertTrue(self.rm.should_time_stop(entry))

    def test_time_stop_boundary(self):
        from config import settings
        hours = settings.trading.time_stop_hours
        entry = datetime.utcnow() - timedelta(hours=hours, seconds=10)
        self.assertTrue(self.rm.should_time_stop(entry))


# ============================================================================
# 4. Position Sizing
# ============================================================================

class TestPositionSizing(unittest.TestCase):

    def test_low_balance_always_min_6(self):
        rm = _make_rm(capital=100.0)
        for tier in [1, 2, 3, 4]:
            size = rm.calculate_position_size(tier=tier, confidence=0.7, coin="BTCUSDT")
            self.assertGreaterEqual(size, 6.0, f"tier {tier} returned {size} < $6")

    def test_low_balance_tier3_higher_than_tier1(self):
        rm = _make_rm(capital=100.0)
        t1 = rm.calculate_position_size(tier=1, confidence=0.7, coin="BTCUSDT")
        t3 = rm.calculate_position_size(tier=3, confidence=0.7, coin="BTCUSDT")
        self.assertGreaterEqual(t3, t1)

    def test_standard_balance_tier_scaling(self):
        rm = _make_rm(capital=1000.0)
        t1 = rm.calculate_position_size(tier=1, confidence=0.7, coin="BTCUSDT")
        t2 = rm.calculate_position_size(tier=2, confidence=0.7, coin="BTCUSDT")
        t3 = rm.calculate_position_size(tier=3, confidence=0.7, coin="BTCUSDT")
        self.assertGreater(t2, t1)
        self.assertGreater(t3, t2)

    def test_high_confidence_boost(self):
        rm = _make_rm(capital=1000.0)
        low  = rm.calculate_position_size(tier=2, confidence=0.70, coin="BTCUSDT")
        high = rm.calculate_position_size(tier=2, confidence=0.90, coin="BTCUSDT")
        self.assertGreater(high, low)

    def test_low_confidence_reduction(self):
        rm = _make_rm(capital=1000.0)
        base    = rm.calculate_position_size(tier=2, confidence=0.70, coin="BTCUSDT")
        reduced = rm.calculate_position_size(tier=2, confidence=0.60, coin="BTCUSDT")
        self.assertLessEqual(reduced, base)

    def test_consecutive_losses_reduction_low_balance(self):
        rm = _make_rm(capital=100.0, consecutive_losses=3)
        size = rm.calculate_position_size(tier=3, confidence=0.9, coin="BTCUSDT")
        self.assertEqual(size, 6.0)

    def test_minimum_floor_always_6(self):
        rm = _make_rm(capital=50.0)
        size = rm.calculate_position_size(tier=4, confidence=0.50, coin="BTCUSDT")
        self.assertGreaterEqual(size, 6.0)


# ============================================================================
# 5. Max Positions (adaptive)
# ============================================================================

class TestMaxPositions(unittest.TestCase):

    def test_very_low_balance_max_1(self):
        rm = _make_rm(capital=50.0)
        self.assertEqual(rm._get_max_positions_for_balance(), 1)

    def test_100_200_max_2(self):
        rm = _make_rm(capital=150.0)
        self.assertEqual(rm._get_max_positions_for_balance(), 2)

    def test_200_300_max_3(self):
        rm = _make_rm(capital=250.0)
        self.assertEqual(rm._get_max_positions_for_balance(), 3)

    def test_above_300_max_5(self):
        rm = _make_rm(capital=500.0)
        from config import settings
        self.assertEqual(rm._get_max_positions_for_balance(), settings.trading.max_positions)


# ============================================================================
# 6. Portfolio Risk Limit (adaptive)
# ============================================================================

class TestPortfolioRiskLimit(unittest.TestCase):

    def test_below_200_30pct(self):
        rm = _make_rm(capital=100.0)
        self.assertAlmostEqual(rm._get_max_portfolio_risk(), 0.30)

    def test_200_300_20pct(self):
        rm = _make_rm(capital=250.0)
        self.assertAlmostEqual(rm._get_max_portfolio_risk(), 0.20)

    def test_above_300_standard(self):
        rm = _make_rm(capital=500.0)
        from config import settings
        self.assertAlmostEqual(rm._get_max_portfolio_risk(), settings.trading.max_portfolio_risk)


# ============================================================================
# 7. can_open_position — all gates
# ============================================================================

class TestCanOpenPosition(unittest.TestCase):

    def _rm(self, **kw):
        rm = _make_rm(**kw)
        rm._sync_with_database = lambda: None
        return rm

    def test_all_clear(self):
        rm = self._rm(capital=1000.0)
        result = rm.can_open_position("BTCUSDT", 20.0, 50000.0)
        self.assertTrue(result.can_trade)

    def test_daily_loss_limit_blocks(self):
        rm = self._rm(capital=1000.0, daily_pnl=-85.0)  # -8.5%
        result = rm.can_open_position("BTCUSDT", 20.0, 50000.0)
        self.assertFalse(result.can_trade)
        self.assertIn("loss", result.reason.lower())

    def test_max_positions_blocks(self):
        rm = self._rm(capital=1000.0, open_positions=5)
        result = rm.can_open_position("BTCUSDT", 20.0, 50000.0)
        self.assertFalse(result.can_trade)
        self.assertIn("positions", result.reason.lower())

    def test_duplicate_coin_blocks(self):
        rm = self._rm(capital=1000.0, open_coins={"ETHUSDT"})
        result = rm.can_open_position("ETHUSDT", 20.0, 3000.0)
        self.assertFalse(result.can_trade)
        self.assertIn("ETHUSDT", result.reason)

    def test_portfolio_risk_exceeded_blocks(self):
        """$95 risk + $20 new = $115 / $1000 = 11.5% > 10% standard limit."""
        rm = self._rm(capital=1000.0, total_risk=95.0)
        result = rm.can_open_position("BTCUSDT", 20.0, 50000.0)
        self.assertFalse(result.can_trade)
        self.assertIn("risk", result.reason.lower())

    def test_position_too_small_hard_blocks(self):
        """$1 is far below the $6 floor — must be rejected."""
        rm = self._rm(capital=1000.0)
        result = rm.can_open_position("BTCUSDT", 1.0, 50000.0)
        self.assertFalse(result.can_trade)
        self.assertIn("minimum", result.reason.lower())

    def test_position_within_01_of_floor_auto_rounded(self):
        """
        On a $600 account min_size = max(6, 600*0.01) = max(6, 6) = $6.
        A position of $5.95 is within $0.10 of $6 → silently auto-rounded up.
        """
        # Use capital=600 so that 1% of capital = $6.00 → min_size = $6.00
        rm = self._rm(capital=600.0)
        result = rm.can_open_position("BTCUSDT", 5.95, 50000.0)
        self.assertTrue(result.can_trade)

    def test_consecutive_losses_allows_trade_with_reduce_hint(self):
        rm = self._rm(capital=1000.0, consecutive_losses=5)
        result = rm.can_open_position("BTCUSDT", 20.0, 50000.0)
        self.assertTrue(result.can_trade)
        self.assertIsNotNone(result.details)
        self.assertTrue(result.details.get("reduce_size"))

    def test_trading_paused_blocks(self):
        rm = self._rm(capital=1000.0)
        rm._pause_trading("test reason", hours=24)
        result = rm.can_open_position("BTCUSDT", 20.0, 50000.0)
        self.assertFalse(result.can_trade)

    def test_trading_paused_auto_resumes_when_expired(self):
        rm = self._rm(capital=1000.0)
        rm.trading_paused = True
        rm.pause_reason   = "old reason"
        rm.pause_until    = datetime.utcnow() - timedelta(seconds=1)
        result = rm.can_open_position("BTCUSDT", 20.0, 50000.0)
        self.assertTrue(result.can_trade)
        self.assertFalse(rm.trading_paused)


# ============================================================================
# 8. record_trade_result
# ============================================================================

class TestRecordTradeResult(unittest.TestCase):

    def test_profit_resets_consecutive_losses(self):
        rm = _make_rm(capital=1000.0, consecutive_losses=3)
        rm.record_trade_result(pnl_usd=50.0, pnl_pct=0.05)
        self.assertEqual(rm.consecutive_losses, 0)

    def test_loss_increments_consecutive_losses(self):
        rm = _make_rm(capital=1000.0, consecutive_losses=0)
        rm.record_trade_result(pnl_usd=-30.0, pnl_pct=-0.03)
        self.assertEqual(rm.consecutive_losses, 1)

    def test_daily_pnl_updated(self):
        rm = _make_rm(capital=1000.0, daily_pnl=0.0)
        rm.record_trade_result(pnl_usd=40.0, pnl_pct=0.04)
        self.assertAlmostEqual(rm.daily_pnl, 40.0)

    def test_capital_updated(self):
        rm = _make_rm(capital=1000.0)
        rm.set_capital(1100.0)
        self.assertAlmostEqual(rm.current_capital, 1100.0)

    def test_daily_loss_triggers_pause(self):
        rm = _make_rm(capital=1000.0, daily_pnl=-70.0)
        rm.record_trade_result(pnl_usd=-15.0, pnl_pct=-0.015)
        self.assertTrue(rm.trading_paused)

    def test_reset_daily(self):
        rm = _make_rm(capital=1000.0, daily_pnl=-50.0)
        rm.trading_paused = True
        rm.reset_daily()
        self.assertAlmostEqual(rm.daily_pnl, 0.0)
        self.assertFalse(rm.trading_paused)



# ============================================================================
# 9. _sync_with_database — _total_risk uses remaining_size (BUG 16)
# ============================================================================

class TestSyncWithDatabaseTotalRisk(unittest.TestCase):
    """
    Verify that _sync_with_database() computes _total_risk from
    remaining_size (post-partial-exit) rather than the static position_size.
    """

    def _make_trade(self, position_size, remaining_size):
        """Return a simple object mimicking a DB Trade row."""
        t = MagicMock()
        t.position_size  = position_size
        t.remaining_size = remaining_size
        t.coin = "BTCUSDT"
        return t

    def test_total_risk_uses_remaining_size_after_partial_exit(self):
        """After TP1, remaining_size is halved — _total_risk must reflect that."""
        trade = self._make_trade(position_size=20.0, remaining_size=10.0)

        import src.trading.risk as _risk_mod
        rm = _make_rm()
        orig_db = _risk_mod.db
        mock_db = MagicMock()
        mock_db.get_daily_stats.return_value = {"pnl": 0.0}
        mock_db.get_open_trades.return_value  = [trade]
        _risk_mod.db = mock_db
        try:
            rm._sync_with_database()
        finally:
            _risk_mod.db = orig_db

        # Should be remaining_size (10), NOT position_size (20)
        self.assertAlmostEqual(rm._total_risk, 10.0)

    def test_total_risk_falls_back_to_position_size_when_remaining_is_none(self):
        """When remaining_size is None (no partial exit yet), use position_size."""
        trade = self._make_trade(position_size=20.0, remaining_size=None)

        import src.trading.risk as _risk_mod
        rm = _make_rm()
        orig_db = _risk_mod.db
        mock_db = MagicMock()
        mock_db.get_daily_stats.return_value = {"pnl": 0.0}
        mock_db.get_open_trades.return_value  = [trade]
        _risk_mod.db = mock_db
        try:
            rm._sync_with_database()
        finally:
            _risk_mod.db = orig_db

        self.assertAlmostEqual(rm._total_risk, 20.0)

    def test_total_risk_sums_multiple_trades_mixed_remaining(self):
        """Multiple open trades — some partial, some not — are summed correctly."""
        trade_a = self._make_trade(position_size=20.0, remaining_size=10.0)   # partial exit
        trade_b = self._make_trade(position_size=15.0, remaining_size=None)   # no partial

        import src.trading.risk as _risk_mod
        rm = _make_rm()
        orig_db = _risk_mod.db
        mock_db = MagicMock()
        mock_db.get_daily_stats.return_value = {"pnl": 0.0}
        mock_db.get_open_trades.return_value  = [trade_a, trade_b]
        _risk_mod.db = mock_db
        try:
            rm._sync_with_database()
        finally:
            _risk_mod.db = orig_db

        # 10.0 (trade_a remaining) + 15.0 (trade_b full size) = 25.0
        self.assertAlmostEqual(rm._total_risk, 25.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
