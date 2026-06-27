"""
Unit tests for PositionManager.check_position() — covers:

  - Stop loss trigger
  - TP1 (partial exit) on normal-size positions (>= $12)
  - TP1 SKIP + tight trailing stop on small positions (< $12)
  - TP2 (full take profit)
  - Standard trailing stop (risk_manager)
  - Tight trailing stop ratchet (only moves up, never down)
  - Tight trailing stop trigger
  - Time stop
  - No action when simply holding

All tests are completely isolated: DB calls, discord, and logging are mocked.
Module-level mock setup is handled by tests/conftest.py.
"""

import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Optional

# conftest.py stubs all heavy deps before these imports.
from src.trading.positions import Position, PositionManager

# ── Grab the MagicMock-db that positions.py will use at runtime ────────────
# conftest registered "src.data.database" as a MagicMock — but by the time
# positions.py imported it, Python may have cached the real module.
# We patch positions.db + positions.logger + positions.risk_manager directly
# on each test class using @patch decorators so we stay isolated.

ENTRY_PRICE  = 100.0
STOP_LOSS    = 97.0           # entry × (1 - 3%)
TAKE_PROFIT  = 107.0          # entry × (1 + 7%)  — TP2
TP1_PRICE    = 103.5          # entry × (1 + 3.5%) — TP1
TIME_HOURS   = 6


def _make_position(
    position_size: float = 20.0,
    highest_price: float = ENTRY_PRICE,
    trailing_stop: Optional[float] = None,
    partial_exit_done: bool = False,
    entry_hours_ago: float = 0.0,
) -> Position:
    return Position(
        trade_id="TEST-001",
        coin="BTCUSDT",
        entry_price=ENTRY_PRICE,
        position_size=position_size,
        position_coins=position_size / ENTRY_PRICE,
        entry_time=datetime.utcnow() - timedelta(hours=entry_hours_ago),
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        tier=1,
        highest_price=highest_price,
        trailing_stop=trailing_stop,
        partial_exit_done=partial_exit_done,
    )


def _make_pm(position: Position) -> PositionManager:
    pm = PositionManager.__new__(PositionManager)
    pm.positions = {position.trade_id: position}
    pm._last_hp_db_write = {}
    return pm


class _StubRM:
    """Minimal stand-in for risk_manager with the exact methods check_position calls."""

    def get_partial_exit_price(self, entry, side="long"):
        return entry * 1.035   # TP1 = +3.5%

    def get_take_profit_price(self, entry, tier, side="long"):
        return entry * 1.07    # TP2 = +7%

    def calculate_trailing_stop(self, entry, current, highest, side="long"):
        pnl = (current - entry) / entry
        if pnl < 0.02:
            return (False, 0)
        return (True, highest * 0.98)  # 2% below highest

    def should_time_stop(self, entry_time):
        elapsed = (datetime.utcnow() - entry_time).total_seconds() / 3600
        return elapsed > TIME_HOURS

    def record_trade_result(self, pnl_usd, pnl_pct):
        pass

    def _invalidate_sync_cache(self):
        pass


_stub_rm    = _StubRM()
_db_mock    = MagicMock(name="db_for_positions")
_log_mock   = MagicMock(name="logger_for_positions")
_disc_mock  = MagicMock(name="discord_for_positions")
_log_pu_mock = MagicMock(name="log_position_update")

_db_mock.update_trade.return_value = None


# Every test class patches the module-level globals that check_position uses.
_PATCHES = dict(
    _db     = patch("src.trading.positions.db",                 _db_mock),
    _logger = patch("src.trading.positions.logger",             _log_mock),
    _disc   = patch("src.trading.positions.discord_notifier",   _disc_mock),
    _lpu    = patch("src.trading.positions.log_position_update", _log_pu_mock),
    _rm     = patch("src.trading.positions.risk_manager",        _stub_rm),
)


def setUpModule():
    for p in _PATCHES.values():
        p.start()


def tearDownModule():
    for p in _PATCHES.values():
        p.stop()


# ============================================================================
# Test Cases
# ============================================================================

class TestCheckPosition(unittest.TestCase):

    def setUp(self):
        _db_mock.reset_mock()
        _log_mock.reset_mock()

    # ── 1. Unknown trade ID ────────────────────────────────────────────────

    def test_unknown_trade_id_returns_none(self):
        pm = _make_pm(_make_position())
        self.assertIsNone(pm.check_position("NONEXISTENT", 100.0))

    # ── 2. Stop loss ───────────────────────────────────────────────────────

    def test_stop_loss_triggers(self):
        pos = _make_position()
        result = _make_pm(pos).check_position(pos.trade_id, STOP_LOSS)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "stop_loss")

    def test_stop_loss_not_triggered_at_entry(self):
        pos = _make_position()
        result = _make_pm(pos).check_position(pos.trade_id, ENTRY_PRICE)
        self.assertNotEqual(result[0] if result else "", "stop_loss")

    # ── 3. TP1 — normal-size position (>= $12) ────────────────────────────

    def test_tp1_triggers_partial_exit_normal_size(self):
        pos = _make_position(position_size=20.0)
        result = _make_pm(pos).check_position(pos.trade_id, TP1_PRICE)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "partial_exit")

    def test_tp1_moves_sl_to_breakeven_normal_size(self):
        pos = _make_position(position_size=20.0)
        _make_pm(pos).check_position(pos.trade_id, TP1_PRICE)
        self.assertAlmostEqual(pos.stop_loss, ENTRY_PRICE, places=6)

    def test_tp1_writes_updated_sl_to_db(self):
        pos = _make_position(position_size=20.0)
        _make_pm(pos).check_position(pos.trade_id, TP1_PRICE)
        _db_mock.update_trade.assert_called_with(pos.trade_id, stop_loss_price=ENTRY_PRICE)

    def test_tp1_not_triggered_when_partial_exit_done(self):
        pos = _make_position(position_size=20.0, partial_exit_done=True)
        result = _make_pm(pos).check_position(pos.trade_id, TP1_PRICE)
        if result is not None:
            self.assertNotEqual(result[0], "partial_exit")

    # ── 4. TP1 SKIP — small position (< $12) ─────────────────────────────

    def test_tp1_skipped_for_small_position(self):
        pos = _make_position(position_size=10.0)
        result = _make_pm(pos).check_position(pos.trade_id, TP1_PRICE)
        self.assertNotEqual(result[0] if result else "", "partial_exit")

    def test_sl_not_moved_when_tp1_skipped(self):
        pos = _make_position(position_size=10.0)
        _make_pm(pos).check_position(pos.trade_id, TP1_PRICE)
        self.assertAlmostEqual(pos.stop_loss, STOP_LOSS, places=6)

    def test_tight_trailing_stop_activated_for_small_position_at_tp1(self):
        """
        Small position whose highest_price >= TP1 must get a trailing_stop
        set to highest_price × (1 - 1.5%).
        """
        pos = _make_position(position_size=10.0, highest_price=TP1_PRICE)
        _make_pm(pos).check_position(pos.trade_id, TP1_PRICE)
        expected_trail = TP1_PRICE * (1 - 0.015)
        self.assertIsNotNone(pos.trailing_stop)
        self.assertAlmostEqual(pos.trailing_stop, expected_trail, places=4)

    def test_tight_trailing_stop_not_activated_below_tp1(self):
        """Price never reached TP1 — no tight trail should activate."""
        below_tp1 = 101.0   # 1% gain, well below TP1 at 103.5
        pos = _make_position(position_size=10.0, highest_price=below_tp1)
        _make_pm(pos).check_position(pos.trade_id, below_tp1)
        # If any trail was set, it should NOT be the tight 1.5% trail
        if pos.trailing_stop is not None:
            not_tight = below_tp1 * (1 - 0.015)
            self.assertNotAlmostEqual(pos.trailing_stop, not_tight, places=2)

    # ── 5. Tight trailing stop ratchet ────────────────────────────────────

    def test_tight_trail_ratchets_upward(self):
        pos = _make_position(position_size=10.0, highest_price=TP1_PRICE)
        pm  = _make_pm(pos)

        pm.check_position(pos.trade_id, TP1_PRICE)
        trail_tick1 = pos.trailing_stop

        # Price rises — update highest_price as the monitoring loop would
        new_high = 106.0
        pos.highest_price = new_high
        pm.check_position(pos.trade_id, new_high)
        trail_tick2 = pos.trailing_stop

        self.assertGreater(trail_tick2, trail_tick1)

    def test_tight_trail_never_moves_down(self):
        """If highest_price somehow drops, the trailing stop must not be lowered."""
        existing_trail = 106.0 * 0.985
        pos = _make_position(position_size=10.0, highest_price=106.0,
                             trailing_stop=existing_trail)
        pm  = _make_pm(pos)

        # Simulate a lower highest_price on next tick
        pos.highest_price = TP1_PRICE   # 103.5 < 106
        pm.check_position(pos.trade_id, TP1_PRICE)

        self.assertGreaterEqual(pos.trailing_stop, existing_trail)

    def test_tight_trail_triggers_stop(self):
        """When current_price <= trailing_stop, action must be 'trailing_stop'."""
        high  = 107.0
        trail = high * (1 - 0.015)   # 105.395
        pos   = _make_position(position_size=10.0, highest_price=high,
                               trailing_stop=trail)
        result = _make_pm(pos).check_position(pos.trade_id, trail - 0.5)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "trailing_stop")

    # ── 6. TP2 (full take profit) ─────────────────────────────────────────

    def test_tp2_triggers_full_exit(self):
        pos = _make_position(position_size=20.0, partial_exit_done=True)
        result = _make_pm(pos).check_position(pos.trade_id, TAKE_PROFIT)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "take_profit")

    def test_tp2_triggers_for_small_position_too(self):
        pos = _make_position(position_size=10.0)
        pos.highest_price = TAKE_PROFIT
        result = _make_pm(pos).check_position(pos.trade_id, TAKE_PROFIT)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "take_profit")

    # ── 7. Standard trailing stop (normal-size positions) ─────────────────

    def test_standard_trail_triggers(self):
        highest = 106.0
        trail   = highest * 0.98    # 103.88
        pos     = _make_position(position_size=20.0, highest_price=highest,
                                 trailing_stop=trail,
                                 partial_exit_done=True)   # TP1 already done
        result = _make_pm(pos).check_position(pos.trade_id, trail - 0.1)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "trailing_stop")


    def test_standard_trail_not_active_below_2pct_gain(self):
        """Standard trail doesn't activate until gain >= 2%."""
        pos = _make_position(position_size=20.0, highest_price=101.0)
        result = _make_pm(pos).check_position(pos.trade_id, 101.0)
        if result is not None:
            self.assertNotEqual(result[0], "trailing_stop")

    # ── 8. Time stop ──────────────────────────────────────────────────────

    def test_time_stop_triggers_when_profit_low(self):
        """After the time limit with PnL < 1.5%, position is closed by time stop."""
        pos = _make_position(position_size=20.0, entry_hours_ago=TIME_HOURS + 0.5)
        result = _make_pm(pos).check_position(pos.trade_id, 101.0)  # +1% < 1.5%
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "time_stop")

    def test_time_stop_does_not_trigger_when_profitable(self):
        """PnL >= 1.5% → time stop must not fire."""
        pos = _make_position(position_size=20.0, entry_hours_ago=TIME_HOURS + 0.5)
        result = _make_pm(pos).check_position(pos.trade_id, 102.0)  # +2% > 1.5%
        if result is not None:
            self.assertNotEqual(result[0], "time_stop")

    def test_time_stop_not_triggered_before_limit(self):
        pos = _make_position(position_size=20.0, entry_hours_ago=1.0)
        result = _make_pm(pos).check_position(pos.trade_id, 101.0)
        if result is not None:
            self.assertNotEqual(result[0], "time_stop")

    # ── 9. No action ──────────────────────────────────────────────────────

    def test_no_action_when_holding(self):
        pos = _make_position(position_size=20.0)
        # +1%: above SL (97), below TP1 (103.5), no trail yet (gain < 2%)
        result = _make_pm(pos).check_position(pos.trade_id, 101.0)
        self.assertIsNone(result)

    # ── 10. Highest price tracking ────────────────────────────────────────

    def test_highest_price_updated(self):
        pos = _make_position(highest_price=ENTRY_PRICE)
        _make_pm(pos).check_position(pos.trade_id, 105.0)
        self.assertAlmostEqual(pos.highest_price, 105.0)

    def test_highest_price_not_reduced(self):
        pos = _make_position(highest_price=105.0)
        _make_pm(pos).check_position(pos.trade_id, 101.0)
        self.assertAlmostEqual(pos.highest_price, 105.0)


# ============================================================================
# 11. Model exit signal (BUG 20 — ensemble.get_exit_signal wired up)
# ============================================================================

class TestModelExitSignal(unittest.TestCase):
    """
    Verify that check_position() delegates to ensemble.get_exit_signal() and
    returns ("signal_exit", reason) when the ensemble signals a trend reversal.
    """

    def setUp(self):
        _db_mock.reset_mock()
        _log_mock.reset_mock()

    def test_signal_exit_returned_when_ensemble_says_exit(self):
        """When get_exit_signal returns ("exit", reason), action must be signal_exit."""
        pos = _make_position(position_size=20.0)
        pm  = _make_pm(pos)

        mock_ensemble = MagicMock()
        mock_ensemble.get_exit_signal.return_value = ("exit", "Bearish trend reversal detected")

        # The function does `from src.models.ensemble import ensemble` lazily,
        # so we patch the object on the source module (not the positions namespace).
        with patch("src.models.ensemble.ensemble", mock_ensemble):
            result = pm.check_position(pos.trade_id, ENTRY_PRICE * 1.01)  # in profit, no SL/TP hit

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "signal_exit")
        self.assertIn("Bearish", result[1])

    def test_no_signal_exit_when_ensemble_returns_hold(self):
        """When get_exit_signal returns something other than 'exit', no signal_exit."""
        pos = _make_position(position_size=20.0)
        pm  = _make_pm(pos)

        mock_ensemble = MagicMock()
        mock_ensemble.get_exit_signal.return_value = ("hold", "")

        with patch("src.models.ensemble.ensemble", mock_ensemble):
            result = pm.check_position(pos.trade_id, ENTRY_PRICE * 1.01)

        if result is not None:
            self.assertNotEqual(result[0], "signal_exit")

    def test_signal_exit_exception_is_swallowed(self):
        """If ensemble.get_exit_signal raises, check_position must not propagate it."""
        pos = _make_position(position_size=20.0)
        pm  = _make_pm(pos)

        mock_ensemble = MagicMock()
        mock_ensemble.get_exit_signal.side_effect = RuntimeError("model unavailable")

        with patch("src.models.ensemble.ensemble", mock_ensemble):
            # Should not raise
            try:
                pm.check_position(pos.trade_id, ENTRY_PRICE * 1.01)
            except RuntimeError:
                self.fail("check_position() propagated RuntimeError from get_exit_signal")


# ============================================================================
# 12. TP migration log — variable shadowing fix (BUG 5 / Minor 17)
# ============================================================================

class TestTPMigrationLog(unittest.TestCase):
    """
    Verify that _load_open_positions() captures the old TP value BEFORE
    overwriting it, so the warning log shows the correct "from" price.
    """

    def test_migration_log_records_old_tp_not_new(self):
        """
        When a stale TP2 <= TP1 is found, the warning must include the OLD TP
        in the 'from' position, not the newly calculated tp2_correct value.
        """
        from unittest.mock import call

        # Build a fake trade that will trigger migration: take_profit <= TP1 (3.5%)
        old_stale_tp = ENTRY_PRICE * 1.03   # 3% — below TP1 of 3.5%, will trigger migration
        tp1          = ENTRY_PRICE * 1.035  # 103.5
        tp2_correct  = ENTRY_PRICE * 1.07   # 107.0

        fake_trade = MagicMock()
        fake_trade.trade_id             = "MIG-001"
        fake_trade.coin                 = "BTCUSDT"
        fake_trade.entry_price          = ENTRY_PRICE
        fake_trade.position_size        = 20.0
        fake_trade.position_size_coins  = 0.20
        fake_trade.entry_time           = datetime.utcnow()
        fake_trade.stop_loss_price      = STOP_LOSS
        fake_trade.take_profit_price    = old_stale_tp   # STALE — below TP1
        fake_trade.consensus_tier       = 1
        fake_trade.highest_price        = ENTRY_PRICE
        fake_trade.trailing_stop_price  = None
        fake_trade.partial_exit_done    = 0
        fake_trade.remaining_size       = 20.0
        fake_trade.ta_confidence        = 0.0
        fake_trade.ml_confidence        = 0.0
        fake_trade.tcn_confidence       = 0.0
        fake_trade.entry_reason         = ""

        with patch("src.trading.positions.db") as mock_db, \
             patch("src.trading.positions.risk_manager") as mock_rm, \
             patch("src.trading.positions.logger") as mock_logger:

            mock_db.get_open_trades.return_value  = [fake_trade]
            mock_db.update_trade.return_value      = None
            mock_rm.get_partial_exit_price.return_value = tp1
            mock_rm.get_take_profit_price.return_value  = tp2_correct

            pm = PositionManager.__new__(PositionManager)
            pm.positions = {}
            pm._load_open_positions()

        # Verify logger.warning was called with a message containing the OLD stale TP
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        migration_calls = [c for c in warning_calls if "Migrated" in c]
        self.assertTrue(
            len(migration_calls) > 0,
            "Expected a migration warning to be logged"
        )
        # The log must contain the old stale TP value, NOT tp2_correct as the "from"
        old_tp_str = f"{old_stale_tp:.4f}"
        new_tp_str = f"{tp2_correct:.4f}"
        log_text = migration_calls[0]
        # The "from" part must reference old_stale_tp
        self.assertIn(old_tp_str, log_text,
                      f"Migration log must contain old TP {old_tp_str}; got: {log_text}")


class TestPartialExitPnLAccumulation(unittest.TestCase):
    """
    Test that full exit correctly retrieves and aggregates the realized partial exit PnL
    so the final PnL logged is the total trade outcome.
    """

    def setUp(self):
        _db_mock.reset_mock()
        _log_mock.reset_mock()
        _disc_mock.reset_mock()

    @patch("src.trading.positions.db", _db_mock)
    @patch("src.trading.positions.logger", _log_mock)
    @patch("src.trading.positions.discord_notifier", _disc_mock)
    @patch("src.trading.positions.risk_manager", _stub_rm)
    def test_close_position_accumulates_partial_pnl(self):
        # Setup a position that has undergone partial exit
        pos = _make_position(position_size=20.0, partial_exit_done=True)
        pos.remaining_size = 10.0
        pos.position_coins = 0.1
        
        pm = _make_pm(pos)

        # Mock the DB to return a Trade object with the first leg's PnL ($0.70)
        mock_trade = MagicMock()
        mock_trade.pnl_usd = 0.70
        _db_mock.get_session.return_value.query.return_value.filter.return_value.first.return_value = mock_trade

        # Close position (final leg has breakeven exit, so final leg PnL is 0.0)
        result = pm.close_position(pos.trade_id, exit_price=100.0, exit_reason="STOP_LOSS")

        # PnL expected to be first_leg ($0.70) + final_leg ($0.0) = $0.70
        # return percentage = 0.70 / 20.0 = 0.035 (3.5%)
        self.assertAlmostEqual(result["pnl_usd"], 0.70)
        self.assertAlmostEqual(result["pnl_pct"], 0.035)

        # Verify update_trade got total_pnl_usd
        _db_mock.update_trade.assert_any_call(
            pos.trade_id,
            exit_time=unittest.mock.ANY,
            exit_price=100.0,
            exit_reason="STOP_LOSS",
            pnl_percent=0.0,
            pnl_usd=0.70,
            status="CLOSED"
        )

        # Verify discord alert sent total pnl_usd and total pnl_pct (3.5%)
        _disc_mock.send_sell_alert.assert_called_once()
        args, kwargs = _disc_mock.send_sell_alert.call_args
        self.assertAlmostEqual(kwargs.get("pnl_usd"), 0.70)
        self.assertAlmostEqual(kwargs.get("pnl_pct"), 3.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
