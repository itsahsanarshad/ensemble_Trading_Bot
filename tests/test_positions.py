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


if __name__ == "__main__":
    unittest.main(verbosity=2)
