"""
Unit tests for ModelPerformanceTracker inside src/models/ensemble.py.

Covers:
  - record_prediction / record_outcome
  - Win rate calculation (rolling 100)
  - Weight adjustment (±20% band)
  - Prediction trimming
  - save / load (using a temp file)

Module-level mock setup is handled by tests/conftest.py.
"""

import sys
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

# conftest.py stubs all heavy deps before these imports.
from src.models.ensemble import ModelPerformanceTracker


# ============================================================================
# Helpers
# ============================================================================

def _make_tracker(tmp_path: str = None) -> ModelPerformanceTracker:
    """Build a tracker with a temp file and empty state — no disk I/O on init."""
    if tmp_path is None:
        tmp_path = tempfile.mktemp(suffix=".json")
    t = ModelPerformanceTracker.__new__(ModelPerformanceTracker)
    t.save_path   = tmp_path
    t.predictions = []
    t.win_rates   = {"ta": 0.5, "ml": 0.5, "tcn": 0.5}
    return t


def _tracker_with_pending(symbol="BTCUSDT", model="ml") -> ModelPerformanceTracker:
    t = _make_tracker()
    ts = (datetime.utcnow() - timedelta(hours=1)).isoformat()
    t.predictions = [{
        "symbol": symbol, "model": model, "signal": "buy",
        "confidence": 0.8, "timestamp": ts,
        "outcome": None, "pnl": None,
    }]
    return t


def _add_outcomes(tracker, model, n_wins, n_losses):
    sym = "BTCUSDT"
    for _ in range(n_wins):
        tracker.predictions.append({
            "symbol": sym, "model": model, "signal": "buy",
            "confidence": 0.8, "timestamp": datetime.utcnow().isoformat(),
            "outcome": "win", "pnl": 1.0,
        })
    for _ in range(n_losses):
        tracker.predictions.append({
            "symbol": sym, "model": model, "signal": "buy",
            "confidence": 0.8, "timestamp": datetime.utcnow().isoformat(),
            "outcome": "loss", "pnl": -1.0,
        })


# ============================================================================
# 1. record_prediction
# ============================================================================

class TestRecordPrediction(unittest.TestCase):

    def test_prediction_appended(self):
        t = _make_tracker()
        with patch.object(t, "save"):
            t.record_prediction("BTCUSDT", "ml", "buy", 0.75)
        self.assertEqual(len(t.predictions), 1)

    def test_prediction_fields(self):
        t = _make_tracker()
        with patch.object(t, "save"):
            t.record_prediction("ETHUSDT", "tcn", "sell", 0.65)
        p = t.predictions[-1]
        self.assertEqual(p["symbol"],     "ETHUSDT")
        self.assertEqual(p["model"],      "tcn")
        self.assertEqual(p["signal"],     "sell")
        self.assertAlmostEqual(p["confidence"], 0.65)
        self.assertIsNone(p["outcome"])

    def test_model_name_lowercased(self):
        t = _make_tracker()
        with patch.object(t, "save"):
            t.record_prediction("BTCUSDT", "ML", "buy", 0.8)
        self.assertEqual(t.predictions[-1]["model"], "ml")

    def test_predictions_trimmed_at_500(self):
        t = _make_tracker()
        for i in range(510):
            t.predictions.append({"dummy": i})
        t._trim_predictions()
        self.assertLessEqual(len(t.predictions), 500)


# ============================================================================
# 2. record_outcome
# ============================================================================

class TestRecordOutcome(unittest.TestCase):

    def test_outcome_set_correctly(self):
        t = _tracker_with_pending()
        with patch.object(t, "save"):
            t.record_outcome("BTCUSDT", "win", pnl=5.0)
        self.assertEqual(t.predictions[0]["outcome"], "win")

    def test_outcome_lowercased(self):
        t = _tracker_with_pending()
        with patch.object(t, "save"):
            t.record_outcome("BTCUSDT", "WIN", pnl=5.0)
        self.assertEqual(t.predictions[0]["outcome"], "win")

    def test_pnl_recorded(self):
        t = _tracker_with_pending()
        with patch.object(t, "save"):
            t.record_outcome("BTCUSDT", "win", pnl=12.5)
        self.assertAlmostEqual(t.predictions[0]["pnl"], 12.5)

    def test_only_matching_symbol_updated(self):
        t = _make_tracker()
        ts = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        t.predictions = [
            {"symbol": "BTCUSDT", "model": "ml", "signal": "buy",
             "confidence": 0.8, "timestamp": ts, "outcome": None, "pnl": None},
            {"symbol": "ETHUSDT", "model": "ml", "signal": "buy",
             "confidence": 0.8, "timestamp": ts, "outcome": None, "pnl": None},
        ]
        with patch.object(t, "save"):
            t.record_outcome("BTCUSDT", "win", pnl=5.0)
        self.assertEqual(t.predictions[0]["outcome"], "win")
        self.assertIsNone(t.predictions[1]["outcome"])

    def test_stale_prediction_outside_lookback_not_updated(self):
        """Predictions older than lookback_hours should not be updated."""
        t = _make_tracker()
        old_ts = (datetime.utcnow() - timedelta(hours=50)).isoformat()
        t.predictions = [{
            "symbol": "BTCUSDT", "model": "ml", "signal": "buy",
            "confidence": 0.8, "timestamp": old_ts,
            "outcome": None, "pnl": None,
        }]
        with patch.object(t, "save"):
            t.record_outcome("BTCUSDT", "win", pnl=5.0, lookback_hours=48)
        self.assertIsNone(t.predictions[0]["outcome"])

    def test_within_lookback_updated(self):
        t = _make_tracker()
        ts = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        t.predictions = [{
            "symbol": "BTCUSDT", "model": "ml", "signal": "buy",
            "confidence": 0.8, "timestamp": ts,
            "outcome": None, "pnl": None,
        }]
        with patch.object(t, "save"):
            t.record_outcome("BTCUSDT", "win", pnl=5.0, lookback_hours=48)
        self.assertEqual(t.predictions[0]["outcome"], "win")


# ============================================================================
# 3. Win-rate calculation
# ============================================================================

class TestWinRates(unittest.TestCase):

    def test_100pct_win_rate(self):
        t = _make_tracker()
        _add_outcomes(t, "ml", n_wins=20, n_losses=0)
        t._calculate_win_rates()
        self.assertAlmostEqual(t.win_rates["ml"], 1.0)

    def test_50pct_win_rate(self):
        t = _make_tracker()
        _add_outcomes(t, "ml", n_wins=15, n_losses=15)
        t._calculate_win_rates()
        self.assertAlmostEqual(t.win_rates["ml"], 0.5)

    def test_0pct_win_rate(self):
        t = _make_tracker()
        _add_outcomes(t, "tcn", n_wins=0, n_losses=20)
        t._calculate_win_rates()
        self.assertAlmostEqual(t.win_rates["tcn"], 0.0)

    def test_fewer_than_10_defaults_to_05(self):
        """With < 10 resolved predictions the win rate stays at 0.5 (neutral)."""
        t = _make_tracker()
        _add_outcomes(t, "ta", n_wins=5, n_losses=0)
        t._calculate_win_rates()
        self.assertAlmostEqual(t.win_rates["ta"], 0.5)

    def test_rolling_window_uses_last_100(self):
        """Only the last 100 predictions matter."""
        t = _make_tracker()
        # 200 old losses then 100 recent wins for "ml"
        for _ in range(200):
            t.predictions.append({
                "symbol": "BTCUSDT", "model": "ml", "signal": "buy",
                "confidence": 0.8, "timestamp": datetime.utcnow().isoformat(),
                "outcome": "loss", "pnl": -1.0,
            })
        for _ in range(100):
            t.predictions.append({
                "symbol": "BTCUSDT", "model": "ml", "signal": "buy",
                "confidence": 0.8, "timestamp": datetime.utcnow().isoformat(),
                "outcome": "win", "pnl": 1.0,
            })
        t._calculate_win_rates()
        # Last 100 are all wins → 100% win rate for ml
        self.assertAlmostEqual(t.win_rates["ml"], 1.0)


# ============================================================================
# 4. Weight adjustment
# ============================================================================

class TestWeightAdjustment(unittest.TestCase):

    def test_weights_sum_to_1(self):
        t = _make_tracker()
        weights = t.get_adjusted_weights()
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_all_keys_present(self):
        t = _make_tracker()
        weights = t.get_adjusted_weights()
        for key in ("ta", "ml", "tcn"):
            self.assertIn(key, weights)

    def test_ml_highest_base_weight(self):
        """At neutral win rates (0.5) ML has the highest base weight."""
        t = _make_tracker()
        weights = t.get_adjusted_weights()
        self.assertGreater(weights["ml"], weights["tcn"])
        self.assertGreater(weights["ml"], weights["ta"])

    def test_high_ta_win_rate_boosts_ta_weight(self):
        t_neutral = _make_tracker()
        w_neutral = t_neutral.get_adjusted_weights()

        t_high = _make_tracker()
        t_high.win_rates["ta"] = 1.0
        w_high = t_high.get_adjusted_weights()

        self.assertGreater(w_high["ta"], w_neutral["ta"])

    def test_low_win_rate_reduces_weight(self):
        t_neutral = _make_tracker()
        w_neutral = t_neutral.get_adjusted_weights()

        t_low = _make_tracker()
        t_low.win_rates["ml"] = 0.0
        w_low = t_low.get_adjusted_weights()

        self.assertLess(w_low["ml"], w_neutral["ml"])

    def test_weights_always_positive(self):
        t = _make_tracker()
        t.win_rates = {"ta": 0.0, "ml": 0.0, "tcn": 0.0}
        weights = t.get_adjusted_weights()
        for model, w in weights.items():
            self.assertGreater(w, 0, f"{model} weight should be > 0")


# ============================================================================
# 5. Save / Load
# ============================================================================

class TestSaveLoad(unittest.TestCase):

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            t1 = _make_tracker(tmp_path=path)
            _add_outcomes(t1, "ml", n_wins=10, n_losses=5)
            t1._calculate_win_rates()
            t1.save()

            t2 = _make_tracker(tmp_path=path)
            t2.load()
            self.assertAlmostEqual(t2.win_rates["ml"], t1.win_rates["ml"], places=4)
        finally:
            os.unlink(path)

    def test_load_from_nonexistent_file_does_not_crash(self):
        t = _make_tracker(tmp_path="/nonexistent/path/tracker.json")
        try:
            t.load()
        except Exception as e:
            self.fail(f"load() raised {e} on missing file")

    def test_save_does_not_exceed_200_predictions(self):
        """save() writes only the last 200 predictions."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                        mode="w") as f:
            path = f.name
        try:
            t = _make_tracker(tmp_path=path)
            for i in range(300):
                t.predictions.append({
                    "symbol": "BTCUSDT", "model": "ml", "signal": "buy",
                    "confidence": 0.8,
                    "timestamp": datetime.utcnow().isoformat(),
                    "outcome": "win", "pnl": 1.0,
                })
            t.save()
            with open(path) as f2:
                data = json.load(f2)
            self.assertLessEqual(len(data["predictions"]), 200)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
