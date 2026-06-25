"""
Consensus Ensemble System — V5 (ML-Gated Architecture)

ARCHITECTURE CHANGE (V5):
    Previously the consensus system allowed TA alone (Tier 1 / Tier 4) to open
    positions. This was the primary source of false positives because rule-based
    heuristics are fragile in non-stationary crypto markets.

    V5 Consensus Rules:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  GATING RULE (hard gate, must pass before any tier check):              │
    │    ml_conf >= ML_GATE_THRESHOLD (0.58)                                  │
    │    ta_signal.blocked == False                                           │
    │                                                                         │
    │  TIER 3 — Full Conviction (all 3 models agree):                         │
    │    ML >= STRONG (0.62) + TCN >= STRONG (0.62) + TA structural >= 0.55  │
    │    Position: Kelly-sized (fractional), max 3.5%                        │
    │    Take Profit: 8%                                                      │
    │                                                                         │
    │  TIER 2 — ML + TCN Consensus:                                           │
    │    ML >= STRONG (0.62) + TCN >= STRONG (0.62)                          │
    │    Position: Kelly-sized, max 2.5%                                     │
    │    Take Profit: 6%                                                      │
    │                                                                         │
    │  TIER 1 — ML Only (high confidence):                                    │
    │    ML >= HIGH (0.70), TCN direction not opposing                        │
    │    Position: Kelly-sized, max 2.0%                                     │
    │    Take Profit: 6%                                                      │
    │                                                                         │
    │  SELL — ML + TA both bearish (for position exit guidance):              │
    │    ml_conf < 0.42 (sell territory) AND TA not blocked AND bearish      │
    │                                                                         │
    │  HOLD — everything else                                                 │
    └─────────────────────────────────────────────────────────────────────────┘

    KEY REMOVALS vs V4:
      - Tier 4 "disagreement override" REMOVED (single-model trades too noisy)
      - TA standalone Tier 1 REMOVED (TA is a filter and feature, not a trader)
      - TCN standalone entry REMOVED (TCN is a confirmation filter only)

    Kelly Criterion Position Sizing:
        f* = p * (TP/SL) - (1-p)
             ─────────────────────
                  TP/SL
        Applied with a 0.15 fractional multiplier and capped per tier.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pickle
import json

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

from config import settings, get_position_size, get_take_profit
from src.utils import logger, log_signal
from src.models.ta_analyzer import ta_analyzer, TASignal
from src.models.ml_model import ml_model, MLSignal
from src.models.tcn_model import tcn_model, TCNSignal
from src.data import collector


# ============================================================================
# Constants
# ============================================================================

# ML hard gate — no trade without this ML confidence
ML_GATE_THRESHOLD = 0.58

# Tier thresholds
ML_HIGH      = 0.70   # Tier 1: ML alone, very high confidence
ML_STRONG    = 0.62   # Tier 2/3: standard strong signal
TCN_STRONG   = 0.62   # TCN must also be this confident for Tier 2/3
TA_STRUCT    = 0.55   # TA structural score for Tier 3 upgrade

# Fractional Kelly multiplier — prevents over-sizing
KELLY_FRACTION = 0.15

# Hard position size caps per tier
TIER1_MAX = 0.020   # 2.0%
TIER2_MAX = 0.025   # 2.5%
TIER3_MAX = 0.035   # 3.5%

# Exit parameters
STOP_LOSS_PCT          = 0.03   # 3% hard stop
TAKE_PROFIT_STANDARD   = 0.06   # 6% standard
TAKE_PROFIT_HIGH       = 0.08   # 8% Tier 3


# ============================================================================
# Model Performance Tracker (unchanged from V4)
# ============================================================================

class ModelPerformanceTracker:
    """
    Track individual model performance (TA, ML, TCN) for data-driven weight
    adjustment. Records predictions and outcomes; calculates rolling win rates
    (last 100 predictions); suggests weight adjustments.
    """

    def __init__(self, save_path: str = None):
        self.save_path = save_path or str(
            Path(__file__).parent.parent.parent / "models" / "performance_tracker.json"
        )
        self.predictions: List[Dict] = []
        self.win_rates = {"ta": 0.5, "ml": 0.5, "tcn": 0.5}
        self.load()

    def record_prediction(self, symbol: str, model: str, signal: str, confidence: float):
        self.predictions.append({
            "symbol":     symbol,
            "model":      model.lower(),
            "signal":     signal,
            "confidence": confidence,
            "timestamp":  datetime.utcnow().isoformat(),
            "outcome":    None,
            "pnl":        None,
        })
        self._trim_predictions()
        self.save()

    def record_outcome(self, symbol: str, outcome: str, pnl: float = 0.0, lookback_hours: int = 8):
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        for pred in self.predictions:
            if (pred["symbol"] == symbol and pred["outcome"] is None
                    and datetime.fromisoformat(pred["timestamp"]) > cutoff):
                pred["outcome"] = outcome
                pred["pnl"]     = pnl
        self._calculate_win_rates()
        self.save()

    def _calculate_win_rates(self, window: int = 100):
        for model in ["ta", "ml", "tcn"]:
            recent = [p for p in self.predictions[-window:]
                      if p["model"] == model and p["outcome"] is not None]
            if len(recent) >= 10:
                wins = sum(1 for p in recent if p["outcome"] == "win")
                self.win_rates[model] = wins / len(recent)
            else:
                self.win_rates[model] = 0.5

    def get_adjusted_weights(self) -> Dict[str, float]:
        """
        Returns performance-adjusted model weights normalised to 1.0.
        V5 base: ML=0.55, TCN=0.30, TA=0.15 (reflecting ML primacy).
        """
        base = {"ta": 0.15, "ml": 0.55, "tcn": 0.30}
        adjusted = {}
        for model, base_weight in base.items():
            perf = (self.win_rates[model] - 0.5) * 0.4   # ±20% max
            adjusted[model] = base_weight * (1 + perf)
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}

    def _trim_predictions(self, max_size: int = 500):
        if len(self.predictions) > max_size:
            self.predictions = self.predictions[-max_size:]

    def save(self):
        try:
            data = {"predictions": self.predictions[-200:], "win_rates": self.win_rates}
            with open(self.save_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save performance tracker: {e}")

    def load(self):
        try:
            if Path(self.save_path).exists():
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                    self.predictions = data.get("predictions", [])
                    self.win_rates   = data.get("win_rates", {"ta": 0.5, "ml": 0.5, "tcn": 0.5})
        except Exception as e:
            logger.warning(f"Failed to load performance tracker: {e}")


# ============================================================================
# Confidence Calibrator (unchanged from V4)
# ============================================================================

class ConfidenceCalibrator:
    """
    Maps raw model confidence → actual probability of success using
    Isotonic Regression. Calibrated probabilities feed Kelly sizing.
    """

    def __init__(self, save_path: str = None):
        self.save_path = save_path or str(
            Path(__file__).parent.parent.parent / "models" / "calibrators.pkl"
        )
        self.calibrators = {"ta": None, "ml": None, "tcn": None}
        self.is_fitted   = {"ta": False, "ml": False, "tcn": False}
        self.load()

    def fit(self, model: str, confidences: List[float], outcomes: List[int]):
        if len(confidences) < 20:
            logger.warning(f"Not enough data to calibrate {model} ({len(confidences)} samples)")
            return
        try:
            from sklearn.isotonic import IsotonicRegression
            cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            cal.fit(confidences, outcomes)
            self.calibrators[model] = cal
            self.is_fitted[model]   = True
            self.save()
            logger.info(f"Calibrated {model} with {len(confidences)} samples")
        except Exception as e:
            logger.warning(f"Calibration failed for {model}: {e}")

    def calibrate(self, model: str, raw_confidence: float) -> float:
        if not self.is_fitted.get(model):
            return raw_confidence
        try:
            cal = self.calibrators.get(model)
            return float(cal.predict([raw_confidence])[0]) if cal else raw_confidence
        except Exception:
            return raw_confidence

    def save(self):
        try:
            with open(self.save_path, "wb") as f:
                pickle.dump({"calibrators": self.calibrators, "is_fitted": self.is_fitted}, f)
        except Exception as e:
            logger.warning(f"Failed to save calibrators: {e}")

    def load(self):
        try:
            if Path(self.save_path).exists():
                with open(self.save_path, "rb") as f:
                    data = pickle.load(f)
                    self.calibrators = data.get("calibrators", {})
                    self.is_fitted   = data.get("is_fitted",   {})
        except Exception as e:
            logger.warning(f"Failed to load calibrators: {e}")


# ============================================================================
# Market Regime Detector (improved thresholds)
# ============================================================================

class MarketRegimeDetector:
    """
    Detect market regime (BULL / BEAR / SIDEWAYS) using BTC 4h data.
    Adjusts consensus thresholds and position multipliers per regime.
    """

    REGIME_THRESHOLDS = {
        "BULL": {
            "ml_strong":          0.60,   # Relaxed in clear trend
            "tcn_strong":         0.60,
            "ml_high":            0.68,
            "position_multiplier": 1.15,  # Slightly larger in bull
        },
        "BEAR": {
            "ml_strong":          0.67,   # Much stricter
            "tcn_strong":         0.67,
            "ml_high":            0.75,
            "position_multiplier": 0.60,  # Smaller positions in bear
        },
        "SIDEWAYS": {
            "ml_strong":          0.62,
            "tcn_strong":         0.62,
            "ml_high":            0.70,
            "position_multiplier": 1.00,
        },
    }

    def __init__(self):
        self.current_regime = "SIDEWAYS"
        self.regime_history: List[Dict] = []
        self._last_check     = None
        self._cache_minutes  = 15

    def detect_regime(self, force_refresh: bool = False) -> str:
        if not force_refresh and self._last_check:
            if datetime.utcnow() - self._last_check < timedelta(minutes=self._cache_minutes):
                return self.current_regime
        try:
            df = collector.get_dataframe("BTCUSDT", timeframe="4h", limit=50)
            if df.empty or len(df) < 30:
                return self.current_regime

            close, high, low = df["close"], df["high"], df["low"]
            ema_50       = close.ewm(span=50, adjust=False).mean()
            price_vs_ema = (close.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
            recent_ret   = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]

            tr       = np.maximum(high - low, np.maximum(
                np.abs(high - close.shift(1)), np.abs(low - close.shift(1))
            ))
            atr      = tr.rolling(14).mean()
            up_move  = high - high.shift(1)
            down_move = low.shift(1) - low
            plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            plus_di  = 100 * np.convolve(plus_dm,  np.ones(14) / 14, mode="valid")[-1] / (atr.iloc[-1] + 1e-10)
            minus_di = 100 * np.convolve(minus_dm, np.ones(14) / 14, mode="valid")[-1] / (atr.iloc[-1] + 1e-10)
            adx      = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

            if price_vs_ema > 0.02 and recent_ret > 0.03 and adx > 20:
                regime = "BULL"
            elif price_vs_ema < -0.02 and recent_ret < -0.03 and adx > 20:
                regime = "BEAR"
            elif adx < 20 or abs(price_vs_ema) < 0.01:
                regime = "SIDEWAYS"
            else:
                regime = self.current_regime

            self.current_regime = regime
            self._last_check    = datetime.utcnow()

            if not self.regime_history or self.regime_history[-1]["regime"] != regime:
                self.regime_history.append({
                    "regime":        regime,
                    "timestamp":     datetime.utcnow().isoformat(),
                    "price_vs_ema":  float(price_vs_ema),
                    "adx":           float(adx),
                })
                logger.info(f"Market regime: {regime} (EMA: {price_vs_ema:.1%}, ADX: {adx:.0f})")

            return regime
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return self.current_regime

    def get_thresholds(self, regime: str = None) -> Dict:
        regime = regime or self.current_regime
        return self.REGIME_THRESHOLDS.get(regime, self.REGIME_THRESHOLDS["SIDEWAYS"])


# ============================================================================
# Kelly Position Sizer
# ============================================================================

def kelly_position_size(
    win_prob: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    kelly_fraction: float = KELLY_FRACTION,
    tier_cap: float = TIER2_MAX,
) -> float:
    """
    Fractional Kelly criterion position size.

    Args:
        win_prob       : Calibrated probability of win (0–1).
        take_profit_pct: Potential gain (e.g. 0.06 for 6%).
        stop_loss_pct  : Maximum loss (e.g. 0.03 for 3%).
        kelly_fraction : Fractional multiplier (default 0.15).
        tier_cap       : Maximum allowed position as fraction of portfolio.

    Returns:
        Position size as a fraction of portfolio (capped at tier_cap).
    """
    if stop_loss_pct <= 0 or win_prob <= 0:
        return 0.0
    b    = take_profit_pct / stop_loss_pct   # odds ratio
    f_star = (win_prob * b - (1 - win_prob)) / b
    f_star = max(f_star, 0.0)               # never go negative
    sized  = f_star * kelly_fraction
    return min(sized, tier_cap)             # respect tier cap


# ============================================================================
# Consensus Signal Output
# ============================================================================

@dataclass
class ConsensusSignal:
    """Combined signal from all models."""
    signal:           str    # 'buy', 'hold', 'sell'
    tier:             int    # 1–3 (0 = no signal)
    confidence:       float  # weighted average confidence
    position_size_pct: float  # recommended position size (fraction of portfolio)
    take_profit_pct:  float
    stop_loss_pct:    float

    ta_signal:  TASignal  = None
    ml_signal:  MLSignal  = None
    tcn_signal: TCNSignal = None

    reasons:   List[str] = field(default_factory=list)
    timestamp: datetime  = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            "signal":            self.signal,
            "tier":              self.tier,
            "confidence":        self.confidence,
            "position_size_pct": self.position_size_pct,
            "take_profit_pct":   self.take_profit_pct,
            "stop_loss_pct":     self.stop_loss_pct,
            "ta_confidence":     self.ta_signal.confidence  if self.ta_signal  else 0,
            "ml_confidence":     self.ml_signal.confidence  if self.ml_signal  else 0,
            "tcn_confidence":    self.tcn_signal.confidence if self.tcn_signal else 0,
            "ta_blocked":        self.ta_signal.blocked     if self.ta_signal  else False,
            "reasons":           self.reasons,
            "timestamp":         self.timestamp.isoformat(),
        }


# ============================================================================
# Consensus Ensemble — V5
# ============================================================================

class ConsensusEnsemble:
    """
    V5 ML-Gated Consensus Ensemble.

    Decision hierarchy (all checks in order):
      1. TA structural filter — if ta_signal.blocked, skip coin entirely.
      2. ML hard gate — ml_conf must be >= ML_GATE_THRESHOLD (0.58).
      3. Tier classification based on ML + TCN confidence levels.
      4. Fractional Kelly position sizing using calibrated win probability.
      5. Regime-based threshold adjustments and position multipliers.

    Model weights (V5 — reflects ML primacy):
      ML:  55%  (primary decision maker)
      TCN: 30%  (sequential confirmation filter)
      TA:  15%  (structural feature / block-only)
    """

    def __init__(self):
        self.ta  = ta_analyzer
        self.ml  = ml_model
        self.tcn = tcn_model

        self.performance_tracker = ModelPerformanceTracker()
        self.calibrator          = ConfidenceCalibrator()
        self.regime_detector     = MarketRegimeDetector()

        # V5 weights — ML is the primary predictor
        self.ml_weight  = 0.55
        self.tcn_weight = 0.30
        self.ta_weight  = 0.15

        self.latest_scan: List[Dict] = []
        self._update_weights_from_performance()

    def _update_weights_from_performance(self):
        """Adjust weights based on rolling performance of each model."""
        try:
            perf_weights = self.performance_tracker.get_adjusted_weights()
            self.ta_weight  = perf_weights.get("ta",  0.15)
            self.ml_weight  = perf_weights.get("ml",  0.55)
            self.tcn_weight = perf_weights.get("tcn", 0.30)
        except Exception:
            pass   # Keep defaults on any failure

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def analyze(self, symbol: str) -> ConsensusSignal:
        """
        Run full consensus analysis on a symbol.

        Returns a ConsensusSignal with signal='buy' only when all V5
        gating rules pass and at least one tier criterion is met.
        """
        try:
            regime      = self.regime_detector.detect_regime()
            thresholds  = self.regime_detector.get_thresholds(regime)

            # --- Step 1: Get predictions from all three models ---
            ta_result  = self.ta.analyze(symbol)
            ml_result  = self.ml.predict(symbol)
            tcn_result = self.tcn.predict(symbol)

            # Calibrated confidences
            ta_conf  = self.calibrator.calibrate("ta",  ta_result.confidence)
            ml_conf  = self.calibrator.calibrate("ml",  ml_result.confidence)
            tcn_conf = self.calibrator.calibrate("tcn", tcn_result.confidence)

            # Record for tracker
            self.performance_tracker.record_prediction(symbol, "ta",  ta_result.signal,  ta_conf)
            self.performance_tracker.record_prediction(symbol, "ml",  ml_result.signal,  ml_conf)
            self.performance_tracker.record_prediction(symbol, "tcn", tcn_result.signal, tcn_conf)

            # --- Step 2: Hard gates (early exits) ---
            # Gate A: TA structural filter veto
            if ta_result.blocked:
                return self._hold(
                    symbol, regime, ta_result, ml_result, tcn_result,
                    ta_conf, ml_conf, tcn_conf,
                    reason=f"TA structural VETO: {ta_result.block_reason}"
                )

            # Gate B: ML confidence gate
            if ml_conf < ML_GATE_THRESHOLD or ml_result.signal != "buy":
                return self._hold(
                    symbol, regime, ta_result, ml_result, tcn_result,
                    ta_conf, ml_conf, tcn_conf,
                    reason=f"ML gate failed (conf={ml_conf:.2f}, signal={ml_result.signal})"
                )

            # --- Step 3: Tier classification ---
            regime_ml_strong  = thresholds["ml_strong"]
            regime_tcn_strong = thresholds["tcn_strong"]
            regime_ml_high    = thresholds["ml_high"]
            pos_mult          = thresholds["position_multiplier"]

            tier, take_profit = self._classify_tier(
                ml_conf, tcn_conf, ta_conf,
                ml_result.signal, tcn_result.signal,
                regime_ml_strong, regime_tcn_strong, regime_ml_high,
            )

            if tier == 0:
                return self._hold(
                    symbol, regime, ta_result, ml_result, tcn_result,
                    ta_conf, ml_conf, tcn_conf,
                    reason=f"No tier matched (ML={ml_conf:.2f}, TCN={tcn_conf:.2f})"
                )

            # --- Step 4: Fractional Kelly position sizing ---
            tier_cap = {1: TIER1_MAX, 2: TIER2_MAX, 3: TIER3_MAX}.get(tier, TIER2_MAX)
            raw_size = kelly_position_size(
                win_prob=ml_conf,
                take_profit_pct=take_profit,
                stop_loss_pct=STOP_LOSS_PCT,
                kelly_fraction=KELLY_FRACTION,
                tier_cap=tier_cap,
            )
            position_size = raw_size * pos_mult    # Regime multiplier
            position_size = min(position_size, tier_cap)   # Never exceed cap

            # --- Step 5: Weighted confidence ---
            weighted_conf = (
                self.ml_weight  * ml_conf +
                self.tcn_weight * tcn_conf +
                self.ta_weight  * ta_conf
            )

            reasons = self._build_reasons(
                tier, regime, ta_result, ml_result, tcn_result,
                ta_conf, ml_conf, tcn_conf
            )

            result = ConsensusSignal(
                signal="buy",
                tier=tier,
                confidence=weighted_conf,
                position_size_pct=position_size,
                take_profit_pct=take_profit,
                stop_loss_pct=STOP_LOSS_PCT,
                ta_signal=ta_result,
                ml_signal=ml_result,
                tcn_signal=tcn_result,
                reasons=reasons,
            )

            log_signal(
                symbol, ta_conf, ml_conf, tcn_conf, "buy", tier,
                ta_signal=ta_result.signal, ml_signal=ml_result.signal,
                tcn_signal=tcn_result.signal, reasons=reasons,
            )
            return result

        except Exception as e:
            logger.error(f"Consensus analysis error for {symbol}: {e}")
            return ConsensusSignal(
                signal="hold", tier=0, confidence=0.0,
                position_size_pct=0, take_profit_pct=0.06, stop_loss_pct=0.03,
                reasons=[f"Error: {str(e)}"],
            )

    # ------------------------------------------------------------------
    # Tier classification
    # ------------------------------------------------------------------
    def _classify_tier(
        self,
        ml_conf:  float, tcn_conf: float, ta_conf: float,
        ml_signal: str,  tcn_signal: str,
        ml_strong: float, tcn_strong: float, ml_high: float,
    ) -> Tuple[int, float]:
        """
        Classify signal tier using V5 rules.

        Returns (tier, take_profit_pct). tier=0 means no signal.
        """
        tcn_bullish = tcn_signal == "buy" or tcn_conf >= tcn_strong

        # Tier 3: All models agree at strong threshold
        if (ml_conf >= ml_strong and tcn_conf >= tcn_strong
                and ta_conf >= TA_STRUCT and tcn_bullish):
            return 3, TAKE_PROFIT_HIGH

        # Tier 2: ML + TCN both strong
        if ml_conf >= ml_strong and tcn_conf >= tcn_strong and tcn_bullish:
            return 2, TAKE_PROFIT_STANDARD

        # Tier 1: ML alone very high confidence, TCN not opposing
        tcn_opposing = tcn_signal == "sell" and tcn_conf > 0.60
        if ml_conf >= ml_high and not tcn_opposing:
            return 1, TAKE_PROFIT_STANDARD

        return 0, TAKE_PROFIT_STANDARD

    # ------------------------------------------------------------------
    # Exit signal (for position monitoring)
    # ------------------------------------------------------------------
    def get_exit_signal(self, symbol: str, entry_price: float, current_price: float) -> Tuple[str, str]:
        """
        Check whether an open position should be closed by models.
        Called by executor during position monitoring.
        """
        try:
            result  = self.analyze(symbol)
            pnl_pct = (current_price - entry_price) / entry_price

            # Both ML and TA turned bearish
            ml_bearish  = result.ml_signal  and result.ml_signal.signal  == "sell"
            ta_bearish  = result.ta_signal  and result.ta_signal.blocked

            if ml_bearish and ta_bearish:
                return "exit", "ML + TA structural both bearish"

            # Confidence collapsed while in profit
            if pnl_pct > 0.03 and result.confidence < 0.38:
                return "exit", "Confidence collapsed while in profit"

            return "hold", ""
        except Exception as e:
            return "hold", f"Exit check error: {e}"

    # ------------------------------------------------------------------
    # Batch scanner
    # ------------------------------------------------------------------
    def scan_for_signals(self, symbols: List[str] = None) -> List[Tuple[str, ConsensusSignal]]:
        """
        Scan a list of symbols and return buy signals sorted by tier then confidence.
        """
        from config import WATCHLIST
        symbols = symbols or WATCHLIST

        signals      = []
        scan_results = []

        for symbol in symbols:
            try:
                result = self.analyze(symbol)
                scan_results.append({
                    "symbol":      symbol,
                    "signal":      result.signal,
                    "tier":        result.tier,
                    "confidence":  result.confidence,
                    "ta_conf":     result.ta_signal.confidence  if result.ta_signal  else 0,
                    "ml_conf":     result.ml_signal.confidence  if result.ml_signal  else 0,
                    "tcn_conf":    result.tcn_signal.confidence if result.tcn_signal else 0,
                    "ta_blocked":  result.ta_signal.blocked     if result.ta_signal  else False,
                    "reasons":     result.reasons,
                })

                # Persist to DB for accuracy tracking
                try:
                    from src.data.database import db
                    if result.ta_signal:
                        db.save_prediction(symbol, "ta",  result.ta_signal.signal,  result.ta_signal.confidence)
                    if result.ml_signal:
                        db.save_prediction(symbol, "ml",  result.ml_signal.signal,  result.ml_signal.confidence)
                    if result.tcn_signal:
                        db.save_prediction(symbol, "tcn", result.tcn_signal.signal, result.tcn_signal.confidence)
                except Exception:
                    pass  # Never block trading on DB write failure

                if result.signal == "buy" and result.tier > 0:
                    signals.append((symbol, result))
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        self.latest_scan = scan_results

        # Sort by tier (desc) then confidence (desc)
        signals.sort(key=lambda x: (x[1].tier, x[1].confidence), reverse=True)
        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _hold(
        self, symbol, regime, ta_result, ml_result, tcn_result,
        ta_conf, ml_conf, tcn_conf, reason: str
    ) -> ConsensusSignal:
        """Return a standardised HOLD signal with reason."""
        log_signal(
            symbol, ta_conf, ml_conf, tcn_conf, "hold", 0,
            ta_signal=ta_result.signal, ml_signal=ml_result.signal,
            tcn_signal=tcn_result.signal if tcn_result else None,
            reasons=[reason],
        )
        return ConsensusSignal(
            signal="hold", tier=0, confidence=0.0,
            position_size_pct=0,
            take_profit_pct=TAKE_PROFIT_STANDARD,
            stop_loss_pct=STOP_LOSS_PCT,
            ta_signal=ta_result, ml_signal=ml_result, tcn_signal=tcn_result,
            reasons=[f"Regime: {regime}", reason],
        )

    def _build_reasons(
        self, tier, regime,
        ta_result, ml_result, tcn_result,
        ta_conf, ml_conf, tcn_conf
    ) -> List[str]:
        tier_labels = {
            1: "Tier 1 — ML High Confidence (TCN not opposing)",
            2: "Tier 2 — ML + TCN Strong Consensus",
            3: "Tier 3 — Full Conviction (ML + TCN + TA structural)",
        }
        reasons = [
            f"Regime: {regime}",
            tier_labels.get(tier, f"Tier {tier}"),
            f"ML: {ml_conf:.0%}  TCN: {tcn_conf:.0%}  TA struct: {ta_conf:.0%}",
        ]
        if ta_result.reasons:
            reasons.extend(ta_result.reasons[:2])
        if ml_result.top_features:
            reasons.append(f"Top feature: {ml_result.top_features[0][0]}")
        if tcn_result and tcn_result.pattern:
            reasons.append(f"TCN pattern: {tcn_result.pattern}")
        return reasons


# ============================================================================
# Global singleton
# ============================================================================
ensemble = ConsensusEnsemble()
