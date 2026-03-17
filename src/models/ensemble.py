"""
Flexible Consensus Ensemble System

Combines TA, XGBoost, and TCN predictions using a smart tiered consensus approach
that adapts to different market situations.

Includes:
- Model Performance Tracking (win rates per model)
- Confidence Calibration (isotonic regression)
- Market Regime Detection (bull/bear/sideways)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pickle
import json

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 4)[0])

from config import settings, get_position_size, get_take_profit
from src.utils import logger, log_signal
from src.models.ta_analyzer import ta_analyzer, TASignal
from src.models.ml_model import ml_model, MLSignal
from src.models.tcn_model import tcn_model, TCNSignal
from src.data import collector


# ============================================================================
# Model Performance Tracking
# ============================================================================

class ModelPerformanceTracker:
    """
    Track individual model performance (TA, ML, TCN) for data-driven weight adjustment.
    
    Features:
    - Records predictions and outcomes
    - Calculates rolling win rates (last 100 predictions)
    - Suggests weight adjustments based on performance
    """
    
    def __init__(self, save_path: str = None):
        self.save_path = save_path or str(Path(__file__).parent.parent.parent / "models" / "performance_tracker.json")
        self.predictions: List[Dict] = []
        self.win_rates = {"ta": 0.5, "ml": 0.5, "tcn": 0.5}
        self.load()
    
    def record_prediction(self, symbol: str, model: str, signal: str, confidence: float):
        """Record a model's prediction before outcome is known."""
        self.predictions.append({
            "symbol": symbol,
            "model": model.lower(),
            "signal": signal,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "outcome": None,  # To be filled later
            "pnl": None
        })
        self._trim_predictions()
        self.save()
    
    def record_outcome(self, symbol: str, outcome: str, pnl: float = 0.0, lookback_hours: int = 8):
        """
        Record outcome for recent predictions on a symbol.
        
        Args:
            symbol: Trading pair
            outcome: 'win' or 'loss'
            pnl: Profit/loss percentage
            lookback_hours: How far back to match predictions
        """
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        for pred in self.predictions:
            if (pred["symbol"] == symbol and 
                pred["outcome"] is None and
                datetime.fromisoformat(pred["timestamp"]) > cutoff):
                pred["outcome"] = outcome
                pred["pnl"] = pnl
        
        self._calculate_win_rates()
        self.save()
    
    def _calculate_win_rates(self, window: int = 100):
        """Calculate rolling win rates for each model."""
        for model in ["ta", "ml", "tcn"]:
            recent = [p for p in self.predictions[-window:] 
                     if p["model"] == model and p["outcome"] is not None]
            
            if len(recent) >= 10:  # Need at least 10 samples
                wins = sum(1 for p in recent if p["outcome"] == "win")
                self.win_rates[model] = wins / len(recent)
            else:
                self.win_rates[model] = 0.5  # Default
    
    def get_adjusted_weights(self) -> Dict[str, float]:
        """
        Get adjusted model weights based on recent performance.
        
        Returns weights normalized to sum to 1.0
        """
        # Base weights (updated to match current performance)
        # TA: 66.7%, ML: 55.2%, TCN: 64.7%
        base = {"ta": 0.35, "ml": 0.25, "tcn": 0.40}
        
        # Performance adjustment (±20% based on win rate vs 50%)
        adjusted = {}
        for model, base_weight in base.items():
            performance_factor = (self.win_rates[model] - 0.5) * 0.4  # ±20%
            adjusted[model] = base_weight * (1 + performance_factor)
        
        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}
    
    def _trim_predictions(self, max_size: int = 500):
        """Keep only recent predictions."""
        if len(self.predictions) > max_size:
            self.predictions = self.predictions[-max_size:]
    
    def save(self):
        """Save tracker state to file."""
        try:
            data = {
                "predictions": self.predictions[-200:],  # Last 200 only
                "win_rates": self.win_rates
            }
            with open(self.save_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save performance tracker: {e}")
    
    def load(self):
        """Load tracker state from file."""
        try:
            if Path(self.save_path).exists():
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get("predictions", [])
                    self.win_rates = data.get("win_rates", {"ta": 0.5, "ml": 0.5, "tcn": 0.5})
        except Exception as e:
            logger.warning(f"Failed to load performance tracker: {e}")


# ============================================================================
# Confidence Calibration
# ============================================================================

class ConfidenceCalibrator:
    """
    Calibrate model confidence using isotonic regression.
    
    Maps raw confidence scores to actual probability of success.
    """
    
    def __init__(self, save_path: str = None):
        self.save_path = save_path or str(Path(__file__).parent.parent.parent / "models" / "calibrators.pkl")
        self.calibrators = {"ta": None, "ml": None, "tcn": None}
        self.is_fitted = {"ta": False, "ml": False, "tcn": False}
        self.load()
    
    def fit(self, model: str, confidences: List[float], outcomes: List[int]):
        """
        Fit calibrator for a model using historical data.
        
        Args:
            model: 'ta', 'ml', or 'tcn'
            confidences: List of raw confidence scores
            outcomes: List of binary outcomes (1=win, 0=loss)
        """
        if len(confidences) < 20:
            logger.warning(f"Not enough data to calibrate {model} ({len(confidences)} samples)")
            return
        
        try:
            from sklearn.isotonic import IsotonicRegression
            
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            calibrator.fit(confidences, outcomes)
            
            self.calibrators[model] = calibrator
            self.is_fitted[model] = True
            self.save()
            
            logger.info(f"Calibrated {model} model with {len(confidences)} samples")
        except ImportError:
            logger.warning("sklearn not available for calibration")
        except Exception as e:
            logger.warning(f"Calibration failed for {model}: {e}")
    
    def calibrate(self, model: str, raw_confidence: float) -> float:
        """
        Calibrate a raw confidence score.
        
        Args:
            model: 'ta', 'ml', or 'tcn'
            raw_confidence: Raw confidence from model
        
        Returns:
            Calibrated probability (or raw if not fitted)
        """
        if not self.is_fitted.get(model):
            return raw_confidence
        
        try:
            calibrator = self.calibrators.get(model)
            if calibrator is None:
                return raw_confidence
            
            calibrated = calibrator.predict([raw_confidence])[0]
            return float(calibrated)
        except Exception:
            return raw_confidence
    
    def save(self):
        """Save calibrators to file."""
        try:
            data = {
                "calibrators": self.calibrators,
                "is_fitted": self.is_fitted
            }
            with open(self.save_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save calibrators: {e}")
    
    def load(self):
        """Load calibrators from file."""
        try:
            if Path(self.save_path).exists():
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                    self.calibrators = data.get("calibrators", {})
                    self.is_fitted = data.get("is_fitted", {})
        except Exception as e:
            logger.warning(f"Failed to load calibrators: {e}")


# ============================================================================
# Market Regime Detection
# ============================================================================

class MarketRegimeDetector:
    """
    Detect current market regime (BULL, BEAR, SIDEWAYS) using BTC as benchmark.
    
    Adjusts ensemble thresholds based on market conditions:
    - BULL: Lower thresholds (more signals)
    - BEAR: Higher thresholds (fewer, safer signals)
    - SIDEWAYS: Standard thresholds
    """
    
    # Regime-specific thresholds
    REGIME_THRESHOLDS = {
        "BULL": {
            "high_confidence": 0.75,
            "strong_threshold": 0.55,  # Relaxed from 0.60 to capture trend
            "position_multiplier": 1.2  # 20% larger positions
        },
        "BEAR": {
            "high_confidence": 0.88,
            "strong_threshold": 0.72,
            "position_multiplier": 0.6  # 40% smaller positions
        },
        "SIDEWAYS": {
            "high_confidence": 0.80,
            "strong_threshold": 0.60,
            "position_multiplier": 1.0
        }
    }
    
    def __init__(self):
        self.current_regime = "SIDEWAYS"
        self.regime_history: List[Dict] = []
        self._last_check = None
        self._cache_minutes = 15  # Recheck every 15 minutes
    
    def detect_regime(self, force_refresh: bool = False) -> str:
        """
        Detect current market regime using BTC as benchmark.
        
        Uses:
        - Price vs 50 EMA (trend direction)
        - ADX (trend strength)
        - Recent volatility
        
        Returns:
            'BULL', 'BEAR', or 'SIDEWAYS'
        """
        # Use cached value if recent
        if not force_refresh and self._last_check:
            if datetime.utcnow() - self._last_check < timedelta(minutes=self._cache_minutes):
                return self.current_regime
        
        try:
            # Get BTC 4h data for regime detection
            df = collector.get_dataframe("BTCUSDT", timeframe="4h", limit=50)
            
            if df.empty or len(df) < 30:
                return self.current_regime
            
            # Calculate indicators
            close = df["close"]
            high = df["high"]
            low = df["low"]
            
            # EMA 50
            ema_50 = close.ewm(span=50, adjust=False).mean()
            price_vs_ema = (close.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
            
            # Recent trend (last 10 candles)
            recent_return = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            
            # ADX calculation (simplified)
            tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - close.shift(1)), 
                               np.abs(low - close.shift(1))))
            atr = tr.rolling(14).mean()
            
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_di = 100 * np.array(plus_dm).astype(float)
            plus_di = np.convolve(plus_di, np.ones(14)/14, mode='valid')[-1] / (atr.iloc[-1] + 1e-10)
            
            minus_di = 100 * np.array(minus_dm).astype(float)
            minus_di = np.convolve(minus_di, np.ones(14)/14, mode='valid')[-1] / (atr.iloc[-1] + 1e-10)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = dx  # Simplified - using DX as proxy
            
            # Volatility (ATR/price)
            volatility = atr.iloc[-1] / close.iloc[-1]
            
            # Determine regime
            if price_vs_ema > 0.02 and recent_return > 0.03 and adx > 20:
                regime = "BULL"
            elif price_vs_ema < -0.02 and recent_return < -0.03 and adx > 20:
                regime = "BEAR"
            elif adx < 20 or abs(price_vs_ema) < 0.01:
                regime = "SIDEWAYS"
            else:
                regime = self.current_regime  # Keep previous
            
            # Update state
            self.current_regime = regime
            self._last_check = datetime.utcnow()
            
            # Log regime change
            if not self.regime_history or self.regime_history[-1]["regime"] != regime:
                self.regime_history.append({
                    "regime": regime,
                    "timestamp": datetime.utcnow().isoformat(),
                    "price_vs_ema": float(price_vs_ema),
                    "adx": float(adx)
                })
                logger.info(f"Market regime: {regime} (EMA: {price_vs_ema:.1%}, ADX: {adx:.0f})")
            
            return regime
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return self.current_regime
    
    def get_adjusted_thresholds(self, regime: str = None) -> Dict:
        """Get thresholds adjusted for current market regime."""
        regime = regime or self.current_regime
        return self.REGIME_THRESHOLDS.get(regime, self.REGIME_THRESHOLDS["SIDEWAYS"])


@dataclass
class ConsensusSignal:
    """Combined signal from all models."""
    signal: str  # 'buy', 'hold', 'sell'
    tier: int  # 1-4 based on consensus rules
    confidence: float  # Weighted average confidence
    position_size_pct: float  # Recommended position size %
    take_profit_pct: float  # Recommended take profit %
    stop_loss_pct: float  # Stop loss % (always 3%)
    
    # Individual model signals
    ta_signal: TASignal = None
    ml_signal: MLSignal = None
    tcn_signal: TCNSignal = None
    
    # Metadata
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "signal": self.signal,
            "tier": self.tier,
            "confidence": self.confidence,
            "position_size_pct": self.position_size_pct,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "ta_confidence": self.ta_signal.confidence if self.ta_signal else 0,
            "ml_confidence": self.ml_signal.confidence if self.ml_signal else 0,
            "tcn_confidence": self.tcn_signal.confidence if self.tcn_signal else 0,
            "reasons": self.reasons,
            "timestamp": self.timestamp.isoformat()
        }


class ConsensusEnsemble:
    """
    Flexible Consensus System (Strategy 3)
    
    Tiered approach for different market situations:
    
    Tier 1 - High Confidence (single model >85%):
        Trade immediately with 2% position
        Example: TA detects explosive volume breakout at 88% → Enter
    
    Tier 2 - Strong Consensus (2/3 models >60%):
        Trade with 2.5% position
        Example: ML=72%, TCN=68%, TA=55% → Enter
    
    Tier 3 - Full Consensus (all 3 models >60%):
        Trade with 3-4% position (highest confidence)
        Example: TA=75%, ML=70%, TCN=68% → Maximum conviction
    
    Tier 4 - Disagreement Override:
        If one model >85% but others neutral (40-60%), still enter with 1.5%
        Catches fast-moving breakouts that slower models haven't recognized
    """
    
    # Consensus thresholds (optimized after TA improvements)
    HIGH_CONFIDENCE = 0.80  # Tier 1 single model
    STRONG_THRESHOLD = 0.60  # Tiers 2 & 3
    NEUTRAL_LOW = 0.30  # Relaxed from 0.35: Allow Tier 4 overrides even if another model is slightly bearish
    NEUTRAL_HIGH = 0.65
    
    # Position sizing (% of portfolio)
    TIER1_SIZE = 0.02  # 2%
    TIER2_SIZE = 0.025  # 2.5%
    TIER3_SIZE = 0.035  # 3.5%
    TIER4_SIZE = 0.015  # 1.5%
    
    # Exit strategy
    STOP_LOSS = 0.03  # 3% hard stop
    TAKE_PROFIT_STANDARD = 0.06  # 6%
    TAKE_PROFIT_HIGH = 0.08  # 8% for Tier 3
    
    def __init__(self):
        """Initialize ensemble with performance tracking, calibration, and regime detection."""
        self.ta = ta_analyzer
        self.ml = ml_model
        self.tcn = tcn_model
        
        # NEW: Performance tracking, calibration, and regime detection
        self.performance_tracker = ModelPerformanceTracker()
        self.calibrator = ConfidenceCalibrator()
        self.regime_detector = MarketRegimeDetector()
        
        # Model weights (updated to match current performance)
        # TA: 66.7% win rate → 35% weight (was 25%)
        # ML: 55.2% win rate → 25% weight (was 40%)
        # TCN: 64.7% win rate → 40% weight (was 35%)
        self.ta_weight = 0.35  # Best win rate - Triple Confirmation
        self.ml_weight = 0.25  # Lowest win rate - XGBoost+CatBoost hybrid
        self.tcn_weight = 0.40  # Second best - Temporal patterns
        
        self.latest_scan = []  # Stores the latest full scan results
        
        # Update weights based on tracked performance
        self._update_weights_from_performance()
    
    def analyze(self, symbol: str) -> ConsensusSignal:
        """
        Analyze a coin using all three models and apply consensus rules.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
        
        Returns:
            ConsensusSignal with tier, signal, and recommendations
        """
        try:
            # Detect market regime and get adjusted thresholds
            regime = self.regime_detector.detect_regime()
            regime_thresholds = self.regime_detector.get_adjusted_thresholds(regime)
            
            # Get predictions from all models
            ta_result = self.ta.analyze(symbol)
            ml_result = self.ml.predict(symbol)
            tcn_result = self.tcn.predict(symbol)
            
            # Extract raw confidences
            ta_conf_raw = ta_result.confidence
            ml_conf_raw = ml_result.confidence
            tcn_conf_raw = tcn_result.confidence
            
            # Apply confidence calibration (if fitted)
            ta_conf = self.calibrator.calibrate("ta", ta_conf_raw)
            ml_conf = self.calibrator.calibrate("ml", ml_conf_raw)
            tcn_conf = self.calibrator.calibrate("tcn", tcn_conf_raw)
            
            # Record predictions for performance tracking
            self.performance_tracker.record_prediction(symbol, "ta", ta_result.signal, ta_conf)
            self.performance_tracker.record_prediction(symbol, "ml", ml_result.signal, ml_conf)
            self.performance_tracker.record_prediction(symbol, "tcn", tcn_result.signal, tcn_conf)
            
            # Apply consensus rules with regime-adjusted thresholds
            tier, signal, position_size = self._apply_consensus_rules(
                ta_conf, ml_conf, tcn_conf,
                ta_result.signal, ml_result.signal, tcn_result.signal,
                high_confidence=regime_thresholds["high_confidence"],
                strong_threshold=regime_thresholds["strong_threshold"]
            )
            
            # Apply regime position multiplier
            position_size *= regime_thresholds.get("position_multiplier", 1.0)
            
            # Determine take profit
            take_profit = self.TAKE_PROFIT_HIGH if tier == 3 else self.TAKE_PROFIT_STANDARD
            
            # Calculate weighted confidence
            if signal == "buy":
                confidence = self._weighted_confidence(ta_conf, ml_conf, tcn_conf)
            else:
                confidence = 1 - self._weighted_confidence(ta_conf, ml_conf, tcn_conf)
            
            # Generate reasons (include regime info)
            reasons = self._generate_reasons(
                tier, ta_result, ml_result, tcn_result,
                ta_conf, ml_conf, tcn_conf
            )
            reasons.insert(0, f"Regime: {regime}")
            
            result = ConsensusSignal(
                signal=signal,
                tier=tier,
                confidence=confidence,
                position_size_pct=position_size,
                take_profit_pct=take_profit,
                stop_loss_pct=self.STOP_LOSS,
                ta_signal=ta_result,
                ml_signal=ml_result,
                tcn_signal=tcn_result,
                reasons=reasons
            )
            
            # Log the signal with full details
            log_signal(
                symbol, ta_conf, ml_conf, tcn_conf, signal, tier,
                ta_signal=ta_result.signal if ta_result else None,
                ml_signal=ml_result.signal if ml_result else None,
                tcn_signal=tcn_result.signal if tcn_result else None,
                reasons=reasons
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Consensus analysis error for {symbol}: {e}")
            return ConsensusSignal(
                signal="hold",
                tier=0,
                confidence=0.0,
                position_size_pct=0,
                take_profit_pct=0.06,
                stop_loss_pct=0.03,
                reasons=[f"Error: {str(e)}"]
            )
    
    def _apply_consensus_rules(
        self,
        ta_conf: float,
        ml_conf: float,
        tcn_conf: float,
        ta_signal: str,
        ml_signal: str,
        tcn_signal: str,
        high_confidence: float = None,
        strong_threshold: float = None
    ) -> Tuple[int, str, float]:
        """
        Apply tiered consensus rules with optional regime-adjusted thresholds.
        
        Priority Order (fixed):
        1. Tier 3: Full Consensus (all 3 models >60%) - Highest conviction
        2. Tier 2: Strong Consensus (2/3 models >60%)
        3. Tier 1: High Confidence (single model >80%)
        4. Tier 4: Disagreement Override (one >80%, others neutral)
        
        Returns:
            Tuple of (tier, signal, position_size)
        """
        # Use regime-adjusted thresholds if provided, otherwise use defaults
        high_conf = high_confidence if high_confidence is not None else self.HIGH_CONFIDENCE
        strong_thresh = strong_threshold if strong_threshold is not None else self.STRONG_THRESHOLD
        
        confidences = [ta_conf, ml_conf, tcn_conf]
        signals = [ta_signal, ml_signal, tcn_signal]
        buy_signals = sum(1 for s in signals if s == "buy")
        
        # PRIORITY 1: Check for Tier 3: Full Consensus (all 3 models >threshold)
        all_above_threshold = all(c > strong_thresh for c in confidences)
        if all_above_threshold and buy_signals == 3:
            return (3, "buy", self.TIER3_SIZE)
        
        # PRIORITY 2: Check for Tier 2: Strong Consensus (2/3 models >threshold)
        above_threshold = sum(1 for c in confidences if c > strong_thresh)
        if above_threshold >= 2 and buy_signals >= 2:
            return (2, "buy", self.TIER2_SIZE)
        
        # PRIORITY 3: Check for Tier 1: High Confidence (any single model >threshold)
        if any(c > high_conf for c in confidences):
            # Find which model has high confidence
            max_conf = max(confidences)
            max_idx = confidences.index(max_conf)
            
            if signals[max_idx] == "buy":
                # SIDEWAYS guard: don't fire Tier 1 if the other two models are both 'sell'
                other_signals = [s for j, s in enumerate(signals) if j != max_idx]
                both_others_sell = all(s == "sell" for s in other_signals)
                if not both_others_sell:
                    return (1, "buy", self.TIER1_SIZE)
        
        # PRIORITY 4: Check for Tier 4: Disagreement Override
        # One model >80% but others in neutral range (35-65%)
        for i, conf in enumerate(confidences):
            if conf > high_conf and signals[i] == "buy":
                other_confs = [c for j, c in enumerate(confidences) if j != i]
                others_neutral = all(
                    self.NEUTRAL_LOW <= c <= self.NEUTRAL_HIGH
                    for c in other_confs
                )
                if others_neutral:
                    return (4, "buy", self.TIER4_SIZE)
        
        # No buy signal - check for sell signals
        sell_signals = sum(1 for s in signals if s == "sell")
        if sell_signals >= 2:
            return (0, "sell", 0)
        
        # Default to hold
        return (0, "hold", 0)
    
    def _weighted_confidence(
        self,
        ta_conf: float,
        ml_conf: float,
        tcn_conf: float
    ) -> float:
        """
        Calculate weighted average confidence.
        
        Weights (adjusted for conservative TA):
        - TA: 25% (conservative filter after improvements)
        - ML: 40% (main predictor, pattern recognition)
        - TCN: 35% (sequential/temporal patterns)
        """
        weights = [self.ta_weight, self.ml_weight, self.tcn_weight]
        confidences = [ta_conf, ml_conf, tcn_conf]
        return sum(w * c for w, c in zip(weights, confidences))
    
    def _generate_reasons(
        self,
        tier: int,
        ta_result: TASignal,
        ml_result: MLSignal,
        tcn_result: TCNSignal,
        ta_conf: float,
        ml_conf: float,
        tcn_conf: float
    ) -> List[str]:
        """Generate human-readable reasons for the signal."""
        reasons = []
        
        tier_names = {
            1: "High Confidence (single model >85%)",
            2: "Strong Consensus (2/3 models >60%)",
            3: "Full Consensus (all models >60%)",
            4: "Disagreement Override (1 strong + neutrals)"
        }
        
        if tier > 0:
            reasons.append(f"Tier {tier}: {tier_names.get(tier, 'Unknown')}")
        
        # Add model-specific reasons
        reasons.append(f"TA: {ta_conf:.0%} ({ta_result.signal})")
        reasons.append(f"ML: {ml_conf:.0%} ({ml_result.signal})")
        reasons.append(f"TCN: {tcn_conf:.0%} ({tcn_result.signal})")
        
        # Add TA-specific reasons
        if ta_result.reasons:
            reasons.extend(ta_result.reasons[:2])
        
        # Add ML top features
        if ml_result.top_features:
            top_feat = ml_result.top_features[0]
            reasons.append(f"Top feature: {top_feat[0]}")
        
        # Add TCN pattern
        if tcn_result.pattern:
            reasons.append(f"Pattern: {tcn_result.pattern}")
        
        return reasons
    
    def scan_for_signals(self, symbols: List[str] = None) -> List[Tuple[str, ConsensusSignal]]:
        """
        Scan all coins and return buy signals sorted by tier and confidence.
        
        Args:
            symbols: List of symbols to scan
        
        Returns:
            List of (symbol, ConsensusSignal) tuples for potential buys
        """
        from config import WATCHLIST
        symbols = symbols or WATCHLIST
        
        signals = []
        scan_results = []
        
        for symbol in symbols:
            try:
                result = self.analyze(symbol)
                
                scan_results.append({
                    "symbol": symbol,
                    "signal": result.signal,
                    "tier": result.tier,
                    "confidence": result.confidence,
                    "ta_conf": result.ta_signal.confidence if result.ta_signal else 0,
                    "ml_conf": result.ml_signal.confidence if result.ml_signal else 0,
                    "tcn_conf": result.tcn_signal.confidence if result.tcn_signal else 0,
                    "reasons": result.reasons
                })
                
                # Persist predictions to DB for model accuracy tracking
                try:
                    from src.data.database import db
                    if result.ta_signal:
                        db.save_prediction(symbol, 'ta', result.ta_signal.signal, result.ta_signal.confidence)
                    if result.ml_signal:
                        db.save_prediction(symbol, 'ml', result.ml_signal.signal, result.ml_signal.confidence)
                    if result.tcn_signal:
                        db.save_prediction(symbol, 'tcn', result.tcn_signal.signal, result.tcn_signal.confidence)
                except Exception:
                    pass  # Never let DB write block trading
                
                if result.signal == "buy" and result.tier > 0:
                    signals.append((symbol, result))
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                
        self.latest_scan = scan_results
        
        # Sort by tier (descending) then confidence (descending)
        signals.sort(key=lambda x: (x[1].tier, x[1].confidence), reverse=True)
        
        return signals
    
    def get_exit_signal(self, symbol: str, entry_price: float, current_price: float) -> Tuple[str, str]:
        """
        Check if position should be exited based on model signals.
        
        Args:
            symbol: Trading pair
            entry_price: Entry price
            current_price: Current price
        
        Returns:
            Tuple of (exit_signal: 'exit'/'hold', reason: str)
        """
        try:
            result = self.analyze(symbol)
            
            # Check P&L
            pnl_pct = (current_price - entry_price) / entry_price
            
            # If models turn bearish
            if result.signal == "sell" and result.tier == 0:
                sell_count = sum(1 for s in [
                    result.ta_signal.signal if result.ta_signal else "hold",
                    result.ml_signal.signal if result.ml_signal else "hold",
                    result.tcn_signal.signal if result.tcn_signal else "hold"
                ] if s == "sell")
                
                if sell_count >= 2:
                    return ("exit", "Model consensus turned bearish")
            
            # If in profit but confidence dropped significantly
            if pnl_pct > 0.03 and result.confidence < 0.40:
                return ("exit", "Confidence dropped while in profit")
            
            return ("hold", "")
            
        except Exception as e:
            logger.error(f"Exit signal error for {symbol}: {e}")
            return ("hold", "")
    
    def _update_weights_from_performance(self):
        """Update model weights based on tracked performance."""
        try:
            adjusted = self.performance_tracker.get_adjusted_weights()
            self.ta_weight = adjusted["ta"]
            self.ml_weight = adjusted["ml"]
            self.tcn_weight = adjusted["tcn"]
        except Exception as e:
            logger.warning(f"Failed to update weights from performance: {e}")
    
    def record_trade_outcome(self, symbol: str, outcome: str, pnl: float = 0.0):
        """
        Record the outcome of a trade for performance tracking.
        
        Args:
            symbol: Trading pair
            outcome: 'win' or 'loss'
            pnl: Profit/loss percentage
        """
        try:
            self.performance_tracker.record_outcome(symbol, outcome, pnl)
            self._update_weights_from_performance()
            logger.info(f"Recorded {outcome} for {symbol} (PnL: {pnl:.1%})")
        except Exception as e:
            logger.warning(f"Failed to record trade outcome: {e}")
    
    def get_model_stats(self) -> Dict:
        """Get current model performance statistics."""
        return {
            "win_rates": self.performance_tracker.win_rates,
            "weights": {
                "ta": self.ta_weight,
                "ml": self.ml_weight,
                "tcn": self.tcn_weight
            },
            "predictions_count": len(self.performance_tracker.predictions),
            "regime": self.regime_detector.current_regime,
            "calibrated": self.calibrator.is_fitted
        }
    
    def fit_calibrators(self):
        """Fit confidence calibrators using historical prediction data."""
        try:
            for model in ["ta", "ml", "tcn"]:
                # Get predictions with outcomes for this model
                data = [p for p in self.performance_tracker.predictions 
                       if p["model"] == model and p["outcome"] is not None]
                
                if len(data) >= 20:
                    confidences = [p["confidence"] for p in data]
                    outcomes = [1 if p["outcome"] == "win" else 0 for p in data]
                    self.calibrator.fit(model, confidences, outcomes)
                    logger.info(f"Fitted calibrator for {model} with {len(data)} samples")
                else:
                    logger.info(f"Not enough data to calibrate {model} ({len(data)} samples)")
        except Exception as e:
            logger.warning(f"Calibrator fitting failed: {e}")


# Global ensemble instance
ensemble = ConsensusEnsemble()

