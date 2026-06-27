"""
Technical Analysis Analyzer — V5 (Feature-Provider Architecture)

ARCHITECTURE CHANGE (V5):
    Previously this module emitted heuristic Buy/Sell signals via a weighted
    scorecard (score >= 75 → "buy"). That approach is fundamentally flawed:
    fixed integer weights applied to binary indicator conditions cannot adapt
    to changing market regimes, and empirical studies show such rule-based
    systems perform no better than a coin flip in crypto.

    V5 Role: Pure structural analysis + feature extraction.
    - Computes all indicators (RSI, MACD, EMAs, BB, ADX, VWAP, patterns, etc.)
    - Applies multi-timeframe structural FILTERS (blocks low-quality setups)
    - Outputs a TASignal with:
        * signal: Always "hold" (TA no longer opens trades unilaterally)
        * confidence: Normalized structural score (0.0–1.0) used as an ML feature
        * feature_dict: Full dict of indicator values for ML feature engineering
        * reasons: Human-readable list of fired conditions

    The ConsensusEnsemble reads `ta_signal.confidence` and `ta_signal.feature_dict`
    as INPUT FEATURES to the XGBoost/CatBoost ensemble. TA never triggers an
    entry alone; it acts as a structural filter and feature provider.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import logger
from src.data import collector
from config import WATCHLIST

# Default Scorecard Constants
VOLUME_SPIKE_MULT = 2.0


# =============================================================================
#  COIN SECTORS (Spot Watchlist)
# =============================================================================
SECTORS = {
    "BTCUSDT":    "btc",
    "ETHUSDT":    "eth",
    "SOLUSDT":    "l1",
    "XRPUSDT":    "payments",
    "ADAUSDT":    "l1",
    "LINKUSDT":   "oracle",
    "AVAXUSDT":   "l1",
    "POLUSDT":    "l2",
    "DOTUSDT":    "l1",
    "TONUSDT":    "l1",
    "NEARUSDT":   "l1",
    "ARBUSDT":    "l2",
    "OPUSDT":     "l2",
    "SUIUSDT":    "l1",
    "APTUSDT":    "l1",
    "GRTUSDT":    "infra",
    "UNIUSDT":    "defi",
    "FILUSDT":    "storage",
    "VETUSDT":    "supply",
    "TAOUSDT":    "ai",
    "ATOMUSDT":   "l1",
    "ALGOUSDT":   "l1",
    "HBARUSDT":   "l1",
    "SEIUSDT":    "l1",
}


# =============================================================================
#  STRUCTURAL SCORE WEIGHTS  (used only to produce a normalised TA confidence
#  feature — NOT used as a standalone buy/sell trigger)
# =============================================================================
LONG_WEIGHTS = {
    "macd_bull_cross":    18,
    "vol_spike":          15,
    "above_vwap":         12,
    "ema_bullish":        12,
    "near_support":       10,
    "rsi_oversold":       10,
    "bb_bounce_lower":     8,
    "macd_hist_bull":      6,
    "stoch_bull_cross":    5,
    "stoch_oversold":      4,
    "above_ema200":        4,
    "bullish_divergence":  4,
}

SHORT_WEIGHTS = {
    "macd_bear_cross":    18,
    "vol_spike":          15,
    "below_vwap":         12,
    "ema_bearish":        12,
    "near_resistance":    10,
    "rsi_overbought":     10,
    "bb_bounce_upper":     8,
    "macd_hist_bear":      6,
    "stoch_bear_cross":    5,
    "stoch_overbought":    4,
    "above_ema200":        4,
    "bearish_divergence":  4,
}

LONG_BONUS = {
    "ema_bull_stack":        12,
    "golden_cross":          10,
    "morning_star":          10,
    "three_white_soldiers":  10,
    "strong_support":        10,
    "bullish_engulfing":      8,
    "hammer":                 8,
    "strong_above_vwap":      7,
    "piercing_line":          6,
    "vol_increasing":         5,
    "bb_squeeze":             4,
    "strong_trend":           4,
}

SHORT_BONUS = {
    "ema_bear_stack":        12,
    "death_cross":           10,
    "evening_star":          10,
    "three_black_crows":     10,
    "strong_resistance":     10,
    "bearish_engulfing":      8,
    "shooting_star":          8,
    "strong_below_vwap":      7,
    "dark_cloud_cover":       6,
    "vol_increasing":         5,
    "bb_squeeze":             4,
    "strong_trend":           4,
}

# Maximum possible score (sum of all weights + bonuses)
_MAX_LONG_SCORE  = sum(LONG_WEIGHTS.values())  + sum(LONG_BONUS.values())
_MAX_SHORT_SCORE = sum(SHORT_WEIGHTS.values()) + sum(SHORT_BONUS.values())


# =============================================================================
#  INDICATOR FUNCTIONS
# =============================================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = _ema(close, fast)
    ema_slow    = _ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger(close: pd.Series, period=20, std_dev=2):
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def _stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(d_period).mean()
    return k.fillna(50), d.fillna(50)

def _detect_patterns(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    o, h, l, c = open_.values, high.values, low.values, close.values
    keys = ["hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing",
            "doji", "morning_star", "evening_star", "three_white_soldiers",
            "three_black_crows", "piercing_line", "dark_cloud_cover"]
    if len(c) < 3:
        return {k: False for k in keys}

    o1, h1, l1, c1 = o[-3], h[-3], l[-3], c[-3]
    o2, h2, l2, c2 = o[-2], h[-2], l[-2], c[-2]
    o3, h3, l3, c3 = o[-1], h[-1], l[-1], c[-1]

    body2 = abs(c2 - o2)
    range2 = h2 - l2 if h2 != l2 else 0.0001
    body3 = abs(c3 - o3)
    range3 = h3 - l3 if h3 != l3 else 0.0001

    lower_wick3 = (min(o3, c3) - l3)
    upper_wick3 = (h3 - max(o3, c3))

    hammer           = (lower_wick3 >= 2 * body3 and upper_wick3 <= body3 * 0.5 and body3 / range3 < 0.4 and c3 > o3)
    shooting_star    = (upper_wick3 >= 2 * body3 and lower_wick3 <= body3 * 0.5 and body3 / range3 < 0.4 and c3 < o3)
    bullish_engulf   = (c2 < o2 and c3 > o3 and o3 <= c2 and c3 >= o2)
    bearish_engulf   = (c2 > o2 and c3 < o3 and o3 >= c2 and c3 <= o2)
    doji             = body3 / range3 < 0.1 if range3 > 0 else False

    body1 = abs(c1 - o1)
    morning_star = (c1 < o1 and body2 / range2 < 0.3 and c3 > o3 and c3 > (o1 + c1) / 2)
    evening_star = (c1 > o1 and body2 / range2 < 0.3 and c3 < o3 and c3 < (o1 + c1) / 2)

    three_white = three_black = False
    if len(c) >= 5:
        o4, h4, l4, c4 = o[-5], h[-5], l[-5], c[-5]
        o5, h5, l5, c5 = o[-4], h[-4], l[-4], c[-4]
        body4, body5 = abs(c4 - o4), abs(c5 - o5)
        three_white = (c4 > o4 and c5 > o5 and c3 > o3 and c5 > c4 and c3 > c5
                       and o5 > o4 and o5 < c4 and o3 > o5 and o3 < c5
                       and body4 > 0 and body5 > 0 and body3 > 0)
        three_black = (c4 < o4 and c5 < o5 and c3 < o3 and c5 < c4 and c3 < c5
                       and o5 < o4 and o5 > c4 and o3 < o5 and o3 > c5
                       and body4 > 0 and body5 > 0 and body3 > 0)

    piercing      = (c2 < o2 and c3 > o3 and o3 < l2 and c3 > (o2 + c2) / 2 and c3 < o2)
    dark_cloud    = (c2 > o2 and c3 < o3 and o3 > h2 and c3 < (o2 + c2) / 2 and c3 > o2)

    return {
        "hammer": bool(hammer), "shooting_star": bool(shooting_star),
        "bullish_engulfing": bool(bullish_engulf), "bearish_engulfing": bool(bearish_engulf),
        "doji": bool(doji), "morning_star": bool(morning_star), "evening_star": bool(evening_star),
        "three_white_soldiers": bool(three_white), "three_black_crows": bool(three_black),
        "piercing_line": bool(piercing), "dark_cloud_cover": bool(dark_cloud)
    }

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> float:
    try:
        prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)
        up_move   = high - prev_high
        down_move = prev_low - low
        plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr    = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_s = tr.ewm(com=period - 1, min_periods=period).mean()
        plus_di  = 100 * pd.Series(plus_dm,  index=high.index).ewm(com=period - 1, min_periods=period).mean() / atr_s.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(com=period - 1, min_periods=period).mean() / atr_s.replace(0, np.nan)

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(com=period - 1, min_periods=period).mean()
        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0
    except Exception:
        return 20.0

def _rsi_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 14) -> dict:
    try:
        prices = close.values[-lookback:]
        rsis   = rsi_series.values[-lookback:]
        # M-7 FIX: Exclude the last bar to find the prior trough/troughs
        prior_prices = prices[:-1]
        prior_rsis   = rsis[:-1]
        if len(prior_prices) > 0:
            price_min_idx = np.argmin(prior_prices)
            price_max_idx = np.argmax(prior_prices)
            bullish_div = prices[-1] <= prior_prices[price_min_idx] and rsis[-1] > prior_rsis[price_min_idx]
            bearish_div = prices[-1] >= prior_prices[price_max_idx] and rsis[-1] < prior_rsis[price_max_idx]
        else:
            bullish_div = bearish_div = False
        return {"bullish_divergence": bool(bullish_div), "bearish_divergence": bool(bearish_div)}
    except Exception:
        return {"bullish_divergence": False, "bearish_divergence": False}

def calculate_vwap(df: pd.DataFrame) -> dict:
    try:
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
        typical_price = (high + low + close) / 3
        # M-2 FIX: Use rolling 24-bar window for VWAP calculation rather than cumsum
        rolling_tp_vol = (typical_price * volume).rolling(24).sum()
        rolling_vol = volume.rolling(24).sum()
        vwap = rolling_tp_vol / rolling_vol
        # Fall back to cumsum if not enough bars for rolling window
        vwap = vwap.fillna((typical_price * volume).cumsum() / volume.cumsum())
        vwap_val      = float(vwap.iloc[-1])
        price         = float(close.iloc[-1])
        vwap_dist     = round(((price - vwap_val) / vwap_val) * 100, 2)
        above_vwap    = price > vwap_val
        return {
            "vwap": round(vwap_val, 6), "vwap_dist_pct": vwap_dist,
            "above_vwap": above_vwap, "below_vwap": not above_vwap,
            "strong_above_vwap": vwap_dist > 1.0, "strong_below_vwap": vwap_dist < -1.0,
        }
    except Exception:
        return {"vwap": 0, "vwap_dist_pct": 0, "above_vwap": False, "below_vwap": False,
                "strong_above_vwap": False, "strong_below_vwap": False}

def calculate_daily_bias(df_daily: pd.DataFrame) -> dict:
    try:
        if df_daily is None or len(df_daily) < 5:
            return {"daily_bias": "neutral", "daily_bullish": False, "daily_bearish": False}
        close, open_ = df_daily["close"], df_daily["open"]
        last3_bullish = sum(1 for i in [-3, -2, -1] if float(close.iloc[i]) > float(open_.iloc[i]))
        last3_bearish = sum(1 for i in [-3, -2, -1] if float(close.iloc[i]) < float(open_.iloc[i]))
        ema20_d = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50_d = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        price_d = float(close.iloc[-1])
        daily_bullish = (last3_bullish >= 2) and (price_d > ema20_d)
        daily_bearish = (last3_bearish >= 2) and (price_d < ema20_d)
        bias = "bullish" if daily_bullish else "bearish" if daily_bearish else "neutral"
        return {"daily_bias": bias, "daily_bullish": daily_bullish, "daily_bearish": daily_bearish,
                "daily_ema20": round(ema20_d, 6), "daily_ema50": round(ema50_d, 6)}
    except Exception:
        return {"daily_bias": "neutral", "daily_bullish": False, "daily_bearish": False,
                "daily_ema20": 0, "daily_ema50": 0}


# =============================================================================
#  OUTPUT DATACLASS
# =============================================================================

@dataclass
class TASignal:
    """
    Technical Analysis signal output.

    V5 Contract:
        signal     : Always "hold". TA no longer opens positions unilaterally.
        confidence : Normalised structural score in [0, 1]. Used as an ML
                     feature by the ConsensusEnsemble.
        feature_dict: Full dict of boolean/float indicator values for ML input.
        key_levels : Support, resistance, ATR, stop-loss, take-profit.
        reasons    : Human-readable list of fired conditions.
        blocked    : True when a hard structural filter vetoed the setup.
        block_reason: Reason string when blocked=True.
    """
    signal:       str   = "hold"
    confidence:   float = 0.0
    feature_dict: Dict  = field(default_factory=dict)
    key_levels:   Dict  = field(default_factory=dict)
    reasons:      List  = field(default_factory=list)
    blocked:      bool  = False
    block_reason: str   = ""

    def to_dict(self) -> Dict:
        return {
            "signal":       self.signal,
            "confidence":   self.confidence,
            "feature_dict": self.feature_dict,
            "key_levels":   self.key_levels,
            "reasons":      self.reasons,
            "blocked":      self.blocked,
            "block_reason": self.block_reason,
        }


# =============================================================================
#  TA ANALYZER CLASS
# =============================================================================

class TAAnalyzer:
    """
    V5 Technical Analysis Analyzer — Feature Provider & Structural Filter.

    What changed vs V4:
        - No more `score >= 75 → "buy"` heuristic. That decision is now owned
          entirely by the XGBoost/CatBoost ensemble inside ConsensusEnsemble.
        - TAAnalyzer still calculates the full indicator stack AND enforces hard
          structural filters (ADX, BTC correlation, dead-cat-bounce guard, daily
          bias). If a filter trips, `blocked=True` is returned; the ensemble
          will respect this veto.
        - The normalised structural score [0–1] is still emitted as `confidence`
          to serve as a weighted feature inside the ML model.
    """

    def __init__(self):
        self._btc_cache = {"bias": "neutral", "timestamp": 0}

    # ------------------------------------------------------------------
    # BTC bias (cached 4 minutes)
    # ------------------------------------------------------------------
    def _get_btc_bias(self) -> str:
        now = time.time()
        if now - self._btc_cache["timestamp"] > 240:
            try:
                df_1h = collector.get_dataframe("BTCUSDT", "1h", limit=60)
                df_4h = collector.get_dataframe("BTCUSDT", "4h", limit=60)
                if df_1h.empty or df_4h.empty:
                    return self._btc_cache["bias"]
                ind_1h = self._calculate_indicators(df_1h)
                ind_4h = self._calculate_indicators(df_4h)
                btc_bull = ind_1h.get("ema_bullish") and ind_4h.get("ema_bullish") and ind_4h.get("macd_bullish")
                btc_bear = ind_1h.get("ema_bearish") and ind_4h.get("macd_bearish")
                bias = "bullish" if btc_bull else "bearish" if btc_bear else "neutral"
                self._btc_cache["bias"]      = bias
                self._btc_cache["timestamp"] = now
            except Exception:
                pass
        return self._btc_cache["bias"]

    # ------------------------------------------------------------------
    # Core indicator computation
    # ------------------------------------------------------------------
    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < 60:
            return {}
        try:
            close, high, low, open_, volume = df["close"], df["high"], df["low"], df["open"], df["volume"]
            price = float(close.iloc[-1])

            rsi_series = _rsi(close, 14)
            rsi        = float(rsi_series.iloc[-1])

            stoch_k, stoch_d = _stochastic(high, low, close)
            stoch_k_val, stoch_d_val = float(stoch_k.iloc[-1]), float(stoch_d.iloc[-1])
            stoch_bull_cross  = (float(stoch_k.iloc[-2]) < float(stoch_d.iloc[-2])) and (stoch_k_val > stoch_d_val) and stoch_k_val < 30
            stoch_bear_cross  = (float(stoch_k.iloc[-2]) > float(stoch_d.iloc[-2])) and (stoch_k_val < stoch_d_val) and stoch_k_val > 70
            stoch_oversold    = stoch_k_val < 20 and stoch_d_val < 20
            stoch_overbought  = stoch_k_val > 80 and stoch_d_val > 80

            macd_line_s, macd_sig_s, macd_hist_s = _macd(close)
            macd_line, macd_signal, macd_hist = float(macd_line_s.iloc[-1]), float(macd_sig_s.iloc[-1]), float(macd_hist_s.iloc[-1])
            macd_bull_cross = (float(macd_line_s.iloc[-2]) < float(macd_sig_s.iloc[-2])) and (macd_line > macd_signal)
            macd_bear_cross = (float(macd_line_s.iloc[-2]) > float(macd_sig_s.iloc[-2])) and (macd_line < macd_signal)
            macd_hist_bull  = macd_hist > 0 and macd_hist > float(macd_hist_s.iloc[-2])
            macd_hist_bear  = macd_hist < 0 and macd_hist < float(macd_hist_s.iloc[-2])

            ema9_s, ema20_s, ema50_s, ema200_s = _ema(close, 9), _ema(close, 20), _ema(close, 50), _ema(close, 200)
            ema9, ema20, ema50, ema200 = float(ema9_s.iloc[-1]), float(ema20_s.iloc[-1]), float(ema50_s.iloc[-1]), float(ema200_s.iloc[-1])
            ema_bull_stack = (price > ema9) and (ema9 > ema20) and (ema20 > ema50)
            ema_bullish    = (price > ema20) and (ema20 > ema50)
            ema_bear_stack = (price < ema9) and (ema9 < ema20) and (ema20 < ema50)
            ema_bearish    = (price < ema20) and (ema20 < ema50)
            golden_cross   = (float(ema20_s.iloc[-2]) < float(ema50_s.iloc[-2])) and (ema20 > ema50)
            death_cross    = (float(ema20_s.iloc[-2]) > float(ema50_s.iloc[-2])) and (ema20 < ema50)

            bb_upper_s, bb_mid_s, bb_lower_s = _bollinger(close)
            bb_upper, bb_mid, bb_lower = float(bb_upper_s.iloc[-1]), float(bb_mid_s.iloc[-1]), float(bb_lower_s.iloc[-1])
            bb_bounce_lower = price <= bb_lower * 1.008
            bb_bounce_upper = price >= bb_upper * 0.992
            bb_width  = (bb_upper - bb_lower) / bb_mid if bb_mid != 0 else 0.0
            bb_squeeze = bb_width < 0.04

            vol_avg       = float(volume.rolling(20).mean().iloc[-1])
            vol_curr      = float(volume.iloc[-1])
            vol_spike     = vol_curr > (vol_avg * VOLUME_SPIKE_MULT)
            vol_increasing = float(volume.iloc[-1]) > float(volume.iloc[-2]) > float(volume.iloc[-3])

            atr_series = _atr(high, low, close, 14)
            atr        = float(atr_series.iloc[-1])
            if np.isnan(atr): atr = price * 0.02
            atr_pct = (atr / price) * 100

            def find_swing_levels(high_s, low_s, lookback=50, tolerance=0.005):
                h, l = high_s.tail(lookback).values, low_s.tail(lookback).values
                swing_highs, swing_lows = [], []
                for i in range(2, len(h) - 2):
                    if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]: swing_highs.append(h[i])
                    if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]: swing_lows.append(l[i])
                def cluster(levels):
                    if not levels: return []
                    levels = sorted(levels)
                    clustered = [levels[0]]
                    for lv in levels[1:]:
                        if abs(lv - clustered[-1]) / clustered[-1] > tolerance: clustered.append(lv)
                    return clustered
                return cluster(swing_highs), cluster(swing_lows)

            swing_highs, swing_lows = find_swing_levels(high, low)
            supports    = [s for s in swing_lows   if s < price]
            resistances = [r for r in swing_highs  if r > price]
            nearest_support    = max(supports)    if supports    else float(low.tail(50).min())
            nearest_resistance = min(resistances) if resistances else float(high.tail(50).max())
            near_support    = price <= nearest_support    * 1.015 if supports    else price <= float(low.tail(20).min())  * 1.015
            near_resistance = price >= nearest_resistance * 0.985 if resistances else price >= float(high.tail(20).max()) * 0.985
            support_strength    = sum(1 for s in swing_lows   if abs(s - nearest_support)    / nearest_support    < 0.01)
            resistance_strength = sum(1 for r in swing_highs  if abs(r - nearest_resistance) / nearest_resistance < 0.01)

            adx        = _adx(high, low, close, 14)
            divergence = _rsi_divergence(close, rsi_series)
            patterns   = _detect_patterns(open_, high, low, close)
            vwap_data  = calculate_vwap(df)

            # Normalised distance features (stationary, safe for ML)
            price_vs_ema20  = (price - ema20)  / ema20  if ema20  else 0.0
            price_vs_ema50  = (price - ema50)  / ema50  if ema50  else 0.0
            price_vs_ema200 = (price - ema200) / ema200 if ema200 else 0.0
            price_vs_vwap   = vwap_data.get("vwap_dist_pct", 0.0) / 100.0

            return {
                # Price
                "price": price, "atr": atr, "atr_pct": atr_pct,
                "price_vs_ema20": price_vs_ema20, "price_vs_ema50": price_vs_ema50,
                "price_vs_ema200": price_vs_ema200, "price_vs_vwap": price_vs_vwap,
                # Momentum
                "rsi": rsi, "stoch_k": stoch_k_val, "stoch_d": stoch_d_val,
                "macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
                # Booleans — trend
                "ema_bullish": ema_bullish, "ema_bull_stack": ema_bull_stack,
                "ema_bearish": ema_bearish, "ema_bear_stack": ema_bear_stack,
                "golden_cross": golden_cross, "death_cross": death_cross,
                "above_ema200": price > ema200, "below_ema200": price < ema200,
                # Booleans — oscillators
                "rsi_oversold": rsi < 30, "rsi_overbought": rsi > 70,
                "stoch_bull_cross": stoch_bull_cross, "stoch_oversold": stoch_oversold,
                "stoch_bear_cross": stoch_bear_cross, "stoch_overbought": stoch_overbought,
                "macd_bull_cross": macd_bull_cross, "macd_bullish": macd_line > macd_signal,
                "macd_bear_cross": macd_bear_cross, "macd_bearish": macd_line < macd_signal,
                "macd_hist_bull": macd_hist_bull, "macd_hist_bear": macd_hist_bear,
                # Booleans — structure
                "bb_bounce_lower": bb_bounce_lower, "bb_bounce_upper": bb_bounce_upper,
                "bb_squeeze": bb_squeeze, "bb_width": bb_width,
                "vol_spike": vol_spike, "vol_increasing": vol_increasing,
                "vol_ratio": vol_curr / vol_avg if vol_avg > 0 else 1.0,
                "near_support": near_support, "strong_support": near_support and support_strength >= 2,
                "near_resistance": near_resistance, "strong_resistance": near_resistance and resistance_strength >= 2,
                "support": nearest_support, "resistance": nearest_resistance,
                "adx": adx, "strong_trend": adx > 40,
                "bullish_divergence": divergence["bullish_divergence"],
                "bearish_divergence": divergence["bearish_divergence"],
                # Patterns
                **{k: patterns[k] for k in patterns},
                # VWAP
                "above_vwap": vwap_data["above_vwap"], "strong_above_vwap": vwap_data["strong_above_vwap"],
                "below_vwap": vwap_data["below_vwap"],  "strong_below_vwap": vwap_data["strong_below_vwap"],
            }
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Internal scoring (produces normalised confidence, NOT a trade signal)
    # ------------------------------------------------------------------
    def _score_symbol(self, indicators: dict, direction: str) -> Tuple[float, List[str]]:
        """Returns a normalised score [0, 1] and list of fired conditions."""
        weights = LONG_WEIGHTS if direction == "LONG" else SHORT_WEIGHTS
        bonuses = LONG_BONUS   if direction == "LONG" else SHORT_BONUS
        max_score = _MAX_LONG_SCORE if direction == "LONG" else _MAX_SHORT_SCORE
        score = 0
        fired = []

        for name, weight in weights.items():
            if name == "above_ema200" and direction == "SHORT":
                if not indicators.get(name, False):
                    score += weight
                    fired.append(f"below_ema200 (+{weight})")
                continue
            if indicators.get(name, False):
                score += weight
                fired.append(f"{name} (+{weight})")

        for name, pts in bonuses.items():
            if indicators.get(name, False):
                score += pts
                fired.append(f"{name} [+Bonus {pts}]")

        normalised = min(score / max(max_score, 1), 1.0)
        return normalised, fired

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def analyze(self, symbol: str, df: pd.DataFrame = None) -> TASignal:
        """
        Analyze symbol and return a TASignal.

        Returns:
            TASignal with:
              - signal      = "hold" always (TA cannot open trades in V5)
              - confidence  = normalised structural score [0, 1]
              - feature_dict= all indicator booleans/floats for ML features
              - blocked     = True if a hard structural filter fired (ensemble
                             should skip this coin regardless of ML confidence)
        """
        try:
            is_evaluation_mode = df is not None

            if df is None:
                primary_df = collector.get_dataframe(symbol, timeframe="1h",  limit=200)
                tf_15m_df  = collector.get_dataframe(symbol, timeframe="15m", limit=100)
                tf_4h_df   = collector.get_dataframe(symbol, timeframe="4h",  limit=100)
                df_daily   = collector.get_dataframe(symbol, timeframe="1d",  limit=60)
            else:
                primary_df = df
                tf_15m_df  = pd.DataFrame()
                tf_4h_df   = pd.DataFrame()
                df_daily   = pd.DataFrame()

            if primary_df.empty or len(primary_df) < 60:
                return TASignal(signal="hold", confidence=0.0, reasons=["Insufficient data"])

            primary = self._calculate_indicators(primary_df)
            tf_15m  = self._calculate_indicators(tf_15m_df) if not tf_15m_df.empty else primary
            tf_4h   = self._calculate_indicators(tf_4h_df)  if not tf_4h_df.empty  else primary
            daily   = calculate_daily_bias(df_daily) if not df_daily.empty else {
                "daily_bias": "neutral", "daily_bullish": False, "daily_bearish": False
            }

            if not primary:
                return TASignal(signal="hold", confidence=0.0, reasons=["Indicator calculation failed"])

            price = primary["price"]
            atr   = primary.get("atr", price * 0.02)
            stop_loss   = price - (1.5 * atr)
            take_profit = price + (2.0 * (price - stop_loss))
            key_levels  = {
                "support":     primary.get("support",    0),
                "resistance":  primary.get("resistance", 0),
                "atr":         atr,
                "stop_loss":   stop_loss,
                "take_profit": take_profit,
            }

            # Evaluation mode: return feature dict without filters
            if is_evaluation_mode:
                long_conf, _ = self._score_symbol(primary, "LONG")
                return TASignal(
                    signal="hold", confidence=long_conf,
                    feature_dict=primary, key_levels=key_levels,
                    reasons=["Evaluation mode — no filter applied"]
                )

            # --- Score both directions ---
            long_conf,  long_fired  = self._score_symbol(primary, "LONG")
            short_conf, short_fired = self._score_symbol(primary, "SHORT")
            direction  = "LONG" if long_conf >= short_conf else "SHORT"
            confidence = long_conf if direction == "LONG" else short_conf
            fired      = long_fired if direction == "LONG" else short_fired

            # --- Hard structural filters (these become VETOES, not signals) ---
            def _block(reason: str) -> TASignal:
                return TASignal(
                    signal="hold", confidence=confidence,
                    feature_dict=primary, key_levels=key_levels,
                    reasons=[f"BLOCKED: {reason}"],
                    blocked=True, block_reason=reason
                )

            # 4h EMA filter
            if direction == "LONG":
                if tf_4h.get("ema_bear_stack", False): return _block("4h EMA stack bearish")
                if tf_4h.get("death_cross",    False): return _block("4h death cross active")
            else:
                if tf_4h.get("ema_bull_stack", False): return _block("4h EMA stack bullish")

            # Dead cat bounce
            if direction == "LONG":
                if (tf_4h.get("ema_bearish", False) and tf_4h.get("macd_bearish", False)
                        and primary.get("vol_ratio", 1.0) < 1.2):
                    return _block("Dead cat bounce — 4h bearish + weak volume")
                if (primary.get("rsi", 50) > 45 and tf_4h.get("rsi", 50) < 42
                        and tf_4h.get("ema_bearish", False)):
                    return _block("RSI dead cat — 4h RSI still bearish")

            # Daily bias
            if direction == "LONG"  and daily["daily_bearish"]: return _block("Daily bearish bias")
            if direction == "SHORT" and daily["daily_bullish"]: return _block("Daily bullish bias")
            if direction == "SHORT" and daily["daily_bias"] == "neutral":
                return _block("Daily neutral — SHORT requires confirmed bearish daily")

            # Choppy market
            adx = primary.get("adx", 20)
            if adx < 15: return _block(f"ADX={round(adx,1)}<15 — too choppy")
            if adx > 55: return _block(f"ADX={round(adx,1)}>55 — super trend, indicators lag")
            if primary.get("atr_pct", 1.0) < 0.3: return _block("ATR% <0.3 — too choppy")

            # BTC correlation
            if symbol != "BTCUSDT":
                btc_bias = self._get_btc_bias()
                if direction == "LONG"  and btc_bias == "bearish" and confidence < 0.75:
                    return _block(f"BTC bearish, conf {confidence:.2f}<0.75")
                if direction == "SHORT" and btc_bias == "bullish" and confidence < 0.75:
                    return _block(f"BTC bullish, conf {confidence:.2f}<0.75")

            # 15m & 4h alignment
            if direction == "LONG":
                tf_15m_agrees = bool(
                    tf_15m.get("macd_bullish") or tf_15m.get("rsi_oversold")
                    or tf_15m.get("stoch_oversold") or tf_15m.get("above_vwap")
                )
                tf_4h_agrees = bool(
                    tf_4h.get("ema_bullish") or tf_4h.get("macd_bullish") or tf_4h.get("above_vwap")
                )
            else:
                tf_15m_agrees = bool(
                    tf_15m.get("macd_bearish") or tf_15m.get("rsi_overbought")
                    or tf_15m.get("stoch_overbought") or tf_15m.get("below_vwap")
                )
                tf_4h_agrees = bool(
                    tf_4h.get("ema_bearish") or tf_4h.get("macd_bearish") or tf_4h.get("below_vwap")
                )

            if not tf_15m_agrees: return _block(f"15m not aligned with {direction}")
            if not tf_4h_agrees:  return _block(f"4h not aligned with {direction}")

            # All filters passed — emit feature dict + normalised confidence
            # NOTE: signal stays "hold". Entry decision is owned by the ML ensemble.
            reasons = [
                f"TA structural score: {confidence:.0%} ({direction})",
                f"Sector: {SECTORS.get(symbol, 'other')}",
            ] + fired[:4]

            return TASignal(
                signal="hold",          # ← V5: TA never opens trades alone
                confidence=confidence,
                feature_dict=primary,
                key_levels=key_levels,
                reasons=reasons,
                blocked=False,
            )

        except Exception as e:
            logger.error(f"TA analyze error for {symbol}: {e}")
            return TASignal(signal="hold", confidence=0.0, reasons=[f"Error: {str(e)}"])


# Global instance
ta_analyzer = TAAnalyzer()
