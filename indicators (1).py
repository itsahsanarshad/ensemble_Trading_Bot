# =============================================================================
#  INDICATORS — pure pandas/numpy. Python 3.14+ compatible.
#  NEW: candlestick patterns, trend strength, divergence detection
# =============================================================================

import pandas as pd
import numpy as np
import logging
from config import RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_MULT

logger = logging.getLogger(__name__)


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


# =============================================================================
#  CANDLESTICK PATTERNS
# =============================================================================

def _detect_patterns(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Detect 6 high-reliability candlestick patterns on the last 3 candles."""

    o  = open_.values
    h  = high.values
    l  = low.values
    c  = close.values

    if len(c) < 3:
        return {k: False for k in ["hammer", "shooting_star", "bullish_engulfing",
                                    "bearish_engulfing", "doji", "morning_star"]}

    # Latest 3 candles
    o1, h1, l1, c1 = o[-3], h[-3], l[-3], c[-3]   # 3 candles ago
    o2, h2, l2, c2 = o[-2], h[-2], l[-2], c[-2]   # previous candle
    o3, h3, l3, c3 = o[-1], h[-1], l[-1], c[-1]   # current candle

    body2  = abs(c2 - o2)
    range2 = h2 - l2 if h2 != l2 else 0.0001
    body3  = abs(c3 - o3)
    range3 = h3 - l3 if h3 != l3 else 0.0001

    lower_wick2 = (min(o2, c2) - l2)
    upper_wick2 = (h2 - max(o2, c2))
    lower_wick3 = (min(o3, c3) - l3)
    upper_wick3 = (h3 - max(o3, c3))

    # Hammer: small body top, long lower wick ≥2× body, bullish reversal
    hammer = (
        lower_wick3 >= 2 * body3 and
        upper_wick3 <= body3 * 0.5 and
        body3 / range3 < 0.4 and
        c3 > o3  # green candle
    )

    # Shooting Star: small body bottom, long upper wick, bearish reversal
    shooting_star = (
        upper_wick3 >= 2 * body3 and
        lower_wick3 <= body3 * 0.5 and
        body3 / range3 < 0.4 and
        c3 < o3  # red candle
    )

    # Bullish Engulfing: red candle then bigger green candle
    bullish_engulfing = (
        c2 < o2 and        # prev candle is red
        c3 > o3 and        # current is green
        o3 <= c2 and       # opens at or below prev close
        c3 >= o2           # closes at or above prev open
    )

    # Bearish Engulfing: green candle then bigger red candle
    bearish_engulfing = (
        c2 > o2 and        # prev candle is green
        c3 < o3 and        # current is red
        o3 >= c2 and       # opens at or above prev close
        c3 <= o2           # closes at or below prev open
    )

    # Doji: open ≈ close (indecision — confirms with next candle)
    doji = body3 / range3 < 0.1 if range3 > 0 else False

    # Morning Star: red, small body (doji-like), green — 3-candle bullish reversal
    body1 = abs(c1 - o1)
    morning_star = (
        c1 < o1 and          # first candle red
        body2 / range2 < 0.3 and  # middle is small (indecision)
        c3 > o3 and          # third is green
        c3 > (o1 + c1) / 2  # closes above midpoint of first candle
    )

    # Evening Star: green, small body, red — 3-candle bearish reversal (opposite of morning star)
    evening_star = (
        c1 > o1 and                    # first candle green (bullish)
        body2 / range2 < 0.3 and       # middle is small (indecision)
        c3 < o3 and                    # third is red (bearish)
        c3 < (o1 + c1) / 2            # closes below midpoint of first candle
    )

    # Three White Soldiers: 3 consecutive bullish candles — strong upward momentum
    # Each opens within previous body and closes higher
    if len(c) >= 5:
        o4, h4, l4, c4 = o[-5], h[-5], l[-5], c[-5]
        o5, h5, l5, c5 = o[-4], h[-4], l[-4], c[-4]
        body4 = abs(c4 - o4)
        body5 = abs(c5 - o5)
        three_white_soldiers = (
            c4 > o4 and c5 > o5 and c3 > o3 and  # all three green
            c5 > c4 and c3 > c5 and               # each closes higher
            o5 > o4 and o5 < c4 and               # opens within prev body
            o3 > o5 and o3 < c5 and               # opens within prev body
            body4 > 0 and body5 > 0 and body3 > 0 # real bodies (not doji)
        )
    else:
        three_white_soldiers = False

    # Three Black Crows: 3 consecutive bearish candles — strong downward momentum
    if len(c) >= 5:
        three_black_crows = (
            c4 < o4 and c5 < o5 and c3 < o3 and  # all three red
            c5 < c4 and c3 < c5 and               # each closes lower
            o5 < o4 and o5 > c4 and               # opens within prev body
            o3 < o5 and o3 > c5 and               # opens within prev body
            body4 > 0 and body5 > 0 and body3 > 0 # real bodies
        )
    else:
        three_black_crows = False

    # Piercing Line: bearish candle then bullish that closes above midpoint — bullish reversal
    piercing_line = (
        c2 < o2 and                    # prev is red
        c3 > o3 and                    # current is green
        o3 < l2 and                    # opens below prev low (gap down)
        c3 > (o2 + c2) / 2 and        # closes above midpoint of prev body
        c3 < o2                        # but doesn't fully engulf
    )

    # Dark Cloud Cover: bullish candle then bearish that closes below midpoint — bearish reversal
    dark_cloud_cover = (
        c2 > o2 and                    # prev is green
        c3 < o3 and                    # current is red
        o3 > h2 and                    # opens above prev high (gap up)
        c3 < (o2 + c2) / 2 and        # closes below midpoint of prev body
        c3 > o2                        # but doesn't fully engulf
    )

    return {
        "hammer":               bool(hammer),
        "shooting_star":        bool(shooting_star),
        "bullish_engulfing":    bool(bullish_engulfing),
        "bearish_engulfing":    bool(bearish_engulfing),
        "doji":                 bool(doji),
        "morning_star":         bool(morning_star),
        "evening_star":         bool(evening_star),
        "three_white_soldiers": bool(three_white_soldiers),
        "three_black_crows":    bool(three_black_crows),
        "piercing_line":        bool(piercing_line),
        "dark_cloud_cover":     bool(dark_cloud_cover),
    }


# =============================================================================
#  TREND STRENGTH
# =============================================================================

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> float:
    """ADX: 0-100. Above 25 = trending. Above 40 = strong trend."""
    try:
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)
        prev_close = close.shift(1)

        up_move   = high - prev_high
        down_move = prev_low - low

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)

        atr_s    = tr.ewm(com=period - 1, min_periods=period).mean()
        plus_di  = 100 * pd.Series(plus_dm,  index=high.index).ewm(com=period - 1, min_periods=period).mean() / atr_s.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(com=period - 1, min_periods=period).mean() / atr_s.replace(0, np.nan)

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(com=period - 1, min_periods=period).mean()
        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0
    except Exception:
        return 20.0


# =============================================================================
#  RSI DIVERGENCE
# =============================================================================

def _rsi_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 14) -> dict:
    """
    Bullish divergence:  price makes lower low BUT rsi makes higher low  → reversal up
    Bearish divergence:  price makes higher high BUT rsi makes lower high → reversal down
    """
    try:
        prices = close.values[-lookback:]
        rsis   = rsi_series.values[-lookback:]

        price_min_idx = np.argmin(prices)
        price_max_idx = np.argmax(prices)

        bullish_div = (
            prices[-1] < prices[price_min_idx] and
            rsis[-1]   > rsis[price_min_idx]
        )
        bearish_div = (
            prices[-1] > prices[price_max_idx] and
            rsis[-1]   < rsis[price_max_idx]
        )

        return {"bullish_divergence": bool(bullish_div), "bearish_divergence": bool(bearish_div)}
    except Exception:
        return {"bullish_divergence": False, "bearish_divergence": False}


# =============================================================================
#  MAIN CALCULATE
# =============================================================================

def calculate_all(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 60:
        return {}

    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df["volume"]
        price  = float(close.iloc[-1])

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi_series = _rsi(close, 14)
        rsi        = float(rsi_series.iloc[-1])

        # ── Stochastic ────────────────────────────────────────────────────────
        stoch_k, stoch_d = _stochastic(high, low, close)
        stoch_k_val = float(stoch_k.iloc[-1])
        stoch_d_val = float(stoch_d.iloc[-1])
        stoch_oversold   = stoch_k_val < 20 and stoch_d_val < 20
        stoch_overbought = stoch_k_val > 80 and stoch_d_val > 80
        stoch_bull_cross = (float(stoch_k.iloc[-2]) < float(stoch_d.iloc[-2])) and (stoch_k_val > stoch_d_val) and stoch_k_val < 30
        stoch_bear_cross = (float(stoch_k.iloc[-2]) > float(stoch_d.iloc[-2])) and (stoch_k_val < stoch_d_val) and stoch_k_val > 70

        # ── MACD ──────────────────────────────────────────────────────────────
        macd_line_s, macd_sig_s, macd_hist_s = _macd(close)
        macd_line   = float(macd_line_s.iloc[-1])
        macd_signal = float(macd_sig_s.iloc[-1])
        macd_hist   = float(macd_hist_s.iloc[-1])
        prev_macd   = float(macd_line_s.iloc[-2])
        prev_sig    = float(macd_sig_s.iloc[-2])
        macd_bull_cross = (prev_macd < prev_sig) and (macd_line > macd_signal)
        macd_bear_cross = (prev_macd > prev_sig) and (macd_line < macd_signal)
        # Histogram growing (momentum building even without cross yet)
        prev_hist   = float(macd_hist_s.iloc[-2])
        macd_hist_growing_bull = macd_hist > 0 and macd_hist > prev_hist
        macd_hist_growing_bear = macd_hist < 0 and macd_hist < prev_hist

        # ── EMA ───────────────────────────────────────────────────────────────
        ema9_s   = _ema(close, 9)
        ema20_s  = _ema(close, 20)
        ema50_s  = _ema(close, 50)
        ema200_s = _ema(close, 200)
        ema9     = float(ema9_s.iloc[-1])
        ema20    = float(ema20_s.iloc[-1])
        ema50    = float(ema50_s.iloc[-1])
        ema200   = float(ema200_s.iloc[-1])
        prev_ema20 = float(ema20_s.iloc[-2])
        prev_ema50 = float(ema50_s.iloc[-2])

        # Full EMA stack alignment (professional trend filter)
        ema_bull_stack = (price > ema9) and (ema9 > ema20) and (ema20 > ema50)  # strong uptrend
        ema_bear_stack = (price < ema9) and (ema9 < ema20) and (ema20 < ema50)  # strong downtrend
        ema_bullish    = (price > ema20) and (ema20 > ema50)
        ema_bearish    = (price < ema20) and (ema20 < ema50)
        above_ema200   = price > ema200
        golden_cross   = (prev_ema20 < prev_ema50) and (ema20 > ema50)
        death_cross    = (prev_ema20 > prev_ema50) and (ema20 < ema50)

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_upper_s, bb_mid_s, bb_lower_s = _bollinger(close)
        bb_upper = float(bb_upper_s.iloc[-1])
        bb_mid   = float(bb_mid_s.iloc[-1])
        bb_lower = float(bb_lower_s.iloc[-1])
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid != 0 else 0.0
        bb_bounce_lower = price <= bb_lower * 1.008
        bb_bounce_upper = price >= bb_upper * 0.992
        bb_squeeze      = bb_width < 0.04   # volatility compression = big move coming

        # ── Volume ────────────────────────────────────────────────────────────
        vol_avg   = float(volume.rolling(20).mean().iloc[-1])
        vol_curr  = float(volume.iloc[-1])
        vol_spike = vol_curr > (vol_avg * VOLUME_SPIKE_MULT)
        vol_ratio = round(vol_curr / vol_avg, 2) if vol_avg > 0 else 1.0
        # Volume trend: increasing volume = conviction
        vol_increasing = float(volume.iloc[-1]) > float(volume.iloc[-2]) > float(volume.iloc[-3])

        # ── ATR ───────────────────────────────────────────────────────────────
        atr_series = _atr(high, low, close, 14)
        atr        = float(atr_series.iloc[-1])
        if np.isnan(atr):
            atr = price * 0.02
        # Volatility relative to price (for filtering choppy markets)
        atr_pct = (atr / price) * 100

        # ── Support & Resistance (Murphy: horizontal price levels) ───────────
        # Method: find swing highs/lows — more reliable than simple min/max
        def find_swing_levels(high_s, low_s, lookback=50, tolerance=0.005):
            """Find horizontal S/R from swing highs and lows — Murphy Ch.4"""
            h = high_s.tail(lookback).values
            l = low_s.tail(lookback).values
            swing_highs, swing_lows = [], []

            for i in range(2, len(h) - 2):
                # Swing high: higher than 2 candles on each side
                if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
                    swing_highs.append(h[i])
                # Swing low: lower than 2 candles on each side
                if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
                    swing_lows.append(l[i])

            # Cluster nearby levels (within tolerance %)
            def cluster(levels):
                if not levels: return []
                levels = sorted(levels)
                clustered = [levels[0]]
                for lv in levels[1:]:
                    if abs(lv - clustered[-1]) / clustered[-1] > tolerance:
                        clustered.append(lv)
                return clustered

            return cluster(swing_highs), cluster(swing_lows)

        swing_highs, swing_lows = find_swing_levels(high, low)

        # Find nearest S/R levels to current price
        supports    = [s for s in swing_lows   if s < price]
        resistances = [r for r in swing_highs  if r > price]

        nearest_support    = max(supports)    if supports    else float(low.tail(50).min())
        nearest_resistance = min(resistances) if resistances else float(high.tail(50).max())

        # Simple fallback for near_support/resistance flags
        support_20    = float(low.tail(20).min())
        resistance_20 = float(high.tail(20).max())

        # Near support: within 1.5% of nearest swing low (Murphy: buy near support)
        near_support    = price <= nearest_support * 1.015 if supports else price <= support_20 * 1.015
        # Near resistance: within 1.5% of nearest swing high (Murphy: sell near resistance)
        near_resistance = price >= nearest_resistance * 0.985 if resistances else price >= resistance_20 * 0.985

        # S/R strength: how many times has price touched this level?
        support_strength    = sum(1 for s in swing_lows   if abs(s - nearest_support)    / nearest_support    < 0.01)
        resistance_strength = sum(1 for r in swing_highs  if abs(r - nearest_resistance) / nearest_resistance < 0.01)
        strong_support    = bool(near_support    and support_strength    >= 2)
        strong_resistance = bool(near_resistance and resistance_strength >= 2)
        near_support      = bool(near_support)
        near_resistance   = bool(near_resistance)

        # ── ADX (Trend Strength) ──────────────────────────────────────────────
        adx = _adx(high, low, close, 14)
        trending       = adx > 25   # market is trending (not choppy)
        strong_trend   = adx > 40   # strong trend — signals more reliable

        # ── RSI Divergence ────────────────────────────────────────────────────
        divergence = _rsi_divergence(close, rsi_series)

        # ── Candlestick Patterns ──────────────────────────────────────────────
        patterns = _detect_patterns(open_, high, low, close)

        # ── VWAP ─────────────────────────────────────────────────────────────
        vwap_data = calculate_vwap(df)

        # ── Trend Bias (daily-like from 4h stack) ─────────────────────────────
        # Is the overall trend up or down? Prevents LONG in downtrend
        ema_trend_up   = ema20 > ema50 > ema200  # confirmed uptrend
        ema_trend_down = ema20 < ema50            # confirmed downtrend

        return {
            # Raw values
            "price":       price,
            "rsi":         round(rsi, 2),
            "stoch_k":     round(stoch_k_val, 2),
            "stoch_d":     round(stoch_d_val, 2),
            "macd_line":   round(macd_line, 6),
            "macd_signal": round(macd_signal, 6),
            "macd_hist":   round(macd_hist, 6),
            "ema9":        round(ema9, 6),
            "ema20":       round(ema20, 6),
            "ema50":       round(ema50, 6),
            "ema200":      round(ema200, 6),
            "bb_upper":    round(bb_upper, 6),
            "bb_mid":      round(bb_mid, 6),
            "bb_lower":    round(bb_lower, 6),
            "bb_width":    round(bb_width, 6),
            "atr":         round(atr, 8),
            "atr_pct":     round(atr_pct, 2),
            "adx":         round(adx, 1),
            "vol_ratio":   vol_ratio,
            "support":     round(nearest_support, 8),
            "resistance":  round(nearest_resistance, 8),

            # Boolean signals
            "rsi_oversold":          rsi < RSI_OVERSOLD,
            "rsi_overbought":        rsi > RSI_OVERBOUGHT,
            "rsi_neutral":           RSI_OVERSOLD <= rsi <= RSI_OVERBOUGHT,
            "stoch_oversold":        stoch_oversold,
            "stoch_overbought":      stoch_overbought,
            "stoch_bull_cross":      stoch_bull_cross,
            "stoch_bear_cross":      stoch_bear_cross,
            "macd_bull_cross":       macd_bull_cross,
            "macd_bear_cross":       macd_bear_cross,
            "macd_bullish":          macd_line > macd_signal,
            "macd_bearish":          macd_line < macd_signal,
            "macd_hist_bull":        macd_hist_growing_bull,
            "macd_hist_bear":        macd_hist_growing_bear,
            "ema_bullish":           ema_bullish,
            "ema_bearish":           ema_bearish,
            "ema_bull_stack":        ema_bull_stack,
            "ema_bear_stack":        ema_bear_stack,
            "ema_trend_up":          ema_trend_up,
            "ema_trend_down":        ema_trend_down,
            "above_ema200":          above_ema200,
            "golden_cross":          golden_cross,
            "death_cross":           death_cross,
            "bb_bounce_lower":       bb_bounce_lower,
            "bb_bounce_upper":       bb_bounce_upper,
            "bb_squeeze":            bb_squeeze,
            "vol_spike":             vol_spike,
            "vol_increasing":        vol_increasing,
            "near_support":          near_support,
            "near_resistance":       near_resistance,
            "strong_support":        strong_support,
            "strong_resistance":     strong_resistance,
            "trending":              trending,
            "strong_trend":          strong_trend,
            "bullish_divergence":    divergence["bullish_divergence"],
            "bearish_divergence":    divergence["bearish_divergence"],

            # Candlestick patterns
            "hammer":                patterns["hammer"],
            "shooting_star":         patterns["shooting_star"],
            "bullish_engulfing":     patterns["bullish_engulfing"],
            "bearish_engulfing":     patterns["bearish_engulfing"],
            "doji":                  patterns["doji"],
            "morning_star":          patterns["morning_star"],
            "evening_star":          patterns["evening_star"],
            "three_white_soldiers":  patterns["three_white_soldiers"],
            "three_black_crows":     patterns["three_black_crows"],
            "piercing_line":         patterns["piercing_line"],
            "dark_cloud_cover":      patterns["dark_cloud_cover"],
            # VWAP
            "vwap":                  vwap_data["vwap"],
            "vwap_dist_pct":         vwap_data["vwap_dist_pct"],
            "above_vwap":            vwap_data["above_vwap"],
            "below_vwap":            vwap_data["below_vwap"],
            "strong_above_vwap":     vwap_data["strong_above_vwap"],
            "strong_below_vwap":     vwap_data["strong_below_vwap"],
        }

    except Exception as e:
        logger.error("Indicator calculation failed: " + str(e))
        return {}

# =============================================================================
#  VWAP — Volume Weighted Average Price
#  Most used indicator by professional intraday traders.
#  Price above VWAP = institutions buying. Below = institutions selling.
# =============================================================================

def calculate_vwap(df: pd.DataFrame) -> dict:
    """
    Calculate VWAP for the current session (resets each day).
    Also returns distance from VWAP as a percentage.
    """
    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        vwap_val  = float(vwap.iloc[-1])
        price     = float(close.iloc[-1])
        vwap_dist = round(((price - vwap_val) / vwap_val) * 100, 2)

        above_vwap = price > vwap_val
        # Strong: price more than 1% above/below VWAP
        strong_above_vwap = vwap_dist >  1.0
        strong_below_vwap = vwap_dist < -1.0

        return {
            "vwap":             round(vwap_val, 6),
            "vwap_dist_pct":    vwap_dist,
            "above_vwap":       above_vwap,
            "below_vwap":       not above_vwap,
            "strong_above_vwap": strong_above_vwap,
            "strong_below_vwap": strong_below_vwap,
        }
    except Exception:
        return {
            "vwap": 0, "vwap_dist_pct": 0,
            "above_vwap": False, "below_vwap": False,
            "strong_above_vwap": False, "strong_below_vwap": False,
        }


# =============================================================================
#  DAILY BIAS — checks the daily (1d) candle direction
#  Call this separately with a 1d dataframe
# =============================================================================

def calculate_daily_bias(df_daily: pd.DataFrame) -> dict:
    """
    Checks the last 3 daily candles to determine macro bias.
    Returns: bullish / bearish / neutral
    Used as a top-level filter: don't LONG in daily downtrend.
    """
    try:
        if df_daily is None or len(df_daily) < 5:
            return {"daily_bias": "neutral", "daily_bullish": False, "daily_bearish": False}

        close = df_daily["close"]
        open_ = df_daily["open"]

        # Last 3 daily candles
        last3_bullish = sum(1 for i in [-3,-2,-1] if float(close.iloc[i]) > float(open_.iloc[i]))
        last3_bearish = sum(1 for i in [-3,-2,-1] if float(close.iloc[i]) < float(open_.iloc[i]))

        # Daily EMA trend
        ema20_d = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50_d = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        price_d = float(close.iloc[-1])

        daily_bullish = (last3_bullish >= 2) and (price_d > ema20_d)
        daily_bearish = (last3_bearish >= 2) and (price_d < ema20_d)

        if daily_bullish:
            bias = "bullish"
        elif daily_bearish:
            bias = "bearish"
        else:
            bias = "neutral"

        return {
            "daily_bias":     bias,
            "daily_bullish":  daily_bullish,
            "daily_bearish":  daily_bearish,
            "daily_ema20":    round(ema20_d, 6),
            "daily_ema50":    round(ema50_d, 6),
        }
    except Exception:
        return {"daily_bias": "neutral", "daily_bullish": False, "daily_bearish": False,
                "daily_ema20": 0, "daily_ema50": 0}
