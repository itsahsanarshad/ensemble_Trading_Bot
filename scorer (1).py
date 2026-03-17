# =============================================================================
#  SCORER v4 — professional confluence engine
#  FIXES: BTC cache staleness, dead cat bounce filter
#  NEW: sector correlation limit
# =============================================================================

import logging
import time
from config import TIMEFRAMES
from indicators import calculate_all, calculate_daily_bias, calculate_vwap
from data_fetcher import get_candles

logger = logging.getLogger(__name__)

# =============================================================================
#  COIN SECTORS — for correlation limiting
# =============================================================================
SECTORS = {
    # Layer 1s
    "BTCUSDT":    "btc",
    "ETHUSDT":    "eth",
    "BNBUSDT":    "l1",
    "SOLUSDT":    "l1",
    "ADAUSDT":    "l1",
    "AVAXUSDT":   "l1",
    "DOTUSDT":    "l1",
    "ATOMUSDT":   "l1",
    "NEARUSDT":   "l1",
    "APTUSDT":    "l1",
    "SUIUSDT":    "l1",
    "SEIUSDT":    "l1",
    "ICPUSDT":    "l1",
    "ALGOUSDT":   "l1",
    "TRXUSDT":    "l1",
    "VETUSDT":    "l1",
    # Layer 2s
    "ARBUSDT":    "l2",
    "OPUSDT":     "l2",
    "MATICUSDT":  "l2",
    # DeFi
    "UNIUSDT":    "defi",
    "AAVEUSDT":   "defi",
    "LDOUSDT":    "defi",
    "MKRUSDT":    "defi",
    "CRVUSDT":    "defi",
    "RUNEUSDT":   "defi",
    # AI / GPU
    "FETUSDT":    "ai",
    "RENDERUSDT": "ai",
    "TAOUSDT":    "ai",
    # Meme coins
    "DOGEUSDT":   "meme",
    "SHIBUSDT":   "meme",
    "PEPEUSDT":   "meme",
    "BONKUSDT":   "meme",
    "WIFUSDT":    "meme",
    "TRUMPUSDT":  "meme",
    # Other
    "XRPUSDT":    "payments",
    "LTCUSDT":    "payments",
    "BCHUSDT":    "payments",
    "LINKUSDT":   "oracle",
    "INJUSDT":    "exchange",
    "THETAUSDT":  "media",
    "FILUSDT":    "storage",
    "ONDOUSDT":   "rwa",
    "FTMUSDT":    "l1",
    "STXUSDT":    "btc_layer",
    "POPCATUSDT": "meme",
    "STXUSDT":    "btc_layer",
}

# =============================================================================
#  SIGNAL WEIGHTS — rebalanced based on backtest analysis
#
#  Key insight from backtest:
#  - RSI oversold/overbought had low predictive value in trending markets
#  - Volume spike + VWAP are most reliable real-time confirmation
#  - EMA alignment (trend confirmation) is more reliable than oscillators
#  - MACD cross is reliable but needs volume confirmation
#  - Support/resistance near price = strongest reversal signal
#
#  Changes:
#  - RSI: 15 → 10 (good but overweighted, often false in trends)
#  - MACD cross: 15 → 18 (more reliable when it fires)
#  - Volume spike: 10 → 15 (strongest real-time confirmation)
#  - VWAP: 7 → 12 (institutions use this, very reliable)
#  - EMA bullish: 10 → 12 (trend confirmation = more weight)
#  - Near support: 8 → 10 (Murphy: most reliable reversal zone)
#  - BB bounce: 10 → 8 (less reliable in trending markets)
# =============================================================================
LONG_WEIGHTS = {
    "macd_bull_cross":    18,   # ↑ most reliable momentum signal
    "vol_spike":          15,   # ↑ volume = real money moving
    "above_vwap":         12,   # ↑ institutions buying above VWAP
    "ema_bullish":        12,   # ↑ trend confirmed = higher weight
    "near_support":       10,   # ↑ Murphy: buy near support
    "rsi_oversold":       10,   # ↓ good but often false in downtrend
    "bb_bounce_lower":     8,   # ↓ less reliable in trends
    "macd_hist_bull":      6,   # momentum building
    "stoch_bull_cross":    5,   # secondary confirmation
    "stoch_oversold":      4,   # tertiary
    "above_ema200":        4,   # long-term trend
    "bullish_divergence":  4,   # often false, low weight
}

SHORT_WEIGHTS = {
    "macd_bear_cross":    18,   # ↑ most reliable momentum signal
    "vol_spike":          15,   # ↑ volume = real selling
    "below_vwap":         12,   # ↑ institutions selling below VWAP
    "ema_bearish":        12,   # ↑ trend confirmed
    "near_resistance":    10,   # ↑ Murphy: sell near resistance
    "rsi_overbought":     10,   # ↓ often false in uptrend
    "bb_bounce_upper":     8,   # ↓ less reliable in trends
    "macd_hist_bear":      6,   # momentum building
    "stoch_bear_cross":    5,   # secondary confirmation
    "stoch_overbought":    4,   # tertiary
    "above_ema200":        4,   # price above = harder to short
    "bearish_divergence":  4,   # often false, low weight
}

LONG_BONUS = {
    "ema_bull_stack":        12,  # ↑ price>ema9>ema20>ema50 = perfect alignment
    "golden_cross":          10,  # ema20 crosses above ema50 = major signal
    "morning_star":          10,  # Nison: strongest 3-candle reversal
    "three_white_soldiers":  10,  # Nison: 3 green candles = momentum
    "strong_support":        10,  # ↑ multiple touches = very reliable
    "bullish_engulfing":      8,  # strong single candle reversal
    "hammer":                 8,  # rejection of lower prices
    "strong_above_vwap":      7,  # ↑ strong VWAP position
    "piercing_line":          6,
    "vol_increasing":         5,  # volume trend confirming
    "bb_squeeze":             4,  # ↓ breakout potential but direction unclear
    "strong_trend":           4,
}

SHORT_BONUS = {
    "ema_bear_stack":        12,  # ↑ price<ema9<ema20<ema50 = perfect alignment
    "death_cross":           10,  # ema20 crosses below ema50 = major signal
    "evening_star":          10,  # Nison: strongest bearish reversal
    "three_black_crows":     10,  # Nison: 3 red candles = momentum
    "strong_resistance":     10,  # ↑ multiple touches = very reliable
    "bearish_engulfing":      8,  # strong single candle reversal
    "shooting_star":          8,  # rejection of higher prices
    "strong_below_vwap":      7,  # ↑ strong VWAP position
    "dark_cloud_cover":       6,
    "vol_increasing":         5,
    "bb_squeeze":             4,
    "strong_trend":           4,
}


def _auto_threshold(fng_value: int, direction: str = "LONG") -> int:
    """
    Thresholds based on backtest v4 (1745 signals with all 6 filters):
    - Overall: 43.2% WR ✅
    - SHORT: 46.4% WR (better historically)
    - LONG:  38.3% WR (better in current Extreme Fear market)

    In Extreme Fear (current market Mar 2026):
    - SHORT collapsed to 28.6% — raise threshold
    - LONG improved to 47.2% — lower threshold to catch more
    """
    from config import SHORT_MIN_SCORE, LONG_ONLY_MODE

    if direction == "SHORT" and LONG_ONLY_MODE:
        return 999

    if fng_value < 20:
        # Extreme Fear: LONG works better, SHORT is unreliable
        base = 75 if direction == "LONG" else 92
    elif fng_value < 30:
        # Fear: both decent, keep standard
        base = 78
    elif fng_value < 50:
        # Neutral: best market — lower threshold
        base = 78
    elif fng_value < 70:
        base = 80
    elif fng_value < 85:
        base = 84
    else:
        base = 90   # Extreme Greed

    if direction == "SHORT":
        return max(base, SHORT_MIN_SCORE)

    return base


def score_symbol(indicators: dict, direction: str) -> tuple:
    weights = LONG_WEIGHTS if direction == "LONG" else SHORT_WEIGHTS
    bonuses = LONG_BONUS   if direction == "LONG" else SHORT_BONUS
    score = 0
    fired = []

    for name, weight in weights.items():
        if name == "above_ema200" and direction == "SHORT":
            if not indicators.get(name, False):
                score += weight
                fired.append("below_ema200")
            continue
        if indicators.get(name, False):
            score += weight
            fired.append(name)

    for name, pts in bonuses.items():
        if indicators.get(name, False):
            score += pts
            fired.append(name + " [+]")

    return score, fired


def _trend_filter(indicators_4h: dict, direction: str) -> tuple:
    if direction == "LONG":
        if indicators_4h.get("ema_bear_stack", False):
            return False, "4h EMA stack bearish"
        if indicators_4h.get("death_cross", False):
            return False, "4h death cross active"
    else:
        if indicators_4h.get("ema_bull_stack", False):
            return False, "4h EMA stack bullish"
    return True, "OK"


def _dead_cat_bounce_filter(indicators_1h: dict, indicators_4h: dict, direction: str) -> tuple:
    """
    Detects dead cat bounces — short relief rallies in a downtrend.
    A LONG signal is a dead cat bounce if:
    - 1h looks bullish BUT 4h is still bearish
    - RSI bounced from oversold but 4h RSI still below 45
    - Volume on bounce is LOWER than average (weak bounce)
    """
    if direction != "LONG":
        return True, "OK"

    rsi_1h = indicators_1h.get("rsi", 50)
    rsi_4h = indicators_4h.get("rsi", 50)
    vol_ratio = indicators_1h.get("vol_ratio", 1.0)
    ema_4h_bearish = indicators_4h.get("ema_bearish", False)
    macd_4h_bearish = indicators_4h.get("macd_bearish", False)

    # Classic dead cat: 1h bouncing but 4h still in downtrend + weak volume
    if ema_4h_bearish and macd_4h_bearish and vol_ratio < 1.2:
        return False, "dead cat bounce — 4h bearish + weak volume"

    # RSI bounce in downtrend: RSI recovered on 1h but 4h RSI still depressed
    if rsi_1h > 45 and rsi_4h < 42 and ema_4h_bearish:
        return False, "RSI dead cat — 4h RSI=" + str(round(rsi_4h, 1)) + " still bearish"

    return True, "OK"


# =============================================================================
#  BTC BIAS — fresh every scan, not cached aggressively
# =============================================================================
_btc_cache = {"bias": "neutral", "timestamp": 0}
BTC_CACHE_SECONDS = 240   # refresh every 4 minutes (was every 5 scans = 25min — too stale)

def get_btc_bias() -> str:
    now = time.time()
    if now - _btc_cache["timestamp"] > BTC_CACHE_SECONDS:
        try:
            df_1h = get_candles("BTCUSDT", "1h")
            df_4h = get_candles("BTCUSDT", "4h")
            if df_1h is None or df_4h is None:
                return _btc_cache["bias"]

            ind_1h = calculate_all(df_1h)
            ind_4h = calculate_all(df_4h)

            # Both 1h AND 4h must agree for a strong bias
            btc_bull = ind_1h.get("ema_bullish") and ind_4h.get("ema_bullish") and ind_4h.get("macd_bullish")
            btc_bear = ind_1h.get("ema_bearish") and ind_4h.get("macd_bearish")

            if btc_bull:
                bias = "bullish"
            elif btc_bear:
                bias = "bearish"
            else:
                bias = "neutral"

            _btc_cache["bias"] = bias
            _btc_cache["timestamp"] = now
            logger.info("BTC bias refreshed: " + bias +
                        "  1h_rsi=" + str(round(ind_1h.get("rsi", 0), 1)) +
                        "  4h_rsi=" + str(round(ind_4h.get("rsi", 0), 1)))
        except Exception:
            pass

    return _btc_cache["bias"]


# =============================================================================
#  MAIN ANALYSIS
# =============================================================================

def analyze_symbol(symbol: str, fng_value: int = 50) -> dict | None:
    from notifier import log_scan_activity
    tf_results = {}
    for tf in TIMEFRAMES:
        df = get_candles(symbol, tf)
        if df is None:
            return None
        ind = calculate_all(df)
        if not ind:
            return None
        tf_results[tf] = ind

    df_daily = get_candles(symbol, "1d")
    daily    = calculate_daily_bias(df_daily) if df_daily is not None else {
        "daily_bias": "neutral", "daily_bullish": False, "daily_bearish": False
    }

    primary = tf_results[TIMEFRAMES[1]]   # 1h
    tf_15m  = tf_results[TIMEFRAMES[0]]
    tf_4h   = tf_results[TIMEFRAMES[2]]

    long_score,  long_fired  = score_symbol(primary, "LONG")
    short_score, short_fired = score_symbol(primary, "SHORT")

    direction = "LONG"  if long_score >= short_score else "SHORT"
    score     = long_score if direction == "LONG" else short_score
    fired     = long_fired if direction == "LONG" else short_fired

    # helper to log and return None cleanly
    def _block(reason: str, status: str):
        log_scan_activity(
            symbol=symbol, direction=direction, score=score,
            threshold=_auto_threshold(fng_value), status=status,
            block_reason=reason,
            rsi=primary.get("rsi", 0), adx=primary.get("adx", 0),
            vwap_pct=primary.get("vwap_dist_pct", 0),
            daily_bias=daily["daily_bias"], fng_value=fng_value,
        )
        logger.debug(f"{symbol} blocked: {reason}")
        return None

    # ── Trend filter (4h) ─────────────────────────────────────────────────────
    passed, reason = _trend_filter(tf_4h, direction)
    if not passed:
        return _block(reason, "BLOCKED_TREND")

    # ── Dead cat bounce filter ────────────────────────────────────────────────
    passed, reason = _dead_cat_bounce_filter(primary, tf_4h, direction)
    if not passed:
        return _block(reason, "BLOCKED_DEADCAT")

    # ── FIX 2: Hard daily bias filter — never trade against the daily trend ───
    # Backtest showed trading against daily trend is the #1 cause of losses
    # LONG in daily downtrend = fighting a waterfall
    # SHORT in daily uptrend  = fighting a rocket
    if direction == "LONG" and daily["daily_bearish"]:
        return _block("daily bearish — hard block, not trading against trend", "BLOCKED_DAILY")
    if direction == "SHORT" and daily["daily_bullish"]:
        return _block("daily bullish — hard block, not trading against trend", "BLOCKED_DAILY")
    # Also block neutral daily for SHORT — SHORT only fires with confirmed bearish daily
    if direction == "SHORT" and daily["daily_bias"] == "neutral":
        return _block("daily neutral — SHORT requires confirmed bearish daily", "BLOCKED_DAILY")

    # ── FIX 1: ADX range filter — sweet spot 15-28 ───────────────────────────
    # Too low (< 15) = no trend, random price = indicators unreliable
    # Too high (> 28) = strong trend = lagging indicators miss the move
    # Sweet spot 15-28 = mild trend = indicators work best
    from config import ADX_MIN, ADX_MAX
    adx = primary.get("adx", 20)
    if adx < ADX_MIN:
        return _block(f"ADX={round(adx,1)}<{ADX_MIN} — too choppy, no trend", "BLOCKED_CHOPPY")
    if adx > ADX_MAX:
        return _block(f"ADX={round(adx,1)}>{ADX_MAX} — strong trend, indicators lag", "BLOCKED_TRENDING")

    # ── BTC correlation ────────────────────────────────────────────────────────
    btc_bias = "neutral"
    if symbol != "BTCUSDT":
        btc_bias = get_btc_bias()
        if direction == "LONG" and btc_bias == "bearish" and score < 78:
            log_scan_activity(
                symbol=symbol, direction=direction, score=score,
                threshold=_auto_threshold(fng_value, direction), status="BLOCKED_BTC",
                block_reason=f"BTC bearish, score {score}<78",
                rsi=primary.get("rsi",0), adx=adx,
                vwap_pct=primary.get("vwap_dist_pct",0),
                daily_bias=daily["daily_bias"], fng_value=fng_value,
            )
        if direction == "SHORT" and btc_bias == "bullish" and score < 78:
            log_scan_activity(
                symbol=symbol, direction=direction, score=score,
                threshold=_auto_threshold(fng_value, direction), status="BLOCKED_BTC",
                block_reason=f"BTC bullish, score {score}<78",
                rsi=primary.get("rsi",0), adx=adx,
                vwap_pct=primary.get("vwap_dist_pct",0),
                daily_bias=daily["daily_bias"], fng_value=fng_value,
            )

    # ── FIX 3: Require BOTH 15m AND 4h to agree — hard filter ─────────────────
    # Old: 15m OR 4h agrees = bonus points (soft)
    # New: BOTH must agree = hard requirement
    # This ensures short-term and long-term momentum both confirm the trade
    if direction == "LONG":
        tf_15m_agrees = bool(
            tf_15m.get("macd_bullish") or tf_15m.get("rsi_oversold") or
            tf_15m.get("stoch_oversold") or tf_15m.get("above_vwap")
        )
        tf_4h_agrees = bool(
            tf_4h.get("ema_bullish") or tf_4h.get("macd_bullish") or tf_4h.get("above_vwap")
        )
    else:
        tf_15m_agrees = bool(
            tf_15m.get("macd_bearish") or tf_15m.get("rsi_overbought") or
            tf_15m.get("stoch_overbought") or tf_15m.get("below_vwap")
        )
        tf_4h_agrees = bool(
            tf_4h.get("ema_bearish") or tf_4h.get("macd_bearish") or tf_4h.get("below_vwap")
        )

    # Both timeframes must agree — hard block if either disagrees
    if not tf_15m_agrees:
        return _block("15m not aligned with direction", "BLOCKED_TREND")
    if not tf_4h_agrees:
        return _block("4h not aligned with direction", "BLOCKED_TREND")

    # Both agreed — small bonus for strong alignment
    score += 8
    # Cap at 97 — backtest showed score 98-100 has WORST WR (30%)
    # When ALL indicators fire simultaneously the move has already happened
    score = min(score, 97)

    # ── Choppy market filter ──────────────────────────────────────────────────
    if primary.get("atr_pct", 1.0) < 0.3:
        return _block("too choppy (ATR%<0.3)", "BLOCKED_CHOPPY")

    threshold = _auto_threshold(fng_value, direction)

    logger.info(
        "  " + symbol.ljust(12) + " " + direction.ljust(5) +
        " score=" + str(score).rjust(3) +
        "  RSI=" + str(round(primary.get("rsi", 0), 1)) +
        "  ADX=" + str(round(adx, 1)) +
        "  VWAP=" + str(primary.get("vwap_dist_pct", 0)) + "%" +
        "  daily=" + daily["daily_bias"] +
        "  thr=" + str(threshold)
    )

    if score < threshold:
        return _block(f"score {score} < threshold {threshold}", "BLOCKED_SCORE")

    return {
        "symbol":     symbol,
        "direction":  direction,
        "score":      score,
        "fired":      fired,
        "threshold":  threshold,
        "daily_bias": daily["daily_bias"],
        "btc_bias":   btc_bias,
        "sector":     SECTORS.get(symbol, "other"),
        "tf_align": {
            "15m": tf_15m_agrees,
            "1h":  True,
            "4h":  tf_4h_agrees,
        },
        "indicators": primary,
    }
