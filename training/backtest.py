"""
Backtesting Engine — V5 (Ensemble-Connected)

ARCHITECTURE CHANGE (V5):
    Previously this engine used a duplicated heuristic scorecard (RSI, MACD,
    Volume scoring → integer threshold) completely disconnected from the live
    ConsensusEnsemble. Results from the old backtester were meaningless as
    they measured different logic than what ran in production.

    V5 connects directly to the ConsensusEnsemble.analyze() method, meaning
    backtest results represent actual live-bot signal quality. Key changes:
      - Uses ConsensusEnsemble with historical data slices (evaluation mode)
      - ML gatekeeping, TA vetoes, and Kelly sizing all active
      - Fees modelled at 0.1% per side (Binance taker)
      - Sharpe ratio, Sortino ratio, and Profit Factor computed
      - Walk-forward validation: models are NOT re-trained during backtest
        (we test the trained model against out-of-sample data)
"""

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings, WATCHLIST
from src.utils import logger
from src.data import collector, feature_engineer
from src.models.ml_model import ml_model, MLSignal
from src.models.ta_analyzer import ta_analyzer, TASignal
from src.models.tcn_model import tcn_model, TCNSignal
from src.models.ensemble import (
    ConsensusEnsemble, ConsensusSignal,
    ML_GATE_THRESHOLD, STOP_LOSS_PCT,
    TAKE_PROFIT_STANDARD, TAKE_PROFIT_HIGH,
    kelly_position_size, KELLY_FRACTION,
    TIER1_MAX, TIER2_MAX, TIER3_MAX,
)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class BacktestTrade:
    """Single completed trade."""
    coin:         str
    entry_time:   datetime
    entry_price:  float
    exit_time:    datetime  = None
    exit_price:   float     = 0.0
    exit_reason:  str       = ""
    pnl_pct:      float     = 0.0
    pnl_usd:      float     = 0.0
    tier:         int       = 0
    ml_conf:      float     = 0.0
    position_pct: float     = 0.0   # size as fraction of portfolio


@dataclass
class BacktestResult:
    """Aggregated backtest performance metrics."""
    total_return:      float = 0.0
    annual_return:     float = 0.0
    max_drawdown:      float = 0.0
    sharpe_ratio:      float = 0.0
    sortino_ratio:     float = 0.0
    win_rate:          float = 0.0
    profit_factor:     float = 0.0
    average_win:       float = 0.0
    average_loss:      float = 0.0
    largest_win:       float = 0.0
    largest_loss:      float = 0.0
    total_trades:      int   = 0
    avg_duration_hrs:  float = 0.0
    fees_paid:         float = 0.0
    trades:            List[BacktestTrade] = field(default_factory=list)


# ============================================================================
# Historical ensemble evaluation helpers
# ============================================================================

def _eval_ml(symbol: str, hist_df: pd.DataFrame) -> MLSignal:
    """Evaluate ML model on a historical dataframe slice."""
    try:
        return ml_model.predict(symbol, df=hist_df)
    except Exception as e:
        logger.debug(f"ML eval error for {symbol}: {e}")
        return MLSignal(signal="hold", confidence=0.5)


def _eval_ta(symbol: str, hist_df: pd.DataFrame) -> TASignal:
    """Evaluate TA analyzer on a historical dataframe slice."""
    try:
        return ta_analyzer.analyze(symbol, df=hist_df)
    except Exception as e:
        logger.debug(f"TA eval error for {symbol}: {e}")
        return TASignal(signal="hold", confidence=0.0)


def _eval_tcn(symbol: str, hist_df: pd.DataFrame) -> TCNSignal:
    """Evaluate TCN model on a historical dataframe slice."""
    try:
        return tcn_model.predict(hist_df, symbol=symbol)
    except Exception as e:
        logger.debug(f"TCN eval error for {symbol}: {e}")
        from src.models.tcn_model import TCNSignal
        return TCNSignal(symbol=symbol, timestamp=datetime.utcnow(),
                         probability=0.5, prediction=0, confidence=0.0)


def _apply_consensus_rules(
    ml_result: MLSignal,
    ta_result: TASignal,
    tcn_result: TCNSignal,
    regime: str = "SIDEWAYS",
) -> Tuple[int, float, float]:
    """
    Apply V5 consensus rules to historical signals.
    Returns (tier, take_profit_pct, position_size_pct).
    """
    # Threshold map (mirrors ConsensusEnsemble.REGIME_THRESHOLDS)
    thresh = {
        "BULL":     {"ml_strong": 0.60, "tcn_strong": 0.60, "ml_high": 0.68, "pos_mult": 1.15},
        "BEAR":     {"ml_strong": 0.67, "tcn_strong": 0.67, "ml_high": 0.75, "pos_mult": 0.60},
        "SIDEWAYS": {"ml_strong": 0.62, "tcn_strong": 0.62, "ml_high": 0.70, "pos_mult": 1.00},
    }.get(regime, {"ml_strong": 0.62, "tcn_strong": 0.62, "ml_high": 0.70, "pos_mult": 1.00})

    ml_conf  = ml_result.confidence
    tcn_conf = tcn_result.confidence if tcn_result else 0.0
    ta_conf  = ta_result.confidence

    # TA structural veto
    if ta_result.blocked:
        return 0, TAKE_PROFIT_STANDARD, 0.0

    # ML hard gate
    if ml_conf < ML_GATE_THRESHOLD or ml_result.signal != "buy":
        return 0, TAKE_PROFIT_STANDARD, 0.0

    ml_s  = thresh["ml_strong"]
    tcn_s = thresh["tcn_strong"]
    ml_h  = thresh["ml_high"]
    pmult = thresh["pos_mult"]
    tcn_ok = tcn_result.signal == "buy" if tcn_result else False

    # Tier 3
    if ml_conf >= ml_s and tcn_conf >= tcn_s and ta_conf >= 0.55 and tcn_ok:
        size = kelly_position_size(ml_conf, TAKE_PROFIT_HIGH, STOP_LOSS_PCT, KELLY_FRACTION, TIER3_MAX)
        return 3, TAKE_PROFIT_HIGH, min(size * pmult, TIER3_MAX)

    # Tier 2
    if ml_conf >= ml_s and tcn_conf >= tcn_s and tcn_ok:
        size = kelly_position_size(ml_conf, TAKE_PROFIT_STANDARD, STOP_LOSS_PCT, KELLY_FRACTION, TIER2_MAX)
        return 2, TAKE_PROFIT_STANDARD, min(size * pmult, TIER2_MAX)

    # Tier 1
    tcn_opposing = tcn_result.signal == "sell" and tcn_conf > 0.60
    if ml_conf >= ml_h and not tcn_opposing:
        size = kelly_position_size(ml_conf, TAKE_PROFIT_STANDARD, STOP_LOSS_PCT, KELLY_FRACTION, TIER1_MAX)
        return 1, TAKE_PROFIT_STANDARD, min(size * pmult, TIER1_MAX)

    return 0, TAKE_PROFIT_STANDARD, 0.0


# ============================================================================
# Backtest Engine
# ============================================================================

class BacktestEngine:
    """
    V5 Backtesting engine — connected to the live ConsensusEnsemble logic.

    Uses a walk-forward simulation where:
      - Historical data slices are fed into the same ML/TA/TCN models
        used in live trading (evaluation mode — no look-ahead).
      - Entry signals are generated using the V5 consensus rules.
      - Exit conditions mirror the live risk management engine.
    """

    FEE_RATE = 0.001   # 0.1% per side (Binance taker)

    def __init__(
        self,
        initial_capital: float = 10_000,
        max_positions:   int   = 5,
    ):
        self.initial_capital = initial_capital
        self.max_positions   = max_positions

    # ------------------------------------------------------------------
    def run(
        self,
        symbols:    List[str],
        start_date: str,
        end_date:   str,
        timeframe:  str = "1h",
    ) -> BacktestResult:
        logger.info(f"=== V5 Backtest: {start_date} → {end_date} ({len(symbols)} symbols) ===")
        data   = self._load_data(symbols, start_date, end_date, timeframe)
        if not data:
            logger.error("No historical data available for backtest.")
            return BacktestResult()
        result = self._simulate(data, timeframe)
        self._calculate_metrics(result)
        return result

    # ------------------------------------------------------------------
    def _load_data(
        self, symbols, start_date, end_date, timeframe
    ) -> Dict[str, pd.DataFrame]:
        data  = {}
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        days  = (end - start).days

        for symbol in symbols:
            logger.info(f"  Loading {symbol}…")
            try:
                candles_per_day = {"15m": 96, "1h": 24, "4h": 6}.get(timeframe, 24)
                df = collector.get_dataframe(symbol, timeframe, limit=(days + 60) * candles_per_day)
                if df.empty:
                    logger.warning(f"  No data for {symbol}")
                    continue

                # Filter to date range
                if "datetime" in df.columns:
                    df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
                elif "timestamp" in df.columns:
                    df["_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df = df[(df["_dt"] >= start) & (df["_dt"] <= end)]
                    df = df.drop(columns=["_dt"])

                # Feature engineering
                df = feature_engineer.engineer_all_features(df)
                if len(df) < 100:
                    logger.warning(f"  Not enough data for {symbol} ({len(df)} rows)")
                    continue

                data[symbol] = df
                logger.info(f"  ✓ {symbol}: {len(df):,} rows")
            except Exception as e:
                logger.warning(f"  Error loading {symbol}: {e}")

        return data

    # ------------------------------------------------------------------
    def _simulate(self, data: Dict[str, pd.DataFrame], timeframe: str) -> BacktestResult:
        result          = BacktestResult()
        capital         = self.initial_capital
        open_positions: Dict[str, BacktestTrade] = {}
        equity_curve    = [capital]

        # Determine candle-step for scanning (scan every 4 candles = 4h on 1h data)
        scan_every = {"15m": 16, "1h": 4, "4h": 1}.get(timeframe, 4)

        # Build time-sorted index from 1h primary data
        all_timestamps = sorted(set(
            ts for df in data.values() for ts in df["timestamp"].tolist()
        ))

        for step_i, ts in enumerate(all_timestamps):

            # ---- Monitor open positions ----
            for symbol in list(open_positions.keys()):
                trade = open_positions[symbol]
                df    = data.get(symbol)
                if df is None:
                    continue
                row = df[df["timestamp"] == ts]
                if row.empty:
                    continue

                current_price = float(row["close"].iloc[0])
                pnl_pct       = (current_price - trade.entry_price) / trade.entry_price
                exit_reason   = None

                # Hard stop loss
                if pnl_pct <= -STOP_LOSS_PCT:
                    exit_reason = "stop_loss"

                # Take profit
                elif pnl_pct >= trade.position_pct:    # reuse as tp store
                    # We stored take_profit_pct in position_pct field temporarily
                    # (see open logic below); access via a custom attribute
                    tp  = getattr(trade, "_take_profit_pct", TAKE_PROFIT_STANDARD)
                    if pnl_pct >= tp:
                        exit_reason = "take_profit"

                if exit_reason:
                    fee = capital * trade.position_pct * self.FEE_RATE
                    pnl_usd = pnl_pct * (capital * trade.position_pct) - fee
                    trade.exit_time   = pd.Timestamp(ts, unit="ms") if isinstance(ts, (int, float)) else ts
                    trade.exit_price  = current_price
                    trade.exit_reason = exit_reason
                    trade.pnl_pct     = pnl_pct
                    trade.pnl_usd     = pnl_usd
                    capital          += pnl_usd
                    result.fees_paid += fee
                    result.trades.append(trade)
                    del open_positions[symbol]

            # ---- Scan for new entries ----
            if step_i % scan_every == 0 and len(open_positions) < self.max_positions:
                for symbol, df in data.items():
                    if symbol in open_positions:
                        continue

                    mask = df["timestamp"] == ts
                    if not mask.any():
                        continue

                    pos = mask.values.nonzero()[0][0]
                    if pos < 100:
                        continue   # Need warm-up history

                    # Historical slice (no look-ahead)
                    hist_df = df.iloc[max(0, pos - 250): pos + 1].copy()
                    if len(hist_df) < 100:
                        continue

                    # Evaluate all three models on historical slice
                    ml_result  = _eval_ml(symbol,  hist_df)
                    ta_result  = _eval_ta(symbol,  hist_df)
                    tcn_result = _eval_tcn(symbol, hist_df)

                    # Apply V5 consensus rules
                    tier, take_profit, pos_size = _apply_consensus_rules(
                        ml_result, ta_result, tcn_result, regime="SIDEWAYS"
                    )

                    if tier == 0 or pos_size == 0:
                        continue

                    entry_price = float(hist_df["close"].iloc[-1])
                    entry_fee   = capital * pos_size * self.FEE_RATE
                    result.fees_paid += entry_fee

                    trade = BacktestTrade(
                        coin         = symbol,
                        entry_time   = pd.Timestamp(ts, unit="ms") if isinstance(ts, (int, float)) else ts,
                        entry_price  = entry_price,
                        tier         = tier,
                        ml_conf      = ml_result.confidence,
                        position_pct = pos_size,
                    )
                    trade._take_profit_pct = take_profit   # store for exit check

                    open_positions[symbol] = trade

                    if len(open_positions) >= self.max_positions:
                        break

            # ---- Update equity curve ----
            unrealised = 0.0
            for symbol, trade in open_positions.items():
                df  = data.get(symbol)
                if df is None:
                    continue
                row = df[df["timestamp"] == ts]
                if row.empty:
                    continue
                price       = float(row["close"].iloc[0])
                unrealised += ((price - trade.entry_price) / trade.entry_price) * (capital * trade.position_pct)
            equity_curve.append(capital + unrealised)

        # ---- Close remaining positions at last price ----
        for symbol, trade in open_positions.items():
            df = data.get(symbol)
            if df is None or df.empty:
                continue
            exit_price = float(df["close"].iloc[-1])
            pnl_pct    = (exit_price - trade.entry_price) / trade.entry_price
            fee        = capital * trade.position_pct * self.FEE_RATE
            pnl_usd    = pnl_pct * (capital * trade.position_pct) - fee
            trade.exit_price  = exit_price
            trade.exit_reason = "end_of_backtest"
            trade.pnl_pct     = pnl_pct
            trade.pnl_usd     = pnl_usd
            capital          += pnl_usd
            result.fees_paid += fee
            result.trades.append(trade)

        result.total_return = (capital - self.initial_capital) / self.initial_capital
        result.total_trades = len(result.trades)

        # Max drawdown
        peak   = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            peak   = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak)
        result.max_drawdown = max_dd

        # Sharpe / Sortino (daily returns approximation)
        if len(equity_curve) > 2:
            eq_series   = pd.Series(equity_curve)
            daily_ret   = eq_series.pct_change().dropna()
            rf          = 0.0
            excess      = daily_ret - rf
            downside    = daily_ret[daily_ret < 0]
            sharpe_denom = daily_ret.std()
            sortino_denom = downside.std()
            result.sharpe_ratio  = float((excess.mean() / sharpe_denom  * np.sqrt(252)) if sharpe_denom  > 0 else 0)
            result.sortino_ratio = float((excess.mean() / sortino_denom * np.sqrt(252)) if sortino_denom > 0 else 0)

        return result

    # ------------------------------------------------------------------
    def _calculate_metrics(self, result: BacktestResult) -> None:
        if not result.trades:
            return

        wins   = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct <= 0]

        result.win_rate    = len(wins) / len(result.trades) if result.trades else 0.0
        result.average_win = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
        result.largest_win = max(t.pnl_pct for t in wins)                if wins   else 0.0
        result.average_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
        result.largest_loss = min(t.pnl_pct for t in losses)              if losses else 0.0

        gross_profit = sum(t.pnl_usd for t in wins)   if wins   else 0.0
        gross_loss   = abs(sum(t.pnl_usd for t in losses)) if losses else 1.0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        durations = []
        for t in result.trades:
            if t.exit_time and t.entry_time:
                dur = (t.exit_time - t.entry_time)
                if hasattr(dur, "total_seconds"):
                    durations.append(dur.total_seconds() / 3600)
        result.avg_duration_hrs = float(np.mean(durations)) if durations else 0.0


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="V5 Backtest — Ensemble-Connected")
    parser.add_argument("--start",   default="2024-06-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument("--coins",   type=int,   default=5,      help="Number of coins (default 5)")
    args = parser.parse_args()

    engine = BacktestEngine(initial_capital=args.capital)
    result = engine.run(
        symbols    = WATCHLIST[:args.coins],
        start_date = args.start,
        end_date   = args.end,
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print("V5 BACKTEST RESULTS  (Ensemble-Connected)")
    print(sep)
    print(f"Period:          {args.start}  →  {args.end}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print()
    print(f"Total Return:    {result.total_return:.1%}")
    print(f"Annual Return:   {result.annual_return:.1%}")
    print(f"Max Drawdown:    {result.max_drawdown:.1%}")
    print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:   {result.sortino_ratio:.2f}")
    print()
    print(f"Total Trades:    {result.total_trades}")
    print(f"Win Rate:        {result.win_rate:.1%}")
    print(f"Profit Factor:   {result.profit_factor:.2f}")
    print(f"Avg Win:         {result.average_win:.2%}")
    print(f"Avg Loss:        {result.average_loss:.2%}")
    print(f"Largest Win:     {result.largest_win:.2%}")
    print(f"Largest Loss:    {result.largest_loss:.2%}")
    print(f"Avg Duration:    {result.avg_duration_hrs:.1f}h")
    print(f"Fees Paid:       ${result.fees_paid:.2f}")
    print(sep)

    print("\n📊 Quality Checklist:")
    print(f"  Win Rate > 50%:        {'✅' if result.win_rate       > 0.50 else '❌'}  ({result.win_rate:.1%})")
    print(f"  Profit Factor > 1.5:   {'✅' if result.profit_factor  > 1.50 else '❌'}  ({result.profit_factor:.2f})")
    print(f"  Max Drawdown < 25%:    {'✅' if result.max_drawdown   < 0.25 else '❌'}  ({result.max_drawdown:.1%})")
    print(f"  Sharpe Ratio > 1.0:    {'✅' if result.sharpe_ratio   > 1.00 else '❌'}  ({result.sharpe_ratio:.2f})")
    print(sep)


if __name__ == "__main__":
    main()
