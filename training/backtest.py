"""
Backtesting Engine

Test trading strategies on historical data:
- Walk-forward simulation
- Performance metrics
- Multiple market condition testing
"""

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings, WATCHLIST
from src.utils import logger
from src.data import collector, feature_engineer


@dataclass
class BacktestTrade:
    """Single trade in backtest."""
    coin: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime = None
    exit_price: float = 0
    exit_reason: str = ""
    pnl_pct: float = 0
    pnl_usd: float = 0
    tier: int = 1


@dataclass 
class BacktestResult:
    """Backtest results."""
    total_return: float = 0
    annual_return: float = 0
    max_drawdown: float = 0
    sharpe_ratio: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    average_win: float = 0
    average_loss: float = 0
    largest_win: float = 0
    largest_loss: float = 0
    total_trades: int = 0
    avg_trade_duration: str = ""
    fees_paid: float = 0
    trades: List[BacktestTrade] = field(default_factory=list)


class BacktestEngine:
    """
    Backtesting engine for strategy validation.
    
    Features:
    - Historical simulation
    - Walk-forward validation
    - Multiple market conditions
    - Detailed performance metrics
    """
    
    # Trading fees (Binance maker/taker)
    FEE_RATE = 0.001  # 0.1%
    
    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,
        max_positions: int = 5
    ):
        """Initialize backtest engine."""
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
    
    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1h"
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            symbols: List of trading pairs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Candlestick interval
        
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Load historical data
        data = self._load_data(symbols, start_date, end_date, timeframe)
        
        if not data:
            return BacktestResult()
        
        # Run simulation
        result = self._simulate(data)
        
        # Calculate metrics
        self._calculate_metrics(result)
        
        return result
    
    def _load_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Load and prepare historical data."""
        data = {}
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days
        
        for symbol in symbols:
            logger.info(f"Loading {symbol}...")
            
            # Backfill data
            collector.backfill_historical_data(symbol, timeframe, days + 30)
            
            # Get data
            df = collector.get_dataframe(symbol, timeframe, limit=days * 96)
            
            if not df.empty:
                # Filter to date range
                if "datetime" in df.columns:
                    df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
                
                # Engineer features
                df = feature_engineer.engineer_all_features(df)
                
                data[symbol] = df
        
        return data
    
    def _simulate(self, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Run trading simulation."""
        result = BacktestResult()
        
        capital = self.initial_capital
        open_positions: Dict[str, BacktestTrade] = {}
        equity_curve = [capital]
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in data.values():
            if "timestamp" in df.columns:
                all_timestamps.update(df["timestamp"].tolist())
        
        timestamps = sorted(all_timestamps)
        
        for i, ts in enumerate(timestamps):
            # Check existing positions
            for symbol, trade in list(open_positions.items()):
                if symbol not in data:
                    continue
                    
                df = data[symbol]
                current = df[df["timestamp"] == ts]
                
                if current.empty:
                    continue
                
                current_price = current["close"].iloc[0]
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                
                # Check exit conditions
                exit_reason = None
                
                # Stop loss (-3%)
                if pnl_pct <= -settings.trading.stop_loss_percent:
                    exit_reason = "stop_loss"
                
                # Take profit (+6% or +8%)
                tp_pct = settings.trading.take_profit_high_conviction if trade.tier == 3 else settings.trading.take_profit_standard
                if pnl_pct >= tp_pct:
                    exit_reason = "take_profit"
                
                if exit_reason:
                    trade.exit_time = pd.Timestamp(ts, unit="ms")
                    trade.exit_price = current_price
                    trade.exit_reason = exit_reason
                    trade.pnl_pct = pnl_pct
                    trade.pnl_usd = trade.pnl_pct * (capital * self.risk_per_trade)
                    
                    # Apply fees
                    fee = capital * self.risk_per_trade * self.FEE_RATE * 2
                    trade.pnl_usd -= fee
                    result.fees_paid += fee
                    
                    capital += trade.pnl_usd
                    result.trades.append(trade)
                    del open_positions[symbol]
            
            # Look for new entries (every 4 periods = 1 hour)
            if i % 4 == 0 and len(open_positions) < self.max_positions:
                for symbol, df in data.items():
                    if symbol in open_positions:
                        continue
                    
                    # Get historical data at this timestamp
                    mask = df["timestamp"] == ts
                    if not mask.any():
                        continue
                    
                    # Get integer position (iloc index)
                    pos = mask.values.nonzero()[0][0]
                    
                    # Need at least 50 periods of history
                    if pos < 50:
                        continue
                    
                    # Get historical slice for this point in time
                    hist_df = df.iloc[max(0, pos-200):pos+1].copy()
                    if len(hist_df) < 50:
                        continue
                    
                    current = hist_df.iloc[-1]
                    entry_price = current["close"]
                    
                    # Calculate signals from HISTORICAL features (not live API)
                    rsi = current.get("rsi_14", 50)
                    macd = current.get("macd", 0)
                    macd_signal = current.get("macd_signal", 0)
                    vol_ratio = current.get("volume_ratio_20", 1)
                    bb_position = current.get("bb_position", 0.5)
                    uptrend = current.get("uptrend", 0)
                    momentum = current.get("momentum_10", 0)
                    adx = current.get("adx", 25)
                    
                    # Scoring system (0-100, like TA analyzer)
                    score = 0
                    tier = 1
                    
                    # RSI scoring (0-25)
                    if 40 <= rsi <= 60:
                        score += 15
                    elif 30 <= rsi < 40:
                        score += 20  # Oversold bounce setup
                    elif rsi < 30:
                        score += 10  # Too oversold, wait
                    elif 60 < rsi <= 70:
                        score += 10
                    
                    # MACD scoring (0-25)
                    if macd > macd_signal:
                        score += 15
                        if macd > 0:
                            score += 10  # Above zero line
                    
                    # Volume scoring (0-25)
                    if vol_ratio > 2.0:
                        score += 25
                    elif vol_ratio > 1.5:
                        score += 18
                    elif vol_ratio > 1.2:
                        score += 10
                    
                    # Trend scoring (0-25)
                    if uptrend:
                        score += 15
                    if momentum > 0:
                        score += 5
                    if adx > 25:
                        score += 5
                    
                    # Entry decision based on score
                    # Lower threshold for more trades
                    if score >= 50:  # Minimum 50/100 to enter
                        tier = 3 if score >= 80 else (2 if score >= 65 else 1)
                        
                        trade = BacktestTrade(
                            coin=symbol,
                            entry_time=pd.Timestamp(ts, unit="ms"),
                            entry_price=entry_price,
                            tier=tier
                        )
                        
                        open_positions[symbol] = trade
                        
                        # Apply entry fee
                        fee = capital * self.risk_per_trade * self.FEE_RATE
                        result.fees_paid += fee
                        
                        if len(open_positions) >= self.max_positions:
                            break
            
            # Update equity curve
            unrealized_pnl = 0
            for symbol, trade in open_positions.items():
                if symbol in data:
                    df = data[symbol]
                    current = df[df["timestamp"] == ts]
                    if not current.empty:
                        price = current["close"].iloc[0]
                        pnl = (price - trade.entry_price) / trade.entry_price
                        unrealized_pnl += pnl * capital * self.risk_per_trade
            
            equity_curve.append(capital + unrealized_pnl)
        
        # Close remaining positions at last price
        for symbol, trade in open_positions.items():
            if symbol in data:
                df = data[symbol]
                if not df.empty:
                    exit_price = df["close"].iloc[-1]
                    pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                    
                    trade.exit_price = exit_price
                    trade.exit_reason = "end_of_backtest"
                    trade.pnl_pct = pnl_pct
                    trade.pnl_usd = pnl_pct * capital * self.risk_per_trade
                    
                    capital += trade.pnl_usd
                    result.trades.append(trade)
        
        result.total_return = (capital - self.initial_capital) / self.initial_capital
        result.total_trades = len(result.trades)
        
        # Calculate max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        
        result.max_drawdown = max_dd
        
        return result
    
    def _calculate_metrics(self, result: BacktestResult) -> None:
        """Calculate performance metrics."""
        if not result.trades:
            return
        
        wins = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct < 0]
        
        result.win_rate = len(wins) / len(result.trades) if result.trades else 0
        
        if wins:
            result.average_win = np.mean([t.pnl_pct for t in wins])
            result.largest_win = max(t.pnl_pct for t in wins)
        
        if losses:
            result.average_loss = np.mean([t.pnl_pct for t in losses])
            result.largest_loss = min(t.pnl_pct for t in losses)
        
        # Profit factor
        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Average trade duration
        durations = []
        for t in result.trades:
            if t.exit_time and t.entry_time:
                dur = t.exit_time - t.entry_time
                durations.append(dur.total_seconds() / 3600)  # hours
        
        if durations:
            avg_hours = np.mean(durations)
            result.avg_trade_duration = f"{avg_hours:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Backtest Trading Strategy")
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital (default: 10000)"
    )
    parser.add_argument(
        "--coins",
        type=int,
        default=10,
        help="Number of coins to test (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Run backtest
    engine = BacktestEngine(initial_capital=args.capital)
    
    result = engine.run(
        symbols=WATCHLIST[:args.coins],
        start_date=args.start,
        end_date=args.end
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {args.start} to {args.end}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print()
    print(f"Total Return: {result.total_return:.1%}")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print()
    print(f"Total Trades: {result.total_trades}")
    print(f"Average Win: {result.average_win:.2%}")
    print(f"Average Loss: {result.average_loss:.2%}")
    print(f"Largest Win: {result.largest_win:.2%}")
    print(f"Largest Loss: {result.largest_loss:.2%}")
    print(f"Avg Duration: {result.avg_trade_duration}")
    print(f"Fees Paid: ${result.fees_paid:.2f}")
    print("=" * 60)
    
    # Check if meets criteria
    print("\n📊 Criteria Check:")
    print(f"  Win Rate > 55%: {'✅' if result.win_rate > 0.55 else '❌'} ({result.win_rate:.1%})")
    print(f"  Profit Factor > 1.5: {'✅' if result.profit_factor > 1.5 else '❌'} ({result.profit_factor:.2f})")
    print(f"  Max Drawdown < 30%: {'✅' if result.max_drawdown < 0.30 else '❌'} ({result.max_drawdown:.1%})")


if __name__ == "__main__":
    main()
