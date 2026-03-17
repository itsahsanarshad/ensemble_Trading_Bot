"""
Ensemble Model Backtest Script - SPOT TRADING (BUY-ONLY)

Tests the actual TA, ML, and TCN models on historical data.
Only counts BUY signal profits (realistic for spot trading).
"""

import argparse
from datetime import datetime
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WATCHLIST
from src.utils import logger
from src.data import feature_engineer


def get_data_from_database(symbol: str, timeframe: str = "1h", days: int = 365) -> pd.DataFrame:
    """Get historical data from SQLite database."""
    try:
        from src.data.database import db as database, PriceData
        from sqlalchemy.orm import Session
        
        session = Session(database.engine)
        
        records = session.query(PriceData).filter_by(
            coin=symbol, timeframe=timeframe
        ).order_by(PriceData.timestamp.asc()).all()
        
        session.close()
        
        if records:
            df = pd.DataFrame([{
                'timestamp': r.timestamp,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            } for r in records])
            
            if len(df) > days * 24:
                df = df.tail(days * 24)
            
            return df
    except Exception as e:
        print(f"  ❌ {symbol}: Database error: {e}")
    
    return pd.DataFrame()


def run_spot_backtest(
    symbols: list,
    lookback_days: int = 365,
    max_hold_hours: int = 24,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06
):
    """
    Run SPOT TRADING backtest (BUY-ONLY).
    
    This simulates real spot trading:
    - Only BUY signals open positions
    - SELL signals are used to AVOID buying (not for shorting)
    - Tracks realistic P&L with stop loss and take profit
    """
    from src.models.ta_analyzer import ta_analyzer
    from src.models.ml_model import ml_model
    from src.models.tcn_model import tcn_model
    
    print("=" * 70)
    print("SPOT TRADING BACKTEST (BUY-ONLY)")
    print("=" * 70)
    print(f"Coins: {', '.join(symbols)}")
    print(f"Lookback: {lookback_days} days")
    print(f"Stop Loss: {stop_loss_pct:.0%} | Take Profit: {take_profit_pct:.0%}")
    print("=" * 70)
    
    results = {
        "ta": {"trades": 0, "wins": 0, "pnl": [], "avoided_losses": 0},
        "ml": {"trades": 0, "wins": 0, "pnl": [], "avoided_losses": 0},
        "tcn": {"trades": 0, "wins": 0, "pnl": [], "avoided_losses": 0},
        "ensemble": {"trades": 0, "wins": 0, "pnl": [], "avoided_losses": 0}
    }
    
    for coin_idx, symbol in enumerate(symbols):
        print(f"\n[{coin_idx+1}/{len(symbols)}] Processing {symbol}...")
        
        df = get_data_from_database(symbol, "1h", lookback_days)
        
        if df.empty or len(df) < 100:
            print(f"  ⚠️ Insufficient data ({len(df)} candles)")
            continue
        
        print(f"  📊 Loaded {len(df)} candles")
        
        print(f"  🔧 Engineering features...", end=" ", flush=True)
        df = feature_engineer.engineer_all_features(df)
        print("✅")
        
        test_indices = list(range(100, len(df) - max_hold_hours, 24))
        total_tests = len(test_indices)
        
        print(f"  📈 Running {total_tests} tests...", end=" ", flush=True)
        
        for test_num, i in enumerate(test_indices):
            if test_num > 0 and test_num % (max(1, total_tests // 4)) == 0:
                print(f"{test_num}/{total_tests}", end=" ", flush=True)
            
            df_slice = df.iloc[:i+1].copy()
            entry_price = df_slice.iloc[-1]['close']
            
            # Get model predictions
            try:
                ta_result = ta_analyzer.analyze(symbol, df_slice)
                ml_result = ml_model.predict(symbol, df_slice)
                tcn_result = tcn_model.predict(df_slice, symbol)
            except:
                continue
            
            # Calculate actual trade outcome with stop/take profit
            future_prices = df.iloc[i+1:i+max_hold_hours+1]['close'].values
            if len(future_prices) < 4:
                continue
            
            # Simulate trade with stop loss and take profit
            def simulate_trade(entry_price, future_prices, sl_pct, tp_pct):
                """Simulate a trade and return PnL."""
                for price in future_prices:
                    pnl = (price - entry_price) / entry_price
                    if pnl <= -sl_pct:
                        return -sl_pct  # Hit stop loss
                    if pnl >= tp_pct:
                        return tp_pct  # Hit take profit
                # Exit at last price if neither hit
                final_pnl = (future_prices[-1] - entry_price) / entry_price
                return final_pnl
            
            # Check what would have happened if we bought
            trade_pnl = simulate_trade(entry_price, future_prices, stop_loss_pct, take_profit_pct)
            was_good_buy = trade_pnl > 0
            
            # Evaluate each model
            models = [
                ("ta", ta_result.signal, ta_result.confidence),
                ("ml", ml_result.signal, ml_result.confidence),
                ("tcn", tcn_result.signal, tcn_result.confidence)
            ]
            
            for model_name, signal, confidence in models:
                if confidence < 0.5:
                    continue
                
                if signal == "buy":
                    # Model said BUY - count the trade
                    results[model_name]["trades"] += 1
                    results[model_name]["pnl"].append(trade_pnl)
                    if trade_pnl > 0:
                        results[model_name]["wins"] += 1
                
                elif signal == "sell" and not was_good_buy:
                    # Model said SELL and avoided a bad trade
                    results[model_name]["avoided_losses"] += 1
            
            # Ensemble: 2/3 buy agreement
            signals = [ta_result.signal, ml_result.signal, tcn_result.signal]
            buy_count = sum(1 for s in signals if s == "buy")
            sell_count = sum(1 for s in signals if s == "sell")
            
            if buy_count >= 2:
                results["ensemble"]["trades"] += 1
                results["ensemble"]["pnl"].append(trade_pnl)
                if trade_pnl > 0:
                    results["ensemble"]["wins"] += 1
            elif sell_count >= 2 and not was_good_buy:
                results["ensemble"]["avoided_losses"] += 1
        
        print(" ✅")
    
    # Print results
    print("\n" + "=" * 70)
    print("SPOT TRADING RESULTS (BUY-ONLY)")
    print("=" * 70)
    
    for model_name in ["ta", "ml", "tcn", "ensemble"]:
        r = results[model_name]
        trades = r["trades"]
        
        if trades == 0:
            continue
        
        win_rate = r["wins"] / trades
        total_pnl = sum(r["pnl"])
        avg_pnl = np.mean(r["pnl"]) if r["pnl"] else 0
        wins = [p for p in r["pnl"] if p > 0]
        losses = [p for p in r["pnl"] if p < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
        
        print(f"\n📊 {model_name.upper()} (BUY signals only):")
        print(f"  Total Trades: {trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg Win: {avg_win:.2%} | Avg Loss: {avg_loss:.2%}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Total PnL: {total_pnl:.1%}")
        print(f"  Bad Trades Avoided (by SELL signals): {r['avoided_losses']}")
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY (BUY-ONLY SPOT TRADING)")
    print("=" * 70)
    print(f"\n{'Model':<12}{'Trades':<10}{'Win Rate':<12}{'PF':<10}{'PnL':<12}{'Avoided'}")
    print("-" * 60)
    
    for model_name in ["ta", "ml", "tcn", "ensemble"]:
        r = results[model_name]
        trades = r["trades"]
        if trades == 0:
            continue
        
        win_rate = r["wins"] / trades
        total_pnl = sum(r["pnl"])
        wins = [p for p in r["pnl"] if p > 0]
        losses = [p for p in r["pnl"] if p < 0]
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
        
        print(f"{model_name.upper():<12}{trades:<10}{win_rate:.1%}{'':>4}{profit_factor:.2f}{'':>6}{total_pnl:.1%}{'':>6}{r['avoided_losses']}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Spot Trading Backtest (BUY-ONLY)")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--coins", type=int, default=5, help="Number of coins")
    parser.add_argument("--sl", type=float, default=0.03, help="Stop loss %")
    parser.add_argument("--tp", type=float, default=0.06, help="Take profit %")
    
    args = parser.parse_args()
    
    run_spot_backtest(
        symbols=WATCHLIST[:args.coins],
        lookback_days=args.days,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp
    )


if __name__ == "__main__":
    main()
