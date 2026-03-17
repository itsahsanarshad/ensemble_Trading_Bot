"""
Individual Model Backtester

Tests each model (TA, ML, TCN) separately to identify performance issues.
Uses database data for reproducible results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.data import db, feature_engineer
from src.data.database import PriceData
from src.models.ta_analyzer import ta_analyzer
from src.models.ml_model import ml_model  
from src.models.tcn_model import tcn_model
from src.utils import logger

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def get_data_from_database(symbol: str, timeframe: str = "1h", days: int = 30) -> pd.DataFrame:
    """
    Load historical data from SQLite database for reproducible evaluation.
    """
    try:
        # Calculate how many candles we need
        candles_per_day = {"15m": 96, "1h": 24, "4h": 6}.get(timeframe, 24)
        limit = days * candles_per_day
        
        # Get data from database
        records = db.get_price_data(symbol, timeframe, limit=limit)
        
        if not records:
            return pd.DataFrame()
        
        # Convert to DataFrame (records are in desc order, so reverse)
        df = pd.DataFrame([{
            'timestamp': r.timestamp,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume
        } for r in reversed(records)])
        
        return df
    except Exception as e:
        print(f"  Error loading {symbol} from database: {e}")
        return pd.DataFrame()


def evaluate_model(model_name: str, symbols: List[str], lookback_days: int = 30) -> Dict:
    """
    Evaluate a single model's predictions against actual price movements.
    Uses database data for reproducibility.
    
    For each signal, we check if price moved in the predicted direction within 24 hours.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} Model")
    print(f"{'='*60}")
    print(f"  Data source: SQLite database (last {lookback_days} days)")
    
    correct = 0
    incorrect = 0
    total_signals = 0
    buy_signals = 0
    sell_signals = 0
    
    total_pnl = 0.0
    wins = []
    losses = []
    
    for symbol in symbols:
        try:
            # Get historical 1h data FROM DATABASE (not live API)
            df = get_data_from_database(symbol, timeframe="1h", days=lookback_days)
            
            if df.empty or len(df) < 100:
                print(f"  ⚠️ {symbol}: Insufficient data ({len(df)} candles)")
                continue
            
            print(f"  📊 {symbol}: Loaded {len(df)} candles from database")
            
            df = feature_engineer.engineer_all_features(df)
            
            # Test predictions at different points in time
            # We'll check every 4th candle to avoid overlap
            for i in range(50, len(df) - 24, 4):
                # Get data up to this point (no forward looking)
                df_slice = df.iloc[:i+1].copy()
                
                # Get model prediction - ALL models now use df_slice!
                try:
                    if model_name == "TA":
                        result = ta_analyzer.analyze(symbol, df_slice)
                        signal = result.signal
                        confidence = result.confidence
                    elif model_name == "ML":
                        # FIX: Pass df_slice to ML model instead of fetching live data
                        result = ml_model.predict(symbol, df_slice)
                        signal = result.signal
                        confidence = result.confidence
                    elif model_name == "TCN":
                        result = tcn_model.predict(df_slice, symbol)
                        signal = result.signal
                        confidence = result.confidence
                    else:
                        continue
                except Exception as e:
                    continue
                
                # Only count high-confidence signals
                if confidence < 0.5:
                    continue
                
                if signal not in ["buy", "sell"]:
                    continue
                
                total_signals += 1
                if signal == "buy":
                    buy_signals += 1
                else:
                    sell_signals += 1
                
                # Check actual outcome (24 hours later = 24 candles on 1h)
                entry_price = df.iloc[i]["close"]
                future_prices = df.iloc[i+1:i+25]["close"]
                
                if len(future_prices) < 12:
                    continue
                
                # Calculate max favorable move and actual move
                if signal == "buy":
                    # BUY signal - count actual PnL (SPOT TRADING)
                    max_price = future_prices.max()
                    pnl = (max_price - entry_price) / entry_price
                    
                    # Win if price went up at least 1%
                    if pnl > 0.01:
                        correct += 1
                        wins.append(pnl)
                        total_pnl += pnl
                    else:
                        incorrect += 1
                        min_price = future_prices.min()
                        loss = (min_price - entry_price) / entry_price
                        losses.append(loss)
                        total_pnl += loss
                        
                else:  # sell signal
                    # SELL signal - in SPOT trading, this is NOT a profit opportunity
                    # It's a warning to NOT BUY. Check if model was correct.
                    min_price = future_prices.min()
                    price_dropped = (entry_price - min_price) / entry_price > 0.01
                    
                    if price_dropped:
                        # Model correctly said "don't buy" - price did drop
                        correct += 1
                        # Note: NO PnL added - we didn't trade, we just avoided a loss
                    else:
                        # Model said "don't buy" but price went up - missed opportunity
                        incorrect += 1
                        
        except Exception as e:
            print(f"  Error evaluating {symbol}: {e}")
            continue
    
    # Calculate metrics
    total = correct + incorrect
    win_rate = correct / total if total > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
    
    print(f"\n📊 {model_name} Results:")
    print(f"  Total Signals: {total_signals}")
    print(f"  Buy Signals: {buy_signals}")
    print(f"  Sell Signals: {sell_signals}")
    print(f"  ")
    print(f"  Correct: {correct}")
    print(f"  Incorrect: {incorrect}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  ")
    print(f"  Avg Win: {avg_win:.2%}")
    print(f"  Avg Loss: {avg_loss:.2%}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total PnL: {total_pnl:.2%}")
    
    return {
        "model": model_name,
        "total_signals": total_signals,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "correct": correct,
        "incorrect": incorrect,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl
    }


def main():
    """Evaluate each model individually."""
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL EVALUATION")
    print("="*60)
    
    # Test symbols
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    
    results = []
    
    # Evaluate each model
    for model_name in ["TA", "ML", "TCN"]:
        result = evaluate_model(model_name, symbols, lookback_days=30)
        results.append(result)
    
    # Summary comparison
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<10} {'Signals':<10} {'Win Rate':<12} {'Profit Factor':<15} {'Total PnL':<12}")
    print("-" * 60)
    
    for r in results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] > 0 else "N/A"
        print(f"{r['model']:<10} {r['total_signals']:<10} {r['win_rate']:.1%}      {pf_str:<15} {r['total_pnl']:.2%}")
    
    # Identify best/worst
    best = max(results, key=lambda x: x['win_rate'])
    worst = min(results, key=lambda x: x['win_rate'])
    
    print(f"\n✅ Best Performer: {best['model']} ({best['win_rate']:.1%} win rate)")
    print(f"❌ Worst Performer: {worst['model']} ({worst['win_rate']:.1%} win rate)")
    

if __name__ == "__main__":
    main()
