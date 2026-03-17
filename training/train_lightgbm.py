"""
Hybrid ML Model Training Script (XGBoost + CatBoost)

Train the hybrid model with:
- 1h as primary timeframe
- 15m data for entry timing confirmation
- 4h data for trend direction
- ATR-based adaptive targets (1.5x ATR, capped 1-4%)

Author: Trading Bot
Version: 4.0 (Hybrid ML)
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WATCHLIST
from src.utils import logger
from src.data import db, collector, feature_engineer
from src.models.ml_model import ml_model


def fetch_training_data(
    symbols: list = None,
    days: int = 180,
    timeframe: str = "1h"
) -> pd.DataFrame:
    """Fetch training data from database and API."""
    symbols = symbols or WATCHLIST
    
    all_data = []
    
    for symbol in symbols:
        logger.info(f"Fetching {days} days of {timeframe} data for {symbol}...")
        
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
                df['symbol'] = symbol
                all_data.append(df)
                logger.info(f"  ✅ Loaded {len(df)} {timeframe} records from database")
            else:
                # Fallback to API
                candles_per_day = {"15m": 96, "1h": 24, "4h": 6}.get(timeframe, 24)
                df = collector.get_dataframe(symbol, timeframe, limit=days * candles_per_day)
                if not df.empty:
                    df["symbol"] = symbol
                    all_data.append(df)
                    logger.info(f"  ✅ Loaded {len(df)} {timeframe} records from API")
        except Exception as e:
            logger.warning(f"  Error fetching {symbol} ({timeframe}): {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"📊 Total {timeframe} samples: {len(combined):,}")
        return combined
    
    return pd.DataFrame()


def train_hybrid_ml(days: int = 180, lookahead: int = 4) -> dict:
    """Train Hybrid XGBoost + CatBoost model with multi-timeframe data."""
    
    logger.info("=" * 70)
    logger.info("🚀 HYBRID ML TRAINING (XGBoost + CatBoost)")
    logger.info("=" * 70)
    logger.info("Target: ADAPTIVE (1.5x ATR, capped 1-4%)")
    logger.info(f"Training data: Last {days} days")
    logger.info("Primary: 1h | Confirmation: 15m + 4h")
    logger.info("=" * 70)
    
    # Fetch 1h data (primary)
    logger.info("\n📊 Fetching 1h data (primary)...")
    df_1h = fetch_training_data(days=days, timeframe="1h")
    
    if df_1h.empty:
        logger.error("❌ No 1h training data available")
        return {"error": "No data"}
    
    # Fetch 15m data (entry timing)
    logger.info("\n📊 Fetching 15m data (entry timing)...")
    df_15m = fetch_training_data(days=days, timeframe="15m")
    
    # Fetch 4h data (trend direction)
    logger.info("\n📊 Fetching 4h data (trend direction)...")
    df_4h = fetch_training_data(days=days, timeframe="4h")
    
    logger.info("\n" + "=" * 70)
    logger.info("DATA SUMMARY:")
    logger.info(f"  1h records:  {len(df_1h):,}")
    logger.info(f"  15m records: {len(df_15m):,}")
    logger.info(f"  4h records:  {len(df_4h):,}")
    logger.info("=" * 70)
    
    # Train hybrid model
    logger.info("\n🔧 Training Hybrid ML model...")
    logger.info("-" * 70)
    
    try:
        metrics = ml_model.train(
            training_data=df_1h,
            df_15m=df_15m if not df_15m.empty else None,
            df_4h=df_4h if not df_4h.empty else None,
            lookahead=lookahead,
            validation_split=0.2,
            use_adaptive_atr=True  # ATR-based targets
        )
        
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("=" * 70)
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid ML Model (XGBoost + CatBoost)")
    parser.add_argument("--days", type=int, default=180, help="Days of training data (default: 180)")
    parser.add_argument("--lookahead", type=int, default=4, help="Lookahead in hours (default: 4)")
    
    args = parser.parse_args()
    
    # Initialize database
    try:
        db.create_tables()
    except:
        pass
    
    # Train hybrid model
    metrics = train_hybrid_ml(days=args.days, lookahead=args.lookahead)
    
    # Display results
    if "error" in metrics:
        print("\n" + "=" * 70)
        print("❌ TRAINING FAILED")
        print("=" * 70)
        print(f"Error: {metrics['error']}")
        return
    
    print("\n" + "=" * 70)
    print("📊 TRAINING RESULTS (Hybrid XGBoost + CatBoost)")
    print("=" * 70)
    print(f"Train Samples:     {metrics.get('train_samples', 0):,}")
    print(f"Val Samples:       {metrics.get('val_samples', 0):,}")
    print(f"Features Used:     {metrics.get('features_used', 0)}")
    print(f"Positive Rate:     {metrics.get('positive_ratio', 0):.1%}")
    print()
    print(f"CatBoost AUC:      {metrics.get('catboost_auc', 0):.4f}")
    print(f"XGBoost AUC:       {metrics.get('xgb_auc', 0):.4f}")
    print(f"Ensemble AUC:      {metrics.get('ensemble_auc', 0):.4f}")
    print("=" * 70)
    
    # Quality check
    ensemble_auc = metrics.get('ensemble_auc', 0)
    if ensemble_auc > 0.70:
        print("✅ Model quality: EXCELLENT (AUC > 0.70)")
    elif ensemble_auc > 0.65:
        print("✅ Model quality: VERY GOOD (AUC > 0.65)")
    elif ensemble_auc > 0.60:
        print("✅ Model quality: GOOD (AUC > 0.60)")
    elif ensemble_auc > 0.55:
        print("⚠️  Model quality: FAIR (AUC 0.55-0.60)")
    else:
        print("❌ Model quality: POOR (AUC < 0.55)")
    
    print("=" * 70)
    print("📁 Model saved to: models/hybrid_ml_model.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()
