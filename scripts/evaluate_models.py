"""
Model Evaluation Script

Comprehensive testing of all three models:
- TA Analyzer
- XGBoost
- TCN

Tests predictions, feature engineering, and model quality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.data import collector
from src.models import ta_analyzer, ml_model, tcn_model, ensemble
from src.data import feature_engineer

def test_ta_analyzer():
    """Test TA analyzer on live data."""
    print("=" * 70)
    print("🔍 Testing TA Analyzer")
    print("=" * 70)
    
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        try:
            result = ta_analyzer.analyze(symbol)
            
            # Get total score from score_breakdown
            total_score = result.score_breakdown.get("total", 0) if result.score_breakdown else 0
            
            print(f"\n{symbol}:")
            print(f"  Signal: {result.signal}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Score: {total_score:.0f}/100")
            print(f"  Breakdown: {result.score_breakdown}")
            print(f"  Reasons: {result.reasons[:3]}")
            
            # Verify it's actually calculating
            if total_score == 0 or result.confidence == 0:
                print(f"  ⚠️  WARNING: Zero score/confidence!")
            else:
                print(f"  ✅ Working")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print()


def test_xgboost():
    """Test XGBoost model predictions."""
    print("=" * 70)
    print("🔍 Testing XGBoost Model")
    print("=" * 70)
    
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        try:
            result = ml_model.predict(symbol)
            
            print(f"\n{symbol}:")
            print(f"  Signal: {result.signal}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Top Features:")
            for feat, imp in result.top_features[:3]:
                print(f"    - {feat}: {imp:.4f}")
            
            # Check if predictions are reasonable
            if 0.3 <= result.confidence <= 0.7:
                print(f"  ✅ Confidence in reasonable range")
            else:
                print(f"  ⚠️  Extreme confidence: {result.confidence:.2%}")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print()


def test_lstm():
    """Test TCN model predictions."""
    print("=" * 70)
    print("🔍 Testing TCN Model")
    print("=" * 70)
    
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        try:
            result = tcn_model.predict(symbol)
            
            print(f"\n{symbol}:")
            print(f"  Signal: {result.signal}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Pattern: {result.sequence_pattern}")
            
            # Check if predictions are reasonable
            if result.confidence == 0.5:
                print(f"  ⚠️  Default confidence (model not loaded?)")
            elif 0.2 <= result.confidence <= 0.8:
                print(f"  ✅ Confidence in reasonable range")
            else:
                print(f"  ⚠️  Extreme confidence: {result.confidence:.2%}")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print()


def test_ensemble():
    """Test ensemble consensus."""
    print("=" * 70)
    print("🔍 Testing Ensemble Consensus")
    print("=" * 70)
    
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        try:
            result = ensemble.analyze(symbol)
            
            ta_conf = result.ta_signal.confidence if result.ta_signal else 0
            ml_conf = result.ml_signal.confidence if result.ml_signal else 0
            tcn_conf = result.tcn_signal.confidence if result.tcn_signal else 0
            
            print(f"\n{symbol}:")
            print(f"  TA:   {ta_conf:.2%} → {result.ta_signal.signal if result.ta_signal else 'N/A'}")
            print(f"  ML:   {ml_conf:.2%} → {result.ml_signal.signal if result.ml_signal else 'N/A'}")
            print(f"  TCN: {tcn_conf:.2%} → {result.tcn_signal.signal if result.tcn_signal else 'N/A'}")
            print(f"  ---")
            print(f"  Consensus: {result.signal} (Tier {result.tier})")
            print(f"  Confidence: {result.confidence:.2%}")
            
            # Check if all models are contributing
            if ta_conf > 0 and ml_conf > 0 and tcn_conf > 0:
                print(f"  ✅ All models active")
            else:
                print(f"  ⚠️  Some models not contributing")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print()


def test_feature_engineering():
    """Test feature engineering."""
    print("=" * 70)
    print("🔍 Testing Feature Engineering")
    print("=" * 70)
    
    symbol = "BTCUSDT"
    df = collector.get_dataframe(symbol, "15m", limit=200)
    
    if df.empty:
        print("❌ No data available")
        return
    
    print(f"\nRaw data shape: {df.shape}")
    
    # Engineer features
    df_engineered = feature_engineer.engineer_all_features(df)
    
    print(f"After engineering: {df_engineered.shape}")
    print(f"Features added: {df_engineered.shape[1] - df.shape[1]}")
    
    # Check for NaN
    nan_cols = df_engineered.columns[df_engineered.isna().any()].tolist()
    if nan_cols:
        print(f"\n⚠️  Columns with NaN: {len(nan_cols)}")
        print(f"   {nan_cols[:5]}")
    else:
        print(f"\n✅ No NaN values")
    
    # Check for inf
    inf_cols = df_engineered.columns[np.isinf(df_engineered).any()].tolist()
    if inf_cols:
        print(f"⚠️  Columns with Inf: {len(inf_cols)}")
        print(f"   {inf_cols[:5]}")
    else:
        print(f"✅ No Inf values")
    
    # Show sample features
    print(f"\nSample features:")
    feature_cols = [c for c in df_engineered.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
    for col in feature_cols[:10]:
        val = df_engineered[col].iloc[-1]
        print(f"  {col}: {val:.4f}")
    
    print()


def test_model_files():
    """Check if model files exist and are valid."""
    print("=" * 70)
    print("🔍 Checking Model Files")
    print("=" * 70)
    
    from pathlib import Path
    import pickle
    
    models_dir = Path(__file__).parent.parent / "models"
    
    # Check XGBoost
    xgb_path = models_dir / "xgboost_model.pkl"
    if xgb_path.exists():
        size_mb = xgb_path.stat().st_size / 1024 / 1024
        print(f"\n✅ XGBoost model exists ({size_mb:.2f} MB)")
        
        try:
            with open(xgb_path, "rb") as f:
                data = pickle.load(f)
            print(f"   Features: {len(data.get('feature_names', []))}")
            print(f"   Scaler: {'✅' if data.get('scaler') else '❌'}")
        except Exception as e:
            print(f"   ⚠️  Error loading: {e}")
    else:
        print(f"\n❌ XGBoost model not found")
    
    # Check TCN
    lstm_path = models_dir / "tcn_model.keras"
    if lstm_path.exists():
        size_mb = lstm_path.stat().st_size / 1024 / 1024
        print(f"\n✅ TCN model exists ({size_mb:.2f} MB)")
    else:
        print(f"\n❌ TCN model not found")
    
    lstm_scaler = models_dir / "lstm_scaler.pkl"
    if lstm_scaler.exists():
        print(f"✅ TCN scaler exists")
        try:
            with open(lstm_scaler, "rb") as f:
                data = pickle.load(f)
            print(f"   Features: {len(data.get('feature_names', []))}")
        except Exception as e:
            print(f"   ⚠️  Error loading: {e}")
    else:
        print(f"❌ TCN scaler not found")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70 + "\n")
    
    test_model_files()
    test_feature_engineering()
    test_ta_analyzer()
    test_xgboost()
    test_lstm()
    test_ensemble()
    
    print("=" * 70)
    print("✅ Evaluation Complete")
    print("=" * 70)
