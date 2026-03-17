"""
Hybrid ML Model: XGBoost + CatBoost Ensemble

Based on research showing:
- XGBoost: Faster, more profitable (7% vs 2% in head-to-head tests)
- CatBoost: Better probability calibration, 81% win rate in studies
- Combination: Best of both worlds

Multi-Timeframe Approach:
- 4h: Overall trend direction (features)
- 1h: Primary prediction timeframe
- 15m: Entry timing precision (features)

Target: ATR-based adaptive (1.5x ATR, capped 1-4%)

Author: Trading Bot
Version: 4.0 (Hybrid XGBoost + CatBoost)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ML imports
try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import logger, log_model_prediction
from src.data import collector, feature_engineer


# Model save path
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class MLSignal:
    """ML model prediction output."""
    signal: str  # 'buy', 'hold', 'sell'
    confidence: float  # 0.0 to 1.0
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "signal": self.signal,
            "confidence": self.confidence,
            "top_features": self.top_features
        }


class HybridMLModel:
    """
    Hybrid XGBoost + CatBoost Model for Crypto Prediction.
    
    Uses soft voting ensemble of:
    - XGBoost: Speed and profitability
    - CatBoost: Probability calibration
    
    Multi-Timeframe Features:
    - 4h features for trend context
    - 1h primary features
    - 15m features for momentum/timing
    
    Target: ATR-based adaptive (1.5x ATR, capped 1-4%)
    """
    
    # Signal thresholds (best performing configuration)
    BUY_THRESHOLD = 0.58       # Original best - 55.2% win rate
    STRONG_BUY_THRESHOLD = 0.70
    SELL_THRESHOLD = 0.42      # Original best
    
    def __init__(self, model_path: str = None):
        """Initialize hybrid model."""
        if not HAS_CATBOOST:
            logger.warning("CatBoost not installed")
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed - will use CatBoost only")
        
        # Initialize CatBoost
        self.catboost_model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=8,
            loss_function='Logloss',
            eval_metric='AUC',
            l2_leaf_reg=3.0,
            min_data_in_leaf=30,
            subsample=0.8,
            colsample_bylevel=0.8,
            auto_class_weights='Balanced',
            early_stopping_rounds=50,
            random_seed=42,
            thread_count=-1,
            verbose=False,
            bootstrap_type='Bernoulli',
        )
        
        # Initialize XGBoost
        if HAS_XGBOOST:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=8,
                min_child_weight=30,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # Will be dynamically set during train()
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=50,
            )
        else:
            self.xgb_model = None
        
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.is_trained: bool = False
        self.model_path = model_path or str(MODELS_DIR / "hybrid_ml_model.pkl")
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load saved model if exists."""
        try:
            model_file = Path(self.model_path)
            if model_file.exists():
                with open(model_file, "rb") as f:
                    saved = pickle.load(f)
                    self.catboost_model = saved.get("catboost_model")
                    self.xgb_model = saved.get("xgb_model")
                    self.scaler = saved.get("scaler")
                    self.feature_names = saved.get("feature_names", [])
                    self.feature_importance = saved.get("feature_importance", {})
                    self.is_trained = True
                logger.info(f"Hybrid ML model loaded from {self.model_path}")
                return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
        return False
    
    def _save_model(self) -> None:
        """Save model to disk."""
        try:
            save_data = {
                "catboost_model": self.catboost_model,
                "xgb_model": self.xgb_model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
                "saved_at": datetime.utcnow().isoformat(),
                "model_type": "hybrid_xgb_catboost"
            }
            with open(self.model_path, "wb") as f:
                pickle.dump(save_data, f)
            logger.info(f"Hybrid model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _get_feature_columns(self) -> List[str]:
        """Get comprehensive feature list for multi-timeframe analysis."""
        return [
            # === 1H PRIMARY FEATURES ===
            # Price action
            "return_1", "return_4", "return_16", "return_96",
            "price_vs_ema20", "price_vs_ema50", "price_vs_sma200",
            "hl_spread", "momentum_10", "momentum_20", "roc_12", "roc_24",
            
            # Volume
            "volume_ratio_20", "volume_ratio_50", "volume_change",
            "volume_change_4", "volume_trend", "vol_price_corr",
            "obv_vs_sma", "price_vs_vwap",
            
            # Technical indicators
            "rsi_7", "rsi_14", "rsi_21",
            "macd", "macd_signal", "macd_histogram",
            "stoch_k", "stoch_d", "bb_position", "bb_width",
            "adx", "williams_r", "cci", "cmf",
            "ema_20_50_cross", "macd_cross",
            
            # Volatility
            "atr_14", "atr_7", "std_24", "std_normalized",
            "volatility_24h", "true_range", "gk_volatility",
            
            # Pattern
            "higher_high", "higher_low", "consec_hh", "consec_hl",
            "uptrend", "downtrend", "consolidation",
            "breakout_magnitude", "dist_from_high_20", "dist_from_low_20",
            
            # Time
            "hour", "day_of_week", "is_weekend",
            "asian_session", "european_session", "american_session",
            
            # === 15M CONFIRMATION FEATURES (Entry Timing) ===
            "rsi_14_15m", "macd_15m", "macd_histogram_15m",
            "volume_ratio_15m", "trend_15m", "momentum_15m", "bullish_15m",
            
            # === 4H CONFIRMATION FEATURES (Trend Direction) ===
            "rsi_14_4h", "macd_4h", "macd_histogram_4h", "adx_4h",
            "trend_4h", "price_vs_ema20_4h", "bullish_4h",
            
            # === MULTI-TIMEFRAME ALIGNMENT ===
            "mtf_alignment", "all_timeframes_bullish"
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for prediction."""
        feature_cols = self.feature_names if self.feature_names else self._get_feature_columns()
        
        X = pd.DataFrame()
        for col in feature_cols:
            if col in df.columns:
                X[col] = df[col].values
            else:
                X[col] = 0
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill().fillna(0)
        
        return X.values, feature_cols
    
    def train(
        self,
        training_data: pd.DataFrame,
        df_15m: pd.DataFrame = None,
        df_4h: pd.DataFrame = None,
        target_pct: float = 0.06,
        lookahead: int = 4,
        validation_split: float = 0.2,
        use_adaptive_atr: bool = True  # ATR-based targets
    ) -> Dict:
        """
        Train the hybrid XGBoost + CatBoost model.
        
        Uses ATR-based adaptive targets (1.5x ATR, capped 1-4%).
        """
        logger.info("=" * 60)
        logger.info("STARTING HYBRID ML TRAINING (XGBoost + CatBoost)")
        logger.info("=" * 60)
        if use_adaptive_atr:
            logger.info("Target: ADAPTIVE (1.5x ATR, capped 1-4%)")
        else:
            logger.info(f"Target: {target_pct:.1%} in {lookahead}h")
        logger.info(f"Primary: 1h | Confirmation: 15m + 4h")
        
        # Engineer 1h features
        logger.info("Engineering 1h features...")
        df = feature_engineer.engineer_all_features(training_data)
        
        # Add multi-timeframe features
        if df_15m is not None or df_4h is not None:
            logger.info("Adding MTF features...")
            df = feature_engineer.engineer_multi_timeframe_features_training(df, df_15m, df_4h)
        
        # Create ATR-based target
        logger.info("Creating ATR-based target...")
        df["target"] = feature_engineer.create_target_variable(
            df, target_pct, lookahead, use_adaptive_atr=use_adaptive_atr
        )
        
        df = df.dropna(subset=["target"])
        
        # Reset feature names for fresh training
        self.feature_names = []
        
        # Prepare features
        logger.info("Preparing features...")
        X, feature_names = self.prepare_features(df)
        y = df["target"].values
        
        positive_rate = y.mean()
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Positive rate: {positive_rate:.2%}")
        
        # Time series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # === TRAIN CATBOOST ===
        logger.info("-" * 60)
        logger.info("Training CatBoost...")
        train_pool = Pool(X_train_scaled, y_train, feature_names=feature_names)
        val_pool = Pool(X_val_scaled, y_val, feature_names=feature_names)
        
        self.catboost_model.fit(train_pool, eval_set=val_pool, use_best_model=True, plot=False)
        catboost_pred = self.catboost_model.predict_proba(X_val_scaled)[:, 1]
        catboost_auc = self._calculate_auc(y_val, catboost_pred)
        logger.info(f"CatBoost AUC: {catboost_auc:.4f}")
        
        # === TRAIN XGBOOST ===
        xgb_auc = 0.0
        if HAS_XGBOOST and self.xgb_model:
            logger.info("-" * 60)
            logger.info("Training XGBoost...")
            
            # Dynamically calc scale_pos_weight to fix severe class imbalance
            pos_weight = float(np.sum(y_train == 0)) / max(1, np.sum(y_train == 1))
            self.xgb_model.set_params(scale_pos_weight=pos_weight)
            logger.info(f"Dynamically set XGBoost scale_pos_weight: {pos_weight:.2f}")
            
            self.xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            xgb_pred = self.xgb_model.predict_proba(X_val_scaled)[:, 1]
            xgb_auc = self._calculate_auc(y_val, xgb_pred)
            logger.info(f"XGBoost AUC: {xgb_auc:.4f}")
        
        # === ENSEMBLE PREDICTION ===
        if HAS_XGBOOST and self.xgb_model:
            # Average probabilities (soft voting)
            ensemble_pred = (catboost_pred + xgb_pred) / 2
            ensemble_auc = self._calculate_auc(y_val, ensemble_pred)
            logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
        else:
            ensemble_auc = catboost_auc
        
        self.feature_names = feature_names
        self.is_trained = True
        
        # Feature importance (average of both models)
        catboost_importance = self.catboost_model.get_feature_importance()
        if HAS_XGBOOST and self.xgb_model:
            xgb_importance = self.xgb_model.feature_importances_
            avg_importance = (catboost_importance + xgb_importance) / 2
        else:
            avg_importance = catboost_importance
        
        self.feature_importance = {
            name: float(imp) for name, imp in zip(feature_names, avg_importance)
        }
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        self._log_feature_importance()
        self._save_model()
        
        metrics = {
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "positive_ratio": float(positive_rate),
            "catboost_auc": float(catboost_auc),
            "xgb_auc": float(xgb_auc),
            "ensemble_auc": float(ensemble_auc),
            "auc": float(ensemble_auc),
            "features_used": len(feature_names),
            "top_features": list(self.feature_importance.items())[:10]
        }
        
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"CatBoost AUC: {catboost_auc:.4f}")
        logger.info(f"XGBoost AUC: {xgb_auc:.4f}")
        logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
        logger.info("=" * 60)
        
        return metrics
    
    def _calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    
    def _log_feature_importance(self):
        if not self.feature_importance:
            return
        
        sorted_features = list(self.feature_importance.items())
        logger.info("=" * 60)
        logger.info("TOP 15 FEATURES:")
        logger.info("=" * 60)
        for i, (feat, imp) in enumerate(sorted_features[:15], 1):
            logger.info(f"  {i:2d}. {feat:30s} | {imp:8.2f}")
    
    def predict(self, symbol: str, df: pd.DataFrame = None) -> MLSignal:
        """Make prediction using ensemble of XGBoost + CatBoost."""
        try:
            # Track if we received pre-loaded data (evaluation mode)
            is_evaluation_mode = df is not None
            
            # Get 1h data if not provided
            if df is None:
                df = collector.get_dataframe(symbol, timeframe="1h", limit=250)
            
            if df.empty or len(df) < 50:
                return MLSignal(signal="hold", confidence=0.0)
            
            # Engineer 1h features
            df = feature_engineer.engineer_all_features(df)
            
            # Add MTF features ONLY in live trading mode (not evaluation)
            # During evaluation, historical df is passed and we can't fetch
            # time-aligned historical 15m/4h data, so we skip MTF features
            if not is_evaluation_mode:
                try:
                    df_15m = collector.get_dataframe(symbol, timeframe="15m", limit=100)
                    df_4h = collector.get_dataframe(symbol, timeframe="4h", limit=50)
                    df = feature_engineer.engineer_multi_timeframe_features(df, df_15m, df_4h)
                except Exception as e:
                    logger.debug(f"MTF data unavailable: {e}")
            
            if not self.is_trained:
                logger.warning("Model not trained")
                return MLSignal(signal="hold", confidence=0.5)
            
            # Prepare features
            X, _ = self.prepare_features(df.tail(1))
            
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Ensemble prediction
            catboost_proba = self.catboost_model.predict_proba(X)[0, 1]
            
            if HAS_XGBOOST and self.xgb_model:
                xgb_proba = self.xgb_model.predict_proba(X)[0, 1]
                proba = (catboost_proba + xgb_proba) / 2  # Soft voting
            else:
                proba = catboost_proba
            
            # Determine signal
            if proba >= self.STRONG_BUY_THRESHOLD:
                signal = "buy"
            elif proba >= self.BUY_THRESHOLD:
                signal = "buy"
            elif proba <= self.SELL_THRESHOLD:
                signal = "sell"
            else:
                signal = "hold"
            
            # Calculate confidence
            if signal == "buy":
                confidence = float(proba)
            elif signal == "sell":
                confidence = float(1 - proba)
            else:
                confidence = float(1 - 2 * abs(proba - 0.5))
            
            top_features = list(self.feature_importance.items())[:5]
            
            result = MLSignal(
                signal=signal,
                confidence=confidence,
                top_features=top_features
            )
            
            log_model_prediction("ML", symbol, signal, confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return MLSignal(signal="hold", confidence=0.5)
    
    def get_quick_score(self, symbol: str) -> Tuple[str, float]:
        result = self.predict(symbol)
        return result.signal, result.confidence


# Backward compatibility aliases
LightGBMModel = HybridMLModel
CatBoostModel = HybridMLModel
XGBoostModel = HybridMLModel

# Global model instance
ml_model = HybridMLModel()
