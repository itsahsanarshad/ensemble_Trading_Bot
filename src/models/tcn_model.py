"""
PyTorch TCN (Temporal Convolutional Network) Model - Improved Version

Improvements:
1. Adaptive Sequence Length (15m=96, 1h=24, 4h=12 candles)
2. Multi-Head Attention mechanism
3. Data augmentation support (4x training data)

Expected improvement: +15-25% accuracy over baseline
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from src.utils import logger


# ============================================================================
# Configuration
# ============================================================================

# Adaptive sequence lengths per timeframe
SEQUENCE_LENGTHS = {
    "15m": 96,   # 24 hours of 15m data
    "1h": 24,    # 24 hours of 1h data  
    "4h": 12,    # 48 hours of 4h data
}
DEFAULT_SEQUENCE_LENGTH = 24  # Default for 1h

# Model architecture
TCN_FILTERS = 32
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8, 16]
ATTENTION_HEADS = 4
ATTENTION_DIM = 32
DROPOUT_RATE = 0.2

# Features used by TCN (19 features including volume)
FEATURES = [
    "return_1", "return_4", "return_16",
    "hl_spread", "close_vs_open",
    "rsi_14", "macd", "momentum_10", "roc_10",
    "bb_position", "atr_14", "volatility_24h",
    "hour", "day_of_week",
    # Volume features (5)
    "volume_ratio_20", "vwap_ratio", "obv_slope", "volume_zscore", "volume_trend"
]


# ============================================================================
# Signal Dataclass
# ============================================================================

@dataclass
class TCNSignal:
    """TCN prediction signal."""
    symbol: str
    timestamp: datetime
    probability: float
    prediction: int  # 0 or 1
    confidence: float
    sequence_pattern: str = ""
    
    @property
    def signal(self) -> str:
        """Convert numeric prediction to buy/hold/sell string for ensemble compatibility."""
        if self.prediction == 1 and self.confidence > 0.6:
            return "buy"
        elif self.prediction == 0 and self.confidence > 0.6:
            return "sell"
        return "hold"
    
    @property
    def pattern(self) -> str:
        """Alias for sequence_pattern for ensemble compatibility."""
        return self.sequence_pattern
    
    @property
    def is_bullish(self) -> bool:
        return self.prediction == 1 and self.confidence > 0.6


# ============================================================================
# Temporal Block (TCN building block)
# ============================================================================

class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolution.
    
    This is the core building block of TCN:
    - Uses dilated causal convolutions
    - Includes residual connections
    - Applies dropout for regularization
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        
        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal trim
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal trim
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ============================================================================
# Multi-Head Attention Layer
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention for temporal sequences.
    
    This allows the model to focus on important timesteps
    in the sequence, improving prediction accuracy.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Add & Norm
        x = self.layer_norm(x + self.dropout(attn_out))
        
        # Back to (batch, channels, seq_len)
        return x.transpose(1, 2)


# ============================================================================
# TCN Model with Attention
# ============================================================================

class TCNWithAttention(nn.Module):
    """
    Temporal Convolutional Network with Multi-Head Attention.
    
    Architecture:
    1. Multiple TemporalBlocks with increasing dilation
    2. Multi-Head Attention layer
    3. Global Average Pooling
    4. Dense layers for classification
    
    Improvements:
    - Attention mechanism focuses on key timesteps
    - Dilated convolutions capture long-range dependencies
    - Residual connections enable deep networks
    """
    
    def __init__(self, input_channels: int, sequence_length: int,
                 num_filters: int = 32, kernel_size: int = 3,
                 dilations: List[int] = None, num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        
        dilations = dilations or [1, 2, 4, 8, 16]
        
        # Temporal blocks
        self.temporal_blocks = nn.ModuleList()
        in_channels = input_channels
        for dilation in dilations:
            self.temporal_blocks.append(
                TemporalBlock(in_channels, num_filters, kernel_size, dilation, dropout)
            )
            in_channels = num_filters
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(num_filters, num_heads, dropout)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters, 32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Temporal blocks
        for block in self.temporal_blocks:
            x = block(x)
        
        # Attention
        x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze(-1)


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_sequences(X: np.ndarray, y: np.ndarray, augment_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Data augmentation for time series sequences.
    
    Techniques:
    1. Gaussian noise injection
    2. Time shifting
    3. Magnitude scaling
    
    Args:
        X: Input sequences (samples, seq_len, features)
        y: Labels
        augment_factor: How many augmented versions to create (1-3)
        
    Returns:
        Augmented X and y arrays (4x original size with factor=3)
    """
    augmented_X = [X]  # Original data
    augmented_y = [y]
    
    # 1. Gaussian noise injection
    if augment_factor >= 1:
        noise_std = 0.02
        noisy = X + np.random.normal(0, noise_std, X.shape)
        augmented_X.append(noisy.astype(np.float32))
        augmented_y.append(y)
    
    # 2. Time shifting (shift by 1 timestep)
    if augment_factor >= 2:
        shifted = np.roll(X, 1, axis=1)
        shifted[:, 0, :] = X[:, 0, :]  # Keep first timestep
        augmented_X.append(shifted.astype(np.float32))
        augmented_y.append(y)
    
    # 3. Magnitude scaling
    if augment_factor >= 3:
        scale_factors = np.random.uniform(0.95, 1.05, X.shape)
        scaled = X * scale_factors
        augmented_X.append(scaled.astype(np.float32))
        augmented_y.append(y)
    
    # Concatenate and shuffle
    result_X = np.concatenate(augmented_X, axis=0)
    result_y = np.concatenate(augmented_y, axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(result_X))
    return result_X[indices], result_y[indices]


# ============================================================================
# TCN Model Wrapper (for production use)
# ============================================================================

class TCNModel:
    """
    Production wrapper for TCN model.
    
    Handles:
    - Model loading/saving
    - Feature scaling
    - Sequence preparation
    - Prediction with confidence
    """
    
    MODEL_PATH = Path("models/tcn_model.pt")
    SCALER_PATH = Path("models/tcn_scaler.pkl")
    
    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        self.sequence_length = SEQUENCE_LENGTHS.get(timeframe, DEFAULT_SEQUENCE_LENGTH)
        self.features = FEATURES
        self.model: Optional[TCNWithAttention] = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model if exists
        self._load_model()
        
    def _load_model(self) -> None:
        """Load trained model and scaler."""
        try:
            if self.MODEL_PATH.exists() and self.SCALER_PATH.exists():
                # Load scaler
                with open(self.SCALER_PATH, 'rb') as f:
                    scaler_data = pickle.load(f)
                    self.scaler = scaler_data.get('scaler')
                    saved_seq_len = scaler_data.get('sequence_length', self.sequence_length)
                    saved_features = scaler_data.get('feature_names', self.features)
                
                # Build model with saved config
                self.model = TCNWithAttention(
                    input_channels=len(saved_features),
                    sequence_length=saved_seq_len,
                    num_filters=TCN_FILTERS,
                    dilations=TCN_DILATIONS,
                    num_heads=ATTENTION_HEADS,
                    dropout=DROPOUT_RATE
                )
                
                # Load weights
                self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Loaded PyTorch TCN model with attention (seq_len={saved_seq_len})")
            else:
                logger.warning("TCN model not found, predictions will be neutral")
        except Exception as e:
            logger.warning(f"Could not load TCN model: {e}")
            self.model = None
            
    def set_timeframe(self, timeframe: str) -> None:
        """Set timeframe and adjust sequence length."""
        self.timeframe = timeframe
        self.sequence_length = SEQUENCE_LENGTHS.get(timeframe, DEFAULT_SEQUENCE_LENGTH)
        logger.info(f"TCN timeframe set to {timeframe}, sequence_length={self.sequence_length}")
    
    def prepare_sequence(self, df) -> Optional[np.ndarray]:
        """Prepare sequence from dataframe."""
        if len(df) < self.sequence_length:
            return None
            
        # Get last sequence_length rows
        data = df[self.features].tail(self.sequence_length).values
        
        # Handle NaN
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Scale if scaler available
        if self.scaler is not None:
            data = self.scaler.transform(data)
        
        return data.astype(np.float32)
    
    def _engineer_tcn_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all 19 TCN features from raw OHLCV data.
        
        Features: return_1, return_4, return_16, hl_spread, close_vs_open,
                  rsi_14, macd, momentum_10, roc_10, bb_position, atr_14,
                  volatility_24h, hour, day_of_week, volume_ratio_20,
                  vwap_ratio, obv_slope, volume_zscore, volume_trend
        """
        df = df.copy()
        
        # Returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_4"] = df["close"].pct_change(4)
        df["return_16"] = df["close"].pct_change(16)
        
        # Price features
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
        df["close_vs_open"] = (df["close"] - df["open"]) / df["open"]
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        
        # Momentum and ROC
        df["momentum_10"] = df["close"] - df["close"].shift(10)
        df["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100
        
        # Bollinger Band position
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # ATR
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift())
        tr3 = abs(df["low"] - df["close"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        
        # Volatility
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["volatility_24h"] = log_ret.rolling(24).std() * np.sqrt(24)
        
        # Time features (from timestamp if available)
        if "timestamp" in df.columns:
            dt = pd.to_datetime(df["timestamp"], unit="ms")
            df["hour"] = dt.dt.hour
            df["day_of_week"] = dt.dt.dayofweek
        else:
            df["hour"] = 12
            df["day_of_week"] = 3
        
        # Volume features
        vol_sma = df["volume"].rolling(20).mean()
        df["volume_ratio_20"] = df["volume"] / vol_sma.replace(0, 1)
        
        # VWAP ratio
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (tp * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        df["vwap_ratio"] = df["close"] / vwap.replace(0, 1)
        
        # OBV slope (rolling)
        signed_vol = np.sign(df["close"].diff()) * df["volume"]
        obv_rolling = signed_vol.rolling(20).sum()
        df["obv_slope"] = obv_rolling.diff(10) / (obv_rolling.rolling(20).std() + 1e-10)
        
        # Volume z-score
        vol_mean = df["volume"].rolling(50).mean()
        vol_std = df["volume"].rolling(50).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, 1)
        
        # Volume trend
        df["volume_trend"] = df["volume"].rolling(10).mean() / df["volume"].rolling(30).mean()
        
        # Fill NaN
        df = df.ffill().bfill().fillna(0)
        
        return df
    
    def predict(self, symbol_or_df, symbol: str = "UNKNOWN") -> TCNSignal:
        """
        Generate prediction signal.
        
        Args:
            symbol_or_df: Either a symbol string (e.g., 'BTCUSDT') or a DataFrame
            symbol: Used as symbol name when df is provided
        """
        from src.data import collector
        
        timestamp = datetime.utcnow()
        
        # Determine if we received a symbol or DataFrame
        if isinstance(symbol_or_df, str):
            # Received symbol string - fetch data
            symbol = symbol_or_df
            try:
                df = collector.get_dataframe(symbol, timeframe="1h", limit=50)
                if df.empty:
                    return TCNSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        probability=0.5,
                        prediction=0,
                        confidence=0.0,
                        sequence_pattern="no_data"
                    )
                # Engineer features for TCN
                df = self._engineer_tcn_features(df)
            except Exception as e:
                logger.warning(f"TCN data fetch error for {symbol}: {e}")
                return TCNSignal(
                    symbol=symbol,
                    timestamp=timestamp,
                    probability=0.5,
                    prediction=0,
                    confidence=0.0,
                    sequence_pattern="data_error"
                )
        else:
            # Received DataFrame - need to engineer TCN-specific features
            df = symbol_or_df
            # Check if TCN features are missing and need to be engineered
            required_features = ["return_1", "close_vs_open", "roc_10", "obv_slope"]
            if not all(f in df.columns for f in required_features):
                df = self._engineer_tcn_features(df)
        
        # Default neutral signal
        if self.model is None:
            return TCNSignal(
                symbol=symbol,
                timestamp=timestamp,
                probability=0.5,
                prediction=0,
                confidence=0.0,
                sequence_pattern="no_model"
            )
        
        try:
            # Prepare sequence
            sequence = self.prepare_sequence(df)
            if sequence is None:
                return TCNSignal(
                    symbol=symbol,
                    timestamp=timestamp,
                    probability=0.5,
                    prediction=0,
                    confidence=0.0,
                    sequence_pattern="insufficient_data"
                )
            
            # Convert to tensor
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                prob = self.model(x).item()
            
            # Generate signal
            prediction = 1 if prob > 0.5 else 0
            confidence = abs(prob - 0.5) * 2  # Scale to 0-1
            
            # Analyze pattern
            pattern = self._analyze_pattern(sequence)
            
            return TCNSignal(
                symbol=symbol,
                timestamp=timestamp,
                probability=prob,
                prediction=prediction,
                confidence=confidence,
                sequence_pattern=pattern
            )
            
        except Exception as e:
            logger.error(f"TCN prediction error: {e}")
            return TCNSignal(
                symbol=symbol,
                timestamp=timestamp,
                probability=0.5,
                prediction=0,
                confidence=0.0,
                sequence_pattern="error"
            )
    
    def _analyze_pattern(self, sequence: np.ndarray) -> str:
        """Analyze sequence pattern for interpretability."""
        if sequence is None or len(sequence) < 5:
            return "unknown"
        
        # Check recent returns (first 3 columns are returns)
        returns = sequence[-5:, :3].mean(axis=1)
        
        if np.all(returns > 0):
            return "uptrend"
        elif np.all(returns < 0):
            return "downtrend"
        elif returns[-1] > returns[0]:
            return "recovering"
        elif returns[-1] < returns[0]:
            return "weakening"
        else:
            return "consolidating"
    
    def save_model(self, model: TCNWithAttention, scaler, metrics: Dict) -> None:
        """Save trained model and scaler."""
        # Ensure directory exists
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), self.MODEL_PATH)
        
        # Save scaler and metadata
        with open(self.SCALER_PATH, 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'feature_names': self.features,
                'sequence_length': self.sequence_length,
                'timeframe': self.timeframe,
                'has_attention': True,
                'version': '2.0-pytorch',
                'metrics': metrics,
                'saved_at': datetime.utcnow().isoformat()
            }, f)
        
        logger.info(f"Saved PyTorch TCN model to {self.MODEL_PATH}")


# ============================================================================
# Global instance for production use
# ============================================================================

# Create default instance
tcn_model = TCNModel(timeframe="1h")
