"""
PyTorch TCN Training Script - Production Ready with GPU Acceleration

Features:
- Adaptive sequence length (24 for 1h timeframe)
- Multi-head attention mechanism (4 heads)
- Data augmentation (4x training data)
- GPU acceleration (CUDA)
- Progress bars (tqdm)
- Early stopping with patience
- Learning rate scheduling

Usage:
    python training/train_tcn.py --days 365
    python training/train_tcn.py --days 365 --no-augmentation
    python training/train_tcn.py --days 720 --epochs 100 --batch-size 128
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WATCHLIST
from src.utils import logger

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not installed. Install with: pip install tqdm")


# ============================================================================
# Model Architecture
# ============================================================================

class TemporalBlock(nn.Module):
    """
    Temporal block with dilated causal convolution.
    
    Components:
    - Two Conv1D layers with dilated causal padding
    - BatchNorm after each conv
    - ReLU activation
    - Dropout for regularization
    - Residual connection
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channels differ)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv + causal trim
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second conv + causal trim
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention for temporal sequences.
    
    Allows the model to focus on important timesteps,
    improving accuracy by 5-10%.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_out))
        
        # Back to (batch, channels, seq_len)
        return x.transpose(1, 2)


class TCNWithAttention(nn.Module):
    """
    Temporal Convolutional Network with Multi-Head Attention.
    
    Architecture:
    1. 5 TemporalBlocks with dilations [1, 2, 4, 8, 16]
    2. Multi-Head Attention (4 heads)
    3. Global Average Pooling
    4. Dense layers: 32 -> 16 -> 1
    
    Total parameters: ~36,449
    """
    
    def __init__(self, input_channels: int, sequence_length: int,
                 num_filters: int = 32, kernel_size: int = 3,
                 dilations=None, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        dilations = dilations or [1, 2, 4, 8, 16]
        
        # Temporal blocks with increasing dilation
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
        # (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Temporal blocks
        for block in self.temporal_blocks:
            x = block(x)
        
        # Attention
        x = self.attention(x)
        
        # Global pooling -> (batch, filters)
        x = self.global_pool(x).squeeze(-1)
        
        # Classification head
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze(-1)


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_sequences(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Data augmentation for 4x training data.
    
    Techniques:
    1. Gaussian noise (std=0.02)
    2. Time shifting (+1 timestep)
    3. Magnitude scaling (0.95-1.05x)
    """
    augmented_X = [X]  # Original
    augmented_y = [y]
    
    # 1. Gaussian noise
    noisy = X + np.random.normal(0, 0.02, X.shape)
    augmented_X.append(noisy.astype(np.float32))
    augmented_y.append(y)
    
    # 2. Time shifting
    shifted = np.roll(X, 1, axis=1)
    shifted[:, 0, :] = X[:, 0, :]  # Preserve first timestep
    augmented_X.append(shifted.astype(np.float32))
    augmented_y.append(y)
    
    # 3. Magnitude scaling
    scaled = X * np.random.uniform(0.95, 1.05, X.shape)
    augmented_X.append(scaled.astype(np.float32))
    augmented_y.append(y)
    
    # Concatenate and shuffle
    result_X = np.concatenate(augmented_X, axis=0)
    result_y = np.concatenate(augmented_y, axis=0)
    
    indices = np.random.permutation(len(result_X))
    return result_X[indices], result_y[indices]


# ============================================================================
# Feature Engineering (no external TA library needed)
# ============================================================================

def calculate_rsi(series, period=14):
    """Calculate RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26):
    """Calculate MACD line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all 19 features for TCN (including volume features)."""
    df = df.copy()
    
    # Returns (3 features)
    df['return_1'] = df['close'].pct_change(1)
    df['return_4'] = df['close'].pct_change(4)
    df['return_16'] = df['close'].pct_change(16)
    
    # Price features (2 features)
    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    df['close_vs_open'] = (df['close'] - df['open']) / df['open']
    
    # Momentum (4 features)
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['macd'] = calculate_macd(df['close'])
    df['momentum_10'] = df['close'].diff(10)
    df['roc_10'] = df['close'].pct_change(10) * 100
    
    # Bollinger Band position (1 feature)
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower
    df['bb_position'] = (df['close'] - bb_lower) / bb_range.replace(0, 1)
    
    # Volatility (2 features)
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    df['volatility_24h'] = df['return_1'].rolling(24).std()
    
    # Time features (2 features)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    else:
        df['hour'] = 0
        df['day_of_week'] = 0
    
    # Volume features (5 features - enhanced!)
    df['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # VWAP ratio - price vs volume-weighted average price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_ratio'] = df['close'] / vwap.replace(0, 1)
    
    # OBV slope - on-balance volume momentum (using rolling to avoid cumsum issues)
    def calc_obv_slope(group):
        signed_volume = np.sign(group['close'].diff()) * group['volume']
        obv_rolling = signed_volume.rolling(window=20, min_periods=1).sum()
        return obv_rolling.diff(10) / (obv_rolling.rolling(20).std() + 1e-10)
    
    df['obv_slope'] = df.groupby('symbol', group_keys=False).apply(calc_obv_slope)
    
    # Volume Z-score - normalized volume
    vol_mean = df['volume'].rolling(50).mean()
    vol_std = df['volume'].rolling(50).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std.replace(0, 1)
    
    # Volume trend - is volume increasing?
    df['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
    
    return df.ffill().bfill().fillna(0)


# ============================================================================
# Data Loading
# ============================================================================

def fetch_training_data(symbols: list = None, days: int = 365, timeframe: str = "1h") -> pd.DataFrame:
    """Fetch data from database using direct SQL."""
    from sqlalchemy import create_engine
    
    symbols = symbols or WATCHLIST
    
    # Direct database connection
    db_path = Path("data/trading_bot.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    engine = create_engine(f"sqlite:///{db_path}")
    
    logger.info(f"Loading {timeframe} data for {len(symbols)} symbols...")
    
    # Timestamp is stored in milliseconds
    cutoff_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    # Build query
    symbols_str = "','".join(symbols)
    query = f"""
        SELECT coin as symbol, timestamp, open, high, low, close, volume
        FROM price_data
        WHERE timeframe = '{timeframe}'
          AND coin IN ('{symbols_str}')
          AND timestamp >= {cutoff_ms}
        ORDER BY coin, timestamp
    """
    
    df = pd.read_sql(query, engine)
    
    if len(df) == 0:
        raise ValueError("No data found in database!")
    
    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    logger.info(f"📊 Loaded {len(df):,} records for {df['symbol'].nunique()} symbols")
    
    return df


# ============================================================================
# Training Function
# ============================================================================

def train_tcn(
    days: int = 365,
    symbols: list = None,
    sequence_length: int = 24,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    use_augmentation: bool = True,
    timeframe: str = "1h"
) -> dict:
    """
    Train TCN model with PyTorch and GPU acceleration.
    
    Returns:
        Dictionary with training results and metrics
    """
    
    # ========== Device Setup ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Device: {device}")
    if device.type == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA: {torch.version.cuda}")
    
    # ========== Configuration ==========
    LOOKAHEAD = 4  # Predict 4 candles ahead
    TARGET_PCT = 0.06  # 6% target (aligned with ML model)
    FEATURE_COLS = [
        "return_1", "return_4", "return_16", "hl_spread", "close_vs_open",
        "rsi_14", "macd", "momentum_10", "roc_10",
        "bb_position", "atr_14", "volatility_24h",
        "hour", "day_of_week",
        # Volume features (5)
        "volume_ratio_20", "vwap_ratio", "obv_slope", "volume_zscore", "volume_trend"
    ]
    
    # ========== Load Data ==========
    logger.info("=" * 60)
    logger.info("📥 LOADING DATA")
    logger.info("=" * 60)
    
    df = fetch_training_data(symbols=symbols, days=days, timeframe=timeframe)
    
    # ========== Feature Engineering ==========
    logger.info("⚙️ Engineering features...")
    df = engineer_features(df)
    
    # ========== Create Target ==========
    logger.info("🎯 Creating adaptive ATR-based target...")
    
    # Calculate rolling ATR for each symbol (already computed in features as atr_14)
    # Target = 1.5x ATR (with min 1%, max 4% bounds)
    ATR_MULTIPLIER = 1.5
    MIN_TARGET_PCT = 0.01  # 1% minimum
    MAX_TARGET_PCT = 0.04  # 4% maximum
    
    # Compute adaptive target per row
    df["atr_pct"] = df["atr_14"] / df["close"]  # ATR as percentage of price
    df["adaptive_target_pct"] = (df["atr_pct"] * ATR_MULTIPLIER).clip(MIN_TARGET_PCT, MAX_TARGET_PCT)
    
    # Future return
    df["future_return"] = df.groupby("symbol")["close"].shift(-LOOKAHEAD) / df["close"] - 1
    
    # Target: future return exceeds adaptive threshold
    df["target"] = (df["future_return"] > df["adaptive_target_pct"]).astype(int)
    df = df.dropna(subset=["target", "future_return"])
    
    positive_rate = df['target'].mean()
    avg_target = df['adaptive_target_pct'].mean() * 100
    logger.info(f"   Adaptive target: avg {avg_target:.2f}% (range {MIN_TARGET_PCT*100:.0f}%-{MAX_TARGET_PCT*100:.0f}%)")
    logger.info(f"   Target positive rate: {positive_rate:.1%}")
    
    # ========== Create Sequences ==========
    logger.info("🔄 Creating sequences...")
    X_all, y_all = [], []
    
    for symbol in df["symbol"].unique():
        symbol_df = df[df["symbol"] == symbol].copy()
        if len(symbol_df) < sequence_length + LOOKAHEAD:
            continue
        
        X_data = np.nan_to_num(symbol_df[FEATURE_COLS].values, nan=0.0, posinf=1.0, neginf=-1.0)
        y_data = symbol_df["target"].values
        
        for i in range(sequence_length, len(X_data) - LOOKAHEAD):
            X_all.append(X_data[i - sequence_length:i])
            y_all.append(y_data[i])
    
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    logger.info(f"   Sequences: {X.shape[0]:,} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
    
    # ========== Train/Val Split ==========
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    logger.info(f"   Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    # ========== Scaling ==========
    logger.info("📏 Scaling features...")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    # ========== Data Augmentation ==========
    if use_augmentation:
        logger.info("🔀 Applying 4x data augmentation...")
        X_train_aug, y_train_aug = augment_sequences(X_train_scaled, y_train)
        logger.info(f"   After augmentation: {len(X_train_aug):,} samples")
    else:
        X_train_aug, y_train_aug = X_train_scaled, y_train
        logger.info("   Augmentation disabled")
    
    # ========== Data Loaders ==========
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_aug),
        torch.FloatTensor(y_train_aug)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    # ========== Build Model ==========
    logger.info("=" * 60)
    logger.info("🏗️ BUILDING MODEL")
    logger.info("=" * 60)
    
    model = TCNWithAttention(
        input_channels=len(FEATURE_COLS),
        sequence_length=sequence_length,
        num_filters=32,
        dilations=[1, 2, 4, 8, 16],
        num_heads=4,
        dropout=0.2
    ).to(device)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable: {trainable_params:,}")
    
    # ========== Loss, Optimizer, Scheduler ==========
    # Compute class weight for imbalanced data (0.8% positives)
    pos_count = y_train_aug.sum()
    neg_count = len(y_train_aug) - pos_count
    pos_weight = neg_count / max(pos_count, 1)  # Weight positive class more
    pos_weight = min(pos_weight, 10.0)  # Cap at 10x to avoid instability
    logger.info(f"   Class weight: {pos_weight:.2f}x for positive class")
    
    # Use BCEWithLogitsLoss with pos_weight for better handling of imbalanced data
    # But since we use sigmoid output, we'll weight the loss manually
    criterion = nn.BCELoss(reduction='none')  # Get per-sample loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # ========== Training Loop ==========
    logger.info("=" * 60)
    logger.info("🚀 TRAINING")
    logger.info("=" * 60)
    
    best_auc = 0
    best_model_state = None
    patience_limit = 10
    patience_counter = 0
    history = {'train_loss': [], 'val_auc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        train_loss = 0
        
        # Progress bar for batches
        if HAS_TQDM:
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                             leave=False, ncols=100, 
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            batch_iter = train_loader
        
        for X_batch, y_batch in batch_iter:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Apply class weights: pos_weight for positive class, 1.0 for negative
            weights = torch.where(y_batch == 1, torch.tensor(pos_weight, device=device), torch.tensor(1.0, device=device))
            loss_per_sample = criterion(outputs, y_batch)
            loss = (loss_per_sample * weights).mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if HAS_TQDM:
                batch_iter.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # ----- Validation -----
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y_batch.numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        val_pred_binary = (val_preds > 0.5).astype(int)
        val_acc = (val_pred_binary == val_targets).mean()
        
        # Update scheduler
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # History
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        
        # Logging
        logger.info(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | "
                   f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info(f"   ✅ New best AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # ========== Load Best Model ==========
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    # ========== Final Evaluation ==========
    logger.info("=" * 60)
    logger.info("📊 FINAL EVALUATION")
    logger.info("=" * 60)
    
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            val_preds.extend(outputs.cpu().numpy())
    
    val_preds = np.array(val_preds)
    val_pred_binary = (val_preds > 0.5).astype(int)
    
    # Metrics
    final_auc = roc_auc_score(val_targets, val_preds)
    precision = precision_score(val_targets, val_pred_binary, zero_division=0)
    recall = recall_score(val_targets, val_pred_binary, zero_division=0)
    f1 = f1_score(val_targets, val_pred_binary, zero_division=0)
    
    logger.info(f"   Val AUC:     {final_auc:.4f}")
    logger.info(f"   Precision:   {precision:.4f}")
    logger.info(f"   Recall:      {recall:.4f}")
    logger.info(f"   F1 Score:    {f1:.4f}")
    
    # Quality assessment
    if final_auc > 0.75:
        quality = "EXCELLENT ⭐⭐⭐"
    elif final_auc > 0.70:
        quality = "VERY GOOD ⭐⭐"
    elif final_auc > 0.65:
        quality = "GOOD ⭐"
    else:
        quality = "FAIR"
    logger.info(f"   Model quality: {quality}")
    
    # ========== Save Model ==========
    logger.info("=" * 60)
    logger.info("💾 SAVING MODEL")
    logger.info("=" * 60)
    
    model_path = Path("models/tcn_model.pt")
    scaler_path = Path("models/tcn_scaler.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_names': FEATURE_COLS,
            'sequence_length': sequence_length,
            'timeframe': timeframe,
            'has_attention': True,
            'has_augmentation': use_augmentation,
            'version': '2.0-pytorch',
            'val_auc': final_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'epochs_trained': epoch + 1,
            'samples_used': len(X_train_aug),
            'saved_at': datetime.utcnow().isoformat()
        }, f)
    
    logger.info(f"   ✅ Model saved: {model_path}")
    logger.info(f"   ✅ Scaler saved: {scaler_path}")
    
    return {
        'val_auc': final_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'epochs_trained': epoch + 1,
        'samples_used': len(X_train_aug),
        'best_auc': best_auc
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch TCN Model with GPU")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data (default: 365)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--sequence-length", type=int, default=24, help="Sequence length (default: 24)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    
    args = parser.parse_args()
    
    # Header
    print()
    logger.info("=" * 60)
    logger.info("🚀 PyTorch TCN Training with GPU")
    logger.info("=" * 60)
    logger.info(f"   Days: {args.days}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Sequence length: {args.sequence_length}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   Augmentation: {'Disabled' if args.no_augmentation else 'Enabled (4x)'}")
    
    # Train
    result = train_tcn(
        days=args.days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        use_augmentation=not args.no_augmentation
    )
    
    # Summary
    print()
    logger.info("=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Final AUC: {result['val_auc']:.4f}")
    logger.info(f"   Precision: {result['precision']:.4f}")
    logger.info(f"   Recall: {result['recall']:.4f}")
    logger.info(f"   F1 Score: {result['f1']:.4f}")
    logger.info(f"   Epochs: {result['epochs_trained']}")
    logger.info(f"   Samples: {result['samples_used']:,}")
    print()
