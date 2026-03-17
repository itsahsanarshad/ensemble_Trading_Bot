"""
Database Models and Connection Manager

Uses SQLAlchemy with async support for SQLite.
"""

from datetime import datetime
from typing import Optional, List
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    DateTime,
    Text,
    Enum as SQLEnum,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

import sys
sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 3)[0])
from config import settings

Base = declarative_base()


class TradeStatus(str, Enum):
    """Trade status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"


class ExitReason(str, Enum):
    """Exit reason enumeration."""
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    MANUAL = "MANUAL"
    PARTIAL_EXIT = "PARTIAL_EXIT"


class PriceData(Base):
    """OHLCV price data storage."""
    
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    coin = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float)
    num_trades = Column(Integer)
    
    __table_args__ = (
        UniqueConstraint("timestamp", "coin", "timeframe", name="uix_price_data"),
        Index("ix_price_lookup", "coin", "timeframe", "timestamp"),
    )


class Indicator(Base):
    """Pre-calculated technical indicators."""
    
    __tablename__ = "indicators"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    coin = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    
    # RSI
    rsi_7 = Column(Float)
    rsi_14 = Column(Float)
    rsi_21 = Column(Float)
    
    # MACD
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    
    # Moving Averages
    ema_20 = Column(Float)
    ema_50 = Column(Float)
    sma_200 = Column(Float)
    
    # Bollinger Bands
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    
    # Volume
    volume_sma_20 = Column(Float)
    volume_ratio = Column(Float)
    
    # Volatility
    atr_14 = Column(Float)
    
    # ADX
    adx = Column(Float)
    
    # Stochastic
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    
    # OBV
    obv = Column(Float)
    
    __table_args__ = (
        UniqueConstraint("timestamp", "coin", "timeframe", name="uix_indicators"),
        Index("ix_indicator_lookup", "coin", "timeframe", "timestamp"),
    )


class Trade(Base):
    """Trade records for tracking and analysis."""
    
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    coin = Column(String(20), nullable=False, index=True)
    
    # Entry
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    entry_price = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    position_size_coins = Column(Float)
    
    # Exit
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    exit_reason = Column(String(50))
    
    # P&L
    pnl_percent = Column(Float)
    pnl_usd = Column(Float)
    
    # Consensus info
    consensus_tier = Column(Integer)
    ta_confidence = Column(Float)
    ml_confidence = Column(Float)
    tcn_confidence = Column(Float)
    entry_reason = Column(Text)
    
    # Status
    status = Column(String(20), default="OPEN")
    
    # Stop/Take profit levels
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)
    trailing_stop_price = Column(Float)
    highest_price = Column(Float)
    
    # Partial exits
    partial_exit_done = Column(Integer, default=0)
    remaining_size = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_trade_status", "status", "coin"),
    )


class ModelPerformance(Base):
    """Track model accuracy over time."""
    
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    auc = Column(Float)
    
    trades_count = Column(Integer)
    win_rate = Column(Float)
    avg_return = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelPrediction(Base):
    """Every prediction from each model per scan.
    
    Replaces performance_tracker.json. Outcomes are backfilled
    when the corresponding trade is closed.
    """
    
    __tablename__ = "model_predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    model = Column(String(10), nullable=False)  # 'ta', 'ml', 'tcn'
    signal = Column(String(10), nullable=False)  # 'buy', 'sell', 'hold'
    confidence = Column(Float, nullable=False)
    predicted_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    trade_id = Column(String(50), nullable=True)  # linked when trade opens
    outcome = Column(String(10), nullable=True)   # 'WIN', 'LOSS', 'NO_TRADE' — set on close
    pnl_pct = Column(Float, nullable=True)        # actual P&L set on close
    
    __table_args__ = (
        Index("ix_prediction_lookup", "symbol", "model", "predicted_at"),
    )


class BotState(Base):
    """Auditable paper balance ledger.
    
    Replaces bot_state.json. Latest row = current balance.
    Full history = see how capital evolved over time.
    """
    
    __tablename__ = "bot_state"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_balance = Column(Float, nullable=False)
    trades_count = Column(Integer, default=0)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    note = Column(String(50), nullable=True)  # 'startup', 'reset', 'trade', 'daily'



class DailyPerformance(Base):
    """Daily portfolio performance tracking."""
    
    __tablename__ = "daily_performance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True)
    
    starting_capital = Column(Float)
    ending_capital = Column(Float)
    daily_pnl = Column(Float)
    daily_pnl_percent = Column(Float)
    
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    max_drawdown = Column(Float)
    best_trade = Column(Float)
    worst_trade = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)



class DatabaseManager:
    """Database connection and operations manager."""
    
    def __init__(self, database_url: str = None):
        """Initialize database connection."""
        self.database_url = database_url or settings.database.database_url
        
        # SQLite specific settings
        if "sqlite" in self.database_url:
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            self.engine = create_engine(self.database_url)
        
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def save_price_data(self, data: List[dict], coin: str, timeframe: str) -> int:
        """
        Save OHLCV data to database.
        
        Args:
            data: List of OHLCV dictionaries
            coin: Trading pair
            timeframe: Candlestick timeframe
        
        Returns:
            Number of records saved
        """
        session = self.get_session()
        try:
            count = 0
            for candle in data:
                existing = session.query(PriceData).filter(
                    PriceData.timestamp == candle["timestamp"],
                    PriceData.coin == coin,
                    PriceData.timeframe == timeframe
                ).first()
                
                if not existing:
                    price_data = PriceData(
                        timestamp=candle["timestamp"],
                        coin=coin,
                        timeframe=timeframe,
                        open=candle["open"],
                        high=candle["high"],
                        low=candle["low"],
                        close=candle["close"],
                        volume=candle["volume"],
                        quote_volume=candle.get("quote_volume"),
                        num_trades=candle.get("num_trades"),
                    )
                    session.add(price_data)
                    count += 1
            
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_price_data(
        self,
        coin: str,
        timeframe: str,
        limit: int = 500,
        start_time: int = None
    ) -> List[PriceData]:
        """
        Retrieve price data from database.
        
        Args:
            coin: Trading pair
            timeframe: Candlestick timeframe
            limit: Maximum records to return
            start_time: Start timestamp (optional)
        
        Returns:
            List of PriceData objects
        """
        session = self.get_session()
        try:
            query = session.query(PriceData).filter(
                PriceData.coin == coin,
                PriceData.timeframe == timeframe
            )
            
            if start_time:
                query = query.filter(PriceData.timestamp >= start_time)
            
            return query.order_by(PriceData.timestamp.desc()).limit(limit).all()
        finally:
            session.close()
    
    def save_trade(self, trade: Trade) -> Trade:
        """Save a trade record."""
        session = self.get_session()
        try:
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def update_trade(self, trade_id: str, **updates) -> Trade:
        """Update an existing trade."""
        session = self.get_session()
        try:
            trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)
                trade.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(trade)
            return trade
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_open_trades(self) -> List[Trade]:
        """Get all open trades."""
        session = self.get_session()
        try:
            return session.query(Trade).filter(Trade.status == "OPEN").all()
        finally:
            session.close()
    
    def get_trades_by_coin(self, coin: str, status: str = None) -> List[Trade]:
        """Get trades for a specific coin."""
        session = self.get_session()
        try:
            query = session.query(Trade).filter(Trade.coin == coin)
            if status:
                query = query.filter(Trade.status == status)
            return query.order_by(Trade.entry_time.desc()).all()
        finally:
            session.close()
    
    def get_daily_stats(self, date: datetime = None) -> dict:
        """Get trading stats for a specific day."""
        date = date or datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        session = self.get_session()
        try:
            trades = session.query(Trade).filter(
                Trade.entry_time >= date,
                Trade.status == "CLOSED"
            ).all()
            
            if not trades:
                return {"total": 0, "wins": 0, "losses": 0, "pnl": 0}
            
            wins = sum(1 for t in trades if t.pnl_percent and t.pnl_percent > 0)
            losses = sum(1 for t in trades if t.pnl_percent and t.pnl_percent < 0)
            pnl = sum(t.pnl_usd or 0 for t in trades)
            
            return {
                "total": len(trades),
                "wins": wins,
                "losses": losses,
                "pnl": pnl,
                "win_rate": wins / len(trades) if trades else 0
            }
        finally:
            session.close()
    
    # -------------------------------------------------------------------------
    # Model Prediction Methods
    # -------------------------------------------------------------------------
    
    def save_prediction(
        self,
        symbol: str,
        model: str,
        signal: str,
        confidence: float,
        trade_id: str = None
    ) -> None:
        """Save a model prediction to DB. Called during every scan."""
        session = self.get_session()
        try:
            pred = ModelPrediction(
                symbol=symbol,
                model=model,
                signal=signal,
                confidence=confidence,
                predicted_at=datetime.utcnow(),
                trade_id=trade_id
            )
            session.add(pred)
            session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()
    
    def backfill_prediction_outcomes(
        self,
        symbol: str,
        trade_id: str,
        outcome: str,
        pnl_pct: float,
        entry_time: datetime
    ) -> int:
        """Backfill outcome on predictions made around entry time.
        
        Called when a trade closes. Matches predictions made within
        10 minutes of entry and updates outcome + pnl_pct.
        Returns number of rows updated.
        """
        session = self.get_session()
        try:
            from datetime import timedelta
            window_start = entry_time - timedelta(minutes=10)
            window_end = entry_time + timedelta(minutes=10)
            
            preds = session.query(ModelPrediction).filter(
                ModelPrediction.symbol == symbol,
                ModelPrediction.predicted_at >= window_start,
                ModelPrediction.predicted_at <= window_end,
                ModelPrediction.outcome == None  # only unfilled
            ).all()
            
            for pred in preds:
                # A prediction is correct if it called 'buy' and we won, or 'sell' and price fell
                pred.outcome = outcome
                pred.pnl_pct = pnl_pct
                pred.trade_id = trade_id
            
            session.commit()
            return len(preds)
        except Exception:
            session.rollback()
            return 0
        finally:
            session.close()
    
    def get_model_accuracy_stats(self, days: int = 30) -> dict:
        """Compute per-model accuracy stats over the last N days.
        
        Accuracy = fraction of 'buy' predictions followed by a WIN outcome.
        Only counts predictions that led to a trade (outcome != null).
        """
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            preds = session.query(ModelPrediction).filter(
                ModelPrediction.predicted_at >= cutoff,
                ModelPrediction.outcome != None
            ).all()
            
            stats = {}
            for model in ('ta', 'ml', 'tcn'):
                model_preds = [p for p in preds if p.model == model]
                buy_preds = [p for p in model_preds if p.signal == 'buy']
                wins = [p for p in buy_preds if p.outcome == 'WIN']
                total_signals = len(model_preds)
                buy_count = len(buy_preds)
                
                stats[model] = {
                    "total_predictions": total_signals,
                    "buy_signals": buy_count,
                    "wins": len(wins),
                    "accuracy": len(wins) / buy_count if buy_count > 0 else None,
                    "avg_pnl": sum(p.pnl_pct or 0 for p in buy_preds) / buy_count if buy_count > 0 else None,
                }
            
            return stats
        finally:
            session.close()
    
    # -------------------------------------------------------------------------
    # Bot State Methods
    # -------------------------------------------------------------------------
    
    def save_bot_state(self, paper_balance: float, trades_count: int = 0, note: str = "trade") -> None:
        """Append current balance to the audit ledger."""
        session = self.get_session()
        try:
            row = BotState(
                paper_balance=paper_balance,
                trades_count=trades_count,
                recorded_at=datetime.utcnow(),
                note=note
            )
            session.add(row)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
    
    def get_latest_bot_state(self) -> dict:
        """Load the latest balance from the BotState ledger."""
        session = self.get_session()
        try:
            row = session.query(BotState).order_by(BotState.recorded_at.desc()).first()
            if row:
                return {
                    "paper_balance": row.paper_balance,
                    "trades_count": row.trades_count,
                    "last_updated": row.recorded_at.isoformat(),
                    "metadata": {}
                }
            return None
        finally:
            session.close()


# Global database manager instance
db = DatabaseManager()
