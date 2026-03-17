import logging
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.executor import executor, ExecutionMode
from src.trading.positions import position_manager
from src.trading.risk import risk_manager
from src.data.database import db, Trade, TradeStatus
from src.data.state import save_state
from src.models.ensemble import ensemble
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.utils import logger
from src.bot import TradingBot

app = FastAPI(title="Trading Bot Dashboard API")

# Setup CORS for easy standalone dev if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.on_event("startup")
async def startup_event():
    """Initial database check on dashboard startup."""
    logger.info("Initializing Database schema check via Dashboard...")
    try:
        db.create_tables()
        logger.info("Database available.")
    except Exception as e:
        logger.error(f"Database sync issue: {e}")

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/status")
async def get_status():
    """Get high-level system status and health."""
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": executor.mode.value,
        "paper_balance": executor.paper_balance if executor.mode == ExecutionMode.PAPER else None,
        "regime": ensemble.regime_detector.current_regime.value if hasattr(ensemble.regime_detector.current_regime, 'value') else str(ensemble.regime_detector.current_regime)
    }


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio metrics, risk exposure, and P&L breakdown."""
    from src.data.database import Trade, TradeStatus
    from src.data.collector import collector as data_collector
    
    # Sync risk metrics
    risk_manager._sync_with_database()
    
    positions = position_manager.get_open_positions()
    total_exposure = sum(p["position_size"] for p in positions)

    # ------------------------------------------------------------------
    # Unrealized P&L — fetch live price for each open position server-side
    # Uses the bot's already-connected Binance public client (no extra auth)
    # ------------------------------------------------------------------
    unrealized_pnl_usd = 0.0
    unrealized_pnl_pct = 0.0
    try:
        if positions:
            for pos in positions:
                try:
                    ticker = data_collector.public_client.get_symbol_ticker(symbol=pos["coin"])
                    live_price = float(ticker["price"])
                    entry_price = pos["entry_price"]
                    # Use remaining_size (in coins) to compute mark-to-market P&L
                    remaining_size = pos.get("remaining_size", pos["position_size"])
                    coins_held = remaining_size / entry_price
                    unrealized_pnl_usd += (live_price - entry_price) * coins_held
                except Exception:
                    pass  # Skip if single ticker fails
            if total_exposure > 0:
                unrealized_pnl_pct = (unrealized_pnl_usd / total_exposure) * 100
    except Exception:
        pass  # Don't let P&L calc error kill the whole endpoint

    # ------------------------------------------------------------------
    # Realized P&L — query CLOSED trades from DB
    # ------------------------------------------------------------------
    realized_pnl_today = 0.0
    realized_pnl_alltime = 0.0
    try:
        session = db.get_session()
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        closed_trades = (
            session.query(Trade)
            .filter(Trade.status == TradeStatus.CLOSED)
            .all()
        )
        for t in closed_trades:
            pnl = t.pnl_usd or 0.0
            realized_pnl_alltime += pnl
            if t.exit_time and t.exit_time >= today_start:
                realized_pnl_today += pnl
        session.close()
    except Exception:
        pass

    return {
        "capital": risk_manager.current_capital,
        "total_risk": risk_manager._total_risk,
        "daily_loss": risk_manager.daily_pnl,
        "open_positions_count": len(positions),
        "total_exposure": total_exposure,
        "risk_limit_reached": risk_manager.get_status().get("is_trading_paused", False),
        # --- P&L ---
        "unrealized_pnl_usd": round(unrealized_pnl_usd, 2),
        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
        "realized_pnl_today": round(realized_pnl_today, 2),
        "realized_pnl_alltime": round(realized_pnl_alltime, 2),
    }


@app.get("/api/positions")
async def get_positions():
    """Get all currently open positions."""
    return position_manager.get_open_positions()


@app.get("/api/signals")
async def get_signals():
    """Get the latest ensemble scan results for all coins."""
    return ensemble.latest_scan


@app.get("/api/model-performance")
async def get_model_performance(days: int = 30):
    """
    Get per-model accuracy stats based on DB prediction outcomes.
    Accuracy = fraction of BUY predictions on traded coins that resulted in a WIN.
    Stats are meaningful once trades have closed and outcomes are backfilled.
    """
    try:
        stats = db.get_model_accuracy_stats(days=days)
        return {
            "window_days": days,
            "models": {
                model: {
                    "total_predictions": v["total_predictions"],
                    "buy_signals": v["buy_signals"],
                    "wins": v["wins"],
                    "accuracy_pct": round(v["accuracy"] * 100, 1) if v["accuracy"] is not None else None,
                    "avg_pnl_pct": round(v["avg_pnl"], 2) if v["avg_pnl"] is not None else None,
                    "label": {"ta": "Technical Analysis", "ml": "ML Ensemble", "tcn": "TCN Deep Learning"}.get(model, model)
                }
                for model, v in stats.items()
            }
        }
    except Exception as e:
        logger.error(f"Error fetching model performance: {e}")
        return {"window_days": days, "models": {}}


@app.get("/api/balance-history")
async def get_balance_history(limit: int = 100):
    """Get paper balance audit trail from BotState ledger."""
    try:
        from src.data.database import BotState
        session = db.SessionLocal()
        rows = (
            session.query(BotState)
            .order_by(BotState.recorded_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "balance": r.paper_balance,
                "trades_count": r.trades_count,
                "recorded_at": r.recorded_at.isoformat(),
                "note": r.note
            }
            for r in reversed(rows)  # chronological order
        ]
    except Exception as e:
        logger.error(f"Error fetching balance history: {e}")
        return []
    finally:
        session.close()


@app.get("/api/history")
async def get_history(limit: int = 50):
    """Get recent closed trades history."""
    try:
        session = db.SessionLocal()
        closed_trades = (
            session.query(Trade)
            .filter(Trade.status == TradeStatus.CLOSED.value)
            .order_by(Trade.exit_time.desc())
            .limit(limit)
            .all()
        )
        
        history = []
        for t in closed_trades:
            # Compute trade duration
            duration_str = ""
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                mins = remainder // 60
                duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
            
            history.append({
                "trade_id": t.trade_id,
                "coin": t.coin,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "duration": duration_str,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "position_size": t.position_size,
                "pnl_percent": t.pnl_percent,
                "pnl_usd": t.pnl_usd,
                "exit_reason": t.exit_reason,
                "tier": t.consensus_tier,
                "ta_confidence": t.ta_confidence,
                "ml_confidence": t.ml_confidence,
                "tcn_confidence": t.tcn_confidence,
                "entry_reason": t.entry_reason,
            })
        return history
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
        return []
    finally:
        session.close()


class ResetRequest(BaseModel):
    new_balance: float


@app.post("/api/reset")
async def reset_bot_state(req: ResetRequest):
    """
    Control Panel: Hard-Reset the Bot's Paper State
    Cancels all open trades and injects a fresh paper balance.
    Closed trade history is preserved.
    """
    if executor.mode != ExecutionMode.PAPER:
        raise HTTPException(status_code=400, detail="Cannot reset state while in LIVE mode.")
    
    if req.new_balance < 100:
        raise HTTPException(status_code=400, detail="Paper balance must be at least $100")
        
    logger.info(f"🚨 CONTROL PANEL: Hard Reset initiated → new balance: ${req.new_balance:.2f}")

    try:
        session = db.SessionLocal()
        
        # 1. Cancel all currently OPEN trades in DB (preserves closed history)
        open_ids = [tid for tid in position_manager.positions.keys()]
        if open_ids:
            session.query(Trade).filter(
                Trade.trade_id.in_(open_ids)
            ).update({"status": "CANCELLED"}, synchronize_session=False)
            session.commit()
        
        # 2. Clear in-memory positions and stale log throttle timestamps
        position_manager.positions.clear()
        # Remove any _last_logged_* dynamic attributes set by check_position()
        stale_attrs = [attr for attr in vars(position_manager) if attr.startswith("_last_logged_")]
        for attr in stale_attrs:
            delattr(position_manager, attr)
        
        # 3. Reset executor balances
        executor.paper_balance = req.new_balance
        executor.paper_trades = {}
        
        # 4. Reset risk manager — capital, risk tracking, and daily counters
        risk_manager.set_capital(req.new_balance)
        risk_manager._total_risk = 0.0
        risk_manager.daily_pnl = 0.0
        risk_manager.daily_start_capital = req.new_balance  # Fix % calculations
        
        # 5. Save new state to disk
        save_state(req.new_balance, 0)
        
        logger.info(f"✅ Reset complete. Paper balance: ${req.new_balance:.2f}, {len(open_ids)} open positions cancelled.")
        return {
            "success": True,
            "message": f"Reset complete. New paper balance: ${req.new_balance:.2f}",
            "cancelled_positions": len(open_ids)
        }
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to reset bot state: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

