"""
State Management

Persist bot state across restarts.
Primary store: BotState DB table (auditable ledger).
Backup store: bot_state.json (crash-safe fallback).
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

STATE_FILE = Path(__file__).parent.parent.parent / "data" / "bot_state.json"


def save_state(paper_balance: float, trades_count: int = 0, metadata: Dict = None, note: str = "trade") -> None:
    """
    Save bot state to DB ledger and JSON backup.

    Args:
        paper_balance: Current paper trading balance
        trades_count: Number of trades executed
        metadata: Additional metadata to save
        note: Reason for this state update ('startup', 'reset', 'trade', 'daily')
    """
    # 1. Write to DB (primary — creates an audit trail row)
    try:
        from src.data.database import db
        db.save_bot_state(paper_balance, trades_count, note=note)
    except Exception as e:
        print(f"Warning: Could not save state to DB: {e}")

    # 2. Write JSON backup (crash-safe secondary store)
    STATE_FILE.parent.mkdir(exist_ok=True)
    state = {
        "paper_balance": paper_balance,
        "trades_count": trades_count,
        "last_updated": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write JSON backup: {e}")


def load_state() -> Dict:
    """
    Load bot state — tries DB first, falls back to JSON.

    Returns:
        Dictionary with state data
    """
    # 1. Try DB (authoritative)
    try:
        from src.data.database import db
        row = db.get_latest_bot_state()
        if row:
            balance = row.get("paper_balance", 10000.0)
            if not isinstance(balance, (int, float)) or balance <= 0:
                print(f"Warning: Invalid DB balance '{balance}', resetting to $10,000")
                row["paper_balance"] = 10000.0
            return row
    except Exception:
        pass  # DB not ready yet on first run — fall through to JSON

    # 2. Fallback: JSON backup
    if not STATE_FILE.exists():
        return {
            "paper_balance": 10000.0,
            "trades_count": 0,
            "last_updated": None,
            "metadata": {}
        }

    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)

        balance = data.get("paper_balance", 10000.0)
        if not isinstance(balance, (int, float)) or balance <= 0:
            print(f"Warning: Invalid paper_balance '{balance}' in JSON, resetting to $10,000")
            data["paper_balance"] = 10000.0

        # Warn if state is very stale (>24 hours)
        last_updated = data.get("last_updated")
        if last_updated:
            try:
                age = datetime.utcnow() - datetime.fromisoformat(last_updated)
                if age.total_seconds() > 86400:
                    print(f"Warning: State file is {age.days}d {age.seconds//3600}h old — consider resetting if balance looks wrong")
            except Exception:
                pass

        return data
    except Exception as e:
        print(f"Error loading state: {e}")
        return {
            "paper_balance": 10000.0,
            "trades_count": 0,
            "last_updated": None,
            "metadata": {}
        }


def reset_state() -> None:
    """Reset state to initial values."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()



