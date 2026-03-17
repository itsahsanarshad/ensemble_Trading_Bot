"""
Daily Stats Report - CLI tool for VPS

View trading bot performance without a dashboard.
Run: python scripts/daily_stats.py
"""

import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_daily_stats():
    """Generate daily stats report."""
    from src.data.database import db, Trade
    from sqlalchemy.orm import Session
    
    print("=" * 60)
    print("📊 TRADING BOT DAILY STATS")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load paper balance
    state_file = Path(__file__).parent.parent / "data" / "bot_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        balance = state.get("paper_balance", 0)
        print(f"\n💰 Paper Balance: ${balance:.2f}")
    
    # Get trades from database
    session = Session(db.engine)
    
    # Open trades
    open_trades = session.query(Trade).filter_by(status="OPEN").all()
    print(f"\n📈 Open Positions: {len(open_trades)}")
    
    if open_trades:
        print("-" * 40)
        for trade in open_trades:
            print(f"  • {trade.coin} | Entry: ${trade.entry_price:.4f} | Size: ${trade.position_size:.2f}")
    
    # Today's trades
    today = datetime.now().date()
    today_start = datetime.combine(today, datetime.min.time())
    
    closed_today = session.query(Trade).filter(
        Trade.status == "CLOSED",
        Trade.exit_time >= today_start
    ).all()
    
    print(f"\n📋 Today's Closed Trades: {len(closed_today)}")
    
    if closed_today:
        print("-" * 40)
        total_pnl = 0
        wins = 0
        for trade in closed_today:
            pnl = trade.pnl_percent or 0
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            emoji = "✅" if pnl > 0 else "❌"
            print(f"  {emoji} {trade.coin} | PnL: {pnl:+.2f}% | {trade.exit_reason}")
        
        win_rate = wins / len(closed_today) * 100 if closed_today else 0
        print("-" * 40)
        print(f"  Today's Win Rate: {win_rate:.0f}%")
        print(f"  Today's Total PnL: {total_pnl:+.2f}%")
    
    # All-time stats
    all_closed = session.query(Trade).filter_by(status="CLOSED").all()
    
    if all_closed:
        print(f"\n📊 All-Time Stats ({len(all_closed)} trades)")
        print("-" * 40)
        
        total_pnl = sum(t.pnl_percent or 0 for t in all_closed)
        wins = sum(1 for t in all_closed if (t.pnl_percent or 0) > 0)
        win_rate = wins / len(all_closed) * 100
        
        print(f"  Total Trades: {len(all_closed)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total PnL: {total_pnl:+.2f}%")
    else:
        print("\n📊 No closed trades yet")
    
    session.close()
    
    # Recent log entries
    log_file = Path(__file__).parent.parent / "logs" / f"bot_{datetime.now().strftime('%Y-%m-%d')}.log"
    if log_file.exists():
        print(f"\n📝 Recent Signals/Trades:")
        print("-" * 40)
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-20:]
            count = 0
            for line in lines:
                if ("SIGNAL" in line or "TRADE" in line or "BUY" in line or "SELL" in line) and count < 5:
                    # Extract just the important part
                    if "|" in line:
                        parts = line.split("|")
                        if len(parts) >= 2:
                            print(f"  {parts[-1].strip()[:70]}")
                            count += 1
    
    print("\n" + "=" * 60)
    print("💡 VPS Commands:")
    print("  python scripts/daily_stats.py  # This report")
    print("  python run.py --scan           # Check signals now")
    print("  python run.py                  # Start bot")
    print("  tail -f logs/bot_*.log         # Watch live logs")
    print("=" * 60)


if __name__ == "__main__":
    get_daily_stats()
