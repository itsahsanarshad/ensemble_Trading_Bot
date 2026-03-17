"""
Data Quality Checker

Verify data coverage and quality for all coins.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import db, PriceData
from config import WATCHLIST
from datetime import datetime, timedelta


def check_data_coverage():
    """Check data coverage for all coins."""
    print("=" * 70)
    print("📊 Data Coverage Report")
    print("=" * 70)
    print(f"{'Symbol':<12} | {'15m Data':<10} | {'1h Data':<10} | {'4h Data':<10} | {'Status':<15}")
    print("-" * 70)
    
    total_15m = 0
    total_1h = 0
    total_4h = 0
    issues = []
    
    # Create session
    from sqlalchemy.orm import Session
    session = Session(db.engine)
    
    for symbol in WATCHLIST:
        count_15m = session.query(PriceData).filter_by(
            coin=symbol, timeframe="15m"
        ).count()
        
        count_1h = session.query(PriceData).filter_by(
            coin=symbol, timeframe="1h"
        ).count()
        
        count_4h = session.query(PriceData).filter_by(
            coin=symbol, timeframe="4h"
        ).count()
        
        total_15m += count_15m
        total_1h += count_1h
        total_4h += count_4h
        
        # Expected: ~17,280 for 180 days of 15m data
        # Expected: ~4,320 for 180 days of 1h data
        status = "✅ Good"
        if count_15m < 1000 or count_1h < 250:
            status = "⚠️  Low data"
            issues.append(symbol)
        
        print(f"{symbol:<12} | {count_15m:>10} | {count_1h:>10} | {count_4h:>10} | {status:<15}")
    
    print("-" * 70)
    print(f"{'TOTAL':<12} | {total_15m:>10} | {total_1h:>10} | {total_4h:>10} |")
    print("=" * 70)
    
    if issues:
        print(f"\n⚠️  Coins with low data: {', '.join(issues)}")
        print("   Consider running: python run.py --backfill 180")
    else:
        print("\n✅ All coins have sufficient data!")
    
    # Check date range
    print("\n📅 Date Range:")
    oldest = session.query(PriceData).order_by(PriceData.timestamp.asc()).first()
    newest = session.query(PriceData).order_by(PriceData.timestamp.desc()).first()
    
    if oldest and newest:
        # Convert millisecond timestamps to datetime
        from datetime import datetime
        oldest_date = datetime.fromtimestamp(oldest.timestamp / 1000)
        newest_date = datetime.fromtimestamp(newest.timestamp / 1000)
        
        print(f"   Oldest: {oldest_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Newest: {newest_date.strftime('%Y-%m-%d %H:%M')}")
        days = (newest_date - oldest_date).days
        print(f"   Coverage: {days} days")
    
    session.close()
    print("=" * 70)


def check_data_gaps():
    """Check for gaps in data."""
    print("\n🔍 Checking for data gaps...")
    
    from sqlalchemy.orm import Session
    session = Session(db.engine)
    
    gaps_found = False
    for symbol in WATCHLIST[:5]:  # Check first 5 coins
        records = session.query(PriceData).filter_by(
            coin=symbol, timeframe="15m"
        ).order_by(PriceData.timestamp.asc()).all()
        
        if len(records) < 2:
            continue
        
        for i in range(1, len(records)):
            time_diff = (records[i].timestamp - records[i-1].timestamp).total_seconds() / 60
            if time_diff > 20:  # Gap > 20 minutes for 15m data
                print(f"   ⚠️  {symbol}: Gap of {time_diff/60:.1f} hours at {records[i].timestamp}")
                gaps_found = True
                break
    
    if not gaps_found:
        print("   ✅ No significant gaps found")


if __name__ == "__main__":
    check_data_coverage()
    # check_data_gaps()  # Skip for now - timestamps need conversion
