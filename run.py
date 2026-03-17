"""
Trading Bot Entry Point

Main entry point for running the crypto trading bot.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def setup():
    """Initial setup for the bot."""
    print("=" * 60)
    print("🚀 Crypto Trading Bot - Initial Setup")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    print("✅ Data directory created")
    
    # Create logs directory
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    print("✅ Logs directory created")
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    print("✅ Models directory created")
    
    # Create database tables
    from src.data import db
    db.create_tables()
    print("✅ Database tables created")
    
    # Check for .env file
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("\n⚠️  No .env file found!")
        print("   Copy .env.example to .env and add your Binance API keys")
        print("   Example: cp .env.example .env")
    else:
        print("✅ .env file found")
    
    print("\n" + "=" * 60)
    print("Setup Complete! Next steps:")
    print("=" * 60)
    print("1. Configure your .env file with Binance API keys")
    print("2. Train the ML models:")
    print("   python training/train_xgboost.py")
    print("   python training/train_lstm.py")
    print("3. Run backtest to validate:")
    print("   python training/backtest.py")
    print("4. Start the bot in paper trading mode:")
    print("   python run.py")
    print("5. Start the dashboard:")
    print("   python dashboard/app.py")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                 # Run bot in paper trading mode
  python run.py --scan          # Scan for signals only
  python run.py --once          # Run once and exit
  python run.py --mode live     # Run in live trading mode (CAREFUL!)
  python run.py --setup         # Initial setup
  python run.py --dashboard     # Start web dashboard
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan for signals and print results"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run initial setup"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the web dashboard server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the dashboard (default: 8000)"
    )
    parser.add_argument(
        "--backfill",
        type=int,
        metavar="DAYS",
        help="Backfill historical data for N days"
    )
    
    args = parser.parse_args()
    
    if args.setup:
        setup()
        return
        
    if args.dashboard:
        import threading
        import uvicorn
        
        def run_dashboard():
            uvicorn.run("src.dashboard.app:app", host="0.0.0.0", port=args.port, access_log=False, log_level="warning")
            
        print("=" * 60)
        print(f"🌐 Starting Dashboard in background on http://localhost:{args.port}")
        print("=" * 60)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        # We do NOT return here, so that the bot's execution continues on the main thread!
    
    if args.backfill:
        from config import WATCHLIST
        from src.data import collector
        
        print(f"Backfilling {args.backfill} days of data (15m, 1h, 4h)...")
        for symbol in WATCHLIST:
            collector.backfill_historical_data(symbol, "15m", args.backfill)
            collector.backfill_historical_data(symbol, "1h", args.backfill)
            collector.backfill_historical_data(symbol, "4h", args.backfill)
        print("Backfill complete!")
        return
    
    # Import bot only when needed
    from src.bot import bot, main as bot_main
    
    # Pass to bot main
    sys.argv = [sys.argv[0]]
    if args.mode == "live":
        sys.argv.extend(["--mode", "live"])
    if args.scan:
        sys.argv.append("--scan")
    if args.once:
        sys.argv.append("--once")
    
    bot_main()


if __name__ == "__main__":
    main()
