"""
Main Trading Bot

Orchestrates all components:
- Data collection
- Model inference
- Signal generation
- Trade execution
- Position monitoring
"""

import time
import schedule
from datetime import datetime
from typing import Dict, Optional

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings, WATCHLIST
from src.utils import logger, log_risk_event
from src.data import db, collector
from src.models import ensemble
from src.trading import executor, position_manager, risk_manager
from src.utils.notifiers import discord_notifier


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Runs continuously:
    - Fetches data every 1 minute
    - Generates signals every 5 minutes
    - Monitors positions every 30 seconds
    - Executes trades when signals fire
    """
    
    def __init__(self):
        """Initialize the trading bot."""
        self.running = False
        self.last_scan = None
        self.last_data_update = None
        self.stats = {
            "signals_generated": 0,
            "trades_executed": 0,
            "trades_rejected": 0,
            "data_updates": 0,
            "position_checks": 0,
            "start_time": None
        }
        
        # Initialize database
        self._setup()
    
    def _setup(self) -> None:
        """Setup bot components."""
        logger.info("Initializing Trading Bot...")
        
        # Create database tables
        db.create_tables()
        logger.info("Database tables created")
        
        # Check API connection
        if collector.check_api_connection():
            logger.info("Binance API connected")
        else:
            logger.error("Binance API connection failed")
        
        # Sanity check: warn if paper balance is dangerously low
        if executor.mode.value == "paper" and executor.paper_balance < 50:
            logger.warning(
                f"⚠️  Paper balance is only ${executor.paper_balance:.2f}! "
                "Bot may not be able to open trades (min $6 per trade). "
                "Reset your state: delete data/bot_state.json to restore $10,000."
            )
    
    def update_data(self) -> None:
        """Update market data for all coins + global BTC bias."""
        try:
            logger.debug("Updating market data...")
            # Update all timeframes that models use
            collector.update_all_coins(timeframe="1d", limit=60)    # Daily trend
            collector.update_all_coins(timeframe="4h", limit=50)    # For MTF confirmation
            collector.update_all_coins(timeframe="1h", limit=200)   # Primary timeframe for all models
            collector.update_all_coins(timeframe="15m", limit=100)  # For MTF momentum
            
            # Explicitly fetch BTC 4h bias even if BTC isn't in WATCHLIST
            btc_klines = collector.fetch_klines("BTCUSDT", "4h", 50, use_public=False)
            if btc_klines:
                collector.data_cache["BTCUSDT_4h"] = pd.DataFrame(btc_klines)
                
            self.stats["data_updates"] += 1
            self.last_data_update = datetime.utcnow()
        except Exception as e:
            logger.error(f"Data update error: {e}")
    
    def scan_for_signals(self) -> None:
        """Scan all coins for trading signals."""
        try:
            logger.info("Scanning for signals...")
            
            # Get buy signals from ensemble
            signals = ensemble.scan_for_signals(WATCHLIST)
            
            self.stats["signals_generated"] += len(signals)
            self.last_scan = datetime.utcnow()
            
            if signals:
                logger.info(f"Found {len(signals)} potential signals")
                
                for symbol, signal in signals:
                    # Try to execute each signal
                    if signal.tier > 0:
                        result = executor.execute_signal(symbol, signal)
                        
                        if result and result.success:
                            self.stats["trades_executed"] += 1
                            logger.info(f"✅ Trade executed: {symbol} @ Tier {signal.tier}")
                        elif result and not result.success:
                            self.stats["trades_rejected"] += 1
                            logger.info(f"❌ Trade rejected: {symbol} - {result.message}")
            else:
                logger.debug("No signals found")
                
        except Exception as e:
            logger.error(f"Signal scan error: {e}")
    
    def monitor_positions(self) -> None:
        """Monitor all open positions."""
        try:
            result = executor.monitor_positions()
            self.stats["position_checks"] += 1
            
            if result["exits"]:
                logger.info(f"Closed {len(result['exits'])} positions")
                
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")

    def send_daily_report(self) -> None:
        """Fetch daily stats and send to Discord."""
        try:
            stats = db.get_daily_stats()
            discord_notifier.send_daily_status(stats)
            logger.info("Daily performance report sent to Discord")
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
    
    def run_once(self) -> Dict:
        """
        Run a single iteration of the bot.
        
        Returns:
            Dictionary with iteration results
        """
        results = {
            "data_updated": False,
            "signals_checked": False,
            "positions_monitored": False,
            "trades": []
        }
        
        # Update data
        self.update_data()
        results["data_updated"] = True
        
        # Scan for signals
        scan_result = executor.scan_and_execute(WATCHLIST)
        results["signals_checked"] = True
        results["trades"] = scan_result.get("trades", [])
        
        # Monitor positions
        self.monitor_positions()
        results["positions_monitored"] = True
        
        return results
    
    def run(self) -> None:
        """
        Run the bot continuously with scheduling.
        """
        self.running = True
        self.stats["start_time"] = datetime.utcnow()
        
        logger.info("=" * 50)
        logger.info("🚀 Trading Bot Started")
        logger.info(f"Mode: {executor.mode.value.upper()}")
        logger.info(f"Monitoring: {len(WATCHLIST)} coins")
        logger.info("=" * 50)
        
        # Schedule tasks
        schedule.every(1).minutes.do(self.update_data)
        schedule.every(5).minutes.do(self.scan_for_signals)
        schedule.every(30).seconds.do(self.monitor_positions)
        
        # Daily reset and report at midnight
        schedule.every().day.at("00:00").do(risk_manager.reset_daily)
        schedule.every().day.at("00:01").do(self.send_daily_report)
        
        # Initial run
        self.update_data()
        self.scan_for_signals()
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.stop()
    
    def stop(self) -> None:
        """Stop the bot."""
        self.running = False
        logger.info("Trading Bot Stopped")
        
        # Log final stats
        if self.stats["start_time"]:
            runtime = datetime.utcnow() - self.stats["start_time"]
            logger.info(f"Runtime: {runtime}")
            logger.info(f"Data updates: {self.stats['data_updates']}")
            logger.info(f"Signals generated: {self.stats['signals_generated']}")
            logger.info(f"Trades executed: {self.stats['trades_executed']}")
            logger.info(f"Trades rejected: {self.stats['trades_rejected']}")
    
    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            "running": self.running,
            "stats": self.stats,
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "last_data_update": self.last_data_update.isoformat() if self.last_data_update else None,
            "executor": executor.get_status(),
            "risk": risk_manager.get_status(),
            "positions": position_manager.get_open_positions()
        }


# Global bot instance
bot = TradingBot()


def main():
    """Entry point for the trading bot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan for signals and print results"
    )
    
    args = parser.parse_args()
    
    # Set execution mode
    if args.mode == "live":
        from src.trading import ExecutionMode
        executor.mode = ExecutionMode.LIVE
        logger.warning("⚠️ LIVE TRADING MODE ENABLED")
    
    if args.scan:
        # Just scan and print signals
        print("\n" + "=" * 60)
        print("🔍 Scanning for trading signals...")
        print("=" * 60)
        
        collector.update_all_coins()
        signals = ensemble.scan_for_signals()
        
        if signals:
            print(f"\n✅ Found {len(signals)} potential trades:\n")
            for symbol, s in signals:
                print(f"  • {symbol} - Tier {s.tier}: confidence {s.confidence:.1%}")
                print(f"    Reasons: {', '.join(s.reasons[:2])}")
                print()
        else:
            print("\n❌ No signals found matching criteria")
        
    elif args.once:
        # Run once
        result = bot.run_once()
        print(f"\nResults: {result}")
    else:
        # Run continuously
        bot.run()


if __name__ == "__main__":
    main()
