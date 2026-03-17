"""
Telegram Notifications

Send trading alerts and updates via Telegram.
"""

import asyncio
from typing import Optional
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings
from src.utils import logger

# Try to import telegram
try:
    from telegram import Bot
    from telegram.error import TelegramError
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


class TelegramNotifier:
    """
    Send trading notifications via Telegram.
    
    Messages include:
    - Trade entries with confidence levels
    - Trade exits with P&L
    - Daily summaries
    - Risk alerts
    """
    
    def __init__(self):
        """Initialize Telegram notifier."""
        self.enabled = (
            HAS_TELEGRAM and
            settings.notifications.telegram_bot_token and
            settings.notifications.telegram_chat_id
        )
        
        if self.enabled:
            self.bot = Bot(token=settings.notifications.telegram_bot_token)
            self.chat_id = settings.notifications.telegram_chat_id
            logger.info("Telegram notifications enabled")
        else:
            self.bot = None
            self.chat_id = None
            logger.info("Telegram notifications disabled")
    
    async def _send_message(self, message: str) -> bool:
        """Send a message via Telegram."""
        if not self.enabled:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="HTML"
            )
            return True
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def send(self, message: str) -> bool:
        """Send message synchronously."""
        if not self.enabled:
            return False
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._send_message(message))
    
    def notify_trade_entry(
        self,
        coin: str,
        entry_price: float,
        position_size: float,
        tier: int,
        confidence: float,
        stop_loss: float,
        take_profit: float,
        reasons: list
    ) -> bool:
        """
        Send trade entry notification.
        
        Args:
            coin: Trading pair
            entry_price: Entry price
            position_size: Position size in USDT
            tier: Consensus tier
            confidence: Overall confidence
            stop_loss: Stop loss price
            take_profit: Take profit price
            reasons: Entry reasons
        """
        tier_emoji = {1: "🔴", 2: "🟡", 3: "🟢", 4: "🔵"}
        
        message = f"""
{tier_emoji.get(tier, "⚪")} <b>PUMP SIGNAL: {coin}</b>

📊 <b>Entry:</b> ${entry_price:.4f}
🎯 <b>Targets:</b> ${take_profit:.4f} (+{((take_profit/entry_price)-1)*100:.1f}%)
🛑 <b>Stop-Loss:</b> ${stop_loss:.4f} ({((stop_loss/entry_price)-1)*100:.1f}%)

💰 <b>Size:</b> ${position_size:.2f}
💪 <b>Confidence:</b> {confidence:.0%}
📈 <b>Tier:</b> {tier}

📝 <b>Reasons:</b>
{chr(10).join('• ' + r for r in reasons[:3])}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
        return self.send(message)
    
    def notify_trade_exit(
        self,
        coin: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        pnl_usd: float,
        exit_reason: str,
        duration: str
    ) -> bool:
        """Send trade exit notification."""
        emoji = "✅" if pnl_pct > 0 else "❌"
        pnl_emoji = "📈" if pnl_pct > 0 else "📉"
        
        message = f"""
{emoji} <b>TRADE CLOSED: {coin}</b>

📊 <b>Entry:</b> ${entry_price:.4f}
📊 <b>Exit:</b> ${exit_price:.4f}

{pnl_emoji} <b>P&L:</b> {'+' if pnl_pct > 0 else ''}{pnl_pct*100:.2f}% (${pnl_usd:+.2f})
🏷️ <b>Reason:</b> {exit_reason}
⏱️ <b>Duration:</b> {duration}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
        return self.send(message)
    
    def notify_daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        daily_pnl: float,
        win_rate: float,
        open_positions: int
    ) -> bool:
        """Send daily summary notification."""
        emoji = "🎉" if daily_pnl > 0 else "😔"
        
        message = f"""
{emoji} <b>DAILY SUMMARY</b>

📊 <b>Trades:</b> {total_trades}
✅ <b>Wins:</b> {wins}
❌ <b>Losses:</b> {losses}
📈 <b>Win Rate:</b> {win_rate:.1%}

💰 <b>Daily P&L:</b> {'+' if daily_pnl > 0 else ''}${daily_pnl:.2f}
📦 <b>Open Positions:</b> {open_positions}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
        return self.send(message)
    
    def notify_risk_alert(self, event: str, details: str) -> bool:
        """Send risk management alert."""
        message = f"""
⚠️ <b>RISK ALERT</b>

🚨 <b>Event:</b> {event}
📝 <b>Details:</b> {details}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
        return self.send(message)
    
    def notify_system_status(self, status: str, message_text: str) -> bool:
        """Send system status notification."""
        emoji = "🟢" if status == "online" else "🔴"
        
        message = f"""
{emoji} <b>SYSTEM {status.upper()}</b>

{message_text}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
        return self.send(message)


# Global notifier instance
notifier = TelegramNotifier()
