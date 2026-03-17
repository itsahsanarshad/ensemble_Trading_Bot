"""
Notification Utilities

Supports Discord Webhooks for real-time trade signals and daily status updates.
"""

import requests
from datetime import datetime
from typing import Dict, Any, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from src.utils.logger import logger


class DiscordNotifier:
    """Send notifications to Discord via Webhooks."""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or settings.notifications.discord_webhook_url
        self.enabled = bool(self.webhook_url)
        
    def _send_embed(self, embed: Dict[str, Any]) -> bool:
        """Send a rich embed to Discord."""
        if not self.enabled:
            return False
            
        try:
            payload = {
                "embeds": [embed],
                "username": "Alpha-Bot Trading",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/2091/2091665.png"
            }
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def send_buy_alert(
        self, 
        symbol: str, 
        price: float, 
        size: float, 
        sl: float, 
        tp: float,   # TP2 (+7%)
        tier: int,
        confidence: float
    ):
        """Send a buy signal alert with TP1, TP2, SL, and risk/reward."""
        # TP1 is always entry * 1.035 (+3.5%)
        tp1 = price * 1.035
        
        # Risk/Reward: from entry to TP2 vs entry to SL
        risk = price - sl
        reward = tp - price
        rr_ratio = reward / risk if risk > 0 else 0
        
        tier_labels = {1: "⚡ High Confidence", 2: "💪 Strong Consensus", 3: "🏆 Full Consensus", 4: "🎯 Breakout Override"}
        tier_color = {1: 3066993, 2: 15844367, 3: 10181046, 4: 16711680}
        
        embed = {
            "title": f"🚀 BUYING  {symbol}",
            "description": f"**{tier_labels.get(tier, f'Tier {tier}')}** signal fired at **{confidence:.1%}** overall confidence.",
            "color": tier_color.get(tier, 3066993),
            "fields": [
                {"name": "Entry Price", "value": f"${price:,.4f}", "inline": True},
                {"name": "Position Size", "value": f"${size:,.2f}", "inline": True},
                {"name": "Tier", "value": f"Tier {tier}", "inline": True},
                {"name": "🎯 TP1 (+3.5%)", "value": f"${tp1:,.4f}", "inline": True},
                {"name": "🎯 TP2 (+7.0%)", "value": f"${tp:,.4f}", "inline": True},
                {"name": "🛡️ Stop Loss (-3%)", "value": f"${sl:,.4f}", "inline": True},
                {"name": "Risk/Reward", "value": f"1 : {rr_ratio:.2f}", "inline": True},
                {"name": "Strategy", "value": "50% exit at TP1 → SL→Breakeven → run to TP2", "inline": False},
            ],
            "footer": {"text": f"Alpha-Bot • {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"},
            "thumbnail": {"url": "https://cdn-icons-png.flaticon.com/512/3443/3443338.png"}
        }
        return self._send_embed(embed)

    def send_sell_alert(
        self, 
        symbol: str, 
        price: float, 
        pnl_pct: float, 
        pnl_usd: float, 
        reason: str,
        entry_price: float = 0.0,
        duration_str: str = ""
    ):
        """Send a sell/exit alert with full trade summary."""
        color = 15158332 if pnl_pct < 0 else 3066993  # Red if loss, Green if profit
        emoji = "🛑" if pnl_pct < 0 else "💰"
        result_label = "LOSS" if pnl_pct < 0 else "WIN"
        
        fields = [
            {"name": f"{emoji} Result", "value": f"**{result_label}**", "inline": True},
            {"name": "Exit Price", "value": f"${price:,.4f}", "inline": True},
            {"name": "P&L %", "value": f"{pnl_pct:+.2f}%", "inline": True},
            {"name": "P&L USD", "value": f"${pnl_usd:+.2f}", "inline": True},
            {"name": "Exit Reason", "value": _format_exit_reason(reason), "inline": True},
        ]
        
        if entry_price > 0:
            fields.append({"name": "Entry Price", "value": f"${entry_price:,.4f}", "inline": True})
        if duration_str:
            fields.append({"name": "⏱️ Duration", "value": duration_str, "inline": True})
        
        embed = {
            "title": f"{emoji} CLOSED: {symbol}",
            "color": color,
            "fields": fields,
            "footer": {"text": f"Alpha-Bot • {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"}
        }
        return self._send_embed(embed)

    def send_partial_exit_alert(self, symbol: str, price: float, pnl_pct: float, pnl_usd: float, remaining_size: float):
        """Send a TP1 partial exit alert — 50% sold, SL moved to breakeven."""
        embed = {
            "title": f"🎯 TP1 HIT — {symbol} (50% closed)",
            "description": "Sold **50%** of position at **TP1**. Stop loss moved to **breakeven**. Letting remaining 50% run to TP2.",
            "color": 16776960,  # Yellow
            "fields": [
                {"name": "Exit Price (TP1)", "value": f"${price:,.4f}", "inline": True},
                {"name": "Partial P&L", "value": f"{pnl_pct:+.2f}% (${pnl_usd:+.2f})", "inline": True},
                {"name": "Remaining Position", "value": f"${remaining_size:,.2f}", "inline": True},
                {"name": "SL Updated", "value": "→ Breakeven (Entry Price)", "inline": True},
                {"name": "Next Target", "value": "TP2 (+7.0%)", "inline": True},
            ],
            "footer": {"text": f"Alpha-Bot • {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"}
        }
        return self._send_embed(embed)

    def send_daily_status(self, stats: Dict[str, Any]):
        """Send comprehensive daily performance summary."""
        pnl = stats.get('pnl', 0)
        win_rate = stats.get('win_rate', 0)
        total = stats.get('total', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        best = stats.get('best_trade', 0)
        worst = stats.get('worst_trade', 0)
        capital = stats.get('capital', 0)
        
        color = 3066993 if pnl >= 0 else 15158332  # Green or Red
        performance_emoji = "📈" if pnl >= 0 else "📉"
        
        embed = {
            "title": f"{performance_emoji} Daily Performance Report — {datetime.now().strftime('%Y-%m-%d')}",
            "color": color,
            "fields": [
                {"name": "💰 Daily P&L", "value": f"${pnl:+.2f}", "inline": True},
                {"name": "📊 Win Rate", "value": f"{win_rate:.1%}", "inline": True},
                {"name": "🔢 Total Trades", "value": str(total), "inline": True},
                {"name": "✅ Wins", "value": str(wins), "inline": True},
                {"name": "❌ Losses", "value": str(losses), "inline": True},
                {"name": "💼 Capital", "value": f"${capital:,.2f}", "inline": True},
                {"name": "🏆 Best Trade", "value": f"${best:+.2f}", "inline": True},
                {"name": "💀 Worst Trade", "value": f"${worst:+.2f}", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        return self._send_embed(embed)


def _format_exit_reason(reason: str) -> str:
    """Human-readable exit reason label."""
    return {
        "TAKE_PROFIT": "✅ Take Profit (TP2)",
        "STOP_LOSS": "🛑 Stop Loss Hit",
        "TRAILING_STOP": "📉 Trailing Stop",
        "TIME_STOP": "⏱️ Time Stop (24h)",
        "PARTIAL_EXIT": "🎯 TP1 Partial Exit",
        "SIGNAL_EXIT": "📡 Signal Reversal",
        "MANUAL": "🖐 Manual Close",
    }.get(str(reason), str(reason))


# Global instance
discord_notifier = DiscordNotifier()


if __name__ == "__main__":
    # Test script
    test_url = settings.notifications.discord_webhook_url
    if test_url:
        notifier = DiscordNotifier(test_url)
        print("Sending test BUY notification...")
        notifier.send_buy_alert("BTCUSDT", 50000.0, 100.0, 48500.0, 53500.0, 3, 0.85)
        print("Sending test SELL notification...")
        notifier.send_sell_alert("BTCUSDT", 53000.0, 6.0, 60.0, "TAKE_PROFIT", entry_price=50000.0, duration_str="4h 22m")
        print("Sending test TP1 partial exit...")
        notifier.send_partial_exit_alert("BTCUSDT", 51750.0, 3.5, 35.0, 50.0)
    else:
        print("No DISCORD_WEBHOOK_URL found in .env")
