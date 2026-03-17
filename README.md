# Crypto Trading Bot

Production-ready cryptocurrency trading bot using a three-layer hybrid approach:
- **Traditional Technical Analysis (TA)** - Fast pattern recognition
- **XGBoost Machine Learning** - Multi-feature pattern detection  
- **LSTM Deep Learning** - Sequential momentum prediction

## Features

- 🎯 **Flexible Consensus System** - 4-tier approach adapts to different market situations
- 📊 **80+ Technical Features** - Comprehensive feature engineering for ML models
- 🛡️ **Risk Management** - Position limits, stop losses, daily loss limits
- 📝 **Paper Trading** - Simulate trades before going live
- 📈 **Web Dashboard** - Real-time monitoring UI
- 🔄 **Automated Exits** - Stop loss, take profit, trailing stops, time stops

## Quick Start

### 1. Install Dependencies

```bash
cd trading-bot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your Binance API keys
```

### 3. Initial Setup

```bash
python run.py --setup
```

### 4. Backfill Historical Data

```bash
python run.py --backfill 30  # Backfill 30 days
```

### 5. Train Models

```bash
# Train XGBoost (faster)
python training/train_xgboost.py --days 180

# Train LSTM (requires GPU, slower)
python training/train_lstm.py --days 180
```

### 6. Run Backtest

```bash
python training/backtest.py --start 2024-01-01 --end 2024-12-31
```

### 7. Start Bot (Paper Trading)

```bash
python run.py
```

### 8. Start Dashboard

```bash
python run.py --dashboard
# Open http://localhost:8000
```

## Trading Strategy

### Consensus Tiers

| Tier | Condition | Position Size |
|------|-----------|---------------|
| 1 | Any model >85% confidence | 2% |
| 2 | 2/3 models >60% | 2.5% |
| 3 | All 3 models >60% | 3.5% |
| 4 | 1 model >85%, others neutral | 1.5% |

### Exit Strategy

- **Stop Loss**: -3% (hard stop)
- **Take Profit**: +6% (standard), +8% (Tier 3)
- **Partial Exit**: 40% at +5%
- **Trailing Stop**: -2% from peak (activates at +5%)
- **Time Stop**: Exit after 6 hours if no movement

### Risk Limits

- Max 5 concurrent positions
- Max 10% portfolio at risk
- Daily -8% loss limit stops trading

## Project Structure

```
trading-bot/
├── config/
│   ├── settings.py      # All configuration
│   └── coins.py         # Coin watchlist
├── src/
│   ├── data/
│   │   ├── collector.py # Binance API
│   │   ├── database.py  # SQLAlchemy models
│   │   └── features.py  # 80+ features
│   ├── models/
│   │   ├── ta_analyzer.py   # Technical analysis
│   │   ├── ml_model.py      # XGBoost
│   │   ├── lstm_model.py    # LSTM
│   │   └── ensemble.py      # Consensus system
│   ├── trading/
│   │   ├── executor.py      # Order execution
│   │   ├── positions.py     # Position tracking
│   │   └── risk.py          # Risk management
│   └── bot.py               # Main orchestrator
├── training/
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   └── backtest.py
├── dashboard/
│   └── app.py           # Web dashboard
└── run.py               # Entry point
```

## Commands

```bash
# Run bot (paper trading)
python run.py

# Run bot (live trading - CAREFUL!)
python run.py --mode live

# Scan for signals only
python run.py --scan

# Run once and exit
python run.py --once

# Start dashboard
python run.py --dashboard

# Backfill data
python run.py --backfill 90
```

## Monitoring

### Dashboard

The web dashboard shows:
- System status and paper balance
- Open positions with live P&L
- Today's trading performance
- Signal scanner

### Logs

Logs are stored in `logs/`:
- `bot_YYYY-MM-DD.log` - General logs
- `trades_YYYY-MM-DD.log` - Trade-specific logs
- `errors.log` - Error logs

## Safety Checklist

Before going live:

- [ ] Run paper trading for 2+ months
- [ ] Achieve >55% win rate
- [ ] Max drawdown <30%
- [ ] All safety systems tested
- [ ] Start with small capital ($100-500)

## License

MIT License - Use at your own risk.

## Disclaimer

⚠️ **Trading cryptocurrencies involves substantial risk of loss. This bot is for educational purposes. Never invest more than you can afford to lose.**
