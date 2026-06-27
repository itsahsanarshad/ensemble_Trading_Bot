# Graph Report - ensemble_trading_bot  (2026-06-27)

## Corpus Check
- 47 files · ~52,027 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 943 nodes · 1472 edges · 55 communities (45 shown, 10 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 235 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a35512cf`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]

## God Nodes (most connected - your core abstractions)
1. `TestCheckPosition` - 28 edges
2. `RiskManager` - 26 edges
3. `_make_position()` - 26 edges
4. `_make_pm()` - 25 edges
5. `_make_rm()` - 25 edges
6. `_make_tracker()` - 24 edges
7. `ConsensusEnsemble` - 23 edges
8. `ModelPerformanceTracker` - 20 edges
9. `FeatureEngineer` - 18 edges
10. `BinanceCollector` - 17 edges

## Surprising Connections (you probably didn't know these)
- `_apply_consensus_rules()` --calls--> `kelly_position_size()`  [INFERRED]
  training/backtest.py → src/models/ensemble.py
- `BacktestTrade` --uses--> `ConsensusSignal`  [INFERRED]
  training/backtest.py → src/models/ensemble.py
- `BacktestResult` --uses--> `ConsensusSignal`  [INFERRED]
  training/backtest.py → src/models/ensemble.py
- `BacktestEngine` --uses--> `ConsensusSignal`  [INFERRED]
  training/backtest.py → src/models/ensemble.py
- `BacktestTrade` --uses--> `ConsensusEnsemble`  [INFERRED]
  training/backtest.py → src/models/ensemble.py

## Hyperedges (group relationships)
- **Model Training & Backtesting** — training_backtest_py, training_train_lightgbm_py, training_train_tcn_py [INFERRED 0.85]

## Communities (55 total, 10 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (15): _make_pm(), _make_position(), Unit tests for PositionManager.check_position() — covers:    - Stop loss trigger, Small position whose highest_price >= TP1 must get a trailing_stop         set t, Price never reached TP1 — no tight trail should activate., If highest_price somehow drops, the trailing stop must not be lowered., When current_price <= trailing_stop, action must be 'trailing_stop'., Standard trail doesn't activate until gain >= 2%. (+7 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (21): Trade records for tracking and analysis., Trade, PositionManager, Open a new position.                  Args:             coin: Trading pair, Close a position (fully or partially).                  Args:             tra, Check a position for exit conditions.                  Args:             trad, Check all positions against current prices.                  Args:, Get list of all open positions. (+13 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (18): ConsensusEnsemble, kelly_position_size(), Fractional Kelly criterion position size.      Args:         win_prob       : Ca, V5 ML-Gated Consensus Ensemble.      Decision hierarchy (all checks in order):, Flexible Consensus System (Strategy 3)          Tiered approach for different, Run full consensus analysis on a symbol.          Returns a ConsensusSignal with, Analyze a coin using all three models and apply consensus rules., Classify signal tier using V5 rules.          Returns (tier, take_profit_pct). t (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (31): get_active_coins(), get_coin_config(), Coin Watchlist Configuration  Top 30 coins by volume on Binance for monitoring, Get configuration for a specific coin., Get list of actively monitored coins., Config package initialization., DatabaseSettings, ExchangeSettings (+23 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (21): Calculate position size based on tier and confidence.         Adaptive for low, Calculate position size based on tier and confidence.         Adaptive for low, Calculate stop loss price., Risk management for the trading bot.          Controls:     - Max 5 concurren, Calculate stop loss price., Calculate take profit price (TP2 - Fib 2.618 equivalent)., Calculate partial exit price TP1 (Fib 1.618 equivalent)., Calculate take profit price (TP2 - Fib 2.618 equivalent). (+13 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (31): Control Panel: Hard-Reset the Bot's Paper State     Cancels all open trades and, reset_bot_state(), load_state(), Save bot state to DB ledger and JSON backup.      Args:         paper_balance, Load bot state — tries DB first, falls back to JSON.      Returns:         Di, save_state(), OrderResult, Execute a trading signal.                  Args:             symbol: Trading (+23 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (18): FeatureEngineer, Create volume-based features (12 features)., Create technical indicators (25 features)., Create volatility features (8 features)., Create pattern recognition features (10 features)., Create time-based features (6 features)., Feature engineering for crypto trading.          Creates 80+ features grouped, Calculate RSI indicator. (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (26): MultiHeadAttention, Single temporal block with dilated causal convolution.          This is the co, Multi-Head Self-Attention for temporal sequences.          This allows the mod, Multi-Head Self-Attention for temporal sequences.          This allows the mod, Temporal Convolutional Network with Multi-Head Attention.          Architectur, Temporal Convolutional Network with Multi-Head Attention.          Architectur, Production wrapper for TCN model.          Handles:     - Model loading/savin, Production wrapper for TCN model.          Handles:     - Model loading/savin (+18 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (10): Send trade exit notification., Send daily summary notification., Send risk management alert., Send system status notification., Send trading notifications via Telegram.          Messages include:     - Tra, Initialize Telegram notifier., Send a message via Telegram., Send message synchronously. (+2 more)

### Community 9 - "Community 9"
Cohesion: 0.09
Nodes (21): get_balance_history(), get_history(), get_model_performance(), get_portfolio(), get_positions(), get_signals(), get_status(), Get all currently open positions. (+13 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (33): Base, BotState, DailyPerformance, DatabaseManager, Indicator, ModelPerformance, ModelPrediction, PriceData (+25 more)

### Community 11 - "Community 11"
Cohesion: 0.1
Nodes (15): BinanceCollector, Fetch candlestick data from Binance.                  Args:             symbo, Get current price for a symbol., Get current prices for multiple symbols., Get 24-hour statistics for a symbol., Backfill historical data for a symbol using public API.                  Args:, Collects OHLCV data from Binance API.          Supports:     - Historical dat, Update data for all monitored coins.                  Args:             timef (+7 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (20): MLSignal, ML model prediction output., TCN prediction signal., TCN prediction signal., TCNSignal, _apply_consensus_rules(), BacktestResult, BacktestTrade (+12 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (25): _adx(), _atr(), _bollinger(), calculate_all(), calculate_daily_bias(), calculate_vwap(), _detect_patterns(), _ema() (+17 more)

### Community 14 - "Community 14"
Cohesion: 0.21
Nodes (13): augment_sequences(), calculate_atr(), calculate_macd(), calculate_rsi(), engineer_features(), fetch_training_data(), PyTorch TCN Training Script - Production Ready with GPU Acceleration  Features, Data augmentation for 4x training data.          Techniques:     1. Gaussian (+5 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (10): HybridMLModel, Load saved model if exists., Get comprehensive feature list for multi-timeframe analysis., Prepare features for prediction., Train the hybrid XGBoost + CatBoost model.                  Uses ATR-based ada, Make prediction using ensemble of XGBoost + CatBoost., Hybrid XGBoost + CatBoost Model for Crypto Prediction.          Uses soft voti, Initialize hybrid model. (+2 more)

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (13): main(), Scan all coins for trading signals., Monitor all open positions., Fetch daily stats and send to Discord., Run a single iteration of the bot.                  Returns:             Dict, Run the bot continuously with scheduling., Get current bot status., Entry point for the trading bot. (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.08
Nodes (31): _adx(), _atr(), _bollinger(), calculate_daily_bias(), calculate_vwap(), _detect_patterns(), _ema(), _macd() (+23 more)

### Community 18 - "Community 18"
Cohesion: 0.27
Nodes (7): iterations, get_data_from_database(), main(), Ensemble Model Backtest Script - SPOT TRADING (BUY-ONLY)  Tests the actual TA,, Get historical data from SQLite database., Run SPOT TRADING backtest (BUY-ONLY).          This simulates real spot tradin, run_spot_backtest()

### Community 19 - "Community 19"
Cohesion: 0.33
Nodes (5): predictions, win_rates, ml, ta, tcn

### Community 20 - "Community 20"
Cohesion: 0.4
Nodes (4): last_updated, metadata, paper_balance, trades_count

### Community 21 - "Community 21"
Cohesion: 0.4
Nodes (4): balance, daily_start_balance, trades, updated_at

### Community 29 - "Community 29"
Cohesion: 0.06
Nodes (33): 1. Install Dependencies, 2. Configure Environment, 3. Initial Setup, 4. Backfill Historical Data, 5. Train Models, 6. Run Backtest, 7. Start Bot (Paper Trading), 8. Start Dashboard (+25 more)

### Community 30 - "Community 30"
Cohesion: 0.13
Nodes (9): ConfidenceCalibrator, Load tracker state from file., Calibrate model confidence using isotonic regression.          Maps raw confid, Fit calibrator for a model using historical data.                  Args:, Maps raw model confidence → actual probability of success using     Isotonic Reg, Calibrate a raw confidence score.                  Args:             model: ', Save calibrators to file., Load calibrators from file. (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.13
Nodes (13): DiscordNotifier, _format_exit_reason(), Send a TP1 partial exit alert — 50% sold, SL moved to breakeven., Send a TP1 partial exit alert — 50% sold, SL moved to breakeven., Send comprehensive daily performance summary., Send comprehensive daily performance summary., Human-readable exit reason label., Human-readable exit reason label. (+5 more)

### Community 32 - "Community 32"
Cohesion: 0.06
Nodes (25): ModelPerformanceTracker, Track individual model performance (TA, ML, TCN) for data-driven weight     adju, Get adjusted model weights based on recent performance.                  Retur, Keep only recent predictions., Save tracker state to file., Returns performance-adjusted model weights normalised to 1.0.         V5 base: M, Track individual model performance (TA, ML, TCN) for data-driven weight adjustme, Record a model's prediction before outcome is known. (+17 more)

### Community 33 - "Community 33"
Cohesion: 0.11
Nodes (12): Feature Engineering Module  Creates 80+ features for machine learning models f, Models package initialization., Hybrid ML Model: XGBoost + CatBoost Ensemble  Based on research showing: - XG, augment_sequences(), PyTorch TCN (Temporal Convolutional Network) Model — V5 (Regularized Filter), Data augmentation for time series sequences.          Techniques:     1. Gaus, Data augmentation for time series sequences.          Techniques:     1. Gaus, Utils package initialization. (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.2
Nodes (5): MarketRegimeDetector, Detect market regime (BULL / BEAR / SIDEWAYS) using BTC 4h data.     Adjusts con, Detect current market regime using BTC as benchmark.                  Uses:, Get thresholds adjusted for current market regime., Initialize ensemble with performance tracking, calibration, and regime detection

### Community 35 - "Community 35"
Cohesion: 0.17
Nodes (10): Binance Data Collector  Fetches OHLCV data from Binance API for all monitored, Data package initialization., Consensus Ensemble System — V5 (ML-Gated Architecture)  ARCHITECTURE CHANGE (V5), Main Trading Bot  Orchestrates all components: - Data collection - Model inf, Trading Executor  Executes trades based on model signals: - Order placement (, Trading package initialization., Position Manager  Tracks and manages open positions: - Entry/exit tracking -, Risk Management Module  Implements all safety controls and risk limits: - Max (+2 more)

### Community 38 - "Community 38"
Cohesion: 0.18
Nodes (13): BaseModel, ResetRequest, ExitReason, Database Models and Connection Manager  Uses SQLAlchemy with async support for, Trade status enumeration., Exit reason enumeration., TradeStatus, Enum (+5 more)

### Community 39 - "Community 39"
Cohesion: 0.22
Nodes (7): ConsensusSignal, Combined signal from all models., Combined signal from all models., Convert to dictionary., Technical Analysis signal output., Technical Analysis signal output.      V5 Contract:         signal     : Always, TASignal

### Community 40 - "Community 40"
Cohesion: 0.13
Nodes (6): _make_rm(), Unit tests for RiskManager — covers all risk checks, position sizing, trailing s, TestMaxPositions, TestPortfolioRiskLimit, TestPositionSizing, TestRecordTradeResult

### Community 41 - "Community 41"
Cohesion: 0.33
Nodes (5): check_data_coverage(), check_data_gaps(), Data Quality Checker  Verify data coverage and quality for all coins., Check data coverage for all coins., Check for gaps in data.

### Community 42 - "Community 42"
Cohesion: 0.14
Nodes (11): Check if a new position can be opened.                  Args:             coi, Get maximum positions based on current balance.         Low balance accounts ge, Get maximum positions based on current balance.         Low balance accounts ge, Get maximum portfolio risk based on current balance.         Low balance accoun, Get maximum portfolio risk based on current balance.         Low balance accoun, Record a trade result and update risk state.                  Args:, Record a trade result and update risk state.                  Args:, Pause trading for specified duration. (+3 more)

### Community 43 - "Community 43"
Cohesion: 0.22
Nodes (6): Adjust weights based on rolling performance of each model., Record the outcome of a trade to update model performance tracking., Scan a list of symbols and return buy signals sorted by tier then confidence., Scan all coins and return buy signals sorted by tier and confidence., Update model weights based on tracked performance., Record the outcome of a trade for performance tracking.                  Args:

### Community 44 - "Community 44"
Cohesion: 0.23
Nodes (4): $95 risk + $20 new = $115 / $1000 = 11.5% > 10% standard limit., $1 is far below the $6 floor — must be rejected., On a $600 account min_size = max(6, 600*0.01) = max(6, 6) = $6.         A positi, TestCanOpenPosition

### Community 45 - "Community 45"
Cohesion: 0.32
Nodes (7): evaluate_model(), get_data_from_database(), main(), Individual Model Backtester  Tests each model (TA, ML, TCN) separately to iden, Evaluate each model individually., Load historical data from SQLite database for reproducible evaluation., Evaluate a single model's predictions against actual price movements.     Uses

### Community 46 - "Community 46"
Cohesion: 0.2
Nodes (6): MultiHeadAttention, Multi-head self-attention for temporal sequences.          Allows the model to, Temporal Convolutional Network with Multi-Head Attention.          Architectur, Temporal block with dilated causal convolution.          Components:     - Tw, TCNWithAttention, TemporalBlock

### Community 47 - "Community 47"
Cohesion: 0.21
Nodes (10): main(), Trading Bot Entry Point  Main entry point for running the crypto trading bot., Initial setup for the bot., setup(), fetch_training_data(), main(), Hybrid ML Model Training Script (XGBoost + CatBoost)  Train the hybrid model w, Fetch training data from database and API. (+2 more)

### Community 48 - "Community 48"
Cohesion: 0.24
Nodes (7): BacktestEngine, main(), Load and prepare historical data., V5 Backtesting engine — connected to the live ConsensusEnsemble logic.      Uses, Calculate performance metrics., Backtesting engine for strategy validation.          Features:     - Historic, Initialize backtest engine.

### Community 49 - "Community 49"
Cohesion: 0.22
Nodes (9): meta, iteration_count, launch_mode, learn_metrics, learn_sets, name, parameters, test_metrics (+1 more)

### Community 51 - "Community 51"
Cohesion: 0.29
Nodes (3): TestTimeStop, Result of a risk check., RiskCheck

## Knowledge Gaps
- **419 isolated node(s):** `Detect 6 high-reliability candlestick patterns on the last 3 candles.`, `ADX: 0-100. Above 25 = trending. Above 40 = strong trend.`, `Bullish divergence:  price makes lower low BUT rsi makes higher low  → reversal`, `Calculate VWAP for the current session (resets each day).     Also returns dista`, `Checks the last 3 daily candles to determine macro bias.     Returns: bullish /` (+414 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RiskManager` connect `Community 4` to `Community 35`, `Community 40`, `Community 42`, `Community 44`, `Community 50`, `Community 51`, `Community 52`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Why does `ModelPerformanceTracker` connect `Community 32` to `Community 34`, `Community 35`, `Community 39`, `Community 12`, `Community 30`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Why does `RiskCheck` connect `Community 51` to `Community 35`, `Community 5`, `Community 38`, `Community 40`, `Community 42`, `Community 44`, `Community 50`, `Community 52`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `TestCheckPosition` (e.g. with `Position` and `PositionManager`) actually correct?**
  _`TestCheckPosition` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `RiskManager` (e.g. with `TestPriceLevels` and `TestTrailingStop`) actually correct?**
  _`RiskManager` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Detect 6 high-reliability candlestick patterns on the last 3 candles.`, `ADX: 0-100. Above 25 = trending. Above 40 = strong trend.`, `Bullish divergence:  price makes lower low BUT rsi makes higher low  → reversal` to the rest of the system?**
  _419 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._