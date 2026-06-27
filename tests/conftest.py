"""
conftest.py — shared pytest fixtures & module stubs.

Loaded automatically by pytest before any test module, so all the heavy
project imports (Binance API, SQLAlchemy, ML models, etc.) are replaced
with lightweight mocks before the real source files get imported.
"""

import sys
import os
from unittest.mock import MagicMock

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── shared mock singletons ────────────────────────────────────────────────────
_db_mock   = MagicMock(name="db")
_log_mock  = MagicMock(name="logger")
_disc_mock = MagicMock(name="discord_notifier")
_coll_mock = MagicMock(name="collector")

_db_mock.get_daily_stats.return_value = {"pnl": 0.0}
_db_mock.get_open_trades.return_value = []
_db_mock.update_trade.return_value    = None

# ── Real enums — import the actual database module directly so we get proper
#    enum types that the source code isinstance()-checks against.
# ─────────────────────────────────────────────────────────────────────────────
# We pre-load the real database module in isolation (it only needs SQLAlchemy).
import importlib as _imp
_db_real = _imp.import_module("src.data.database")

TradeStatus = _db_real.TradeStatus
ExitReason  = _db_real.ExitReason
Trade       = _db_real.Trade

# ── Register ALL module stubs before any project package is imported ──────────
_stub_modules = {
    # ── data layer ─────────────────────────────────────────────────────────
    "src.data": MagicMock(
        db=_db_mock,
        Trade=Trade,
        TradeStatus=TradeStatus,
        ExitReason=ExitReason,
        collector=_coll_mock,
        BinanceCollector=MagicMock(),
        feature_engineer=MagicMock(),
        FeatureEngineer=MagicMock(),
    ),
    "src.data.database": MagicMock(
        db=_db_mock,
        Trade=Trade,
        TradeStatus=TradeStatus,
        ExitReason=ExitReason,
        DatabaseManager=MagicMock(),
        PriceData=MagicMock(),
        Indicator=MagicMock(),
        ModelPerformance=MagicMock(),
        DailyPerformance=MagicMock(),
    ),
    "src.data.collector":  MagicMock(collector=_coll_mock, BinanceCollector=MagicMock()),
    "src.data.features":   MagicMock(feature_engineer=MagicMock(), FeatureEngineer=MagicMock()),
    "src.data.state":      MagicMock(save_state=MagicMock(), load_state=MagicMock()),

    # ── utils layer ─────────────────────────────────────────────────────────
    "src.utils": MagicMock(
        logger=_log_mock,
        log_risk_event=MagicMock(),
        log_trade=MagicMock(),
        log_signal=MagicMock(),
        log_position_update=MagicMock(),
    ),
    "src.utils.logger": MagicMock(
        logger=_log_mock,
        log_risk_event=MagicMock(),
        log_trade=MagicMock(),
        log_signal=MagicMock(),
        log_position_update=MagicMock(),
        setup_logging=MagicMock(),
    ),
    "src.utils.notifiers": MagicMock(discord_notifier=_disc_mock),

    # ── model layer (heavy ML deps) ──────────────────────────────────────────
    "src.models.ta_analyzer": MagicMock(ta_analyzer=MagicMock(), TASignal=MagicMock()),
    "src.models.ml_model":    MagicMock(ml_model=MagicMock(),  MLSignal=MagicMock()),
    "src.models.tcn_model":   MagicMock(tcn_model=MagicMock(), TCNSignal=MagicMock()),

    # ── executor (pulls in state, binance, etc.) ─────────────────────────────
    "src.trading.executor": MagicMock(
        executor=MagicMock(),
        TradingExecutor=MagicMock(),
        ExecutionMode=MagicMock(),
        OrderResult=MagicMock(),
    ),
}

for mod_name, stub in _stub_modules.items():
    sys.modules.setdefault(mod_name, stub)

# Make the shared mocks available to test files via pytest fixtures if needed.
import pytest   # noqa: E402

@pytest.fixture
def db_mock():
    _db_mock.reset_mock()
    _db_mock.get_daily_stats.return_value = {"pnl": 0.0}
    _db_mock.get_open_trades.return_value = []
    _db_mock.update_trade.return_value    = None
    return _db_mock

@pytest.fixture
def log_mock():
    _log_mock.reset_mock()
    return _log_mock
