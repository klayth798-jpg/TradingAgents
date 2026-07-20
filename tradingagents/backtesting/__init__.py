"""Point-in-time simulation primitives for TradingAgents."""

from .engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    CommissionModel,
    Fill,
    Order,
    Portfolio,
    SimulatedBroker,
    SlippageModel,
)


def run_agent_backtest(*args, **kwargs):
    """Lazy import so the deterministic engine can be used without yfinance."""
    from .runner import run_agent_backtest as _run_agent_backtest

    return _run_agent_backtest(*args, **kwargs)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "CommissionModel",
    "Fill",
    "Order",
    "Portfolio",
    "SimulatedBroker",
    "SlippageModel",
    "run_agent_backtest",
]
