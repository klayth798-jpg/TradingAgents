"""Bridge between the LLM analysis graph and the deterministic simulator."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

from .engine import BacktestConfig, BacktestEngine, BacktestResult

if TYPE_CHECKING:
    from tradingagents.graph.trading_graph import TradingAgentsGraph


def _download_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    end_exclusive = (pd.Timestamp(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    prices = yf.download(
        ticker,
        start=start_date,
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    return prices


def run_agent_backtest(
    graph: TradingAgentsGraph,
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    rebalance_days: int = 5,
    asset_type: str = "stock",
    config: BacktestConfig | None = None,
    signal_provider: Callable[[str], str] | None = None,
    prices: pd.DataFrame | None = None,
) -> BacktestResult:
    """Generate point-in-time agent signals and simulate next-open execution.

    ``signal_provider`` and ``prices`` are injectable for deterministic tests or
    for replaying previously saved decisions without paying for new LLM calls.
    """
    if not graph.config.get("historical_mode"):
        raise ValueError("agent backtests require config['historical_mode'] = True")
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be at least 1")
    prices = prices if prices is not None else _download_prices(ticker, start_date, end_date)
    normalized = BacktestEngine._normalize_prices(prices)
    signal_dates = [d.strftime("%Y-%m-%d") for d in normalized.index[::rebalance_days]][:-1]

    if signal_provider is None:
        def signal_provider(date: str) -> str:
            _, rating = graph.propagate(
                ticker,
                date,
                asset_type=asset_type,
                store_artifacts=False,
                resolve_memory=False,
            )
            return rating

    signals = {date: signal_provider(date) for date in signal_dates}
    return BacktestEngine(config).run(ticker, normalized, signals)
