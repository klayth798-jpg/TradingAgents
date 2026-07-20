"""Deterministic long-only portfolio simulator.

Signals are generated after a signal day's close and execute at the next
available session's open. This timing rule is deliberate: executing a decision
against the same close used to produce it would introduce look-ahead bias.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from tradingagents.agents.utils.rating import parse_rating


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    annualization_days: int = 252
    target_weights: dict[str, float | None] = field(default_factory=lambda: {
        "Buy": 1.0,
        "Overweight": 0.75,
        "Hold": None,
        "Underweight": 0.25,
        "Sell": 0.0,
    })

    def __post_init__(self):
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("commission_bps and slippage_bps must be non-negative")
        for rating, weight in self.target_weights.items():
            if weight is not None and not 0 <= weight <= 1:
                raise ValueError(f"target weight for {rating} must be between 0 and 1")


@dataclass(frozen=True)
class Order:
    ticker: str
    signal_date: str
    execution_date: str
    rating: str
    quantity: int
    target_weight: float


@dataclass(frozen=True)
class Fill:
    ticker: str
    signal_date: str
    execution_date: str
    rating: str
    quantity: int
    price: float
    commission: float
    slippage_cost: float


@dataclass
class Portfolio:
    cash: float
    quantity: int = 0

    def equity(self, mark_price: float) -> float:
        return self.cash + self.quantity * mark_price


@dataclass(frozen=True)
class CommissionModel:
    bps: float = 5.0

    def calculate(self, notional: float) -> float:
        return abs(notional) * self.bps / 10_000


@dataclass(frozen=True)
class SlippageModel:
    bps: float = 5.0

    def execution_price(self, reference_price: float, quantity: int) -> float:
        if quantity == 0:
            return reference_price
        direction = 1 if quantity > 0 else -1
        return reference_price * (1 + direction * self.bps / 10_000)


class SimulatedBroker:
    """Execute whole-share market orders with costs and no shorting."""

    def __init__(self, commission: CommissionModel, slippage: SlippageModel):
        self.commission = commission
        self.slippage = slippage

    def execute(self, portfolio: Portfolio, order: Order, open_price: float) -> Fill | None:
        quantity = order.quantity
        if quantity > 0:
            execution_price = self.slippage.execution_price(open_price, quantity)
            per_share_cost = execution_price * (1 + self.commission.bps / 10_000)
            quantity = min(quantity, math.floor(portfolio.cash / per_share_cost))
        else:
            quantity = max(quantity, -portfolio.quantity)
            execution_price = self.slippage.execution_price(open_price, quantity)

        if quantity == 0:
            return None

        notional = quantity * execution_price
        commission = self.commission.calculate(notional)
        portfolio.cash -= notional + commission
        portfolio.quantity += quantity
        return Fill(
            ticker=order.ticker,
            signal_date=order.signal_date,
            execution_date=order.execution_date,
            rating=order.rating,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage_cost=abs(quantity) * abs(execution_price - open_price),
        )


@dataclass
class BacktestResult:
    ticker: str
    config: BacktestConfig
    signals: dict[str, str]
    fills: list[Fill]
    equity_curve: pd.DataFrame
    metrics: dict[str, float]

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.equity_curve.to_csv(directory / "equity_curve.csv", index=False)
        pd.DataFrame([asdict(fill) for fill in self.fills]).to_csv(
            directory / "fills.csv", index=False
        )
        (directory / "signals.json").write_text(
            json.dumps(self.signals, indent=2), encoding="utf-8"
        )
        summary = {
            "ticker": self.ticker,
            "config": asdict(self.config),
            "metrics": self.metrics,
        }
        path = directory / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.broker = SimulatedBroker(
            CommissionModel(self.config.commission_bps),
            SlippageModel(self.config.slippage_bps),
        )

    def run(
        self,
        ticker: str,
        prices: pd.DataFrame,
        signals: dict[str, str],
    ) -> BacktestResult:
        prices = self._normalize_prices(prices)
        if len(prices) < 2:
            raise ValueError("backtest requires at least two trading sessions")

        normalized_signals = {
            pd.Timestamp(date).strftime("%Y-%m-%d"): parse_rating(signal)
            for date, signal in signals.items()
        }
        sessions = list(prices.index)
        session_lookup = {date.strftime("%Y-%m-%d"): idx for idx, date in enumerate(sessions)}
        execution_schedule: dict[str, tuple[str, str]] = {}
        for signal_date, rating in normalized_signals.items():
            idx = session_lookup.get(signal_date)
            if idx is None or idx + 1 >= len(sessions):
                continue
            execution_date = sessions[idx + 1].strftime("%Y-%m-%d")
            execution_schedule[execution_date] = (signal_date, rating)

        portfolio = Portfolio(cash=self.config.initial_cash)
        fills: list[Fill] = []
        rows = []
        for date, row in prices.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            scheduled = execution_schedule.get(date_str)
            if scheduled:
                signal_date, rating = scheduled
                target_weight = self.config.target_weights[rating]
                if target_weight is not None:
                    equity_at_open = portfolio.equity(open_price)
                    desired_quantity = math.floor(equity_at_open * target_weight / open_price)
                    order = Order(
                        ticker=ticker,
                        signal_date=signal_date,
                        execution_date=date_str,
                        rating=rating,
                        quantity=desired_quantity - portfolio.quantity,
                        target_weight=target_weight,
                    )
                    fill = self.broker.execute(portfolio, order, open_price)
                    if fill:
                        fills.append(fill)
            rows.append({
                "date": date_str,
                "open": open_price,
                "close": close_price,
                "cash": portfolio.cash,
                "quantity": portfolio.quantity,
                "equity": portfolio.equity(close_price),
            })

        curve = pd.DataFrame(rows)
        metrics = self._metrics(curve, prices)
        metrics["trade_count"] = float(len(fills))
        return BacktestResult(
            ticker=ticker,
            config=self.config,
            signals=normalized_signals,
            fills=fills,
            equity_curve=curve,
            metrics=metrics,
        )

    @staticmethod
    def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        required = {"Open", "Close"}
        if not required.issubset(prices.columns):
            raise ValueError("prices must contain Open and Close columns")
        frame = prices.loc[:, ["Open", "Close"]].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame.sort_index().dropna()
        if (frame <= 0).any().any():
            raise ValueError("prices must be positive")
        return frame

    def _metrics(self, curve: pd.DataFrame, prices: pd.DataFrame) -> dict[str, float]:
        equity = curve["equity"].astype(float)
        daily_returns = equity.pct_change().dropna()
        total_return = equity.iloc[-1] / self.config.initial_cash - 1
        years = max((len(equity) - 1) / self.config.annualization_days, 1 / 252)
        annualized_return = (equity.iloc[-1] / self.config.initial_cash) ** (1 / years) - 1
        running_max = equity.cummax()
        max_drawdown = float((equity / running_max - 1).min())
        volatility = float(daily_returns.std(ddof=0)) if len(daily_returns) else 0.0
        sharpe = (
            float(daily_returns.mean() / volatility * math.sqrt(self.config.annualization_days))
            if volatility > 0 else 0.0
        )
        benchmark_return = float(prices["Close"].iloc[-1] / prices["Close"].iloc[0] - 1)
        return {
            "initial_equity": float(self.config.initial_cash),
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "benchmark_return": benchmark_return,
            "alpha_vs_buy_hold": float(total_return - benchmark_return),
        }
