import pandas as pd
import pytest

from tradingagents.backtesting.engine import BacktestConfig, BacktestEngine


def _prices():
    return pd.DataFrame(
        {
            "Open": [100.0, 110.0, 120.0, 115.0],
            "Close": [105.0, 115.0, 118.0, 117.0],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
    )


@pytest.mark.unit
def test_signal_executes_at_next_open_not_same_close():
    result = BacktestEngine(BacktestConfig(
        initial_cash=1_000,
        commission_bps=0,
        slippage_bps=0,
    )).run("AAPL", _prices(), {"2025-01-02": "Buy"})
    assert len(result.fills) == 1
    assert result.fills[0].signal_date == "2025-01-02"
    assert result.fills[0].execution_date == "2025-01-03"
    assert result.fills[0].price == 110.0
    assert result.fills[0].quantity == 9


@pytest.mark.unit
def test_sell_exits_existing_position_at_next_open():
    result = BacktestEngine(BacktestConfig(
        initial_cash=1_000,
        commission_bps=0,
        slippage_bps=0,
    )).run(
        "AAPL",
        _prices(),
        {"2025-01-02": "Buy", "2025-01-06": "Sell"},
    )
    assert [fill.quantity for fill in result.fills] == [9, -9]
    assert result.equity_curve.iloc[-1]["quantity"] == 0


@pytest.mark.unit
def test_hold_does_not_force_an_arbitrary_target_weight():
    result = BacktestEngine(BacktestConfig(
        initial_cash=1_000,
        commission_bps=0,
        slippage_bps=0,
    )).run("AAPL", _prices(), {"2025-01-02": "Hold"})
    assert result.fills == []
    assert result.metrics["final_equity"] == pytest.approx(1_000)


@pytest.mark.unit
def test_costs_reduce_equity():
    no_cost = BacktestEngine(BacktestConfig(
        initial_cash=1_000, commission_bps=0, slippage_bps=0
    )).run("AAPL", _prices(), {"2025-01-02": "Buy"})
    with_cost = BacktestEngine(BacktestConfig(
        initial_cash=1_000, commission_bps=10, slippage_bps=10
    )).run("AAPL", _prices(), {"2025-01-02": "Buy"})
    assert with_cost.metrics["final_equity"] < no_cost.metrics["final_equity"]
