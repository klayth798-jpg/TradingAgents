"""Strict point-in-time mode must fail closed on live-only sources."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.schemas import SentimentBand, SentimentReport
from tradingagents.dataflows import alpha_vantage_fundamentals as av_fundamentals, interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.stockstats_utils import filter_financials_by_date


@pytest.mark.unit
def test_prediction_markets_disabled_in_historical_mode():
    set_config({"historical_mode": True, "as_of": "2025-01-15"})
    vendor = MagicMock(return_value="LIVE DATA")
    with patch.dict(interface.VENDOR_METHODS, {"get_prediction_markets": {"polymarket": vendor}}):
        result = interface.route_to_vendor(
            "get_prediction_markets", "Fed", _as_of="2025-01-15"
        )
    assert "disabled in strict historical mode" in result
    assert "as_of=2025-01-15" in result
    vendor.assert_not_called()


@pytest.mark.unit
def test_financial_period_waits_for_publication_lag():
    frame = pd.DataFrame({pd.Timestamp("2024-12-31"): [100.0]}, index=["Revenue"])
    assert filter_financials_by_date(
        frame, "2025-01-15", publication_lag_days=45
    ).empty
    assert not filter_financials_by_date(
        frame, "2025-02-20", publication_lag_days=45
    ).empty


@pytest.mark.unit
def test_alpha_vantage_reports_use_reported_date_in_strict_mode(monkeypatch):
    set_config({"historical_mode": True})
    payload = (
        '{"quarterlyReports": ['
        '{"fiscalDateEnding": "2024-12-31", "reportedDate": "2025-02-10"}'
        ']}'
    )
    monkeypatch.setattr(av_fundamentals, "_make_api_request", lambda *args: payload)
    before = av_fundamentals.get_balance_sheet("AAPL", curr_date="2025-02-01")
    after = av_fundamentals.get_balance_sheet("AAPL", curr_date="2025-02-15")
    assert '"quarterlyReports": []' in before
    assert '"_publishedAt": "2025-02-10"' in after


@pytest.mark.unit
def test_sentiment_historical_mode_never_fetches_live_social(monkeypatch):
    report = SentimentReport(
        overall_band=SentimentBand.NEUTRAL,
        overall_score=5,
        confidence="low",
        narrative="Live social sources unavailable by policy.",
    )
    structured = MagicMock()
    structured.invoke.return_value = report
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(
        "tradingagents.agents.analysts.sentiment_analyst.get_news.func",
        lambda *args: "historical news",
    )
    stocktwits = MagicMock(side_effect=AssertionError("must not fetch live StockTwits"))
    reddit = MagicMock(side_effect=AssertionError("must not fetch live Reddit"))
    monkeypatch.setattr(
        "tradingagents.agents.analysts.sentiment_analyst.fetch_stocktwits_messages",
        stocktwits,
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.sentiment_analyst.fetch_reddit_posts",
        reddit,
    )
    state = {
        "company_of_interest": "NVDA",
        "trade_date": "2025-01-15",
        "asset_type": "stock",
        "historical_mode": True,
        "messages": [],
    }
    create_sentiment_analyst(llm)(state)
    stocktwits.assert_not_called()
    reddit.assert_not_called()
    prompt = structured.invoke.call_args.args[0]
    assert "disabled in strict historical mode" in str(prompt)
