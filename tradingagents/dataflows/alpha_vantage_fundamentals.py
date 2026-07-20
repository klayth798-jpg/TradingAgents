import json
from datetime import datetime, timedelta

from .alpha_vantage_common import _make_api_request
from .config import get_config


def _availability_date(report: dict, *, annual: bool) -> tuple[str, str]:
    """Return the best available publication date and its provenance."""
    for key in ("filingDate", "reportedDate", "publishedAt", "acceptedDate"):
        value = str(report.get(key) or "")[:10]
        if value:
            return value, key
    fiscal_end = report.get("fiscalDateEnding", "")
    if not fiscal_end:
        return "", "missing"
    lags = get_config().get("fundamentals_publication_lag_days", {})
    lag = int(lags.get("annual" if annual else "quarterly", 90 if annual else 45))
    try:
        estimated = datetime.strptime(fiscal_end, "%Y-%m-%d") + timedelta(days=lag)
    except ValueError:
        return "", "invalid_fiscal_date"
    return estimated.strftime("%Y-%m-%d"), f"estimated_fiscal_end_plus_{lag}d"


def _filter_reports_by_date(result, curr_date: str):
    """Keep only reports available to the market by ``curr_date``.

    ``_make_api_request`` returns the fundamentals payload as a JSON string, so
    parse, filter, and re-serialize. A non-JSON body or an unset ``curr_date`` is
    returned unchanged.
    """
    if not curr_date or not isinstance(result, str):
        return result
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return result
    if not isinstance(payload, dict):
        return result
    strict_historical = bool(get_config().get("historical_mode"))
    for key in ("annualReports", "quarterlyReports"):
        if isinstance(payload.get(key), list):
            if not strict_historical:
                payload[key] = [
                    report for report in payload[key]
                    if report.get("fiscalDateEnding", "") <= curr_date
                ]
                continue
            kept = []
            for report in payload[key]:
                available, source = _availability_date(report, annual=key == "annualReports")
                if available and available <= curr_date:
                    report = dict(report)
                    report["_publishedAt"] = available
                    report["_publishedAtSource"] = source
                    kept.append(report)
            payload[key] = kept
    return json.dumps(payload)


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    if curr_date and get_config().get("historical_mode"):
        return (
            "DATA_UNAVAILABLE: Alpha Vantage OVERVIEW is a current snapshot "
            f"without published_at metadata and is excluded as of {curr_date}."
        )
    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)
