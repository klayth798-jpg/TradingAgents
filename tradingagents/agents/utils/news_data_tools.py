from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.provenance import annotate_data


@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return annotate_data(
        route_to_vendor("get_news", ticker, start_date, end_date),
        source="ticker_news",
        as_of=end_date,
    )

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int | None, "Days to look back; omit to use the configured default"] = None,
    limit: Annotated[int | None, "Max articles to return; omit to use the configured default"] = None,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor. Defaults for look_back_days and
    limit come from DEFAULT_CONFIG (global_news_lookback_days,
    global_news_article_limit); pass explicit values to override.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back; omit to inherit config
        limit (int): Maximum number of articles to return; omit to inherit config

    Returns:
        str: A formatted string containing global news data
    """
    return annotate_data(
        route_to_vendor("get_global_news", curr_date, look_back_days, limit),
        source="global_news",
        as_of=curr_date,
    )

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    as_of: Annotated[str | None, "Point-in-time cutoff; required in historical mode"] = None,
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    from tradingagents.dataflows.config import get_config
    from tradingagents.dataflows.provenance import historical_source_disabled

    if get_config().get("historical_mode"):
        return historical_source_disabled("insider_transactions", as_of or "historical cutoff")
    return annotate_data(
        route_to_vendor("get_insider_transactions", ticker),
        source="insider_transactions",
        as_of=as_of,
    )
