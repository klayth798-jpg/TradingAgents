"""Point-in-time provenance markers for agent-facing data payloads."""

from __future__ import annotations

from datetime import datetime, timezone

from tradingagents.dataflows.config import get_config


def live_as_of() -> str:
    """Return an explicit UTC timestamp for live data retrieval."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def annotate_data(
    payload: object,
    *,
    source: str,
    as_of: str | None = None,
    historical: bool | None = None,
) -> str:
    """Attach a small, machine-readable provenance header to tool output.

    Agent tools continue returning strings for LangChain compatibility. The
    first line is a stable contract that downstream extractors and reports can
    parse without guessing which time boundary applied to the payload.
    """
    config = get_config()
    if not config.get("data_provenance_enabled", True):
        return str(payload)
    if historical is None:
        historical = bool(config.get("historical_mode"))
    cutoff = as_of or live_as_of()
    mode = "historical" if historical else "live"
    return f"DATA_PROVENANCE: source={source}; as_of={cutoff}; mode={mode}\n{payload}"


def historical_source_disabled(source: str, as_of: str) -> str:
    """Fail-closed sentinel for a source without point-in-time history."""
    return annotate_data(
        f"DATA_UNAVAILABLE: {source} is live-only and is disabled in strict "
        f"historical mode. No value was estimated or substituted.",
        source=source,
        as_of=as_of,
        historical=True,
    )
