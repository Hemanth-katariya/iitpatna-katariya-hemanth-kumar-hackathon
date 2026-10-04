"""Data contracts between ingestion, the engine and downstream modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Document:
    """One raw item from any source, before analysis."""

    doc_id: str
    ts: str  # ISO-8601 UTC
    source: str  # "news" | "social"
    publisher: str
    text: str
    url: str = ""
    weight: float = 1.0  # how many raw documents this one stands for (sampled sources)


@dataclass(frozen=True)
class Signal:
    """One structured risk signal: a document's reading for one ticker (or the market)."""

    doc_id: str
    ts: str
    source: str
    publisher: str
    ticker: str
    sector: str
    text: str
    sentiment: float  # -1 .. 1
    sentiment_label: str
    sentiment_conf: float
    event_type: str
    event_conf: float
    impact: float  # 1 .. 10
    attention: float  # today's document volume for the ticker relative to its trailing norm
    drivers: dict[str, float] = field(default_factory=dict)  # additive contributions to the impact score

    def to_dict(self) -> dict:
        return asdict(self)
