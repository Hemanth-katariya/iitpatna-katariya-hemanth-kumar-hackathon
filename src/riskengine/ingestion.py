"""Source adapters. Each one yields `Document`s; the engine does not care where they came from.

Historical (replay) : CsvSource over the committed samples in data/.
Live news           : Google News RSS, Yahoo Finance RSS, GDELT DOC API.
Live social         : StockTwits public symbol streams.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import quote_plus

import pandas as pd
import requests

from riskengine.config import BY_TICKER, DATA, INDEX_TICKERS
from riskengine.nlp.text import doc_id
from riskengine.schema import Document

_HEADERS = {"User-Agent": "Mozilla/5.0 (riskengine hackathon prototype)"}
_TIMEOUT = 30


class Source(Protocol):
    name: str

    def fetch(self) -> Iterator[Document]: ...


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _doc(source: str, publisher: str, ts: str, text: str, url: str = "", weight: float = 1.0) -> Document:
    return Document(doc_id(source, ts, text), ts, source, publisher, text, url, weight)


class CsvSource:
    """Replays a committed sample file, optionally restricted to [start, end] dates (inclusive)."""

    def __init__(self, path: Path | str, start: str | None = None, end: str | None = None):
        self.path, self.start, self.end = Path(path), start, end
        self.name = self.path.stem

    def fetch(self) -> Iterator[Document]:
        df = pd.read_csv(self.path, keep_default_na=False)
        day = df["ts"].str[:10]
        if self.start:
            df = df[day >= self.start]
        if self.end:
            df = df[day <= self.end]
        for r in df.itertuples(index=False):
            yield Document(r.doc_id, r.ts, r.source, r.publisher, r.text, r.url, float(r.weight))


def replay_sources(start: str | None = None, end: str | None = None) -> list[CsvSource]:
    return [CsvSource(DATA / "news.csv", start, end), CsvSource(DATA / "tweets.csv", start, end)]


class GoogleNewsSource:
    name = "google_news"

    def __init__(self, tickers: tuple[str, ...] = INDEX_TICKERS, window: str = "2d", extra_queries: tuple[str, ...] = ()):
        self.queries = [f'"{BY_TICKER[t].name}" stock' for t in tickers] + list(extra_queries)
        self.window = window

    def fetch(self) -> Iterator[Document]:
        import feedparser

        for q in self.queries:
            url = f"https://news.google.com/rss/search?q={quote_plus(q)}+when:{self.window}&hl=en-US&gl=US&ceid=US:en"
            for e in feedparser.parse(url, request_headers=_HEADERS).entries:
                publisher = e.get("source", {}).get("title", "Google News")
                # Google appends " - Publisher" to every title.
                title = e.title.rsplit(" - ", 1)[0] if e.title.endswith(f" - {publisher}") else e.title
                yield _doc("news", publisher, _iso(parsedate_to_datetime(e.published)), title, e.link)


class YahooFinanceSource:
    name = "yahoo_finance"

    def __init__(self, tickers: tuple[str, ...] = INDEX_TICKERS):
        self.tickers = tickers

    def fetch(self) -> Iterator[Document]:
        import feedparser

        for t in self.tickers:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
            for e in feedparser.parse(url, request_headers=_HEADERS).entries:
                yield _doc("news", "Yahoo Finance", _iso(parsedate_to_datetime(e.published)), e.title, e.link)


class GdeltSource:
    """GDELT DOC 2.0 article search. GDELT allows one request every five seconds."""

    name = "gdelt"

    def __init__(self, queries: tuple[str, ...], timespan: str = "1d", max_records: int = 75):
        self.queries, self.timespan, self.max_records = queries, timespan, max_records

    def fetch(self) -> Iterator[Document]:
        for i, q in enumerate(self.queries):
            if i:
                time.sleep(5.5)
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": f"{q} sourcelang:english",
                    "mode": "ArtList",
                    "format": "json",
                    "timespan": self.timespan,
                    "maxrecords": self.max_records,
                    "sort": "DateDesc",
                },
                headers=_HEADERS,
                timeout=_TIMEOUT,
            )
            if r.status_code != 200 or not r.text.lstrip().startswith("{"):
                continue  # throttled or empty; the next poll will pick it up
            for a in r.json().get("articles", []):
                ts = _iso(datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))
                yield _doc("news", a.get("domain", "GDELT"), ts, a["title"], a.get("url", ""))


class StockTwitsSource:
    name = "stocktwits"

    def __init__(self, tickers: tuple[str, ...] = INDEX_TICKERS):
        self.tickers = tickers

    def fetch(self) -> Iterator[Document]:
        for t in self.tickers:
            r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{t}.json", headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code != 200:
                continue
            for m in r.json().get("messages", []):
                yield _doc("social", "stocktwits", m["created_at"], m["body"], f"https://stocktwits.com/message/{m['id']}")


MACRO_QUERIES = ("Federal Reserve interest rates", "sanctions OR tariffs economy", "oil prices OPEC", "credit rating downgrade OR default")


def live_sources(tickers: tuple[str, ...] = INDEX_TICKERS) -> list[Source]:
    return [
        GoogleNewsSource(tickers, extra_queries=MACRO_QUERIES),
        YahooFinanceSource(tickers),
        StockTwitsSource(tickers),
    ]


def fetch_all(sources: list[Source]) -> list[Document]:
    """Pull every source, tolerating individual failures, and return documents oldest first."""
    docs: list[Document] = []
    for s in sources:
        try:
            docs.extend(s.fetch())
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"[ingestion] {s.name} failed: {type(e).__name__}: {e}")
    return sorted(docs, key=lambda d: d.ts)
