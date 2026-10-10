"""Sentiment and event models behind one call, with an optional on-disk cache.

Scoring text is the only slow step on CPU and it is stateless, so results are cached by text
hash. scripts/prescore.py fills the cache in parallel shards; the engine then replays from it.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pandas as pd

from riskengine.config import RAW
from riskengine.nlp.events import Event, EventClassifier
from riskengine.nlp.sentiment import FinBertSentiment, Sentiment

CACHE_DIR = RAW / "score_cache"
_FIELDS = ["sentiment", "sentiment_label", "sentiment_conf", "event_type", "event_conf"]


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def load_cache(cache_dir: Path = CACHE_DIR) -> dict[str, tuple]:
    cache: dict[str, tuple] = {}
    if cache_dir.exists():
        for f in sorted(cache_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            cache.update(zip(df["key"], df[_FIELDS].itertuples(index=False, name=None)))
    return cache


class TextScorer:
    def __init__(self, sentiment: FinBertSentiment | None = None, events: EventClassifier | None = None, cache_dir: Path | None = CACHE_DIR):
        self._sentiment, self._events = sentiment, events
        self.cache = load_cache(cache_dir) if cache_dir else {}
        self._load_lock = threading.Lock()

    # Models load lazily: a fully cached replay never needs them. The lock matters because the
    # dashboard and API share one scorer across threads, and two threads importing transformers
    # for the first time at once fail with "cannot import name".
    @property
    def sentiment(self) -> FinBertSentiment:
        with self._load_lock:
            if self._sentiment is None:
                self._sentiment = FinBertSentiment()
        return self._sentiment

    @property
    def events(self) -> EventClassifier:
        with self._load_lock:
            if self._events is None:
                self._events = EventClassifier()
        return self._events

    def score(self, texts: list[str]) -> list[tuple[Sentiment, Event]]:
        keys = [text_key(t) for t in texts]
        missing = list({k: t for k, t in zip(keys, texts) if k not in self.cache}.items())
        if missing:
            new_texts = [t for _, t in missing]
            for (k, _), s, e in zip(missing, self.sentiment.score(new_texts), self.events.classify(new_texts)):
                self.cache[k] = (s.score, s.label, s.confidence, e.label, e.confidence)
        return [(Sentiment(*self.cache[k][:3]), Event(*self.cache[k][3:])) for k in keys]

    def rows(self, texts: list[str]) -> pd.DataFrame:
        """Cache entries for `texts` as a frame, ready to be written to a shard file."""
        keys = list(dict.fromkeys(text_key(t) for t in texts))
        return pd.DataFrame([(k, *self.cache[k]) for k in keys], columns=["key", *_FIELDS])
