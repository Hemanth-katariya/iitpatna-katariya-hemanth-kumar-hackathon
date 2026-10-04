"""The risk engine: documents in, structured risk signals out."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

from riskengine.config import BY_TICKER, MARKET
from riskengine.nlp.entities import link
from riskengine.nlp.impact import ImpactModel
from riskengine.nlp.scoring import TextScorer
from riskengine.nlp.text import clean, dedup_key, is_boilerplate
from riskengine.schema import Document, Signal

MIN_CHARS = 20
MAX_TICKERS_PER_DOC = 3  # round-ups naming many companies say little about any one of them


class AttentionTracker:
    """Document volume per (ticker, source) today, relative to that stream's own trailing mean.

    Streams are tracked per source so that switching on a new feed does not look like a burst
    of attention, and a stream reports a neutral 1.0 until it has `min_history` days behind it.
    """

    def __init__(self, window: int = 20, min_history: int = 5):
        self.window, self.min_history = window, min_history
        self._history: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._sources_seen: dict[str, int] = defaultdict(int)  # days observed per source

    def update(self, counts: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
        """Record one day of counts keyed by (ticker, source); return each stream's attention ratio."""
        ratios = {}
        for key, n in counts.items():
            hist = self._history[key]
            if self._sources_seen[key[1]] < self.min_history:
                ratios[key] = 1.0
            else:
                # Days on which the stream was silent count as zero volume.
                mean = sum(hist) / min(self._sources_seen[key[1]], self.window)
                ratios[key] = (n + 1.0) / (mean + 1.0)
        for key in set(self._history) | set(counts):
            self._history[key].append(counts.get(key, 0.0))
        for source in {s for _, s in counts}:
            self._sources_seen[source] += 1
        return ratios


class RiskEngine:
    def __init__(
        self,
        scorer: TextScorer | None = None,
        impact: ImpactModel | None = None,
        attention: AttentionTracker | None = None,
    ):
        self.scorer = scorer or TextScorer()
        self.impact = impact or ImpactModel.load()
        self.attention = attention or AttentionTracker()
        self._seen: set[str] = set()

    def process(self, docs: Iterable[Document]) -> list[Signal]:
        """Analyse documents in chronological day batches and return one signal per (document, ticker)."""
        by_day: dict[str, list[tuple[Document, str, list[str]]]] = defaultdict(list)
        for doc in docs:
            text = clean(doc.text)
            key = dedup_key(text)
            if len(text) < MIN_CHARS or key in self._seen or is_boilerplate(text):
                continue
            tickers = link(text, doc.source)
            if not tickers or len(tickers) > MAX_TICKERS_PER_DOC:
                continue
            self._seen.add(key)
            by_day[doc.ts[:10]].append((doc, text, tickers))

        signals: list[Signal] = []
        for day in sorted(by_day):
            signals.extend(self._process_day(by_day[day]))
        return signals

    def _process_day(self, items: list[tuple[Document, str, list[str]]]) -> list[Signal]:
        scores = self.scorer.score([text for _, text, _ in items])

        counts: dict[tuple[str, str], float] = defaultdict(float)
        for doc, _, tickers in items:
            for t in tickers:
                counts[(t, doc.source)] += doc.weight
        attention = self.attention.update(dict(counts))

        out = []
        for (doc, text, tickers), (s, e) in zip(items, scores):
            for t in tickers:
                att = attention[(t, doc.source)]
                impact, drivers = self.impact.score(t, e.label, s.score, att, doc.source)
                out.append(
                    Signal(
                        doc_id=doc.doc_id,
                        ts=doc.ts,
                        source=doc.source,
                        publisher=doc.publisher,
                        ticker=t,
                        sector="Market" if t == MARKET else BY_TICKER[t].sector,
                        text=text,
                        sentiment=round(s.score, 4),
                        sentiment_label=s.label,
                        sentiment_conf=round(s.confidence, 4),
                        event_type=e.label,
                        event_conf=round(e.confidence, 4),
                        impact=impact,
                        attention=round(att, 3),
                        drivers=drivers,
                    )
                )
        return out
