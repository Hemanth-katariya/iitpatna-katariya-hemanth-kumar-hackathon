from riskengine.config import MARKET
from riskengine.engine import RiskEngine
from riskengine.nlp.events import Event
from riskengine.nlp.impact import ImpactModel, ScopeModel
from riskengine.nlp.sentiment import Sentiment
from riskengine.schema import Document


class StubScorer:
    """Stands in for the NLP models: negative if the text says 'falls', and counts what it scores."""

    def __init__(self):
        self.scored: list[str] = []

    def score(self, texts):
        self.scored.extend(texts)
        return [
            (Sentiment(-0.8, "negative", 0.9) if "falls" in t else Sentiment(0.6, "positive", 0.8), Event("Earnings", 0.7))
            for t in texts
        ]


def _engine() -> tuple[RiskEngine, StubScorer]:
    scope = ScopeModel(1.0, {"Earnings": 0.3}, {"news": 0.0, "social": 0.0}, 0.5, 0.1, 0.3, [0.8 + 0.2 * i for i in range(10)])
    scorer = StubScorer()
    return RiskEngine(scorer, ImpactModel(scope, scope)), scorer


def _doc(i: int, text: str, source: str = "news", day: str = "2020-03-02") -> Document:
    return Document(f"d{i}", f"{day}T10:00:00Z", source, "pub", text)


def test_one_signal_per_document_and_ticker_with_structured_fields():
    engine, _ = _engine()
    signals = engine.process([_doc(1, "Apple falls as Microsoft cuts its outlook for the quarter")])
    assert [s.ticker for s in signals] == ["AAPL", "MSFT"]
    s = signals[0]
    assert s.sector == "Information Technology" and s.event_type == "Earnings"
    assert -1 <= s.sentiment <= 1 and 1 <= s.impact <= 10
    assert set(s.drivers) == {"baseline", "event_type", "tone", "attention", "source"}


def test_duplicates_boilerplate_and_unlinked_text_are_dropped():
    engine, scorer = _engine()
    signals = engine.process(
        [
            _doc(1, "Boeing falls after regulators extend the 737 Max grounding"),
            _doc(2, "BOEING FALLS after regulators extend the 737 Max grounding!"),  # duplicate
            _doc(3, "Oakbrook Investments Llc Buys Pfizer Inc, Target Corp, Sells Apple"),  # fund-holdings boilerplate
            _doc(4, "Local bakery wins a regional award for its sourdough"),  # nothing we cover
            _doc(5, "Fed cuts interest rates to zero in an emergency move"),
        ]
    )
    assert [s.ticker for s in signals] == ["BA", MARKET]
    assert len(scorer.scored) == 2  # filtered documents never reach the models


def test_social_posts_link_by_cashtag_only():
    engine, _ = _engine()
    assert engine.process([_doc(1, "Tesla is overvalued here, change my mind please", "social")]) == []
    assert [s.ticker for s in engine.process([_doc(2, "$TSLA is overvalued here, change my mind", "social")])] == ["TSLA"]


def test_days_are_processed_in_order_and_negative_news_scores_higher():
    engine, _ = _engine()
    signals = engine.process(
        [
            _doc(1, "Intel rallies after strong data centre demand", day="2020-03-03"),
            _doc(2, "Intel falls after delaying its next chip generation", day="2020-03-02"),
        ]
    )
    assert [s.ts[:10] for s in signals] == ["2020-03-02", "2020-03-03"]
    assert signals[0].impact > signals[1].impact
