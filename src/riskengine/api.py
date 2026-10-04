"""HTTP API over the signal store. Run with: uvicorn riskengine.api:app --port 8000"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from riskengine.config import DB_PATH, EVENT_TYPES, INDEX_TICKERS, LIVE_DB_PATH
from riskengine.market import load_prices
from riskengine.modules import rebalancer, stress
from riskengine.nlp.text import doc_id
from riskengine.schema import Document
from riskengine.store import SignalStore

app = FastAPI(
    title="AI/NLP Financial Risk Engine",
    description="Structured risk signals (sentiment, event type, impact) from news and social media, "
    "with index-rebalancing and portfolio stress-testing applications.",
    version="0.1.0",
)


def _store(feed: str) -> SignalStore:
    return SignalStore(LIVE_DB_PATH if feed == "live" else DB_PATH)


@lru_cache(maxsize=1)
def _engine():
    from riskengine.engine import RiskEngine

    return RiskEngine()


def _records(df) -> list[dict]:
    return df.to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "replay_signals": _store("replay").count(), "live_signals": _store("live").count()}


@app.get("/signals")
def signals(
    feed: str = Query("replay", pattern="^(replay|live)$"),
    ticker: str | None = Query(None, description="Ticker, or MARKET for market-wide signals"),
    event_type: str | None = Query(None, description=" | ".join(EVENT_TYPES)),
    source: str | None = Query(None, pattern="^(news|social)$"),
    start: str | None = Query(None, description="Inclusive ISO date, e.g. 2020-03-01"),
    end: str | None = Query(None, description="Inclusive ISO date"),
    min_impact: float | None = Query(None, ge=1, le=10),
    limit: int = Query(100, ge=1, le=5000),
) -> list[dict]:
    """Structured risk signals, newest first."""
    df = _store(feed).query(ticker, event_type, source, start, end, min_impact, limit, newest_first=True)
    return _records(df)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, examples=["Boeing halts 737 Max production after second fatal crash"])
    source: str = Field("news", pattern="^(news|social)$")


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> list[dict]:
    """Score one piece of text on demand; returns one signal per company (or the market) it concerns."""
    from datetime import datetime, timezone

    from riskengine.engine import RiskEngine

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    shared = _engine()
    # Reuse the loaded models but not the stream state: ad-hoc requests must not count as market attention.
    engine = RiskEngine(shared.scorer, shared.impact)
    out = engine.process([Document(doc_id(req.source, ts, req.text), ts, req.source, "api", req.text)])
    if not out:
        raise HTTPException(422, "Text does not mention a covered company or a market-wide topic.")
    return [s.to_dict() for s in out]


@app.get("/index/weights")
def index_weights(date: str | None = Query(None, description="As-of ISO date; default is the latest")) -> dict:
    """Module A: target weights of the sentiment-tilted index."""
    sig = _store("replay").query()
    if sig.empty:
        raise HTTPException(404, "No signals in the store. Run scripts/run_pipeline.py restore first.")
    res = rebalancer.run(sig, load_prices())
    targets = res.targets if date is None else res.targets.loc[:date]
    if targets.empty:
        raise HTTPException(404, f"No weights on or before {date}.")
    row = targets.iloc[-1]
    return {
        "date": row.name.strftime("%Y-%m-%d"),
        "weights": {t: round(float(row[t]), 5) for t in INDEX_TICKERS},
        "sentiment": {t: round(float(res.state.loc[row.name, t]), 4) for t in INDEX_TICKERS},
    }


@app.get("/stress/triggers")
def stress_triggers() -> list[dict]:
    """Module B: stress tests triggered by high-impact systemic events in the signal stream."""
    return _records(stress.detect_triggers(_store("replay").query()))


@app.get("/stress/run")
def stress_run(
    event_type: str = Query(..., description=" | ".join(stress.SCENARIOS)),
    impact: float = Query(..., ge=1, le=10),
    focus_sector: str | None = None,
) -> dict:
    """Module B: apply the scenario for an event type at a given impact score to the synthetic portfolio."""
    if event_type not in stress.SCENARIOS:
        raise HTTPException(422, f"event_type must be one of {list(stress.SCENARIOS)}")
    scenario, _, summary = stress.run_stress(stress.load_portfolio(), event_type, impact, focus_sector)
    return {"scenario": scenario.__dict__, "result": summary}
