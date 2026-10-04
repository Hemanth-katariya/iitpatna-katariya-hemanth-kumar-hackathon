"""API tests against the committed engine output (no NLP models are loaded)."""
import pytest
from fastapi.testclient import TestClient

from riskengine import api
from riskengine.config import SIGNALS_EXPORT
from riskengine.store import SignalStore

pytestmark = pytest.mark.skipif(not SIGNALS_EXPORT.exists(), reason="needs data/signals.csv.gz")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("api") / "signals.db"
    SignalStore(db).import_csv(SIGNALS_EXPORT)
    original = api._store
    api._store = lambda feed: SignalStore(db)
    yield TestClient(api.app)
    api._store = original


def test_health_reports_signal_count(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["replay_signals"] > 10_000


def test_signals_are_structured_and_filterable(client):
    rows = client.get("/signals", params={"ticker": "AAPL", "min_impact": 7, "limit": 5}).json()
    assert 0 < len(rows) <= 5
    for r in rows:
        assert r["ticker"] == "AAPL" and r["impact"] >= 7
        assert -1 <= r["sentiment"] <= 1 and r["event_type"] and isinstance(r["drivers"], dict)
    assert client.get("/signals", params={"source": "telegram"}).status_code == 422


def test_index_weights_sum_to_one(client):
    body = client.get("/index/weights", params={"date": "2020-03-31"}).json()
    assert body["date"] <= "2020-03-31"
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-3)


def test_stress_endpoints(client):
    triggers = client.get("/stress/triggers").json()
    assert triggers and {"date", "event_type", "impact", "severity"} <= set(triggers[0])
    run = client.get("/stress/run", params={"event_type": "Credit Event", "impact": 9}).json()
    assert run["result"]["value_after"] < run["result"]["value_before"]
    assert client.get("/stress/run", params={"event_type": "Alien Invasion", "impact": 9}).status_code == 422
