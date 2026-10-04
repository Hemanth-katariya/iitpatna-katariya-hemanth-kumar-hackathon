"""Smoke test: the dashboard renders every tab from the committed engine output without raising."""
import pytest

from riskengine.config import SIGNALS_EXPORT

pytestmark = pytest.mark.skipif(not SIGNALS_EXPORT.exists(), reason="needs data/signals.csv.gz")


def test_dashboard_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/dashboard.py", default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) >= 10  # KPI tiles on the signal, rebalancer and stress tabs
    assert at.title[0].value == "AI/NLP Financial Risk Engine"
