"""Impact score: an additive, market-calibrated estimate of how hard an event will move prices.

The raw score is the expected size of the abnormal price move over the event day and the next,
in units of the name's own daily volatility ("sigma move"):

    raw = baseline + w[event type] + w[source] + tone + attention

Every term is a fitted coefficient (see scripts/calibrate_impact.py), so each signal can report
exactly why it scored the way it did. The raw score is then mapped to 1-10 through fixed
quantile breakpoints taken from the calibration sample, which keeps "impact > 7" rare by design.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from riskengine.config import MARKET, MODELS

MODEL_PATH = MODELS / "impact_model.json"

# Share of calibration signals at or below each score 1..10: a score above 7 is the top ~8%.
SCORE_QUANTILES = (0.0, 0.30, 0.50, 0.65, 0.77, 0.86, 0.92, 0.96, 0.985, 0.998)
ATTENTION_CLIP = (-1.0, 3.0)


def log_attention(attention: float) -> float:
    return float(np.clip(math.log(max(attention, 1e-6)), *ATTENTION_CLIP))


@dataclass
class ScopeModel:
    """Coefficients for one scope: single-name signals or market-wide signals."""

    baseline: float
    event: dict[str, float]
    source: dict[str, float]
    negative_tone: float  # per unit of negative sentiment magnitude
    positive_tone: float  # per unit of positive sentiment magnitude
    attention: float  # per unit of log attention ratio
    breakpoints: list[float]  # raw scores that map to 1, 2, ..., 10

    def drivers(self, event_type: str, sentiment: float, attention: float, source: str) -> dict[str, float]:
        tone = self.negative_tone * max(-sentiment, 0.0) + self.positive_tone * max(sentiment, 0.0)
        return {
            "baseline": self.baseline,
            "event_type": self.event.get(event_type, 0.0),
            "tone": tone,
            "attention": self.attention * log_attention(attention),
            "source": self.source.get(source, 0.0),
        }

    def to_score(self, raw: float) -> float:
        return float(np.interp(raw, self.breakpoints, np.arange(1, 11)))


@dataclass
class ImpactModel:
    company: ScopeModel
    market: ScopeModel

    def scope(self, ticker: str) -> ScopeModel:
        return self.market if ticker == MARKET else self.company

    def score(self, ticker: str, event_type: str, sentiment: float, attention: float, source: str) -> tuple[float, dict[str, float]]:
        """Return (impact on the 1-10 scale, additive drivers in sigma-move units)."""
        scope = self.scope(ticker)
        drivers = scope.drivers(event_type, sentiment, attention, source)
        return round(scope.to_score(sum(drivers.values())), 2), {k: round(v, 4) for k, v in drivers.items()}

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ImpactModel":
        d = json.loads(path.read_text())
        return cls(company=ScopeModel(**d["company"]), market=ScopeModel(**d["market"]))
