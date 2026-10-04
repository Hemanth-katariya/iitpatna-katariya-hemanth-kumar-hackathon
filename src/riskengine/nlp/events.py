"""Event classification: sentence embeddings + TF-IDF into a linear model, with a keyword baseline."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from riskengine.config import EVENT_TYPES, MODELS

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = MODELS / "event_classifier.joblib"

# Maps the 20 topics of the public `twitter-financial-news-topic` dataset onto our taxonomy.
TOPIC_TO_EVENT: dict[int, str] = {
    0: "Analyst Rating",  # Analyst Update
    1: "Macroeconomic",  # Fed | Central Banks
    2: "Product/Company News",  # Company | Product News
    3: "Macroeconomic",  # Treasuries | Corporate Debt
    4: "Earnings",  # Dividend
    5: "Earnings",  # Earnings
    6: "Commodity/Energy",  # Energy | Oil
    7: "Earnings",  # Financials (reported results)
    8: "Macroeconomic",  # Currencies
    9: "Market Commentary",  # General News | Opinion
    10: "Commodity/Energy",  # Gold | Metals | Materials
    11: "Merger/Acquisition",  # IPO
    12: "Legal/Regulatory",  # Legal | Regulation
    13: "Merger/Acquisition",  # M&A | Investments
    14: "Macroeconomic",  # Macro
    15: "Market Commentary",  # Markets
    16: "Geopolitical",  # Politics
    17: "Management Change",  # Personnel Change
    18: "Market Commentary",  # Stock Commentary
    19: "Market Commentary",  # Stock Movement
}

# High-precision credit vocabulary. The public topic data has no credit-event class, so this
# pattern both weak-labels training examples and overrides the model at inference time.
CREDIT_PATTERN = re.compile(
    r"\b(?:bankrupt\w*|chapter (?:11|7)|insolven\w*|defaults? on|defaulted|bond default|debt default|credit default"
    r"|missed (?:an? )?(?:interest|bond|debt|coupon|loan) payments?|debt restructuring|restructur\w+ (?:its |their )?debt"
    r"|credit rating|rating (?:cut|downgrade)|cut(?:s)? (?:\w+ )?to junk|junk status|downgrad\w+ (?:\w+ ){0,3}(?:to )?junk"
    r"|(?:moody's|fitch|s&p global|s&p) (?:cuts|downgrades|lowers|slashes)|going concern|liquidity crisis"
    r"|distressed debt|creditors?|covenant breach|forbearance)\b",
    re.IGNORECASE,
)

# Naive keyword classifier, first match wins. This is the baseline the trained model must beat.
_KEYWORD_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Credit Event", CREDIT_PATTERN),
    (
        "Merger/Acquisition",
        re.compile(r"\b(?:acquir\w+|acquisition|merger|merge|takeover|buyout|to buy|stake|ipo|spin-?off|bid for)\b", re.I),
    ),
    (
        "Analyst Rating",
        re.compile(
            r"\b(?:upgrad\w+|downgrad\w+|price target|initiat\w+|reiterat\w+|analysts?|outperform|underperform"
            r"|overweight|underweight|buy rating|sell rating)\b",
            re.I,
        ),
    ),
    (
        "Earnings",
        re.compile(
            r"\b(?:earnings|eps|revenue|quarter(?:ly)?|q[1-4]|profit|guidance|dividend|beats|misses|results)\b", re.I
        ),
    ),
    (
        "Legal/Regulatory",
        re.compile(
            r"\b(?:lawsuit|sues?|sued|court|probe|investigation|fine[sd]?|regulators?|antitrust|settle\w*|sec|ftc|doj|fda)\b",
            re.I,
        ),
    ),
    (
        "Management Change",
        re.compile(r"\b(?:ceo|cfo|chief executive|steps? down|resign\w*|appoint\w*|names|hires|successor)\b", re.I),
    ),
    (
        "Geopolitical",
        re.compile(
            r"\b(?:sanctions?|tariffs?|trade war|trade deal|invasion|war|military|missiles?|geopolitic\w*|nato|putin"
            r"|ukraine|russia\w*|taiwan|north korea|iran\w*|brexit|election|white house|congress|senate|trump|biden)\b",
            re.I,
        ),
    ),
    (
        "Commodity/Energy",
        re.compile(r"\b(?:oil|crude|opec|natural gas|gold|copper|brent|wti|metals?|silver)\b", re.I),
    ),
    (
        "Macroeconomic",
        re.compile(
            r"\b(?:fed|federal reserve|inflation|gdp|interest rates?|central bank|jobs|unemployment|economy|treasury"
            r"|yields?|dollar|currency|recession|ecb)\b",
            re.I,
        ),
    ),
    (
        "Product/Company News",
        re.compile(r"\b(?:launch\w*|unveil\w*|releas\w+|introduc\w+|rolls? out|partnership|announces?|new)\b", re.I),
    ),
]


def keyword_classify(texts: list[str]) -> list[str]:
    out = []
    for t in texts:
        out.append(next((label for label, pat in _KEYWORD_RULES if pat.search(t)), "Market Commentary"))
    return out


@dataclass(frozen=True)
class Event:
    label: str
    confidence: float


class EventClassifier:
    """Linear classifier over [sentence embedding | TF-IDF] features, plus the credit-event override."""

    def __init__(self, path: Path = MODEL_PATH):
        import joblib
        from sentence_transformers import SentenceTransformer

        art = joblib.load(path)
        self.clf, self.tfidf, self.labels = art["clf"], art["tfidf"], list(art["labels"])
        self.encoder = SentenceTransformer(art["encoder"], device="cpu")
        assert set(self.labels) <= set(EVENT_TYPES)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    def probabilities(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict_proba(features(self.embed(texts), texts, self.tfidf))

    def classify(self, texts: list[str]) -> list[Event]:
        probs = self.probabilities(texts)
        out = []
        for t, p in zip(texts, probs):
            k = int(p.argmax())
            if CREDIT_PATTERN.search(t):
                out.append(Event("Credit Event", max(0.9, float(p[self.labels.index("Credit Event")]))))
            else:
                out.append(Event(self.labels[k], float(p[k])))
        return out


def features(embeddings: np.ndarray, texts: list[str], tfidf):
    """Stack dense embeddings with sparse TF-IDF (if the artifact has a vectoriser)."""
    if tfidf is None:
        return embeddings
    from scipy.sparse import csr_matrix, hstack

    return hstack([csr_matrix(embeddings), tfidf.transform(texts)]).tocsr()
