"""Sentiment scoring: FinBERT as the production model, a VADER lexicon scorer as the naive baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FINBERT = "ProsusAI/finbert"


@dataclass(frozen=True)
class Sentiment:
    score: float  # P(positive) - P(negative), in [-1, 1]
    label: str  # positive | negative | neutral
    confidence: float  # probability of the predicted label


class FinBertSentiment:
    """Batch scorer around ProsusAI/finbert, sized for short texts on CPU."""

    def __init__(self, model_name: str = FINBERT, batch_size: int = 64, max_length: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.labels = [id2label[i] for i in range(len(id2label))]
        self._pos, self._neg = self.labels.index("positive"), self.labels.index("negative")

    def probabilities(self, texts: list[str]) -> np.ndarray:
        """Class probabilities, one row per text, columns ordered as `self.labels`."""
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        # Sorting by length keeps padding (and so CPU time) per batch low.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with self._torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                enc = self.tokenizer(
                    [texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                )
                out[idx] = self._torch.softmax(self.model(**enc).logits, dim=-1).numpy()
        return out

    def score(self, texts: list[str]) -> list[Sentiment]:
        probs = self.probabilities(texts)
        top = probs.argmax(axis=1)
        return [
            Sentiment(float(p[self._pos] - p[self._neg]), self.labels[k], float(p[k])) for p, k in zip(probs, top)
        ]


class LexiconSentiment:
    """VADER compound score: the general-purpose lexicon baseline FinBERT is measured against."""

    def __init__(self, neutral_band: float = 0.05):
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            self._sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            self._sia = SentimentIntensityAnalyzer()
        self.neutral_band = neutral_band

    def score(self, texts: list[str]) -> list[Sentiment]:
        out = []
        for t in texts:
            s = self._sia.polarity_scores(t)["compound"]
            label = "positive" if s > self.neutral_band else "negative" if s < -self.neutral_band else "neutral"
            out.append(Sentiment(float(s), label, abs(float(s))))
        return out
