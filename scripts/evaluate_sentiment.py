"""Benchmark FinBERT against the VADER lexicon baseline; writes docs/metrics/sentiment.json.

Two public test sets:
  * twitter-financial-news-sentiment (validation split): unseen by FinBERT, so this is the fair test.
  * Financial PhraseBank (all-agree): FinBERT was fine-tuned on PhraseBank, so its score there is
    an in-sample upper bound and is reported only for reference.
"""
from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from riskengine.config import RAW, ROOT
from riskengine.nlp.sentiment import FinBertSentiment, LexiconSentiment
from riskengine.nlp.text import clean


def load_twitter() -> pd.DataFrame:
    df = pd.read_csv(RAW / "sent_valid.csv")
    df["label"] = df["label"].map({0: "negative", 1: "positive", 2: "neutral"})  # bearish / bullish / neutral
    df["text"] = df["text"].map(clean)
    return df


def load_phrasebank() -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "FinancialPhraseBank-v1.0.zip") as z:
        raw = z.read("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt").decode("latin-1")
    rows = [line.rsplit("@", 1) for line in io.StringIO(raw).read().splitlines() if "@" in line]
    return pd.DataFrame(rows, columns=["text", "label"])


def evaluate(model, df: pd.DataFrame) -> dict:
    pred = [s.label for s in model.score(df.text.tolist())]
    return {
        "accuracy": round(accuracy_score(df.label, pred), 4),
        "macro_f1": round(f1_score(df.label, pred, average="macro"), 4),
    }


def main() -> None:
    models = {"vader_baseline": LexiconSentiment(), "finbert": FinBertSentiment()}
    sets = {
        "twitter_financial_news (out-of-sample)": load_twitter(),
        "financial_phrasebank_allagree (in-sample for FinBERT)": load_phrasebank(),
    }
    results = {}
    for set_name, df in sets.items():
        results[set_name] = {"n": len(df), **{m: evaluate(model, df) for m, model in models.items()}}
        print(set_name, json.dumps(results[set_name]))
    out = ROOT / "docs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sentiment.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
