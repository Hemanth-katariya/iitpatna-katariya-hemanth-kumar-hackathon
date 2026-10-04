"""Train and evaluate the event classifier; writes models/event_classifier.joblib and docs/metrics/events.json.

Labels come from the public `twitter-financial-news-topic` dataset (MIT), remapped to our taxonomy.
That dataset has no credit-event class, so Credit Event examples are weak-labelled with a
high-precision regex, on the topic data and on pre-2020 news headlines. Credit Event metrics are
therefore measured against rule-derived labels and are reported separately as such.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from riskengine.config import EVENT_TYPES, MODELS, RAW, REPLAY_START, ROOT
from riskengine.nlp.events import CREDIT_PATTERN, ENCODER, MODEL_PATH, TOPIC_TO_EVENT, features, keyword_classify
from riskengine.nlp.text import clean, dedup_key

SEED = 7
N_CREDIT_NEWS = 1500


def load_topic(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name)
    df["text"] = df["text"].map(clean)
    df["label"] = df["label"].map(TOPIC_TO_EVENT)
    df.loc[df["text"].str.contains(CREDIT_PATTERN), "label"] = "Credit Event"
    return df[df["text"].str.len() >= 15].drop_duplicates("text")


def load_credit_news() -> pd.DataFrame:
    """Pre-2020 headlines that match the credit pattern: out-of-domain examples of the weak class."""
    news = pd.read_parquet(RAW / "financial_news.parquet", columns=["headline", "date"])
    news = news[news["date"] < REPLAY_START]
    news["text"] = news["headline"].map(clean)
    hit = news[news["text"].str.contains(CREDIT_PATTERN)].copy()
    hit["key"] = hit["text"].map(dedup_key)
    hit = hit.drop_duplicates("key")
    hit = hit.sample(min(len(hit), N_CREDIT_NEWS), random_state=SEED)
    hit["label"] = "Credit Event"
    return hit[["text", "label"]]


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
    }


def main() -> None:
    train, valid = load_topic("topic_train.csv"), load_topic("topic_valid.csv")
    credit = load_credit_news()
    cut = int(len(credit) * 0.8)
    train = pd.concat([train[["text", "label"]], credit.iloc[:cut]], ignore_index=True)
    valid = pd.concat([valid[["text", "label"]], credit.iloc[cut:]], ignore_index=True)
    valid = valid[~valid["text"].isin(set(train["text"]))]
    print(f"train={len(train):,} valid={len(valid):,}")
    print(train["label"].value_counts().to_string())

    encoder = SentenceTransformer(ENCODER, device="cpu")
    emb = lambda texts: encoder.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)  # noqa: E731
    e_train, e_valid = emb(train.text.tolist()), emb(valid.text.tolist())
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60_000).fit(train.text)

    variants = {
        "tfidf_logreg": (None, tfidf),
        "embedding_logreg": (e_train, None),
        "embedding_plus_tfidf_logreg": (e_train, tfidf),
    }
    y_train, y_valid = train.label.to_numpy(), valid.label.to_numpy()
    # Credit Event labels are regex-derived, so headline numbers exclude that class.
    organic = y_valid != "Credit Event"
    results: dict[str, dict] = {"keyword_baseline": scores(y_valid[organic], np.array(keyword_classify(valid.text.tolist()))[organic])}
    fitted = {}
    for name, (dense, vec) in variants.items():
        if dense is None:
            x_tr, x_va = vec.transform(train.text), vec.transform(valid.text)
        else:
            x_tr = features(dense, train.text.tolist(), vec)
            x_va = features(e_valid, valid.text.tolist(), vec)
        clf = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced", random_state=SEED).fit(x_tr, y_train)
        pred = clf.predict(x_va)
        results[name] = scores(y_valid[organic], pred[organic])
        fitted[name] = (clf, vec, pred)
        print(f"{name:32s} {results[name]}")
    print(f"{'keyword_baseline':32s} {results['keyword_baseline']}")

    deployable = {k: v for k, v in results.items() if k not in ("keyword_baseline", "tfidf_logreg")}
    best = max(deployable, key=lambda k: deployable[k]["macro_f1"])
    clf, vec, pred = fitted[best]
    report = classification_report(y_valid, pred, output_dict=True, zero_division=0)
    print(f"\nselected: {best}\n" + classification_report(y_valid, pred, zero_division=0))

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"clf": clf, "tfidf": vec, "labels": list(clf.classes_), "encoder": ENCODER}, MODEL_PATH, compress=3)
    out = ROOT / "docs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(
        json.dumps(
            {
                "dataset": "zeroshot/twitter-financial-news-topic (validation split), remapped to 11 event types",
                "n_train": len(train),
                "n_valid": int(organic.sum()),
                "note": "Headline metrics exclude Credit Event, whose labels are regex-derived (weak supervision).",
                "variants": results,
                "selected": best,
                "per_class": {k: {m: round(v, 4) for m, v in report[k].items()} for k in EVENT_TYPES if k in report},
            },
            indent=2,
        )
    )
    print(f"saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
