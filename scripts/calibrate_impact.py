"""Calibrate the impact score against realised price reactions.

For ~35k company and market headlines from 2015-2019 this script
  1. scores each headline with the production sentiment and event models,
  2. measures the abnormal price move around it, in units of the name's own volatility,
  3. fits the additive impact model on 2015-2018 and tests it out-of-time on 2019.

Writes models/impact_model.json and docs/metrics/impact.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from riskengine.config import EVENT_TYPES, MARKET, RAW, ROOT
from riskengine.engine import AttentionTracker
from riskengine.market import event_day, load_prices, sigma_moves
from riskengine.nlp.impact import SCORE_QUANTILES, ImpactModel, ScopeModel, log_attention
from riskengine.nlp.scoring import TextScorer
from riskengine.nlp.text import clean

TEST_START = "2019-01-01"
TARGET_CLIP = 6.0
REFERENCE_EVENT = "Market Commentary"
BIG_MOVE = 2.0  # a two-sigma abnormal move


def score_headlines() -> pd.DataFrame:
    """Run the NLP models over the calibration headlines.

    This is the slow step on CPU; run `scripts/prescore.py calibration i/n` first to fill the cache in parallel.
    """
    df = pd.read_csv(RAW / "calibration_news.csv")
    scores = TextScorer().score(df["text"].map(clean).tolist())
    df["sentiment"] = [s.score for s, _ in scores]
    df["event_type"] = [e.label for _, e in scores]
    df["source"] = "news"
    return df


def add_attention(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the same attention ratio the engine computes online."""
    tracker, ratios = AttentionTracker(), {}
    counts = df.groupby(["ts", "ticker", "source"]).size()
    for day, day_counts in counts.groupby(level="ts"):
        day_dict = {(t, s): float(n) for (_, t, s), n in day_counts.items()}
        for key, ratio in tracker.update(day_dict).items():
            ratios[(day, *key)] = ratio
    df["attention"] = [ratios[k] for k in zip(df.ts, df.ticker, df.source)]
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    prices = load_prices(RAW / "prices_full.csv")
    two_day, next_day = sigma_moves(prices)
    df["event_day"] = event_day(prices.index, df["ts"])
    df = df.dropna(subset=["event_day"])
    for name, frame in (("sigma_move", two_day), ("sigma_next", next_day)):
        long = frame.stack().rename(name)
        df = df.join(long, on=["event_day", "ticker"])
    return df.dropna(subset=["sigma_move", "sigma_next"])


def design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    events = [e for e in EVENT_TYPES if e != REFERENCE_EVENT]
    cols = [(df["event_type"] == e).to_numpy(float) for e in events]
    cols += [
        np.maximum(-df["sentiment"].to_numpy(), 0.0),
        np.maximum(df["sentiment"].to_numpy(), 0.0),
        df["attention"].map(log_attention).to_numpy(),
    ]
    return np.column_stack(cols), [*events, "negative_tone", "positive_tone", "attention"]


def fit_scope(train: pd.DataFrame) -> ScopeModel:
    x, names = design(train)
    reg = Ridge(alpha=1.0).fit(x, train["sigma_move"].clip(upper=TARGET_CLIP))
    coef = dict(zip(names, reg.coef_))
    event = {e: float(coef.get(e, 0.0)) for e in EVENT_TYPES}
    raw = reg.predict(x)
    breakpoints = np.maximum.accumulate(np.quantile(raw, SCORE_QUANTILES) + np.arange(10) * 1e-9)
    return ScopeModel(
        baseline=float(reg.intercept_),
        event=event,
        source={"news": 0.0, "social": 0.0},
        negative_tone=float(coef["negative_tone"]),
        positive_tone=float(coef["positive_tone"]),
        attention=float(coef["attention"]),
        breakpoints=[float(b) for b in breakpoints],
    )


def raw_scores(scope: ScopeModel, df: pd.DataFrame) -> np.ndarray:
    return np.array(
        [sum(scope.drivers(e, s, a, src).values()) for e, s, a, src in zip(df.event_type, df.sentiment, df.attention, df.source)]
    )


def evaluate(scope: ScopeModel, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    raw = raw_scores(scope, test)
    score = np.array([scope.to_score(r) for r in raw])
    y, y_next = test["sigma_move"].to_numpy(), test["sigma_next"].to_numpy()
    naive = test["sentiment"].abs().to_numpy()  # baseline: impact = strength of sentiment
    gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=7)
    gbm.fit(design(train)[0], train["sigma_move"].clip(upper=TARGET_CLIP))
    gbm_raw = gbm.predict(design(test)[0])

    order = np.argsort(raw)
    top, bottom = order[-len(order) // 10 :], order[: len(order) // 2]
    high = score > 7
    rho = lambda a, b: round(float(spearmanr(a, b).statistic), 4)  # noqa: E731
    return {
        "n_test": int(len(test)),
        "spearman_vs_two_day_move": {"impact_model": rho(raw, y), "naive_abs_sentiment": rho(naive, y), "gradient_boosting": rho(gbm_raw, y)},
        "spearman_vs_next_day_move": {"impact_model": rho(raw, y_next), "naive_abs_sentiment": rho(naive, y_next)},
        "mean_sigma_move": {
            "all": round(float(y.mean()), 3),
            "top_decile_by_impact": round(float(y[top].mean()), 3),
            "bottom_half_by_impact": round(float(y[bottom].mean()), 3),
        },
        "share_scored_above_7": round(float(high.mean()), 4),
        "big_move_rate": {
            "base_rate": round(float((y > BIG_MOVE).mean()), 4),
            "when_impact_above_7": round(float((y[high] > BIG_MOVE).mean()), 4) if high.any() else None,
        },
        "mean_sigma_move_by_score_band": {
            band: round(float(y[(score >= lo) & (score < hi)].mean()), 3)
            for band, lo, hi in (("1-3", 1, 3), ("3-5", 3, 5), ("5-7", 5, 7), ("7-10", 7, 10.01))
            if ((score >= lo) & (score < hi)).any()
        },
    }


def main() -> None:
    df = score_headlines()
    df["ticker"] = df["tickers"].str.split()
    df = df.explode("ticker")
    df = add_targets(add_attention(df))
    print(f"calibration signals with price targets: {len(df):,}")

    scopes, metrics = {}, {}
    for name, part in (("company", df[df.ticker != MARKET]), ("market", df[df.ticker == MARKET])):
        train, test = part[part.ts < TEST_START], part[part.ts >= TEST_START]
        scopes[name] = fit_scope(train)
        metrics[name] = {"n_train": int(len(train)), **evaluate(scopes[name], train, test)}
        print(f"\n[{name}] " + json.dumps(metrics[name], indent=2))
        print(json.dumps({k: round(v, 3) for k, v in scopes[name].event.items()}), f"neg={scopes[name].negative_tone:.3f} pos={scopes[name].positive_tone:.3f} att={scopes[name].attention:.3f} base={scopes[name].baseline:.3f}")

    ImpactModel(**scopes).save()
    out = ROOT / "docs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "impact.json").write_text(
        json.dumps(
            {
                "target": "abs. cumulative abnormal return over [t, t+1] / (sqrt(2) x trailing 60d volatility)",
                "train": "2015-2018 headlines",
                "test": "2019 headlines (out-of-time)",
                **metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
