"""Turn the raw public downloads into the small, committed sample in data/.

Outputs
  data/news.csv               headlines mentioning an index member or the market (replay window)
  data/tweets.csv             de-duplicated, spam-filtered tweets for the same universe, capped per day
  data/prices.csv             daily adjusted closes for the index, the benchmark and market proxies
  data/raw/calibration_news.csv  2015-2019 company headlines used only to calibrate the impact score
  data/raw/prices_full.csv       price history backing that calibration
"""
from __future__ import annotations

import re
import sys

import pandas as pd
import yfinance as yf

from riskengine.config import ALL_TICKERS, BENCHMARK, DATA, INDEX_TICKERS, MARKET, RAW, REPLAY_END, REPLAY_START
from riskengine.nlp.entities import link
from riskengine.nlp.text import cashtag_count, clean, dedup_key, doc_id, is_boilerplate

TWEETS_PER_TICKER_DAY = 25
MAX_CASHTAGS = 4  # tweets listing many tickers are almost always promotional spam
MAX_MARKET_CALIBRATION = 30_000
MARKET_PROXIES = ["^VIX", "TLT", "HYG", "USO", "GLD", "UUP"]
SEED = 7


def _wanted(tickers: list[str]) -> bool:
    return any(t in INDEX_TICKERS or t == MARKET for t in tickers)


# In this mirror every GuruFocus headline has its digits 0, 1 and 2 replaced by the
# characters "…", "–" and "—" ("COVID-–9", "$5…… Million"). Undo that where the character
# sits next to a digit, another garbled digit, "$" or "-", which leaves real dashes alone.
_GARBLED = re.compile(r"(?<=[\d$…–—\-/])[…–—]|[…–—](?=[\d…–—])")
_DIGIT = {"…": "0", "–": "1", "—": "2"}


def repair_digits(text: str) -> str:
    return _GARBLED.sub(lambda m: _DIGIT[m.group()], text)


def load_news() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "financial_news.parquet", columns=["headline", "url", "publisher", "date"])
    df["ts"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts", "headline"])
    df = df[df["ts"] >= "2015-01-01"]
    guru = df["publisher"] == "GuruFocus"
    df.loc[guru, "headline"] = df.loc[guru, "headline"].map(repair_digits)
    df["text"] = df["headline"].map(clean)
    df = df[(df["text"].str.len() >= 20) & ~df["text"].map(is_boilerplate)]
    df["key"] = df["text"].map(dedup_key)
    return df.sort_values("ts").drop_duplicates("key")


def build_news(news: pd.DataFrame) -> pd.DataFrame:
    df = news[(news.ts >= REPLAY_START) & (news.ts <= pd.Timestamp(REPLAY_END, tz="UTC") + pd.Timedelta(days=1))].copy()
    df = df[df["text"].map(lambda t: _wanted(link(t)))]
    df["source"] = "news"
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["doc_id"] = [doc_id("news", ts, tx) for ts, tx in zip(df.ts, df.text)]
    df["weight"] = 1.0
    return df[["doc_id", "ts", "source", "publisher", "text", "url", "weight"]]


def build_calibration_news(news: pd.DataFrame) -> pd.DataFrame:
    df = news[(news.ts >= "2015-01-01") & (news.ts < REPLAY_START)].copy()
    df["tickers"] = df["text"].map(link)
    df = df[df["tickers"].map(len).between(1, 2)]  # round-ups naming many companies carry no single-name signal
    df["tickers"] = df["tickers"].map(" ".join)
    market = df["tickers"] == MARKET
    df = pd.concat([df[~market], df[market].sample(min(int(market.sum()), MAX_MARKET_CALIBRATION), random_state=SEED)])
    df = df.sort_values("ts")
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%d")
    return df[["ts", "publisher", "text", "tickers"]]


def build_tweets() -> pd.DataFrame:
    df = pd.read_csv(RAW / "stock_tweets_2020.csv", lineterminator="\n", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df["ts"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts", "text"])
    df["text"] = df["text"].map(clean)
    df = df[(df["text"].str.len() >= 25) & (df["text"].map(cashtag_count) <= MAX_CASHTAGS)]
    df["key"] = df["text"].map(dedup_key)
    df = df.sort_values("ts").drop_duplicates("key")  # collapses retweets onto the earliest copy
    df["tickers"] = df["text"].map(lambda t: link(t, "social"))
    df = df[df["tickers"].map(_wanted)]

    # Cap volume per (first ticker, day) so that mega-caps do not swamp the sample.
    df["bucket"] = df["tickers"].map(lambda ts: next(t for t in ts if t in INDEX_TICKERS or t == MARKET))
    df["day"] = df["ts"].dt.date
    df = df.sample(frac=1.0, random_state=SEED)
    group = df.groupby(["bucket", "day"])
    # `weight` records how many raw tweets each kept tweet stands for, so attention is not lost to the cap.
    df["weight"] = (group["text"].transform("size") / TWEETS_PER_TICKER_DAY).clip(lower=1.0).round(2)
    df = df[group.cumcount() < TWEETS_PER_TICKER_DAY].sort_values("ts")
    df["source"] = "social"
    df["publisher"] = "twitter"
    df["url"] = ""
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["doc_id"] = [doc_id("social", ts, tx) for ts, tx in zip(df.ts, df.text)]
    return df[["doc_id", "ts", "source", "publisher", "text", "url", "weight"]]


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"].dropna(how="all")
    long = close.stack().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    long["close"] = long["close"].round(4)
    return long.sort_values(["ticker", "date"])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    news = load_news()

    sample = build_news(news)
    sample.to_csv(DATA / "news.csv", index=False)
    print(f"news.csv               {len(sample):>7,} rows  {sample.ts.min()} .. {sample.ts.max()}")

    calib = build_calibration_news(news)
    calib.to_csv(RAW / "calibration_news.csv", index=False)
    print(f"calibration_news.csv   {len(calib):>7,} rows  {calib.ts.min()} .. {calib.ts.max()}")

    tweets = build_tweets()
    tweets.to_csv(DATA / "tweets.csv", index=False)
    print(f"tweets.csv             {len(tweets):>7,} rows  {tweets.ts.min()} .. {tweets.ts.max()}")

    if "--skip-prices" in sys.argv and (RAW / "prices_full.csv").exists():
        return
    full = fetch_prices([*ALL_TICKERS, BENCHMARK, *MARKET_PROXIES], "2014-12-01", "2020-08-01")
    full.to_csv(RAW / "prices_full.csv", index=False)
    keep = full[full.ticker.isin([*INDEX_TICKERS, BENCHMARK, *MARKET_PROXIES]) & (full.date >= "2019-07-01")]
    keep.to_csv(DATA / "prices.csv", index=False)
    print(f"prices.csv             {len(keep):>7,} rows  tickers={keep.ticker.nunique()}  (full: {len(full):,})")


if __name__ == "__main__":
    main()
