"""Run the risk engine.

  python scripts/run_pipeline.py restore            load the committed engine output into the signal store (seconds)
  python scripts/run_pipeline.py replay [--export]  recompute every signal from data/news.csv and data/tweets.csv
  python scripts/run_pipeline.py live               pull the live feeds once and score them
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from itertools import groupby

from riskengine.config import DB_PATH, LIVE_DB_PATH, REPLAY_END, REPLAY_START, SIGNALS_EXPORT
from riskengine.ingestion import fetch_all, live_sources, replay_sources
from riskengine.store import SignalStore


def restore() -> None:
    store = SignalStore(DB_PATH)
    store.clear()
    print(f"restored {store.import_csv(SIGNALS_EXPORT):,} signals from {SIGNALS_EXPORT.name} into {DB_PATH.name}")


def replay(start: str, end: str, export: bool) -> None:
    from riskengine.engine import RiskEngine

    docs = fetch_all(replay_sources(start, end))
    print(f"replaying {len(docs):,} documents from {start} to {end}")
    engine, store = RiskEngine(), SignalStore(DB_PATH)
    store.clear()
    t0, done, written = time.time(), 0, 0
    # Feed the engine one calendar day at a time, as a live deployment would.
    for day, batch in groupby(docs, key=lambda d: d.ts[:10]):
        batch = list(batch)
        written += store.write(engine.process(batch))
        done += len(batch)
        print(f"{day}  docs {done:>6,}/{len(docs):,}  signals {written:>6,}  {done / (time.time() - t0):5.1f} docs/s", flush=True)
    if export:
        print(f"exported {store.export_csv(SIGNALS_EXPORT):,} signals to {SIGNALS_EXPORT.name}")


def live() -> None:
    from riskengine.engine import FixedAttention, RiskEngine

    docs = fetch_all(live_sources())
    print(f"fetched {len(docs):,} live documents: {dict(Counter(d.source for d in docs))}")
    signals = RiskEngine(attention=FixedAttention()).process(docs)
    store = SignalStore(LIVE_DB_PATH)
    store.clear()  # each run is a fresh snapshot of the feeds
    store.write(signals)
    print(f"wrote {len(signals):,} signals to {LIVE_DB_PATH.name} (store now holds {store.count():,})")
    for s in sorted(signals, key=lambda s: -s.impact)[:8]:
        print(f"  impact {s.impact:4.1f}  sent {s.sentiment:+.2f}  {s.ticker:6s} {s.event_type:22s} {s.text[:90]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["restore", "replay", "live"])
    ap.add_argument("--start", default=REPLAY_START)
    ap.add_argument("--end", default=REPLAY_END)
    ap.add_argument("--export", action="store_true", help="after a replay, rewrite data/signals.csv.gz")
    args = ap.parse_args()
    {"restore": restore, "replay": lambda: replay(args.start, args.end, args.export), "live": live}[args.mode]()


if __name__ == "__main__":
    main()
