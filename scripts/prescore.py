"""Score texts into the on-disk cache, in parallel shards, checkpointing as it goes.

  python scripts/prescore.py replay 0/3        # shard 0 of 3 of data/news.csv + data/tweets.csv
  python scripts/prescore.py calibration 1/3   # shard 1 of 3 of data/raw/calibration_news.csv

BERT inference on CPU stops scaling at about four threads, so several four-thread shards
finish sooner than one process using every core. Re-running a shard resumes where it stopped.
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import torch

from riskengine.config import DATA, RAW
from riskengine.nlp.scoring import CACHE_DIR, TextScorer
from riskengine.nlp.text import clean

CHUNK = 512
THREADS = 4


def texts_for(dataset: str) -> list[str]:
    if dataset == "calibration":
        frames = [pd.read_csv(RAW / "calibration_news.csv")]
    else:
        frames = [pd.read_csv(DATA / "news.csv"), pd.read_csv(DATA / "tweets.csv")]
    texts = pd.concat(frames)["text"].map(clean)
    return sorted(set(texts))


def main() -> None:
    dataset, shard = sys.argv[1], sys.argv[2]
    i, n = map(int, shard.split("/"))
    torch.set_num_threads(THREADS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{dataset}_{i}of{n}.parquet"

    scorer = TextScorer()  # loads every existing shard, so already-scored texts cost nothing
    todo = texts_for(dataset)[i::n]
    t0 = time.time()
    for start in range(0, len(todo), CHUNK):
        scorer.score(todo[start : start + CHUNK])
        scorer.rows(todo[: start + CHUNK]).to_parquet(out, index=False)
        done = min(start + CHUNK, len(todo))
        print(f"[{dataset} {shard}] {done:,}/{len(todo):,}  {done / (time.time() - t0):.1f} docs/s", flush=True)


if __name__ == "__main__":
    main()
