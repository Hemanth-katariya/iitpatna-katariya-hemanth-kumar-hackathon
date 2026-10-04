"""Download the public raw datasets into data/raw/ (not committed; see README 'Dataset Used')."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# local filename -> (Hugging Face dataset repo, path inside repo)
FILES = {
    "topic_train.csv": ("zeroshot/twitter-financial-news-topic", "topic_train.csv"),
    "topic_valid.csv": ("zeroshot/twitter-financial-news-topic", "topic_valid.csv"),
    "sent_train.csv": ("zeroshot/twitter-financial-news-sentiment", "sent_train.csv"),
    "sent_valid.csv": ("zeroshot/twitter-financial-news-sentiment", "sent_valid.csv"),
    "FinancialPhraseBank-v1.0.zip": ("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip"),
    "financial_news.parquet": ("ashraq/financial-news", "data/train-00000-of-00001-8ec327f23bbe0948.parquet"),
    "stock_tweets_2020.csv": ("StephanAkkerman/stock-market-tweets-data", "stock-market-tweets-data.csv"),
    "stock_tweets_2015_2019.csv": ("mjw/stock_market_tweets", "stock_market_tweets.csv"),
}


def download(name: str, repo: str, path: str, retries: int = 30) -> None:
    dest = RAW / name
    if dest.exists():
        print(f"skip   {name} (already downloaded)", flush=True)
        return
    url = HF.format(repo=repo, path=path)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        try:
            # Large files drop mid-stream on slow links, so resume from the bytes already on disk.
            with requests.get(url, stream=True, timeout=120, headers={"Range": f"bytes={done}-"}) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            break
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            print(f"retry  {name} attempt {attempt} after {type(e).__name__}", flush=True)
            time.sleep(min(5 * attempt, 30))
    else:
        raise RuntimeError(f"could not download {name} after {retries} attempts")
    tmp.replace(dest)
    print(f"saved  {name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:] or list(FILES)
    for name in wanted:
        download(name, *FILES[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
