"""SQLite signal store: the hand-off point between the engine and downstream modules."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from riskengine.config import DB_PATH
from riskengine.schema import Signal

_COLUMNS = (
    "doc_id", "ts", "source", "publisher", "ticker", "sector", "text", "sentiment", "sentiment_label",
    "sentiment_conf", "event_type", "event_conf", "impact", "attention", "drivers",
)  # fmt: skip

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    doc_id TEXT NOT NULL, ts TEXT NOT NULL, source TEXT NOT NULL, publisher TEXT, ticker TEXT NOT NULL,
    sector TEXT, text TEXT, sentiment REAL, sentiment_label TEXT, sentiment_conf REAL,
    event_type TEXT, event_conf REAL, impact REAL, attention REAL, drivers TEXT,
    PRIMARY KEY (doc_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals (ts);
CREATE INDEX IF NOT EXISTS idx_signals_ticker_ts ON signals (ticker, ts);
"""


class SignalStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def write(self, signals: Iterable[Signal]) -> int:
        rows = [tuple(json.dumps(v) if k == "drivers" else v for k, v in s.to_dict().items()) for s in signals]
        with self._connect() as con:
            con.executemany(f"INSERT OR REPLACE INTO signals VALUES ({','.join('?' * len(_COLUMNS))})", rows)
        return len(rows)

    def clear(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM signals")

    def query(
        self,
        ticker: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
        start: str | None = None,
        end: str | None = None,
        min_impact: float | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> pd.DataFrame:
        """Signals as a DataFrame; `start`/`end` are inclusive ISO date or datetime prefixes."""
        where, params = [], []
        for clause, value in (
            ("ticker = ?", ticker),
            ("event_type = ?", event_type),
            ("source = ?", source),
            ("ts >= ?", start),
            ("ts <= ?", end + "￿" if end else None),
            ("impact >= ?", min_impact),
        ):
            if value is not None:
                where.append(clause)
                params.append(value)
        sql = "SELECT * FROM signals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts " + ("DESC" if newest_first else "ASC")
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as con:
            df = pd.read_sql_query(sql, con, params=params)
        df["drivers"] = df["drivers"].map(json.loads)
        return df

    def count(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
