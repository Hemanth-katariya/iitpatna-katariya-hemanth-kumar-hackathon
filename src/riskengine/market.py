"""Price utilities: returns, market-adjusted (abnormal) returns and standardised event moves."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.config import BENCHMARK, DATA, MARKET

BETA_WINDOW = 120
VOL_WINDOW = 60


def load_prices(path: Path | str = DATA / "prices.csv") -> pd.DataFrame:
    """Wide frame of adjusted closes: index = trading date, columns = tickers."""
    long = pd.read_csv(path, parse_dates=["date"])
    return long.pivot(index="date", columns="ticker", values="close").sort_index()


def abnormal_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily return minus beta times the benchmark return; beta is a trailing estimate known the day before."""
    r = prices.pct_change()
    m = r[BENCHMARK]
    beta = (r.rolling(BETA_WINDOW).cov(m)).div(m.rolling(BETA_WINDOW).var(), axis=0).shift(1)
    ar = r.sub(beta.mul(m, axis=0))
    ar[BENCHMARK] = m  # for the market itself the "abnormal" return is just its return
    return ar


def sigma_moves(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardised event-window moves, in units of each name's trailing daily volatility.

    Returns (two_day, next_day): the absolute cumulative abnormal return over [t, t+1] scaled by
    sqrt(2) sigma, and the absolute abnormal return on t+1 alone. News here is stamped by date
    only, so the two-day window is the materiality measure and t+1 the strictly predictive one.
    """
    ar = abnormal_returns(prices)
    vol = ar.rolling(VOL_WINDOW).std().shift(1)
    two_day = (ar + ar.shift(-1)).abs() / (vol * np.sqrt(2))
    next_day = ar.shift(-1).abs() / vol
    rename = {BENCHMARK: MARKET}
    return two_day.rename(columns=rename), next_day.rename(columns=rename)


def event_day(trading_days: pd.DatetimeIndex, dates: pd.Series) -> pd.Series:
    """First trading day on or after each date (NaT if beyond the price history)."""
    pos = trading_days.searchsorted(pd.to_datetime(dates).to_numpy())
    padded = trading_days.append(pd.DatetimeIndex([pd.NaT]))
    return pd.Series(padded[np.minimum(pos, len(trading_days))], index=dates.index)
