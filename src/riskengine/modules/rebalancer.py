"""Module A: tactical index rebalancing driven by the engine's sentiment scores.

Pipeline
  signals -> daily sentiment per stock (impact- and confidence-weighted, shrunk when thin)
          -> exponentially smoothed sentiment state
          -> cross-sectional z-score -> tilt on equal weights
          -> position limits and a daily turnover budget
          -> backtest with a one-day execution lag and transaction costs
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskengine.config import BENCHMARK, INDEX_TICKERS

TRADING_DAYS = 252


@dataclass(frozen=True)
class RebalanceConfig:
    half_life_days: float = 3.0  # memory of the sentiment state
    shrinkage_docs: float = 3.0  # a day needs about this many documents before its mean is trusted
    social_weight: float = 0.5  # a tweet counts for less than a news headline
    tilt: float = 0.35  # weight multiplier is exp(tilt * z-score)
    min_multiple: float = 0.4  # floor and cap on a stock's weight, as multiples of its base weight
    max_multiple: float = 2.0
    max_daily_turnover: float = 0.10  # one-way turnover budget per day
    cost_bps: float = 5.0  # transaction cost per unit of turnover
    execution_lag: int = 1  # trading days between observing sentiment and holding the new weights


def daily_sentiment(signals: pd.DataFrame, trading_days: pd.DatetimeIndex, cfg: RebalanceConfig) -> pd.DataFrame:
    """Sentiment per (trading day, ticker). Weekend and holiday signals roll into the next trading day."""
    s = signals[signals["ticker"].isin(INDEX_TICKERS)].copy()
    day = pd.to_datetime(s["ts"].str[:10])
    pos = trading_days.searchsorted(day.to_numpy())
    s = s[pos < len(trading_days)]
    s["day"] = trading_days[pos[pos < len(trading_days)]]
    s["w"] = s["sentiment_conf"] * s["impact"] * np.where(s["source"] == "social", cfg.social_weight, 1.0)
    s["ws"] = s["w"] * s["sentiment"]
    g = s.groupby(["day", "ticker"])[["w", "ws"]].sum()
    mean_w = s["w"].mean()
    # Shrink toward neutral: a day with one tweet should not move the index like a day with fifty.
    score = g["ws"] / (g["w"] + cfg.shrinkage_docs * mean_w)
    return score.unstack("ticker").reindex(index=trading_days, columns=list(INDEX_TICKERS))


def sentiment_state(daily: pd.DataFrame, cfg: RebalanceConfig) -> pd.DataFrame:
    """Exponentially weighted sentiment; days without news decay the state toward neutral."""
    alpha = 1 - 0.5 ** (1 / cfg.half_life_days)
    return daily.fillna(0.0).ewm(alpha=alpha, adjust=False).mean()


def target_weights(state: pd.DataFrame, cfg: RebalanceConfig) -> pd.DataFrame:
    """Tilt equal weights by each stock's sentiment relative to the rest of the index."""
    base = 1.0 / state.shape[1]
    z = state.sub(state.mean(axis=1), axis=0).div(state.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    raw = base * np.exp(cfg.tilt * z.clip(-3, 3))
    return raw.apply(lambda row: _bounded_normalise(row.to_numpy(), base * cfg.min_multiple, base * cfg.max_multiple), axis=1, result_type="broadcast")


def _bounded_normalise(w: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale weights to sum to one while respecting [lo, hi] per position."""
    w = np.clip(w, lo, hi)
    for _ in range(50):
        free = (w > lo + 1e-12) & (w < hi - 1e-12)
        gap = 1.0 - w.sum()
        if abs(gap) < 1e-10 or not free.any():
            break
        w[free] = np.clip(w[free] + gap * w[free] / w[free].sum(), lo, hi)
    return w / w.sum()


def naive_weights(daily: pd.DataFrame) -> pd.DataFrame:
    """Naive rebalancer for comparison: raw same-day sentiment, no smoothing, no limits."""
    raw = (1.0 + daily.fillna(0.0)).clip(lower=0.05)
    return raw.div(raw.sum(axis=1), axis=0)


def backtest(targets: pd.DataFrame, prices: pd.DataFrame, cfg: RebalanceConfig, max_turnover: float | None = None) -> pd.DataFrame:
    """Simulate holding `targets`, traded `execution_lag` days after they are known.

    Returns one row per day: portfolio return net of costs, turnover, and the held weights.
    """
    tickers = list(targets.columns)
    returns = prices[tickers].pct_change().reindex(targets.index).fillna(0.0)
    desired = targets.shift(cfg.execution_lag)
    held = np.full(len(tickers), 1.0 / len(tickers))
    rows = []
    for day in targets.index:
        r = returns.loc[day].to_numpy()
        gross = float(held @ r)
        held = held * (1 + r) / (1 + gross)  # weights drift with the day's returns
        turnover = 0.0
        if not desired.loc[day].isna().any():
            trade = desired.loc[day].to_numpy() - held
            one_way = np.abs(trade).sum() / 2
            if max_turnover is not None and one_way > max_turnover:
                trade *= max_turnover / one_way
                one_way = max_turnover
            held, turnover = held + trade, one_way
        rows.append([gross - turnover * 2 * cfg.cost_bps / 1e4, turnover, *held])
    return pd.DataFrame(rows, index=targets.index, columns=["ret", "turnover", *tickers])


def performance(ret: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    curve = (1 + ret).cumprod()
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    out = {
        "total_return": float(curve.iloc[-1] - 1),
        "annualised_vol": float(vol),
        "sharpe": float(ret.mean() * TRADING_DAYS / vol) if vol else 0.0,
        "max_drawdown": float((curve / curve.cummax() - 1).min()),
    }
    if benchmark is not None:
        active = ret - benchmark
        te = active.std() * np.sqrt(TRADING_DAYS)
        out["excess_return"] = float(curve.iloc[-1] - (1 + benchmark).cumprod().iloc[-1])
        out["tracking_error"] = float(te)
        out["information_ratio"] = float(active.mean() * TRADING_DAYS / te) if te else 0.0
    return out


def information_coefficient(state: pd.DataFrame, prices: pd.DataFrame, horizon: int = 1, lag: int = 1) -> pd.Series:
    """Daily rank correlation between the sentiment state and the return it would actually earn.

    With a one-day execution lag, sentiment seen on day t is traded at the close of t+1 and
    first earns the return of t+2, so that is the return it is compared with.
    """
    fwd = prices[list(state.columns)].pct_change(horizon).shift(-horizon - lag).reindex(state.index)
    return state.corrwith(fwd, axis=1, method="spearman").dropna()


@dataclass
class RebalanceResult:
    daily: pd.DataFrame  # raw daily sentiment
    state: pd.DataFrame  # smoothed sentiment
    targets: pd.DataFrame  # target weights
    strategy: pd.DataFrame  # backtest of the sentiment-tilted index
    naive: pd.DataFrame  # backtest of the naive rebalancer
    equal_weight: pd.Series  # daily returns of the untilted index
    spy: pd.Series
    metrics: dict[str, dict[str, float]]


def run(signals: pd.DataFrame, prices: pd.DataFrame, cfg: RebalanceConfig = RebalanceConfig(), start: str | None = None, end: str | None = None) -> RebalanceResult:
    days = prices.index[(prices.index >= (start or signals["ts"].min()[:10])) & (prices.index <= (end or signals["ts"].max()[:10]))]
    daily = daily_sentiment(signals, days, cfg)
    state = sentiment_state(daily, cfg)
    targets = target_weights(state, cfg)

    strategy = backtest(targets, prices, cfg, cfg.max_daily_turnover)
    naive = backtest(naive_weights(daily), prices, cfg)
    base = pd.DataFrame(1.0 / len(INDEX_TICKERS), index=days, columns=list(INDEX_TICKERS))
    equal_bt = backtest(base, prices, cfg)
    equal = equal_bt["ret"]
    spy = prices[BENCHMARK].pct_change().reindex(days).fillna(0.0)

    ic = information_coefficient(state, prices)
    # Same-day check: does a day's raw sentiment line up with that day's return? (descriptive, not tradable)
    same_day = daily.corrwith(prices[list(INDEX_TICKERS)].pct_change().reindex(days), axis=1, method="spearman").dropna()
    metrics = {
        "sentiment_tilted": {**performance(strategy["ret"], equal), "avg_daily_turnover": float(strategy["turnover"].mean())},
        "naive_rebalancer": {**performance(naive["ret"], equal), "avg_daily_turnover": float(naive["turnover"].mean())},
        "equal_weight_index": {**performance(equal), "avg_daily_turnover": float(equal_bt["turnover"].mean())},
        "spy": performance(spy),
        "signal_quality": {
            "mean_ic": float(ic.mean()),
            "ic_t_stat": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 1 else 0.0,
            "ic_hit_rate": float((ic > 0).mean()),
            "same_day_ic": float(same_day.mean()),
            "same_day_ic_t_stat": float(same_day.mean() / same_day.std() * np.sqrt(len(same_day))) if len(same_day) > 1 else 0.0,
        },
    }
    return RebalanceResult(daily, state, targets, strategy, naive, equal, spy, metrics)
