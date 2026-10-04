import numpy as np
import pandas as pd
import pytest

from riskengine.config import INDEX_TICKERS, MARKET
from riskengine.engine import AttentionTracker
from riskengine.modules import rebalancer, stress
from riskengine.nlp.impact import ImpactModel, ScopeModel
from riskengine.schema import Signal
from riskengine.store import SignalStore


def _signal(**kw) -> Signal:
    base = dict(
        doc_id="d1", ts="2020-03-02T10:00:00Z", source="news", publisher="p", ticker="AAPL", sector="Information Technology",
        text="Apple cuts guidance", sentiment=-0.8, sentiment_label="negative", sentiment_conf=0.9,
        event_type="Earnings", event_conf=0.8, impact=7.5, attention=2.0, drivers={"baseline": 1.0},
    )  # fmt: skip
    return Signal(**{**base, **kw})


# ---------------------------------------------------------------- store


def test_store_roundtrip_filters_and_csv(tmp_path):
    store = SignalStore(tmp_path / "s.db")
    store.write([_signal(), _signal(doc_id="d2", ticker="MSFT", impact=3.0, ts="2020-03-05T00:00:00Z")])
    store.write([_signal()])  # same (doc_id, ticker) replaces, not duplicates
    assert store.count() == 2
    assert store.query(min_impact=7)["ticker"].tolist() == ["AAPL"]
    assert store.query(start="2020-03-03", end="2020-03-05")["ticker"].tolist() == ["MSFT"]
    assert store.query(ticker="AAPL").iloc[0]["drivers"] == {"baseline": 1.0}

    store.export_csv(tmp_path / "out.csv.gz")
    other = SignalStore(tmp_path / "t.db")
    assert other.import_csv(tmp_path / "out.csv.gz") == 2
    assert other.query(ticker="AAPL").iloc[0]["drivers"] == {"baseline": 1.0}


# ---------------------------------------------------------------- attention and impact


def test_attention_is_neutral_until_history_exists_then_detects_bursts():
    tracker = AttentionTracker(window=20, min_history=5)
    key = ("AAPL", "news")
    for _ in range(5):
        assert tracker.update({key: 4.0})[key] == 1.0
    assert tracker.update({key: 4.0})[key] == pytest.approx(1.0)
    assert tracker.update({key: 24.0})[key] == pytest.approx(5.0)


def test_new_source_does_not_look_like_a_burst():
    tracker = AttentionTracker(min_history=5)
    for _ in range(10):
        tracker.update({("AAPL", "news"): 4.0})
    assert tracker.update({("AAPL", "news"): 4.0, ("AAPL", "social"): 500.0})[("AAPL", "social")] == 1.0


def test_feed_outage_is_not_mistaken_for_silence():
    tracker = AttentionTracker(min_history=5)
    both = {("AAPL", "news"): 4.0, ("AAPL", "social"): 30.0, ("MSFT", "social"): 10.0}
    for _ in range(8):
        tracker.update(both)
    for _ in range(15):  # the social feed goes down; news keeps flowing
        tracker.update({("AAPL", "news"): 4.0})
    back = tracker.update(both)
    assert back[("AAPL", "social")] == pytest.approx(1.0)


def test_social_attention_uses_share_of_voice_not_raw_volume():
    tracker = AttentionTracker(min_history=5)
    day = {("AAPL", "social"): 30.0, ("MSFT", "social"): 10.0}
    for _ in range(6):
        tracker.update(day)
    # The collector captures ten times more posts for every ticker: no one's share changed.
    flood = tracker.update({k: n * 10 for k, n in day.items()})
    assert flood[("AAPL", "social")] == pytest.approx(1.0)
    # AAPL takes most of an ordinary day's volume: that is attention.
    shift = tracker.update({("AAPL", "social"): 39.0, ("MSFT", "social"): 1.0})
    assert shift[("AAPL", "social")] > 1.2 and shift[("MSFT", "social")] < 0.5


def _scope() -> ScopeModel:
    return ScopeModel(
        baseline=1.0, event={"Credit Event": 0.5}, source={"news": 0.0, "social": -0.1},
        negative_tone=0.4, positive_tone=0.1, attention=0.3, breakpoints=[0.8 + 0.2 * i for i in range(10)],
    )  # fmt: skip


def test_impact_drivers_sum_to_raw_score_and_stay_in_range():
    model = ImpactModel(company=_scope(), market=_scope())
    score, drivers = model.score("AAPL", "Credit Event", -1.0, np.e, "news")
    assert sum(drivers.values()) == pytest.approx(1.0 + 0.5 + 0.4 + 0.3)
    assert 1.0 <= score <= 10.0
    calm, _ = model.score("AAPL", "Market Commentary", 0.0, 1.0, "social")
    assert calm < score


# ---------------------------------------------------------------- Module A


def _prices(days: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-02", periods=days)
    rng = np.random.default_rng(0)
    cols = [*INDEX_TICKERS, "SPY"]
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (days, len(cols))), axis=0), index=idx, columns=cols)


def _signals_frame(prices: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for day in prices.index[:-2]:
        for t in INDEX_TICKERS:
            rows.append(_signal(doc_id=f"{day}{t}", ts=day.strftime("%Y-%m-%dT12:00:00Z"), ticker=t, sentiment=float(rng.uniform(-1, 1)), impact=5.0).to_dict())
    return pd.DataFrame(rows)


def test_target_weights_respect_limits_and_sum_to_one():
    cfg = rebalancer.RebalanceConfig()
    prices = _prices()
    res = rebalancer.run(_signals_frame(prices), prices, cfg)
    base = 1 / len(INDEX_TICKERS)
    assert np.allclose(res.targets.sum(axis=1), 1.0)
    assert res.targets.min().min() >= base * cfg.min_multiple - 1e-9
    assert res.targets.max().max() <= base * cfg.max_multiple + 1e-9
    assert res.strategy["turnover"].max() <= cfg.max_daily_turnover + 1e-9
    assert np.allclose(res.strategy[list(INDEX_TICKERS)].sum(axis=1), 1.0)


def test_positive_sentiment_raises_weight_and_negative_lowers_it():
    cfg = rebalancer.RebalanceConfig()
    state = pd.DataFrame([np.zeros(len(INDEX_TICKERS))], columns=list(INDEX_TICKERS))
    state.iloc[0, 0], state.iloc[0, 1] = 0.6, -0.6
    w = rebalancer.target_weights(state, cfg).iloc[0]
    base = 1 / len(INDEX_TICKERS)
    assert w.iloc[0] > base > w.iloc[1]


def test_backtest_trades_only_after_the_execution_lag():
    cfg = rebalancer.RebalanceConfig(cost_bps=0.0)
    prices = _prices(10)
    tickers = list(INDEX_TICKERS)
    targets = pd.DataFrame(1 / len(tickers), index=prices.index, columns=tickers)
    targets.iloc[3:] = 0.0
    targets.iloc[3:, 0] = 1.0  # from day 3 the target is 100% in the first stock
    bt = rebalancer.backtest(targets, prices, cfg)
    r = prices[tickers].pct_change().fillna(0.0)
    assert bt["turnover"].iloc[3] < 0.05  # day 3: the new target is known but not yet tradable
    assert bt[tickers[0]].iloc[4] == pytest.approx(1.0)  # traded at the close of day 4
    assert bt["ret"].iloc[5] == pytest.approx(r[tickers[0]].iloc[5])  # and earns from day 5


# ---------------------------------------------------------------- Module B


def test_scenario_scaling_and_severity_mapping():
    assert stress.severity_from_impact(7.0) == pytest.approx(stress.MIN_SEVERITY)
    assert stress.severity_from_impact(10.0) == pytest.approx(1.0)
    half = stress.SCENARIOS["Credit Event"].scaled(0.5)
    assert half.hy_spread_bp == pytest.approx(300)
    assert half.pd_multiplier == pytest.approx(1.75)


def test_stress_attribution_adds_up_and_harder_events_lose_more():
    p = stress.load_portfolio()
    _, stressed, mild = stress.run_stress(p, "Credit Event", 7.0)
    _, _, severe = stress.run_stress(p, "Credit Event", 10.0)
    assert sum(mild["by_risk_factor"].values()) == pytest.approx(mild["pnl"])
    assert sum(mild["by_sector"].values()) == pytest.approx(mild["pnl"])
    assert severe["pnl"] < mild["pnl"] < 0
    assert (stressed["stressed_pd"] <= 1).all() and (stressed["stressed_pd"] >= stressed["pd_1y"]).all()


def test_focus_sector_concentrates_credit_losses():
    p = stress.load_portfolio()
    _, a, _ = stress.run_stress(p, "Credit Event", 9.0)
    _, b, _ = stress.run_stress(p, "Credit Event", 9.0, focus_sector="Energy")
    energy = p["sector"] == "Energy"
    assert b.loc[energy, "pnl_credit_loss"].sum() < a.loc[energy, "pnl_credit_loss"].sum()
    assert b.loc[~energy, "pnl_total"].sum() == pytest.approx(a.loc[~energy, "pnl_total"].sum())


def test_triggers_need_a_cluster_of_systemic_high_impact_signals():
    def rows(n, event, impact, day="2020-03-09", ticker=MARKET):
        return [_signal(doc_id=f"{event}{day}{i}", ts=f"{day}T00:00:00Z", ticker=ticker, sector="Market", event_type=event, impact=impact).to_dict() for i in range(n)]

    df = pd.DataFrame(
        rows(4, "Geopolitical", 8.5)  # fires
        + rows(2, "Macroeconomic", 9.0)  # too few signals
        + rows(5, "Earnings", 9.5)  # not a systemic event type
        + rows(5, "Credit Event", 6.0)  # below the impact threshold
        + rows(4, "Geopolitical", 9.0, day="2020-03-11")  # inside the cooldown window
        + rows(4, "Geopolitical", 9.0, day="2020-03-20")  # fires again
    )
    trig = stress.detect_triggers(df)
    assert trig[["date", "event_type"]].values.tolist() == [["2020-03-09", "Geopolitical"], ["2020-03-20", "Geopolitical"]]
    assert trig.iloc[0]["impact"] == pytest.approx(8.5)
    assert trig.iloc[0]["focus_sector"] is None


def test_triggers_ignore_good_news_and_unsure_labels_and_detect_sell_offs():
    def rows(n, tag, **kw):
        return [_signal(doc_id=f"{tag}{i}", ts="2020-03-09T00:00:00Z", ticker=MARKET, sector="Market", impact=8.0, **kw).to_dict() for i in range(n)]

    df = pd.DataFrame(
        rows(4, "good", event_type="Commodity/Energy", sentiment=0.7)  # oil rebounding is not a stress event
        + rows(4, "unsure", event_type="Macroeconomic", event_conf=0.4)
        + rows(4, "selloff", event_type="Market Commentary", sentiment=-0.9)
    )
    trig = stress.detect_triggers(df)
    assert trig["event_type"].tolist() == [stress.SELL_OFF]
    assert stress.SELL_OFF in stress.SCENARIOS
