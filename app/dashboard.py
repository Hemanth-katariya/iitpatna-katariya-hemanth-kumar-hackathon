"""Streamlit dashboard. Run with: streamlit run app/dashboard.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import theme as T  # noqa: E402
from riskengine.config import BY_TICKER, DB_PATH, EVENT_TYPES, INDEX_TICKERS, LIVE_DB_PATH, MARKET, ROOT, SIGNALS_EXPORT  # noqa: E402
from riskengine.market import load_prices  # noqa: E402
from riskengine.modules import rebalancer, stress  # noqa: E402
from riskengine.store import SignalStore  # noqa: E402

st.set_page_config(page_title="AI/NLP Risk Engine", page_icon=None, layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2.2rem; max-width: 1400px;}
    [data-testid="stMetricValue"] {font-size: 1.7rem;}
    [data-testid="stMetric"] {background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10); border-radius: 8px; padding: 12px 16px;}
    </style>""",
    unsafe_allow_html=True,
)
CHART = dict(use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------------ data access


@st.cache_data(show_spinner="Loading signals...")
def load_signals(feed: str, _version: float) -> pd.DataFrame:
    path = LIVE_DB_PATH if feed == "Live" else DB_PATH
    store = SignalStore(path)
    if feed == "Replay" and store.count() == 0 and SIGNALS_EXPORT.exists():
        store.import_csv(SIGNALS_EXPORT)  # first run after cloning: restore the committed engine output
    df = store.query()
    df["date"] = pd.to_datetime(df["ts"].str[:10])
    return df


def signals_for(feed: str) -> pd.DataFrame:
    path = LIVE_DB_PATH if feed == "Live" else DB_PATH
    return load_signals(feed, path.stat().st_mtime if path.exists() else 0.0)


@st.cache_data
def prices() -> pd.DataFrame:
    return load_prices()


@st.cache_data
def portfolio() -> pd.DataFrame:
    return stress.load_portfolio()


@st.cache_data(show_spinner="Rebalancing index...")
def rebalance(_signals: pd.DataFrame, key: tuple, cfg: rebalancer.RebalanceConfig) -> rebalancer.RebalanceResult:
    return rebalancer.run(_signals, prices(), cfg)


@st.cache_resource(show_spinner="Loading NLP models (about a minute on first use)...")
def models():
    from riskengine.engine import RiskEngine

    return RiskEngine()


def metrics_file(name: str) -> dict | None:
    path = ROOT / "docs" / "metrics" / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


# ------------------------------------------------------------------ page: signals

IMPACT_BANDS = [(7, "High"), (5, "Elevated"), (0, "Routine")]


def band(impact: float) -> str:
    return next(label for lo, label in IMPACT_BANDS if impact >= lo)


def page_signals(sig: pd.DataFrame) -> None:
    st.caption("Every document from the news and social feeds becomes a structured signal: who it concerns, its sentiment, what kind of event it is, and how hard that event is likely to move prices.")
    f1, f2, f3, f4 = st.columns([1.2, 1.4, 1, 1])
    tickers = f1.multiselect("Company", [MARKET, *sorted(set(sig["ticker"]) - {MARKET})], placeholder="All companies and market")
    events = f2.multiselect("Event type", [e for e in EVENT_TYPES if e in set(sig["event_type"])], placeholder="All event types")
    source = f3.selectbox("Source", ["All", "news", "social"])
    min_impact = f4.slider("Minimum impact", 1.0, 10.0, 1.0, 0.5)
    view = sig[sig["impact"] >= min_impact]
    if tickers:
        view = view[view["ticker"].isin(tickers)]
    if events:
        view = view[view["event_type"].isin(events)]
    if source != "All":
        view = view[view["source"] == source]
    if view.empty:
        st.info("No signals match these filters.")
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Signals", f"{len(view):,}")
    k2.metric("From news", f"{(view['source'] == 'news').sum():,}")
    k3.metric("From social", f"{(view['source'] == 'social').sum():,}")
    k4.metric("High impact (7+)", f"{(view['impact'] >= 7).sum():,}")
    k5.metric("Mean sentiment", f"{view['sentiment'].mean():+.2f}")

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("**High-impact signals per day**")
        daily = view[view["impact"] >= 7].groupby("date").size().reindex(pd.date_range(view["date"].min(), view["date"].max()), fill_value=0)
        fig = go.Figure(go.Bar(x=daily.index, y=daily.values, marker_color=T.BLUE, hovertemplate="%{y} signals<extra></extra>"))
        st.plotly_chart(T.style(fig, 300, legend=False, y_title="Signals with impact 7+"), **CHART)
    with right:
        st.markdown("**Average impact by event type**")
        by_event = view.groupby("event_type")["impact"].agg(["mean", "size"]).sort_values("mean")
        fig = go.Figure(
            go.Bar(
                x=by_event["mean"], y=by_event.index, orientation="h", marker_color=T.BLUE, customdata=by_event["size"],
                hovertemplate="mean impact %{x:.2f}<br>%{customdata:,} signals<extra></extra>",
            )
        )  # fmt: skip
        fig = T.style(fig, 300, legend=False, x_title="Mean impact (1-10)")
        fig.update_layout(hovermode="closest", margin=dict(l=150))
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, **CHART)

    st.markdown("**Signal feed** (highest impact first)")
    top = view.sort_values(["impact", "ts"], ascending=[False, False]).head(300).reset_index(drop=True)
    table = top[["ts", "ticker", "source", "event_type", "sentiment", "impact", "text"]].copy()
    table.insert(6, "level", top["impact"].map(band))
    table["ts"] = table["ts"].str.replace("T", " ").str[:16]
    picked = st.dataframe(
        table,
        use_container_width=True,
        height=330,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "ts": "Time (UTC)",
            "ticker": "Ticker",
            "source": "Source",
            "event_type": "Event type",
            "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
            "impact": st.column_config.ProgressColumn("Impact", min_value=1, max_value=10, format="%.1f"),
            "level": "Level",
            "text": st.column_config.TextColumn("Text", width="large"),
        },
    )
    rows = picked.selection.rows if picked and picked.selection.rows else [0]
    explain(top.iloc[rows[0]])


def explain(row: pd.Series) -> None:
    """Show why one signal received its impact score."""
    st.markdown("**Why this impact score?** Select a row above to inspect another signal.")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown(f"> {row['text']}")
        st.markdown(
            f"**{row['ticker']}** · {row['sector']} · {row['source']} ({row['publisher']})  \n"
            f"Event type: **{row['event_type']}** (confidence {row['event_conf']:.0%})  \n"
            f"Sentiment: **{row['sentiment']:+.2f}** ({row['sentiment_label']}, confidence {row['sentiment_conf']:.0%})  \n"
            f"Attention: **{row['attention']:.1f}x** the usual daily volume  \n"
            f"Impact: **{row['impact']:.1f} / 10** ({band(row['impact'])})"
        )
    with c2:
        labels = {"baseline": "Baseline", "event_type": "Event type", "tone": "Tone of the text", "attention": "Attention burst", "source": "Source"}
        d = pd.Series(row["drivers"]).rename(labels)
        d = d[[v for v in labels.values() if v in d.index]][::-1]
        fig = go.Figure(
            go.Bar(
                x=d.values, y=d.index, orientation="h", marker_color=[T.BLUE if v >= 0 else T.RED for v in d.values],
                text=[f"{v:+.2f}" for v in d.values], textposition="outside", textfont=dict(color=T.INK_2),
                hovertemplate="%{y}: %{x:+.3f} sigma<extra></extra>",
            )
        )  # fmt: skip
        fig = T.style(fig, 240, legend=False, x_title=f"Contribution to the expected price move (total {d.sum():.2f} sigma)")
        span = max(abs(d.values).max(), 0.1)
        fig.update_layout(hovermode="closest", margin=dict(l=120, r=24))
        fig.update_xaxes(range=[min(d.values.min(), 0) - 0.18 * span, max(d.values.max(), 0) + 0.18 * span])
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, **CHART)


def page_analyze() -> None:
    st.caption("Type any headline or post. The same engine that produced the feed scores it on the spot.")
    examples = [
        "Boeing halts 737 Max production after regulators extend grounding",
        "Fed cuts interest rates to zero in emergency move as markets plunge",
        "Chesapeake Energy files for Chapter 11 bankruptcy protection after oil price collapse",
        "$TSLA crushing delivery estimates, record quarter incoming",
    ]
    choice = st.selectbox("Example", examples)
    text = st.text_area("Text to analyse", choice, height=90)
    source = st.radio("Treat as", ["news", "social"], horizontal=True)
    if not st.button("Analyse", type="primary"):
        return
    from riskengine.engine import RiskEngine
    from riskengine.schema import Document

    shared = models()
    engine = RiskEngine(shared.scorer, shared.impact)
    out = engine.process([Document("adhoc", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), source, "manual input", text)])
    if not out:
        st.warning("This text does not mention a covered company or a market-wide topic, so the engine emits no signal for it.")
        return
    for s in out:
        explain(pd.Series(s.to_dict()))
        if s.event_type in stress.SCENARIOS and s.impact >= stress.TRIGGER_IMPACT:
            st.error(f"This single signal clears the stress-test threshold ({s.event_type}, impact {s.impact:.1f}). A cluster of {stress.MIN_SIGNALS} such signals in a day triggers the '{stress.SCENARIOS[s.event_type].name}' scenario.")


# ------------------------------------------------------------------ page: Module A


def page_rebalancer(sig: pd.DataFrame) -> None:
    st.caption("A mock index of 18 S&P 100 stocks starts equal-weighted. Each day the engine's sentiment tilts weights toward stocks with better news flow than their peers, within position limits and a turnover budget.")
    with st.expander("Rebalancing rules", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        cfg = rebalancer.RebalanceConfig(
            tilt=c1.slider("Tilt strength", 0.0, 1.0, 0.35, 0.05, help="Weight multiplier is exp(tilt x sentiment z-score)."),
            half_life_days=c2.slider("Sentiment half-life (days)", 1.0, 10.0, 3.0, 0.5),
            max_daily_turnover=c3.slider("Daily turnover budget", 0.02, 0.50, 0.10, 0.01),
            cost_bps=c4.slider("Transaction cost (bps)", 0.0, 20.0, 5.0, 1.0),
        )
    res = rebalance(sig, (len(sig), str(sig["ts"].max())), cfg)
    m = res.metrics
    s, e, n = m["sentiment_tilted"], m["equal_weight_index"], m["naive_rebalancer"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sentiment index return", f"{s['total_return']:+.1%}", f"{s['excess_return']:+.1%} vs equal weight")
    k2.metric("Equal-weight index", f"{e['total_return']:+.1%}")
    k3.metric("S&P 500 (SPY)", f"{m['spy']['total_return']:+.1%}")
    k4.metric("Information ratio", f"{s['information_ratio']:.2f}", help="Annualised active return over tracking error, against the equal-weight index.")
    k5.metric("Avg daily turnover", f"{s['avg_daily_turnover']:.1%}", f"naive rule: {n['avg_daily_turnover']:.0%}", delta_color="off")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("**Growth of 100, net of costs**")
        curves = {
            "Sentiment-tilted index": (1 + res.strategy["ret"]).cumprod() * 100,
            "Equal-weight index": (1 + res.equal_weight).cumprod() * 100,
            "S&P 500 (SPY)": (1 + res.spy).cumprod() * 100,
        }
        fig = go.Figure()
        for (name, c), color in zip(curves.items(), T.SERIES):
            fig.add_trace(go.Scatter(x=c.index, y=c.values, name=name, mode="lines", line=dict(width=2, color=color), hovertemplate="%{y:.1f}"))
        # Direct-label the two ends that differ; the tilted and equal-weight lines finish too close to label both.
        for c in (curves["Sentiment-tilted index"], curves["S&P 500 (SPY)"]):
            fig.add_annotation(x=c.index[-1], y=c.iloc[-1], text=f"{c.iloc[-1]:.0f}", showarrow=False, xanchor="left", xshift=6, font=dict(color=T.INK_2, size=12))
        fig = T.style(fig, 360, y_title="Index level (start = 100)")
        fig.update_layout(margin=dict(r=48))
        st.plotly_chart(fig, **CHART)
    with right:
        st.markdown("**Index weight by sector over time**")
        held = res.strategy[list(INDEX_TICKERS)]
        by_sector = held.T.groupby(lambda t: BY_TICKER[t].sector).sum().T
        fig = go.Figure()
        for sector, color in T.SECTOR_COLOR.items():
            if sector in by_sector:
                fig.add_trace(
                    go.Scatter(
                        x=by_sector.index, y=by_sector[sector], name=sector, mode="lines", stackgroup="w",
                        line=dict(width=1.5, color=T.SURFACE), fillcolor=color, hovertemplate="%{y:.1%}",
                    )
                )  # fmt: skip
        fig = T.style(fig, 360, y_title="Share of index")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=1, yanchor="top", traceorder="reversed", font=dict(size=11)), margin=dict(t=16, r=8))
        st.plotly_chart(fig, **CHART)

    st.markdown("**How far each stock's weight sits from its equal weight**")
    base = 1 / len(INDEX_TICKERS)
    dev = (held / base - 1).T
    order = dev.mean(axis=1).sort_values().index
    lim = float(np.abs(dev.values).max())
    fig = go.Figure(
        go.Heatmap(
            z=dev.loc[order].values, x=dev.columns, y=order, colorscale=T.DIVERGING, zmin=-lim, zmax=lim, xgap=0, ygap=2,
            colorbar=dict(title=dict(text="vs equal weight", font=dict(size=11)), tickformat="+.0%", thickness=10, outlinewidth=0),
            hovertemplate="%{y} on %{x|%d %b %Y}<br>%{z:+.0%} vs equal weight<extra></extra>",
        )
    )  # fmt: skip
    fig = T.style(fig, 470, legend=False)
    fig.update_layout(hovermode="closest", margin=dict(l=64))
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, **CHART)

    left, right = st.columns([1, 1])
    with left:
        day = st.select_slider("Weights and sentiment on", options=list(held.index), value=held.index[-1], format_func=lambda d: d.strftime("%d %b %Y"))
        snap = pd.DataFrame({"weight": held.loc[day], "sentiment": res.state.loc[day]}).sort_values("weight")
        fig = go.Figure(
            go.Bar(
                x=snap["weight"], y=snap.index, orientation="h", marker_color=[T.BLUE if w >= base else T.GRAY for w in snap["weight"]],
                customdata=snap["sentiment"], hovertemplate="weight %{x:.2%}<br>sentiment state %{customdata:+.3f}<extra></extra>",
            )
        )  # fmt: skip
        fig.add_vline(x=base, line=dict(color=T.INK_2, width=1))
        fig.add_annotation(x=base, y=1.04, yref="paper", text="equal weight", showarrow=False, font=dict(color=T.INK_2, size=11))
        fig = T.style(fig, 470, legend=False, x_title="Index weight (blue = overweight)")
        fig.update_xaxes(tickformat=".0%", showgrid=True)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(hovermode="closest", margin=dict(l=64))
        st.plotly_chart(fig, **CHART)
    with right:
        st.markdown("**Backtest summary**")
        rows = {"Sentiment-tilted": s, "Naive rule": n, "Equal weight": e, "S&P 500": m["spy"]}
        summary = pd.DataFrame(rows).T[["total_return", "annualised_vol", "sharpe", "max_drawdown", "avg_daily_turnover"]]
        st.dataframe(
            summary.style.format({"total_return": "{:+.1%}", "annualised_vol": "{:.1%}", "sharpe": "{:.2f}", "max_drawdown": "{:.1%}", "avg_daily_turnover": "{:.1%}"}, na_rep="-"),
            use_container_width=True,
            column_config={"total_return": "Return", "annualised_vol": "Vol", "sharpe": "Sharpe", "max_drawdown": "Drawdown", "avg_daily_turnover": "Turnover"},
        )
        st.caption("Naive rule: weights follow raw same-day sentiment with no smoothing or limits.")
        q = m["signal_quality"]
        st.markdown(
            f"**Does sentiment predict returns?** Daily rank correlation between a stock's sentiment and the return it goes on to earn: "
            f"**{q['mean_ic']:+.3f}** on average (t-stat {q['ic_t_stat']:.2f}, positive on {q['ic_hit_rate']:.0%} of days)."
        )
        st.caption(
            f"Backtest over {len(held)} trading days. Sentiment seen on day t is traded at the close of t+1 and earns from t+2, "
            "because news is stamped by date only. One short window is not proof of a durable edge."
        )


# ------------------------------------------------------------------ page: Module B


def page_stress(sig: pd.DataFrame) -> None:
    st.caption("A synthetic $7bn wholesale banking book of loans, bonds and derivatives. When the engine sees a cluster of high-impact systemic events, the matching shock scenario is applied, scaled by the impact score.")
    port = portfolio()
    triggers = stress.detect_triggers(sig)
    mode = st.radio("Stress test", ["Triggered by the engine", "What-if"], horizontal=True, label_visibility="collapsed")

    if mode == "Triggered by the engine":
        if triggers.empty:
            st.info(f"No trigger yet: the engine has not seen {stress.MIN_SIGNALS} or more systemic signals with impact {stress.TRIGGER_IMPACT:.0f}+ on one day. Try the what-if mode.")
            return
        st.markdown(f"**{len(triggers)} stress tests triggered** (rule: {stress.MIN_SIGNALS}+ signals of a systemic event type with impact {stress.TRIGGER_IMPACT:.0f}+ on one day)")
        show = triggers.assign(headline=triggers["headlines"].str[0], severity=triggers["severity"] * 100, focus_sector=triggers["focus_sector"].fillna("Broad market")).drop(columns="headlines")
        picked = st.dataframe(
            show, use_container_width=True, hide_index=True, height=min(38 * len(show) + 40, 250), on_select="rerun", selection_mode="single-row",
            column_config={
                "date": "Date", "event_type": "Event type", "impact": st.column_config.NumberColumn("Impact", format="%.1f"),
                "severity": st.column_config.NumberColumn("Severity", format="%.0f%%", help="Share of the full scenario applied."),
                "n_signals": "Signals", "focus_sector": "Sector in focus", "headline": st.column_config.TextColumn("Lead headline", width="large"),
            },
        )  # fmt: skip
        row = triggers.iloc[picked.selection.rows[0] if picked and picked.selection.rows else len(triggers) - 1]
        event, impact, focus = row["event_type"], float(row["impact"]), row["focus_sector"]
        st.markdown(f"**Showing: {event} on {row['date']}** · impact {impact:.1f}" + (f" · concentrated in {focus}" if focus else ""))
        with st.expander("Headlines behind this trigger"):
            for h in row["headlines"]:
                st.markdown(f"- {h}")
    else:
        c1, c2, c3 = st.columns([1.2, 1.4, 1.2])
        event = c1.selectbox("Event type", list(stress.SCENARIOS))
        impact = c2.slider("Impact score", stress.TRIGGER_IMPACT, 10.0, 8.5, 0.1)
        focus = c3.selectbox("Sector in focus (optional)", ["None", *sorted(set(port["sector"]) - {"Sovereign"})])
        focus = None if focus == "None" else focus

    scenario, stressed, summary = stress.run_stress(port, event, impact, focus if isinstance(focus, str) else None)
    st.markdown(f"**Scenario: {scenario.name}.** {scenario.narrative}")
    shocks = {
        "Rates": f"{scenario.rates_bp:+.0f} bp", "IG spreads": f"{scenario.ig_spread_bp:+.0f} bp", "HY spreads": f"{scenario.hy_spread_bp:+.0f} bp",
        "Equities": f"{scenario.equity_pct:+.1f}%", "US dollar": f"{scenario.usd_pct:+.1f}%", "Oil": f"{scenario.oil_pct:+.1f}%",
        "Default rates": f"x{scenario.pd_multiplier:.2f}",
    }  # fmt: skip
    st.caption("Shocks applied: " + " · ".join(f"{k} {v}" for k, v in shocks.items()))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio value before", T.money(summary["value_before"]))
    k2.metric("Portfolio value after", T.money(summary["value_after"]), f"{summary['pnl_pct']:+.2%}")
    k3.metric("Stress loss", T.money(summary["pnl"]))
    k4.metric("Expected credit loss", T.money(summary["expected_loss_after"]), f"from {T.money(summary['expected_loss_before'])}", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.markdown("**Where the loss comes from** (by risk factor)")
        f = pd.Series(summary["by_risk_factor"]).sort_values(ascending=False)
        fig = go.Figure(
            go.Bar(
                x=f.values / 1e6, y=f.index, orientation="h", marker_color=[T.BLUE if v >= 0 else T.RED for v in f.values],
                text=[f"{v / 1e6:+,.0f}" for v in f.values], textposition="outside", textfont=dict(color=T.INK_2),
                hovertemplate="%{y}: %{x:+,.1f}m<extra></extra>",
            )
        )  # fmt: skip
        fig = T.style(fig, 320, legend=False, x_title="Profit or loss, $m (red = loss)")
        span = max(abs(f.values).max() / 1e6, 1.0)
        fig.update_layout(hovermode="closest", margin=dict(l=150, r=24))
        fig.update_yaxes(showgrid=False, autorange="reversed")
        fig.update_xaxes(showgrid=True, range=[min(f.values.min() / 1e6, 0) - 0.2 * span, max(f.values.max() / 1e6, 0) + 0.2 * span])
        st.plotly_chart(fig, **CHART)
    with right:
        st.markdown("**Value before and after, by asset class**")
        g = stressed.groupby("asset_class")[["market_value_usd", "stressed_value_usd", "pnl_total"]].sum() / 1e6
        g = g.loc[[a for a in ("Loan", "Bond", "Derivative") if a in g.index]]
        g.loc["Total"] = g.sum()
        g["pct"] = g["pnl_total"] / g["market_value_usd"].where(g["market_value_usd"].abs() > 50)
        st.dataframe(
            g.style.format({"market_value_usd": "{:,.0f}", "stressed_value_usd": "{:,.0f}", "pnl_total": "{:+,.1f}", "pct": "{:+.2%}"}, na_rep="n/a"),
            use_container_width=True,
            column_config={"asset_class": "Asset class", "market_value_usd": "Before $m", "stressed_value_usd": "After $m", "pnl_total": "Change $m", "pct": "Change %"},
        )
        st.caption("Loans lose value through higher expected credit loss; bonds and derivatives are marked to market. Derivatives carry little market value, so their change is shown in dollars only.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Loss by sector**")
        sec = pd.Series(summary["by_sector"]) / 1e6
        fig = go.Figure(
            go.Bar(
                x=sec.values, y=sec.index, orientation="h", marker_color=[T.BLUE if v >= 0 else T.RED for v in sec.values],
                hovertemplate="%{y}: %{x:+,.1f}m<extra></extra>",
            )
        )  # fmt: skip
        fig = T.style(fig, 400, legend=False, x_title="Profit or loss, $m")
        fig.update_layout(hovermode="closest", margin=dict(l=170))
        fig.update_yaxes(showgrid=False, autorange="reversed")
        fig.update_xaxes(showgrid=True)
        st.plotly_chart(fig, **CHART)
    with right:
        st.markdown("**Ten largest position losses**")
        worst = stressed.nsmallest(10, "pnl_total")[["pnl_total", "instrument", "sector", "rating", "position_id"]]
        worst = worst.assign(pnl_total=worst["pnl_total"] / 1e6)
        st.dataframe(
            worst, use_container_width=True, hide_index=True, height=400,
            column_config={
                "pnl_total": st.column_config.NumberColumn("Loss $m", format="%.1f"),
                "instrument": "Instrument", "sector": "Sector", "rating": "Rating", "position_id": "Position",
            },
        )  # fmt: skip
    st.caption("All positions and counterparties are synthetic. The stress model is a simplified sensitivity-based revaluation, not a regulatory stress test.")


# ------------------------------------------------------------------ page: validation


def page_validation() -> None:
    st.caption("Each model is measured against a naive alternative on data it did not train on.")
    sent, events, impact = metrics_file("sentiment"), metrics_file("events"), metrics_file("impact")
    c1, c2 = st.columns(2)
    if sent:
        with c1:
            key = next(k for k in sent if "out-of-sample" in k)
            st.markdown(f"**Sentiment: accuracy on {sent[key]['n']:,} unseen financial tweets**")
            _versus({"Lexicon baseline (VADER)": sent[key]["vader_baseline"]["accuracy"], "FinBERT": sent[key]["finbert"]["accuracy"]})
            st.caption("FinBERT's 97% on Financial PhraseBank is not shown as a headline: it was trained on that dataset.")
    if events:
        with c2:
            st.markdown(f"**Event classification: accuracy on {events['n_valid']:,} held-out headlines**")
            v = events["variants"]
            _versus({"Keyword rules": v["keyword_baseline"]["accuracy"], "TF-IDF only": v["tfidf_logreg"]["accuracy"], "Embeddings + TF-IDF": v[events["selected"]]["accuracy"]})
            st.caption("Credit Event is excluded here because its labels come from a rule, not from human annotation.")
    if impact:
        st.markdown("**Impact score: do higher scores precede bigger price moves?** (2019 headlines, not used in fitting)")
        c, day = impact["company"], impact["company"]["per_stock_day"]
        left, right = st.columns([1.2, 1])
        with left:
            bands = c["mean_sigma_move_by_score_band"]
            fig = go.Figure(go.Bar(x=list(bands), y=list(bands.values()), marker_color=T.BLUE, text=[f"{v:.2f}" for v in bands.values()], textposition="outside", textfont=dict(color=T.INK_2), hovertemplate="score %{x}: %{y:.2f} sigma<extra></extra>"))
            fig = T.style(fig, 300, legend=False, x_title="Impact score band", y_title="Mean abnormal move (sigma)")
            fig.update_xaxes(type="category")
            st.markdown(f"*Single-company signals* ({c['n_test']:,} signals)")
            st.plotly_chart(fig, **CHART)
        with right:
            st.markdown("*Per stock and day (the conservative view)*")
            a, b = st.columns(2)
            a.metric("Big move, impact above 7", f"{day['big_move_rate']['when_impact_above_7']:.0%}", help="Share of stock-days with an abnormal move above two sigma when the day's strongest signal scored above 7.")
            b.metric("Big move, any day", f"{day['big_move_rate']['base_rate']:.0%}")
            sp = c["spearman_vs_two_day_move"]
            st.markdown(
                f"Rank correlation with the realised move, per signal: **{sp['impact_model']:+.2f}** for the impact model, "
                f"**{sp['naive_abs_sentiment']:+.2f}** for sentiment strength alone. Per stock-day it is {day['spearman_vs_two_day_move']:+.2f}, "
                f"and {day['spearman_vs_next_day_move']:+.2f} against the next day's move only."
            )
        st.warning(
            "Market-wide impact is not validated. A separate fit on market-level headlines showed no relationship with S&P 500 moves in 2019 "
            f"(rank correlation {impact['market']['separate_market_fit_spearman_2019']:+.2f}), so market-wide text reuses the single-company coefficients. "
            "The public headline set is stock commentary, not a macro newswire."
        )


def _versus(values: dict[str, float]) -> None:
    names = list(values)
    fig = go.Figure(
        go.Bar(
            x=list(values.values()), y=names, orientation="h", marker_color=[T.GRAY] * (len(names) - 1) + [T.BLUE],
            text=[f"{v:.1%}" for v in values.values()], textposition="outside", textfont=dict(color=T.INK_2),
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )  # fmt: skip
    fig = T.style(fig, 90 + 46 * len(names), legend=False)
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", showgrid=True)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(hovermode="closest", margin=dict(l=170, r=24))
    st.plotly_chart(fig, **CHART)


# ------------------------------------------------------------------ layout


def main() -> None:
    st.title("AI/NLP Financial Risk Engine")
    with st.sidebar:
        st.header("Data feed")
        feed = st.radio("Feed", ["Replay", "Live"], label_visibility="collapsed", captions=["Jan-Jul 2020 news and tweets", "Google News, Yahoo Finance, StockTwits"])
        if feed == "Live" and st.button("Fetch live data now"):
            from riskengine.ingestion import fetch_all, live_sources

            with st.spinner("Pulling live feeds and scoring..."):
                docs = fetch_all(live_sources())
                n = SignalStore(LIVE_DB_PATH).write(models().process(docs))
            st.success(f"{len(docs):,} documents fetched, {n:,} new signals.")
        sig = signals_for(feed)
        if sig.empty:
            st.warning("No signals yet. " + ("Press 'Fetch live data now'." if feed == "Live" else "Run `python scripts/run_pipeline.py restore`."))
            st.stop()
        if feed == "Replay":
            days = sorted(sig["date"].unique())
            as_of = st.select_slider("Replay up to", options=days, value=days[-1], format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"), help="Everything on the dashboard uses only signals up to this date, as if it were today.")
            sig = sig[sig["date"] <= as_of]
        st.caption(f"{len(sig):,} signals · {sig['doc_id'].nunique():,} documents · {sig['date'].min():%d %b %Y} to {sig['date'].max():%d %b %Y}")

    tabs = st.tabs(["Risk signals", "Analyse a headline", "Module A: Index rebalancer", "Module B: Stress testing", "Model validation"])
    with tabs[0]:
        page_signals(sig)
    with tabs[1]:
        page_analyze()
    with tabs[2]:
        if feed == "Live":
            st.info("The rebalancer backtest needs a price history, so it runs on the replay feed.")
        else:
            page_rebalancer(sig)
    with tabs[3]:
        page_stress(sig)
    with tabs[4]:
        page_validation()


main()
