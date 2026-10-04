"""Module B: event-driven stress testing of a synthetic wholesale banking portfolio.

The engine's Event Classification and Impact Score decide *when* to stress and *how hard*:
  1. `detect_triggers` watches the signal stream for clusters of high-impact systemic events.
  2. Each event type maps to a shock scenario (rates, credit spreads, equity, FX, oil, defaults),
     scaled by the impact score.
  3. `apply_scenario` revalues every position through its risk sensitivities.

Trading-book style positions (bonds, derivatives) take a mark-to-market hit through their
sensitivities; banking-book loans take theirs through higher expected credit loss (PD and LGD).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from riskengine.config import DATA, MARKET

TRIGGER_IMPACT = 7.0  # the brief's example threshold
MIN_SIGNALS = 3  # distinct high-impact documents needed on a day: one headline is not an event
COOLDOWN_DAYS = 5  # do not re-trigger the same event type inside this window
MIN_SEVERITY = 0.4  # severity applied when the impact score only just clears the trigger


@dataclass(frozen=True)
class Scenario:
    """Market and credit shocks at full severity (impact score 10)."""

    name: str
    narrative: str
    rates_bp: float = 0.0  # parallel shift in the risk-free curve
    ig_spread_bp: float = 0.0
    hy_spread_bp: float = 0.0
    equity_pct: float = 0.0
    usd_pct: float = 0.0  # USD appreciation against foreign currencies
    oil_pct: float = 0.0
    pd_multiplier: float = 1.0  # applied to every borrower's default probability
    sector_pd_multiplier: dict[str, float] = field(default_factory=dict)  # additional, per sector
    lgd_addon: float = 0.0  # absolute increase in loss given default

    def scaled(self, severity: float) -> "Scenario":
        s = float(np.clip(severity, 0.0, 1.5))
        return replace(
            self,
            rates_bp=self.rates_bp * s,
            ig_spread_bp=self.ig_spread_bp * s,
            hy_spread_bp=self.hy_spread_bp * s,
            equity_pct=self.equity_pct * s,
            usd_pct=self.usd_pct * s,
            oil_pct=self.oil_pct * s,
            pd_multiplier=1 + (self.pd_multiplier - 1) * s,
            sector_pd_multiplier={k: 1 + (v - 1) * s for k, v in self.sector_pd_multiplier.items()},
            lgd_addon=self.lgd_addon * s,
        )


# Simplified scenario library. Magnitudes are anchored on public episodes (for example March 2020:
# equities about -34% peak to trough, high-yield spreads about +700bp, oil about -65%) and on the
# brief's own example for the rate shock (rates +2%, equities -10%).
SCENARIOS: dict[str, Scenario] = {
    "Geopolitical": Scenario(
        "Geopolitical risk-off",
        "Conflict or sanctions drive a flight to quality: equities and credit sell off, government "
        "yields fall, oil and the dollar rise.",
        rates_bp=-50, ig_spread_bp=100, hy_spread_bp=400, equity_pct=-20, usd_pct=5, oil_pct=30, pd_multiplier=1.6,
        sector_pd_multiplier={"Industrials": 1.3, "Consumer Discretionary": 1.3, "Financials": 1.2},
    ),
    "Macroeconomic": Scenario(
        "Rate shock and growth slowdown",
        "Rates reprice sharply higher, equities fall and credit weakens, with rate-sensitive sectors hit hardest.",
        rates_bp=200, ig_spread_bp=75, hy_spread_bp=250, equity_pct=-10, usd_pct=4, oil_pct=-10, pd_multiplier=1.5,
        sector_pd_multiplier={"Real Estate": 1.5, "Consumer Discretionary": 1.3, "Utilities": 1.2},
    ),
    "Credit Event": Scenario(
        "Credit crunch",
        "Defaults or downgrades trigger a repricing of credit: spreads gap wider, default rates and loss severities rise.",
        rates_bp=-25, ig_spread_bp=150, hy_spread_bp=600, equity_pct=-15, usd_pct=2, pd_multiplier=2.5, lgd_addon=0.05,
        sector_pd_multiplier={"Financials": 1.3, "Real Estate": 1.3},
    ),
    "Commodity/Energy": Scenario(
        "Oil price collapse",
        "A supply glut or demand shock crushes oil prices, stressing energy borrowers and high-yield credit.",
        rates_bp=-20, ig_spread_bp=60, hy_spread_bp=350, equity_pct=-8, oil_pct=-50, pd_multiplier=1.2,
        sector_pd_multiplier={"Energy": 3.0, "Materials": 1.5, "Industrials": 1.2},
    ),
}  # fmt: skip
SYSTEMIC_EVENTS = tuple(SCENARIOS)
FOCUS_SECTOR_MULTIPLIER = 1.5  # extra default stress on the sector the triggering news concentrates on


def severity_from_impact(impact: float) -> float:
    """Map an impact score at or above the trigger onto [MIN_SEVERITY, 1.0]."""
    return float(np.clip(MIN_SEVERITY + (1 - MIN_SEVERITY) * (impact - TRIGGER_IMPACT) / (10 - TRIGGER_IMPACT), 0.0, 1.0))


def load_portfolio(path=DATA / "portfolio.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def apply_scenario(portfolio: pd.DataFrame, scenario: Scenario, focus_sector: str | None = None) -> pd.DataFrame:
    """Revalue the portfolio under `scenario`; returns the positions with P&L attribution columns."""
    p = portfolio.copy()
    spread_bp = np.where(p["grade"] == "IG", scenario.ig_spread_bp, np.where(p["grade"] == "HY", scenario.hy_spread_bp, 0.0))
    dy = scenario.rates_bp / 1e4

    p["pnl_rates"] = p["rate_dv01_usd"] * scenario.rates_bp + 0.5 * p["convexity"] * p["market_value_usd"] * dy**2
    p["pnl_credit_spread"] = p["spread_cs01_usd"] * spread_bp
    p["pnl_equity"] = p["equity_delta_usd"] * scenario.equity_pct
    p["pnl_fx"] = p["fx_delta_usd"] * scenario.usd_pct
    p["pnl_commodity"] = p["commodity_delta_usd"] * scenario.oil_pct

    multiplier = scenario.pd_multiplier * p["sector"].map(scenario.sector_pd_multiplier).fillna(1.0)
    if focus_sector:
        multiplier = multiplier * np.where(p["sector"] == focus_sector, FOCUS_SECTOR_MULTIPLIER, 1.0)
    p["stressed_pd"] = (p["pd_1y"] * multiplier).clip(upper=1.0)
    p["stressed_lgd"] = (p["lgd"] + np.where(p["lgd"] > 0, scenario.lgd_addon, 0.0)).clip(upper=1.0)
    p["expected_loss"] = p["ead_usd"] * p["pd_1y"] * p["lgd"]
    p["stressed_expected_loss"] = p["ead_usd"] * p["stressed_pd"] * p["stressed_lgd"]
    p["pnl_credit_loss"] = -(p["stressed_expected_loss"] - p["expected_loss"])

    pnl_cols = ["pnl_rates", "pnl_credit_spread", "pnl_equity", "pnl_fx", "pnl_commodity", "pnl_credit_loss"]
    p["pnl_total"] = p[pnl_cols].sum(axis=1)
    p["stressed_value_usd"] = p["market_value_usd"] + p["pnl_total"]
    return p


PNL_LABELS = {
    "pnl_rates": "Interest rates",
    "pnl_credit_spread": "Credit spreads",
    "pnl_equity": "Equity",
    "pnl_fx": "FX",
    "pnl_commodity": "Commodity",
    "pnl_credit_loss": "Expected credit loss",
}


def summarise(stressed: pd.DataFrame) -> dict:
    before, after = stressed["market_value_usd"].sum(), stressed["stressed_value_usd"].sum()
    return {
        "value_before": float(before),
        "value_after": float(after),
        "pnl": float(after - before),
        "pnl_pct": float((after - before) / before),
        "expected_loss_before": float(stressed["expected_loss"].sum()),
        "expected_loss_after": float(stressed["stressed_expected_loss"].sum()),
        "by_risk_factor": {label: float(stressed[col].sum()) for col, label in PNL_LABELS.items()},
        "by_asset_class": stressed.groupby("asset_class")["pnl_total"].sum().to_dict(),
        "by_sector": stressed.groupby("sector")["pnl_total"].sum().sort_values().to_dict(),
    }


def detect_triggers(
    signals: pd.DataFrame,
    threshold: float = TRIGGER_IMPACT,
    min_signals: int = MIN_SIGNALS,
    cooldown_days: int = COOLDOWN_DAYS,
) -> pd.DataFrame:
    """Find days on which a systemic event type clusters above the impact threshold.

    Returns one row per trigger: date, event_type, impact (mean of the three strongest signals),
    severity, n_signals, focus_sector (set when the news concentrates on one sector) and headlines.
    """
    hot = signals[(signals["impact"] >= threshold) & signals["event_type"].isin(SYSTEMIC_EVENTS)].copy()
    hot["date"] = hot["ts"].str[:10]
    rows, last_fired = [], {}
    for (date, event), g in hot.groupby(["date", "event_type"]):
        docs = g.sort_values("impact", ascending=False).drop_duplicates("doc_id")
        if len(docs) < min_signals:
            continue
        day = pd.Timestamp(date)
        if event in last_fired and (day - last_fired[event]).days < cooldown_days:
            continue
        last_fired[event] = day
        impact = float(docs["impact"].head(3).mean())
        sectors = docs.loc[docs["ticker"] != MARKET, "sector"]
        top = sectors.value_counts(normalize=True)
        focus = top.index[0] if len(top) and top.iloc[0] >= 0.5 and len(sectors) >= len(docs) / 2 else None
        rows.append(
            {
                "date": date,
                "event_type": event,
                "impact": round(impact, 2),
                "severity": round(severity_from_impact(impact), 3),
                "n_signals": int(len(docs)),
                "focus_sector": focus,
                "headlines": docs["text"].head(3).tolist(),
            }
        )
    return pd.DataFrame(rows, columns=["date", "event_type", "impact", "severity", "n_signals", "focus_sector", "headlines"])


def run_stress(portfolio: pd.DataFrame, event_type: str, impact: float, focus_sector: str | None = None) -> tuple[Scenario, pd.DataFrame, dict]:
    """Stress the portfolio for one detected (or hypothetical) event."""
    scenario = SCENARIOS[event_type].scaled(severity_from_impact(impact))
    stressed = apply_scenario(portfolio, scenario, focus_sector)
    return scenario, stressed, summarise(stressed)
