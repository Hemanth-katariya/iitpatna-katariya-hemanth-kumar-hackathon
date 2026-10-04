"""Generate the synthetic wholesale banking portfolio used by the stress-testing module.

Everything here is invented: counterparties are numbered placeholders and all amounts are drawn
from a seeded random generator. The brief's "provided sample transaction data" was not supplied,
and the suggested public transaction sets are retail card data, so a wholesale book of loans,
bonds and derivatives is synthesised instead. Writes data/portfolio.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from riskengine.config import DATA

SEED = 20260
SECTORS = {  # sector -> share of corporate exposure
    "Financials": 0.15, "Industrials": 0.14, "Energy": 0.12, "Information Technology": 0.12,
    "Health Care": 0.10, "Consumer Discretionary": 0.10, "Communication Services": 0.07,
    "Consumer Staples": 0.06, "Utilities": 0.05, "Materials": 0.05, "Real Estate": 0.04,
}  # fmt: skip
# One-year default probabilities by rating: illustrative values of the same order of magnitude as
# long-run corporate default studies published by the rating agencies.
PD = {"AAA": 0.0001, "AA": 0.0002, "A": 0.0006, "BBB": 0.0018, "BB": 0.0065, "B": 0.032, "CCC": 0.25}
LOAN_RATINGS = (["A", "BBB", "BB", "B", "CCC"], [0.10, 0.35, 0.33, 0.19, 0.03])
BOND_RATINGS = (["AA", "A", "BBB", "BB", "B"], [0.08, 0.30, 0.37, 0.17, 0.08])
IG = {"AAA", "AA", "A", "BBB"}

COLUMNS = [
    "position_id", "asset_class", "instrument", "counterparty", "sector", "rating", "grade", "notional_usd",
    "market_value_usd", "maturity_years", "rate_type", "rate_dv01_usd", "convexity", "spread_cs01_usd",
    "equity_delta_usd", "fx_delta_usd", "commodity_delta_usd", "ead_usd", "pd_1y", "lgd",
]  # fmt: skip


def _row(**kw) -> dict:
    row = dict.fromkeys(COLUMNS, 0.0)
    row.update(kw)
    row["grade"] = "Sovereign" if row["sector"] == "Sovereign" else "IG" if row["rating"] in IG else "HY"
    return row


def loans(rng: np.random.Generator, n: int = 80) -> list[dict]:
    rows = []
    for i in range(n):
        sector = rng.choice(list(SECTORS), p=list(SECTORS.values()))
        rating = rng.choice(LOAN_RATINGS[0], p=LOAN_RATINGS[1])
        revolver = rng.random() < 0.35
        commitment = float(rng.lognormal(np.log(60e6), 0.7))
        drawn = commitment * (rng.uniform(0.2, 0.6) if revolver else 1.0)
        maturity = float(rng.uniform(1, 5) if revolver else rng.uniform(2, 7))
        fixed = (not revolver) and rng.random() < 0.3
        duration = maturity * 0.85 if fixed else 0.2  # floating loans reprice within a quarter
        pd_1y, lgd = PD[rating], float(rng.uniform(0.30, 0.50))
        ead = drawn + 0.5 * (commitment - drawn)  # 50% credit conversion factor on the undrawn part
        rows.append(
            _row(
                asset_class="Loan",
                instrument="Revolving Credit Facility" if revolver else "Term Loan",
                counterparty=f"Borrower {i + 1:03d}",
                sector=sector,
                rating=rating,
                notional_usd=commitment,
                market_value_usd=drawn - ead * pd_1y * lgd,  # carrying value net of expected-loss provision
                maturity_years=maturity,
                rate_type="Fixed" if fixed else "Floating",
                rate_dv01_usd=-drawn * duration / 1e4,
                ead_usd=ead,
                pd_1y=pd_1y,
                lgd=lgd,
            )
        )
    return rows


def bonds(rng: np.random.Generator, n_corp: int = 38, n_sov: int = 8) -> list[dict]:
    rows = []
    for i in range(n_corp + n_sov):
        sovereign = i >= n_corp
        sector = "Sovereign" if sovereign else rng.choice(list(SECTORS), p=list(SECTORS.values()))
        rating = "AAA" if sovereign else rng.choice(BOND_RATINGS[0], p=BOND_RATINGS[1])
        face = float(rng.lognormal(np.log(90e6 if sovereign else 45e6), 0.6))
        maturity = float(rng.uniform(1, 20 if sovereign else 12))
        duration = maturity / (1 + 0.035 * maturity) ** 0.9  # rough modified duration for a par coupon bond
        mv = face * rng.uniform(0.94, 1.04)
        rows.append(
            _row(
                asset_class="Bond",
                instrument="Sovereign Bond" if sovereign else "Corporate Bond",
                counterparty=f"Sovereign {i - n_corp + 1:02d}" if sovereign else f"Issuer {i + 1:03d}",
                sector=sector,
                rating=rating,
                notional_usd=face,
                market_value_usd=mv,
                maturity_years=maturity,
                rate_type="Fixed",
                rate_dv01_usd=-mv * duration / 1e4,
                convexity=duration**2 + duration,
                spread_cs01_usd=0.0 if sovereign else -mv * duration / 1e4,
            )
        )
    return rows


def derivatives(rng: np.random.Generator) -> list[dict]:
    rows = []

    def add(instrument, k, sector="Financials", rating="A", **kw):
        rows.append(_row(asset_class="Derivative", instrument=instrument, counterparty=f"Dealer {k:02d}", sector=sector, rating=rating, **kw))

    for k in range(10):  # interest rate swaps: mostly pay-fixed hedges of the fixed-rate assets
        notional, tenor = float(rng.lognormal(np.log(150e6), 0.5)), float(rng.uniform(2, 10))
        pay_fixed = rng.random() < 0.7
        add(
            "Interest Rate Swap (pay fixed)" if pay_fixed else "Interest Rate Swap (receive fixed)", k + 1,
            notional_usd=notional, market_value_usd=float(rng.normal(0, 0.01) * notional), maturity_years=tenor,
            rate_type="Fixed", rate_dv01_usd=(1 if pay_fixed else -1) * notional * tenor * 0.9 / 1e4,
        )  # fmt: skip
    for k in range(6):  # FX forwards: long or short USD against a foreign currency
        notional = float(rng.lognormal(np.log(80e6), 0.5))
        add("FX Forward", k + 11, notional_usd=notional, market_value_usd=float(rng.normal(0, 0.008) * notional),
            maturity_years=float(rng.uniform(0.25, 1.5)), fx_delta_usd=float(rng.choice([1, -1], p=[0.4, 0.6])) * notional / 100)  # fmt: skip
    for k in range(4):  # equity total return swaps: long equity exposure
        notional = float(rng.lognormal(np.log(60e6), 0.4))
        add("Equity Total Return Swap", k + 17, notional_usd=notional, market_value_usd=float(rng.normal(0, 0.01) * notional),
            maturity_years=float(rng.uniform(0.5, 2)), equity_delta_usd=notional / 100)  # fmt: skip
    for k in range(3):  # commodity swaps written for energy clients: the bank is long oil
        notional = float(rng.lognormal(np.log(40e6), 0.4))
        add("Commodity Swap (oil)", k + 21, sector="Energy", rating="BBB", notional_usd=notional,
            market_value_usd=float(rng.normal(0, 0.01) * notional), maturity_years=float(rng.uniform(0.5, 3)),
            commodity_delta_usd=notional / 100)  # fmt: skip
    for k in range(5):  # credit default swaps: bought protection gains when spreads widen
        notional, tenor = float(rng.lognormal(np.log(50e6), 0.4)), 5.0
        bought = rng.random() < 0.6
        add("Credit Default Swap (protection bought)" if bought else "Credit Default Swap (protection sold)", k + 24,
            sector=str(rng.choice(list(SECTORS))), rating=str(rng.choice(["BBB", "BB", "B"])), notional_usd=notional,
            market_value_usd=float(rng.normal(0, 0.005) * notional), maturity_years=tenor,
            spread_cs01_usd=(1 if bought else -1) * notional * 4.5 / 1e4)  # fmt: skip
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame([*loans(rng), *bonds(rng), *derivatives(rng)], columns=COLUMNS)
    df["position_id"] = [f"P{i + 1:04d}" for i in range(len(df))]
    money = [c for c in df.columns if c.endswith("_usd")]
    df[money] = df[money].round(0)
    df[["maturity_years", "convexity", "lgd"]] = df[["maturity_years", "convexity", "lgd"]].round(3)
    df.to_csv(DATA / "portfolio.csv", index=False)
    summary = df.groupby("asset_class").agg(positions=("position_id", "size"), notional=("notional_usd", "sum"), value=("market_value_usd", "sum"))
    print((summary / [1, 1e9, 1e9]).round(2).to_string())
    print(f"total market value ${df.market_value_usd.sum() / 1e9:.2f}bn across {len(df)} positions")


if __name__ == "__main__":
    main()
