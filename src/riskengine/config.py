"""Static configuration: paths, the company universe and the event taxonomy."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = ROOT / "models"
DB_PATH = DATA / "signals.db"

MARKET = "MARKET"  # pseudo-ticker for macro / market-wide documents
BENCHMARK = "SPY"


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    sector: str
    aliases: tuple[str, ...]  # regex fragments matched against text, case-sensitive
    cashtags: tuple[str, ...] = field(default=())  # extra historical cashtags, e.g. FB for META
    in_index: bool = False  # member of the mock index used by the rebalancer


def _c(ticker, name, sector, aliases, cashtags=(), in_index=False) -> Company:
    return Company(ticker, name, sector, tuple(aliases), tuple(cashtags), in_index)


# The 18 index members are current S&P 100 constituents chosen for sector spread and for
# coverage in both text sources. The remaining companies only widen the impact-calibration sample.
COMPANIES: tuple[Company, ...] = (
    _c("AAPL", "Apple", "Information Technology", [r"Apple"], in_index=True),
    _c("MSFT", "Microsoft", "Information Technology", [r"Microsoft"], in_index=True),
    _c("INTC", "Intel", "Information Technology", [r"Intel"], in_index=True),
    _c("GOOGL", "Alphabet", "Communication Services", [r"Alphabet", r"Google"], ["GOOG"], in_index=True),
    _c("META", "Meta Platforms", "Communication Services", [r"Facebook", r"Meta Platforms"], ["FB"], in_index=True),
    _c("DIS", "Walt Disney", "Communication Services", [r"Disney"], in_index=True),
    _c("NFLX", "Netflix", "Communication Services", [r"Netflix"], in_index=True),
    _c("AMZN", "Amazon", "Consumer Discretionary", [r"Amazon"], in_index=True),
    _c("TSLA", "Tesla", "Consumer Discretionary", [r"Tesla"], in_index=True),
    _c("WMT", "Walmart", "Consumer Staples", [r"Walmart", r"Wal-Mart"], in_index=True),
    _c("JPM", "JPMorgan Chase", "Financials", [r"JPMorgan", r"JP Morgan", r"J\.P\. Morgan"], in_index=True),
    _c("BAC", "Bank of America", "Financials", [r"Bank of America", r"BofA"], in_index=True),
    _c("GS", "Goldman Sachs", "Financials", [r"Goldman Sachs", r"Goldman"], in_index=True),
    _c("XOM", "Exxon Mobil", "Energy", [r"Exxon"], in_index=True),
    _c("CVX", "Chevron", "Energy", [r"Chevron"], in_index=True),
    _c("BA", "Boeing", "Industrials", [r"Boeing"], in_index=True),
    _c("GILD", "Gilead Sciences", "Health Care", [r"Gilead"], in_index=True),
    _c("PFE", "Pfizer", "Health Care", [r"Pfizer"], in_index=True),
    _c("NVDA", "Nvidia", "Information Technology", [r"Nvidia", r"NVIDIA"]),
    _c("AMD", "Advanced Micro Devices", "Information Technology", [r"Advanced Micro Devices", r"AMD"]),
    _c("CSCO", "Cisco", "Information Technology", [r"Cisco"]),
    _c("IBM", "IBM", "Information Technology", [r"IBM"]),
    _c("V", "Visa", "Financials", [r"Visa Inc"]),
    _c("PYPL", "PayPal", "Financials", [r"PayPal"]),
    _c("AXP", "American Express", "Financials", [r"American Express", r"AmEx"]),
    _c("WFC", "Wells Fargo", "Financials", [r"Wells Fargo"]),
    _c("C", "Citigroup", "Financials", [r"Citigroup"]),
    _c("MS", "Morgan Stanley", "Financials", [r"Morgan Stanley"]),
    _c("JNJ", "Johnson & Johnson", "Health Care", [r"Johnson & Johnson", r"J&J"]),
    _c("MRK", "Merck", "Health Care", [r"Merck"]),
    _c("ABBV", "AbbVie", "Health Care", [r"AbbVie"]),
    _c("UNH", "UnitedHealth", "Health Care", [r"UnitedHealth"]),
    _c("HD", "Home Depot", "Consumer Discretionary", [r"Home Depot"]),
    _c("MCD", "McDonald's", "Consumer Discretionary", [r"McDonald's", r"McDonalds"]),
    _c("NKE", "Nike", "Consumer Discretionary", [r"Nike"]),
    _c("SBUX", "Starbucks", "Consumer Discretionary", [r"Starbucks"]),
    _c("GM", "General Motors", "Consumer Discretionary", [r"General Motors"]),
    _c("F", "Ford Motor", "Consumer Discretionary", [r"Ford Motor"]),
    _c("KO", "Coca-Cola", "Consumer Staples", [r"Coca-Cola"]),
    _c("COST", "Costco", "Consumer Staples", [r"Costco"]),
    _c("T", "AT&T", "Communication Services", [r"AT&T"]),
    _c("VZ", "Verizon", "Communication Services", [r"Verizon"]),
    _c("GE", "General Electric", "Industrials", [r"General Electric"]),
    _c("CAT", "Caterpillar", "Industrials", [r"Caterpillar"]),
)

BY_TICKER: dict[str, Company] = {c.ticker: c for c in COMPANIES}
INDEX_TICKERS: tuple[str, ...] = tuple(c.ticker for c in COMPANIES if c.in_index)
ALL_TICKERS: tuple[str, ...] = tuple(c.ticker for c in COMPANIES)

# Event taxonomy: the five classes named in the brief plus the classes a credit or
# equity analyst would track separately. "Market Commentary" is the low-information bucket.
EVENT_TYPES: tuple[str, ...] = (
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Merger/Acquisition",
    "Product/Company News",
    "Earnings",
    "Analyst Rating",
    "Legal/Regulatory",
    "Management Change",
    "Commodity/Energy",
    "Market Commentary",
)

# Replay window covered by the committed sample data.
REPLAY_START = "2020-01-02"
REPLAY_END = "2020-07-16"
