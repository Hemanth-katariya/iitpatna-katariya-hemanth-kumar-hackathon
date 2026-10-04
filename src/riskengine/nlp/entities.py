"""Entity linking: map free text to tickers in the universe, or to the market as a whole."""
from __future__ import annotations

import re

from riskengine.config import COMPANIES, MARKET


_TAG_TO_TICKER = {tag: c.ticker for c in COMPANIES for tag in (c.ticker, *c.cashtags)}

# (ticker, plain substrings used as a cheap pre-check, boundary-aware regex). Names stay
# case-sensitive so "Apple" the company is not confused with the fruit.
_NAME_PATTERNS = [
    (
        c.ticker,
        tuple(re.sub(r"\\(.)", r"\1", a) for a in c.aliases),
        re.compile(rf"(?<![A-Za-z])(?:{'|'.join(c.aliases)})(?![A-Za-z])"),
    )
    for c in COMPANIES
]

_MARKET_TAGS = {"SPX", "SPY", "QQQ", "NDX", "DJIA", "DJI", "DIA", "IWM", "RUT", "VIX", "ES", "NQ"}
_CASHTAG = re.compile(r"\$([A-Za-z]{1,6})(?![A-Za-z.])")
_MACRO = re.compile(
    r"\b(?:S&P 500|S&P500|Dow Jones|Nasdaq|Wall Street|Federal Reserve|the Fed|Fed's|FOMC|Powell|Treasury yields?"
    r"|Treasuries|central banks?|ECB|interest rates?|rate cuts?|rate hikes?|inflation|GDP|recession|unemployment"
    r"|jobless|payrolls|stimulus|OPEC|crude|oil prices?|tariffs?|trade war|trade deal|sanctions?|election|Brexit"
    r"|geopolitical|stock market|bond yields?|yield curve)\b"
)

# Banks are often named as the broker issuing an opinion rather than as the subject of the news
# ("Buy rating reaffirmed at JPMorgan", "Goldman Sachs downgrades Ford"). Those mentions are skipped.
_BROKERS = {"JPM", "BAC", "GS", "MS", "C", "WFC"}
_BROKER_BEFORE = re.compile(r"(?:\bat|\bby|\bat the|\bvia)\s+$", re.IGNORECASE)
_BROKER_AFTER = re.compile(
    r"^(?:\s+(?:Chase|Sachs|& Co\.?|Securities|Merrill Lynch|Group))*\s+"
    r"(?:upgrades?|downgrades?|initiates?|reiterates?|maintains|analysts?|says|sees|starts|strategists?"
    r"|(?:raises|cuts|lowers|boosts|trims)\s+(?:\S+\s+){0,3}(?:price )?target"
    r"|(?:\S+\s+){0,4}conference)",
    re.IGNORECASE,
)


def _is_broker_mention(text: str, m: re.Match[str]) -> bool:
    return bool(_BROKER_BEFORE.search(text[: m.start()]) or _BROKER_AFTER.match(text[m.end() :]))


def link(text: str, source: str = "news") -> list[str]:
    """Tickers that `text` is about; [MARKET] for macro text naming no company; [] if neither.

    Social posts are linked by cashtag only: on Twitter a company name without its cashtag is
    usually an aside, and cashtag matching keeps precision high.
    """
    first_seen: dict[str, int] = {}
    tags: set[str] = set()
    if "$" in text:
        for m in _CASHTAG.finditer(text):
            tag = m.group(1).upper()
            tags.add(tag)
            if tag in _TAG_TO_TICKER:
                first_seen.setdefault(_TAG_TO_TICKER[tag], m.start())
    if source != "social":
        for ticker, literals, pattern in _NAME_PATTERNS:
            if ticker in first_seen or not any(lit in text for lit in literals):
                continue
            for m in pattern.finditer(text):
                if not (ticker in _BROKERS and _is_broker_mention(text, m)):
                    first_seen[ticker] = m.start()
                    break
    if first_seen:
        return sorted(first_seen, key=first_seen.get)

    if tags - _MARKET_TAGS:
        return []  # about some company outside the universe
    return [MARKET] if tags or _MACRO.search(text) else []
