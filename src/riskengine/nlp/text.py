"""Text normalisation shared by every source adapter."""
from __future__ import annotations

import hashlib
import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_RT = re.compile(r"^RT @\w+:\s*")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9$ ]+")
_CASHTAG = re.compile(r"\$[A-Za-z]{1,6}\b")


def clean(text: str) -> str:
    """Unescape HTML entities, drop URLs and retweet prefixes, collapse whitespace."""
    text = html.unescape(str(text))
    text = _RT.sub("", text)
    text = _URL.sub("", text)
    return _WS.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Key that is equal for texts differing only in case, punctuation or links."""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", clean(text).lower())).strip()


def doc_id(source: str, ts: str, text: str) -> str:
    return hashlib.sha1(f"{source}|{ts}|{text}".encode("utf-8")).hexdigest()[:16]


# Headlines that name companies but carry no news about them: fund-holdings disclosures,
# syndicated market-research adverts, conference schedules and screener round-ups.
_BOILERPLATE = re.compile(
    r"\bBuys\b[^,]+,[^,]+(?:,|\bSells\b)"
    r"|\b(?:LLC|L\.?P\.?|Ltd|Management|Advis[oe]rs?|Capital|Partners|Trust|Associates|Wealth|Investments?|Counsel)\.?,?"
    r"\s+(?:Inc\.?\s+)?Buys\b"
    r"|^Thinking about (?:trading|buying)"
    r"|\bMarket\b.*\b(?:Industry|Forecast|CAGR|Research Report|Size|Share|Top Players|New PDF)\b"
    r"|\bto Present at\b|\bto Participate in\b|\bConference\b|\bWebcast\b|\bAnnounces Availability\b"
    r"|^Stock Research Reports for|^New Strong (?:Buy|Sell) Stocks|\bFiles? (?:10-[KQ]|8-K|Form)\b",
    re.IGNORECASE,
)


def is_boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE.search(text))


def cashtag_count(text: str) -> int:
    return len(set(m.upper() for m in _CASHTAG.findall(text)))
