import pytest

from riskengine.config import MARKET
from riskengine.nlp.entities import link
from riskengine.nlp.text import clean, dedup_key, is_boilerplate


@pytest.mark.parametrize(
    "text, source, expected",
    [
        ("Apple unveils new iPhone as Microsoft cuts prices", "news", ["AAPL", "MSFT"]),
        ("An apple a day keeps the doctor away", "news", []),
        ("Intelligent Systems posts record quarter", "news", []),
        ("$fb and $GOOG both under antitrust scrutiny", "social", ["META", "GOOGL"]),
        ("$CAT breaking out on volume", "social", ["CAT"]),
        ("Tesla is overvalued, change my mind", "social", []),
        ("Amicus Therapeutics Buy Rating Reaffirmed at JPMorgan Chase & Co.", "news", []),
        ("Goldman Sachs downgrades Ford Motor to neutral", "news", ["F"]),
        ("Goldman Sachs profit falls 46% as bank sets aside cash for loan losses", "news", ["GS"]),
        ("JPMorgan raises Netflix price target to $500", "news", ["NFLX"]),
        ("Fed cuts interest rates to zero in emergency move", "news", [MARKET]),
        ("$SPX looking weak into the close", "social", [MARKET]),
        ("$HUBG given average recommendation of Hold #stocks", "social", []),
        ("Local bakery wins award", "news", []),
    ],
)
def test_link(text, source, expected):
    assert link(text, source) == expected


def test_clean_strips_retweet_url_and_entities():
    assert clean("RT @user: Apple &amp; Tesla   rally https://t.co/abc") == "Apple & Tesla rally"


def test_dedup_key_ignores_case_punctuation_and_links():
    assert dedup_key("Apple RALLIES!! https://t.co/x") == dedup_key("apple rallies")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Oakbrook Investments Llc Buys State Street Corporation, Pfizer Inc, Target Corp, Sells Apple", True),
        ("Thinking about trading options or stock in Beyond Meat, Walt Disney, LYFT?", True),
        ("FibroGen to Present at Bank of America Securities Health Care Conference", True),
        ("Medical Plastics Market (New PDF) | Top Players - Exxon", True),
        ("Amazon buys Zoox for $1.2 billion", False),
        ("Boeing halts 737 Max production", False),
    ],
)
def test_is_boilerplate(text, expected):
    assert is_boilerplate(text) is expected
