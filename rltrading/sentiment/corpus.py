"""Build the text corpus to be scored: one row per headline / post, with the date at which
it may first be used by the agent (`eff_date`) and asset tags."""
import re

import duckdb
import numpy as np
import pandas as pd

from rltrading.config import CACHE_DIR, RAW_DIR
from rltrading.data.news import effective_date

B = RAW_DIR / "news" / "data"

FINANCE_RE = (r"(stock|shares|market|wall st|wall street|dow |dow jones|nasdaq|s&p|economy|economic|"
              r"recession|inflation|fed |federal reserve|interest rate|rate hike|rate cut|earnings|profit|"
              r"revenue|ipo|investor|bank|treasury|bond|yield|dollar|oil price|crude|opec|gold price|"
              r"gdp|unemployment|jobless|payroll|tariff|trade war|bailout|bankrupt|merger|acquisition|"
              r"sell-off|selloff|rally|bull market|bear market|crash|volatility)")

# Keyword tags used to build per-asset sentiment. Lower-cased regex.
ASSET_KEYWORDS = {
    "SPY": r"s&p|wall st|wall street|stock market|stocks|dow |dow jones|equities",
    "QQQ": r"nasdaq|tech stocks|technology stocks|big tech",
    "IWM": r"small cap|small-cap|russell 2000",
    "DIA": r"dow |dow jones|blue chip|blue-chip",
    "XLF": r"\bbanks?\b|banking|financials|wall street banks|lender",
    "XLE": r"\boil\b|crude|opec|energy stocks|natural gas|petroleum",
    "XLK": r"tech stocks|technology|semiconductor|chipmaker|silicon valley",
    "XLV": r"health ?care|pharma|drugmaker|biotech|hospital",
    "GLD": r"\bgold\b|bullion|precious metal",
    "TLT": r"treasur(y|ies)|bond yield|bonds?\b|10-year|federal reserve|\bfed\b|interest rate",
    "AAPL": r"\bapple\b|iphone|\baapl\b|tim cook",
    "MSFT": r"microsoft|\bmsft\b|windows|azure|satya nadella",
    "AMZN": r"amazon|\bamzn\b|bezos|aws\b",
    "GOOGL": r"google|alphabet|\bgoogl?\b|youtube",
    "JPM": r"jpmorgan|jp morgan|j\.p\. morgan|jamie dimon|\bjpm\b",
    "XOM": r"exxon|\bxom\b",
    "JNJ": r"johnson & johnson|johnson and johnson|\bjnj\b|j&j",
    "WMT": r"walmart|wal-mart|\bwmt\b",
    "NVDA": r"nvidia|\bnvda\b",
    "BAC": r"bank of america|\bbac\b|bofa",
    "INTC": r"\bintel\b|\bintc\b",
    "KO": r"coca-cola|coca cola|\bcoke\b",
    "PG": r"procter|p&g\b",
    "CSCO": r"\bcisco\b|\bcsco\b",
    "PFE": r"pfizer|\bpfe\b",
}


def _q(sql):
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    return con.execute(sql).df()


def _sources():
    pub = "json_extract_string(extra_fields,'$.publication')"
    yield "cnbc", _q(f"select date, text, extra_fields from read_parquet('{B}/headlines_10sites_2007_2022/*.parquet') where {pub}='CNBC'")
    yield "news10", _q(f"select date, text, extra_fields from read_parquet('{B}/headlines_10sites_2007_2022/*.parquet') "
                       f"where {pub}<>'CNBC' and regexp_matches(lower(text), '{FINANCE_RE}')")
    desk = "json_extract_string(extra_fields,'$.news_desk')"
    nyt = _q(f"select date, text, extra_fields from read_parquet('{B}/nyt_headlines_2010_2021/*.parquet') "
             f"where {desk} in ('Business','SundayBusiness','Your Money')")
    # NYT text = abstract \n\n headline \n\n lead. Keep headline + abstract.
    parts = nyt.text.str.split("\n\n")
    nyt["text"] = parts.str[1].fillna("") + ". " + parts.str[0].fillna("")
    yield "nyt_biz", nyt
    for name in ["sp500_daily_headlines", "finsen_us_2007_2023", "cnbc_headlines"]:
        yield "misc", _q(f"select date, text, extra_fields from read_parquet('{B}/{name}/*.parquet')")
    yield "reddit", _q(f"select date, text, extra_fields from read_parquet('{B}/reddit_finance_sp500/*.parquet')")


def build(max_chars=400):
    frames = []
    for src, df in _sources():
        df = df.dropna(subset=["text"])
        df["eff_date"] = effective_date(df)
        df["source"] = src
        if src == "reddit":
            import json
            ex = df.extra_fields.map(json.loads)
            df["weight"] = np.log1p(ex.map(lambda e: max(e.get("score") or 0, 0)))
            df["subreddit"] = ex.map(lambda e: e.get("subreddit"))
        else:
            df["weight"] = 1.0
            df["subreddit"] = None
        df["text"] = df.text.str.replace(r"\s+", " ", regex=True).str.strip().str[:max_chars]
        df = df[df.text.str.split().str.len() >= 4]  # drop menu items / tickers-only junk
        df = df.drop(columns=["extra_fields", "date"]).drop_duplicates(subset=["text"])
        frames.append(df)
        print(src, len(df), flush=True)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["eff_date"])
    out = out.drop_duplicates(subset=["source", "text"]).reset_index(drop=True)
    low = out.text.str.lower()
    for t, kw in ASSET_KEYWORDS.items():
        out["tag_" + t] = low.str.contains(kw, regex=True)
    out["id"] = np.arange(len(out))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE_DIR / "corpus.parquet")
    print(out.groupby("source").size(), out[[c for c in out if c.startswith("tag_")]].sum().to_dict())
    return out


if __name__ == "__main__":
    build()
