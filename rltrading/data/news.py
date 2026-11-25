"""Free news / forum text from the Hugging Face corpus `financial-news-multisource`.

The corpus unifies ~24 public datasets into (date, text, extra_fields). We only pull the
finance-relevant subsets. No API key needed. Research/non-commercial use only.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from rltrading.config import RAW_DIR

REPO = "idleengine/financial-news-multisource"  # ungated mirror of Brianferrell787/financial-news-multisource
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data?recursive=true"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"

DEFAULT_SUBSETS = [
    "cnbc_headlines", "djia_stock_headlines", "finsen_us_2007_2023", "sp500_daily_headlines",
    "reddit_finance_sp500", "benzinga_6000stocks", "headlines_10sites_2007_2022",
    "yahoo_finance_felixdrinkall", "sentarl_combined", "nyt_headlines_2010_2021",
    "bloomberg_reuters", "us_financial_news_2018_jeet", "yahoo_finance_articles",
]


def list_files(subsets):
    files = requests.get(API, timeout=60).json()
    return [f["path"] for f in files if f["type"] == "file" and f["path"].split("/")[1] in subsets]


def _download(path):
    dest = RAW_DIR / "news" / path
    if dest.exists() and dest.stat().st_size > 0:
        return path
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with requests.get(RESOLVE + path, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return path


def download(subsets=None):
    paths = list_files(subsets or DEFAULT_SUBSETS)
    with ThreadPoolExecutor(6) as ex:
        for p in ex.map(_download, paths):
            print("ok", p, flush=True)


def load_subset(name, columns=("date", "text", "extra_fields")) -> pd.DataFrame:
    files = sorted((RAW_DIR / "news" / "data" / name).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=list(columns)) for f in files], ignore_index=True)


def effective_date(df: pd.DataFrame, cutoff_hour_et: float = 15.5) -> pd.Series:
    """Calendar date (naive) of the first daily close at which an item may be used.

    The agent decides at the close of day t. To avoid look-ahead:
      * minute-precision items published before 15:30 ET on day t -> usable at t, else t+1
      * day-precision items (time unknown) dated t -> usable only at t+1
    Weekend/holiday dates are later rolled forward to the next trading day by the caller.
    """
    ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
    ex = df["extra_fields"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
    day = ex.map(lambda e: e.get("time_precision") == "day").astype(bool)
    raw_day = pd.to_datetime(ex.map(lambda e: e.get("date_raw")), errors="coerce")
    et = ts.dt.tz_convert("America/New_York")
    late = (et.dt.hour + et.dt.minute / 60.0) >= cutoff_hour_et
    eff = et.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(late.astype(int), unit="D")
    day_eff = raw_day.fillna(ts.dt.tz_localize(None).dt.normalize()).dt.normalize() + pd.Timedelta(days=1)
    return eff.where(~day, day_eff)


if __name__ == "__main__":
    download(sys.argv[1:] or None)
