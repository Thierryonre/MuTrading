"""Global settings: paths, universe, date splits."""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
# Large raw data lives outside the repo (WSL native disk / Modal volume) for speed.
DATA_DIR = Path(os.environ.get("RLT_DATA_DIR", REPO_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
# Small derived artefacts that are committed to git.
PROCESSED_DIR = REPO_DIR / "data" / "processed"
RESULTS_DIR = REPO_DIR / "results"

ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "GLD", "TLT"]
STOCKS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "WMT",
          "NVDA", "BAC", "INTC", "KO", "PG", "CSCO", "PFE"]
UNIVERSE = ETFS + STOCKS
MARKET_TICKERS = ["^VIX"]

PRICE_START = "2007-01-01"
PRICE_END = "2025-11-01"

# Chronological splits (no shuffling across time). Chosen so that the main news sources
# (CNBC 2007-2022, NYT business 2010-2021, keyword-filtered general news 2007-2022) cover all three.
TRAIN_START, TRAIN_END = "2010-01-01", "2017-12-31"
VAL_START, VAL_END = "2018-01-01", "2019-06-30"
TEST_START, TEST_END = "2019-07-01", "2022-12-31"
