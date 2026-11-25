"""Download daily OHLCV from Yahoo Finance (free, no key) and cache as parquet."""
import pandas as pd

from rltrading.config import CACHE_DIR, MARKET_TICKERS, PRICE_END, PRICE_START, UNIVERSE


def download_prices(tickers=None, start=PRICE_START, end=PRICE_END, force=False) -> pd.DataFrame:
    """Long-format frame: date, ticker, open, high, low, close, volume (split/dividend adjusted)."""
    tickers = tickers or (UNIVERSE + MARKET_TICKERS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "prices.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    import yfinance as yf  # lazy: not needed when prices are cached
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by="ticker",
                      progress=False, threads=True)
    frames = []
    for t in tickers:
        df = raw[t].dropna(how="all").copy()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = t
        df.index.name = "date"
        frames.append(df.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out.to_parquet(path)
    return out


if __name__ == "__main__":
    df = download_prices(force=True)
    print(df.groupby("ticker")["date"].agg(["min", "max", "count"]))
