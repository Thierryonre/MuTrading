"""Per-asset daily feature panel used by the trading environment.

Everything at row t uses information available at the close of day t only.
`fwd_ret[t]` = log(close[t+1]/close[t]) is the reward-relevant quantity (never an input).
"""
import json

import numpy as np
import pandas as pd

from rltrading import config as C
from rltrading.data.prices import download_prices

PRICE_FEATURES = ["ret1", "ret5", "ret20", "logvol20", "volratio", "dist50", "dist200", "rsi14",
                  "volu_z", "range", "mkt_ret1", "mkt_ret20", "vix_z", "vix_chg5"]
LAG_RETS = 10  # last N vol-normalised daily returns
SENT_FEATURES = ["mkt_sent1", "mkt_sent5", "mkt_sent20", "mkt_newsvol", "asset_sent5", "asset_newscnt"]
FORUM_FEATURES = ["forum_sent5", "forum_vol"]


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def price_features(px: pd.DataFrame) -> pd.DataFrame:
    wide = {f: px.pivot(index="date", columns="ticker", values=f) for f in ["open", "high", "low", "close", "volume"]}
    close = wide["close"]
    lr = np.log(close).diff()
    spy, vix = lr["SPY"], close["^VIX"]
    out = []
    for t in C.UNIVERSE:
        c, r = close[t], lr[t]
        vol20 = r.rolling(20).std()
        vol5 = r.rolling(5).std()
        f = pd.DataFrame(index=close.index)
        f["ret1"] = r / vol20
        f["ret5"] = r.rolling(5).sum() / (vol20 * np.sqrt(5))
        f["ret20"] = r.rolling(20).sum() / (vol20 * np.sqrt(20))
        f["logvol20"] = np.log(vol20 * np.sqrt(252))
        f["volratio"] = np.log(vol5 / vol20)
        f["dist50"] = np.log(c / c.rolling(50).mean()) / vol20 / np.sqrt(50)
        f["dist200"] = np.log(c / c.rolling(200).mean()) / vol20 / np.sqrt(200)
        f["rsi14"] = (_rsi(c) - 50) / 50
        lv = np.log(wide["volume"][t].replace(0, np.nan))
        f["volu_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
        f["range"] = np.log(wide["high"][t] / wide["low"][t]) / vol20
        svol = spy.rolling(20).std()
        f["mkt_ret1"] = spy / svol
        f["mkt_ret20"] = spy.rolling(20).sum() / (svol * np.sqrt(20))
        lvix = np.log(vix)
        f["vix_z"] = (lvix - lvix.rolling(252).mean()) / lvix.rolling(252).std()
        f["vix_chg5"] = lvix.diff(5)
        for k in range(LAG_RETS):
            f[f"lag{k}"] = (r / vol20).shift(k)
        f["fwd_ret"] = r.shift(-1)
        f["close"] = c
        f["ticker"] = t
        out.append(f.reset_index())
    return pd.concat(out, ignore_index=True)


def build_panel(sentiment_model: str | None = "finbert", force=False) -> pd.DataFrame:
    tag = sentiment_model or "none"
    path = C.CACHE_DIR / f"panel_{tag}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    px = download_prices()
    panel = price_features(px)
    if sentiment_model:
        s = pd.read_csv(C.PROCESSED_DIR / f"sentiment_daily_{sentiment_model}.csv.gz", parse_dates=["date"])
        panel = panel.merge(s, on=["date", "ticker"], how="left")
        for col in SENT_FEATURES + FORUM_FEATURES:
            panel[col] = panel[col].fillna(0.0)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel.to_parquet(path)
    return panel


def feature_columns(use_sentiment: bool, use_forum: bool = False):
    cols = PRICE_FEATURES + [f"lag{k}" for k in range(LAG_RETS)]
    if use_sentiment:
        cols += SENT_FEATURES
    if use_forum:
        cols += FORUM_FEATURES
    return cols


def make_arrays(sentiment_model: str | None, use_sentiment: bool, use_forum: bool = False, force=False):
    """Save normalised arrays for fast loading by env workers. Returns path to .npz."""
    tag = f"{sentiment_model or 'none'}_s{int(use_sentiment)}_f{int(use_forum)}"
    path = C.CACHE_DIR / f"arrays_{tag}.npz"
    if path.exists() and not force:
        return path
    panel = build_panel(sentiment_model if (use_sentiment or use_forum) else None)
    cols = feature_columns(use_sentiment, use_forum)
    panel = panel[(panel.date >= "2008-06-01") & (panel.date <= C.TEST_END)].copy()
    tr = (panel.date >= C.TRAIN_START) & (panel.date <= C.TRAIN_END)
    mu = panel.loc[tr, cols].mean()
    sd = panel.loc[tr, cols].std().replace(0, 1.0)
    X = ((panel[cols] - mu) / sd).clip(-5, 5).fillna(0.0).astype(np.float32)
    dates = panel.date.values.astype("datetime64[D]")
    tick = panel.ticker.values
    np.savez(path, X=X.values, fwd=panel.fwd_ret.fillna(0.0).values.astype(np.float32),
             close=panel.close.values.astype(np.float64), dates=dates, tickers=tick,
             cols=np.array(cols), mu=mu.values, sd=sd.values)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump({"cols": cols, "n": int(len(panel))}, f)
    return path
