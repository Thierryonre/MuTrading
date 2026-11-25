"""Aggregate scored texts into daily, look-ahead-free sentiment features per ticker."""
import sys

import numpy as np
import pandas as pd

from rltrading import config as C
from rltrading.data.prices import download_prices

NEWS_SOURCES = ["cnbc", "news10", "nyt_biz", "misc"]


def _trailing_z(s: pd.Series, win=252, minp=60):
    m = s.rolling(win, min_periods=minp).mean().shift(1)
    sd = s.rolling(win, min_periods=minp).std().shift(1)
    return (s - m) / sd


def _ewm_ratio(sum_s, cnt_s, halflife):
    num = sum_s.fillna(0).ewm(halflife=halflife).mean()
    den = cnt_s.fillna(0).ewm(halflife=halflife).mean()
    return num / (den + 0.05), np.log1p(den)


def build(model: str):
    corpus = pd.read_parquet(C.CACHE_DIR / "corpus.parquet")
    scores = pd.read_parquet(C.CACHE_DIR / f"scores_{model}.parquet")
    df = corpus.merge(scores, on="id")
    df["s"] = df.p_pos - df.p_neg
    px = download_prices()
    tdays = np.sort(px.loc[px.ticker == "SPY", "date"].unique())
    pos = np.searchsorted(tdays, df.eff_date.values.astype("datetime64[ns]"))
    df = df[pos < len(tdays)].copy()
    df["date"] = tdays[pos[pos < len(tdays)]]
    idx = pd.DatetimeIndex(tdays)

    news = df[df.source.isin(NEWS_SOURCES)]
    # --- market level: per-source trailing z-score, then average available sources
    zs, cnts = [], []
    for src in NEWS_SOURCES:
        g = news[news.source == src].groupby("date").s.agg(["mean", "size"]).reindex(idx)
        zs.append(_trailing_z(g["mean"]).rename(src))
        cnts.append(g["size"].fillna(0).rename(src))
    Z = pd.concat(zs, axis=1)
    mkt_raw = Z.mean(axis=1)  # NaN where no source has data
    m = pd.DataFrame(index=idx)
    m["mkt_sent1"] = mkt_raw.fillna(0)
    m["mkt_sent5"] = mkt_raw.fillna(0).ewm(span=5).mean()
    m["mkt_sent20"] = mkt_raw.fillna(0).ewm(span=20).mean()
    tot = np.log1p(pd.concat(cnts, axis=1).sum(axis=1))
    m["mkt_newsvol"] = _trailing_z(tot).fillna(0)

    # --- forum (reddit) level, upvote-weighted
    fr = df[df.source == "reddit"].assign(ws=lambda d: d.s * (1 + d.weight), w=lambda d: 1 + d.weight)
    g = fr.groupby("date")[["ws", "w"]].sum().reindex(idx)
    m["forum_sent5"], m["forum_vol"] = _ewm_ratio(g.ws, g.w, 5)

    # --- asset level from keyword-tagged news
    rows = []
    for t in C.UNIVERSE:
        sub = news[news["tag_" + t]]
        g = sub.groupby("date").s.agg(["sum", "size"]).reindex(idx)
        a = m.copy()
        a["asset_sent5"], a["asset_newscnt"] = _ewm_ratio(g["sum"], g["size"], 5)
        a["ticker"] = t
        rows.append(a.reset_index(names="date"))
    out = pd.concat(rows, ignore_index=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.PROCESSED_DIR / f"sentiment_daily_{model}.csv.gz", index=False, float_format="%.5f", compression="gzip")
    # coverage summary
    cov = Z.notna().groupby(Z.index.year).mean().round(2)
    print(cov)
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "finbert")
