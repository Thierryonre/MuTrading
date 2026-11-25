"""Does the sentiment signal predict returns at all? Rank-IC per split, before any RL.

python scripts/sentiment_ic.py finbert nosible
"""
import sys

import numpy as np
import pandas as pd

from rltrading import config as C
from rltrading.data.features import price_features
from rltrading.data.prices import download_prices

FEATS = ["mkt_sent1", "mkt_sent5", "mkt_sent20", "mkt_newsvol", "asset_sent5", "asset_newscnt",
         "forum_sent5", "forum_vol"]
SPLITS = {"train": (C.TRAIN_START, C.TRAIN_END), "val": (C.VAL_START, C.VAL_END),
          "test": (C.TEST_START, C.TEST_END)}


def ic_table(model):
    pf = price_features(download_prices())[["date", "ticker", "fwd_ret", "close"]]
    # 5-day forward return
    pf["fwd5"] = pf.groupby("ticker").close.transform(lambda c: np.log(c.shift(-5) / c))
    s = pd.read_csv(C.PROCESSED_DIR / f"sentiment_daily_{model}.csv.gz", parse_dates=["date"])
    df = pf.merge(s, on=["date", "ticker"])
    rows = []
    for split, (lo, hi) in SPLITS.items():
        d = df[(df.date >= lo) & (df.date <= hi)]
        spy = d[d.ticker == "SPY"]
        for f in FEATS:
            # market-wide features: evaluate on SPY; asset features: pooled cross-section by day
            if f.startswith(("mkt", "forum")):
                ic1 = spy[f].corr(spy.fwd_ret, method="spearman")
                ic5 = spy[f].corr(spy.fwd5, method="spearman")
                n = len(spy)
            else:
                g = d.groupby("date")
                ic1 = g.apply(lambda x: x[f].corr(x.fwd_ret, method="spearman")).mean()
                ic5 = g.apply(lambda x: x[f].corr(x.fwd5, method="spearman")).mean()
                n = d.date.nunique()
            rows.append(dict(model=model, split=split, feature=f, ic_1d=ic1, ic_5d=ic5, n=n))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    models = sys.argv[1:] or ["finbert"]
    out = pd.concat([ic_table(m) for m in models])
    piv = out.pivot_table(index=["feature"], columns=["model", "split"], values="ic_1d").round(3)
    piv5 = out.pivot_table(index=["feature"], columns=["model", "split"], values="ic_5d").round(3)
    print("1-day rank IC\n", piv, "\n\n5-day rank IC\n", piv5)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(C.RESULTS_DIR / "sentiment_ic.csv", index=False, float_format="%.4f")
    with open(C.RESULTS_DIR / "sentiment_ic.md", "w") as f:
        f.write("### 1-day rank IC (Spearman)\n\n" + piv.to_markdown() + "\n\n### 5-day rank IC\n\n" + piv5.to_markdown() + "\n")
