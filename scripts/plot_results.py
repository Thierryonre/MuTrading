"""Equity curves of the equal-weight test portfolios: best seed-0 run per algorithm vs buy & hold.

python scripts/plot_results.py [run_name ...]
"""
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rltrading import config as C
from rltrading.metrics import portfolio


def curve(path):
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    return np.exp(portfolio(d).cumsum())


def main(names):
    runs = sorted(glob.glob(str(C.RESULTS_DIR / "runs" / "*" / "summary.json")))
    if names:
        runs = [r for r in runs if any(n in r for n in names)]
    fig, ax = plt.subplots(figsize=(11, 6))
    bh = curve(C.RESULTS_DIR / "baseline_buyhold_test_daily.csv.gz")
    ax.plot(bh.index, bh.values, color="black", lw=2.2, label="Buy & hold (EW 25)")
    for r in runs:
        s = json.load(open(r))
        p = r.replace("summary.json", "test_daily_logret.csv.gz")
        try:
            c = curve(p)
        except FileNotFoundError:
            continue
        ax.plot(c.index, c.values, lw=1.1, label=f"{s['name']} (Sharpe {s['test']['sharpe']:.2f})")
    ax.axhline(1, color="grey", lw=0.6)
    ax.set_title("Test period 2019-07 → 2022-12, equal-weight portfolio of per-asset agents (net of 5 bps)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = C.RESULTS_DIR / "equity_curves_test.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1:])
