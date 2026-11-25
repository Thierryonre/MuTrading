"""Performance metrics from daily log-return series."""
import numpy as np
import pandas as pd


def summarize(logret: pd.Series, pos: pd.Series | None = None) -> dict:
    r = logret.fillna(0.0)
    n = len(r)
    if n == 0:
        return {}
    simple = np.expm1(r)
    eq = np.exp(r.cumsum())
    dd = eq / eq.cummax() - 1
    sd = simple.std()
    out = dict(
        total_return=float(eq.iloc[-1] - 1),
        cagr=float(eq.iloc[-1] ** (252 / n) - 1),
        ann_vol=float(sd * np.sqrt(252)),
        sharpe=float(simple.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        max_drawdown=float(dd.min()),
        days=int(n),
    )
    if pos is not None:
        out.update(turnover=float(pos.diff().abs().fillna(pos.abs()).sum() / n * 252),
                   pct_long=float((pos > 0.05).mean()), pct_short=float((pos < -0.05).mean()),
                   avg_pos=float(pos.mean()))
    return out


def portfolio(daily: pd.DataFrame) -> pd.Series:
    """Equal-weight portfolio of per-asset strategies (daily rebalanced). daily: date x ticker log returns."""
    return np.log1p(np.expm1(daily).mean(axis=1))
