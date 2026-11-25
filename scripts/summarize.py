"""Collect results/runs/*/summary.json into results/summary.csv and a markdown table."""
import glob
import json

import pandas as pd

from rltrading import config as C


def main():
    rows = []
    for p in glob.glob(str(C.RESULTS_DIR / "runs" / "*" / "summary.json")):
        s = json.load(open(p))
        args = s.get("args", {})
        rows.append(dict(
            run=s["name"], algo=s["algo"], sentiment=s["sentiment"] + ("+forum" if s.get("forum") else ""),
            seed=s["seed"], steps=args.get("max_env_step", args.get("steps")), ckpt=s["best_ckpt"],
            val_sharpe=s["val"]["sharpe"], test_sharpe=s["test"]["sharpe"], test_cagr=s["test"]["cagr"],
            test_maxdd=s["test"]["max_drawdown"], test_turnover=s["test"].get("turnover"),
            test_avg_pos=s["test"].get("avg_pos"), test_mean_asset_sharpe=s["test"].get("mean_asset_sharpe"),
            train_hours=(args.get("wall_sec") or 0) / 3600))
    df = pd.DataFrame(rows).sort_values(["algo", "sentiment", "seed"])
    df.to_csv(C.RESULTS_DIR / "summary.csv", index=False, float_format="%.4f")
    fmt = df.copy()
    for c in ["val_sharpe", "test_sharpe", "test_mean_asset_sharpe", "test_avg_pos", "train_hours"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2f}")
    for c in ["test_cagr", "test_maxdd"]:
        fmt[c] = fmt[c].map(lambda x: f"{100 * x:.1f}%")
    fmt["test_turnover"] = fmt.test_turnover.map(lambda x: f"{x:.0f}")
    (C.RESULTS_DIR / "summary.md").write_text(fmt.to_markdown(index=False))
    print(fmt.to_markdown(index=False))
    # mean over seeds
    agg = df.groupby(["algo", "sentiment"])[["val_sharpe", "test_sharpe", "test_cagr", "test_maxdd"]].agg(["mean", "count"])
    print(agg.round(3))


if __name__ == "__main__":
    main()
