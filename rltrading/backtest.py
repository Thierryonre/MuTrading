"""Backtest trained LightZero agents and simple baselines on the val / test splits.

python -m rltrading.backtest --exp <exp_dir>            # picks checkpoint by val Sharpe, then tests it
python -m rltrading.backtest --baselines
"""
import argparse
import glob
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from rltrading import config as C
from rltrading.envs.trading_env import TradingCore
from rltrading.metrics import portfolio, summarize


def run_cores(cores, act_fn):
    """Step all single-ticker cores in lockstep. act_fn(obs[N,D], t) -> list of actions."""
    obs = np.stack([c.reset(ticker=c._tk) for c in cores])
    done = np.zeros(len(cores), bool)
    step = 0
    while not done.all():
        idx = np.where(~done)[0]
        acts = act_fn(obs[idx], idx, step)
        for k, i in enumerate(idx):
            o, _, d = cores[i].step(acts[k])
            obs[i] = o
            done[i] = d
        step += 1
    frames = {}
    for c in cores:
        h = pd.DataFrame(c.history, columns=["date", "pos", "fwd", "r"])
        h["date"] = pd.to_datetime(h.date)
        frames[c._tk] = h.set_index("date")
    return frames


def make_cores(arrays_path, split, **kw):
    probe = TradingCore(arrays_path, split=split, full_episode=True, **kw)
    cores = []
    for t in probe.names:
        c = TradingCore(arrays_path, split=split, full_episode=True, **kw)
        c._tk = t
        cores.append(c)
    return cores


def report(frames, name):
    daily = pd.DataFrame({t: f.r for t, f in frames.items()}).fillna(0.0)
    pos = pd.DataFrame({t: f.pos for t, f in frames.items()})
    per = pd.DataFrame({t: summarize(daily[t], pos[t]) for t in daily}).T
    port = summarize(portfolio(daily), pos.mean(axis=1))
    port["mean_asset_sharpe"] = float(per.sharpe.mean())
    port["frac_assets_beat_bh"] = np.nan
    return dict(name=name, portfolio=port, per_asset=per, daily=daily, pos=pos)


# ----------------------------------------------------------------------------- agents
def load_policy(exp_dir, ckpt):
    from ding.config import compile_config
    from ding.policy import create_policy

    from rltrading.train_lz import build_cfg, parse
    args = json.load(open(os.path.join(exp_dir, "run_args.json")))
    a = parse([])
    for k, v in args.items():
        if hasattr(a, k):
            setattr(a, k, v)
    a.collectors, a.evaluators = 1, 1
    main_cfg, create_cfg = build_cfg(a)
    if a.algo == "unizero":
        main_cfg.policy.model.world_model_cfg.env_num = 8
        main_cfg.policy.model.world_model_cfg.evaluator_env_num = 8
    main_cfg.exp_name = "/tmp/bt_" + os.path.basename(exp_dir)
    cfg = compile_config(main_cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
    policy = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval"])
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    policy.learn_mode.load_state_dict(sd)
    return policy, a, cfg


def agent_frames(policy, a, cfg, split):
    from rltrading.data.features import make_arrays
    use_sent = a.sentiment != "none"
    arrays = str(make_arrays(a.sentiment if use_sent else None, use_sentiment=use_sent, use_forum=a.forum))
    continuous = a.algo.startswith("sampled")
    positions = (0.0, 1.0) if a.long_only else (-1.0, 0.0, 1.0)
    cores = make_cores(arrays, split, cost_bps=a.cost_bps, positions=positions, continuous=continuous,
                       long_only=a.long_only)
    n_act = 1 if continuous else len(positions)
    em = policy.eval_mode
    # UniZero keeps per-env KV caches that grow over a 374-883 step episode: evaluate a few assets at a time
    chunk = 5 if a.algo == "unizero" else len(cores)
    frames = {}
    for c0 in range(0, len(cores), chunk):
        sub = cores[c0:c0 + chunk]
        em.reset(list(range(len(sub))))

        def act(obs, idx, step):
            data = torch.from_numpy(obs).float()
            mask = [np.ones(n_act, "int8") for _ in idx]
            out = em.forward(data, action_mask=mask, to_play=[-1] * len(idx), ready_env_id=np.array(idx),
                             timestep=[step] * len(idx))
            return [out[i]["action"] for i in idx]

        frames.update(run_cores(sub, act))
        if a.algo == "unizero":
            import gc
            gc.collect()
    return frames


def list_ckpts(exp_dir):
    paths = glob.glob(os.path.join(exp_dir, "ckpt", "iteration_*.pth.tar"))
    return sorted(paths, key=lambda p: int(re.findall(r"iteration_(\d+)", p)[0]))


def evaluate_exp(exp_dir, max_ckpts=8):
    ckpts = list_ckpts(exp_dir)
    if len(ckpts) > max_ckpts:  # evenly spaced subset (always include last)
        sel = np.unique(np.linspace(0, len(ckpts) - 1, max_ckpts).round().astype(int))
        ckpts = [ckpts[i] for i in sel]
    val_rows = []
    best = None
    for ck in ckpts:
        policy, a, cfg = load_policy(exp_dir, ck)
        r = report(agent_frames(policy, a, cfg, "val"), os.path.basename(ck))
        it = int(re.findall(r"iteration_(\d+)", ck)[0])
        val_rows.append(dict(iter=it, **{k: r["portfolio"][k] for k in ["sharpe", "cagr", "max_drawdown", "turnover", "avg_pos", "mean_asset_sharpe"]}))
        print(val_rows[-1], flush=True)
        if best is None or r["portfolio"]["sharpe"] > best[1]:
            best = (ck, r["portfolio"]["sharpe"])
    policy, a, cfg = load_policy(exp_dir, best[0])
    test = report(agent_frames(policy, a, cfg, "test"), os.path.basename(best[0]))
    val = report(agent_frames(policy, a, cfg, "val"), os.path.basename(best[0]))
    name = re.sub(r"_\d{6}_\d{6}$", "", os.path.basename(exp_dir.rstrip("/")))  # drop LightZero timestamp suffix
    out_dir = C.RESULTS_DIR / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(val_rows).to_csv(out_dir / "val_curve.csv", index=False)
    test["per_asset"].to_csv(out_dir / "test_per_asset.csv")
    test["daily"].to_csv(out_dir / "test_daily_logret.csv.gz", compression="gzip", float_format="%.6f")
    test["pos"].to_csv(out_dir / "test_positions.csv.gz", compression="gzip", float_format="%.3f")
    summary = dict(name=name, algo=a.algo, sentiment=a.sentiment, forum=a.forum, long_only=a.long_only,
                   seed=a.seed, best_ckpt=os.path.basename(best[0]), val=val["portfolio"], test=test["portfolio"],
                   args=json.load(open(os.path.join(exp_dir, "run_args.json"))))
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1, default=float)
    print("TEST", name, json.dumps(test["portfolio"], default=float))
    return summary


# ----------------------------------------------------------------------------- baselines
def baseline_frames(split, kind, arrays):
    cores = make_cores(arrays, split, cost_bps=5.0, positions=(-1.0, 0.0, 1.0), continuous=True)
    d = np.load(arrays, allow_pickle=True)
    cols = list(d["cols"])

    def act(obs, idx, step):
        if kind == "buy_hold":
            return [np.array([1.0])] * len(idx)
        if kind == "flat":
            return [np.array([0.0])] * len(idx)
        if kind == "trend_sma200":  # long when above 200d SMA, else flat
            j = cols.index("dist200")
            mu, sd = d["mu"][j], d["sd"][j]
            return [np.array([1.0 if o[j] * sd + mu > 0 else 0.0]) for o in obs]
        if kind == "momentum_20d_ls":  # long/short on 20d return sign
            j = cols.index("ret20")
            mu, sd = d["mu"][j], d["sd"][j]
            return [np.array([np.sign(o[j] * sd + mu)]) for o in obs]
        if kind in ("sent_contrarian", "sent_follow"):  # rule on 5d market news sentiment
            j = cols.index("mkt_sent5")
            sgn = -1 if kind == "sent_contrarian" else 1
            return [np.array([1.0 if sgn * (o[j] * d["sd"][j] + d["mu"][j]) > 0 else 0.0]) for o in obs]
        raise ValueError(kind)
    return run_cores(cores, act)


def run_baselines(sentiment="finbert"):
    from rltrading.data.features import make_arrays
    arrays = str(make_arrays(sentiment, use_sentiment=True))
    rows = []
    for kind in ["buy_hold", "flat", "trend_sma200", "momentum_20d_ls", "sent_contrarian", "sent_follow"]:
        for split in ["val", "test"]:
            r = report(baseline_frames(split, kind, arrays), kind)
            rows.append(dict(strategy=kind, split=split, **r["portfolio"]))
            if kind == "buy_hold" and split == "test":
                r["per_asset"].to_csv(C.RESULTS_DIR / "baseline_buyhold_test_per_asset.csv")
                r["daily"].to_csv(C.RESULTS_DIR / "baseline_buyhold_test_daily.csv.gz", compression="gzip", float_format="%.6f")
    df = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(C.RESULTS_DIR / f"baselines_{sentiment}.csv", index=False, float_format="%.4f")
    print(df.to_string())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="*", default=[])
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--sentiment", default="finbert")
    p.add_argument("--max_ckpts", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.baselines:
        run_baselines(a.sentiment)
    for e in a.exp:
        evaluate_exp(e, a.max_ckpts)
