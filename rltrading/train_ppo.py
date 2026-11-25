"""Model-free reference: PPO (stable-baselines3) on the same environment and features.

python -m rltrading.train_ppo --sentiment finbert --seed 0
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from rltrading import config as C
from rltrading.backtest import make_cores, report, run_cores
from rltrading.data.features import make_arrays
from rltrading.envs.trading_env import TradingCore


class GymTrading(gym.Env):
    def __init__(self, arrays, seed, **kw):
        self.core = TradingCore(arrays, split="train", seed=seed, **kw)
        self.observation_space = gym.spaces.Box(-6, 6, (self.core.obs_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(len(self.core.positions))

    def reset(self, seed=None, options=None):
        return self.core.reset(), {}

    def step(self, a):
        o, r, d = self.core.step(a)
        return o, r / 10.0, d, False, {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sentiment", default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--evals", type=int, default=10)
    a = p.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(2)
    use = a.sentiment != "none"
    arrays = str(make_arrays(a.sentiment if use else None, use_sentiment=use))
    env = DummyVecEnv([lambda i=i: GymTrading(arrays, seed=a.seed * 100 + i) for i in range(8)])
    model = PPO("MlpPolicy", env, seed=a.seed, n_steps=256, batch_size=512, learning_rate=3e-4, gamma=0.95,
                ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]), verbose=0)

    def evaluate(split):
        cores = make_cores(arrays, split)

        def act(obs, idx, step):
            return model.predict(obs, deterministic=True)[0]
        return report(run_cores(cores, act), "ppo")

    name = f"ppo_{a.sentiment}_seed{a.seed}"
    best, rows = None, []
    for k in range(a.evals):
        model.learn(a.steps // a.evals, reset_num_timesteps=False)
        r = evaluate("val")["portfolio"]
        rows.append(dict(step=(k + 1) * a.steps // a.evals, **{x: r[x] for x in ["sharpe", "cagr", "avg_pos", "turnover"]}))
        print(rows[-1], flush=True)
        if best is None or r["sharpe"] > best[0]:
            best = (r["sharpe"], k)
            model.save(f"/tmp/{name}_best")
    model = PPO.load(f"/tmp/{name}_best")
    test, val = evaluate("test"), evaluate("val")
    out = C.RESULTS_DIR / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "val_curve.csv", index=False)
    test["per_asset"].to_csv(out / "test_per_asset.csv")
    test["daily"].to_csv(out / "test_daily_logret.csv.gz", compression="gzip", float_format="%.6f")
    json.dump(dict(name=name, algo="ppo", sentiment=a.sentiment, seed=a.seed, best_ckpt=f"eval{best[1]}",
                   val=val["portfolio"], test=test["portfolio"], args=vars(a)),
              open(out / "summary.json", "w"), indent=1, default=float)
    print("TEST", name, json.dumps(test["portfolio"], default=float))


if __name__ == "__main__":
    main()
