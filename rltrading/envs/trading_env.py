"""Single-asset daily trading environment.

At the close of day t the agent picks a target position p_t in [-1, 1] (fraction of capital,
short allowed unless long_only). Reward (in percent log-return units):

    r_t = reward_scale * ( p_t * fwd_ret_t - cost * |p_t - p_{t-1}| )   (default scale 1000 -> units of 0.1%)

Training episodes: random asset, random window of `episode_len` days inside the split.
Eval episodes: asset chosen by the env seed, full split window (deterministic).
"""
import numpy as np

from rltrading import config as C

SPLITS = {"train": (C.TRAIN_START, C.TRAIN_END), "val": (C.VAL_START, C.VAL_END),
          "test": (C.TEST_START, C.TEST_END)}


class TradingCore:
    def __init__(self, arrays_path, split="train", episode_len=128, cost_bps=5.0, positions=(-1.0, 0.0, 1.0),
                 continuous=False, long_only=False, full_episode=None, tickers=None, seed=0,
                 reward_scale=1000.0):
        d = np.load(arrays_path, allow_pickle=True)
        self.X, self.fwd, self.dates, self.tick = d["X"], d["fwd"], d["dates"], d["tickers"]
        lo, hi = np.datetime64(SPLITS[split][0]), np.datetime64(SPLITS[split][1])
        self.split = split
        self.full_episode = (split != "train") if full_episode is None else full_episode
        self.episode_len = episode_len
        self.cost = cost_bps / 1e4
        self.positions = np.array(positions, dtype=np.float32)
        self.continuous = continuous
        self.reward_scale = reward_scale
        self.long_only = long_only
        self.rng = np.random.default_rng(seed)
        self.seed_val = seed
        # index ranges per ticker, restricted to the split
        names = tickers or list(dict.fromkeys(self.tick.tolist()))
        self.ranges = {}
        for t in names:
            idx = np.where((self.tick == t) & (self.dates >= lo) & (self.dates <= hi))[0]
            if len(idx) > 10:
                self.ranges[t] = (idx[0], idx[-1])
        self.names = list(self.ranges)
        self.obs_dim = self.X.shape[1] + 1
        self.eval_counter = 0

    def reset(self, ticker=None):
        if ticker is None:
            if self.full_episode:
                # fixed asset per eval env -> comparable validation scores across evaluations
                ticker = self.names[self.seed_val % len(self.names)]
            else:
                ticker = self.names[self.rng.integers(len(self.names))]
        a, b = self.ranges[ticker]
        if self.full_episode:
            self.t, self.end = a, b
        else:
            L = min(self.episode_len, b - a)
            self.t = int(self.rng.integers(a, b - L + 1))
            self.end = self.t + L
        self.ticker = ticker
        self.pos = 0.0
        self.ret_sum = 0.0
        self.history = []
        return self._obs()

    def _obs(self):
        return np.concatenate([self.X[self.t], [self.pos]]).astype(np.float32)

    def action_to_pos(self, action):
        if self.continuous:
            p = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))
            return max(p, 0.0) if self.long_only else p
        return float(self.positions[int(np.asarray(action).reshape(-1)[0])])

    def step(self, action):
        p = self.action_to_pos(action)
        r = p * float(self.fwd[self.t]) - self.cost * abs(p - self.pos)
        self.history.append((self.dates[self.t], p, float(self.fwd[self.t]), r))
        self.pos = p
        self.ret_sum += r
        self.t += 1
        done = self.t >= self.end
        return self._obs(), self.reward_scale * r, done
