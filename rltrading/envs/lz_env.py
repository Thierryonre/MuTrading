"""LightZero (DI-engine BaseEnv) wrapper around TradingCore."""
import copy

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from rltrading.envs.trading_env import TradingCore


@ENV_REGISTRY.register('trading_lightzero')
class TradingLightZeroEnv(BaseEnv):
    config = dict(
        arrays_path=None,
        split='train',
        episode_len=128,
        cost_bps=5.0,
        positions=(-1.0, 0.0, 1.0),
        continuous=False,
        long_only=False,
        is_eval=False,
        reward_scale=1000.0,
    )

    @classmethod
    def default_config(cls):
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None):
        self._cfg = cfg or {}
        self._core = None
        self._seed = 0
        self._dynamic_seed = True

    def _make(self):
        c = self._cfg
        split = c.get('split', 'train')
        self._core = TradingCore(c['arrays_path'], split=split, episode_len=c.get('episode_len', 128),
                                 cost_bps=c.get('cost_bps', 5.0), positions=tuple(c.get('positions', (-1., 0., 1.))),
                                 continuous=c.get('continuous', False), long_only=c.get('long_only', False),
                                 seed=self._seed, reward_scale=c.get('reward_scale', 1000.0))
        n_act = 1 if self._core.continuous else len(self._core.positions)
        self._observation_space = gym.spaces.Box(-6, 6, (self._core.obs_dim,), np.float32)
        if self._core.continuous:
            self._action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        else:
            self._action_space = gym.spaces.Discrete(n_act)
        self._reward_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)

    def _wrap(self, obs):
        n = 1 if self._core.continuous else len(self._core.positions)
        return {'observation': obs, 'action_mask': np.ones(n, 'int8'), 'to_play': -1, 'timestep': self._step}

    def reset(self):
        if self._core is None:
            self._make()
        if self._dynamic_seed and not self._core.full_episode:
            self._core.rng = np.random.default_rng(self._core.rng.integers(1 << 31))
        self._step = 0
        self._ep_ret = 0.0
        return self._wrap(self._core.reset())

    def step(self, action):
        obs, rew, done = self._core.step(action)
        self._step += 1
        self._ep_ret += rew
        info = {}
        if done:
            info['eval_episode_return'] = self._ep_ret
            info['ticker'] = self._core.ticker
        return BaseEnvTimestep(self._wrap(obs), np.array([rew], dtype=np.float32), done, info)

    def seed(self, seed, dynamic_seed=True):
        self._seed = int(seed)
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        if self._core is not None:
            self._core.seed_val = self._seed
            self._core.rng = np.random.default_rng(self._seed)

    def random_action(self):
        if self._core is None:
            self._make()
        return np.array([self._action_space.sample()]).reshape(-1)

    def close(self):
        pass

    @property
    def observation_space(self):
        if self._core is None:
            self._make()
        return self._observation_space

    @property
    def action_space(self):
        if self._core is None:
            self._make()
        return self._action_space

    @property
    def reward_space(self):
        if self._core is None:
            self._make()
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg):
        n = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.split = 'train'
        return [cfg for _ in range(n)]

    @staticmethod
    def create_evaluator_env_cfg(cfg):
        n = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.split = 'val'
        return [cfg for _ in range(n)]

    def __repr__(self):
        return 'TradingLightZeroEnv'
