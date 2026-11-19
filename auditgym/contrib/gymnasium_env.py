"""Optional Gymnasium adapter (pip install auditgym[gym]): observations/actions are JSON strings.

    import gymnasium as gym, auditgym.contrib.gymnasium_env  # registers AuditGym-v1
    env = gym.make("AuditGym-v1", mode="train")
    obs, info = env.reset(seed=7)
"""
from __future__ import annotations

import json
import string

import gymnasium as gym
from gymnasium import spaces

from ..env import AuditEnv

_CHARS = string.printable + "".join(chr(c) for c in range(0xA0, 0x500)) + "ёЁ—–…«»№"
MAX_LEN = 2_000_000


class AuditGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mode: str = "train", config: dict | None = None, time_limit_s: float | None = None):
        """time_limit_s=None: no wall-clock limit (slow LLM rollouts); pass 60 to mirror the official limit."""
        self._env = AuditEnv(mode, time_limit_s=time_limit_s)
        self._config = config
        self.observation_space = spaces.Text(max_length=MAX_LEN, charset=_CHARS)
        self.action_space = spaces.Text(max_length=200_000, charset=_CHARS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = seed if seed is not None else int(self.np_random.integers(0, 2**62))
        obs, info = self._env.reset(seed=ep_seed, config=self._config, options=options)
        return json.dumps(obs, ensure_ascii=True), info

    def step(self, action):
        try:
            act = json.loads(action) if isinstance(action, str) else action
        except ValueError:
            act = action
        obs, r, term, trunc, info = self._env.step(act)
        return json.dumps(obs, ensure_ascii=True), float(r), term, trunc, info


gym.register(id="AuditGym-v1", entry_point="auditgym.contrib.gymnasium_env:AuditGymEnv", max_episode_steps=None)
