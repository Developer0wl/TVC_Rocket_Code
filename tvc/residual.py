"""Residual-on-PID wrapper: total gimbal command = PID command + scale * policy action.

The policy action in [-1, 1] is multiplied by `scale` (fraction of the gimbal limit) and added to
the PID command, then clipped to [-1, 1]. With a zero action the wrapped env behaves exactly like
the PID baseline, so training starts from the safe controller. The PID is built from NOMINAL
params while the plant may be randomized, as on a real vehicle. The PID command (normalized) is
appended to the observation so the policy can see what it is correcting.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .controllers.pid import PIDController
from .params import VehicleParams


class ResidualPIDWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale: float = 0.3, pid_kwargs: dict | None = None):
        super().__init__(env)
        self.scale = scale
        self._pid_kwargs = pid_kwargs or {}
        self.pid = PIDController(VehicleParams(), **self._pid_kwargs)
        n = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n + 2,), dtype=np.float32)
        self._pid_action = np.zeros(2)

    def _pid_normalized(self) -> np.ndarray:
        base = self.env.unwrapped
        if base.sensing == "imu":
            cmd = self.pid.command_from_estimate(
                base.est.u, base.est.rates, base.sim.t, base.control_dt
            )
        else:
            cmd = self.pid.command(base.sim.x, base.sim.t, base.control_dt)
        return cmd / base.params.gimbal_max

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, self._pid_action]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.pid.reset()
        self._pid_action = self._pid_normalized()
        return self._augment(obs), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        total = np.clip(self._pid_action + self.scale * action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(total)
        info["pid_action"] = self._pid_action.copy()
        info["total_action"] = total
        self._pid_action = self._pid_normalized()
        return self._augment(obs), reward, terminated, truncated, info
