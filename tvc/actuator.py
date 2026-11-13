"""Gimbal servo model: command delay, first-order lag, rate limit, angle limit."""

from collections import deque

import numpy as np

from .params import VehicleParams


class GimbalServo:
    """Two-axis gimbal. Angles are [about body y (pitch), about body z (yaw)] in rad."""

    def __init__(self, params: VehicleParams, dt: float):
        self.p = params
        self.dt = dt
        self.delay_steps = int(round(params.servo_delay / dt))
        self.reset()

    def reset(self, angle=(0.0, 0.0)):
        self.angle = np.array(angle, dtype=float)
        # Prefill with the initial angle so a nonzero start isn't pulled toward 0 during the delay.
        self._queue = deque(
            [self.angle.copy() for _ in range(self.delay_steps)], maxlen=self.delay_steps or None
        )

    def step(self, command) -> np.ndarray:
        cmd = np.clip(np.asarray(command, dtype=float), -self.p.gimbal_max, self.p.gimbal_max)
        if self.delay_steps:
            delayed = self._queue[0]
            self._queue.append(cmd)
        else:
            delayed = cmd
        # Exact discretization of the first-order lag (stable for any dt/tau), then rate limit.
        alpha = 1.0 - np.exp(-self.dt / self.p.servo_tau)
        max_step = self.p.servo_rate_max * self.dt
        delta = np.clip((delayed - self.angle) * alpha, -max_step, max_step)
        self.angle = np.clip(self.angle + delta, -self.p.gimbal_max, self.p.gimbal_max)
        return self.angle.copy()
