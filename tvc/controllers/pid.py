"""Cascaded attitude PID for pitch/yaw gimbal control.

Outer loop: attitude error -> rate command (P + clamped I).
Inner loop: rate error -> desired angular acceleration (P).
Gain scheduling by dynamic inversion: torque = I * alpha_des and, for small gimbal angles,
tau_y = -d*T*theta, tau_z = -d*T*phi, so theta = -I*alpha_y/(d*T), phi = -I*alpha_z/(d*T).
That keeps the loop gains independent of the (time-varying) thrust, mass and inertia.
Roll is not controllable with a 2-axis gimbal and is ignored.
"""

import numpy as np

from ..dynamics import quat_to_rot
from ..motor import Motor
from ..params import VehicleParams


class PIDController:
    def __init__(
        self,
        params: VehicleParams,
        kp_att: float = 6.0,
        ki_att: float = 1.0,
        kp_rate: float = 15.0,
        i_limit: float = 0.3,  # rad/s clamp on the integral's rate contribution
        min_thrust: float = 2.0,  # N, below this the gimbal has no authority
    ):
        self.p = params
        self.motor = Motor(params)
        self.kp_att, self.ki_att, self.kp_rate = kp_att, ki_att, kp_rate
        self.i_limit, self.min_thrust = i_limit, min_thrust
        self.reset()

    def reset(self):
        self._int = np.zeros(2)  # integral of attitude error about [y, z]

    def command(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Gimbal command [theta, phi] (rad) from the TRUE state x at time t."""
        up_body = quat_to_rot(x[6:10]).T @ np.array([0.0, 0.0, 1.0])
        return self.command_from_estimate(up_body, x[10:13], t, dt)

    def command_from_estimate(
        self, up_body: np.ndarray, rates: np.ndarray, t: float, dt: float
    ) -> np.ndarray:
        """Gimbal command from (estimated) world-up in the body frame and body rates [3]."""
        thrust = self.motor.thrust(t)
        if thrust < self.min_thrust:
            return np.zeros(2)
        m, d, inertia = self.motor.mass_properties(t)

        # Rotation vector taking body x toward world-up: e = x_hat cross up_body.
        err = np.array([-up_body[2], up_body[1]])  # about [y, z]

        # Anti-windup: freeze the integral when its contribution is already at the clamp.
        i_term = self.ki_att * (self._int + err * dt)
        if np.all(np.abs(i_term) <= self.i_limit):
            self._int += err * dt
        i_term = np.clip(self.ki_att * self._int, -self.i_limit, self.i_limit)

        rate_cmd = self.kp_att * err + i_term
        alpha_des = self.kp_rate * (rate_cmd - np.asarray(rates)[1:3])
        gimbal = -inertia[1:3] * alpha_des / (d * thrust)
        return np.clip(gimbal, -self.p.gimbal_max, self.p.gimbal_max)
