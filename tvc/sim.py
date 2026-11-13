"""Simulation wrapper: motor + servo + dynamics stepped at a fixed physics rate.

Launch-rail phase (idea borrowed from RocketPy, MIT, https://github.com/RocketPy-Team/RocketPy,
reimplemented here): until the rocket has travelled `rail_length - rail_exit_offset` along the
rail axis, it is constrained to 1-DOF translation along that axis with attitude frozen, and it
cannot slide backwards. Gimbal authority is irrelevant on the rail.
"""

import numpy as np

from .actuator import GimbalServo
from .atmosphere import WindModel
from .dynamics import GRAVITY, derivative, quat_to_rot, rk4, rk4_step, upright_quaternion
from .motor import Motor
from .params import VehicleParams


class Sim:
    def __init__(self, params: VehicleParams | None = None, wind: WindModel | None = None, dt=1e-3):
        self.p = params or VehicleParams()
        self.wind = wind or WindModel()
        self.dt = dt
        self.motor = Motor(self.p)
        self.servo = GimbalServo(self.p, dt)
        self.reset()

    def reset(self, x0: np.ndarray | None = None, t0: float = 0.0, on_rail: bool | None = None):
        """Reset the state. By default a pad start (x0 is None) begins on the rail and an
        explicit x0 (free-flight test or perturbed start) does not."""
        if on_rail is None:
            on_rail = x0 is None and self.p.rail_length > 0
        if x0 is None:
            x0 = np.zeros(13)
            x0[6:10] = upright_quaternion()
        self.x = np.array(x0, dtype=float)
        self.x[6:10] /= np.linalg.norm(self.x[6:10])
        self.t = t0
        self.on_rail = bool(on_rail)
        self.rail_axis = quat_to_rot(self.x[6:10])[:, 0]
        self._rail_origin = self.x[0:3].copy()
        self.rail_exit_speed = None
        self.servo.reset()
        return self.x.copy()

    @property
    def rail_travel(self) -> float:
        return float(np.dot(self.x[0:3] - self._rail_origin, self.rail_axis))

    def _rail_derivative(self, gimbal):
        axis = self.rail_axis

        def f(t, x):
            full = derivative(t, x, gimbal, self.p, self.motor, self.wind)
            speed = float(np.dot(x[3:6], axis))
            a_axis = float(np.dot(full[3:6], axis))
            if speed <= 0.0 and a_axis < 0.0:
                a_axis = 0.0  # the rail supports the rocket; it cannot slide back
            out = np.zeros(13)
            out[0:3] = axis * speed
            out[3:6] = axis * a_axis
            return out  # attitude and rates frozen (q_dot = w_dot = 0)

        return f

    def step(self, gimbal_cmd) -> np.ndarray:
        """Advance one physics step with the commanded gimbal angles [pitch, yaw] (rad)."""
        angle = self.servo.step(gimbal_cmd)
        if self.on_rail:
            self.x = rk4(self._rail_derivative(angle), self.t, self.x, self.dt)
            self.x[3:6] = self.rail_axis * max(float(np.dot(self.x[3:6], self.rail_axis)), 0.0)
            if self.rail_travel >= self.p.rail_length - self.p.rail_exit_offset:
                self.on_rail = False
                self.rail_exit_speed = float(np.linalg.norm(self.x[3:6]))
        else:
            self.x = rk4_step(self.t, self.x, self.dt, angle, self.p, self.motor, self.wind)
        self.t += self.dt
        return self.x.copy()

    def specific_force(self) -> np.ndarray:
        """Body-frame specific force (m/s^2) an ideal accelerometer at the CoM would read.

        Excludes gravity: free fall reads zero; sitting on the pad reads +g along body x (up).
        """
        rot = quat_to_rot(self.x[6:10])
        acc = derivative(self.t, self.x, self.servo.angle, self.p, self.motor, self.wind)[3:6]
        if self.on_rail:
            # Constrained to the rail axis; before liftoff the rail holds the rocket still.
            along = max(float(np.dot(acc, self.rail_axis)), 0.0)
            acc = self.rail_axis * along
        return rot.T @ (acc - GRAVITY)

    @property
    def tilt(self) -> float:
        """Angle (rad) between the rocket axis and world vertical."""
        q = self.x[6:10]
        # Column 0 of R is the body x axis in world coordinates; atan2 stays accurate near 0.
        horizontal = np.hypot(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]))
        vertical = 2 * (q[1] * q[3] - q[0] * q[2])
        return float(np.arctan2(horizontal, vertical))
