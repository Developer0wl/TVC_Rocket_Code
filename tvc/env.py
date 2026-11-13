"""Gymnasium environment TVCRocketEnv (bump ENV_VERSION on any obs/action/reward change).

Control runs at `control_dt` (default 100 Hz); physics at 1 kHz. Action is the commanded gimbal
[pitch, yaw] normalized to [-1, 1] of the gimbal limit. The reward is a PLACEHOLDER.

Sensing modes (WO-D2):
  sensing="true" (default, v0 observation, 14-dim): observation built from the true state.
  sensing="imu"  (v1 observation, 11-dim): noisy IMU + attitude estimator after a 1 s pad
      alignment. No true state, velocity, or position is exposed (no such sensors yet);
      obs = [est. world-up in body (3), est. body rates (3), accel/30 (3), previous action (2)].
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .atmosphere import WindModel
from .dynamics import quat_mult, quat_to_rot, upright_quaternion
from .estimator import G0, AttitudeEstimator
from .params import VehicleParams
from .randomization import sample as sample_params
from .sensors import IMU
from .sim import Sim

ENV_VERSION = "v1"  # v1 adds sensing="imu"; sensing="true" keeps the v0 observation layout
TILT_LIMIT = np.deg2rad(45.0)


def tilted_quaternion(tilt: float, azimuth: float) -> np.ndarray:
    """Upright attitude tilted by `tilt` rad toward world azimuth `azimuth` (about world z)."""
    axis = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])  # horizontal axis, world frame
    dq = np.concatenate(([np.cos(tilt / 2)], np.sin(tilt / 2) * axis))
    return quat_mult(dq, upright_quaternion())


class TVCRocketEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        params: VehicleParams | None = None,
        wind: WindModel | None = None,
        control_dt: float = 0.01,
        physics_dt: float = 1e-3,
        max_init_tilt_deg: float = 5.0,
        randomize: dict | bool = False,
        rand_scale: float = 1.0,
        sensing: str = "true",
    ):
        super().__init__()
        if sensing not in ("true", "imu"):
            raise ValueError("sensing must be 'true' or 'imu'")
        self.sensing = sensing
        self.imu: IMU | None = None
        self.est = AttitudeEstimator()
        self._accel = np.array([G0, 0.0, 0.0])
        self.params = params or VehicleParams()
        self.wind = wind or WindModel()
        self.control_dt = control_dt
        self.substeps = int(round(control_dt / physics_dt))
        self.max_init_tilt = np.deg2rad(max_init_tilt_deg)
        self.randomize = randomize  # False, True (default spec), or a spec dict
        self.rand_scale = rand_scale
        self.nominal_params, self.nominal_wind = self.params, self.wind
        self.drawn: dict = {}
        self._physics_dt = physics_dt
        self.sim = Sim(self.params, self.wind, dt=physics_dt)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        obs_dim = 14 if sensing == "true" else 11
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._prev_action = np.zeros(2)

    # -- helpers ---------------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        if self.sensing == "imu":
            return np.concatenate(
                [self.est.u, self.est.rates, self._accel / 30.0, self._prev_action]
            ).astype(np.float32)
        x = self.sim.x
        rot = quat_to_rot(x[6:10])
        up_body = rot.T @ np.array([0.0, 0.0, 1.0])
        v_body = rot.T @ x[3:6] / 20.0
        obs = np.concatenate(
            [up_body, x[10:13], v_body, [x[2] / 50.0], x[0:2] / 20.0, self._prev_action]
        )
        return obs.astype(np.float32)

    # -- gym API ---------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if self.randomize:
            spec = None if self.randomize is True else self.randomize
            self.params, self.wind, self.drawn = sample_params(
                self.np_random, self.nominal_params, spec, self.rand_scale
            )
            self.sim = Sim(self.params, self.wind, dt=self._physics_dt)
        scenario = options.get("scenario", "perturbed")
        if scenario == "pad":
            self.sim.reset()
        else:
            tilt = options.get("tilt_deg")
            tilt = (
                np.deg2rad(tilt)
                if tilt is not None
                else self.np_random.uniform(0, self.max_init_tilt)
            )
            azimuth = options.get("azimuth_deg")
            azimuth = (
                np.deg2rad(azimuth) if azimuth is not None else self.np_random.uniform(0, 2 * np.pi)
            )
            x0 = np.zeros(13)
            x0[2] = 1.0
            x0[6:10] = tilted_quaternion(tilt, azimuth)
            x0[3:6] = quat_to_rot(x0[6:10])[:, 0] * 5.0  # moving along the rocket axis
            x0[10:13] = self.np_random.normal(0.0, 0.02, 3) * np.array([0.0, 1.0, 1.0])
            self.sim.reset(x0, t0=options.get("t0", 0.4))
        self._prev_action = np.zeros(2)
        if self.sensing == "imu":
            self._align_imu()
        return self._obs(), self._info()

    def _align_imu(self):
        """1 s stationary pad alignment (gravity -> tilt, gyro mean -> bias), then fly."""
        seed = int(self.np_random.integers(2**31 - 1))
        self.imu = IMU(seed=seed)
        self.imu.reset(seed=seed, start_time=-1.0)
        up = quat_to_rot(self.sim.x[6:10]).T @ np.array([0.0, 0.0, 1.0])
        gyro, accel = [], []
        for k in range(1000):
            g, a = self.imu.read(-1.0 + k * 1e-3, np.zeros(3), G0 * up)
            gyro.append(g)
            accel.append(a)
        self.est.align(gyro, accel)
        self.imu.start_at(self.sim.t)
        self._accel = np.mean(accel, axis=0)

    def _info(self) -> dict:
        return {
            "t": self.sim.t,
            "tilt": self.sim.tilt,
            "altitude": float(self.sim.x[2]),
            "gimbal": self.sim.servo.angle.copy(),
            "on_rail": self.sim.on_rail,
            "randomized": dict(self.drawn),
            "sensing": self.sensing,
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        cmd = action * self.params.gimbal_max
        for _ in range(self.substeps):
            self.sim.step(cmd)
            if self.sensing == "imu":
                g, a = self.imu.read(self.sim.t, self.sim.x[10:13], self.sim.specific_force())
                self.est.update(g, a, self.sim.dt)
                self._accel = a

        tilt = self.sim.tilt
        w = self.sim.x[10:13]
        # PLACEHOLDER reward: stay upright, low rates, smooth actions.
        reward = (
            1.0
            - (tilt / TILT_LIMIT) ** 2
            - 0.01 * float(w @ w)
            - 0.01 * float(np.sum((action - self._prev_action) ** 2))
        )
        self._prev_action = action

        crashed = (not self.sim.on_rail) and self.sim.x[2] <= 0.0 and self.sim.x[5] < 0.0
        terminated = bool(tilt > TILT_LIMIT or crashed)
        if terminated:
            reward -= 10.0
        truncated = bool(self.sim.t >= self.params.motor.time[-1])  # motor burnout
        return self._obs(), float(reward), terminated, truncated, self._info()
