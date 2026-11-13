"""IMU model: gyro + accelerometer between the true state and the controller.

Per-axis pipeline (noise structure follows RocketPy's inertial sensor, MIT, reimplemented):
    ideal -> + white noise + bias (constant + random walk) -> saturate -> quantize
then sampled at `rate` Hz with zero-order hold and a fixed latency (RocketPy has no latency or
hold model; both matter for a real MCU loop). All randomness comes from a seeded Generator.
White noise std = noise_density * sqrt(rate / 2)  (one-sided density in units/sqrt(Hz)).
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

DEG = np.pi / 180.0
G0 = 9.80665


@dataclass(frozen=True)
class AxisSpec:
    noise_density: float  # units / sqrt(Hz)
    bias_const_std: float  # units; per-flight constant bias ~ N(0, std)
    bias_walk: float  # units / sqrt(s); random-walk rate
    range: float  # +/- saturation, units
    resolution: float  # quantization step, units (0 disables)


# Roughly hobby-grade MEMS IMU values (placeholders).
GYRO = AxisSpec(0.005 * DEG, 0.2 * DEG, 0.02 * DEG, 2000 * DEG, 2000 * DEG / 32768)
ACCEL = AxisSpec(0.0004 * G0, 0.05, 0.005, 16 * G0, 16 * G0 / 32768)


class IMU:
    def __init__(
        self,
        rate: float = 500.0,
        latency: float = 0.002,
        gyro: AxisSpec = GYRO,
        accel: AxisSpec = ACCEL,
        seed: int | None = None,
        ideal: bool = False,
    ):
        self.rate, self.gyro_spec, self.accel_spec = rate, gyro, accel
        self.period = 1.0 / rate
        self.latency = latency
        self.ideal = ideal
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: int | None = None, start_time: float = 0.0):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._bias = {
            "gyro": self.rng.normal(0.0, self.gyro_spec.bias_const_std, 3),
            "accel": self.rng.normal(0.0, self.accel_spec.bias_const_std, 3),
        }
        self._walk = {"gyro": np.zeros(3), "accel": np.zeros(3)}
        self._next_sample = start_time  # may be negative: lets a pre-launch window run first
        self._pending = deque()  # (ready_time, gyro, accel)
        self._out = (np.zeros(3), np.zeros(3))

    def start_at(self, t: float):
        """Resume sampling at time t, keeping biases/walk (e.g. after a pre-launch window)."""
        self._next_sample = t
        self._pending.clear()

    def _corrupt(self, key: str, spec: AxisSpec, ideal: np.ndarray) -> np.ndarray:
        self._walk[key] += self.rng.normal(0.0, spec.bias_walk * np.sqrt(self.period), 3)
        std = spec.noise_density * np.sqrt(self.rate / 2.0)
        y = ideal + self.rng.normal(0.0, std, 3) + self._bias[key] + self._walk[key]
        y = np.clip(y, -spec.range, spec.range)
        if spec.resolution > 0:
            y = np.round(y / spec.resolution) * spec.resolution
        return y

    def read(self, t: float, true_rates, true_specific_force):
        """Call every physics step with the current truth. Returns (gyro, accel) as the
        controller sees them at time t: held between samples and delayed by `latency`."""
        rates = np.asarray(true_rates, dtype=float)
        force = np.asarray(true_specific_force, dtype=float)
        while t + 1e-12 >= self._next_sample:
            if self.ideal:
                g, a = rates.copy(), force.copy()
            else:
                g = self._corrupt("gyro", self.gyro_spec, rates)
                a = self._corrupt("accel", self.accel_spec, force)
            self._pending.append((self._next_sample + self.latency, g, a))
            self._next_sample += self.period
        while self._pending and self._pending[0][0] <= t + 1e-12:
            _, g, a = self._pending.popleft()
            self._out = (g, a)
        return self._out[0].copy(), self._out[1].copy()
