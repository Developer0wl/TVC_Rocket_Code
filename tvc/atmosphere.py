"""ISA air density and a simple deterministic wind model (world frame, z-up)."""

from dataclasses import dataclass, field

import numpy as np


def air_density(z: float) -> float:
    """ISA troposphere density at altitude z (m)."""
    z = min(max(z, 0.0), 11000.0)
    temp = 288.15 - 0.0065 * z
    pressure = 101325.0 * (temp / 288.15) ** 5.25588
    return pressure / (287.05 * temp)


@dataclass(frozen=True)
class WindModel:
    """Steady wind with power-law shear plus an optional 1-cos gust."""

    mean_10m: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s at 10 m
    shear_exponent: float = 1.0 / 7.0
    gust_amplitude: float = 0.0  # m/s
    gust_start: float = 0.0  # s
    gust_duration: float = 1.0  # s
    gust_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def velocity(self, t: float, z: float) -> np.ndarray:
        w = np.asarray(self.mean_10m, dtype=float) * (max(z, 1.0) / 10.0) ** self.shear_exponent
        if self.gust_amplitude and self.gust_start <= t <= self.gust_start + self.gust_duration:
            phase = 2.0 * np.pi * (t - self.gust_start) / self.gust_duration
            w = w + self.gust_amplitude * 0.5 * (1.0 - np.cos(phase)) * np.asarray(
                self.gust_direction, dtype=float
            )
        return w
