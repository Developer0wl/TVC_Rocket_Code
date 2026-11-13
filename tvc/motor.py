"""Motor thrust curve, propellant burn, and time-varying mass properties."""

import numpy as np

from .params import VehicleParams


class Motor:
    """Piecewise-linear thrust curve with exact cumulative impulse for mass burn."""

    def __init__(self, params: VehicleParams):
        self.p = params
        self._t = np.asarray(params.motor.time, dtype=float)
        self._T = np.asarray(params.motor.thrust, dtype=float)
        dt = np.diff(self._t)
        seg_impulse = 0.5 * (self._T[:-1] + self._T[1:]) * dt
        self._cum = np.concatenate(([0.0], np.cumsum(seg_impulse)))
        self.total_impulse = float(self._cum[-1])
        self.burn_time = float(self._t[-1])

    def thrust(self, t: float) -> float:
        if t <= self._t[0] or t >= self.burn_time:
            return 0.0
        return float(np.interp(t, self._t, self._T))

    def impulse_to(self, t: float) -> float:
        """Exact integral of thrust from 0 to t."""
        if t <= self._t[0]:
            return 0.0
        if t >= self.burn_time:
            return self.total_impulse
        k = int(np.searchsorted(self._t, t, side="right")) - 1
        h = t - self._t[k]
        slope = (self._T[k + 1] - self._T[k]) / (self._t[k + 1] - self._t[k])
        return float(self._cum[k] + self._T[k] * h + 0.5 * slope * h * h)

    def prop_mass(self, t: float) -> float:
        frac = self.impulse_to(t) / self.total_impulse
        return self.p.motor.prop_mass * (1.0 - frac)

    def mass_properties(self, t: float):
        """Return (mass, d, I) at time t.

        d: distance of the CoM forward of the gimbal pivot (the thrust moment arm).
        I: principal inertia diagonal [Ixx, Iyy, Izz] about the CoM, body frame.
        """
        p = self.p
        if p.mass_table is not None:
            return self._table_mass_properties(t)
        md, mp = p.dry_mass, self.prop_mass(t)
        m = md + mp
        d = (md * p.x_dry_cm + mp * p.x_prop_cm) / m

        # Dry body: uniform cylinder about its own CoM.
        ixx_d = 0.5 * md * p.radius**2
        iyy_d = md * (3 * p.radius**2 + p.length**2) / 12.0
        # Propellant grain: solid cylinder.
        ixx_p = 0.5 * mp * p.prop_radius**2
        iyy_p = mp * (3 * p.prop_radius**2 + p.prop_length**2) / 12.0

        # Parallel-axis shift to the combined CoM (lateral axes only).
        iyy = iyy_d + md * (p.x_dry_cm - d) ** 2 + iyy_p + mp * (p.x_prop_cm - d) ** 2
        ixx = ixx_d + ixx_p
        return m, d, np.array([ixx, iyy, iyy])

    def _table_mass_properties(self, t: float):
        """CAD mass table, interpolated by remaining propellant fraction f (1 at ignition).

        Mass and the first moment (mass * x_cm) are exactly linear in propellant mass;
        inertia about the CoM is interpolated linearly (a good approximation).
        """
        (m0, x0, ixx0, iyy0), (m1, x1, ixx1, iyy1) = self.p.mass_table
        f = self.prop_mass(t) / self.p.motor.prop_mass
        mass = m1 + (m0 - m1) * f
        d = (m1 * x1 + (m0 * x0 - m1 * x1) * f) / mass
        ixx = ixx1 + (ixx0 - ixx1) * f
        iyy = iyy1 + (iyy0 - iyy1) * f
        return mass, d, np.array([ixx, iyy, iyy])

    def inertia_rate(self, t: float, h: float = 1e-3) -> np.ndarray:
        """dI/dt by central difference (inertia changes as propellant burns)."""
        t0, t1 = max(t - h, 0.0), t + h
        return (self.mass_properties(t1)[2] - self.mass_properties(t0)[2]) / (t1 - t0)
