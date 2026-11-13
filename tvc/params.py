"""Vehicle parameters. All values are SI. PLACEHOLDERS for a ~1 kg hobby TVC rocket.

Body frame: x along the rocket axis pointing at the nose, y/z lateral, origin at the CoM.
Positions along the axis ("x_*") are measured FORWARD from the gimbal pivot (aft end).
"""

from dataclasses import dataclass, field

import numpy as np

G0 = 9.80665


@dataclass(frozen=True)
class MotorParams:
    # Piecewise-linear thrust curve, roughly an F-class motor (~72 Ns total impulse).
    time: tuple = (0.0, 0.1, 0.5, 2.5, 3.0, 3.3)
    thrust: tuple = (0.0, 32.0, 28.0, 22.0, 8.0, 0.0)
    prop_mass: float = 0.045  # kg


@dataclass(frozen=True)
class VehicleParams:
    dry_mass: float = 0.955  # kg, includes motor casing, avionics, servos
    length: float = 0.60  # m
    radius: float = 0.03  # m
    x_dry_cm: float = 0.28  # dry CoM, forward of gimbal pivot
    x_prop_cm: float = 0.10  # propellant CoM, forward of gimbal pivot
    prop_length: float = 0.10  # m, propellant grain length (for its own inertia)
    prop_radius: float = 0.012  # m
    x_cp: float = 0.20  # centre of pressure, forward of pivot (aft of CoM => stable)
    cd_axial: float = 0.5
    cd_normal: float = 1.0  # cross-flow drag coefficient on the side area
    cn_alpha: float = 8.0  # normal-force slope per rad (finned slender body), ref area = pi r^2
    # Optional CAD mass table: ((mass, x_cm, Ixx, Iyy) at ignition, (...) at burnout), SI units,
    # x_cm forward of the gimbal pivot, inertias about the CoM. When set it REPLACES the
    # dry_mass/x_dry_cm/x_prop_cm/cylinder-inertia model above.
    mass_table: tuple | None = None
    # Launch rail (0 disables). Effective travel = rail_length - rail_exit_offset.
    rail_length: float = 1.0  # m
    rail_exit_offset: float = 0.10  # m, nozzle-to-rail-button distance
    # Build imperfections (domain-randomization knobs; zero = ideal)
    thrust_misalign: tuple = (0.0, 0.0)  # rad, static offset added to gimbal [theta, phi]
    cg_offset: tuple = (0.0, 0.0)  # m, lateral CoM offset from the geometric axis [y, z]
    # Gimbal servo
    gimbal_max: float = np.deg2rad(8.0)  # rad, per axis
    servo_rate_max: float = 5.0  # rad/s
    servo_tau: float = 0.03  # s, first-order lag
    servo_delay: float = 0.01  # s, command latency
    motor: MotorParams = field(default_factory=MotorParams)

    @property
    def ref_area_axial(self) -> float:
        return np.pi * self.radius**2

    @property
    def ref_area_side(self) -> float:
        return 2.0 * self.radius * self.length
