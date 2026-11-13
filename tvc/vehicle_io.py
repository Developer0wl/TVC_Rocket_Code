"""Load a vehicle description (JSON, SI units) exported from CAD into VehicleParams.

See docs/cad_export_checklist.md for what to read out of SolidWorks/Onshape. The file is
validated hard: wrong frames or unit slips (mm vs m) are the most likely CAD-import bug.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .params import MotorParams, VehicleParams


class VehicleFileError(ValueError):
    pass


def _need(d: dict, key: str, where: str):
    if key not in d:
        raise VehicleFileError(f"missing '{key}' in {where}")
    return d[key]


def _mass_row(d: dict, where: str):
    return tuple(float(_need(d, k, where)) for k in ("mass_kg", "x_cm_m", "Ixx_kgm2", "Iyy_kgm2"))


def params_from_dict(data: dict) -> VehicleParams:
    dims = _need(data, "dimensions", "root")
    length = float(_need(dims, "length_m", "dimensions"))
    radius = float(_need(dims, "radius_m", "dimensions"))
    if not (0.05 <= length <= 5.0 and 0.005 <= radius <= 0.3):
        raise VehicleFileError(
            f"length {length} m / radius {radius} m look wrong (mm instead of m?)"
        )

    mp = _need(data, "mass_properties", "root")
    ign = _mass_row(_need(mp, "ignition", "mass_properties"), "mass_properties.ignition")
    bo = _mass_row(_need(mp, "burnout", "mass_properties"), "mass_properties.burnout")
    for name, row in (("ignition", ign), ("burnout", bo)):
        if not (0.01 <= row[0] <= 100.0):
            raise VehicleFileError(f"{name} mass {row[0]} kg looks wrong (g instead of kg?)")
        if not (0.0 < row[1] < length):
            raise VehicleFileError(f"{name} x_cm {row[1]} m must lie within 0..length={length} m")
        if row[2] <= 0 or row[3] <= 0:
            raise VehicleFileError(f"{name} inertias must be positive")
    if bo[0] >= ign[0]:
        raise VehicleFileError("burnout mass must be less than ignition mass")
    if bo[3] < bo[2]:
        raise VehicleFileError("Iyy (lateral) should exceed Ixx (roll) for a slender rocket")

    motor = _need(data, "motor", "root")
    time = tuple(float(v) for v in _need(motor, "time_s", "motor"))
    thrust = tuple(float(v) for v in _need(motor, "thrust_N", "motor"))
    if len(time) != len(thrust) or len(time) < 2 or time[0] != 0.0 or thrust[-1] != 0.0:
        raise VehicleFileError("motor curve needs matching time/thrust, start at t=0, end at 0 N")
    prop = float(_need(motor, "propellant_mass_kg", "motor"))
    if abs((ign[0] - bo[0]) - prop) > 0.1 * prop + 1e-3:
        raise VehicleFileError(
            f"ignition-burnout mass ({ign[0] - bo[0]:.4f} kg) disagrees with propellant mass "
            f"({prop:.4f} kg) by more than 10%"
        )

    tvc = _need(data, "tvc", "root")
    aero = data.get("aero", {})
    rail = data.get("launch_rail", {})
    base = VehicleParams()
    return replace(
        base,
        length=length,
        radius=radius,
        dry_mass=bo[0],
        mass_table=(ign, bo),
        x_cp=float(aero.get("x_cp_m", base.x_cp)),
        cn_alpha=float(aero.get("cn_alpha_per_rad", base.cn_alpha)),
        cd_axial=float(aero.get("cd_axial", base.cd_axial)),
        cd_normal=float(aero.get("cd_normal", base.cd_normal)),
        rail_length=float(rail.get("length_m", base.rail_length)),
        rail_exit_offset=float(rail.get("exit_offset_m", base.rail_exit_offset)),
        gimbal_max=float(np.deg2rad(_need(tvc, "gimbal_max_deg", "tvc"))),
        servo_rate_max=float(np.deg2rad(_need(tvc, "servo_rate_max_deg_s", "tvc"))),
        servo_tau=float(_need(tvc, "servo_tau_s", "tvc")),
        servo_delay=float(_need(tvc, "servo_delay_s", "tvc")),
        motor=MotorParams(time=time, thrust=thrust, prop_mass=prop),
    )


def load_vehicle(path: str | Path) -> VehicleParams:
    return params_from_dict(json.loads(Path(path).read_text()))
