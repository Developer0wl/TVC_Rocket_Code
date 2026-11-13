"""Domain randomization: sample VehicleParams + wind per episode from a declarative spec.

Spec pattern (idea from RocketPy's stochastic classes, MIT, reimplemented): a dict
    {name: (nominal, spread)}               -> normal(nominal, spread)
    {name: (nominal, spread, "uniform")}    -> uniform(nominal - spread, nominal + spread)
    {name: (nominal, spread, "normal")}     -> same as the 2-tuple form
Draws are clipped to `CLIPS` (RocketPy can draw negative masses; we don't). All randomness comes
from the numpy Generator passed in (the env's `self.np_random`), so episodes are reproducible.

Names are VehicleParams fields, plus these virtual knobs:
    impulse_scale      multiplies the thrust curve (total impulse, propellant mass unchanged)
    burn_scale         stretches the burn time (thrust level unchanged, so impulse changes too)
    thrust_misalign_y/_z   static thrust angle offsets (rad)
    cg_offset_y/_z         lateral CoM offsets (m)
    wind_speed, wind_azimuth_deg, gust_amplitude   wind (returned as a WindModel)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .atmosphere import WindModel
from .params import MotorParams, VehicleParams

CLIPS = {
    "dry_mass": (0.3, 3.0),
    "x_dry_cm": (0.05, 0.55),
    "x_cp": (0.05, 0.55),
    "cd_axial": (0.1, 2.0),
    "cd_normal": (0.2, 2.5),
    "cn_alpha": (1.0, 20.0),
    "servo_tau": (0.005, 0.2),
    "servo_delay": (0.0, 0.05),
    "servo_rate_max": (1.0, 20.0),
    "gimbal_max": (np.deg2rad(2.0), np.deg2rad(20.0)),
    "impulse_scale": (0.7, 1.3),
    "burn_scale": (0.7, 1.3),
    "wind_speed": (0.0, 15.0),
    "gust_amplitude": (0.0, 10.0),
}

# A conservative default: roughly +/-5-10% manufacturing/assembly spread, mild wind.
DEFAULT_SPEC = {
    "dry_mass": (0.955, 0.03),
    "x_dry_cm": (0.28, 0.01),
    "x_cp": (0.20, 0.01),
    "cn_alpha": (8.0, 1.0),
    "cd_axial": (0.5, 0.05),
    "servo_tau": (0.03, 0.005),
    "servo_delay": (0.01, 0.002),
    "impulse_scale": (1.0, 0.05),
    "thrust_misalign_y": (0.0, np.deg2rad(0.3)),
    "thrust_misalign_z": (0.0, np.deg2rad(0.3)),
    "cg_offset_y": (0.0, 0.002),
    "cg_offset_z": (0.0, 0.002),
    "wind_speed": (3.0, 2.0),
    "wind_azimuth_deg": (0.0, 180.0, "uniform"),
    "gust_amplitude": (2.0, 1.5),
}

_VIRTUAL = {
    "impulse_scale",
    "burn_scale",
    "thrust_misalign_y",
    "thrust_misalign_z",
    "cg_offset_y",
    "cg_offset_z",
    "wind_speed",
    "wind_azimuth_deg",
    "gust_amplitude",
}


def _draw(rng: np.random.Generator, name: str, spec: tuple) -> float:
    nominal, spread = float(spec[0]), float(spec[1])
    dist = spec[2] if len(spec) > 2 else "normal"
    if dist == "uniform":
        value = rng.uniform(nominal - spread, nominal + spread)
    elif dist == "normal":
        value = rng.normal(nominal, spread)
    else:
        raise ValueError(f"unknown distribution {dist!r} for {name}")
    if name in CLIPS:
        value = float(np.clip(value, *CLIPS[name]))
    return float(value)


def sample(
    rng: np.random.Generator,
    base: VehicleParams | None = None,
    spec: dict | None = None,
    scale: float = 1.0,
):
    """Return (VehicleParams, WindModel, drawn_values) for one episode.

    `scale` widens or narrows every spread (0 = nominal; use it as a curriculum knob).
    """
    base = base or VehicleParams()
    spec = DEFAULT_SPEC if spec is None else spec
    valid = set(base.__dataclass_fields__) | _VIRTUAL
    unknown = set(spec) - valid
    if unknown:
        raise KeyError(f"unknown randomization keys: {sorted(unknown)}")

    drawn = {k: _draw(rng, k, (v[0], v[1] * scale, *v[2:])) for k, v in spec.items()}

    fields = {k: v for k, v in drawn.items() if k not in _VIRTUAL}
    p = replace(base, **fields)

    motor = p.motor
    thrust = np.asarray(motor.thrust, dtype=float) * drawn.get("impulse_scale", 1.0)
    time = np.asarray(motor.time, dtype=float) * drawn.get("burn_scale", 1.0)
    p = replace(p, motor=MotorParams(tuple(time), tuple(thrust), motor.prop_mass))

    mis = np.asarray(p.thrust_misalign) + [
        drawn.get("thrust_misalign_y", 0.0),
        drawn.get("thrust_misalign_z", 0.0),
    ]
    cg = np.asarray(p.cg_offset) + [drawn.get("cg_offset_y", 0.0), drawn.get("cg_offset_z", 0.0)]
    p = replace(p, thrust_misalign=tuple(mis), cg_offset=tuple(cg))

    az = np.deg2rad(drawn.get("wind_azimuth_deg", 0.0))
    speed = drawn.get("wind_speed", 0.0)
    gust_az = az + np.pi / 2
    wind = WindModel(
        mean_10m=np.array([speed * np.cos(az), speed * np.sin(az), 0.0]),
        gust_amplitude=drawn.get("gust_amplitude", 0.0),
        gust_start=float(rng.uniform(0.5, 2.5)),
        gust_duration=float(rng.uniform(0.5, 1.5)),
        gust_direction=np.array([np.cos(gust_az), np.sin(gust_az), 0.0]),
    )
    return p, wind, drawn
