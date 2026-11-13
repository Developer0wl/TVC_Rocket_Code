"""Fly the PID baseline through the standard perturbation set; print metrics, save plots.

Usage: .venv/Scripts/python.exe scripts/fly_pid.py [--out outputs]
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tvc.atmosphere import WindModel
from tvc.controllers.pid import PIDController
from tvc.env import TVCRocketEnv
from tvc.params import VehicleParams

GUST = WindModel(
    mean_10m=np.array([2.0, 0.0, 0.0]),
    gust_amplitude=5.0,
    gust_start=1.0,
    gust_duration=1.0,
    gust_direction=np.array([0.0, 1.0, 0.0]),
)


def fly(params, wind, options, seed=0):
    env = TVCRocketEnv(params=params, wind=wind)
    ctrl = PIDController(params)
    env.reset(seed=seed, options=options)
    sim = env.unwrapped.sim
    ctrl.reset()
    log = {
        k: []
        for k in (
            "t",
            "alt",
            "tilt",
            "wy",
            "wz",
            "cmd_y",
            "cmd_z",
            "ang_y",
            "ang_z",
            "thrust",
            "mass",
        )
    }
    terminated = truncated = False
    sat_steps = n_steps = 0
    while not (terminated or truncated):
        cmd = ctrl.command(sim.x, sim.t, env.control_dt)
        _, _, terminated, truncated, info = env.step(cmd / params.gimbal_max)
        n_steps += 1
        sat_steps += int(np.any(np.abs(cmd) >= params.gimbal_max * 0.999))
        for k, v in (
            ("t", sim.t),
            ("alt", sim.x[2]),
            ("tilt", np.rad2deg(sim.tilt)),
            ("wy", np.rad2deg(sim.x[11])),
            ("wz", np.rad2deg(sim.x[12])),
            ("cmd_y", np.rad2deg(cmd[0])),
            ("cmd_z", np.rad2deg(cmd[1])),
            ("ang_y", np.rad2deg(info["gimbal"][0])),
            ("ang_z", np.rad2deg(info["gimbal"][1])),
            ("thrust", sim.motor.thrust(sim.t)),
            ("mass", sim.motor.mass_properties(sim.t)[0]),
        ):
            log[k].append(v)
    log = {k: np.array(v) for k, v in log.items()}
    tilt = log["tilt"]
    unsettled = np.nonzero(tilt > 2.0)[0]
    settle = float(log["t"][unsettled[-1]]) if len(unsettled) else float(log["t"][0])
    metrics = {
        "survived": not terminated,
        "peak_tilt_deg": float(tilt.max()),
        "settle_t_s": settle,
        "sat_frac": sat_steps / max(n_steps, 1),
        "final_alt_m": float(log["alt"][-1]),
    }
    return metrics, log


def scenarios():
    base = VehicleParams()
    slow_servo = replace(base, servo_delay=base.servo_delay * 2)
    imperfect = replace(base, thrust_misalign=(0.0, np.deg2rad(0.5)), cg_offset=(0.003, -0.002))
    return {
        "tilt10_gust": (base, GUST, {"tilt_deg": 10.0, "azimuth_deg": 30.0}),
        "pad_gust": (base, GUST, {"scenario": "pad"}),
        "tilt10_delay_x2": (slow_servo, GUST, {"tilt_deg": 10.0, "azimuth_deg": 30.0}),
        "tilt10_imperfect": (imperfect, GUST, {"tilt_deg": 10.0, "azimuth_deg": 30.0}),
    }


def plot(log, path):
    t = log["t"]
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    ax[0].plot(t, log["alt"])
    ax[0].set_ylabel("altitude (m)")
    ax[1].plot(t, log["tilt"])
    ax[1].axhline(2, ls=":", c="gray")
    ax[1].set_ylabel("tilt (deg)")
    ax[2].plot(t, log["wy"], label="pitch rate")
    ax[2].plot(t, log["wz"], label="yaw rate")
    ax[2].set_ylabel("body rate (deg/s)")
    ax[2].legend()
    ax[3].plot(t, log["cmd_y"], "--", label="cmd pitch")
    ax[3].plot(t, log["ang_y"], label="act pitch")
    ax[3].plot(t, log["cmd_z"], "--", label="cmd yaw")
    ax[3].plot(t, log["ang_z"], label="act yaw")
    ax[3].set_ylabel("gimbal (deg)")
    ax[3].legend(ncol=4, fontsize=8)
    ax[4].plot(t, log["thrust"], label="thrust (N)")
    ax4b = ax[4].twinx()
    ax4b.plot(t, log["mass"], c="C1", label="mass (kg)")
    ax[4].set_ylabel("thrust (N)")
    ax4b.set_ylabel("mass (kg)")
    ax[4].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print(
        f"{'scenario':18s} {'survived':9s} {'peak tilt':>9s} {'settle(s)':>9s} "
        f"{'sat%':>6s} {'alt(m)':>7s}"
    )
    for name, (params, wind, options) in scenarios().items():
        m, log = fly(params, wind, options)
        print(
            f"{name:18s} {str(m['survived']):9s} {m['peak_tilt_deg']:9.2f} {m['settle_t_s']:9.2f} "
            f"{100 * m['sat_frac']:6.1f} {m['final_alt_m']:7.1f}"
        )
        plot(log, out / f"fly_pid_{name}.png")
    print(f"plots written to {out.resolve()}")


if __name__ == "__main__":
    main()
