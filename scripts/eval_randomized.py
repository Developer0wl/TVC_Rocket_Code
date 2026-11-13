"""Evaluate the PID baseline over randomized vehicles/wind (fixed seeds, reproducible).

The controller is built from NOMINAL params, as on a real vehicle; the plant is randomized.
Usage: .venv/Scripts/python.exe scripts/eval_randomized.py [--n 40] [--scale 1.0]
"""

import argparse

import numpy as np

from tvc.controllers.pid import PIDController
from tvc.env import TVCRocketEnv
from tvc.params import VehicleParams


def run_episode(seed: int, scale: float, scenario: str):
    env = TVCRocketEnv(randomize=True, rand_scale=scale)
    ctrl = PIDController(VehicleParams())
    options = {"scenario": "pad"} if scenario == "pad" else {"tilt_deg": 10.0}
    env.reset(seed=seed, options=options)
    sim, p = env.unwrapped.sim, env.unwrapped.params
    tilts, terminated, truncated, info = [], False, False, {}
    while not (terminated or truncated):
        cmd = ctrl.command(sim.x, sim.t, env.control_dt)
        _, _, terminated, truncated, info = env.step(cmd / p.gimbal_max)
        tilts.append(np.rad2deg(info["tilt"]))
    return not terminated, max(tilts), tilts[-1], info["altitude"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    for scenario in ("tilt10", "pad"):
        res = [run_episode(s, args.scale, scenario) for s in range(args.n)]
        ok = np.array([r[0] for r in res])
        peak = np.array([r[1] for r in res])
        final = np.array([r[2] for r in res])
        print(
            f"{scenario:7s} n={args.n} scale={args.scale}: survived {ok.mean():.0%}, "
            f"peak tilt mean {peak.mean():.1f} / worst {peak.max():.1f} deg, "
            f"final tilt mean {final.mean():.2f} / worst {final.max():.2f} deg"
        )


if __name__ == "__main__":
    main()
