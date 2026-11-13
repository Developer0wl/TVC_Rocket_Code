"""Evaluate the attitude estimator in flight over randomized plants (WO-D1).

The PID flies on TRUE state; the estimator runs in parallel on the noisy IMU model after a 1 s
stationary pad alignment. Reports attitude error (angle between estimated and true world-up in
the body frame) and body-rate error.

Usage: .venv/Scripts/python.exe scripts/eval_estimator.py [--n 20] [--scale 1.0]
"""

import argparse

import numpy as np

from tvc.controllers.pid import PIDController
from tvc.dynamics import quat_to_rot
from tvc.env import TILT_LIMIT, TVCRocketEnv
from tvc.estimator import G0, AttitudeEstimator
from tvc.params import VehicleParams
from tvc.sensors import IMU

DEG = 180.0 / np.pi


def run_episode(seed: int, scenario: str, scale: float, imu_kwargs: dict | None = None):
    nominal = VehicleParams()
    env = TVCRocketEnv(params=nominal, randomize=True, rand_scale=scale)
    options = {"scenario": "pad"} if scenario == "pad" else {"tilt_deg": 10.0}
    env.reset(seed=seed, options=options)
    sim = env.sim
    ctrl = PIDController(nominal)
    ctrl.reset()

    imu = IMU(seed=seed, **(imu_kwargs or {}))
    imu.reset(seed=seed, start_time=-1.0)
    up_true = quat_to_rot(sim.x[6:10]).T @ np.array([0.0, 0.0, 1.0])
    g_s, a_s = [], []
    for k in range(1000):  # 1 s stationary alignment before launch
        g, a = imu.read(-1.0 + k * 1e-3, np.zeros(3), G0 * up_true)
        g_s.append(g)
        a_s.append(a)
    est = AttitudeEstimator()
    est.align(g_s, a_s)

    errs, rate_errs, times = [], [], []
    t0 = sim.t
    while sim.t < nominal.motor.time[-1] and sim.tilt < TILT_LIMIT:
        cmd = ctrl.command(sim.x, sim.t, env.control_dt)
        for _ in range(env.substeps):
            sim.step(cmd)
            g, a = imu.read(sim.t, sim.x[10:13], sim.specific_force())
            est.update(g, a, sim.dt)
        up = quat_to_rot(sim.x[6:10]).T @ np.array([0.0, 0.0, 1.0])
        errs.append(np.arccos(np.clip(np.dot(est.u, up), -1, 1)) * DEG)
        rate_errs.append(np.linalg.norm((est.rates - sim.x[10:13])[1:]) * DEG)
        times.append(sim.t - t0)
    errs, rate_errs = np.array(errs), np.array(rate_errs)
    return {
        "mean": errs.mean(),
        "max": errs.max(),
        "final": errs[-1],
        "rate_rms": float(np.sqrt(np.mean(rate_errs**2))),
        "align0": errs[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    for scenario in ("tilt10", "pad"):
        rows = [run_episode(1000 + s, scenario, args.scale) for s in range(args.n)]
        a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        print(
            f"{scenario:7s} n={args.n} spread x{args.scale}: "
            f"attitude err mean {a['mean'].mean():.2f}"
            f" / worst-max {a['max'].max():.2f} deg | final mean {a['final'].mean():.2f}"
            f" / worst {a['final'].max():.2f} deg | rate err RMS {a['rate_rms'].mean():.2f} deg/s"
            f" | post-alignment err {a['align0'].mean():.2f} deg"
        )


if __name__ == "__main__":
    main()
