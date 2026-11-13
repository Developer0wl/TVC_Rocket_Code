"""PID tuning helpers: run one randomized episode with given gains, and score it.

Used by scripts/tune_pid.py (CMA-ES). The controller is built from NOMINAL params while the plant
is randomized (domain randomization), as on a real vehicle.
"""

from __future__ import annotations

import numpy as np

from .controllers.pid import PIDController
from .env import TVCRocketEnv
from .params import VehicleParams

GAIN_NAMES = ("kp_att", "ki_att", "kp_rate", "i_limit")
SETTLE_TIME = 1.0  # s after episode start; metrics "after settling" start here


def run_pid_episode(
    gains: dict,
    seed: int,
    scenario: str = "tilt10",
    rand_scale: float = 1.0,
    vehicle: VehicleParams | None = None,
) -> dict:
    """Fly one episode; return metrics (tilt in deg)."""
    nominal = vehicle or VehicleParams()
    env = TVCRocketEnv(params=nominal, randomize=True, rand_scale=rand_scale)
    ctrl = PIDController(nominal, **gains)
    options = {"scenario": "pad"} if scenario == "pad" else {"tilt_deg": 10.0}
    env.reset(seed=seed, options=options)
    sim, p = env.sim, env.params
    t0 = sim.t
    ctrl.reset()

    tilts, times, sat, rates = [], [], 0, []
    prev = np.zeros(2)
    terminated = truncated = False
    n = 0
    while not (terminated or truncated):
        cmd = ctrl.command(sim.x, sim.t, env.control_dt)
        norm = cmd / p.gimbal_max
        _, _, terminated, truncated, info = env.step(norm)
        tilts.append(np.rad2deg(info["tilt"]))
        times.append(sim.t - t0)
        sat += int(np.any(np.abs(norm) >= 0.999))
        rates.append(float(np.abs(norm - prev).mean()))
        prev = norm
        n += 1
    tilts, times = np.array(tilts), np.array(times)
    after = tilts[times >= SETTLE_TIME]
    if len(after) == 0:
        after = tilts
    return {
        "survived": not terminated,
        "peak_deg": float(tilts.max()),
        "peak_after_settle_deg": float(after.max()),
        "mean_after_settle_deg": float(after.mean()),
        "final_deg": float(tilts[-1]),
        "sat_frac": sat / max(n, 1),
        "action_rate": float(np.mean(rates)),
    }


def episode_cost(m: dict) -> float:
    """Scalar cost for one episode (lower is better)."""
    cost = (
        1.0 * m["final_deg"]
        + 1.0 * m["mean_after_settle_deg"]
        + 1.0 * m["peak_after_settle_deg"]
        + 0.2 * m["peak_deg"]
        + 5.0 * m["sat_frac"]
        + 2.0 * m["action_rate"] * 10.0
    )
    if not m["survived"]:
        cost += 100.0
    return float(cost)


def aggregate(costs: list[float], worst_weight: float = 0.5) -> float:
    """Mean cost plus a worst-case term, so the tuner cannot trade tail risk for the average."""
    c = np.asarray(costs, dtype=float)
    return float(c.mean() + worst_weight * c.max())


def _job(args):
    gains, seed, scenario, rand_scale = args
    return run_pid_episode(gains, seed, scenario, rand_scale)


def summarize(metrics: list[dict]) -> dict:
    """Mean/worst summary used in before/after tables."""
    g = lambda k: np.array([m[k] for m in metrics], dtype=float)  # noqa: E731
    return {
        "n": len(metrics),
        "survived": float(g("survived").mean()),
        "peak_mean": float(g("peak_deg").mean()),
        "peak_worst": float(g("peak_deg").max()),
        "final_mean": float(g("final_deg").mean()),
        "final_worst": float(g("final_deg").max()),
        "after_settle_mean": float(g("mean_after_settle_deg").mean()),
        "sat_frac_mean": float(g("sat_frac").mean()),
        "action_rate_mean": float(g("action_rate").mean()),
    }
