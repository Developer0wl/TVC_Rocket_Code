"""Shared evaluation protocol (WO-C1): one evaluator for any controller on a locked seed set.

A "controller" is a PID baseline (zero residual) or a PPO residual policy, both run through
ResidualPIDWrapper so they face identical plants. Metrics are reported as mean AND worst case.
The eval seed set is locked here and must never be used for training or tuning (those use 0-999).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .env import TVCRocketEnv
from .residual import ResidualPIDWrapper

EVAL_SEEDS = tuple(range(1000, 1100))  # LOCKED. Training/tuning use seeds 0-999 only.
SCENARIOS = ("tilt10", "pad")
SETTLE_DEG = 2.0

_MODEL_CACHE: dict = {}


def episode_metrics(tilts, times, norm_actions, terminated: bool) -> dict:
    tilts, times = np.asarray(tilts), np.asarray(times)
    acts = np.asarray(norm_actions)
    above = np.nonzero(tilts > SETTLE_DEG)[0]
    settle = float(times[above[-1]]) if len(above) else 0.0
    after = tilts[times >= 1.0]
    after = after if len(after) else tilts
    da = np.abs(np.diff(acts, axis=0)).mean() if len(acts) > 1 else 0.0
    return {
        "survived": not terminated,
        "peak_deg": float(tilts.max()),
        "final_deg": float(tilts[-1]),
        "settle_s": settle,
        "mean_after_settle_deg": float(after.mean()),
        "sat_frac": float(np.mean(np.any(np.abs(acts) >= 0.999, axis=1))),
        "action_rate": float(da),
    }


def _load_policy(model_path: str | None):
    if model_path is None:
        return lambda obs: np.zeros(2)
    if model_path not in _MODEL_CACHE:
        from stable_baselines3 import PPO

        model = PPO.load(model_path, device="cpu")
        _MODEL_CACHE[model_path] = lambda obs: model.predict(obs, deterministic=True)[0]
    return _MODEL_CACHE[model_path]


def run_episode(job: dict) -> dict:
    policy = _load_policy(job["model_path"])
    env = ResidualPIDWrapper(
        TVCRocketEnv(randomize=True, rand_scale=job["scale"]),
        scale=job["residual_scale"],
        pid_kwargs=job["pid"],
    )
    options = {"scenario": "pad"} if job["scenario"] == "pad" else {"tilt_deg": 10.0}
    obs, _ = env.reset(seed=job["seed"], options=options)
    tilts, times, acts, done, terminated = [], [], [], False, False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        tilts.append(np.rad2deg(info["tilt"]))
        times.append(info["t"] - job.get("t0", 0.0))
        acts.append(info["total_action"])
        done = terminated or truncated
    times = np.array(times) - times[0]
    return episode_metrics(tilts, times, acts, terminated)


def summarize(metrics: list[dict]) -> dict:
    g = lambda k: np.array([m[k] for m in metrics], dtype=float)  # noqa: E731
    return {
        "n": len(metrics),
        "survival": float(g("survived").mean()),
        "peak_mean": float(g("peak_deg").mean()),
        "peak_worst": float(g("peak_deg").max()),
        "final_mean": float(g("final_deg").mean()),
        "final_worst": float(g("final_deg").max()),
        "settle_mean": float(g("settle_s").mean()),
        "settle_worst": float(g("settle_s").max()),
        "sat_frac_mean": float(g("sat_frac").mean()),
        "action_rate_mean": float(g("action_rate").mean()),
    }


def evaluate(
    model_path: str | None = None,
    n: int = 100,
    scale: float = 1.0,
    pid: dict | None = None,
    residual_scale: float = 0.3,
    workers: int = 1,
) -> dict:
    """Evaluate on EVAL_SEEDS[:n] per scenario. Returns {scenario: summary, 'episodes': ...}."""
    if n > len(EVAL_SEEDS):
        raise ValueError(f"n={n} exceeds the locked eval set ({len(EVAL_SEEDS)} seeds)")
    jobs = [
        {
            "model_path": model_path,
            "seed": s,
            "scenario": sc,
            "scale": scale,
            "pid": pid or {},
            "residual_scale": residual_scale,
        }
        for sc in SCENARIOS
        for s in EVAL_SEEDS[:n]
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_episode, jobs, chunksize=4))
    else:
        results = [run_episode(j) for j in jobs]
    report = {"scale": scale, "n": n, "residual_scale": residual_scale, "pid": pid or {}}
    for i, sc in enumerate(SCENARIOS):
        chunk = results[i * n : (i + 1) * n]
        report[sc] = summarize(chunk)
    report["episodes"] = {sc: results[i * n : (i + 1) * n] for i, sc in enumerate(SCENARIOS)}
    return report


def check_exit_criteria(
    policy_1x: dict, policy_3x: dict, reference_1x: dict, criteria: dict
) -> dict:
    """Machine-checkable Milestone RL criteria (docs/WORK_ORDERS.md). Returns pass/fail per item.

    Criteria 4 (estimator-based observations) and 5 (reproducibility) are manual checks.
    """
    out = {}
    for sc in ("tilt10", "pad"):
        p1, p3, ref = policy_1x[sc], policy_3x[sc], reference_1x[sc]
        out[sc] = {
            "survival_1x": p1["survival"] >= criteria["survival_1x"],
            "survival_3x": p3["survival"] >= criteria["survival_3x"],
            "worst_final_not_worse_than_ref": p1["final_worst"] <= ref["final_worst"],
            "mean_final_improvement": p1["final_mean"]
            <= (1.0 - criteria["mean_final_improvement"]) * ref["final_mean"],
            "action_rate_ok": p1["action_rate_mean"]
            <= criteria["action_rate_max_ratio"] * ref["action_rate_mean"],
        }
    out["all_pass"] = all(v for sc in ("tilt10", "pad") for v in out[sc].values())
    return out


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
