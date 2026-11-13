"""CMA-ES tuning of the cascaded PID gains over randomized plants (WO-B1).

Training uses seeds 0-999 (fixed set, shared by all candidates = common random numbers).
Held-out comparison uses seeds 1000+ and is run once at the end against the hand-tuned baseline.

Usage: .venv/Scripts/python.exe scripts/tune_pid.py --name tune_pid_001 [--gens 20] [--pop 14]
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import numpy as np

from tvc.tuning import _job, aggregate, episode_cost, summarize

# name: (low, high) searched in log10 space, mapped to [0, 1] for CMA.
BOUNDS = {
    "kp_att": (1.0, 30.0),
    "ki_att": (0.05, 10.0),
    "kp_rate": (3.0, 60.0),
    "i_limit": (0.05, 1.0),
}
HAND_TUNED = {"kp_att": 6.0, "ki_att": 1.0, "kp_rate": 15.0, "i_limit": 0.3}
NAMES = list(BOUNDS)


def to_unit(gains: dict) -> np.ndarray:
    return np.array(
        [
            (np.log10(gains[k]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            for k, (lo, hi) in BOUNDS.items()
        ]
    )


def from_unit(x) -> dict:
    return {
        k: float(10 ** (np.log10(lo) + np.clip(v, 0, 1) * (np.log10(hi) - np.log10(lo))))
        for v, (k, (lo, hi)) in zip(x, BOUNDS.items(), strict=True)
    }


def train_tasks(gains, n_per_scenario, rand_scale):
    tasks = [(gains, s, "tilt10", rand_scale) for s in range(n_per_scenario)]
    tasks += [(gains, 100 + s, "pad", rand_scale) for s in range(n_per_scenario)]
    return tasks


def evaluate_gains(pool, gains, tasks_fn):
    metrics = list(pool.map(_job, tasks_fn(gains)))
    return aggregate([episode_cost(m) for m in metrics]), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--pop", type=int, default=14)
    ap.add_argument("--n-train", type=int, default=8, help="train episodes per scenario")
    ap.add_argument("--n-heldout", type=int, default=50, help="held-out episodes per scenario")
    ap.add_argument("--rand-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path("runs") / args.name
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new --name")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "bounds": BOUNDS}, indent=2))

    def tasks_fn(gains):
        return train_tasks(gains, args.n_train, args.rand_scale)

    es = cma.CMAEvolutionStrategy(
        to_unit(HAND_TUNED),
        0.25,
        {"bounds": [0, 1], "popsize": args.pop, "seed": args.seed + 1, "verbose": -9},
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        base_cost, _ = evaluate_gains(pool, HAND_TUNED, tasks_fn)
        print(f"hand-tuned train cost: {base_cost:.2f}")
        best_cost, best_gains = base_cost, dict(HAND_TUNED)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gen", "best_gen_cost", "best_so_far", *NAMES])
            for gen in range(args.gens):
                xs = es.ask()
                costs = [evaluate_gains(pool, from_unit(x), tasks_fn)[0] for x in xs]
                es.tell(xs, costs)
                i = int(np.argmin(costs))
                if costs[i] < best_cost:
                    best_cost, best_gains = costs[i], from_unit(xs[i])
                w.writerow([gen, f"{costs[i]:.3f}", f"{best_cost:.3f}", *best_gains.values()])
                f.flush()
                print(
                    f"gen {gen:2d}  gen-best {costs[i]:7.2f}  best {best_cost:7.2f}  "
                    + " ".join(f"{k}={v:.3g}" for k, v in best_gains.items()),
                    flush=True,
                )

        (out / "best_gains.json").write_text(json.dumps(best_gains, indent=2))

        # One-shot held-out comparison (seeds 1000+), never used during the search.
        def held(gains):
            n = args.n_heldout
            t = [(gains, 1000 + s, "tilt10", args.rand_scale) for s in range(n)]
            t += [(gains, 2000 + s, "pad", args.rand_scale) for s in range(n)]
            return t

        report = {}
        for label, g in (("hand_tuned", HAND_TUNED), ("cma_tuned", best_gains)):
            ms = list(pool.map(_job, held(g)))
            half = len(ms) // 2
            report[label] = {"tilt10": summarize(ms[:half]), "pad": summarize(ms[half:])}
    (out / "heldout_report.json").write_text(json.dumps(report, indent=2))
    for scen in ("tilt10", "pad"):
        print(f"\nheld-out {scen} (n={args.n_heldout}), mean/worst:")
        for label in ("hand_tuned", "cma_tuned"):
            r = report[label][scen]
            print(
                f"  {label:10s} survived {r['survived']:.0%} | peak {r['peak_mean']:.1f}/"
                f"{r['peak_worst']:.1f} | final {r['final_mean']:.2f}/{r['final_worst']:.2f} | "
                f"sat {r['sat_frac_mean']:.3f} | act-rate {r['action_rate_mean']:.4f}"
            )
    print(f"\nbest gains: {best_gains}\nsaved to {out}")


if __name__ == "__main__":
    main()
