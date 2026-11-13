"""Fixed-seed evaluation of PID and/or a PPO residual policy (WO-C1 protocol).

Uses the locked eval seeds (tvc.evaluation.EVAL_SEEDS, 1000-1099). Never train or tune on them.

Examples:
  .venv/Scripts/python.exe scripts/eval_policy.py --n 40 --workers 12            # PID only
  .venv/Scripts/python.exe scripts/eval_policy.py --model runs/<name>/model.zip --n 100 \\
      --pid-config runs/tune_pid_001/best_gains.json --json outputs/eval_<name>.json
Writes machine-readable JSON (per-scenario summaries + per-episode metrics) with --json, and
checks the draft Milestone RL exit criteria when --model is given.
"""

import argparse
import json
from pathlib import Path

from tvc.evaluation import check_exit_criteria, evaluate, load_json


def fmt(name, r):
    return (
        f"  {name:14s} survival {r['survival']:.0%}"
        f" | peak {r['peak_mean']:5.1f}/{r['peak_worst']:5.1f}"
        f" | final {r['final_mean']:5.2f}/{r['final_worst']:5.2f} deg"
        f" | settle {r['settle_mean']:4.2f}/{r['settle_worst']:4.2f} s"
        f" | sat {r['sat_frac_mean']:.3f} | act-rate {r['action_rate_mean']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="PPO residual model .zip (omit for PID only)")
    ap.add_argument("--pid-config", default=None, help="JSON of PID gains (e.g. best_gains.json)")
    ap.add_argument("--n", type=int, default=100, help="episodes per scenario (max 100)")
    ap.add_argument("--scale", type=float, default=1.0, help="randomization spread multiplier")
    ap.add_argument("--residual-scale", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--criteria", default="configs/exit_criteria.json")
    args = ap.parse_args()

    pid = load_json(args.pid_config) if args.pid_config else {}
    common = dict(n=args.n, pid=pid, residual_scale=args.residual_scale, workers=args.workers)
    out = {"pid": evaluate(None, scale=args.scale, **common)}
    if args.model:
        out["policy"] = evaluate(args.model, scale=args.scale, **common)

    print(f"n={args.n} per scenario, spread x{args.scale}   (peak/final/settle: mean/worst)")
    for scenario in ("tilt10", "pad"):
        print(f"{scenario}:")
        for name, rep in out.items():
            print(fmt(name, rep[scenario]))

    if args.model:
        extra = dict(common, scale=3.0)
        out["pid_3x"] = evaluate(None, **{k: v for k, v in extra.items()})
        out["policy_3x"] = evaluate(args.model, **extra)
        verdict = check_exit_criteria(
            out["policy"], out["policy_3x"], out["pid"], load_json(args.criteria)
        )
        out["exit_criteria"] = verdict
        print("\nDRAFT exit-criteria check (automatic items only):")
        for sc in ("tilt10", "pad"):
            print(
                f"  {sc}: "
                + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in verdict[sc].items())
            )
        print(f"  ALL PASS: {verdict['all_pass']}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
