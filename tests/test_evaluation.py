import json

import numpy as np
import pytest

from tvc.evaluation import (
    EVAL_SEEDS,
    check_exit_criteria,
    episode_metrics,
    evaluate,
    summarize,
)

CRITERIA = {
    "survival_1x": 1.0,
    "survival_3x": 0.95,
    "mean_final_improvement": 0.2,
    "action_rate_max_ratio": 1.5,
}


def test_eval_seed_set_is_locked_and_disjoint_from_training():
    assert EVAL_SEEDS == tuple(range(1000, 1100))
    assert min(EVAL_SEEDS) >= 1000  # training/tuning use 0-999


def test_episode_metrics_values():
    t = np.linspace(0, 3, 301)
    tilts = np.where(t < 0.5, 10.0, 0.5)
    acts = np.zeros((301, 2))
    acts[:10, 0] = 1.0  # saturated for 10 of 301 steps
    m = episode_metrics(tilts, t, acts, terminated=False)
    assert m["survived"] and m["peak_deg"] == 10.0 and m["final_deg"] == 0.5
    assert m["settle_s"] == pytest.approx(0.49, abs=0.02)
    assert m["sat_frac"] == pytest.approx(10 / 301)
    assert m["action_rate"] > 0


def _summary(survival=1.0, final_mean=1.0, final_worst=3.0, rate=0.01):
    s = {"survival": survival, "final_mean": final_mean, "final_worst": final_worst}
    s["action_rate_mean"] = rate
    return {"tilt10": dict(s), "pad": dict(s)}


def test_exit_criteria_pass_and_each_failure_mode():
    ref = _summary(final_mean=4.0, final_worst=10.0, rate=0.01)
    good = _summary(final_mean=2.0, final_worst=8.0, rate=0.012)
    assert check_exit_criteria(good, _summary(survival=0.96), ref, CRITERIA)["all_pass"]

    assert not check_exit_criteria(good, _summary(survival=0.90), ref, CRITERIA)["all_pass"]
    assert not check_exit_criteria(
        _summary(survival=0.98, final_mean=2.0, final_worst=8.0), good, ref, CRITERIA
    )["all_pass"]
    worse_tail = _summary(final_mean=2.0, final_worst=12.0)
    assert not check_exit_criteria(worse_tail, good, ref, CRITERIA)["all_pass"]
    small_gain = _summary(final_mean=3.5, final_worst=8.0)
    assert not check_exit_criteria(small_gain, good, ref, CRITERIA)["all_pass"]
    chatter = _summary(final_mean=2.0, final_worst=8.0, rate=0.05)
    assert not check_exit_criteria(chatter, good, ref, CRITERIA)["all_pass"]


def test_evaluate_pid_is_reproducible_and_json_serializable():
    a = evaluate(None, n=2)
    b = evaluate(None, n=2)
    assert a["tilt10"] == b["tilt10"] and a["pad"] == b["pad"]
    assert a["tilt10"]["n"] == 2 and 0.0 <= a["pad"]["survival"] <= 1.0
    json.dumps(a)  # machine-readable


def test_n_beyond_locked_set_is_rejected():
    with pytest.raises(ValueError):
        evaluate(None, n=101)


def test_summarize_reports_worst_case():
    ms = [
        {
            "survived": True,
            "peak_deg": 5,
            "final_deg": 1,
            "settle_s": 0.2,
            "sat_frac": 0.0,
            "action_rate": 0.01,
            "mean_after_settle_deg": 1,
        },
        {
            "survived": False,
            "peak_deg": 40,
            "final_deg": 30,
            "settle_s": 2.0,
            "sat_frac": 0.5,
            "action_rate": 0.05,
            "mean_after_settle_deg": 20,
        },
    ]
    s = summarize(ms)
    assert s["survival"] == 0.5 and s["final_worst"] == 30 and s["peak_worst"] == 40
