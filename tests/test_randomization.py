import numpy as np
import pytest

from tvc.env import TVCRocketEnv
from tvc.motor import Motor
from tvc.params import VehicleParams
from tvc.randomization import CLIPS, DEFAULT_SPEC, sample


def test_same_seed_same_draw_different_seed_differs():
    a = sample(np.random.default_rng(1))[2]
    b = sample(np.random.default_rng(1))[2]
    c = sample(np.random.default_rng(2))[2]
    assert a == b and a != c


def test_scale_zero_gives_nominal_vehicle():
    p, wind, drawn = sample(np.random.default_rng(0), scale=0.0)
    base = VehicleParams()
    assert p.dry_mass == pytest.approx(DEFAULT_SPEC["dry_mass"][0])
    assert p.cn_alpha == pytest.approx(8.0)
    assert p.thrust_misalign == (0.0, 0.0) and p.cg_offset == (0.0, 0.0)
    assert Motor(p).total_impulse == pytest.approx(Motor(base).total_impulse)


def test_draws_are_clipped_to_physical_ranges():
    spec = {"dry_mass": (0.955, 50.0), "servo_tau": (0.03, 5.0), "impulse_scale": (1.0, 9.0)}
    rng = np.random.default_rng(0)
    for _ in range(200):
        p, _, drawn = sample(rng, spec=spec)
        assert CLIPS["dry_mass"][0] <= p.dry_mass <= CLIPS["dry_mass"][1]
        assert CLIPS["servo_tau"][0] <= p.servo_tau <= CLIPS["servo_tau"][1]
        assert CLIPS["impulse_scale"][0] <= drawn["impulse_scale"] <= CLIPS["impulse_scale"][1]


def test_impulse_scale_scales_total_impulse_and_burn_scale_stretches_time():
    base = Motor(VehicleParams())
    p, _, _ = sample(np.random.default_rng(0), spec={"impulse_scale": (1.2, 0.0)})
    assert Motor(p).total_impulse == pytest.approx(1.2 * base.total_impulse)
    p, _, _ = sample(np.random.default_rng(0), spec={"burn_scale": (1.1, 0.0)})
    assert Motor(p).burn_time == pytest.approx(1.1 * base.burn_time)


def test_virtual_knobs_map_to_params_and_wind():
    spec = {
        "thrust_misalign_z": (0.01, 0.0),
        "cg_offset_y": (0.003, 0.0),
        "wind_speed": (6.0, 0.0),
        "wind_azimuth_deg": (90.0, 0.0),
    }
    p, wind, _ = sample(np.random.default_rng(0), spec=spec)
    assert p.thrust_misalign == pytest.approx((0.0, 0.01))
    assert p.cg_offset == pytest.approx((0.003, 0.0))
    np.testing.assert_allclose(wind.mean_10m, [0.0, 6.0, 0.0], atol=1e-12)


def test_unknown_key_and_bad_distribution_raise():
    with pytest.raises(KeyError):
        sample(np.random.default_rng(0), spec={"not_a_param": (1.0, 0.1)})
    with pytest.raises(ValueError):
        sample(np.random.default_rng(0), spec={"dry_mass": (1.0, 0.1, "weird")})


def test_spread_matches_spec_statistically():
    rng = np.random.default_rng(0)
    masses = [sample(rng, spec={"dry_mass": (0.955, 0.03)})[0].dry_mass for _ in range(2000)]
    assert np.mean(masses) == pytest.approx(0.955, abs=0.005)
    assert np.std(masses) == pytest.approx(0.03, rel=0.1)


def test_env_randomizes_per_reset_and_is_reproducible():
    env = TVCRocketEnv(randomize=True)
    env.reset(seed=4)
    d1 = dict(env.drawn)
    p1 = env.params
    env.reset()  # next episode draws fresh values from the same stream
    assert env.drawn != d1
    env2 = TVCRocketEnv(randomize=True)
    env2.reset(seed=4)
    assert env2.drawn == d1 and env2.params == p1
    # Default (no randomization) leaves the nominal vehicle untouched.
    plain = TVCRocketEnv()
    plain.reset(seed=4)
    assert plain.params == VehicleParams() and plain.drawn == {}


def test_randomized_env_still_flies():
    env = TVCRocketEnv(randomize=True)
    env.reset(seed=0, options={"scenario": "pad"})
    for _ in range(50):
        _, _, terminated, _, info = env.step(np.zeros(2))
    assert not terminated and info["altitude"] > 0.0
