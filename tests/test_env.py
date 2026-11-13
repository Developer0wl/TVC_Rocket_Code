import numpy as np
from gymnasium.utils.env_checker import check_env

from tvc.env import TVCRocketEnv


def test_env_passes_gymnasium_checker():
    check_env(TVCRocketEnv(), skip_render_check=True)


def test_reset_is_reproducible_with_seed():
    a, _ = TVCRocketEnv().reset(seed=3)
    b, _ = TVCRocketEnv().reset(seed=3)
    np.testing.assert_array_equal(a, b)


def test_pad_start_runs_on_rail_then_flies():
    env = TVCRocketEnv()
    env.reset(options={"scenario": "pad"})
    assert env.unwrapped.sim.on_rail
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        if terminated or truncated:
            break
    assert not info["on_rail"] and info["altitude"] > 1.0 and not terminated


def test_large_tilt_terminates():
    env = TVCRocketEnv()
    env.reset(options={"tilt_deg": 40.0, "azimuth_deg": 0.0})
    terminated = False
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        if terminated or truncated:
            break
    assert terminated  # uncontrolled, it tumbles past the 45 deg limit
