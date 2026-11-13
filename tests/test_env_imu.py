import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from tvc.controllers.pid import PIDController
from tvc.dynamics import quat_to_rot
from tvc.env import ENV_VERSION, TVCRocketEnv
from tvc.params import VehicleParams
from tvc.residual import ResidualPIDWrapper


def test_imu_env_passes_checker_and_has_11_dim_obs_without_true_state():
    env = TVCRocketEnv(sensing="imu", randomize=True)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=1)
    assert obs.shape == (11,) and info["sensing"] == "imu"
    assert ENV_VERSION == "v1"
    # true-state mode keeps the v0 layout
    assert TVCRocketEnv().reset(seed=1)[0].shape == (14,)


def test_invalid_sensing_mode_rejected():
    with pytest.raises(ValueError):
        TVCRocketEnv(sensing="gps")


def test_alignment_gives_small_attitude_error_at_reset():
    env = TVCRocketEnv(sensing="imu", randomize=True)
    errs = []
    for seed in range(8):
        env.reset(seed=seed, options={"tilt_deg": 10.0, "azimuth_deg": 30.0})
        up = quat_to_rot(env.sim.x[6:10]).T @ np.array([0.0, 0.0, 1.0])
        errs.append(np.degrees(np.arccos(np.clip(np.dot(env.est.u, up), -1, 1))))
    assert max(errs) < 1.5 and np.mean(errs) < 0.8


def test_imu_env_is_reproducible_with_seed():
    def rollout(seed):
        env = TVCRocketEnv(sensing="imu", randomize=True)
        obs, _ = env.reset(seed=seed, options={"tilt_deg": 5.0})
        out = [obs]
        for _ in range(20):
            obs, *_ = env.step(np.array([0.1, -0.1]))
            out.append(obs)
        return np.array(out)

    np.testing.assert_array_equal(rollout(3), rollout(3))
    assert not np.array_equal(rollout(3), rollout(4))


def test_pid_on_estimates_still_flies_tilt_and_pad():
    for options in ({"tilt_deg": 10.0, "azimuth_deg": 30.0}, {"scenario": "pad"}):
        env = ResidualPIDWrapper(TVCRocketEnv(sensing="imu", randomize=True))
        obs, _ = env.reset(seed=2, options=options)
        assert obs.shape == (13,)  # 11 + PID command
        done = terminated = False
        peak, last = 0.0, 0.0
        while not done:
            obs, _, terminated, truncated, info = env.step(np.zeros(2))
            peak, last = max(peak, np.degrees(info["tilt"])), np.degrees(info["tilt"])
            done = terminated or truncated
        assert not terminated and last < 6.0


def test_pid_command_from_estimate_matches_true_state_command_when_exact():
    p = VehicleParams()
    ctrl_a, ctrl_b = PIDController(p), PIDController(p)
    env = TVCRocketEnv()
    env.reset(seed=0, options={"tilt_deg": 8.0, "azimuth_deg": 20.0})
    x, t = env.sim.x, env.sim.t
    up = quat_to_rot(x[6:10]).T @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        ctrl_a.command(x, t, 0.01), ctrl_b.command_from_estimate(up, x[10:13], t, 0.01)
    )
