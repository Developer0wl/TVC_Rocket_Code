import numpy as np
from gymnasium.utils.env_checker import check_env

from tvc.controllers.pid import PIDController
from tvc.env import TVCRocketEnv
from tvc.params import VehicleParams
from tvc.residual import ResidualPIDWrapper


def _rollout(env, policy, seed=0):
    obs, _ = env.reset(seed=seed, options={"tilt_deg": 10.0, "azimuth_deg": 30.0})
    states, done = [], False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs))
        states.append(info["tilt"])
        done = term or trunc
    return np.array(states)


def test_zero_residual_reproduces_pid_exactly():
    wrapped = ResidualPIDWrapper(TVCRocketEnv(), scale=0.3)
    res = _rollout(wrapped, lambda o: np.zeros(2))

    env = TVCRocketEnv()
    pid = PIDController(VehicleParams())
    obs, _ = env.reset(seed=0, options={"tilt_deg": 10.0, "azimuth_deg": 30.0})
    pid.reset()
    sim, ref, done = env.sim, [], False
    while not done:
        cmd = pid.command(sim.x, sim.t, env.control_dt) / env.params.gimbal_max
        _, _, term, trunc, info = env.step(cmd)
        ref.append(info["tilt"])
        done = term or trunc
    np.testing.assert_allclose(res, ref, atol=1e-12)


def test_residual_is_scaled_and_total_is_clipped():
    wrapped = ResidualPIDWrapper(TVCRocketEnv(), scale=0.3)
    wrapped.reset(seed=0, options={"tilt_deg": 10.0})
    _, _, _, _, info = wrapped.step(np.array([1.0, -1.0]))
    expected = np.clip(info["pid_action"] + 0.3 * np.array([1.0, -1.0]), -1, 1)
    np.testing.assert_allclose(info["total_action"], expected)
    assert np.all(np.abs(info["total_action"]) <= 1.0)


def test_wrapped_env_passes_checker_and_exposes_pid_in_obs():
    env = ResidualPIDWrapper(TVCRocketEnv(randomize=True))
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (16,)
