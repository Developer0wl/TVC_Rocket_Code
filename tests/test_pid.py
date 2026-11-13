import numpy as np

from tvc.atmosphere import WindModel
from tvc.controllers.pid import PIDController
from tvc.env import TVCRocketEnv
from tvc.params import VehicleParams


def _fly(controlled: bool, tilt_deg: float, wind: WindModel):
    p = VehicleParams()
    env = TVCRocketEnv(params=p, wind=wind)
    ctrl = PIDController(p)
    env.reset(seed=0, options={"tilt_deg": tilt_deg, "azimuth_deg": 30.0})
    sim = env.unwrapped.sim
    tilts, terminated, truncated = [], False, False
    while not (terminated or truncated):
        cmd = ctrl.command(sim.x, sim.t, env.control_dt) if controlled else np.zeros(2)
        _, _, terminated, truncated, info = env.step(cmd / p.gimbal_max)
        tilts.append((sim.t, np.rad2deg(info["tilt"])))
    return np.array(tilts), terminated


def test_pid_recovers_from_tilt_and_holds_under_gust():
    wind = WindModel(
        mean_10m=np.array([2.0, 0.0, 0.0]),
        gust_amplitude=5.0,
        gust_start=1.0,
        gust_duration=1.0,
        gust_direction=np.array([0.0, 1.0, 0.0]),
    )
    tilts, terminated = _fly(True, 10.0, wind)
    assert not terminated
    assert tilts[:, 1].max() <= 10.5  # never worse than the initial tilt
    after_recovery = tilts[tilts[:, 0] > 0.9, 1]
    assert after_recovery.max() < 4.0  # holds attitude through the gust


def test_pid_beats_uncontrolled_on_a_large_tilt():
    wind = WindModel()
    controlled, term_c = _fly(True, 20.0, wind)
    free, _ = _fly(False, 20.0, wind)
    assert not term_c
    assert controlled[:, 1].max() < free[:, 1].max()
    assert controlled[-1, 1] < 3.0
