from dataclasses import replace

import numpy as np
import pytest

from tvc.dynamics import upright_quaternion
from tvc.params import G0
from tvc.sensors import ACCEL, GYRO, IMU
from tvc.sim import Sim

DT = 1e-3


def run(imu, t_end, rates=(0.1, 0.0, 0.0), accel=(0.0, 0.0, G0)):
    out = [imu.read(k * DT, rates, accel) for k in range(int(t_end / DT))]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def test_ideal_imu_is_passthrough():
    g, a = run(IMU(rate=500, latency=0.0, ideal=True), 0.1)
    np.testing.assert_allclose(g[5:], np.tile([0.1, 0, 0], (95, 1)))
    np.testing.assert_allclose(a[5:], np.tile([0, 0, G0], (95, 1)))


def test_sample_and_hold_at_rate():
    imu = IMU(rate=100, latency=0.0, ideal=True)
    seq = [imu.read(k * DT, (k * 0.01, 0, 0), (0, 0, 0))[0][0] for k in range(100)]
    assert 8 <= np.count_nonzero(np.diff(seq)) <= 10  # ~10 samples in 0.1 s, held between


def test_latency_delays_signal():
    imu = IMU(rate=1000, latency=0.010, ideal=True)
    arrival = None
    for k in range(40):
        g, _ = imu.read(k * DT, (1.0, 0, 0), (0, 0, 0))
        if g[0] == 1.0 and arrival is None:
            arrival = k
    assert arrival == 10


def test_noise_std_matches_density():
    gyro = replace(
        GYRO, noise_density=0.01, bias_const_std=0.0, bias_walk=0.0, range=100.0, resolution=0.0
    )
    g, _ = run(IMU(rate=500, latency=0.0, gyro=gyro, seed=1), 40.0, rates=(0, 0, 0))
    assert g[::2].std() == pytest.approx(0.01 * np.sqrt(500 / 2), rel=0.05)


def test_saturation_and_quantization():
    spec = replace(
        GYRO, noise_density=0.0, bias_const_std=0.0, bias_walk=0.0, range=1.0, resolution=0.25
    )
    g, _ = run(IMU(rate=1000, latency=0.0, gyro=spec, seed=0), 0.05, rates=(5.0, 0.3, -0.37))
    np.testing.assert_allclose(g[-1], [1.0, 0.25, -0.25])


def test_bias_random_walk_grows_like_sqrt_t():
    spec = replace(
        GYRO, noise_density=0.0, bias_const_std=0.0, bias_walk=0.1, range=1e3, resolution=0.0
    )
    ends = [
        run(IMU(rate=100, latency=0.0, gyro=spec, seed=s), 4.0, rates=(0, 0, 0))[0][-1, 0]
        for s in range(60)
    ]
    assert np.std(ends) == pytest.approx(0.1 * np.sqrt(4.0), rel=0.35)


def test_seed_reproducible_and_different_seeds_differ():
    a, b, c = (run(IMU(seed=s), 0.2)[0] for s in (5, 5, 6))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sim_specific_force_pad_free_fall_and_burn():
    sim = Sim(dt=DT)  # on the pad, thrust ~0: reads +g along body x (up)
    np.testing.assert_allclose(sim.specific_force(), [G0, 0, 0], atol=1e-9)
    x0 = np.zeros(13)
    x0[6:10] = upright_quaternion()
    sim.reset(x0, t0=100.0)  # engine cold, free fall: reads ~0
    np.testing.assert_allclose(sim.specific_force(), 0.0, atol=1e-9)
    sim.reset(x0, t0=1.0)  # burning: reads thrust/mass along body x
    m = sim.motor.mass_properties(1.0)[0]
    assert sim.specific_force()[0] == pytest.approx(sim.motor.thrust(1.0) / m, rel=1e-6)


def test_default_accel_spec_sane():
    assert ACCEL.range > 10 * G0
