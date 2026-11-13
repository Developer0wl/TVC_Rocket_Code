import numpy as np
import pytest

from tvc.estimator import G0, AttitudeEstimator, _rotate

DEG = np.pi / 180


def angle(a, b):
    return float(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def test_rotate_matches_vector_kinematics():
    # u' = -w x u: rotating about body y at 1 rad/s turns x toward +z... check against Euler step
    u = np.array([1.0, 0.0, 0.0])
    w = np.array([0.0, 1.0, 0.0])
    dt = 1e-6
    np.testing.assert_allclose(_rotate(u, w * dt), u - np.cross(w, u) * dt, atol=1e-10)
    # Full 90 degrees about y: x -> +z (since -y x x = +z)
    np.testing.assert_allclose(_rotate(u, w * (np.pi / 2)), [0, 0, 1], atol=1e-12)


def test_pad_alignment_recovers_tilt_and_bias():
    tilt = 7 * DEG
    up_body = np.array([np.cos(tilt), np.sin(tilt), 0.0])
    rng = np.random.default_rng(0)
    bias = np.array([0.003, -0.002, 0.001])
    gyro = bias + rng.normal(0, 0.002, (500, 3))
    accel = G0 * up_body + rng.normal(0, 0.05, (500, 3))
    est = AttitudeEstimator()
    est.align(gyro, accel)
    assert angle(est.u, up_body) < 0.3 * DEG
    np.testing.assert_allclose(est.bias, bias, atol=3e-4)


def test_gyro_integration_tracks_known_rotation():
    est = AttitudeEstimator()
    est.reset(up_body=(1, 0, 0))
    w = np.array([0.0, 0.5, 0.2])
    dt, true_u = 1e-3, np.array([1.0, 0.0, 0.0])
    accel = np.array([30.0, 0.0, 0.0])  # thrust-dominated: gate closed, no correction
    for _ in range(1000):
        true_u = _rotate(true_u, w * dt)
        est.update(w, accel, dt)
    assert angle(est.u, true_u) < 1e-6


def test_gravity_correction_is_gated_off_under_thrust():
    est = AttitudeEstimator()
    assert est.gate(np.array([G0, 0, 0])) == pytest.approx(1.0)
    assert est.gate(np.array([30.0, 0, 0])) < 1e-6
    est.reset(up_body=(1, 0, 0))
    wrong = np.array([20.0, 8.0, 0.0])  # a large lateral "measurement" during boost
    for _ in range(500):
        est.update(np.zeros(3), wrong, 1e-3)
    np.testing.assert_allclose(est.u, [1, 0, 0], atol=1e-9)


def test_stationary_correction_pulls_in_a_wrong_initial_guess():
    tilt = 10 * DEG
    up_body = np.array([np.cos(tilt), np.sin(tilt), 0.0])
    est = AttitudeEstimator()
    est.reset(up_body=(1, 0, 0))  # starts 10 deg off
    for _ in range(3000):
        est.update(np.zeros(3), G0 * up_body, 1e-3)
    assert angle(est.u, up_body) < 0.2 * DEG


def test_bias_is_learned_while_stationary_with_gravity_correction():
    up_body = np.array([1.0, 0.0, 0.0])
    bias = np.array([0.0, 0.01, -0.01])  # ~0.6 deg/s
    est = AttitudeEstimator(kp=2.0, ki=0.5)
    est.reset(up_body=up_body)
    for _ in range(40000):  # 40 s
        est.update(bias, G0 * up_body, 1e-3)
    np.testing.assert_allclose(est.bias[1:], bias[1:], atol=1.5e-3)
    assert angle(est.u, up_body) < 0.2 * DEG


def test_output_is_unit_vector_and_rates_are_bias_corrected():
    est = AttitudeEstimator()
    est.reset(up_body=(1, 0, 0), bias=(0.0, 0.1, 0.0))
    u, rates = est.update(np.array([0.0, 0.3, 0.0]), np.array([30.0, 0, 0]), 1e-3)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    np.testing.assert_allclose(rates, [0.0, 0.2, 0.0])
