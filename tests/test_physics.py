from dataclasses import replace

import numpy as np
import pytest

from tvc.atmosphere import WindModel, air_density
from tvc.dynamics import (
    GRAVITY,
    aero_force_torque,
    derivative,
    quat_to_rot,
    rk4_step,
    thrust_force_torque,
    upright_quaternion,
)
from tvc.motor import Motor
from tvc.params import G0, VehicleParams
from tvc.sim import Sim

NO_AERO = replace(VehicleParams(), cd_axial=0.0, cd_normal=0.0, cn_alpha=0.0)
T_AFTER_BURN = 100.0  # motor is cold: pure rigid body + gravity


def upright_state():
    x = np.zeros(13)
    x[6:10] = upright_quaternion()
    return x


def test_upright_quaternion_points_x_up():
    axis_world = quat_to_rot(upright_quaternion()) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(axis_world, [0, 0, 1], atol=1e-12)


def test_free_fall_matches_analytic():
    motor, wind = Motor(NO_AERO), WindModel()
    x = upright_state()
    x[2] = 100.0
    t, dt = T_AFTER_BURN, 1e-3
    for _ in range(1000):
        x = rk4_step(t, x, dt, (0, 0), NO_AERO, motor, wind)
        t += dt
    np.testing.assert_allclose(x[2], 100.0 - 0.5 * G0 * 1.0**2, atol=1e-9)
    np.testing.assert_allclose(x[5], -G0 * 1.0, atol=1e-9)


def test_quaternion_stays_unit_norm_under_tumble():
    motor, wind = Motor(NO_AERO), WindModel()
    x = upright_state()
    x[10:13] = [3.0, 7.0, -5.0]
    t, dt = T_AFTER_BURN, 1e-3
    for _ in range(5000):
        x = rk4_step(t, x, dt, (0, 0), NO_AERO, motor, wind)
        t += dt
    assert abs(np.linalg.norm(x[6:10]) - 1.0) < 1e-12


def test_torque_free_conserves_angular_momentum_and_energy():
    motor, wind = Motor(NO_AERO), WindModel()
    _, _, inertia = motor.mass_properties(T_AFTER_BURN)
    x = upright_state()
    x[10:13] = [2.0, 4.0, -3.0]

    def invariants(s):
        L_world = quat_to_rot(s[6:10]) @ (inertia * s[10:13])
        return L_world, 0.5 * np.dot(s[10:13], inertia * s[10:13])

    L0, E0 = invariants(x)
    t, dt = T_AFTER_BURN, 1e-3
    for _ in range(5000):
        x = rk4_step(t, x, dt, (0, 0), NO_AERO, motor, wind)
        t += dt
    L1, E1 = invariants(x)
    np.testing.assert_allclose(L1, L0, rtol=1e-8)
    assert E1 == pytest.approx(E0, rel=1e-8)


def test_zero_gimbal_thrust_is_pure_axial_with_no_torque():
    force, torque = thrust_force_torque(25.0, (0.0, 0.0), d=0.25)
    np.testing.assert_allclose(force, [25.0, 0, 0])
    np.testing.assert_allclose(torque, 0, atol=1e-12)


def test_gimbal_torque_sign_and_magnitude():
    d, T = 0.25, 25.0
    # Positive phi (about z) pushes the pivot toward +y; pivot is aft => torque about -z.
    force, torque = thrust_force_torque(T, (0.0, 0.1), d)
    assert force[1] > 0
    assert torque[2] == pytest.approx(-d * force[1])
    assert torque[0] == pytest.approx(0.0) and torque[1] == pytest.approx(0.0)
    # Positive theta (about y) pushes thrust toward -z body; the aft pivot goes down, the nose
    # goes up (+z), which is a NEGATIVE rotation about y (x -> -z is positive about y).
    force, torque = thrust_force_torque(T, (0.1, 0.0), d)
    assert force[2] < 0
    assert torque[1] == pytest.approx(d * force[2])
    assert torque[1] < 0


def test_vertical_thrust_acceleration_at_zero_gimbal():
    p, t = NO_AERO, 1.0
    motor = Motor(p)
    a = derivative(t, upright_state(), (0, 0), p, motor, WindModel())[3:6]
    m, _, _ = motor.mass_properties(t)
    np.testing.assert_allclose(a, [0, 0, motor.thrust(t) / m - G0], atol=1e-9)


def test_short_burst_angular_acceleration_matches_torque_over_inertia():
    p, t0, dt = NO_AERO, 1.0, 1e-3
    motor = Motor(p)
    sim = Sim(p, dt=dt)
    sim.servo.delay_steps = 0  # apply gimbal instantly for this check
    sim.servo.p = replace(p, servo_tau=1e-6, servo_rate_max=1e6)
    sim.reset(upright_state(), t0=t0)
    gimbal = np.array([0.0, 0.05])
    for _ in range(10):
        sim.step(gimbal)
    m, d, inertia = motor.mass_properties(t0)
    _, tau = thrust_force_torque(motor.thrust(t0 + 0.005), gimbal, d)
    expected_wz = tau[2] / inertia[2] * 10 * dt
    assert sim.x[12] == pytest.approx(expected_wz, rel=0.05)


def test_mass_burn_matches_thrust_impulse():
    p = VehicleParams()
    motor = Motor(p)
    assert motor.mass_properties(0.0)[0] == pytest.approx(p.dry_mass + p.motor.prop_mass)
    assert motor.mass_properties(motor.burn_time + 1)[0] == pytest.approx(p.dry_mass)
    # Mid-burn mass drop is proportional to impulse delivered.
    t = 1.5
    frac = motor.impulse_to(t) / motor.total_impulse
    assert motor.prop_mass(t) == pytest.approx(p.motor.prop_mass * (1 - frac))
    # Exact impulse agrees with brute-force trapezoid integration.
    ts = np.linspace(0, t, 20001)
    brute = np.trapezoid([motor.thrust(s) for s in ts], ts)
    assert motor.impulse_to(t) == pytest.approx(brute, rel=1e-4)


def test_cg_moves_aft_as_propellant_burns():
    motor = Motor(VehicleParams())
    d_start = motor.mass_properties(0.0)[1]
    d_end = motor.mass_properties(10.0)[1]
    assert d_end > d_start  # propellant sits aft of the dry CoM, so CoM moves forward


def test_upright_flight_stays_upright_and_climbs():
    sim = Sim(dt=1e-3)
    for _ in range(1000):
        sim.step((0.0, 0.0))
    assert sim.x[2] > 1.0
    assert sim.tilt < 1e-6
    np.testing.assert_allclose(sim.x[[0, 1]], 0.0, atol=1e-9)


def test_stable_weathercock_under_crosswind():
    # CP aft of CoM: a crosswind should rotate the nose into the wind, not away from it.
    wind = WindModel(mean_10m=np.array([0.0, 8.0, 0.0]))
    sim = Sim(replace(VehicleParams(), cd_axial=0.5), wind=wind, dt=1e-3)
    x0 = upright_state()
    x0[2] = 50.0
    x0[5] = 30.0
    sim.reset(x0, t0=T_AFTER_BURN)
    for _ in range(200):
        sim.step((0.0, 0.0))
    # Wind blows toward +y, so it comes from -y; the nose should turn into it (toward -y).
    nose_world = quat_to_rot(sim.x[6:10]) @ np.array([1.0, 0, 0])
    assert nose_world[1] < 0


def test_air_density_decreases_with_altitude():
    assert air_density(0) == pytest.approx(1.225, rel=0.01)
    assert air_density(1000) < air_density(0)


def test_gravity_constant():
    assert GRAVITY[2] == -G0


# --- Added after physics review ---


def test_variable_inertia_term_conserves_angular_momentum_during_burn():
    # Spin about a principal axis with no torque: d(I w)/dt = 0 only if the I' term is present.
    p, motor, wind = NO_AERO, Motor(NO_AERO), WindModel()
    x = upright_state()
    x[12] = 0.5
    t, dt = 0.5, 1e-3
    h0 = motor.mass_properties(t)[2][2] * x[12]
    for _ in range(2000):
        x = rk4_step(t, x, dt, (0, 0), p, motor, wind)
        t += dt
    i_end = motor.mass_properties(t)[2][2]
    assert i_end < motor.mass_properties(0.5)[2][2] * 0.99  # inertia really changed
    assert i_end * x[12] == pytest.approx(h0, rel=1e-5)


@pytest.mark.parametrize("axis,sign", [(0, -1.0), (1, -1.0)])
def test_gimbal_axes_drive_expected_body_rates(axis, sign):
    # axis 0 = theta (about y, torque_y = d*Fz < 0); axis 1 = phi (about z, torque_z = -d*Fy < 0).
    p, t0, dt = NO_AERO, 1.0, 1e-3
    motor = Motor(p)
    sim = Sim(p, dt=dt)
    sim.servo.delay_steps = 0
    sim.servo.p = replace(p, servo_tau=1e-6, servo_rate_max=1e6)
    sim.reset(upright_state(), t0=t0)
    gimbal = np.zeros(2)
    gimbal[axis] = 0.05
    for _ in range(10):
        sim.step(gimbal)
    m, d, inertia = motor.mass_properties(t0)
    _, tau = thrust_force_torque(motor.thrust(t0 + 0.005), gimbal, d)
    rate_idx, tau_idx = (11, 1) if axis == 0 else (12, 2)
    expected = tau[tau_idx] / inertia[tau_idx] * 10 * dt
    assert np.sign(sim.x[rate_idx]) == sign
    assert sim.x[rate_idx] == pytest.approx(expected, rel=0.05)
    other = 12 if axis == 0 else 11
    assert abs(sim.x[other]) < 1e-9  # axes are independent


def test_axial_drag_magnitude_matches_formula():
    p = replace(NO_AERO, cd_axial=0.5)
    motor, wind = Motor(p), WindModel()
    x = upright_state()
    x[2], x[5] = 0.0, -50.0  # falling nose-up at 50 m/s, sea level
    a = derivative(T_AFTER_BURN, x, (0, 0), p, motor, wind)[3:6]
    m = motor.mass_properties(T_AFTER_BURN)[0]
    drag = 0.5 * air_density(0.0) * 0.5 * p.ref_area_axial * 50.0**2
    np.testing.assert_allclose(a, [0, 0, drag / m - G0], rtol=1e-9, atol=1e-12)


def test_aero_restoring_moment_is_linear_in_alpha_and_stabilizing():
    p = replace(VehicleParams(), cd_normal=0.0)  # isolate the linear cn_alpha term
    rho, d = 1.225, 0.27
    speed = 100.0

    def moment(alpha):
        v = np.array([speed * np.cos(alpha), 0.0, speed * np.sin(alpha)])
        return aero_force_torque(p, v, np.zeros(3), rho, d)[1][1]

    m1, m2 = moment(np.deg2rad(2)), moment(np.deg2rad(4))
    assert m2 == pytest.approx(2 * m1, rel=0.01)
    # Lateral air-relative velocity +z at the aft CP gives a -z force, hence a negative torque
    # about y. Negative rotation about y turns the nose toward +z, i.e. toward the velocity
    # vector, which is the stabilizing (weathercock) direction.
    assert m1 < 0


def test_pitch_rate_is_damped_by_aero():
    p = replace(VehicleParams(), cd_normal=0.0)
    w = np.array([0.0, 2.0, 0.0])  # pitch rate about +y
    v = np.array([80.0, 0.0, 0.0])
    tau = aero_force_torque(p, v, w, 1.225, 0.27)[1]
    assert tau[1] * w[1] < 0  # opposes the rotation


def test_wind_shear_and_gust_shape():
    wm = WindModel(
        mean_10m=np.array([5.0, 0, 0]),
        gust_amplitude=4.0,
        gust_start=1.0,
        gust_duration=2.0,
        gust_direction=np.array([0.0, 1.0, 0.0]),
    )
    assert wm.velocity(0.0, 10.0)[0] == pytest.approx(5.0)
    assert wm.velocity(0.0, 40.0)[0] == pytest.approx(5.0 * 4.0 ** (1 / 7))
    assert wm.velocity(2.0, 10.0)[1] == pytest.approx(4.0)  # mid-gust peak
    assert wm.velocity(0.5, 10.0)[1] == 0.0 and wm.velocity(3.5, 10.0)[1] == 0.0


def test_air_density_tropopause():
    assert air_density(11000.0) == pytest.approx(0.3639, rel=0.01)


def test_tilt_matches_known_angle_and_reset_normalizes():
    sim = Sim(dt=1e-3)
    x0 = upright_state()
    tilt = np.deg2rad(10.0)
    # Pitch the nose by `tilt` about world y (rotation of the upright body toward +x world).
    half = tilt / 2
    dq = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    from tvc.dynamics import quat_mult

    x0[6:10] = quat_mult(dq, x0[6:10]) * 3.0  # deliberately unnormalized
    sim.reset(x0)
    assert np.linalg.norm(sim.x[6:10]) == pytest.approx(1.0)
    assert sim.tilt == pytest.approx(tilt, rel=1e-9)
