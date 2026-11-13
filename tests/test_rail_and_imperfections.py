from dataclasses import replace

import numpy as np
import pytest

from tvc.dynamics import thrust_force_torque, upright_quaternion
from tvc.motor import Motor
from tvc.params import G0, VehicleParams
from tvc.sim import Sim


def test_rocket_sits_on_pad_until_thrust_exceeds_weight():
    sim = Sim(dt=1e-3)
    motor = Motor(sim.p)
    assert sim.on_rail
    t_liftoff = None
    for _ in range(300):
        before = sim.x[2]
        sim.step((0.0, 0.0))
        m = motor.mass_properties(sim.t)[0]
        if t_liftoff is None and motor.thrust(sim.t) < m * G0:
            assert sim.x[2] == pytest.approx(before, abs=1e-12)  # no backward/downward slide
        if sim.x[2] > 1e-6 and t_liftoff is None:
            t_liftoff = sim.t
    assert t_liftoff is not None and sim.x[2] >= 0.0


def test_attitude_frozen_and_axis_motion_only_on_rail():
    sim = Sim(dt=1e-3)
    q0 = sim.x[6:10].copy()
    steps = 0
    while sim.on_rail and steps < 3000:
        sim.step((0.1, 0.1))  # big gimbal commands must not rotate the rocket on the rail
        np.testing.assert_allclose(sim.x[6:10], q0, atol=1e-12)
        np.testing.assert_allclose(sim.x[10:13], 0.0, atol=1e-12)
        np.testing.assert_allclose(sim.x[0:2], 0.0, atol=1e-12)
        steps += 1
    assert not sim.on_rail


def test_rail_exit_after_effective_travel_with_sensible_speed():
    sim = Sim(dt=1e-3)
    while sim.on_rail:
        sim.step((0.0, 0.0))
    p = sim.p
    assert sim.rail_travel >= p.rail_length - p.rail_exit_offset
    assert sim.rail_travel < p.rail_length  # exits promptly, not far beyond
    # Energy sanity: v^2 ~ 2 a s with a ~ (T/m - g) ~ 10-20 m/s^2 over 0.9 m.
    assert 3.0 < sim.rail_exit_speed < 9.0


def test_explicit_state_start_skips_rail_and_rail_can_be_disabled():
    x0 = np.zeros(13)
    x0[6:10] = upright_quaternion()
    sim = Sim(dt=1e-3)
    sim.reset(x0)
    assert not sim.on_rail
    sim2 = Sim(replace(VehicleParams(), rail_length=0.0), dt=1e-3)
    assert not sim2.on_rail


def test_thrust_misalignment_creates_torque_even_at_zero_gimbal():
    p = replace(VehicleParams(), thrust_misalign=(0.0, 0.02))
    sim = Sim(p, dt=1e-3)
    x0 = np.zeros(13)
    x0[6:10] = upright_quaternion()
    sim.reset(x0, t0=1.0)
    for _ in range(50):
        sim.step((0.0, 0.0))
    assert sim.x[12] < 0  # phi > 0 gives negative torque about z
    assert abs(sim.x[11]) < 1e-9


def test_cg_offset_torque_is_offset_times_thrust():
    d, T, cy = 0.25, 20.0, 0.004
    force, torque = thrust_force_torque(T, (0.0, 0.0), d, cg_offset=(cy, 0.0))
    np.testing.assert_allclose(force, [T, 0, 0])
    # pivot at [-d, -cy, 0] relative to CoM; tau_z = r_x*F_y - r_y*F_x = +cy*T
    assert torque[2] == pytest.approx(cy * T)
    assert torque[0] == pytest.approx(0.0) and torque[1] == pytest.approx(0.0)
