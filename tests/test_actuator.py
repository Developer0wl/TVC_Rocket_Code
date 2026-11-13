from dataclasses import replace

import numpy as np
import pytest

from tvc.actuator import GimbalServo
from tvc.params import VehicleParams

DT = 1e-3


def run(servo, cmd, steps):
    out = []
    for _ in range(steps):
        out.append(servo.step(cmd))
    return np.array(out)


def test_angle_limit_is_enforced():
    p = VehicleParams()
    servo = GimbalServo(p, DT)
    out = run(servo, (1.0, -1.0), 2000)
    assert out[-1, 0] == pytest.approx(p.gimbal_max, abs=1e-6)
    assert out[-1, 1] == pytest.approx(-p.gimbal_max, abs=1e-6)


def test_rate_limit_is_enforced():
    p = replace(VehicleParams(), servo_rate_max=2.0, servo_tau=0.001)
    servo = GimbalServo(p, DT)
    out = run(servo, (p.gimbal_max, 0.0), 50)
    rates = np.abs(np.diff(out[:, 0])) / DT
    assert rates.max() <= 2.0 + 1e-9


def test_command_delay():
    p = replace(VehicleParams(), servo_delay=0.010, servo_tau=0.001)
    servo = GimbalServo(p, DT)
    out = run(servo, (0.05, 0.0), 30)
    first_move = int(np.argmax(np.abs(out[:, 0]) > 1e-9))
    assert first_move == 10  # delay_steps = 10


def test_first_order_lag_time_constant():
    p = replace(VehicleParams(), servo_delay=0.0, servo_tau=0.05, servo_rate_max=1e3)
    servo = GimbalServo(p, DT)
    cmd = 0.05  # within the 8 deg limit and the rate cap, so a pure linear step response
    out = run(servo, (cmd, 0.0), int(0.05 / DT))
    assert out[-1, 0] == pytest.approx(cmd * (1 - np.exp(-1)), rel=0.02)
