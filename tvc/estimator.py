"""Attitude estimator for the controller: world-up direction in the body frame + body rates.

Why not a plain accelerometer tilt estimate: under thrust the accelerometer reads ~T/m along the
body axis regardless of tilt, so it carries almost no tilt information during boost. This filter:

1. aligns on the pad (stationary: accelerometer reads +g along up, gyro mean = bias);
2. dead-reckons with the bias-corrected gyro through boost;
3. applies a gravity correction (Mahony-style, vector form) only while |accel| ~ g, with a smooth
   gate, and learns the gyro bias from that correction when it is active.

State: unit vector `u` (world +z expressed in the body frame) and gyro bias `b`.
    u' = -(w_meas - b) x u + kp * gate * (u_m - u (u . u_m))
    b' = ki * gate * (u x u_m)
Yaw is unobservable from gravity and irrelevant to the pitch/yaw gimbal control, so it is not
estimated.
"""

from __future__ import annotations

import numpy as np

G0 = 9.80665


def _rotate(u: np.ndarray, omega_dt: np.ndarray) -> np.ndarray:
    """Rotate vector u by the rotation vector -omega_dt (Rodrigues), matching u' = -omega x u."""
    theta = np.linalg.norm(omega_dt)
    if theta < 1e-12:
        return u - np.cross(omega_dt, u)
    k = -omega_dt / theta
    return (
        u * np.cos(theta) + np.cross(k, u) * np.sin(theta) + k * np.dot(k, u) * (1 - np.cos(theta))
    )


class AttitudeEstimator:
    def __init__(self, kp: float = 2.0, ki: float = 0.05, gate_width: float = 0.15):
        self.kp, self.ki, self.gate_width = kp, ki, gate_width
        self.reset()

    def reset(self, up_body=(1.0, 0.0, 0.0), bias=(0.0, 0.0, 0.0)):
        u = np.asarray(up_body, dtype=float)
        self.u = u / np.linalg.norm(u)
        self.bias = np.asarray(bias, dtype=float).copy()
        self.rates = np.zeros(3)

    def align(self, gyro_samples, accel_samples):
        """Pad alignment from a stationary window (Nx3 arrays)."""
        accel = np.mean(np.asarray(accel_samples, dtype=float), axis=0)
        self.reset(accel / np.linalg.norm(accel), np.mean(np.asarray(gyro_samples), axis=0))

    def gate(self, accel: np.ndarray) -> float:
        """~1 when |accel| ~ g (gravity dominates), ~0 under thrust."""
        err = (np.linalg.norm(accel) - G0) / (self.gate_width * G0)
        return float(np.exp(-err * err))

    def update(self, gyro, accel, dt: float):
        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)
        self.rates = gyro - self.bias
        u = _rotate(self.u, self.rates * dt)
        w = self.gate(accel)
        if w > 1e-4:
            u_m = accel / np.linalg.norm(accel)
            u = u + self.kp * w * (u_m - u * np.dot(u, u_m)) * dt
            self.bias = self.bias + self.ki * w * np.cross(u, u_m) * dt
        self.u = u / np.linalg.norm(u)
        return self.u.copy(), self.rates.copy()
