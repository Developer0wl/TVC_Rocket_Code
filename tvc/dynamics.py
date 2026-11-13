"""6-DoF rigid-body dynamics with quaternion attitude and RK4 integration.

State x (13): position world (3), velocity world (3), quaternion [w,x,y,z] body->world (4),
body angular rate (3). World frame is z-up; body x points along the rocket axis to the nose.
"""

import numpy as np

from .atmosphere import WindModel, air_density
from .motor import Motor
from .params import G0, VehicleParams

GRAVITY = np.array([0.0, 0.0, -G0])


def quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that v_world = R @ v_body."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def upright_quaternion() -> np.ndarray:
    """Body x axis pointing along world +z (rocket vertical, nose up)."""
    a = -np.pi / 4  # half of -90 deg rotation about body/world y
    return np.array([np.cos(a), 0.0, np.sin(a), 0.0])


def thrust_force_torque(thrust: float, gimbal, d: float, cg_offset=(0.0, 0.0)):
    """Body-frame thrust force and torque about the CoM.

    gimbal = [theta (about y), phi (about z)]. Thrust direction is Rz(phi) Ry(theta) x_hat.
    The pivot sits a distance d aft of the CoM, and the CoM may sit off the geometric axis by
    cg_offset = [y, z], so the pivot is at [-d, -cy, -cz] relative to the CoM.
    """
    th, ph = gimbal
    direction = np.array([np.cos(ph) * np.cos(th), np.sin(ph) * np.cos(th), -np.sin(th)])
    force = thrust * direction
    torque = np.cross(np.array([-d, -cg_offset[0], -cg_offset[1]]), force)
    return force, torque


def aero_force_torque(p: VehicleParams, v_body: np.ndarray, w: np.ndarray, rho: float, d: float):
    """Axial drag plus a normal force at the CP.

    Normal force = linear lift term (cn_alpha, gives a restoring moment ~ alpha at small angles
    of attack) + quadratic cross-flow drag. The lateral velocity is evaluated AT the CP
    (v + w x r_cp), so body rotation produces aerodynamic damping.
    """
    r_cp = np.array([p.x_cp - d, -p.cg_offset[0], -p.cg_offset[1]])
    v_cp = v_body + np.cross(w, r_cp)
    f_axial = np.array(
        [-0.5 * rho * p.cd_axial * p.ref_area_axial * v_body[0] * abs(v_body[0]), 0, 0]
    )
    v_lat = np.array([0.0, v_cp[1], v_cp[2]])
    f_lift = -0.5 * rho * p.cn_alpha * p.ref_area_axial * abs(v_body[0]) * v_lat
    f_cross = -0.5 * rho * p.cd_normal * p.ref_area_side * np.linalg.norm(v_lat) * v_lat
    f_normal = f_lift + f_cross
    return f_axial + f_normal, np.cross(r_cp, f_normal)


def derivative(
    t: float,
    x: np.ndarray,
    gimbal,
    p: VehicleParams,
    motor: Motor,
    wind: WindModel,
) -> np.ndarray:
    vel, q, w = x[3:6], x[6:10], x[10:13]
    rot = quat_to_rot(q)
    m, d, inertia = motor.mass_properties(t)
    inertia_dot = motor.inertia_rate(t)

    gimbal_eff = np.asarray(gimbal, dtype=float) + np.asarray(p.thrust_misalign)
    f_t, tau_t = thrust_force_torque(motor.thrust(t), gimbal_eff, d, p.cg_offset)
    v_body = rot.T @ (vel - wind.velocity(t, x[2]))
    f_a, tau_a = aero_force_torque(p, v_body, w, air_density(x[2]), d)

    acc = rot @ (f_t + f_a) / m + GRAVITY
    # Euler's equations with time-varying inertia: I w' = tau - w x (I w) - I' w
    w_dot = (tau_t + tau_a - np.cross(w, inertia * w) - inertia_dot * w) / inertia
    q_dot = 0.5 * quat_mult(q, np.concatenate(([0.0], w)))

    out = np.empty(13)
    out[0:3] = vel
    out[3:6] = acc
    out[6:10] = q_dot
    out[10:13] = w_dot
    return out


def rk4(f, t: float, x: np.ndarray, dt: float) -> np.ndarray:
    """Classic RK4 for x' = f(t, x), then renormalize the quaternion block."""
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_new[6:10] /= np.linalg.norm(x_new[6:10])
    return x_new


def rk4_step(t: float, x: np.ndarray, dt: float, gimbal, p, motor, wind) -> np.ndarray:
    """One RK4 step with the gimbal angles held constant over the step."""
    return rk4(lambda tt, xx: derivative(tt, xx, gimbal, p, motor, wind), t, x, dt)
