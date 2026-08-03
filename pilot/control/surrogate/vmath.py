"""Batched versions of the quaternion algebra in `plant.py`, and nothing else.

Every function here is the `[n, ...]` form of a function in `plant.py` and is written to
produce bit-identical results for n = 1, because `selfcheck.py` compares the surrogate's
integrator against `plant.Sim` directly and any rewrite of the algebra would turn that
test into a test of the rewrite.

**No sign flips live in this file.** The simulator's -1 rate mirror belongs to the link
layer (`CONVENTIONS.md`); everything in the surrogate is canonical body/world NED.
"""

import numpy as np


def quat_to_rot(q):
    """Body -> world rotation matrices from (w, x, y, z) quaternions `[n, 4]`."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty(q.shape[:1] + (3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def quat_integrate(q, omega_body, dt):
    """Advance `[n, 4]` quaternions by `[n, 3]` body rates over `[n]` (or scalar) dt."""
    q = np.asarray(q, dtype=np.float64)
    o = np.asarray(omega_body, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1, 1) if np.ndim(dt) else dt
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    p, r, s = o[:, 0], o[:, 1], o[:, 2]
    dq = 0.5 * np.stack([-x * p - y * r - z * s,
                         w * p + y * s - z * r,
                         w * r - x * s + z * p,
                         w * s + x * r - y * p], axis=1)
    out = q + dq * dt
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def quat_from_euler(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) Euler angles `[n]` -> `[n, 4]` quaternions."""
    roll = np.asarray(roll, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy], axis=1)


def euler_from_rot(r):
    """(roll, pitch, yaw) from body->world rotation matrices `[n, 3, 3]`.

    Roll and pitch are what a gravity vector observes; yaw is used only inside the
    surrogate (course construction, start states) and never reaches the observation.
    """
    roll = np.arctan2(r[:, 2, 1], r[:, 2, 2])
    pitch = -np.arcsin(np.clip(r[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(r[:, 1, 0], r[:, 0, 0])
    return roll, pitch, yaw


def rot_apply(r, v):
    """world = R @ v_body, batched."""
    return np.einsum('nij,nj->ni', r, v)


def rot_apply_t(r, v):
    """body = R^T @ v_world, batched."""
    return np.einsum('nji,nj->ni', r, v)


def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)
