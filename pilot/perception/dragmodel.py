"""The drag speed model for VQ2, and the estimator that inverts it.

WHAT THE DATA SAYS (dragfit_report.txt; VQ1 truth pose, airborne frames, contact dropped).
The horizontal body specific force is quadratic in the IN-PLANE speed component, not in
the total speed:

    a_body[0:2] = -k * |v_xy| * v_xy          v_xy = v_body[0:2],  k ~ 0.044 /m

Binning by total speed |v| makes the implied coefficient SATURATE above ~14 m/s and looks
like a broken law; binning by |v_xy| makes it flat to within a few percent over 2-25 m/s.
That is anisotropic drag -- the airframe has its own coefficient in the rotor plane -- and
it matters because it turns the inversion into an exact algebraic one with no assumption
in it at all:

    |v_xy| = sqrt(|a_xy| / k)                 direction: -a_xy / |a_xy|

which is precisely the quantity producer._selfobs already publishes the BEARING of.
No integration, no filter, nothing to diverge -- the same standing that vel_bearing_rad has.

THE THIRD COMPONENT IS REFUSED, and that refusal is the main design decision in this file.

Thrust acts along body -z, so body-z drag is buried under it -- the same fact that makes
the horizontal pair clean makes the third one blind. It is not negligible: at race pace
the airframe is pitched 20-40 deg, so the body-z component is a third to a half of the
total speed. The barometer that could supply it reads nan throughout this simulator.

The obvious substitute was tried and FAILED VALIDATION. Assume the velocity is
WORLD-HORIZONTAL, which in the body frame reads v_body . g_hat_body = 0, and the gravity
direction ImuFilter already maintains closes the system:

    v_z = -(v_x*gx + v_y*gy) / gz

Against VQ1 truth, leave-one-session-out, that estimator returns the total speed at
median 1.24 m/s but p90 14.96 m/s, with a -14.3 m/s bias in the 22-40 m/s band -- because
the recorded flight-path angle has a median of 26-34 deg and only a third to two thirds of
frames sit under 20 deg. The assumption is simply false in this data. estimate() below
keeps that path for the record and for anyone who wants to re-measure it; NOTHING SHIPS
ON IT. The producer calls inplane().

So (vel_bearing_rad, speed_est_mps) together are the body-horizontal velocity VECTOR, and
nothing in this file claims to know how fast the aircraft is climbing.
"""
from __future__ import annotations

import numpy as np

K_DRAG = 0.04249         # 1/m, a_xy = K_DRAG * |v_xy| * v_xy. Fit: dragval.py
A_NOISE = 0.08           # m/s^2 horizontal specific force below which |v_xy| is noise
                         # (0.08 -> |v_xy| = 1.35 m/s; below that drag carries no signal)
SPEED_MAX = 40.0         # clamp; fastest recorded VQ1 frame is 34.3 m/s
GZ_MIN = 0.35            # |g_hat_z| below this: >69 deg bank, the horizontal-velocity
                         # constraint is singular and v_z is refused
A_MAX = 52.0             # m/s^2. CONTACT GUARD, and it is load-bearing rather than
                         # cosmetic: a collision reads 879 m/s^2 of horizontal specific
                         # force in the VM race session, which the drag inversion would
                         # cheerfully report as 143 m/s in whatever direction the impact
                         # pushed. 52 m/s^2 is the drag at the fastest frame ever recorded
                         # (34.3 m/s -> 50.0), so nothing aerodynamic is being discarded;
                         # above it the accelerometer is measuring something that is not
                         # air, and the honest answer is no answer.
                         # NOT a filter -- a REFUSAL. Nothing is smoothed or carried over,
                         # so the estimator keeps the algebraic, non-diverging property
                         # that vel_bearing_rad has.


def estimate(accel, g_hat, k=K_DRAG):
    """accel: canonical body specific force (3,). g_hat: unit gravity DOWN in body, or None.

    -> (speed_mps, conf, v_body) where v_body is a 3-vector or None. speed is the TOTAL
    speed when g_hat is usable and the in-plane speed otherwise (conf reports which).
    """
    ax, ay = float(accel[0]), float(accel[1])
    a_h = float(np.hypot(ax, ay))
    if a_h < A_NOISE or a_h > A_MAX:
        return 0.0, 0.0, None
    v_xy = float(np.sqrt(a_h / k))
    ux, uy = -ax / a_h, -ay / a_h
    conf_a = float(np.clip((a_h - A_NOISE) / 0.4, 0.0, 1.0))
    if g_hat is None:
        return min(v_xy, SPEED_MAX), 0.5 * conf_a, None
    gz = float(g_hat[2])
    if abs(gz) < GZ_MIN:
        return min(v_xy, SPEED_MAX), 0.5 * conf_a, None
    uz = -(ux * float(g_hat[0]) + uy * float(g_hat[1])) / gz
    v = np.array([ux * v_xy, uy * v_xy, uz * v_xy])
    s = float(min(np.linalg.norm(v), SPEED_MAX))
    conf = conf_a * float(np.clip((abs(gz) - GZ_MIN) / 0.25, 0.0, 1.0))
    return s, conf, v


def estimate_batch(accel, g_hat, k=K_DRAG):
    """Vectorised. accel (n,3), g_hat (n,3) -> s (n,), conf (n,), v_body (n,3),
    v_xy (n,) -- the assumption-free in-plane speed, reported separately."""
    accel = np.asarray(accel, float)
    g = np.asarray(g_hat, float)
    ax, ay = accel[:, 0], accel[:, 1]
    a_h = np.hypot(ax, ay)
    live = a_h >= A_NOISE
    ah = np.maximum(a_h, 1e-9)
    v_xy = np.where(live, np.sqrt(a_h / k), 0.0)
    ux, uy = -ax / ah, -ay / ah
    gz = g[:, 2]
    okz = live & (np.abs(gz) >= GZ_MIN)
    uz = np.where(okz, -(ux * g[:, 0] + uy * g[:, 1]) / np.where(np.abs(gz) < 1e-9, 1.0, gz), 0.0)
    v = np.stack([ux * v_xy, uy * v_xy, uz * v_xy], 1)
    s = np.minimum(np.linalg.norm(v, axis=1), SPEED_MAX)
    conf_a = np.clip((a_h - A_NOISE) / 0.4, 0.0, 1.0)
    conf = np.where(okz, conf_a * np.clip((np.abs(gz) - GZ_MIN) / 0.25, 0.0, 1.0),
                    0.5 * conf_a)
    return s, conf * live, v, np.minimum(v_xy, SPEED_MAX)


def inplane(accel, k=K_DRAG):
    """The assumption-free estimate: BODY IN-PLANE velocity only.

    -> (speed_mps, conf, v_xy_body 3-vector with z = 0), or (0, 0, None).

    This is the estimator the producer uses, and it deliberately stops short of the
    total speed: |v_xy| = sqrt(|a_xy| / k) needs nothing but the accelerometer and the
    fitted coefficient, whereas the third component needs an assumption about the flight
    path that the recordings do not support (see the module docstring).
    """
    ax, ay = float(accel[0]), float(accel[1])
    a_h = float(np.hypot(ax, ay))
    if a_h < A_NOISE or a_h > A_MAX:
        return 0.0, 0.0, None
    v = float(np.sqrt(a_h / k))
    conf = float(np.clip((a_h - A_NOISE) / 0.4, 0.0, 1.0))
    return v, conf, np.array([-ax / a_h * v, -ay / a_h * v, 0.0])


def inplane_batch(accel, k=K_DRAG):
    accel = np.asarray(accel, float)
    ax, ay = accel[:, 0], accel[:, 1]
    a_h = np.hypot(ax, ay)
    ok = (a_h >= A_NOISE) & (a_h <= A_MAX)
    ah = np.maximum(a_h, 1e-9)
    v = np.where(ok, np.sqrt(a_h / k), 0.0)
    conf = np.where(ok, np.clip((a_h - A_NOISE) / 0.4, 0.0, 1.0), 0.0)
    vec = np.stack([-ax / ah * v, -ay / ah * v, np.zeros_like(v)], 1)
    return v, conf, vec
