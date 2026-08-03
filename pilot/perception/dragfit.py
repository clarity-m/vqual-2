"""Fit the drag/speed model from VQ1 telemetry (the only build that streams truth velocity).

interface.SelfObs's contract: thrust acts along body -z BY DEFINITION, so the horizontal
BODY accelerometer components measure ONLY drag, and drag is antiparallel to airspeed:

    a_body[0:2] = -(c(s) / m) * v_body[0:2]          s = |v_body|

Both sides are body-frame, so no world frame and no yaw appears in the fit itself.
Attitude is needed only to rotate LOCAL_POSITION_NED's world velocity into the body frame;
that rotation is then CHECKED against the drag bearing, which is an independent statement
about the same quantity (producer._selfobs's vel_bearing_rad).

CONVENTIONS.md: gyro mirrored, accel NOT, LOCAL_POSITION_NED canonical,
truth_roll = ATTITUDE.roll, truth_pitch = ODOMETRY.pitch, truth_yaw = -ATTITUDE.yaw.
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.join(os.path.dirname(HERE), 'sessions')

VQ1 = {
    '20260731-130744': 'free flight, 30.0 m/s',
    '20260731-131305': 'long mixed, 29.9 m/s',
    '20260731-132659': 'long slow, 8.7 m/s',
    '20260731-143025': 'rate doublets, 20.6 m/s',
    '20260731-144815': 'single-axis taps, 21.2 m/s',
    '20260731-150712': 'thrust/cruise/skids, 34.3 m/s',
    '20260731-195307': 'lap w/ resets, 15.9 m/s',
    '20260731-203428': 'yaw taps, 4.1 m/s',
    '20260731-204841-vq1-lap-slow': 'clean lap, 17.8 m/s',
}


def load_csv(path, cols):
    out = {c: [] for c in cols}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                vals = [float(row[c]) for c in cols]
            except (TypeError, ValueError):
                continue
            if any(math.isnan(v) for v in vals):
                continue
            for c, v in zip(cols, vals):
                out[c].append(v)
    return {c: np.asarray(v) for c, v in out.items()}


def session(name):
    d = os.path.join(SESS, name)
    imu = load_csv(os.path.join(d, 'imu.csv'),
                   ['t_wall_ns', 'xacc', 'yacc', 'zacc', 'xgyro', 'ygyro', 'zgyro'])
    pos = load_csv(os.path.join(d, 'position.csv'),
                   ['t_wall_ns', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    att = load_csv(os.path.join(d, 'attitude.csv'), ['t_wall_ns', 'roll', 'pitch', 'yaw'])
    odo = load_csv(os.path.join(d, 'odometry.csv'), ['t_wall_ns', 'qw', 'qx', 'qy', 'qz'])
    if min(len(imu['t_wall_ns']), len(pos['t_wall_ns']), len(att['t_wall_ns'])) < 100:
        return None

    t0 = imu['t_wall_ns'][0] / 1e9
    t = imu['t_wall_ns'] / 1e9 - t0
    tp = pos['t_wall_ns'] / 1e9 - t0
    ta = att['t_wall_ns'] / 1e9 - t0
    to = odo['t_wall_ns'] / 1e9 - t0

    def ip(src_t, y):
        return np.interp(t, src_t, y)

    v_world = np.stack([ip(tp, pos['vx']), ip(tp, pos['vy']), ip(tp, pos['vz'])], 1)
    p_world = np.stack([ip(tp, pos['x']), ip(tp, pos['y']), ip(tp, pos['z'])], 1)

    roll = ip(ta, np.unwrap(att['roll']))
    yaw = -ip(ta, np.unwrap(att['yaw']))
    qw, qx, qy, qz = odo['qw'], odo['qx'], odo['qy'], odo['qz']
    s2 = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = ip(to, np.arcsin(s2))

    accel = np.stack([imu['xacc'], imu['yacc'], imu['zacc']], 1)      # canonical, no flip
    gyro = -np.stack([imu['xgyro'], imu['ygyro'], imu['zgyro']], 1)   # THE mirror

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.empty((len(t), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    v_body = np.einsum('nji,nj->ni', R, v_world)
    return dict(name=name, t=t, accel=accel, gyro=gyro, v_body=v_body, v_world=v_world,
                p_world=p_world, roll=roll, pitch=pitch, yaw=yaw, R=R)


def build_rows(sessions, gyro_max=None, speed_min=0.0):
    A, V, S, W, T, N = [], [], [], [], [], []
    for s in sessions:
        sp = np.linalg.norm(s['v_body'], axis=1)
        w = np.linalg.norm(s['gyro'], axis=1)
        m = sp >= speed_min
        if gyro_max is not None:
            m = m & (w <= gyro_max)
        A.append(s['accel'][m, :2])
        V.append(s['v_body'][m])
        S.append(sp[m])
        W.append(w[m])
        T.append(s['t'][m])
        N += [s['name']] * int(m.sum())
    return (np.concatenate(A), np.concatenate(V), np.concatenate(S),
            np.concatenate(W), np.concatenate(T), np.asarray(N))


def fit_c(A, V, S, model='quad'):
    """a_xy = -(k1 + k2*s) * v_xy, least squares over stacked x and y components."""
    vxy = V[:, :2].reshape(-1, 1)
    sv = (V[:, :2] * S[:, None]).reshape(-1, 1)
    y = -A.reshape(-1)
    if model == 'quad':
        X = np.concatenate([vxy, sv], 1)
    elif model == 'lin':
        X = vxy
    else:
        X = sv
    k, *_ = np.linalg.lstsq(X, y, rcond=None)
    return k


def c_of_s(k, s):
    s = np.asarray(s, float)
    return k[0] + k[1] * s if len(k) == 2 else k[0] * np.ones_like(s)


def invert_speed(k, a_h):
    """|a_xy| = c(s) |v_xy|; take |v_xy| ~ s -> k2 s^2 + k1 s - a_h = 0."""
    a_h = np.asarray(a_h, float)
    if len(k) == 2 and abs(k[1]) > 1e-9:
        disc = k[0] ** 2 + 4.0 * k[1] * a_h
        return (-k[0] + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * k[1])
    return a_h / max(k[0], 1e-9)


def main():
    which = sys.argv[1:] or list(VQ1)
    out = []
    for n in which:
        s = session(n)
        if s is None:
            print('  %s: no truth streams, skipped' % n)
            continue
        out.append(s)
        sp = np.linalg.norm(s['v_body'], axis=1)
        ah = np.linalg.norm(s['accel'][:, :2], axis=1)
        print('%-34s n=%6d dur=%6.1fs  speed p50/p90/max %5.2f/%5.2f/%5.2f  '
              '|a_xy| p50/p90 %5.2f/%5.2f  frac s>8 %.2f'
              % (n, len(s['t']), s['t'][-1], np.percentile(sp, 50), np.percentile(sp, 90),
                 sp.max(), np.percentile(ah, 50), np.percentile(ah, 90), np.mean(sp > 8)))
    return out


if __name__ == '__main__':
    main()
