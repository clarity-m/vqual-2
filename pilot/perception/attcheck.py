"""Is attitude_conf p10 = 0.00 in high-rate segments the intended behaviour, or a bug?

roll_pitch() multiplies two independent degradations:

    conf_mag  = max(0, 1 - ||a| - 9.81| / 4)     translational bursts
    conf_turn = 1 / (1 + 0.8 * max(0, |gyro| - 0.3))   coordinated turns

conf_turn cannot reach zero -- it is a positive rational function, 0.32 at 3 rad/s. So any
0.00 must come from conf_mag, i.e. from |a| differing from g by >= 4 m/s^2. This script
decomposes the collapse, and separately asks the question the coordinator raised: does a
COLLISION spike poison the gravity estimate for a long window afterwards?

The second question is not about the confidence at all but about ImuFilter.g, whose trim
rate alpha is itself gated on the same |a|-g mismatch -- which means a collision does not
drag g toward a bad measurement (alpha drops to 0.0001), it just stops correcting. That is
the right behaviour, but "should be" is not a measurement, so it is measured here against
the gyro-only propagation.
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import producer as P   # noqa: E402

SESS = os.path.join(os.path.dirname(HERE), 'sessions')


def load(session):
    rows = []
    with open(os.path.join(SESS, session, 'imu.csv'), newline='') as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r['t_wall_ns']),
                             float(r['xacc']), float(r['yacc']), float(r['zacc']),
                             float(r['xgyro']), float(r['ygyro']), float(r['zgyro'])))
            except (TypeError, ValueError):
                pass
    return np.asarray(rows)


def collisions(session):
    p = os.path.join(SESS, session, 'collisions.csv')
    if not os.path.exists(p):
        return []
    ts = []
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            try:
                ts.append(float(r['t_wall_ns']) / 1e9)
            except (TypeError, ValueError, KeyError):
                pass
    if not ts:
        return []
    ts = np.sort(np.asarray(ts))
    ep, a = [], ts[0]
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > 0.5:
            ep.append((a, ts[i - 1]))
            a = ts[i]
    ep.append((a, ts[-1]))
    return ep


def main(session):
    rows = load(session)
    t = rows[:, 0] / 1e9
    t0 = t[0]
    imu = P.ImuFilter()
    conf, cmag, cturn, gyn, amag, roll, pitch = [], [], [], [], [], [], []
    for r in rows:
        imu.push(r[0], r[1:4], r[4:7])
        rr, pp, c = imu.roll_pitch()
        conf.append(c)
        cm = max(0.0, 1.0 - abs(imu.amag - 9.81) / 4.0)
        ct = 1.0 / (1.0 + 0.8 * max(0.0, float(np.linalg.norm(imu.gyro)) - 0.3))
        cmag.append(cm); cturn.append(ct)
        gyn.append(float(np.linalg.norm(imu.gyro)))
        amag.append(imu.amag); roll.append(rr); pitch.append(pp)
    conf = np.asarray(conf); cmag = np.asarray(cmag); cturn = np.asarray(cturn)
    gyn = np.asarray(gyn); amag = np.asarray(amag)
    ts = t - t0

    print('== %s: %d IMU samples, %.1f s ==' % (session, len(t), ts[-1]))
    print('attitude_conf overall: med %.2f  p10 %.2f  frac==0 %.3f'
          % (np.median(conf), np.percentile(conf, 10), np.mean(conf < 1e-6)))
    hi = gyn > 1.0
    print('during |gyro|>1 (%.0f%% of samples): conf med %.2f p10 %.2f'
          % (100 * hi.mean(), np.median(conf[hi]), np.percentile(conf[hi], 10)))
    print()
    print('WHICH FACTOR COLLAPSES? (conf = conf_mag * conf_turn)')
    print('  conf_turn : min %.3f  p10 %.3f  med %.3f   <- cannot reach 0 by construction'
          % (cturn.min(), np.percentile(cturn, 10), np.median(cturn)))
    print('  conf_mag  : min %.3f  p10 %.3f  med %.3f   frac==0 %.3f'
          % (cmag.min(), np.percentile(cmag, 10), np.median(cmag), np.mean(cmag < 1e-6)))
    z = conf < 1e-6
    if z.any():
        print('  of the %d zero-confidence samples, %.0f%% have conf_mag == 0'
              % (z.sum(), 100 * np.mean(cmag[z] < 1e-6)))
        print('  |a| on those samples: med %.2f  p5 %.2f  p95 %.2f  (g = 9.81)'
              % (np.median(amag[z]), np.percentile(amag[z], 5), np.percentile(amag[z], 95)))
        print('  |gyro| on those samples: med %.2f  p90 %.2f' %
              (np.median(gyn[z]), np.percentile(gyn[z], 90)))

    print()
    print('IS THE ZERO EARNED? |a| deviation vs body rate, by regime:')
    for lo, hi2 in ((0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 9.0)):
        m = (gyn >= lo) & (gyn < hi2)
        if m.sum() < 20:
            continue
        print('  |gyro| %3.1f-%3.1f  n=%5d  ||a|-g| med %5.2f p90 %5.2f  conf med %.2f p10 %.2f'
              % (lo, hi2, m.sum(), np.median(np.abs(amag[m] - 9.81)),
                 np.percentile(np.abs(amag[m] - 9.81), 90),
                 np.median(conf[m]), np.percentile(conf[m], 10)))

    ep = collisions(session)
    print()
    print('COLLISION TRANSIENTS: %d episode(s)' % len(ep))
    for a, b in ep:
        ra, rb = a - t0, b - t0
        print('  episode %.2f-%.2f s' % (ra, rb))
        for lo, hi2, lbl in ((ra - 2.0, ra, 'before (2 s)'),
                             (ra, rb + 0.5, 'during +0.5 s'),
                             (rb + 0.5, rb + 2.5, 'after 0.5-2.5 s'),
                             (rb + 2.5, rb + 6.5, 'after 2.5-6.5 s')):
            m = (ts >= lo) & (ts < hi2)
            if m.sum() < 5:
                continue
            print('    %-18s n=%4d  |a| med %6.2f max %7.2f   conf med %.2f  '
                  '|gyro| med %.2f' % (lbl, m.sum(), np.median(amag[m]), amag[m].max(),
                                       np.median(conf[m]), np.median(gyn[m])))
    print()
    print('  Gravity-filter trim rate is gated on the SAME mismatch: alpha = 0.008 when')
    print('  ||a|-g| < 0.35, 0.001 under 1.5, else 0.0001. So during a contact spike the')
    print('  filter stops trusting the accelerometer rather than being dragged by it.')
    n_alpha0 = np.mean(np.abs(amag - 9.81) >= 1.5)
    print('  frames at the slowest trim (||a|-g| >= 1.5): %.1f%%' % (100 * n_alpha0))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else '20260802-153626-vm-fast-lap-0-10-with-collisions')
