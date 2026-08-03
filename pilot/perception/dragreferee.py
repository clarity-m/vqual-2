"""INDEPENDENT REFEREE for the drag speed model, on VQ2 data.

The fit was made against VQ1's truth velocity. VQ1 is a different build, and CONVENTIONS.md
is explicit that "everything was measured on the VQ1 build" is an inference, not a
measurement. So the model needs a referee that lives entirely inside VQ2, and gate PnP is
one: a gate is static, so the rate of change of its measured RANGE is the closing speed.

    -d(range)/dt  =  component of own velocity along the line of sight

The left-hand side comes from the camera and the known 1.5 m aperture; the right-hand side
from the accelerometer and a coefficient fitted on a different build. Nothing is shared.

WHY A WINDOW AND NOT A DIFFERENCE. Consecutive-frame differencing does not work here and
the reason is worth recording: PnP range noise is ~3% of range, so at 15 m it is ~0.45 m,
while one frame at 5 m/s moves 0.17 m. The derivative is four times smaller than the noise
on it, and the first version of this script duly returned corr = -0.09 -- a null result
manufactured entirely by the measurement setup, not by the model. The slope is therefore
fitted over a window long enough for the signal to clear the noise.

WHAT THIS REFEREE CANNOT SEE: it compares only the horizontal projection, so the vertical
term the model already refuses to estimate leaks in whenever the gate is well above or
below the flight path. Frames are restricted to small gate bearings and low body rate, and
the residual is reported as a distribution.
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dragmodel as DM   # noqa: E402

SESS = os.path.join(os.path.dirname(HERE), 'sessions')
RUNS = os.path.join(HERE, 'producer_runs')

WIN_S = 0.60               # slope window: at 5 m/s that is 3 m of closure vs ~0.45 m noise
MIN_PTS = 10               # pose_valid samples required inside the window
BEARING_MAX_DEG = 30.0     # gate near the nose: the LOS projection is a cosine
GYRO_MAX = 1.0             # straight-ish, so the range change is translation
SPEED_MIN = 2.5
R2_MIN = 0.80              # the window must actually look like a straight approach


def imu_arrays(session):
    t, a = [], []
    with open(os.path.join(SESS, session, 'imu.csv'), newline='') as f:
        for r in csv.DictReader(f):
            try:
                t.append(float(r['t_wall_ns']) / 1e9)
                a.append((float(r['xacc']), float(r['yacc']), float(r['zacc'])))
            except (TypeError, ValueError):
                pass
    return np.asarray(t), np.asarray(a)


def frame_t0(session):
    with open(os.path.join(SESS, session, 'frames.csv'), newline='') as f:
        for r in csv.DictReader(f):
            if r.get('file'):
                return float(r['t_recv_wall_ns']) / 1e9
    return None


def run(session, rundir):
    dg = np.genfromtxt(os.path.join(rundir, 'diag.csv'), delimiter=',', names=True)
    t, rng, pv = dg['t_s'], dg['range_m'], dg['pose_valid']
    br, gy = np.radians(dg['bearing_deg']), dg['gyro_norm']

    ti, ai = imu_arrays(session)
    ti = ti - frame_t0(session)
    idx = np.clip(np.searchsorted(ti, t), 0, len(ti) - 1)
    v_xy, conf, vec = DM.inplane_batch(ai[idx])
    bvel = np.arctan2(vec[:, 1], vec[:, 0])

    ok = (pv > 0) & np.isfinite(rng)
    rows = []
    i = 0
    n = len(t)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and t[j + 1] - t[i] <= WIN_S:
            j += 1
        w = np.arange(i, j + 1)
        w = w[ok[w]]
        if len(w) < MIN_PTS or t[w[-1]] - t[w[0]] < 0.5 * WIN_S:
            i += 1
            continue
        if gy[w].max() > GYRO_MAX or np.max(np.abs(np.degrees(br[w]))) > BEARING_MAX_DEG:
            i += 1
            continue
        if np.min(conf[w]) < 0.3 or np.median(v_xy[w]) < SPEED_MIN:
            i += 1
            continue
        tt, rr = t[w], rng[w]
        k = np.polyfit(tt, rr, 1)
        res = rr - np.polyval(k, tt)
        ssr = 1.0 - np.sum(res ** 2) / max(np.sum((rr - rr.mean()) ** 2), 1e-9)
        if ssr < R2_MIN:
            i += 1
            continue
        closing_meas = -float(k[0])
        d = br[w] - bvel[w]
        closing_pred = float(np.median(v_xy[w] * np.cos(d)))
        rows.append((float(tt[0]), closing_meas, closing_pred, float(np.median(v_xy[w])),
                     float(np.median(np.degrees(d))), float(np.median(rr))))
        i = j + 1                      # non-overlapping windows: independent samples
    return np.asarray(rows) if rows else np.zeros((0, 6))


def report(name, r):
    if len(r) < 15:
        print('  %-46s n=%3d  too few' % (name[:46], len(r)))
        return None
    meas, pred = r[:, 1], r[:, 2]
    e = pred - meas
    k = np.polyfit(meas, pred, 1)
    print('  %-46s n=%4d' % (name[:46], len(r)))
    print('      closing: meas med %5.2f   pred med %5.2f m/s' %
          (np.median(meas), np.median(pred)))
    print('      residual (pred-meas): bias %+5.2f  med|e| %5.2f  p90 %5.2f m/s' %
          (np.mean(e), np.median(np.abs(e)), np.percentile(np.abs(e), 90)))
    print('      pred = %.2f*meas %+.2f   corr %+.2f' %
          (k[0], k[1], float(np.corrcoef(meas, pred)[0, 1])))
    return r


def main():
    pairs = []
    for a in sys.argv[1:]:
        s, d = a.split('=')
        pairs.append((s, d))
    if not pairs:
        pairs = [('20260802-153626-vm-fast-lap-0-10-with-collisions', 'BEFORE-vm'),
                 ('20260801-121520-vq2-lap-0-15', 'BEFORE-lap121520'),
                 ('20260801-144858-vqual2-lap-pausing', 'BEFORE-pausing')]
    allr = []
    print('=== gate-PnP range closure vs drag-derived closing speed (VQ2 only) ===')
    print('    window %.2f s, |bearing|<%.0f deg, |gyro|<%.1f, window R2>%.2f'
          % (WIN_S, BEARING_MAX_DEG, GYRO_MAX, R2_MIN))
    for s, d in pairs:
        p = os.path.join(RUNS, d)
        if not os.path.exists(os.path.join(p, 'diag.csv')):
            print('  %-46s (no diag.csv)' % s[:46])
            continue
        rr = report(s, run(s, p))
        if rr is not None:
            allr.append(rr)
    if allr:
        r = np.concatenate(allr)
        meas, pred = r[:, 1], r[:, 2]
        e = pred - meas
        k = np.polyfit(meas, pred, 1)
        print()
        print('  POOLED n=%d  bias %+.2f  med|e| %.2f  p90 %.2f m/s  corr %+.2f  '
              'slope %.2f' % (len(r), np.mean(e), np.median(np.abs(e)),
                              np.percentile(np.abs(e), 90),
                              float(np.corrcoef(meas, pred)[0, 1]), k[0]))
        for lo, hi in ((2, 5), (5, 8), (8, 12), (12, 25)):
            m = (meas >= lo) & (meas < hi)
            if m.sum() < 8:
                continue
            print('    meas %2.0f-%-2.0f  n=%4d  pred med %5.2f  bias %+5.2f  med|e| %5.2f'
                  % (lo, hi, m.sum(), np.median(pred[m]), np.mean(e[m]),
                     np.median(np.abs(e[m]))))
        np.savetxt(os.path.join(RUNS, 'dragreferee.csv'), r, delimiter=',', fmt='%.4f',
                   header='t_s,closing_meas,closing_pred,v_xy,delta_deg,range_m')


if __name__ == '__main__':
    main()
