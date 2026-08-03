"""20260802-163548 "vm-straight-fly": what the session contains, established by the CAMERA.

THIS FILE PREVIOUSLY RECORDED THE SEGMENTATION BACKWARDS, and the way it did so is the
lesson, not the session. Claire's note says "marker 1 indicates when straight fly/drift
begins". The predecessor pass overruled her from the IMU alone and concluded the opposite.
Its evidence looked strong:

    t 4-46 s  (pre-reset)   |a_xy| pinned at 2.77-2.78 m/s^2, |gyro| < 0.008 rad/s
    t 55-82 s (post-marker) |a_xy| pinned at 3.000 m/s^2,     |gyro| EXACTLY 0.0000

and it read the second as a parked aircraft, because a drone resting on the tilted VQ2 launch
pad reads |a_xy| = 3.00 at zero speed. It then fitted/validated on the first.

BOTH SEGMENTS LOOK IDENTICAL TO AN ACCELEROMETER, AND THAT IS NOT A COINCIDENCE. Steady
flight at CONSTANT VELOCITY has zero net acceleration, so the accelerometer reads pure
gravity in the body frame: a constant vector at the trim tilt, |a| = g. A parked aircraft on
a tilted pad reads a constant vector at the pad tilt, |a| = g. Here the drift trim (17.8 deg)
and the pad tilt (17.8 deg) happen to match to three digits. No statistic of the IMU can
separate them -- and in particular "how many IMU rows repeat" cannot, because repeated rows
are what a steady state produces. (The claimed 79.6% of exactly-repeated rows after the
marker was also simply wrong: it is 8.2% there and 7.3% over the session.)

WHAT THE CAMERA SAYS (staticdet.py, mean |dI| between consecutive frames):

    t  3.5-47.3 s   0.073   STATIC    <- the segment that was fitted on
    t 55-82 s       5.63    MOVING    <- the segment that was discarded

The frames say it plainly: at t=20 s and t=45 s the image is a single flat orange/yellow
surface filling the frame, pixel-identical between them. The aircraft took off at t=0, hit
something at t=2.1 s (collisions.csv, one episode) and came to rest jammed against it with
the lens blocked, and stayed there for 44 s until Claire reset the sim at t=46.76 s. Two
further facts, each independent of both the camera and the accelerometer, agree:

    * active_gate_index advances 0 -> 1 at t=53.97 s. A gate was CROSSED after the reset.
      Nothing parked on the pad crosses a gate.
    * events.jsonl: armed=false at t=47.3, armed=true at t=49.3 -- the aircraft was
      re-armed for a second flight, which is the one the markers annotate.

SO THE SESSION IS EXACTLY WHAT CLAIRE SAID IT WAS: take-off, a crash at t=2 s, 44 s of a
jammed aircraft, a reset, then from marker 1 a clean straight drift at ~8.4 m/s to the end.

WHAT IT CAN STILL DO FOR THE DRAG MODEL: less than hoped, and it is worth saying why. The
drift is at ONE speed, and VQ2 streams no velocity, so it cannot enter the fit. As a
range-closure referee it is nearly powerless in BOTH directions: the frozen segment has no
motion to measure, and the real drift is flown down an empty hangar aisle with only 0.1%
pose-valid frames -- the aircraft passes gate 1 abeam and then has nothing in view. What
survives is a single fly-by of gate 1, 12 fresh detections over 0.43 s, run below.
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dragmodel as DM     # noqa: E402
import staticdet as SD     # noqa: E402

SESS = os.path.join(os.path.dirname(HERE), 'sessions')
RUNS = os.path.join(HERE, 'producer_runs')
SESSION = '20260802-163548'
RUN = 'F-drift-POST'          # replay of the POST-MARKER segment (t >= 54.98 s)
MARKER1_S = 54.98


def imu():
    t, a, g = [], [], []
    with open(os.path.join(SESS, SESSION, 'imu.csv'), newline='') as f:
        for r in csv.DictReader(f):
            try:
                t.append(float(r['t_wall_ns']) / 1e9)
                a.append((float(r['xacc']), float(r['yacc']), float(r['zacc'])))
                g.append((float(r['xgyro']), float(r['ygyro']), float(r['zgyro'])))
            except (TypeError, ValueError):
                pass
    t = np.asarray(t)
    return t - t[0], np.asarray(a), np.asarray(g)


def segmentation():
    t, a, g = imu()
    ah = np.hypot(a[:, 0], a[:, 1])
    gn = np.linalg.norm(g, axis=1)
    dup = np.zeros(len(t), bool)
    dup[1:] = np.all(np.diff(np.concatenate([a, g], 1), axis=0) == 0, axis=1)
    print('=== %s: IMU says these two segments are the same thing ===' % SESSION)
    print('  %-22s %6s %10s %10s %9s %7s' %
          ('segment', 'n', '|a_xy| p50', '|a| p50', '|gyro|p50', 'dup%'))
    for lbl, lo, hi in (('pre-reset  4-46 s', 4, 46), ('post-marker 55-82 s', MARKER1_S, 82)):
        m = (t >= lo) & (t < hi)
        print('  %-22s %6d %10.4f %10.4f %9.5f %6.1f%%'
              % (lbl, m.sum(), np.median(ah[m]),
                 np.median(np.linalg.norm(a[m], axis=1)), np.median(gn[m]),
                 100 * dup[m].mean()))
    print('  exact-duplicate consecutive IMU rows over the whole session: %.1f%%'
          % (100 * dup.mean()))
    print()
    print('=== the camera separates them (staticdet: mean |dI|, M_STATIC = %.2f) ==='
          % SD.M_STATIC)
    tm, mm = SD.motion_series(SESSION)
    for lbl, lo, hi in (('pre-reset  4-46 s', 4, 46), ('post-marker 55-82 s', MARKER1_S, 82)):
        w = (tm >= lo) & (tm < hi)
        mv = SD.moving_mask(SESSION, tm[w])
        print('  %-22s n=%4d  mean|dI| p50 %8.3f   MOVING %6.2f%% of frames'
              % (lbl, w.sum(), np.median(mm[w]), 100 * mv.mean()))
    print('  parked segments found: ' +
          ', '.join('%.1f-%.1f s' % s for s in SD.segments(SESSION)))
    print()
    print('=== independent of both: the race packet ===')
    rt, ai = [], []
    with open(os.path.join(SESS, SESSION, 'race.csv'), newline='') as f:
        for r in csv.DictReader(f):
            try:
                rt.append(float(r['t_wall_ns']) / 1e9)
                ai.append(int(r['active_gate_index']))
            except (TypeError, ValueError):
                pass
    rt = np.asarray(rt) - rt[0]
    ai = np.asarray(ai)
    for i in np.nonzero(np.diff(ai))[0]:
        print('  active_gate_index %d -> %d at t = %.2f s  (a gate was CROSSED)'
              % (ai[i], ai[i + 1], rt[i + 1]))


def flyby(rundir):
    """The one referee this session can still supply: the gate-1 fly-by.

    |gyro| is EXACTLY 0.0000 after the marker, so the body frame is non-rotating and a gate's
    body bearing is an inertial bearing. Straight flight past a static point at closest
    approach b:   bearing(t) = atan2(b, x0 - v t),  range(t) = hypot(b, x0 - v t).
    Fit (v, b, x0) to the FRESH detections; bearings carry the shape, the size-derived range
    only sets the scale -- which is exactly where its metre-level error lands.
    """
    p = os.path.join(rundir, 'diag.csv')
    if not os.path.exists(p):
        print('  no %s -- replay the post-marker segment first (see __main__ note)' % p)
        return
    d = np.genfromtxt(p, delimiter=',', names=True)
    fresh = (d['valid'] > 0) & (d['staleness_s'] < 0.001)
    print()
    print('=== gate-1 fly-by, the only VQ2-native speed check left in this session ===')
    print('  pose_valid over the drift: %.1f%% of %d frames -- the aisle is empty, which is'
          % (100 * np.mean(d['pose_valid'] > 0), len(d)))
    print('  why the range-closure referee cannot run here on EITHER segment.')
    if fresh.sum() < 8:
        print('  fresh detections: %d -- too few even for the fly-by' % fresh.sum())
        return
    t = d['t_s'][fresh]
    bm = np.radians(d['bearing_deg'][fresh])
    rm = d['range_m'][fresh]
    sp = d['speed_mps'][fresh]
    print('  n fresh = %d over %.2f s, bearing %.1f -> %.1f deg, drag speed med %.2f m/s'
          % (fresh.sum(), t[-1] - t[0], math.degrees(bm[0]), math.degrees(bm[-1]),
             np.median(sp)))
    try:
        from scipy.optimize import least_squares
    except ImportError:
        print('  scipy unavailable; skipping the fit')
        return
    sig_b, sig_r = math.radians(1.0), 0.15

    def resid(q, tt=t, b=bm, r=rm):
        v, cpa, x0 = q
        x = x0 - v * (tt - tt[0])
        return np.concatenate([(np.arctan2(cpa, x) - b) / sig_b,
                               (np.hypot(cpa, x) - r) / (sig_r * r)])
    best = None
    for v0 in (2.0, 5.0, 8.0, 12.0, 20.0):
        s = least_squares(resid, [v0, 3.0, 7.0], bounds=([.1, .1, .1], [40, 30, 60]))
        if best is None or s.cost < best.cost:
            best = s
    rng = np.random.default_rng(0)
    vs = []
    for _ in range(1000):
        i = rng.integers(0, len(t), len(t))
        try:
            vs.append(least_squares(lambda q: resid(q, t[i], bm[i], rm[i]), best.x,
                                    bounds=([.1, .1, .1], [40, 30, 60])).x[0])
        except Exception:
            pass
    vs = np.asarray(vs)
    print('  geometric v = %.2f m/s (CPA %.2f m); resample sd %.2f, 95%% CI [%.2f, %.2f]'
          % (best.x[0], best.x[1], vs.std(), *np.percentile(vs, [2.5, 97.5])))
    print('  drag model at k = %.5f predicts %.2f m/s   ->  ratio geom/drag = %.2f'
          % (DM.K_DRAG, np.median(sp), best.x[0] / np.median(sp)))
    print('  NOT DECISIVE, and the reason is structural: v scales linearly with the closest-')
    print('  approach distance, which comes from the SIZE-derived range of a gate sitting at')
    print('  the frame edge, and the sight line drops 6 deg in elevation across the 0.43 s so')
    print('  the flight is not in the plane the model assumes. The statistical CI above is a')
    print('  small fraction of those systematics. The 24-leg odometry referee (dragleg.py),')
    print('  which spans 8-25 m/s and uses only crossings + the map, is the one with power.')


def parked_speed_warning():
    """What the producer reported while the aircraft was jammed against a wall."""
    p = os.path.join(RUNS, 'F-drift', 'diag.csv')
    if not os.path.exists(p):
        return
    d = np.genfromtxt(p, delimiter=',', names=True)
    m = (d['t_s'] >= 4) & (d['t_s'] < 46)
    if m.sum() < 10:
        return
    print()
    print('=== and the consequence downstream, which is the reason this matters ===')
    print('  over the 42 s in which the aircraft did not move at all, producer reported')
    print('    speed_est_mps  p10/p50/p90 = %.2f / %.2f / %.2f m/s   speed_conf median %.2f'
          % (*np.percentile(d['speed_mps'][m], [10, 50, 90]), np.median(d['speed_conf'][m])))
    print('  at FULL confidence. sqrt(|a_xy|/k) has no zero, so a tilted stationary airframe')
    print('  is reported as ~8 m/s and COAST_TRANSLATE will duly translate every coasted')
    print('  track at that speed. Nothing in the IMU can catch it; staticdet.moving_mask can.')


if __name__ == '__main__':
    segmentation()
    flyby(os.path.join(RUNS, sys.argv[1] if len(sys.argv) > 1 else RUN))
    parked_speed_warning()
