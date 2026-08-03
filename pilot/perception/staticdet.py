"""STATIC / PARKED DETECTOR from IMAGE MOTION -- the referee the accelerometer cannot be.

WHY THIS EXISTS. A quadrotor in steady straight flight at CONSTANT VELOCITY has zero net
acceleration, so its accelerometer reads pure gravity in the body frame: a constant vector,
tilted by the trim attitude. A quadrotor PARKED on the VQ2 launch pad -- which sits on top of
a parked aircraft, tilted -- reads a constant, tilted gravity vector too. In this simulator
the two are not merely similar, they are numerically almost the same:

    20260802-163548, parked on the pad   t 48-52 s   a = (-2.999, +0.001, -9.340)  |a|=9.81
    20260802-163548, steady drift        t 62-82 s   a = (-3.000,  0.000, -9.338)  |a|=9.81

Same magnitude, same 17.8 deg tilt, both with |gyro| = 0.0000. **No function of the IMU
alone can separate them**, and neither can "how many IMU rows repeat", because a constant
acceleration is exactly what steady flight produces. A predecessor read this session
backwards on precisely that evidence and fitted on the frozen segment. The camera settles it
in one number.

THE MEASUREMENT. Mean absolute difference between consecutive frames, 160x120 greyscale:

    m(i) = mean |I_i - I_{i-1}|          (8-bit levels)

Nothing about the drone is assumed -- only that the world outside is static and textured, so
a moving camera changes its image and a stationary one does not.

CALIBRATION (all measured; rerun calibrate() / validate() after any change to SIZE):
    STATIC  163548 t 4-46 s   aircraft lodged after a contact, image frozen
                              p50 0.073  p90 0.100  p99 0.125
    MOVING  163548 t 55-82 s  the straight drift, 8.4 m/s      p50 5.63  p10 ~1.7
    MOVING  163548 t 0-4 s    take-off                         p50 7.48
    MOVING  VQ1 airborne frames, truth speed > 2 m/s (n=14989 across 6 sessions)

M_STATIC = 0.25 sits 2x above the frozen-image p99 and ~7x below the moving p10. It is not
delicate: any threshold in 0.15-0.30 gives the same verdicts (validate() sweeps it).

STATIC_MIN_S is the second half of the rule and it is what makes the low threshold safe.
"Parked" is a SUSTAINED state. A single glance at an untextured wall, or a frame pair spanning
a stall in the video stream, reads low for a moment while the aircraft is moving fast --
requiring the low-motion run to last 2 s removes those. Measured against VQ1 truth velocity:

    threshold 0.25, no run-length rule   1.16% of truth-moving (>2 m/s) frames called static
    threshold 0.25, run >= 2 s           1.16%       (the floor: genuinely textureless views)
    threshold 0.50, no run-length rule   6.74%
    threshold 0.50, run >= 2 s           3.12%

and 100% of the known-static 42 s is caught with 0% of the known-moving 27 s lost.

WHAT IT CANNOT DO. It reports MOTION OF THE IMAGE, not speed, and it is a NEGATIVE claim:
"the image is not changing, therefore the aircraft is not moving through the scene". A lens
staring at untextured sky reads low while moving (the residual 1.16%); a pure yaw on the spot
reads high while not translating. Both are visible in the series itself.
"""
from __future__ import annotations

import csv
import os

import numpy as np

try:
    import cv2
except ImportError:                                       # pragma: no cover
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.join(os.path.dirname(HERE), 'sessions')

SIZE = (160, 120)          # downscale: kills JPEG block noise, keeps every real motion
M_STATIC = 0.25            # mean |dI| below this -> the image is not changing
SMOOTH_S = 0.5             # median-filter the series over this first, so one dropped or
                           # duplicated frame cannot flip a verdict
STATIC_MIN_S = 2.0         # ... and it must STAY below for this long to count as parked
CACHE = '_imgmotion.npz'


def _session_dir(session):
    return session if os.path.isdir(session) else os.path.join(SESS, session)


def _imu_t0(d):
    with open(os.path.join(d, 'imu.csv'), newline='') as f:
        return float(next(csv.DictReader(f))['t_wall_ns']) / 1e9


def motion_series(session, rebuild=False):
    """-> (t_s, m), t_s measured from the FIRST IMU ROW so it shares the time axis every
    other tool in this project uses. Cached in the session directory (sessions/ is
    gitignored, so the cache never reaches git). ([], []) when the session has no frames."""
    d = _session_dir(session)
    cache = os.path.join(d, CACHE)
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        return z['t'], z['m']
    rows = []
    fcsv = os.path.join(d, 'frames.csv')
    if os.path.exists(fcsv):
        with open(fcsv, newline='') as f:
            for r in csv.DictReader(f):
                if r.get('file'):
                    rows.append((float(r['t_recv_wall_ns']) / 1e9, r['file']))
    if len(rows) < 2 or cv2 is None:
        t = np.zeros(0); m = np.zeros(0)
        np.savez(cache, t=t, m=m)
        return t, m
    t0 = _imu_t0(d)
    prev = None
    T, M = [], []
    for ts, fn in rows:
        im = cv2.imread(os.path.join(d, 'frames', fn), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, SIZE).astype(np.float32)
        if prev is not None:
            T.append(ts - t0)
            M.append(float(np.mean(np.abs(im - prev))))
        prev = im
    t = np.asarray(T); m = np.asarray(M)
    np.savez(cache, t=t, m=m)
    return t, m


def _median_smooth(t, m, win_s):
    if len(m) == 0:
        return m
    out = np.empty_like(m)
    for i in range(len(m)):
        w = (t >= t[i] - 0.5 * win_s) & (t <= t[i] + 0.5 * win_s)
        out[i] = np.median(m[w])
    return out


def _static_flags(t, m, m_static, min_s):
    """low-motion AND sustained for min_s."""
    low = _median_smooth(t, m, SMOOTH_S) < m_static
    out = np.zeros(len(low), bool)
    i, n = 0, len(low)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and low[j + 1]:
            j += 1
        if t[j] - t[i] >= min_s:
            out[i:j + 1] = True
        i = j + 1
    return out


def moving_mask(session, t_query, m_static=M_STATIC, min_s=STATIC_MIN_S, default=True):
    """THE REUSABLE CALL. -> bool array over t_query (seconds from the first IMU row):
    True where the aircraft is moving through the scene, False where it is parked.

    `default` is used wherever the camera cannot answer -- a session with no frames, or a
    query time outside the frame span. It is True, so the detector only ever REMOVES frames
    it has positive evidence against; a frameless session behaves exactly as before.
    """
    t_query = np.atleast_1d(np.asarray(t_query, float))
    t, m = motion_series(session)
    if len(t) < 3:
        return np.full(t_query.shape, bool(default))
    st = _static_flags(t, m, m_static, min_s)
    idx = np.clip(np.searchsorted(t, t_query), 0, len(t) - 1)
    out = ~st[idx]
    out[(t_query < t[0] - 1.0) | (t_query > t[-1] + 1.0)] = bool(default)
    return out


def static_fraction(session):
    t, m = motion_series(session)
    if len(t) < 3:
        return float('nan'), 0
    return float(np.mean(_static_flags(t, m, M_STATIC, STATIC_MIN_S))), len(t)


def segments(session, m_static=M_STATIC, min_s=STATIC_MIN_S):
    """Contiguous parked stretches as (t0, t1) seconds -- for reading a session by eye."""
    t, m = motion_series(session)
    if len(t) < 3:
        return []
    st = _static_flags(t, m, m_static, min_s)
    out, i, n = [], 0, len(st)
    while i < n:
        if not st[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and st[j + 1]:
            j += 1
        out.append((float(t[i]), float(t[j])))
        i = j + 1
    return out


KNOWN = [('20260802-163548', 4, 46, False, 'lodged after a contact, image frozen'),
         ('20260802-163548', 55, 82, True, 'the straight drift, 8.4 m/s'),
         ('20260802-163548', 0, 4, True, 'take-off')]


def calibrate():
    print('=== image-motion calibration (mean |dI|, %dx%d greyscale) ===' % SIZE)
    print('    M_STATIC = %.2f   STATIC_MIN_S = %.1f s' % (M_STATIC, STATIC_MIN_S))
    for s, lo, hi, moving, lbl in KNOWN:
        t, m = motion_series(s)
        w = (t >= lo) & (t < hi)
        if w.sum() < 3:
            print('  %-16s %5.0f-%-5.0f  no frames' % (s, lo, hi))
            continue
        st = _static_flags(t, m, M_STATIC, STATIC_MIN_S)
        print('  %-16s %3.0f-%-3.0f n=%4d  p10 %7.3f p50 %7.3f p99 %7.3f  '
              'called static %6.2f%%  [truth: %s -- %s]'
              % (s, lo, hi, w.sum(), np.percentile(m[w], 10), np.median(m[w]),
                 np.percentile(m[w], 99), 100 * st[w].mean(),
                 'MOVING' if moving else 'STATIC', lbl))
    print()
    for s, _, _, _, _ in KNOWN[:1]:
        print('  parked segments found in %s:' % s)
        for a, b in segments(s):
            print('     %6.2f - %6.2f s  (%.1f s)' % (a, b, b - a))


def validate():
    """Score the rule against VQ1 TRUTH velocity -- the only ground truth available."""
    import sys
    sys.path.insert(0, HERE)
    import dragfit as DF              # noqa: E402
    print()
    print('=== validation against VQ1 truth velocity ===')
    data = []
    for n in DF.VQ1:
        t, m = motion_series(n)
        if len(t) < 50:
            print('  %-32s no frames -- detector abstains (default True)' % n)
            continue
        s = DF.session(n)
        if s is None:
            continue
        sp = np.interp(t, s['t'], np.linalg.norm(s['v_world'], axis=1))
        data.append((n, t, m, sp))
    for n, t, m, sp in data:
        st = _static_flags(t, m, M_STATIC, STATIC_MIN_S)
        k = sp > 2.0
        print('  %-32s n=%5d  parked %5.2f%%  of truth-moving frames called parked %5.2f%%'
              % (n, len(t), 100 * st.mean(), 100 * st[k].mean() if k.sum() else float('nan')))
    print()
    print('  %6s %6s | %-34s | %s' % ('thr', 'minS', 'VQ1 truth-moving called parked',
                                      '163548 known windows'))
    t3, m3 = motion_series('20260802-163548')
    ks = (t3 >= 4) & (t3 < 46)
    km = (t3 >= 55) & (t3 < 82)
    for thr in (0.15, 0.25, 0.50, 1.00):
        for min_s in (0.0, 2.0):
            fp = n2 = 0
            for _, t, m, sp in data:
                st = _static_flags(t, m, thr, min_s)
                k = sp > 2.0
                fp += int(st[k].sum()); n2 += int(k.sum())
            s3 = _static_flags(t3, m3, thr, min_s)
            print('  %6.2f %6.1f | s>2 m/s: %6.2f%%                       | '
                  'static caught %6.2f%%  moving lost %5.2f%%'
                  % (thr, min_s, 100 * fp / max(n2, 1), 100 * s3[ks].mean(),
                     100 * s3[km].mean()))


if __name__ == '__main__':
    calibrate()
    validate()
