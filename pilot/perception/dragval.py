"""Fit k, then validate OUT OF SAMPLE. Selection is stated, not implied.

WHAT IS CUT AND WHY (every one of these has already bitten this project once):
  parked      -- a drone on the 17.8 deg start pad reads |a_xy| = 3.00 m/s^2 at zero
                 speed (pad normal force through a tilted airframe). Pooled in, it is
                 34-100% of a session and it fixes the intercept at a pure artefact.
  contact     -- collision EPISODES (>0.5 s gaps), plus a guard window. A contact impulse
                 is a horizontal specific force unrelated to airspeed; |a_xy| reaches
                 106 m/s^2 in the raw data.
  slow        -- below 2 m/s drag is at the accelerometer noise floor.
  rotating    -- high |gyro| frames: the truth-velocity rotation into the body frame is
                 interpolated from three streams, and interpolation error scales with
                 rate. Reported both ways so the cut can be seen to matter or not.
  not moving  -- IMAGE-MOTION cut (staticdet.moving_mask), added 2026-08-02 after a
                 predecessor fitted 42 s of a jammed aircraft believing it was straight
                 flight. The accelerometer CANNOT tell steady constant-velocity flight
                 from a park on a tilted surface -- both read a constant gravity vector at
                 the trim tilt -- so this cut is the only one here that does not come from
                 the same instrument as the thing being fitted. On the VQ1 sessions it
                 removes 328 frames (5.4 s, all of them in 150712) and leaves k unchanged
                 at 0.04249: the airborne-altitude cut was already doing the job here. It
                 is wired in anyway, because the next analyst will not have truth altitude.
                 Sessions with no frames abstain (kept), so it only ever removes frames it
                 has positive evidence against.
"""
import csv
import os

import numpy as np
import dragfit as DF
import dragmodel as DM
import staticdet as SD

AIR_Z = -0.5
S_MIN = 2.0
CONTACT_GUARD_S = 0.5
EPISODE_GAP_S = 0.5


def episodes(name):
    """Collision EPISODES (project convention: gaps > 0.5 s), as (t0, t1) wall seconds."""
    p = os.path.join(DF.SESS, name, 'collisions.csv')
    if not os.path.exists(p):
        return []
    ts = []
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row['t_wall_ns']) / 1e9)
            except (TypeError, ValueError, KeyError):
                pass
    if not ts:
        return []
    ts = np.sort(np.asarray(ts))
    out, a = [], ts[0]
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > EPISODE_GAP_S:
            out.append((a, ts[i - 1]))
            a = ts[i]
    out.append((a, ts[-1]))
    return out


def imu_t0(name):
    with open(os.path.join(DF.SESS, name, 'imu.csv'), newline='') as f:
        return float(next(csv.DictReader(f))['t_wall_ns']) / 1e9


def load(gyro_max=None, use_static=True):
    out = []
    for n in DF.VQ1:
        s = DF.session(n)
        if s is None:
            continue
        m = s['p_world'][:, 2] < AIR_Z
        t0 = imu_t0(n)
        for a, b in episodes(n):
            m &= ~((s['t'] > a - t0 - CONTACT_GUARD_S) & (s['t'] < b - t0 + CONTACT_GUARD_S))
        if gyro_max is not None:
            m &= np.linalg.norm(s['gyro'], axis=1) <= gyro_max
        if use_static:
            m &= SD.moving_mask(n, s['t'])
        if m.sum() < 200:
            continue
        g_body = np.einsum('nji,j->ni', s['R'][m], np.array([0.0, 0.0, 1.0]))
        out.append(dict(name=n, accel=s['accel'][m], v_body=s['v_body'][m],
                        v_world=s['v_world'][m], g_body=g_body, t=s['t'][m],
                        gyro=s['gyro'][m]))
    return out


def fit_k(rows, model='inplane'):
    """model 'inplane': a_xy = -k |v_xy| v_xy.  'total': a_xy = -k |v| v_xy.
    'inplane+lin': a_xy = -(k1 + k2 |v_xy|) v_xy."""
    A = np.concatenate([r['accel'][:, :2] for r in rows])
    V = np.concatenate([r['v_body'] for r in rows])
    s = np.linalg.norm(V, axis=1)
    vxy = np.linalg.norm(V[:, :2], axis=1)
    m = s >= S_MIN
    A, V, s, vxy = A[m], V[m], s[m], vxy[m]
    y = -A.reshape(-1)
    if model == 'inplane':
        X = (V[:, :2] * vxy[:, None]).reshape(-1, 1)
    elif model == 'total':
        X = (V[:, :2] * s[:, None]).reshape(-1, 1)
    elif model == 'inplane+lin':
        X = np.concatenate([V[:, :2].reshape(-1, 1),
                            (V[:, :2] * vxy[:, None]).reshape(-1, 1)], 1)
    else:
        X = np.concatenate([V[:, :2].reshape(-1, 1),
                            (V[:, :2] * s[:, None]).reshape(-1, 1)], 1)
    k, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ k
    return k, 1.0 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2), int(m.sum())


def band(true, est, conf, bands=((2, 5), (5, 8), (8, 12), (12, 16), (16, 22), (22, 40))):
    print('      %-12s %7s %8s %8s %8s %8s' % ('band', 'n', 'bias', 'med|e|', 'p90|e|', 'med%'))
    for lo, hi in bands:
        k = (true >= lo) & (true < hi) & (conf > 0.2)
        if k.sum() < 30:
            continue
        e = est[k] - true[k]
        print('      %5.0f-%-6.0f %7d %8.2f %8.2f %8.2f %7.1f%%'
              % (lo, hi, k.sum(), np.mean(e), np.median(np.abs(e)),
                 np.percentile(np.abs(e), 90),
                 100 * np.median(np.abs(e) / np.maximum(true[k], 1e-6))))
    k = (true >= 2) & (conf > 0.2)
    e = est[k] - true[k]
    print('      %-12s %7d %8.2f %8.2f %8.2f %7.1f%%'
          % ('ALL >=2', k.sum(), np.mean(e), np.median(np.abs(e)),
             np.percentile(np.abs(e), 90), 100 * np.median(np.abs(e) / true[k])))


def main():
    rows = load()
    tot = sum(len(r['t']) for r in rows)
    print('=== usable VQ1 frames: airborne, no contact, %d total (%.0f s at 61 Hz) ==='
          % (tot, tot / 61.0))
    for r in rows:
        s = np.linalg.norm(r['v_body'], axis=1)
        print('  %-32s n=%6d (%5.1f s)  s med %5.2f max %5.2f  frac s>8 %.2f'
              % (r['name'], len(s), len(s) / 61.0, np.median(s), s.max(), np.mean(s > 8)))

    print()
    print('=== which drag law? ===')
    for m in ('total', 'inplane', 'total+lin', 'inplane+lin'):
        k, r2, n = fit_k(rows, m)
        print('  R2 %.4f   %-14s  k = %s   n = %d'
              % (r2, m, np.array2string(k, precision=5), n))

    print()
    print('=== per-session k (in-plane quadratic) -- the stability test the pooled')
    print('    total-speed law failed (2.35x spread) ===')
    ks = []
    for r in rows:
        k, r2, n = fit_k([r], 'inplane')
        if n < 100:
            continue
        ks.append(float(k[0]))
        print('  %-32s k = %.5f   R2 %.4f   n %5d' % (r['name'], k[0], r2, n))
    print('  spread: median %.5f  min %.5f  max %.5f  (max/min = %.2f)'
          % (np.median(ks), min(ks), max(ks), max(ks) / min(ks)))

    print()
    print('=== does a gyro cut change k? (interpolation-error check) ===')
    for gm in (None, 2.0, 1.0, 0.5):
        rr = load(gyro_max=gm)
        k, r2, n = fit_k(rr, 'inplane')
        print('  |gyro| <= %-5s  k = %.5f  R2 %.4f  n = %6d (%.0f s)'
              % (str(gm), k[0], r2, n, n / 61.0))

    print()
    print('=== flight-path angle (the world-horizontal assumption behind v_z) ===')
    vw = np.concatenate([r['v_world'] for r in rows])
    sw = np.linalg.norm(vw, axis=1)
    for lo, hi in ((2, 8), (8, 15), (15, 40)):
        m = (sw >= lo) & (sw < hi)
        if m.sum() < 50:
            continue
        f = np.degrees(np.arcsin(np.clip(-vw[m, 2] / np.maximum(sw[m], 1e-6), -1, 1)))
        print('  s %2.0f-%-2.0f n=%6d  |fpa| med %5.1f deg  p90 %5.1f  frac<20deg %.2f'
              % (lo, hi, m.sum(), np.median(np.abs(f)), np.percentile(np.abs(f), 90),
                 np.mean(np.abs(f) < 20)))

    # ---- LEAVE-ONE-SESSION-OUT -----------------------------------------------------
    print()
    print('=== LEAVE-ONE-SESSION-OUT (fit on the rest, predict the held-out) ===')
    print('    two estimands: |v_xy| (assumption-free) and |v| (needs the gravity constraint)')
    exy, e3, ev = [], [], []
    for i, held in enumerate(rows):
        k, _, _ = fit_k([r for j, r in enumerate(rows) if j != i], 'inplane')
        se, cf, ve, vxye = DM.estimate_batch(held['accel'], held['g_body'], k=float(k[0]))
        st = np.linalg.norm(held['v_body'], axis=1)
        sxy = np.linalg.norm(held['v_body'][:, :2], axis=1)
        m = (st >= 2) & (cf > 0.2)
        if m.sum() < 50:
            continue
        a = vxye[m] - sxy[m]
        b = se[m] - st[m]
        c = np.linalg.norm(ve[m] - held['v_body'][m], axis=1)
        exy.append(a); e3.append(b); ev.append(c)
        print('  %-30s k=%.5f n=%5d | vxy med|e| %5.2f p90 %5.2f | '
              '|v| med|e| %5.2f p90 %5.2f | vec med %5.2f p90 %5.2f'
              % (held['name'][:30], k[0], m.sum(), np.median(np.abs(a)),
                 np.percentile(np.abs(a), 90), np.median(np.abs(b)),
                 np.percentile(np.abs(b), 90), np.median(c), np.percentile(c, 90)))
    exy, e3, ev = np.concatenate(exy), np.concatenate(e3), np.concatenate(ev)
    print('  POOLED OOS  n=%d' % len(exy))
    print('    |v_xy|  bias %+.2f  med|e| %.2f  p90 %.2f m/s' %
          (np.mean(exy), np.median(np.abs(exy)), np.percentile(np.abs(exy), 90)))
    print('    |v|     bias %+.2f  med|e| %.2f  p90 %.2f m/s' %
          (np.mean(e3), np.median(np.abs(e3)), np.percentile(np.abs(e3), 90)))
    print('    vector  med %.2f  p90 %.2f m/s  (this is what coast translation uses)'
          % (np.median(ev), np.percentile(ev, 90)))

    kf, r2, n = fit_k(rows, 'inplane')
    print()
    print('=== final k = %.5f (R2 %.4f, n %d) -- in-sample error by band ===' % (kf[0], r2, n))
    ST, SE, CF, SXY, VXY, EV = [], [], [], [], [], []
    for r in rows:
        se, cf, ve, vxye = DM.estimate_batch(r['accel'], r['g_body'], k=float(kf[0]))
        ST.append(np.linalg.norm(r['v_body'], axis=1))
        SXY.append(np.linalg.norm(r['v_body'][:, :2], axis=1))
        SE.append(se); CF.append(cf); VXY.append(vxye)
        EV.append(np.linalg.norm(ve - r['v_body'], axis=1))
    ST, SE, CF = np.concatenate(ST), np.concatenate(SE), np.concatenate(CF)
    SXY, VXY, EV = np.concatenate(SXY), np.concatenate(VXY), np.concatenate(EV)
    print('    |v_xy| (drag only, no assumption):')
    band(SXY, VXY, CF)
    print('    |v| (with the gravity constraint):')
    band(ST, SE, CF)
    m = (ST >= 2) & (CF > 0.2)
    print('    3-D velocity VECTOR error: med %.2f  p90 %.2f m/s'
          % (np.median(EV[m]), np.percentile(EV[m], 90)))
    np.savez('dragval.npz', k=kf, ST=ST, SE=SE, CF=CF, SXY=SXY, VXY=VXY, EV=EV)
    print()
    print('K_DRAG = %.5f' % kf[0])


if __name__ == '__main__':
    main()
