"""Two confounds before quoting any coefficient.

(1) CONTACT. |a_xy| reaches 24.7 m/s^2 (2.5 g horizontal) in the fastest bin. That is not
    plausible aerodynamic drag on this airframe; skids and wall contact are in the data
    (150712 is literally a skid card) and collisions.csv timestamps them. A contact
    impulse is a horizontal specific force with no relation to airspeed, so leaving it in
    inflates high-speed drag and drags the coefficient toward a wrong law.

(2) TIME ALIGNMENT. imu.csv and position.csv are separate MAVLink streams. A lag between
    them is invisible in slow flight and dominant in fast flight, which is exactly the
    band the fit is supposed to be about. Scan it rather than assume it.
"""
import csv
import os

import numpy as np
import dragfit as DF

AIR_Z = -0.5
S_MIN = 2.0
CONTACT_GUARD_S = 0.5


def collision_times(name):
    p = os.path.join(DF.SESS, name, 'collisions.csv')
    if not os.path.exists(p):
        return np.zeros(0)
    ts = []
    with open(p, newline='') as f:
        rd = csv.DictReader(f)
        key = None
        for row in rd:
            if key is None:
                key = 't_wall_ns' if 't_wall_ns' in row else list(row)[0]
            try:
                ts.append(float(row[key]))
            except (TypeError, ValueError):
                pass
    return np.asarray(ts)


def load(lag_s=0.0, drop_contact=True):
    out = []
    for n in DF.VQ1:
        s = DF.session(n)
        if s is None:
            continue
        m = s['p_world'][:, 2] < AIR_Z
        if drop_contact:
            ct = collision_times(n)
            if len(ct):
                t0 = ct.min()
                # collisions.csv shares t_wall_ns with imu.csv
                ctr = ct / 1e9 - (t0 / 1e9 - (t0 / 1e9))
                ct_s = ct / 1e9
                base = ct_s.min() * 0  # placeholder, real offset applied below
                # rebuild session-relative collision times
                imu_t0 = None
                p = os.path.join(DF.SESS, n, 'imu.csv')
                with open(p, newline='') as f:
                    r0 = next(csv.DictReader(f))
                    imu_t0 = float(r0['t_wall_ns']) / 1e9
                ct_rel = ct_s - imu_t0
                bad = np.zeros(len(s['t']), bool)
                idx = np.searchsorted(s['t'], ct_rel)
                for i in idx:
                    lo = max(0, i - 1)
                    tt = s['t'][min(lo, len(s['t']) - 1)]
                    bad |= np.abs(s['t'] - tt) < CONTACT_GUARD_S
                m = m & ~bad
        if m.sum() < 200:
            continue
        g_body = np.einsum('nji,j->ni', s['R'][m], np.array([0.0, 0.0, 1.0]))
        out.append(dict(name=n, accel=s['accel'][m], v_body=s['v_body'][m],
                        v_world=s['v_world'][m], g_body=g_body, t=s['t'][m],
                        gyro=s['gyro'][m]))
    return out


def fit_cs(rows):
    """a_xy = -(k1 + k2 s) v_xy"""
    A = np.concatenate([r['accel'][:, :2] for r in rows])
    V = np.concatenate([r['v_body'] for r in rows])
    s = np.linalg.norm(V, axis=1)
    m = s >= S_MIN
    A, V, s = A[m], V[m], s[m]
    X = np.concatenate([V[:, :2].reshape(-1, 1), (V[:, :2] * s[:, None]).reshape(-1, 1)], 1)
    y = -A.reshape(-1)
    k, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ k
    return k, 1.0 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2), len(s)


def main():
    print('=== lag scan: shift v_body relative to accel, refit, report R2 ===')
    base = load(drop_contact=False)
    for lag_ms in range(-200, 201, 25):
        rows = []
        for r in base:
            n = len(r['t'])
            sh = int(round(lag_ms / 1000.0 * 61.0))     # imu ~61 Hz
            if sh == 0:
                rr = r
            elif sh > 0:
                rr = dict(r, accel=r['accel'][sh:], v_body=r['v_body'][:n - sh])
            else:
                rr = dict(r, accel=r['accel'][:n + sh], v_body=r['v_body'][-sh:])
            rows.append(rr)
        k, r2, n = fit_cs(rows)
        print('  lag %+5d ms   R2 %.4f   k = %s' % (lag_ms, r2, np.array2string(k, precision=5)))

    print()
    print('=== contact removal ===')
    for tag, rows in (('with contact', load(drop_contact=False)),
                      ('contact dropped', load(drop_contact=True))):
        k, r2, n = fit_cs(rows)
        A = np.concatenate([r['accel'][:, :2] for r in rows])
        ah = np.linalg.norm(A, axis=1)
        print('  %-18s n=%6d  R2 %.4f  k = %s  |a_xy| p99 %.2f  max %.2f'
              % (tag, n, r2, np.array2string(k, precision=5), np.percentile(ah, 99), ah.max()))

    print()
    print('=== |a_xy| vs s after contact removal, binned (is it really quadratic?) ===')
    rows = load(drop_contact=True)
    A = np.concatenate([r['accel'][:, :2] for r in rows])
    V = np.concatenate([r['v_body'] for r in rows])
    s = np.linalg.norm(V, axis=1)
    vxy = np.linalg.norm(V[:, :2], axis=1)
    ah = np.linalg.norm(A, axis=1)
    print('   s bin        n    |a_xy|/|v_xy| p25/p50/p75    implied c(s)')
    for lo, hi in ((2, 4), (4, 6), (6, 8), (8, 11), (11, 14), (14, 18), (18, 23),
                   (23, 28), (28, 40)):
        k = (s >= lo) & (s < hi)
        if k.sum() < 30:
            continue
        c = ah[k] / np.maximum(vxy[k], 1e-6)
        print('  %4.0f-%-4.0f %7d      %6.3f %6.3f %6.3f' %
              (lo, hi, k.sum(), np.percentile(c, 25), np.percentile(c, 50),
               np.percentile(c, 75)))


if __name__ == '__main__':
    main()
