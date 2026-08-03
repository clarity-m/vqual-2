"""TRANSIENT VALIDITY OF THE DRAG SPEED INVERSION -- measured against VQ1 truth velocity.

THE HYPOTHESIS UNDER TEST (posed 2026-08-02): |v_xy| = sqrt(|a_xy|/k) assumes STEADY
STATE; in steady flight the horizontal specific force is drag alone, but while ACCELERATING
it is drag + the linear acceleration, so speed_est is biased during exactly the aggressive
transients a race consists of.

THE HYPOTHESIS IS FALSE, and section 2 measures it at the force level rather than arguing.
An accelerometer measures SPECIFIC FORCE -- non-gravitational forces divided by mass --
not coordinate acceleration. Thrust acts along body -z by definition, so the horizontal
body pair is drag ALONE whether or not the aircraft is accelerating: the acceleration is
the CONSEQUENCE of thrust + drag + gravity, not a further additive term in what the sensor
reads. Regressing the drag residual on the coordinate acceleration returns a slope of
0.007-0.041 where a full leak would return 1.000.

WHAT IS REAL is milder and differently caused: the estimate is roughly twice as noisy when
|a_xy| or the body rate is changing fast (section 1), it is not a truth-stream timing
artefact (section 3), and a bounded confidence factor sorts the tail (section 4). That
factor is what producer.transient_factor() implements; the constants there are read off
these tables and are not fitted.

Run: python3 dragtransient.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import dragmodel as DM      # noqa: E402
import dragval as DV        # noqa: E402

K = DM.K_DRAG


def deriv(t, y):
    dt = np.gradient(t)
    dt[np.abs(dt) < 1e-6] = 1e-6
    if y.ndim == 1:
        return np.gradient(y) / dt
    return np.stack([np.gradient(y[:, j]) / dt for j in range(y.shape[1])], 1)


def gather(rows):
    """One flat table over every usable VQ1 frame (airborne, no contact, not parked,
    |v_xy| >= 2 m/s, |a_xy| inside the estimator's own accept band)."""
    out = {k: [] for k in ('true', 'est', 'jerk', 'gyro', 'dvdt_w', 'dvdt_b', 'ah',
                           'conf_noise', 'resid', 'dv_b')}
    for r in rows:
        t, a, v_b = r['t'], r['accel'], r['v_body']
        sxy = np.linalg.norm(v_b[:, :2], axis=1)
        ah = np.hypot(a[:, 0], a[:, 1])
        est = np.where((ah >= DM.A_NOISE) & (ah <= DM.A_MAX), np.sqrt(ah / K), np.nan)
        m = np.isfinite(est) & (sxy >= 2.0)
        dv_b = deriv(t, v_b)[:, :2]
        out['true'].append(sxy[m])
        out['est'].append(est[m])
        out['jerk'].append(np.abs(deriv(t, ah))[m])
        out['gyro'].append(np.linalg.norm(r['gyro'], axis=1)[m])
        out['dvdt_w'].append(np.linalg.norm(deriv(t, r['v_world']), axis=1)[m])
        out['dvdt_b'].append(np.linalg.norm(dv_b, axis=1)[m])
        out['ah'].append(ah[m])
        out['conf_noise'].append(np.clip((ah[m] - DM.A_NOISE) / 0.4, 0.0, 1.0))
        out['resid'].append((a[:, :2] + K * sxy[:, None] * v_b[:, :2])[m])
        out['dv_b'].append(dv_b[m])
    return {k: (np.concatenate(v) if v[0].ndim == 1 else np.concatenate(v, 0))
            for k, v in out.items()}


def hstr(hi):
    return 'inf' if hi > 1e8 else ('%g' % hi)


def bands(x, e, true, edges, label, unit):
    print('    %-24s %7s %8s %8s %8s %8s' % (label + ' [' + unit + ']', 'n', 'bias',
                                             'med|e|', 'p90|e|', 'med%'))
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        if k.sum() < 50:
            continue
        print('    %10g - %-11s %7d %8.2f %8.2f %8.2f %7.1f%%'
              % (lo, hstr(hi), k.sum(), np.mean(e[k]), np.median(np.abs(e[k])),
                 np.percentile(np.abs(e[k]), 90),
                 100 * np.median(np.abs(e[k]) / np.maximum(true[k], 1e-6))))


def main():
    rows = DV.load()
    D = gather(rows)
    e = D['est'] - D['true']
    true = D['true']
    print('=== VQ1, n=%d frames (airborne, no contact, not parked, |v_xy| >= 2 m/s) ==='
          % len(e))
    print('    k = %.5f    error e = est|v_xy| - true|v_xy|' % K)
    print('    OVERALL  bias %+.2f  med|e| %.2f  p90|e| %.2f  median relative %.1f%%'
          % (np.mean(e), np.median(np.abs(e)), np.percentile(np.abs(e), 90),
             100 * np.median(np.abs(e) / true)))

    print()
    print('--- 1. error vs every transient proxy ------------------------------------')
    bands(D['dvdt_w'], e, true, [0, 2, 5, 10, 20, 40, 1e9], '|dv/dt| inertial', 'm/s2')
    print()
    bands(D['dvdt_b'], e, true, [0, 2, 5, 10, 20, 40, 80, 1e9], '|dv_body/dt|', 'm/s2')
    print()
    bands(D['gyro'], e, true, [0, 0.2, 0.5, 1, 2, 4, 1e9], '|gyro|', 'rad/s')
    print()
    bands(D['jerk'], e, true, [0, 2, 5, 10, 25, 50, 1e9], 'd|a_xy|/dt', 'm/s3')
    print()
    for lbl, m in (('steady   |dv/dt| < 10', D['dvdt_w'] < 10),
                   ('accel    |dv/dt| >= 10', D['dvdt_w'] >= 10),
                   ('hard     |dv/dt| >= 25', D['dvdt_w'] >= 25)):
        print('    %-24s n=%6d  bias %+.2f  med|e| %.2f  p90|e| %.2f  rel %.1f%%'
              % (lbl, m.sum(), np.mean(e[m]), np.median(np.abs(e[m])),
                 np.percentile(np.abs(e[m]), 90),
                 100 * np.median(np.abs(e[m]) / true[m])))

    print()
    print('--- 2. THE FALSIFICATION: is there a linear-acceleration term at all? -----')
    print('    residual  r = a_meas_xy + k|v_xy|v_xy   (zero if the pair is pure drag)')
    print('    regress r on the body-horizontal COORDINATE acceleration; a full leak')
    print('    would give slope 1.000 on each axis.')
    R, AC, G = D['resid'], D['dv_b'], D['gyro']
    for j, ax in ((0, 'x'), (1, 'y')):
        s = np.polyfit(AC[:, j], R[:, j], 1)[0]
        c = np.corrcoef(AC[:, j], R[:, j])[0, 1]
        print('      body-%s  slope %+.4f  corr %+.3f   (all frames, n=%d)'
              % (ax, s, c, len(R)))
    q = G < 0.2
    for j, ax in ((0, 'x'), (1, 'y')):
        s = np.polyfit(AC[q, j], R[q, j], 1)[0]
        print('      body-%s  slope %+.4f  corr %+.3f   (|gyro| < 0.2 rad/s, n=%d)'
              % (ax, s, np.corrcoef(AC[q, j], R[q, j])[0, 1], q.sum()))
    rn, mag = np.linalg.norm(R, axis=1), np.linalg.norm(AC, axis=1)
    print('    residual MAGNITUDE by |dv/dt| -- it grows, but as a FRACTION of the')
    print('    acceleration it collapses, which is noise amplification, not a leak:')
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)):
        m = (mag >= lo) & (mag < hi)
        if m.sum() < 50:
            continue
        print('      %5g-%-9s n=%6d  |r| med %.3f p90 %.3f m/s2   |r|/|dv/dt| med %.3f'
              % (lo, hstr(hi), m.sum(), np.median(rn[m]),
                 np.percentile(rn[m], 90), np.median(rn[m] / np.maximum(mag[m], 1e-6))))

    print()
    print('--- 3. is the transient residual a TRUTH-ALIGNMENT artefact? -------------')
    print('    truth velocity is interpolated from a slower position stream onto the')
    print('    61 Hz IMU clock; a constant offset would produce a residual proportional')
    print('    to rate of change. Sweep the shift -- a flat minimum at 0 means no.')
    print('      shift_ms   med|resid|   med|resid| (|dv/dt|>10)   med|e|   med|e| (>10)')
    for sh in (-0.10, -0.05, -0.033, -0.016, 0.0, 0.016, 0.033, 0.05, 0.10):
        RN, MG, EE = [], [], []
        for r in rows:
            t, a, v_b = r['t'], r['accel'], r['v_body']
            vs = np.stack([np.interp(t + sh, t, v_b[:, j]) for j in range(3)], 1)
            sxy = np.linalg.norm(vs[:, :2], axis=1)
            ah = np.hypot(a[:, 0], a[:, 1])
            est = np.where((ah >= DM.A_NOISE) & (ah <= DM.A_MAX), np.sqrt(ah / K), np.nan)
            m = np.isfinite(est) & (sxy >= 2.0)
            RN.append(np.linalg.norm((a[:, :2] + K * sxy[:, None] * vs[:, :2])[m], axis=1))
            MG.append(np.linalg.norm(deriv(t, vs)[m, :2], axis=1))
            EE.append(est[m] - sxy[m])
        RN, MG, EE = map(np.concatenate, (RN, MG, EE))
        hi = MG > 10
        print('      %+8.0f   %10.3f   %22.3f   %6.3f   %11.3f'
              % (sh * 1000, np.median(RN), np.median(RN[hi]),
                 np.median(np.abs(EE)), np.median(np.abs(EE[hi]))))

    print()
    print('--- 4. the shipped confidence, scored ------------------------------------')
    print('    producer.transient_factor() x the existing |a_xy| noise confidence.')
    import producer as PR      # noqa: E402
    f = np.array([PR.transient_factor(j, g) for j, g in zip(D['jerk'], D['gyro'])])
    conf = D['conf_noise'] * f
    print('    transient factor: median %.2f  p10 %.2f  frames degraded below 1.0: %.1f%%'
          % (np.median(f), np.percentile(f, 10), 100 * np.mean(f < 0.999)))
    print('    corr(1 - factor, |e|) = %+.3f' % np.corrcoef(1 - f, np.abs(e))[0, 1])
    for lo, hi in ((0.0, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 30:
            continue
        print('      conf %4.2f-%4.2f  n=%6d (%5.1f%%)  med|e| %.2f  p90|e| %.2f  '
              'bias %+.2f' % (lo, hi, m.sum(), 100 * m.mean(), np.median(np.abs(e[m])),
                              np.percentile(np.abs(e[m]), 90), np.mean(e[m])))
    print('    NOTE: the factor changes no ESTIMATE, only the advertised confidence.')
    print('    speed_est_mps is bit-identical before and after; what moves is what the')
    print('    consumer is told about it.')
    np.savez(os.path.join(HERE, 'dragtransient.npz'), e=e, conf=conf, **D)


if __name__ == '__main__':
    main()
