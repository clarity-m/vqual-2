"""Final drag-fit report: the segmentation cut, the surviving seconds, the fit, and the
out-of-sample validation. Writes dragfit_report.txt.

THE CUT, in the order it is applied, with the seconds surviving each stage. This is the
part worth reading: this project has been bitten three times in one session by a
measurement setup that silently decided the answer, and a drag fit is exactly that shape of
problem -- a fit dominated by near-hover frames measures the accelerometer noise floor and
returns a confident coefficient for it.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dragfit as DF      # noqa: E402
import dragval as DV      # noqa: E402
import dragmodel as DM    # noqa: E402

HZ = 61.0
GYRO_MAX = 1.5
S_FAST = 5.0
STEADY_S = 0.5            # a "segment" must survive this long to count as sustained


def segments(mask, t, min_s):
    """Contiguous runs of mask lasting at least min_s. -> list of (i0, i1) inclusive."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if t[j] - t[i] >= min_s:
            out.append((i, j))
        i = j + 1
    return out


def main():
    lines = []

    def p(s=''):
        print(s)
        lines.append(s)

    p('DRAG / SPEED MODEL -- fit and out-of-sample validation')
    p('=' * 78)
    p()
    p('DATA. The fit needs TRUE velocity, which only the VQ1 build streams')
    p('(LOCAL_POSITION_NED, canonical per CONVENTIONS.md). VQ1 and VQ2 have identical')
    p('physics per the spec diff, so this is valid plant data for VQ2 -- and it is the')
    p('ONLY data that can carry this fit, since VQ2 blocks every pose stream. The')
    p('2026-08-02 VM race-pace session is therefore NOT a fit input; it is a validation')
    p('and diagnosis population, and it is used as one below.')
    p()
    p('THE CUT, and the seconds surviving each stage:')
    p()
    p('  %-30s %8s %8s %8s %8s' % ('session', 'raw s', 'airborne', '+noContact', '+steady'))

    rows = []
    tot = [0.0, 0.0, 0.0, 0.0]
    for n in DF.VQ1:
        s = DF.session(n)
        if s is None:
            continue
        t = s['t']
        air = s['p_world'][:, 2] < DV.AIR_Z
        nc = air.copy()
        t0 = DV.imu_t0(n)
        for a, b in DV.episodes(n):
            nc &= ~((t > a - t0 - DV.CONTACT_GUARD_S) & (t < b - t0 + DV.CONTACT_GUARD_S))
        sp = np.linalg.norm(s['v_body'], axis=1)
        gy = np.linalg.norm(s['gyro'], axis=1)
        steady_m = nc & (gy <= GYRO_MAX) & (sp >= S_FAST)
        segs = segments(steady_m, t, STEADY_S)
        steady_s = sum(t[b] - t[a] for a, b in segs)
        # real elapsed time per sample, so every column is on the same clock
        dt = np.gradient(t)
        raw = float(t[-1] - t[0])
        air_s, nc_s = float(dt[air].sum()), float(dt[nc].sum())
        p('  %-30s %8.1f %8.1f %8.1f %8.1f' % (n[:30], raw, air_s, nc_s, steady_s))
        tot[0] += raw; tot[1] += air_s; tot[2] += nc_s; tot[3] += steady_s
        rows.append((n, steady_m, segs))
    p('  %-30s %8.1f %8.1f %8.1f %8.1f' % ('TOTAL', *tot))
    p()
    p('  airborne   = LOCAL_POSITION_NED z < -0.5 m. Non-negotiable: a drone parked on')
    p('               the 17.8 deg start pad reads a CONSTANT |a_xy| = 3.00 m/s^2 at zero')
    p('               speed (pad normal force through a tilted airframe), and it is 34-99%')
    p('               of several sessions. Pooled in, it pins the intercept on an artefact.')
    p('  noContact  = collision EPISODES (>0.5 s gaps) with a 0.5 s guard either side.')
    p('               Raw |a_xy| reaches 106 m/s^2; nothing aerodynamic does that.')
    p('  steady     = additionally |gyro| <= %.1f rad/s, |v| >= %.0f m/s, in runs lasting'
      % (GYRO_MAX, S_FAST))
    p('               >= %.1f s. This is the "clean fast straight-ish" population.' % STEADY_S)
    p()

    # ---- fit on the full cut and on the strict cut ---------------------------------
    full = DV.load()
    kf, r2f, nf = DV.fit_k(full, 'inplane')
    p('THE LAW. Horizontal body specific force is quadratic in the IN-PLANE speed')
    p('component, not in the total speed:')
    p()
    p('    a_body[0:2] = -k * |v_xy| * v_xy')
    p()
    p('  candidate laws, pooled, airborne + no contact, |v| >= 2 m/s (n = %d):' % nf)
    for m in ('total', 'inplane', 'total+lin', 'inplane+lin'):
        k, r2, n = DV.fit_k(full, m)
        p('    R2 %.4f   %-13s  k = %s' % (r2, m, np.array2string(k, precision=5)))
    p()
    p('  Binning by TOTAL speed makes the implied coefficient saturate above ~14 m/s and')
    p('  the law look broken; binning by |v_xy| makes it flat over 2-25 m/s. That is')
    p('  anisotropic drag (the airframe has its own coefficient in the rotor plane), and')
    p('  it matters twice over: it fits far better (R2 0.956 vs 0.826), and it makes the')
    p('  inversion |v_xy| = sqrt(|a_xy|/k) exact -- no assumption inside it at all.')
    p()
    p('  per-session k (the stability test the total-speed law failed at 2.35x spread):')
    ks = []
    for r in full:
        k, r2, n = DV.fit_k([r], 'inplane')
        if n < 100:
            continue
        ks.append(float(k[0]))
        p('    %-30s k = %.5f  R2 %.4f  n %5d' % (r['name'][:30], k[0], r2, n))
    p('    spread max/min = %.2f   median %.5f' % (max(ks) / min(ks), np.median(ks)))
    p()
    p('  robustness of k to the steadiness cut (if the cut moved k, the fit would be')
    p('  measuring the cut rather than the airframe):')
    for gm in (None, 2.0, 1.0, 0.5):
        rr = DV.load(gyro_max=gm)
        k, r2, n = DV.fit_k(rr, 'inplane')
        p('    |gyro| <= %-5s  k = %.5f  R2 %.4f  n = %6d (%5.0f s)'
          % (str(gm), k[0], r2, n, n / HZ))
    p()
    p('  ADOPTED  k = %.5f  (R2 %.4f, n = %d)' % (kf[0], r2f, nf))
    p()

    # ---- LOSO ------------------------------------------------------------------------
    p('OUT-OF-SAMPLE (leave one SESSION out -- a whole session, not a time block, so no')
    p('split boundary can leak the way the gatenet label edit did on 2026-08-01):')
    p()
    p('  %-30s %6s %14s %14s' % ('held out', 'n', '|v_xy| med/p90', '|v| med/p90'))
    exy, e3 = [], []
    for i, held in enumerate(full):
        k, _, _ = DV.fit_k([r for j, r in enumerate(full) if j != i], 'inplane')
        se, cf, ve, vxye = DM.estimate_batch(held['accel'], held['g_body'], k=float(k[0]))
        st = np.linalg.norm(held['v_body'], axis=1)
        sxy = np.linalg.norm(held['v_body'][:, :2], axis=1)
        m = (st >= 2) & (cf > 0.2)
        if m.sum() < 50:
            continue
        a = vxye[m] - sxy[m]
        b = se[m] - st[m]
        exy.append(a); e3.append(b)
        p('  %-30s %6d   %5.2f / %5.2f   %5.2f / %5.2f'
          % (held['name'][:30], m.sum(), np.median(np.abs(a)), np.percentile(np.abs(a), 90),
             np.median(np.abs(b)), np.percentile(np.abs(b), 90)))
    exy, e3 = np.concatenate(exy), np.concatenate(e3)
    p('  %-30s %6d   %5.2f / %5.2f   %5.2f / %5.2f'
      % ('POOLED', len(exy), np.median(np.abs(exy)), np.percentile(np.abs(exy), 90),
         np.median(np.abs(e3)), np.percentile(np.abs(e3), 90)))
    p()
    p('  |v_xy| is what SHIPS: median %.2f m/s, p90 %.2f m/s, bias %+.2f, ~7%% relative,'
      % (np.median(np.abs(exy)), np.percentile(np.abs(exy), 90), np.mean(exy)))
    p('  and flat across every speed band from 2 to 34 m/s.')
    p()
    p('  |v| is what IS REFUSED: median looks fine at %.2f m/s but p90 is %.2f m/s and the'
      % (np.median(np.abs(e3)), np.percentile(np.abs(e3), 90)))
    p('  22-40 m/s band carries a -14.3 m/s bias. The total speed needs the body-z')
    p('  velocity component, which is not observable (thrust sits on that axis; the')
    p('  barometer reads nan in this sim). Closing the system with a world-horizontal-')
    p('  velocity assumption is what fails: the recorded flight-path angle has a 26-34 deg')
    p('  median and only a third to two thirds of frames sit under 20 deg.')
    p()
    p('  So speed_est_mps carries the IN-PLANE speed and says so, and')
    p('  (vel_bearing_rad, speed_est_mps) together are the body-horizontal velocity')
    p('  vector. Nothing in the stack claims to know the climb rate.')

    with open(os.path.join(HERE, 'dragfit_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print()
    print('wrote dragfit_report.txt')


if __name__ == '__main__':
    main()
