"""Explore before fitting: is a_body_xy really -c(s)*v_body_xy in FLIGHT?

Three things have to be true before any coefficient is worth quoting:
  1. the parked/grounded frames are gone (a parked drone on a 17.8 deg pad reads
     |a_xy| = 3.00 m/s^2 at zero speed -- pad normal force, not drag),
  2. the drag DIRECTION is antiparallel to body velocity (this simultaneously checks
     the body rotation built from the three truth streams AND producer's vel_bearing),
  3. |a_xy| actually grows with speed, and the growth law is identifiable.
"""
import numpy as np
import dragfit as DF

AIR_Z = -0.5      # NED down-positive: airborne means z below -0.5 m


def airborne(s):
    return s['p_world'][:, 2] < AIR_Z


def main():
    S = [x for x in (DF.session(n) for n in DF.VQ1) if x is not None]
    print('=== airborne fraction and in-flight ranges ===')
    for s in S:
        m = airborne(s)
        sp = np.linalg.norm(s['v_body'], axis=1)
        ah = np.linalg.norm(s['accel'][:, :2], axis=1)
        if m.sum() < 50:
            print('%-34s airborne %5.1f%%  -- too few' % (s['name'], 100 * m.mean()))
            continue
        print('%-34s airborne %5.1f%% (n=%6d)  s p50/p90/max %5.2f/%5.2f/%5.2f  '
              '|a_xy| p50/p90 %5.2f/%5.2f' %
              (s['name'], 100 * m.mean(), m.sum(), np.percentile(sp[m], 50),
               np.percentile(sp[m], 90), sp[m].max(), np.percentile(ah[m], 50),
               np.percentile(ah[m], 90)))

    # --- direction check -----------------------------------------------------------
    print()
    print('=== drag bearing vs truth body-velocity bearing (in flight) ===')
    print('   the producer computes atan2(-ay, -ax); truth is atan2(vy, vx) body')
    for s in S:
        m = airborne(s)
        if m.sum() < 200:
            continue
        ax, ay = s['accel'][m, 0], s['accel'][m, 1]
        vx, vy = s['v_body'][m, 0], s['v_body'][m, 1]
        sp = np.linalg.norm(s['v_body'][m], axis=1)
        ah = np.hypot(ax, ay)
        for lo, hi in ((0.0, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, 99.0)):
            k = (sp >= lo) & (sp < hi) & (ah > 0.05)
            if k.sum() < 50:
                continue
            d = np.degrees(np.angle(np.exp(1j * (np.arctan2(-ay[k], -ax[k])
                                                 - np.arctan2(vy[k], vx[k])))))
            print('%-30s s %4.0f-%4.0f n=%6d  bearing err med %7.1f deg  '
                  'p90|err| %6.1f  frac<20deg %.2f'
                  % (s['name'][:30], lo, hi, k.sum(), np.median(d),
                     np.percentile(np.abs(d), 90), np.mean(np.abs(d) < 20)))

    # --- magnitude law -------------------------------------------------------------
    print()
    print('=== |a_xy| vs |v_xy| (all airborne, pooled), binned ===')
    A, V, SP, W, T, N = [], [], [], [], [], []
    for s in S:
        m = airborne(s)
        if m.sum() < 50:
            continue
        A.append(s['accel'][m, :2])
        V.append(s['v_body'][m])
        SP.append(np.linalg.norm(s['v_body'][m], axis=1))
        W.append(np.linalg.norm(s['gyro'][m], axis=1))
        N += [s['name']] * int(m.sum())
    A = np.concatenate(A); V = np.concatenate(V); SP = np.concatenate(SP)
    W = np.concatenate(W); N = np.asarray(N)
    vxy = np.linalg.norm(V[:, :2], axis=1)
    ah = np.linalg.norm(A, axis=1)
    edges = [0, 1, 2, 3, 5, 7, 10, 13, 16, 20, 25, 30, 40]
    print(' |v_xy| bin      n    |a_xy| p25/p50/p75    a/v med   |v_z|/|v| med')
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (vxy >= lo) & (vxy < hi)
        if k.sum() < 30:
            continue
        r = ah[k] / np.maximum(vxy[k], 1e-6)
        print('%5.0f-%-5.0f %7d   %6.3f %6.3f %6.3f   %7.4f   %6.3f'
              % (lo, hi, k.sum(), np.percentile(ah[k], 25), np.percentile(ah[k], 50),
                 np.percentile(ah[k], 75), np.median(r),
                 np.median(np.abs(V[k, 2]) / np.maximum(SP[k], 1e-6))))
    np.savez('dragexplore.npz', A=A, V=V, SP=SP, W=W, N=N)
    print()
    print('saved dragexplore.npz  n=%d' % len(SP))


if __name__ == '__main__':
    main()
