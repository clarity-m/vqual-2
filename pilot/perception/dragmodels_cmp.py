"""Model comparison. Within a session the quadratic law fits at R2 0.93-0.98, but the
coefficient moves 2.35x BETWEEN sessions -- so a regressor is missing, not the law.

The standard quadrotor decomposition says which one: horizontal force on a rotorcraft has
TWO terms, and only one of them is the airframe's parasitic drag.

  rotor drag / blade flapping:  F = -T * kh * v_perp     linear in v, PROPORTIONAL TO THRUST
  parasitic body drag:          F = -0.5 rho Cd A |v| v  quadratic in v, thrust-independent

Thrust is not directly observable, but its specific value is: |a_z| body IS the thrust
specific force (drag contributes to it only through the small body-z drag component).
So test a_xy = -(kh*|az| + kq*s + k1) * v_xy and see which terms survive.

Everything on the right is available under VQ2 (accelerometer only), which is the point.
"""
import numpy as np
import dragfit as DF

AIR_Z = -0.5
S_MIN = 2.0


def load():
    out = []
    for n in DF.VQ1:
        s = DF.session(n)
        if s is None:
            continue
        m = s['p_world'][:, 2] < AIR_Z
        if m.sum() < 200:
            continue
        g_body = np.einsum('nji,j->ni', s['R'][m], np.array([0.0, 0.0, 1.0]))
        out.append(dict(name=n, accel=s['accel'][m], gyro=s['gyro'][m],
                        v_body=s['v_body'][m], v_world=s['v_world'][m],
                        g_body=g_body, t=s['t'][m]))
    return out


TERMS = {
    'const': lambda az, s: np.ones_like(s),
    'speed': lambda az, s: s,
    'thrust': lambda az, s: az,
    'thrust*speed': lambda az, s: az * s,
}


def design(rows, terms):
    A = np.concatenate([r['accel'][:, :2] for r in rows])
    AZ = np.abs(np.concatenate([r['accel'][:, 2] for r in rows]))
    V = np.concatenate([r['v_body'] for r in rows])
    s = np.linalg.norm(V, axis=1)
    m = s >= S_MIN
    A, AZ, V, s = A[m], AZ[m], V[m], s[m]
    cols = [(V[:, :2] * TERMS[t](AZ, s)[:, None]).reshape(-1, 1) for t in terms]
    return np.concatenate(cols, 1), -A.reshape(-1), s


def fit(rows, terms):
    X, y, s = design(rows, terms)
    k, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ k
    r2 = 1.0 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)
    return k, r2, len(s)


def main():
    rows = load()
    combos = [('const',), ('speed',), ('thrust',), ('const', 'speed'),
              ('thrust', 'speed'), ('thrust', 'const'), ('thrust', 'const', 'speed'),
              ('thrust*speed',), ('thrust', 'thrust*speed'),
              ('thrust', 'thrust*speed', 'speed')]
    print('=== pooled fit, airborne, s >= %.0f m/s  (n = %d) ===' % (S_MIN, fit(rows, ('const',))[2]))
    for c in combos:
        k, r2, n = fit(rows, c)
        print('  R2 %.4f   %-38s  k = %s' % (r2, '+'.join(c), np.array2string(k, precision=5)))

    print()
    print('=== per-session stability of the best 2-term model (thrust + speed) ===')
    for r in rows:
        k, r2, n = fit([r], ('thrust', 'speed'))
        if n < 100:
            continue
        az = np.median(np.abs(r['accel'][:, 2]))
        print('  %-32s kh %.5f  kq %.5f  R2 %.3f  n %5d  med|az| %5.2f'
              % (r['name'], k[0], k[1], r2, n, az))

    print()
    print('=== per-session stability of thrust-only ===')
    for r in rows:
        k, r2, n = fit([r], ('thrust',))
        if n < 100:
            continue
        print('  %-32s kh %.5f  R2 %.3f  n %5d' % (r['name'], k[0], r2, n))

    print()
    print('=== thrust range check: does |az| actually vary between sessions? ===')
    for r in rows:
        az = np.abs(r['accel'][:, 2])
        s = np.linalg.norm(r['v_body'], axis=1)
        m = s >= S_MIN
        print('  %-32s |az| p10/p50/p90 %5.2f/%5.2f/%5.2f' %
              (r['name'], np.percentile(az[m], 10), np.percentile(az[m], 50),
               np.percentile(az[m], 90)))


if __name__ == '__main__':
    main()
