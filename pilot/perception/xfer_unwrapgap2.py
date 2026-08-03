"""How comfortable was every branch re-anchor on the 121520 lap? A branch slip needs the
gyro-propagated psi to be more than 45 deg wrong when the next confident frame arrives."""
import csv
import numpy as np

HERE = 'C:/Users/USER/Projects/vqual-2/pilot/perception/'
for SESS in ['20260801-121520-vq2-lap-0-15']:
    rows = list(csv.DictReader(open(HERE + 'skylight_' + SESS + '.csv')))
    t = np.array([float(r['t_recv_wall_ns']) for r in rows])
    t = (t - t[0]) / 1e9
    conf = np.array([int(r['confident']) for r in rows]) > 0
    psi = np.array([float(r['psi_unwrapped']) if r['psi_unwrapped'] else np.nan for r in rows])
    pm = np.array([float(r['psi_mod90']) if r['psi_mod90'] else np.nan for r in rows])

    def w90(a):
        return (a + 45.0) % 90.0 - 45.0

    idx = np.where(conf)[0]
    gaps = np.diff(t[idx])
    out = []
    for k in np.argsort(-gaps)[:10]:
        i0, i1 = idx[k], idx[k + 1]
        # psi propagated by the gyro across the gap, as skylight wrote it, vs the fresh
        # mod-90 observation at re-anchor
        pred = psi[i1 - 1] if i1 - 1 > i0 else psi[i0]
        resid = w90(pm[i1] - pred)
        out.append((t[i0], gaps[k], resid))
    print('== %s ==' % SESS)
    print('  ten longest unconfident stretches: t_start, length, |snap residual| (deg)')
    for a, g, r in out:
        print('    t=%7.1f s  gap %5.2f s  snap residual %+6.2f deg  '
              '(margin to a 45 deg slip: %.1f deg)' % (a, g, r, 45 - abs(r)))
    # every re-anchor
    allres = np.array([w90(pm[idx[k + 1]] - psi[idx[k + 1] - 1])
                       for k in range(len(idx) - 1) if idx[k + 1] - idx[k] > 1])
    if len(allres):
        print('  all %d re-anchors after a gap: |residual| median %.2f  p90 %.2f  max %.2f deg'
              % (len(allres), np.median(np.abs(allres)),
                 np.percentile(np.abs(allres), 90), np.abs(allres).max()))
