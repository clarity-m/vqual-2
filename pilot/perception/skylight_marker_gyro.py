"""Marker-hover constancy, gyro-compensated: within each marker window subtract the
levelled gyro yaw integral from psi so real aircraft yaw does not count against the
compass. Residual p-p/std is the compass's own noise."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import skylight as SK
import skylight_validate as SV

allpp = []
for name, pad in SV.MARKER_SESSIONS:
    rows = SV.load_skylight(name)
    imu = SK.Imu(os.path.join(SV.SESS, name))
    t = np.array([r['t'] for r in rows])
    for k, tm in enumerate(SV.markers(name)):
        lo, hi = np.searchsorted(t, [tm - pad * 1e9, tm + pad * 1e9])
        win = [r for r in rows[lo:hi] if r['conf']]
        if len(win) < 5:
            continue
        # residual = psi - gyro integral from window start, wrapped incrementally
        res = [0.0]
        for i in range(1, len(win)):
            dg = imu.yaw_lev_integral(win[i - 1]['t'], win[i]['t'])
            dm = float(SV.wrap90(win[i]['psi'] - win[i - 1]['psi']))
            res.append(res[-1] + dm - dg)
        res = np.array(res)
        pp, sd = float(res.max() - res.min()), float(res.std())
        gy = imu.yaw_lev_integral(win[0]['t'], win[-1]['t'])
        allpp.append(pp)
        print(f'{name[-24:]} marker {k:2d}: n={len(win):3d} true-yaw {gy:+6.2f} deg  '
              f'residual p-p {pp:5.2f} std {sd:4.2f}', flush=True)
pp = np.array(allpp)
print(f'>> {len(pp)} windows: gyro-compensated p-p median {np.median(pp):.2f} deg '
      f'(p90 {np.percentile(pp, 90):.2f}); {int((pp < 1.0).sum())}/{len(pp)} under 1 deg')
