"""Which grid axis is along-hangar? The strafe (gates 15->16) is along-hangar
translation; the chained light-motion increments live in grid coordinates, so their
dominant direction names the axis."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import skylight_pitch as SP

obs = SP._strafe_obs(SP.STRAFE)
idx = [i for i, o in enumerate(obs) if o]
cont = {}
th = obs[idx[0]]['theta']
for i in idx:
    th = th + float(SP.wrap90(obs[i]['theta'] - th))
    cont[i] = th


def rz(t):
    c, s = math.cos(math.radians(t)), math.sin(math.radians(t))
    return np.array([[c, s], [-s, c]])


moves = []
for a in range(len(idx) - 1):
    i, j = idx[a], idx[a + 1]
    if (obs[j]['t'] - obs[i]['t']) > 0.2e9:
        continue
    RA, RB = rz(cont[i]), rz(cont[j])
    mv = []
    for pa_lev, _c in obs[i]['lights']:
        pa = RA @ pa_lev
        best, bd = None, 1e9
        for pb_lev, _c2 in obs[j]['lights']:
            pb = RB @ pb_lev
            d = float(np.linalg.norm(pa - pb))
            if d < bd:
                best, bd = pb, d
        if best is not None and bd < 0.08:
            mv.append(best - pa)
    if len(mv) >= 2:
        moves.append(np.median(np.array(mv), axis=0))
M = np.array(moves)
tot = np.abs(M).sum(axis=0)
net = M.sum(axis=0)
ang = np.degrees(np.arctan2(net[1], net[0])) % 180.0
print(f'steps n={len(M)}  sum|d| axis0={tot[0]:.2f} axis1={tot[1]:.2f} (units of H)')
print(f'net displacement grid-frame: ({net[0]:+.2f}, {net[1]:+.2f}) H-units, '
      f'direction {ang:.1f} deg mod 180 (0 = axis0)')
