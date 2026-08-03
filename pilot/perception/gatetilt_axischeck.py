"""gatetilt_axischeck.py -- can the gate-9 lean DIRECTION be a twin artefact?

The IPPE twin is the one failure mode that could fake a direction: the mirror
n_twin = 2(n.l)l - n flips the normal's vertical component, so picking the wrong twin
gives the OPPOSITE lean. If every gate-9 row were seen from one vantage, a systematically
wrong pick would look perfectly consistent.

THE TEST. The twin is a mirror ABOUT THE LINE OF SIGHT, so the wrong branch's lean
direction MUST move as the line of sight moves, while the true one must not. So:
  1. how wide is the spread of viewing azimuths over the accepted rows?
  2. does the SELECTED lean direction depend on viewing azimuth? (must not)
  3. does the REJECTED (forced-twin) lean direction depend on it? (must, if vantages vary)
  4. do the two halves of the vantage range, split at the median, give the same answer?

Also cross-checks the lean azimuth against the map's independently measured gate-plane
normal azimuth (mod 180) -- a lean must be perpendicular to the gate plane, so the two
have to agree mod 180. That channel used no twin resolution at all.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gatetilt_strat as GS      # noqa: E402

GATE = 9


def circ(vals):
    a = np.radians(np.asarray(vals, float))
    c, s = np.cos(a).mean(), np.sin(a).mean()
    return (math.degrees(math.atan2(s, c)) % 360.0, float(np.hypot(c, s)))


def lean_az(n):
    d = -float(n[2]) * np.array([n[0], n[1]])
    if np.linalg.norm(d) < 1e-6:
        return None
    return math.degrees(math.atan2(d[1], d[0])) % 360.0


def main():
    R = GS.load_rows()
    mapj = json.load(open(GS.MAP))
    rot, refl, offs = GS.sketch_frame(mapj)
    rows = [r for r in R[GATE] if r['size_px'] >= GS.NEAR_PX]

    prim, twin, view, sess = [], [], [], []
    for r in rows:
        b = float(offs.get(r['s'], 0.0))
        prim.append(GS.to_sketch(r['n_lev'], r['psi'], b, rot, refl))
        twin.append(GS.to_sketch(r['twin'], r['psi'], b, rot, refl))
        l = GS.to_sketch(r['l_lev'], r['psi'], b, rot, refl)
        view.append(math.degrees(math.atan2(l[1], l[0])) % 360.0)
        sess.append(r['s'])
    prim, twin, view = np.array(prim), np.array(twin), np.array(view)

    d = GS.direction(R, GATE, mapj)
    u = np.array(d['axis_sketch'], float)
    dn, dt = np.abs(prim @ u), np.abs(twin @ u)
    take_prim = dn >= dt
    sel = np.where(take_prim[:, None], prim, twin)
    rej = np.where(take_prim[:, None], twin, prim)

    vm, vR = circ(view)
    dev = np.abs((view - vm + 180.0) % 360.0 - 180.0)
    print('viewing azimuth (camera->gate, export frame): median %.1f deg, spread '
          'p05-p95 = %.1f..%.1f deg, half-range %.1f deg, n=%d'
          % (vm, np.percentile(view, 5), np.percentile(view, 95), dev.max(), len(view)))

    sa = [lean_az(n) for n in sel]
    ra = [lean_az(n) for n in rej]
    ma, mR = circ(sa)
    ta, tR = circ(ra)
    print('SELECTED  lean azimuth %.1f deg, concentration R=%.3f' % (ma, mR))
    print('REJECTED  lean azimuth %.1f deg, concentration R=%.3f   (twin branch)'
          % (ta, tR))
    print('separation between the two branches: %.1f deg'
          % abs((ma - ta + 180.0) % 360.0 - 180.0))

    # dependence on viewing azimuth: split at the median vantage
    lo = view < np.median(view)
    for name, arr in (('selected', sa), ('rejected', ra)):
        a1, r1 = circ([x for x, k in zip(arr, lo) if k])
        a2, r2 = circ([x for x, k in zip(arr, lo) if not k])
        print('%-9s  near-vantage half %.1f (R=%.2f, n=%d) | far half %.1f (R=%.2f, n=%d)'
              '  -> shift %.1f deg'
              % (name, a1, r1, int(lo.sum()), a2, r2, int((~lo).sum()),
                 abs((a1 - a2 + 180.0) % 360.0 - 180.0)))

    # per session
    print('\nper session (independent flights, independent compass branches):')
    for s in sorted(set(sess)):
        k = [i for i in range(len(sess)) if sess[i] == s]
        if len(k) < 8:
            continue
        a, r = circ([sa[i] for i in k])
        vmm, _ = circ(view[k])
        print('  %-40s n=%3d  view_az %6.1f  lean_az %6.1f  R=%.3f  branch +%.0f'
              % (s.split('-', 1)[1], len(k), vmm, a, r, offs.get(s, 0.0)))

    # independent channel: the map's own gate-plane azimuth (no twin resolution)
    ga = mapj['layout_directions_2026_08_02']['gate_plane_angles_deg'].get(str(GATE))
    if ga:
        az180 = float(ga['normal_az_sketch_frame_mod180'])
        err = abs((ma - az180 + 90.0) % 180.0 - 90.0)
        print('\nmap gate-plane normal azimuth (mod 180, n=%d, MAD %.1f): %.2f deg'
              % (ga['n'], ga['mad_deg'], az180))
        print('lean azimuth vs that plane normal, mod 180: %.1f deg apart '
              '(a lean must be PERPENDICULAR to the gate plane, i.e. 0 apart)' % err)

    # what the tilt means for an approach: angle between the gate normal and the
    # horizontal race line through the gate
    P = {int(k): np.array([v['x'], v['y'], v['z_up_m']], float) for k, v in
         mapj['layout_directions_2026_08_02']['positions_m'].items()}
    t_in = P[GATE] - P[GATE - 1]
    t_out = P[GATE + 1] - P[GATE]
    for nm, t in (('in 8->9', t_in), ('out 9->10', t_out)):
        taz = math.degrees(math.atan2(t[1], t[0])) % 360.0
        print('race leg %-10s azimuth %6.1f deg  ->  lean is %+6.1f deg from it'
              % (nm, taz, (ma - taz + 180.0) % 360.0 - 180.0))


if __name__ == '__main__':
    main()
