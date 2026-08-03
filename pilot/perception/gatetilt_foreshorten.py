"""gatetilt_foreshorten.py -- a THIRD tilt channel that shares no failure mode with the
other two, built to settle the DIRECTION of the gate-9 lean (which way it leans, not just
by how much).

WHY A THIRD CHANNEL. Direction is the one thing the other two cannot give:

  * the PnP normal knows the direction but carries the IPPE TWIN, and the twin is exactly
    a mirror about the line of sight -- so it flips the lean direction. Resolving it by
    "which branch is more self-consistent" is close to circular, and the gate-9 vantages
    span only ~18 deg of azimuth, which is not enough to break the tie by geometry alone.
  * edge_tilt (image edge line vs gravity) is PnP-free but goes BLIND exactly here: the
    fitted lean azimuth is within 2 deg of the line of sight, i.e. the gate leans almost
    straight toward or away from the camera, and that is the one lean whose side edges
    still project as vertical lines. Gate 9 reads only ~4 deg on that channel, which is
    the blind spot behaving as documented, not a contradiction.

THE OBSERVABLE USED HERE. If a gate leans AWAY from the camera, its top edge is farther
away than its bottom edge and therefore projects SHORTER; leaning toward the camera makes
it longer. For a 2.7 m frame tilted 23 deg the top is displaced ~1.05 m in depth, which is
a 10-15% length difference at typical range -- far above quad noise. This uses only the
QUAD and the RANGE. No PnP rotation, no twin, no compass, no gravity vanishing point for
the estimate itself (gravity is used only to say which of the two edges is the top one).

Camera elevation is the confound and it is MODELLED, not assumed away: looking down on a
vertical gate also shortens its bottom edge. The forward model places the gate at the
measured range along the measured line of sight and projects it, so the vertical
hypothesis already predicts whatever ratio the viewing elevation implies. What is fitted
is the DEPARTURE from that.

VALIDATION IS BUILT IN: the same scan is run on every gate. The control gates must come
back near 0 deg. If they did not, the channel would be measuring the camera model.
"""

from __future__ import annotations

import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L                # noqa: E402
import gatetilt_strat as GS      # noqa: E402

K = np.array([[L.FX, 0.0, L.CX], [0.0, L.FY, L.CY], [0.0, 0.0, 1.0]])


def measured_ratio(row):
    """(len_top / len_bottom) of the quad's horizontal edge pair, or None.

    The horizontal pair is the one whose image lines are FARTHER from gravity-vertical
    (the complement of the pair edge_tilt uses). Top vs bottom is decided by which
    midpoint sits farther against gravity in the image.
    """
    q = np.asarray(row['quad'], float).reshape(4, 2)
    g = np.asarray(row['g_body'], float)
    g = g / max(np.linalg.norm(g), 1e-9)
    g_cam = L.body_to_cam() @ g                    # gravity DOWN, camera frame
    ang = []
    for j in range(4):
        p1 = np.array([q[j][0], q[j][1], 1.0])
        p2 = np.array([q[(j + 1) % 4][0], q[(j + 1) % 4][1], 1.0])
        n = K.T @ np.cross(p1, p2)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return None
        ang.append(abs(math.degrees(math.asin(
            max(-1.0, min(1.0, float(n @ g_cam) / nn))))))
    a02, a13 = 0.5 * (ang[0] + ang[2]), 0.5 * (ang[1] + ang[3])
    pair = (0, 2) if a02 > a13 else (1, 3)         # the HORIZONTAL pair
    segs = []
    for j in pair:
        p1, p2 = q[j], q[(j + 1) % 4]
        segs.append((0.5 * (p1 + p2), float(np.linalg.norm(p2 - p1))))
    # image direction of DOWN at the gate: toward gravity's vanishing point if it is in
    # front of the camera, away from it if behind
    vp = K @ g_cam
    c = q.mean(axis=0)
    if abs(vp[2]) > 1e-6:
        d = (vp[:2] / vp[2]) - c
        down = d * (1.0 if g_cam[2] > 0 else -1.0)
    else:
        down = np.array([L.FX * g_cam[0], L.FY * g_cam[1]])
    nd = np.linalg.norm(down)
    if nd < 1e-9:
        return None
    down /= nd
    s0 = float((segs[0][0] - c) @ down)
    top, bot = (segs[1], segs[0]) if s0 > 0 else (segs[0], segs[1])
    if bot[1] < 1e-6:
        return None
    return top[1] / bot[1]


def predicted_ratio(l_e, rng, s, phi_deg, lean_az_e_deg):
    """Forward-model len_top/len_bottom for a gate of half-size s at range rng.

    Frame: levelled, z UP, gate centre at the origin, camera at -rng * l_e.
    Normal n = (cos phi * d, -sin phi) with d the horizontal top-lean direction, so the
    top of the gate is displaced toward d -- the same convention as gatetilt_strat.
    """
    phi = math.radians(phi_deg)
    a = math.radians(lean_az_e_deg)
    d = np.array([math.cos(a), math.sin(a), 0.0])
    n = np.array([math.cos(phi) * d[0], math.cos(phi) * d[1], -math.sin(phi)])
    z = np.array([0.0, 0.0, 1.0])
    r = np.cross(z, n)
    nr = np.linalg.norm(r)
    if nr < 1e-6:
        return None
    r /= nr
    u = np.cross(n, r)
    u /= np.linalg.norm(u)
    if u[2] < 0:
        u = -u
    C = -rng * l_e
    e1 = np.cross(l_e, z)
    if np.linalg.norm(e1) < 1e-6:
        return None
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(l_e, e1)

    def proj(X):
        v = X - C
        dep = float(v @ l_e)
        if dep < 1e-3:
            return None
        return np.array([float(v @ e1) / dep, float(v @ e2) / dep])

    top = [proj(s * u + k * s * r) for k in (-1, 1)]
    bot = [proj(-s * u + k * s * r) for k in (-1, 1)]
    if any(p is None for p in top + bot):
        return None
    lb = float(np.linalg.norm(bot[1] - bot[0]))
    if lb < 1e-9:
        return None
    return float(np.linalg.norm(top[1] - top[0])) / lb


def rows_for(R, gate, min_px=GS.NEAR_PX):
    return [r for r in R.get(gate, []) if r['size_px'] >= min_px]


def scan(R, gate, mapj, phis=None, azs=None, min_px=GS.NEAR_PX):
    """Grid-search (tilt, lean azimuth) against the measured top/bottom length ratio."""
    rot, refl, offs = GS.sketch_frame(mapj)
    phis = np.arange(0.0, 36.0, 2.5) if phis is None else phis
    azs = np.arange(0.0, 360.0, 10.0) if azs is None else azs
    data = []
    for r in rows_for(R, gate, min_px):
        m = measured_ratio(r)
        if m is None or not (0.2 < m < 5.0):
            continue
        l = np.asarray(r['l_lev'], float)
        l = l / max(np.linalg.norm(l), 1e-9)
        l_e = np.array([l[0], l[1], -l[2]])          # levelled, z UP
        # this row's grid->sketch azimuth offset, so hypotheses live in the EXPORT frame
        b = float(offs.get(r['s'], 0.0))
        base = r['psi'] + b + rot                    # az_sketch = -(az_lev + base)
        s = 0.75 if r.get('src') == 'inner' else 1.35
        data.append((l_e, float(r['range']), s, base, math.log(m), r['s']))
    if len(data) < 20:
        return None
    best = None
    for phi in phis:
        for az in ([0.0] if phi == 0 else azs):
            res = []
            for l_e, rng, s, base, lm, _s in data:
                az_e = -(az) - base                  # export az -> levelled az
                p = predicted_ratio(l_e, rng, s, phi, az_e)
                if p is None or p <= 0:
                    res.append(1e3)
                    continue
                res.append(abs(lm - math.log(p)))
            sc = float(np.median(res))
            if best is None or sc < best[0]:
                best = (sc, float(phi), float(az % 360.0))
    # score of the vertical hypothesis for comparison
    res0 = []
    for l_e, rng, s, base, lm, _s in data:
        p = predicted_ratio(l_e, rng, s, 0.0, 0.0)
        res0.append(abs(lm - math.log(p)) if p and p > 0 else 1e3)
    return {'n': len(data), 'best_tilt_deg': round(best[1], 1),
            'best_lean_az_deg': round(best[2], 1),
            'resid_best': round(best[0], 4), 'resid_vertical': round(float(
                np.median(res0)), 4),
            'improvement': round(float(np.median(res0)) - best[0], 4)}


def gate9_two_hypotheses(R, mapj, az_a=129.7, phi=23.0):
    """Head-to-head: lean toward az_a vs the opposite, per session and pooled."""
    rot, refl, offs = GS.sketch_frame(mapj)
    per = collections.defaultdict(lambda: [[], [], []])
    for r in rows_for(R, 9):
        m = measured_ratio(r)
        if m is None or not (0.2 < m < 5.0):
            continue
        l = np.asarray(r['l_lev'], float)
        l = l / max(np.linalg.norm(l), 1e-9)
        l_e = np.array([l[0], l[1], -l[2]])
        base = r['psi'] + float(offs.get(r['s'], 0.0)) + rot
        s = 0.75 if r.get('src') == 'inner' else 1.35
        lm = math.log(m)
        for k, az in enumerate((None, az_a, (az_a + 180.0) % 360.0)):
            ph = 0.0 if az is None else phi
            aze = 0.0 if az is None else (-(az) - base)
            p = predicted_ratio(l_e, r['range'], s, ph, aze)
            per[r['s']][k].append(abs(lm - math.log(p)) if p and p > 0 else 1e3)
    print('%-42s %7s %9s %9s   verdict' % ('session', 'vert', 'az %.0f' % az_a,
                                           'az %.0f' % ((az_a + 180) % 360)))
    tot = [[], [], []]
    for s, v in sorted(per.items()):
        if len(v[0]) < 8:
            continue
        med = [float(np.median(x)) for x in v]
        for k in range(3):
            tot[k].extend(v[k])
        win = ('LEANS AWAY (az %.0f)' % az_a if med[1] == min(med) else
               'LEANS TOWARD (az %.0f)' % ((az_a + 180) % 360)
               if med[2] == min(med) else 'vertical')
        print('%-42s %7.4f %9.4f %9.4f   %s  (n=%d)'
              % (s.split('-', 1)[1], med[0], med[1], med[2], win, len(v[0])))
    med = [float(np.median(x)) for x in tot]
    print('%-42s %7.4f %9.4f %9.4f   pooled n=%d'
          % ('ALL', med[0], med[1], med[2], len(tot[0])))
    return med


def main():
    R = GS.load_rows()
    mapj = json.load(open(GS.MAP))
    print('=== free (tilt, lean-azimuth) fit to the top/bottom edge-length ratio')
    print('%-5s %6s %10s %12s %10s %10s %10s'
          % ('gate', 'n', 'tilt_deg', 'lean_az_deg', 'resid', 'resid_vert', 'gain'))
    out = {}
    for g in range(GS.N_GATES):
        s = scan(R, g, mapj)
        out[g] = s
        if s is None:
            print('%-5d   (too few usable near rows)' % g)
            continue
        print('%-5d %6d %10.1f %12.1f %10.4f %10.4f %10.4f'
              % (g, s['n'], s['best_tilt_deg'], s['best_lean_az_deg'],
                 s['resid_best'], s['resid_vertical'], s['improvement']))

    print('\n=== gate 9 head-to-head: which way does it lean? (lower = better fit)')
    med = gate9_two_hypotheses(R, mapj)
    json.dump({'scan': {str(k): v for k, v in out.items()},
               'gate9_head_to_head': {'vertical': med[0], 'lean_az_129.7': med[1],
                                      'lean_az_309.7': med[2]}},
              open(os.path.join(HERE, 'gatetilt_foreshorten.json'), 'w'), indent=1)
    print('\nwrote gatetilt_foreshorten.json')


if __name__ == '__main__':
    main()
