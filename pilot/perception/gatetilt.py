"""gatetilt.py -- is any VQ2 gate plane TILTED out of vertical?

WHY. mapdir.collect_angles() measures the gate plane's normal AZIMUTH (mod 180) in the
grid frame and REFUSES gates 8, 12, 13 at MAD > 20 deg -- the IPPE head-on valley: when
the gate is viewed nearly along its normal, the rotation is weakly observed and the two
IPPE solutions straddle the truth, so azimuth scatters. The pilot separately reports that
one gate is visibly TILTED (an edge-on screenshot shows the gate as a bar off plumb), so
the question is whether tilt survives the same degeneracy. It does -- tilt is a different
component and is far better conditioned.

TWO ESTIMATORS, in increasing order of trust:

  1. |elevation of the PnP normal| (tilt_of). The two IPPE solutions are mirror images
     about the line of sight, n_twin = 2(n.l)l - n, so with a near-horizontal l the twin
     flips the SIGN of the normal's elevation but preserves its magnitude. Robust to the
     twin, NOT robust to IPPE's rotation noise -- control gates read 3-15 deg here, so
     this channel is only good for ranking.

  2. edge_tilt(): PnP-FREE. The plane through the camera centre and a gate edge's image
     line has normal N = K^T l, and a gravity-vertical edge satisfies N.g = 0, so
     asin(N_hat . g_hat) is that edge's lean out of vertical -- image line and gravity
     only. Validated on an independent frame (gate8.png): the edge-on bar reads 19.7 deg
     against 3.9 deg of perspective lean for a true vertical at that image position and
     1.0-1.2 deg on two hangar columns in the same frame. Control gates read 1.3-4.9 deg.
     THIS is the estimator the export uses.

BLIND SPOT, stated because it decides how to read the result: a gate that leans BACK or
FORWARD projects its side edges as vertical lines when viewed HEAD-ON. The lean only
appears off-axis, and the detector's aspect filter discards the most oblique views, so
this measurement is weakest on approach-only gates. Absence of evidence is weak evidence
of absence here.

RESULT -- CORRECTED 2026-08-02. This module's --report POOLS every detection per gate and
reads the median, and on that statistic no gate looks tilted (every gate 1.3-4.9 deg,
gate 8 with the heaviest tail). THAT CONCLUSION IS AN ARTEFACT OF THE POOLING and is
overturned. Most rows are far and small, where a 20 deg lean is geometrically invisible
and the estimator returns noise, so pooling drags a tilted gate DOWN to the noise floor
and pushes vertical gates UP to it. Cut the SAME rows by measurement quality and gate 9
rises (14.6 -> 24.0 deg) while every other gate falls toward 1-4 deg; gate 8 is vertical.

    gatetilt_strat.py       quality-stratified magnitudes + the twin-resolved lean
                            DIRECTION (gate 9 leans toward azimuth 130 deg, export frame)
    gatetilt_obliquity.py   the cut that actually decides the MAGNITUDE: both estimators'
                            blind spots depend on view obliquity, not on size or range
    gatetilt_foreshorten.py a fourth channel, DISCARDED for failing its own controls

Standing verdict: gate 9 leans 21 +- 5 deg, top toward azimuth 130 deg; the other 16
gates are vertical to 1.5-4.0 deg. Do not read a per-gate verdict off --report's pooled
table. Full reasoning: pilot/course/build_tilt.py and pilot/course/README.md.

    python3 pilot/perception/gatetilt.py --collect     # -> gatetilt_rows.json
    python3 pilot/perception/gatetilt.py --report      # POOLED -- see the warning above
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L            # noqa: E402
import mapvq2 as M           # noqa: E402
import mapedges as ME_mod    # noqa: E402
import mapdir as MD          # noqa: E402

ROWS = os.path.join(HERE, 'gatetilt_rows.json')


def collect():
    """One row per clean NAMED detection at a compass-confident frame.

    Same selection as mapdir.collect_angles (so the comparison against the accepted
    azimuths is apples to apples), but keeps the full levelled normal, the line of
    sight, and the IPPE twin candidate.
    """
    out = collections.defaultdict(list)
    for sess in MD.BASE_SESSIONS:
        res = M.run(sess, verbose=False)
        S = res['S']
        C = MD.Compass(sess)
        if not C.ok:
            print('  no compass: %s' % os.path.basename(sess))
            continue
        byframe, _c = ME_mod.named_clean_by_frame(S, res['good'], res['ident'])
        n = 0
        for i, gd in byframe.items():
            terms, _why = C.bearing_terms(S.frames[i]['file'], float(S.t[i]))
            if terms is None:
                continue
            psi_u, R_lb = terms
            for g, d in gd.items():
                nb = np.asarray(d['normal_body'], float)
                pb = np.asarray(d['pos_body'], float)
                if not np.all(np.isfinite(nb)) or not np.all(np.isfinite(pb)):
                    continue
                nrm = np.linalg.norm(nb)
                if nrm < 1e-6:
                    continue
                n_lev = R_lb @ (nb / nrm)
                v_lev = R_lb @ pb
                rng = float(np.linalg.norm(v_lev))
                if rng < 1e-6:
                    continue
                l_lev = v_lev / rng
                twin = 2.0 * float(n_lev @ l_lev) * l_lev - n_lev
                g_body = R_lb.T @ np.array([0.0, 0.0, 1.0])   # gravity DOWN, body frame
                out[int(g)].append({
                    's': S.name,
                    't': float(S.t[i]),
                    'psi': float(psi_u),
                    'n_lev': [round(float(x), 5) for x in n_lev],
                    'twin': [round(float(x), 5) for x in twin],
                    'l_lev': [round(float(x), 5) for x in l_lev],
                    'g_body': [round(float(x), 5) for x in g_body],
                    'quad': [[round(float(c), 2) for c in p]
                             for p in np.asarray(d['quad'], float).reshape(4, 2)],
                    'range': round(rng, 3),
                    'size_px': round(float(d.get('size_px', 0.0)), 2),
                    'src': d.get('source', '?'),
                })
                n += 1
        print('  %-38s %5d rows' % (os.path.basename(sess), n))
    json.dump({str(g): v for g, v in sorted(out.items())}, open(ROWS, 'w'))
    print('wrote %s: %s' % (ROWS, {g: len(v) for g, v in sorted(out.items())}))


# ---------------------------------------------------------------------------------------


def _circ_median_mod180(vals):
    a = np.radians(np.asarray(vals, float) * 2.0)
    c, s = np.cos(a).mean(), np.sin(a).mean()
    m = (math.degrees(math.atan2(s, c)) / 2.0) % 180.0
    dev = np.abs((np.asarray(vals, float) - m + 90.0) % 180.0 - 90.0)
    return m, float(np.median(dev))


def tilt_of(row, which='n_lev'):
    """|elevation| of the plane normal, degrees. z of the levelled frame is DOWN."""
    z = float(row[which][2])
    return abs(math.degrees(math.asin(max(-1.0, min(1.0, z)))))


def signed_tilt(row, which='n_lev'):
    """+ = normal tilts UP (gate leans back, top away from the approach); - = down."""
    z = float(row[which][2])
    return -math.degrees(math.asin(max(-1.0, min(1.0, z))))


def los_elev(row):
    return -math.degrees(math.asin(max(-1.0, min(1.0, float(row['l_lev'][2])))))


def edge_tilt(row):
    """PnP-FREE tilt: how far the gate's side edges are from GRAVITY-vertical.

    The plane through the camera centre and the image line of one gate edge has normal
    N = K^T l  (a point X on the line satisfies l.(K X) = 0, i.e. (K^T l).X = 0).  The
    edge's 3-D direction lies in that plane, so if the edge were vertical, gravity would
    lie in it too:  N_hat . g_hat = 0.  Hence

        phi = asin(N_hat . g_hat)

    is the edge's lean out of vertical, computed from the IMAGE LINE and GRAVITY only --
    no PnP rotation, no IPPE twin, no compass.  Of the quad's two edge pairs we take the
    pair that is closer to vertical (unambiguous for any tilt below 45 deg) and return
    the mean of its two edges. The other pair is returned as horiz_lean purely as a
    sanity channel: it must read near 90 deg (those edges are the horizontal ones), and
    it does -- 63-86 deg across all 17 gates. If it ever came back small, the pair
    selection would be broken.

    NOTE the two side edges genuinely DISAGREE on a strongly leaning gate seen in
    perspective (they converge on the tilted vanishing point), so do NOT filter rows on
    pair_disagree -- that silently deletes exactly the tilted gates.

    This is the estimator that matches what the eye sees in an edge-on frame: a gate
    rendered as a narrow bar, and the bar visibly off plumb.
    """
    q = np.asarray(row['quad'], float).reshape(4, 2)
    g = np.asarray(row['g_body'], float)
    g = g / max(np.linalg.norm(g), 1e-9)
    g_cam = L.body_to_cam() @ g
    K = np.array([[L.FX, 0.0, L.CX], [0.0, L.FY, L.CY], [0.0, 0.0, 1.0]])
    KT = K.T
    ang = []
    for j in range(4):
        p1 = np.array([q[j][0], q[j][1], 1.0])
        p2 = np.array([q[(j + 1) % 4][0], q[(j + 1) % 4][1], 1.0])
        lin = np.cross(p1, p2)
        n = KT @ lin
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return None
        n /= nn
        ang.append(math.degrees(math.asin(max(-1.0, min(1.0, float(n @ g_cam))))))
    # pair (0,2) and (1,3) are the opposite-edge pairs of an ordered quad
    a02 = 0.5 * (abs(ang[0]) + abs(ang[2]))
    a13 = 0.5 * (abs(ang[1]) + abs(ang[3]))
    if a02 <= a13:
        vert, horiz = (ang[0], ang[2]), (ang[1], ang[3])
    else:
        vert, horiz = (ang[1], ang[3]), (ang[0], ang[2])
    lean = 0.5 * (vert[0] + vert[1]) if vert[0] * vert[1] > 0 else \
        0.5 * (abs(vert[0]) + abs(vert[1])) * (1 if vert[0] + vert[1] >= 0 else -1)
    return {
        'lean': float(lean),                       # signed, camera-relative
        'lean_abs': float(0.5 * (abs(vert[0]) + abs(vert[1]))),
        'pair_disagree': float(abs(vert[0] - vert[1])),
        'horiz_lean': float(0.5 * (abs(horiz[0]) + abs(horiz[1]))),
    }


def resolve_axis(rows, iters=4000, seed=0):
    """Vote the two IPPE candidates of every row onto ONE plane axis (mod 180).

    Different vantages mirror about different lines of sight, so the twin that is a
    measurement artefact moves between rows while the true normal does not. RANSAC on
    the axis (|dot| -- the normal's sign is not observable from a square).
    """
    if len(rows) < 6:
        return None
    cand = np.array([r['n_lev'] for r in rows] + [r['twin'] for r in rows], float)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True).clip(1e-9)
    rng = np.random.default_rng(seed)
    n = len(rows)
    best, best_in = None, -1
    TOL = math.cos(math.radians(12.0))
    for _ in range(iters):
        u = cand[rng.integers(0, len(cand))]
        for _refine in range(3):
            d = np.abs(cand @ u)
            keep = cand[d > TOL] * np.sign(cand[d > TOL] @ u)[:, None]
            if len(keep) < 3:
                break
            u = keep.mean(axis=0)
            u /= np.linalg.norm(u)
        # inlier count per ROW (a row is an inlier if EITHER of its candidates fits)
        dn = np.abs(np.array([r['n_lev'] for r in rows], float) @ u)
        dt = np.abs(np.array([r['twin'] for r in rows], float) @ u)
        ninl = int(np.sum(np.maximum(dn, dt) > TOL))
        if ninl > best_in:
            best_in, best = ninl, u.copy()
    u = best
    dn = np.abs(np.array([r['n_lev'] for r in rows], float) @ u)
    dt = np.abs(np.array([r['twin'] for r in rows], float) @ u)
    pick = np.where(dn >= dt, 0, 1)
    sel = np.array([rows[i]['n_lev'] if pick[i] == 0 else rows[i]['twin']
                    for i in range(n)], float)
    resid = np.degrees(np.arccos(np.abs(sel @ u).clip(0, 1)))
    inl = resid < 12.0
    if inl.sum() >= 3:
        S = sel[inl] * np.sign(sel[inl] @ u)[:, None]
        u = S.mean(axis=0)
        u /= np.linalg.norm(u)
        resid = np.degrees(np.arccos(np.abs(sel @ u).clip(0, 1)))
        inl = resid < 12.0
    if u[2] > 0:            # report the normal pointing UP-ish (z is DOWN in the frame)
        u = -u
    return {
        'axis_lev': [round(float(x), 4) for x in u],
        'tilt_deg': round(float(abs(math.degrees(math.asin(max(-1.0, min(1.0, u[2])))))), 2),
        'inlier_frac': round(float(inl.mean()), 3),
        'n': n,
        'resid_median_deg': round(float(np.median(resid[inl])) if inl.any() else float('nan'), 2),
    }


def report():
    rows = {int(k): v for k, v in json.load(open(ROWS)).items()}
    mapj = json.load(open(os.path.join(HERE, 'map_vq2.json')))
    accepted = mapj['layout_directions_2026_08_02']['gate_plane_angles_deg']

    print('\n=== |normal elevation| = plane tilt out of vertical (gravity-referenced, '
          'no compass)')
    print('%-5s %6s %7s %7s %7s %7s %7s %7s  %s' %
          ('gate', 'n', 'med', 'MAD', 'p25', 'p75', 'sgn-med', 'losEl', 'map azimuth'))
    summary = {}
    for g in sorted(rows):
        rr = rows[g]
        t = np.array([tilt_of(r) for r in rr])
        st = np.array([signed_tilt(r) for r in rr])
        le = np.array([los_elev(r) for r in rr])
        med = float(np.median(t))
        mad = float(np.median(np.abs(t - med)))
        acc = accepted.get(str(g))
        tag = ('az %6.1f MAD %4.1f n=%d' % (acc['normal_az_sketch_frame_mod180'],
                                            acc['mad_deg'], acc['n'])) if acc \
            else 'REFUSED (head-on valley)'
        print('%-5d %6d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f  %s' %
              (g, len(rr), med, mad, float(np.percentile(t, 25)),
               float(np.percentile(t, 75)), float(np.median(st)),
               float(np.median(le)), tag))
        summary[g] = {'n': len(rr), 'tilt_median_deg': round(med, 2),
                      'tilt_mad_deg': round(mad, 2),
                      'signed_median_deg': round(float(np.median(st)), 2),
                      'los_elev_median_deg': round(float(np.median(le)), 2),
                      'map_azimuth_accepted': bool(acc)}

    print('\n=== viewpoint dependence: median |tilt| split by SESSION and by RANGE')
    for g in sorted(rows):
        rr = rows[g]
        by = collections.defaultdict(list)
        for r in rr:
            by[r['s']].append(tilt_of(r))
        parts = ' '.join('%s:%.1f(n%d)' % (s.split('-')[1], float(np.median(v)), len(v))
                         for s, v in sorted(by.items()))
        near = [tilt_of(r) for r in rr if r['range'] < 15]
        far = [tilt_of(r) for r in rr if r['range'] >= 15]
        rp = 'near<15m %.1f(n%d)  far %.1f(n%d)' % (
            float(np.median(near)) if near else float('nan'), len(near),
            float(np.median(far)) if far else float('nan'), len(far))
        print(' gate %2d  %s   | %s' % (g, rp, parts))

    # --- PnP-FREE referee -------------------------------------------------------------
    # edge_tilt() uses only the IMAGE LINE of the gate's side edges and GRAVITY, so it
    # shares no failure mode with the IPPE twin problem it is refereeing. This is the
    # channel that decides Claire's verticality claim.
    print('\n=== EDGE TILT (PnP-free: image edge line vs gravity). |lean| of the near-'
          'vertical edge pair; horiz = the consistency channel (should track lean)')
    print('%-5s %6s %8s %8s %8s %9s %9s %8s' %
          ('gate', 'n', 'lean_med', 'lean_MAD', 'sgn_med', 'pairdis', 'horiz_med',
           'n>10deg'))
    edge = {}
    for g in sorted(rows):
        vals, sgn, pdis, hz = [], [], [], []
        for r in rows[g]:
            e = edge_tilt(r)
            if e is None or r['size_px'] < 25.0:
                continue           # sub-25 px quads: edge lines are corner noise
            vals.append(e['lean_abs'])
            sgn.append(e['lean'])
            pdis.append(e['pair_disagree'])
            hz.append(e['horiz_lean'])
        if len(vals) < 20:
            print('%-5d %6d  (too few >=25 px rows)' % (g, len(vals)))
            continue
        v = np.array(vals)
        med = float(np.median(v))
        edge[g] = {'n': len(v), 'lean_median_deg': round(med, 2),
                   'lean_mad_deg': round(float(np.median(np.abs(v - med))), 2),
                   'signed_median_deg': round(float(np.median(sgn)), 2),
                   'pair_disagree_median_deg': round(float(np.median(pdis)), 2),
                   'horiz_lean_median_deg': round(float(np.median(hz)), 2),
                   'frac_gt10deg': round(float((v > 10).mean()), 3)}
        print('%-5d %6d %8.2f %8.2f %8.2f %9.2f %9.2f %7.1f%%' %
              (g, len(v), med, edge[g]['lean_mad_deg'], edge[g]['signed_median_deg'],
               edge[g]['pair_disagree_median_deg'], edge[g]['horiz_lean_median_deg'],
               100 * edge[g]['frac_gt10deg']))

    print('\n=== edge tilt vs VIEWPOINT (a real tilt is a gate property: must not move)')
    for g in sorted(edge):
        by = collections.defaultdict(list)
        for r in rows[g]:
            e = edge_tilt(r)
            if e is not None and r['size_px'] >= 25.0:
                by[r['s']].append(e['lean_abs'])
        parts = ' '.join('%s:%.1f(n%d)' % (s.split('-')[1], float(np.median(v)), len(v))
                         for s, v in sorted(by.items()) if len(v) >= 8)
        print(' gate %2d  %s' % (g, parts))

    print('\n=== twin-resolved 3-D plane axis (RANSAC vote over both IPPE solutions)')
    res = {}
    for g in sorted(rows):
        r = resolve_axis(rows[g])
        res[g] = r
        if r:
            print(' gate %2d  tilt %5.2f deg  inliers %5.1f%%  resid_med %4.1f deg  n=%d'
                  % (g, r['tilt_deg'], 100 * r['inlier_frac'], r['resid_median_deg'],
                     r['n']))
    json.dump({'per_gate': {str(k): v for k, v in summary.items()},
               'edge_tilt': {str(k): v for k, v in edge.items()},
               'resolved_axis': {str(k): v for k, v in res.items()}},
              open(os.path.join(HERE, 'gatetilt_summary.json'), 'w'), indent=1)
    print('\nwrote gatetilt_summary.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.collect:
        collect()
    if a.report:
        report()
    if not (a.collect or a.report):
        ap.print_help()


if __name__ == '__main__':
    main()
