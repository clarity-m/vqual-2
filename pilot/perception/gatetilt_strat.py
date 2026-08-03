"""gatetilt_strat.py -- STRATIFIED re-derivation of per-gate tilt out of vertical.

WHY THIS FILE EXISTS. gatetilt.py --report pooled every detection and read the MEDIAN
over all 8500 rows. That median is dominated by FAR, SMALL detections, where the tilt of
a plane is geometrically almost invisible (a 20 deg lean changes a 30 px quad by a couple
of pixels) and the estimator returns noise centred a few degrees above zero. Pooling
therefore pulls a genuinely tilted gate DOWN toward the noise floor and pushes every
vertical gate UP to it -- the two become indistinguishable. The verdict "no gate reads
consistently tilted" was an artefact of that pooling, not a measurement.

THE FIX is to stratify by MEASUREMENT QUALITY and look at the TREND, not the level:

    a real tilt is a property of the gate, so it must SURVIVE as views get closer and
    larger; estimator noise must SHRINK.

Gate 9 rises 14.6 -> 24.0 deg going from all rows to >=60 px rows. Every other gate falls
toward 1-4 deg. That divergence is the signal; no single pooled number can show it.

ACCEPTED RULE (frozen before looking at the direction result):
    rows with size_px >= 60 (about 25 m range or nearer for a 2.7 m frame),
    per-session medians first, then the median OVER SESSIONS -- so one long flight
    cannot outvote three others -- and sigma from the spread of the session medians.

TWO INDEPENDENT CHANNELS, both reported:
    pnp  : |elevation| of the PnP plane normal, deg(asin|n_lev.z|). Twin-robust in
           MAGNITUDE (the IPPE mirror about a near-horizontal line of sight flips the
           sign of the normal's elevation but preserves its size), NOT in DIRECTION.
    edge : gatetilt.edge_tilt -- PnP-free, image edge line vs gravity. Shares no failure
           mode with PnP. Blind to a lean seen exactly head-on.

DIRECTION (task 2). The magnitude alone is useless to a policy: it needs to know which
way the gate leans. The plane normal is an AXIS -- its overall sign is not observable
from a square -- but the LEAN DIRECTION of the gate's top is invariant under n -> -n:

    top-lean direction   d ~ -n_z * n_horiz          (both factors flip together)
    tilt magnitude       phi = |asin(n_z)|

so d is well defined once the IPPE TWIN is resolved (the twin is a different plane, not
a sign flip: it flips n_z while mirroring n_horiz about the line of sight). We resolve it
by RANSAC voting in the SKETCH frame -- the twin artefact moves between vantages, the
true normal does not -- and then require the answer to repeat per session.

Frame conversion, row -> export (sketch) frame:
    az_grid   = psi_unwrapped + atan2(n_lev.y, n_lev.x) + session_branch_offset
    az_sketch = -(az_grid + grid_to_sketch_rotation_deg)      [reflection = True]
    z_up      = -n_lev.z                                      [levelled frame z is DOWN]

    python3 pilot/perception/gatetilt_strat.py
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
import gatetilt as GT             # noqa: E402

ROWS = os.path.join(HERE, 'gatetilt_rows.json')
MAP = os.path.join(HERE, 'map_vq2.json')
OUT = os.path.join(HERE, 'gatetilt_strat.json')

NEAR_PX = 60.0          # the accepted quality rule
MIN_SESS_ROWS = 8       # a session median needs this many rows to count
N_GATES = 17


def load_rows():
    return {int(k): v for k, v in json.load(open(ROWS)).items()}


def pnp_tilt(r):
    n = np.asarray(r['n_lev'], float)
    n = n / max(np.linalg.norm(n), 1e-9)
    return math.degrees(math.asin(min(1.0, abs(float(n[2])))))


def edge_lean(r):
    e = GT.edge_tilt(r)
    return None if e is None else e['lean_abs']


def _med(v):
    return float(np.median(v)) if len(v) else float('nan')


def sess_pooled(vals_by_sess, min_rows=MIN_SESS_ROWS):
    """(median over session medians, robust sigma, n_sessions, n_rows, detail)."""
    meds, tot, detail = [], 0, []
    for s, v in sorted(vals_by_sess.items()):
        if len(v) < min_rows:
            continue
        m = _med(v)
        meds.append(m)
        tot += len(v)
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        detail.append({'session': s, 'median_deg': round(m, 2), 'n': len(v),
                       'iqr_deg': round(iqr, 2)})
    if not meds:
        return None
    meds = np.array(meds)
    pooled = float(np.median(meds))
    # robust spread of the session medians; floor at 1 deg so a lucky 2-session
    # agreement cannot claim sub-degree knowledge of a vision measurement
    if len(meds) >= 2:
        sig = max(1.0, float(1.4826 * np.median(np.abs(meds - pooled))))
    else:
        sig = 4.0
    return {'deg': pooled, 'sigma_deg': sig, 'n_sessions': len(meds), 'n_rows': tot,
            'per_session': detail}


# ---------------------------------------------------------------------------------------
# Part 1 -- the stratification that overturns the pooled verdict


def strata_table(R):
    strata = [
        ('all', lambda r: True),
        ('>=60px', lambda r: r['size_px'] >= 60.0),
        ('>=120px', lambda r: r['size_px'] >= 120.0),
        ('<=15m', lambda r: r['range'] <= 15.0),
    ]
    print('=== PnP-normal tilt, deg(asin|n_lev.z|): pooled median by stratum')
    print('%-5s %-22s %-22s %-22s %-22s  %s'
          % ('gate', 'all', '>=60px', '>=120px', '<=15m', 'trend'))
    tab = {}
    for g in range(N_GATES):
        rows = R.get(g, [])
        cells, vals = [], {}
        for name, f in strata:
            v = [pnp_tilt(r) for r in rows if f(r)]
            vals[name] = v
            cells.append('%6.1f (n=%4d)' % (_med(v), len(v)) if v else '    --       ')
        a, b = _med(vals['all']), _med(vals['>=60px'])
        trend = 'RISES  <<<' if (len(vals['>=60px']) >= 20 and b > a + 2.0) else (
            'falls' if len(vals['>=60px']) >= 20 else 'thin')
        print('%-5d %-22s %-22s %-22s %-22s  %s'
              % (g, cells[0], cells[1], cells[2], cells[3], trend))
        tab[g] = {name: (round(_med(vals[name]), 2) if vals[name] else None,
                         len(vals[name])) for name, _f in strata}
    return tab


def magnitudes(R):
    """The accepted estimate: >=60 px rows, per-session medians, median over sessions."""
    print('\n=== ACCEPTED RULE: size_px >= %.0f, per-session medians -> median over '
          'sessions' % NEAR_PX)
    print('%-5s %8s %8s %6s %6s %8s %8s   %s'
          % ('gate', 'pnp_deg', 'sigma', 'nsess', 'nrows', 'edge_deg', 'edge_sig',
             'per-session pnp medians'))
    out = {}
    for g in range(N_GATES):
        rows = [r for r in R.get(g, []) if r['size_px'] >= NEAR_PX]
        by_p, by_e = collections.defaultdict(list), collections.defaultdict(list)
        for r in rows:
            by_p[r['s']].append(pnp_tilt(r))
            e = edge_lean(r)
            if e is not None:
                by_e[r['s']].append(e)
        P = sess_pooled(by_p)
        E = sess_pooled(by_e)
        if P is None:
            print('%-5d %8s  (too few near rows: %d)' % (g, '--', len(rows)))
            out[g] = {'pnp': None, 'edge': None, 'n_near_rows': len(rows)}
            continue
        det = ' '.join('%.1f(n%d,iqr%.1f)' % (d['median_deg'], d['n'], d['iqr_deg'])
                       for d in P['per_session'])
        print('%-5d %8.1f %8.1f %6d %6d %8s %8s   %s'
              % (g, P['deg'], P['sigma_deg'], P['n_sessions'], P['n_rows'],
                 '%.1f' % E['deg'] if E else '--',
                 '%.1f' % E['sigma_deg'] if E else '--', det))
        out[g] = {'pnp': P, 'edge': E, 'n_near_rows': len(rows)}
    return out


# ---------------------------------------------------------------------------------------
# Part 2 -- direction


def sketch_frame(mapj):
    F = mapj['layout_directions_2026_08_02']['frame']
    rot = float(F['grid_to_sketch_rotation_deg'])
    refl = bool(F['grid_to_sketch_reflection'])
    offs = mapj['layout_directions_2026_08_02']['session_branch_offsets_deg']
    return rot, refl, offs


def to_sketch(vec_lev, psi_u, branch, rot, refl):
    """Levelled-body-frame (z DOWN) unit vector -> export/sketch frame (z UP)."""
    v = np.asarray(vec_lev, float)
    v = v / max(np.linalg.norm(v), 1e-9)
    h = float(np.hypot(v[0], v[1]))
    az_grid = psi_u + math.degrees(math.atan2(v[1], v[0])) + branch
    az = math.radians((-(az_grid + rot)) % 360.0 if refl else (az_grid + rot) % 360.0)
    return np.array([h * math.cos(az), h * math.sin(az), -float(v[2])])


def _ransac_axis(cand, iters=6000, tol_deg=15.0, seed=0):
    """Vote candidate unit vectors (both twins of every row) onto one axis."""
    rng = np.random.default_rng(seed)
    TOL = math.cos(math.radians(tol_deg))
    best, best_in = None, -1
    for _ in range(iters):
        u = cand[rng.integers(0, len(cand))]
        for _r in range(3):
            d = np.abs(cand @ u)
            keep = cand[d > TOL]
            if len(keep) < 3:
                break
            keep = keep * np.sign(keep @ u)[:, None]
            u = keep.mean(axis=0)
            u /= np.linalg.norm(u)
        ninl = int((np.abs(cand @ u) > TOL).sum())
        if ninl > best_in:
            best_in, best = ninl, u.copy()
    return best


def direction(R, gate, mapj, min_px=NEAR_PX):
    """Resolve the IPPE twin in the SKETCH frame and report the gate's top-lean vector."""
    rot, refl, offs = sketch_frame(mapj)
    rows = [r for r in R.get(gate, []) if r['size_px'] >= min_px]
    prim, twin, sess = [], [], []
    for r in rows:
        b = float(offs.get(r['s'], 0.0))
        prim.append(to_sketch(r['n_lev'], r['psi'], b, rot, refl))
        twin.append(to_sketch(r['twin'], r['psi'], b, rot, refl))
        sess.append(r['s'])
    if len(prim) < 10:
        return None
    prim, twin = np.array(prim), np.array(twin)
    u = _ransac_axis(np.vstack([prim, twin]))
    dn, dt = np.abs(prim @ u), np.abs(twin @ u)
    sel = np.where((dn >= dt)[:, None], prim, twin)
    pick_prim = float((dn >= dt).mean())
    # refine on inliers
    resid = np.degrees(np.arccos(np.abs(sel @ u).clip(0, 1)))
    inl = resid < 15.0
    if inl.sum() >= 3:
        S = sel[inl] * np.sign(sel[inl] @ u)[:, None]
        u = S.mean(axis=0)
        u /= np.linalg.norm(u)
        resid = np.degrees(np.arccos(np.abs(sel @ u).clip(0, 1)))
        inl = resid < 15.0

    # top-lean direction per row: d ~ -n_z * n_horiz  (invariant under n -> -n)
    def lean_vec(n):
        d = -float(n[2]) * np.array([n[0], n[1]])
        m = np.linalg.norm(d)
        return None if m < 1e-6 else d / m

    az_rows, by_s = [], collections.defaultdict(list)
    for i in range(len(sel)):
        d = lean_vec(sel[i])
        if d is None:
            continue
        a = math.degrees(math.atan2(d[1], d[0])) % 360.0
        az_rows.append(a)
        by_s[sess[i]].append(a)

    def circ(vals):
        a = np.radians(np.asarray(vals, float))
        c, s = np.cos(a).mean(), np.sin(a).mean()
        return (math.degrees(math.atan2(s, c)) % 360.0,
                float(np.hypot(c, s)))            # R = concentration, 1 = perfect

    az, conc = circ(az_rows)
    per_s = {s: (round(circ(v)[0], 1), round(circ(v)[1], 3), len(v))
             for s, v in sorted(by_s.items()) if len(v) >= MIN_SESS_ROWS}
    dev = np.abs((np.asarray(az_rows) - az + 180.0) % 360.0 - 180.0)
    # tilt magnitude implied by the resolved axis
    tilt_axis = abs(math.degrees(math.asin(min(1.0, abs(float(u[2]))))))
    return {
        'n_rows': len(sel), 'axis_sketch': [round(float(x), 4) for x in u],
        'axis_tilt_deg': round(tilt_axis, 2),
        'inlier_frac': round(float(inl.mean()), 3),
        'resid_median_deg': round(float(np.median(resid[inl])), 2) if inl.any() else None,
        'twin_pick_primary_frac': round(pick_prim, 3),
        'lean_azimuth_deg': round(az, 1),
        'lean_concentration_R': round(conc, 3),
        'lean_dev_median_deg': round(float(np.median(dev)), 1),
        'lean_unit_xy': [round(math.cos(math.radians(az)), 3),
                         round(math.sin(math.radians(az)), 3)],
        'per_session': per_s,
    }


def describe(az):
    """Plain-language direction in the export frame."""
    x, y = math.cos(math.radians(az)), math.sin(math.radians(az))
    parts = []
    parts.append('%s (%s)' % ('+x' if x >= 0 else '-x',
                              'back toward the start' if x >= 0 else 'along the race'))
    parts.append('%s (%s)' % ('+y' if y >= 0 else '-y',
                              'toward the 21-29 column row' if y >= 0
                              else 'toward the 12-20 column row'))
    return 'top leans azimuth %.0f deg = (%.2f, %.2f): %s dominant' % (
        az, x, y, parts[0] if abs(x) >= abs(y) else parts[1])


def main():
    R = load_rows()
    mapj = json.load(open(MAP))
    tab = strata_table(R)
    mags = magnitudes(R)

    print('\n=== DIRECTION (twin-resolved in the export frame, >=%.0f px rows)' % NEAR_PX)
    dirs = {}
    for g in range(N_GATES):
        d = direction(R, g, mapj)
        dirs[g] = d
        if d is None:
            continue
        print(' gate %2d  axis_tilt %5.1f  inl %4.1f%%  lean_az %6.1f  R=%.2f  '
              'dev_med %4.1f  n=%d' %
              (g, d['axis_tilt_deg'], 100 * d['inlier_frac'], d['lean_azimuth_deg'],
               d['lean_concentration_R'], d['lean_dev_median_deg'], d['n_rows']))
        if g == 9:
            for s, v in d['per_session'].items():
                print('            %-40s lean_az %6.1f  R=%.2f  n=%d'
                      % (s.split('-', 1)[1], v[0], v[1], v[2]))
    if dirs.get(9):
        print('\n gate 9: ' + describe(dirs[9]['lean_azimuth_deg']))

    json.dump({'rule': {'min_size_px': NEAR_PX, 'min_session_rows': MIN_SESS_ROWS,
                        'pooling': 'per-session medians, then median over sessions'},
               'strata': {str(g): tab[g] for g in tab},
               'accepted': {str(g): mags[g] for g in mags},
               'direction': {str(g): dirs[g] for g in dirs}},
              open(OUT, 'w'), indent=1, default=float)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
