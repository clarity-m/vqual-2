"""gatetilt_obliquity.py -- settle the gate-9 tilt MAGNITUDE by stratifying on VIEW
OBLIQUITY, which is the variable both estimators' blind spots depend on.

THE PROBLEM WITH SIZE/RANGE AS THE STRATIFIER. Two channels disagree on gate 9:

  (a) PnP normal elevation      : ~24 deg on near/large rows, rising with proximity
  (b) edge lean vs gravity      : ~4 deg on the same rows (PnP-free)

and BOTH channels' failure modes are functions of the same geometric quantity -- the
angle between the line of sight and the gate's plane normal:

  theta = 0    HEAD-ON.  IPPE is near-degenerate here, so (a) is worst conditioned and
               biased upward. And (b) is BLIND here: a gate leaning toward or away from
               the camera still projects its side edges as vertical lines.
  theta = 90   EDGE-ON.  (a) is well conditioned; (b) sees the lean at full size. The
               gate8.png screenshot that started this is an edge-on view.

Near, large detections on this course are overwhelmingly HEAD-ON APPROACHES, so
stratifying by size or range selects rows where BOTH channels are at their worst. That is
what the earlier passes did -- in opposite directions.

THE MODEL THAT MAKES (b) QUANTITATIVE INSTEAD OF MERELY BLIND. Let phi be the gate's true
lean and theta the obliquity. edge_tilt returns psi = asin(N_hat . g_hat) where N is the
normal of the plane through the camera centre and the edge's image line. Since N is
perpendicular to the edge direction u, and z_hat = cos(phi) u + sin(phi) d_hat with d_hat
the horizontal lean direction,

    sin(psi) = sin(phi) * (N_hat . d_hat) = sin(phi) * sin(theta)

because N_hat . d_hat is the component of the viewing ray along the in-plane horizontal,
which is sin(theta). So the edge channel does not merely go blind head-on -- it makes a
PREDICTION there, and a small non-zero reading at small theta is evidence FOR a large
phi, not against it. Inverting per row:

    phi_hat = asin( sin(psi) / sin(theta) )

which is exactly the correction the earlier passes were missing.

WHAT IS TESTED (all four are reported for gate 9 and for control gates):
  1. tilt vs obliquity bucket, both channels;
  2. does (a) FALL toward (b) as theta grows (would mean (a) is a conditioning artefact)?
  3. does (b) RISE with theta as sin(theta) demands (would mean the tilt is real)?
  4. the inverted phi_hat from (b) -- flat in theta if the model is right.

Controls are the discipline: a vertical gate must return phi_hat ~ 0 at every theta.

    python3 pilot/perception/gatetilt_obliquity.py
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
import gatetilt_strat as GS       # noqa: E402

MIN_PX = 40.0            # quad-quality floor only; NOT the stratifier
OUT = os.path.join(HERE, 'gatetilt_obliquity.json')


def gate_normal_az(mapj, g):
    """The map's independently measured plane normal azimuth (mod 180), export frame."""
    a = mapj['layout_directions_2026_08_02']['gate_plane_angles_deg'].get(str(g))
    return None if a is None else float(a['normal_az_sketch_frame_mod180'])


def rows_with_theta(R, g, mapj, min_px=MIN_PX):
    """Attach obliquity theta (0 = head-on, 90 = edge-on) to every usable row."""
    rot, refl, offs = GS.sketch_frame(mapj)
    naz = gate_normal_az(mapj, g)
    if naz is None:
        return []
    out = []
    for r in R.get(g, []):
        if r['size_px'] < min_px:
            continue
        e = GT.edge_tilt(r)
        if e is None:
            continue
        b = float(offs.get(r['s'], 0.0))
        l = GS.to_sketch(r['l_lev'], r['psi'], b, rot, refl)
        vaz = math.degrees(math.atan2(l[1], l[0]))
        theta = abs((vaz - naz + 90.0) % 180.0 - 90.0)      # 0..90
        out.append({'theta': theta, 'pnp': GS.pnp_tilt(r), 'edge': e['lean_abs'],
                    's': r['s'], 'size': r['size_px'], 'range': r['range']})
    return out


def phi_from_edge(psi_deg, theta_deg):
    """Invert sin(psi) = sin(phi) sin(theta). None where theta is too small to invert."""
    st = math.sin(math.radians(theta_deg))
    if st < math.sin(math.radians(12.0)):
        return None
    v = math.sin(math.radians(psi_deg)) / st
    return math.degrees(math.asin(min(1.0, v)))


BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 90)]


def edge_slope_fit(rows, nboot=400, seed=0):
    """Robust fit of sin(psi) = a*sin(theta) + b over ALL rows -> phi = asin(a).

    Per-row inversion phi = asin(sin psi / sin theta) divides by sin(theta) and so
    explodes exactly where the rows are (theta < 20 deg): a 3 deg noise reading at
    theta = 13 deg inverts to 13 deg of fictitious tilt. A regression uses the same
    rows without that amplification, and the intercept b ABSORBS the estimator's noise
    floor instead of letting it masquerade as tilt. A vertical gate must return a ~ 0
    whatever its noise floor is.

    Theil-Sen slope (median of pairwise slopes) for robustness against the tail of bad
    quads; bootstrap over rows for the interval.
    """
    if len(rows) < 25:
        return None
    x = np.array([math.sin(math.radians(r['theta'])) for r in rows])
    y = np.array([math.sin(math.radians(r['edge'])) for r in rows])
    if x.max() - x.min() < 0.05:
        return None

    def theil_sen(xx, yy):
        n = len(xx)
        idx = np.random.default_rng(seed).integers(0, n, size=(min(20000, n * 8), 2))
        dx = xx[idx[:, 1]] - xx[idx[:, 0]]
        dy = yy[idx[:, 1]] - yy[idx[:, 0]]
        ok = np.abs(dx) > 0.02
        if ok.sum() < 20:
            return None
        return float(np.median(dy[ok] / dx[ok]))

    a = theil_sen(x, y)
    if a is None:
        return None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(nboot):
        k = rng.integers(0, len(x), len(x))
        v = theil_sen(x[k], y[k])
        if v is not None:
            boot.append(v)
    boot = np.array(boot)
    b = float(np.median(y - a * x))

    def to_phi(s):
        return math.degrees(math.asin(min(1.0, max(0.0, s))))
    return {'slope': round(a, 4), 'intercept': round(b, 4),
            'phi_deg': round(to_phi(a), 1),
            'phi_lo_deg': round(to_phi(float(np.percentile(boot, 5))), 1),
            'phi_hi_deg': round(to_phi(float(np.percentile(boot, 95))), 1),
            'noise_floor_deg': round(math.degrees(math.asin(min(1.0, max(0.0, b)))), 1),
            'n': len(rows), 'theta_span_deg': round(float(
                max(r['theta'] for r in rows) - min(r['theta'] for r in rows)), 1)}


def report(R, mapj, gates):
    res = {}
    for g in gates:
        rows = rows_with_theta(R, g, mapj)
        if not rows:
            print('gate %2d: no measured plane azimuth -- obliquity undefined' % g)
            continue
        th = np.array([r['theta'] for r in rows])
        print('\ngate %2d   n=%d usable rows (>=%.0f px)   theta: median %.1f  '
              'p90 %.1f  max %.1f  |  rows above 30 deg: %d'
              % (g, len(rows), MIN_PX, np.median(th), np.percentile(th, 90), th.max(),
                 int((th >= 30).sum())))
        print('   %-12s %5s %8s %8s %10s   %s'
              % ('theta bucket', 'n', 'pnp', 'edge', 'phi(edge)', 'sessions'))
        buck = {}
        for lo, hi in BUCKETS:
            sel = [r for r in rows if lo <= r['theta'] < hi]
            if not sel:
                continue
            pnp = float(np.median([r['pnp'] for r in sel]))
            edg = float(np.median([r['edge'] for r in sel]))
            ph = [phi_from_edge(r['edge'], r['theta']) for r in sel]
            ph = [x for x in ph if x is not None]
            phm = float(np.median(ph)) if len(ph) >= 5 else None
            ns = len({r['s'] for r in sel})
            print('   %-12s %5d %8.1f %8.1f %10s   %d'
                  % ('%d-%d' % (lo, hi), len(sel), pnp, edg,
                     '%.1f (n%d)' % (phm, len(ph)) if phm is not None else '-', ns))
            buck['%d-%d' % (lo, hi)] = {'n': len(sel), 'pnp_deg': round(pnp, 2),
                                        'edge_deg': round(edg, 2),
                                        'phi_from_edge_deg': (round(phm, 2) if phm
                                                              is not None else None),
                                        'n_sessions': ns}
        # pooled inversion over every row with enough obliquity to invert
        ph = [(phi_from_edge(r['edge'], r['theta']), r['s']) for r in rows]
        ph = [(x, s) for x, s in ph if x is not None]
        by_s = collections.defaultdict(list)
        for x, s in ph:
            by_s[s].append(x)
        pooled = GS.sess_pooled(by_s, min_rows=5)
        if pooled:
            print('   PnP-FREE phi over all theta>=12 deg rows: %.1f deg '
                  '(sigma %.1f, %d sessions, %d rows)'
                  % (pooled['deg'], pooled['sigma_deg'], pooled['n_sessions'],
                     pooled['n_rows']))
            for d in pooled['per_session']:
                print('        %-40s %5.1f deg (n=%d, iqr %.1f)'
                      % (d['session'].split('-', 1)[1], d['median_deg'], d['n'],
                         d['iqr_deg']))
        fit = edge_slope_fit(rows)
        if fit:
            print('   PnP-FREE slope fit sin(psi)=a*sin(theta)+b over all %d rows '
                  '(theta span %.1f deg): phi = %.1f deg  [90%% CI %.1f-%.1f], '
                  'noise floor %.1f deg'
                  % (fit['n'], fit['theta_span_deg'], fit['phi_deg'], fit['phi_lo_deg'],
                     fit['phi_hi_deg'], fit['noise_floor_deg']))
        res[g] = {'edge_slope_fit': fit,
                  'n_rows': len(rows), 'theta_median': round(float(np.median(th)), 1),
                  'theta_p90': round(float(np.percentile(th, 90)), 1),
                  'n_theta_ge_30': int((th >= 30).sum()),
                  'buckets': buck,
                  'phi_pnp_free': (None if not pooled else
                                   {'deg': round(pooled['deg'], 2),
                                    'sigma_deg': round(pooled['sigma_deg'], 2),
                                    'n_sessions': pooled['n_sessions'],
                                    'n_rows': pooled['n_rows'],
                                    'per_session': pooled['per_session']})}
    return res


def main():
    R = GS.load_rows()
    mapj = json.load(open(GS.MAP))
    print('=== obliquity theta = angle(line of sight, gate plane normal). '
          '0 = head-on, 90 = edge-on.')
    print('=== pnp  = |elevation of the PnP normal| (biased UP where theta is small)')
    print('=== edge = image edge line vs gravity  (goes to zero as theta -> 0 even for '
          'a real tilt)')
    print('=== phi(edge) = the PnP-FREE tilt, edge reading corrected by '
          'sin(psi)=sin(phi)sin(theta)')
    res = report(R, mapj, list(range(GS.N_GATES)))
    json.dump({'model': 'sin(psi) = sin(phi) * sin(theta)', 'min_size_px': MIN_PX,
               'per_gate': {str(k): v for k, v in res.items()}},
              open(OUT, 'w'), indent=1)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
