"""xfer_gate9roll.py -- does gate 9 carry an IN-PLANE ROLL about its own normal?

READ-ONLY analysis of pilot/perception/gatetilt_rows.json.

CHANNEL. The observed aperture quad's ORIENTATION IN THE IMAGE relative to the image
projection of GRAVITY at the gate's location, compared against the SAME quantity computed
for a synthetic 1.5 m square placed at the resolved pose with ZERO in-plane roll. The
residual is the in-plane roll, modulo the square's 4-fold symmetry (so only (-45, 45] is
meaningful, which is exactly what the 4x-angle circular mean returns).

WHY THIS CHANNEL. A gate that merely LEANS about a horizontal axis projects, viewed
HEAD-ON, with its side edges along the image plumb: the lean is along the line of sight
and foreshortens away. Head-on rows -- which is all gate 9 has -- are therefore the CLEAN
regime for in-plane roll and the blind regime for lean. Lean contamination grows roughly
as tan(lean)*sin(theta), hence stratification on theta and the intercept at theta -> 0.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

REPO = 'C:/Users/USER/Projects/vqual-2'
sys.path.insert(0, REPO + '/pilot')
sys.path.insert(0, REPO + '/pilot/perception')
import label as L                      # noqa: E402
import producer as P                   # noqa: E402

ROWS = REPO + '/pilot/perception/gatetilt_rows.json'
OUT = 'C:/Users/USER/xfer_scratch'
GATE_M = 1.5
rng = np.random.default_rng(7)


def levelling(g_body):
    """body -> levelled-body frame (z DOWN along gravity, yaw anchored to body forward)."""
    z = np.asarray(g_body, float)
    z = z / np.linalg.norm(z)
    f = np.array([1.0, 0.0, 0.0])
    x = f - (f @ z) * z
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x = x / nx
    return np.stack([x, np.cross(z, x), z])


def quad_orientation(quad, plumb_uv):
    q = np.asarray(quad, float)
    a0 = math.atan2(plumb_uv[1], plumb_uv[0])
    acc = 0j
    for i in range(4):
        e = q[(i + 1) % 4] - q[i]
        if np.linalg.norm(e) < 1e-6:
            return None
        acc += np.exp(4j * (math.atan2(e[1], e[0]) - a0))
    return math.degrees(np.angle(acc)) / 4.0


def plumb_dir(P_body, g_hat):
    a = P.project_body(P_body)
    b = P.project_body(P_body + 0.35 * g_hat)
    if a is None or b is None:
        return None
    d = np.array([b[0] - a[0], b[1] - a[1]])
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else None


def synth_quad(P_body, n_body, g_hat, roll_deg=0.0):
    n = n_body / np.linalg.norm(n_body)
    eh = np.cross(g_hat, n)
    if np.linalg.norm(eh) < 1e-6:
        return None
    eh = eh / np.linalg.norm(eh)
    ev = np.cross(n, eh)
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    a1, a2 = c * eh + s * ev, -s * eh + c * ev
    h = GATE_M / 2.0
    uv = [P.project_body(P_body + h * (sx * a1 + sy * a2))
          for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
    if any(x is None for x in uv):
        return None
    return np.array(uv, float)


def wrap90(x):
    return (x + 45.0) % 90.0 - 45.0


def build():
    D = json.load(open(ROWS))
    recs, frame_err = [], []
    for gs, rows in D.items():
        g = int(gs)
        for r in rows:
            q = np.asarray(r['quad'], float)
            if q.shape != (4, 2):
                continue
            gb = np.asarray(r['g_body'], float)
            Rlb = levelling(gb)
            if Rlb is None:
                continue
            g_hat = gb / np.linalg.norm(gb)
            ctr = q.mean(0)
            rhat = P.pixel_ray_body(ctr[0], ctr[1])
            l_lev = np.asarray(r['l_lev'], float)
            frame_err.append(math.degrees(math.acos(
                max(-1.0, min(1.0, float((Rlb @ rhat) @ l_lev))))))
            n_lev = np.asarray(r['n_lev'], float)
            th = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(n_lev @ l_lev))))))
            Pb = rhat * float(r['range'])
            pu = plumb_dir(Pb, g_hat)
            if pu is None:
                continue
            phi = quad_orientation(q, pu)
            if phi is None:
                continue
            preds = {}
            for key, nm in (('n', 'n_lev'), ('t', 'twin')):
                sq = synth_quad(Pb, Rlb.T @ np.asarray(r[nm], float), g_hat, 0.0)
                preds[key] = quad_orientation(sq, pu) if sq is not None else None
            js = []
            for _ in range(24):
                qq = q + rng.normal(0.0, 1.0, q.shape)
                v = quad_orientation(qq, pu)
                if v is not None:
                    js.append(wrap90(v - phi))
            rec = dict(gate=g, sess=r['s'], theta=th, size=float(r['size_px']),
                       rangem=float(r['range']), phi=phi,
                       pred=preds['n'], pred_t=preds['t'],
                       sig=float(np.std(js)) if js else float('nan'))
            rec['dphi'] = wrap90(phi - preds['n']) if preds['n'] is not None else None
            recs.append(rec)
    print('FRAME CHECK  angle(R_lb@ray_centroid, l_lev): median %.2f deg  p90 %.2f  n=%d'
          % (np.median(frame_err), np.percentile(frame_err, 90), len(frame_err)))
    json.dump(recs, open(OUT + '/xfer_gate9roll_rows.json', 'w'))
    print('rows: %d' % len(recs))
    return recs


def sessmed(vals_by_sess):
    """median over per-session medians; returns (value, n_sess, spread)."""
    ms = [float(np.median(v)) for v in vals_by_sess.values() if len(v) >= 3]
    if not ms:
        return float('nan'), 0, float('nan')
    return float(np.median(ms)), len(ms), float(np.std(ms))


def table(recs, key, label, theta_bins, size_min):
    import collections
    print('\n=== %s   (size_px >= %d)' % (label, size_min))
    hdr = 'gate |' + ''.join(' th %2d-%2d      |' % b for b in theta_bins)
    print(hdr)
    groups = [('9', lambda g: g == 9), ('3', lambda g: g == 3),
              ('8', lambda g: g == 8), ('13', lambda g: g == 13),
              ('oth', lambda g: g not in (9,))]
    for name, sel in groups:
        cells = []
        for lo, hi in theta_bins:
            by = collections.defaultdict(list)
            for r in recs:
                if r[key] is None or not sel(r['gate']):
                    continue
                if r['size'] < size_min or not (lo <= r['theta'] < hi):
                    continue
                by[(r['gate'], r['sess'])].append(r[key])
            n = sum(len(v) for v in by.values())
            med, ns, sp = sessmed(by)
            cells.append('%6.1f n%-5d|' % (med, n) if n else '   --  n0    |')
        print('%-4s |' % name + ''.join(cells))


def main():
    recs = build()
    tb = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 90)]
    for smin in (40, 80, 140):
        table(recs, 'phi', 'RAW quad-vs-plumb orientation phi (deg, mod 90)', tb, smin)
        table(recs, 'dphi', 'RESIDUAL dphi = phi_obs - phi_unrolled_model (deg)', tb, smin)
    # noise floor
    import collections
    for gsel, nm in ((9, 'gate 9'), (3, 'gate 3'), (8, 'gate 8'), (13, 'gate 13')):
        v = [r['sig'] for r in recs if r['gate'] == gsel and r['size'] >= 80]
        if v:
            print('MC 1px sigma of phi, %s size>=80: median %.2f deg (n=%d)'
                  % (nm, float(np.median(v)), len(v)))
    plot(recs)


def plot(recs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = {9: 'tab:red', 3: 'tab:blue', 8: 'tab:green', 13: 'tab:orange'}
    for ax, key, ttl in ((axes[0], 'phi', 'raw phi (quad vs plumb)'),
                         (axes[1], 'dphi', 'residual vs unrolled model')):
        for g, c in cols.items():
            xs = [r['theta'] for r in recs if r['gate'] == g and r['size'] >= 80 and r[key] is not None]
            ys = [r[key] for r in recs if r['gate'] == g and r['size'] >= 80 and r[key] is not None]
            if not xs:
                continue
            ax.scatter(xs, ys, s=6, alpha=0.35, color=c, label='gate %d (n=%d)' % (g, len(xs)))
            # binned medians
            bx, by = [], []
            for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 25), (25, 90)):
                v = [y for x, y in zip(xs, ys) if lo <= x < hi]
                if len(v) >= 5:
                    bx.append((lo + min(hi, 40)) / 2.0)
                    by.append(np.median(v))
            ax.plot(bx, by, '-o', color=c, lw=2)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xlabel('obliquity theta (deg)')
        ax.set_ylabel('in-plane angle (deg)')
        ax.set_title(ttl + '  [size_px >= 80]')
        ax.set_ylim(-45, 45)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    p = OUT + '/xfer_gate9roll.png'
    fig.savefig(p, dpi=110)
    print('wrote ' + p)


if __name__ == '__main__':
    main()
