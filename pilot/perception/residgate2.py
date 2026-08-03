"""residgate2.py -- Bug 1, second pass: judge the residual gate by the POSE it protects.

residgate.py's absolute >10 px error criterion misclassifies big gates: 10 px on a
500 px gate is 2% -- a fine pose. The gate exists to keep BAD POSES out of the tracks,
so the referee here is pose-level: PnP on the net quad vs PnP on the GT quad.

  CATASTROPHE: |range - range_gt| / range_gt > 0.30, or the position direction off
               by > 10 deg (a wrong gate / garbage quad).
  GOOD:        range error < 10% and direction < 3 deg.

Rule under test: accept iff corrected_resid <= max(ABS, k * size_px).

    python3 pilot/perception/residgate2.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import gatenet as G                    # noqa: E402
import gatenet_conf as GC              # noqa: E402
import detect as D                     # noqa: E402
from producer import pnp_pose          # noqa: E402
from residgate import hand_items, BANDS, q          # noqa: E402

import torch                           # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

TRUNK = os.path.join(HERE, 'gatenet_runs', 'colab-v3-rot', 'best.pt')


def pose_of(quad):
    return pnp_pose(np.ascontiguousarray(D.order_quad(np.asarray(quad, np.float64))))


def gather(items, trunk):
    loader = DataLoader(G.GateCrops(items, False), batch_size=64, shuffle=False,
                        num_workers=0)
    preds, geos, idxs = [], [], []
    with torch.no_grad():
        for x, y, g, i in loader:
            preds.append(trunk(x).numpy())
            geos.append(g.numpy())
            idxs.append(i.numpy())
    preds, geos = np.concatenate(preds), np.concatenate(geos)
    idxs = np.concatenate(idxs)
    rows = []
    for j in range(len(preds)):
        cx, cy, S = geos[j]
        quad = G.from_norm(preds[j], cx, cy, S).astype(np.float64)
        d = items[idxs[j]]
        gt = np.asarray(d['corners'], np.float64)
        pg = pose_of(gt)
        pq = pose_of(quad)
        if pg is None:
            continue
        if pq is None:
            rows.append({'size': d['size_px'], 'resid': float('inf'),
                         'rng_err': float('inf'), 'dir_err': 180.0,
                         'clipped': d['clipped']})
            continue
        rng_err = abs(pq['range'] - pg['range']) / max(pg['range'], 1e-6)
        u1 = pq['pos'] / max(np.linalg.norm(pq['pos']), 1e-9)
        u2 = pg['pos'] / max(np.linalg.norm(pg['pos']), 1e-9)
        dir_err = math.degrees(math.acos(float(np.clip(u1 @ u2, -1, 1))))
        rows.append({'size': d['size_px'], 'resid': pq['rms'], 'rng_err': rng_err,
                     'dir_err': dir_err, 'clipped': d['clipped']})
    return rows


def classify(rows):
    size = np.array([r['size'] for r in rows])
    resid = np.array([r['resid'] for r in rows])
    bad = np.array([r['rng_err'] > 0.30 or r['dir_err'] > 10.0 for r in rows])
    good = np.array([r['rng_err'] < 0.10 and r['dir_err'] < 3.0 for r in rows])
    return size, resid, bad, good


def table(tag, size, resid, bad, good):
    print(f'\n== {tag}: pose-level classes by size bucket ==')
    print('| bucket | n | good | resid med/p90 good | bad | resid med/p10 bad |')
    for lo, hi in BANDS:
        m = (size >= lo) & (size < hi)
        g, b = m & good, m & bad
        name = f'{lo:.0f}-{hi:.0f}' if hi < 1e8 else f'>={lo:.0f}'
        print(f'| {name} | {m.sum()} | {g.sum()} | {q(resid[g],50):.3f}/'
              f'{q(resid[g],90):.3f} | {b.sum()} | {q(resid[b],50):.3f}/'
              f'{q(resid[b],10):.3f} |')


def sweep(tag, size, resid, bad, good):
    print(f'\n== rule sweep, {tag} (pose-level referee) ==')

    def ev(name, thr):
        rej = ~np.isfinite(resid) | (resid > thr)
        catch = 100 * rej[bad].mean() if bad.any() else float('nan')
        keep = 100 * (~rej)[good].mean() if good.any() else float('nan')
        gbig = good & (size >= 120)
        keep_big = 100 * (~rej)[gbig].mean() if gbig.any() else float('nan')
        print(f'  {name:30s} catch {catch:5.1f}%  keep good {keep:5.1f}%  '
              f'keep good>=120 {keep_big:5.1f}% (n={int(gbig.sum())})')

    ev('0.3 abs (HUD today)', 0.3 + 0 * size)
    ev('max(6, .05s) (producer today)', np.maximum(6.0, 0.05 * size))
    for k in (0.004, 0.006, 0.008, 0.010, 0.015):
        ev(f'max(0.3, {k}*size)', np.maximum(0.3, k * size))
    for k in (0.006, 0.008, 0.010):
        ev(f'max(0.5, {k}*size)', np.maximum(0.5, k * size))


def main():
    trunk = GC.load_trunk(TRUNK, torch.device('cpu'))
    items = G.load_index()
    _, va = G.split_index(items, 'block')
    rows = gather(va, trunk)
    size, resid, bad, good = classify(rows)
    print(f'VQ1 val: {len(rows)} instances, good {good.sum()} bad {bad.sum()}')
    table('NET, VQ1 val', size, resid, bad, good)
    sweep('NET VQ1 val', size, resid, bad, good)

    hi = hand_items()
    rows_h = gather(hi, trunk)
    sh, rh, bh, gh = classify(rows_h)
    print(f'\nVQ2 hand: {len(rows_h)} instances, good {gh.sum()} bad {bh.sum()}')
    table('NET, VQ2 hand', sh, rh, bh, gh)
    sweep('NET VQ2 hand', sh, rh, bh, gh)

    # pooled: the deployed rule must hold on both
    size2 = np.concatenate([size, sh])
    resid2 = np.concatenate([resid, rh])
    bad2 = np.concatenate([bad, bh])
    good2 = np.concatenate([good, gh])
    sweep('POOLED', size2, resid2, bad2, good2)


if __name__ == '__main__':
    main()
