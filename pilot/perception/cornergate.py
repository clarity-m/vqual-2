"""cornergate.py -- is IN-FRAME CORNER COUNT a better trust variable than SIZE? (Bug 1)

Claire's proposal (2026-08-02): for CLOSE gates, size is the wrong variable. What decides
whether a pose is constrained is how much of the gate is actually OBSERVED. Three in-frame
corners of a known 1.5 m square under known K leave only a small discrete solution set;
with two or fewer the quad is mostly extrapolation, which is where the large close-range
errors live.

This measures rather than assumes. Same pose-level referee as residgate2.py (PnP on the
predicted quad vs PnP on the GT quad):

  CATASTROPHE: range error > 30% or position direction off by > 10 deg
  GOOD:        range error < 10% and direction < 3 deg

bucketed by n_corners_in_frame of the GROUND-TRUTH quad (amodal corners -- ones the
labeller placed outside the image -- count as OUT, which is the whole point).

    python3 pilot/perception/cornergate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import gatenet as G                      # noqa: E402
import gatenet_conf as GC                # noqa: E402
import label as L                        # noqa: E402
from residgate import hand_items, q      # noqa: E402
from residgate2 import gather, TRUNK     # noqa: E402

import torch                             # noqa: E402

MARGIN = 0.0        # a corner is IN if it lies inside the image rectangle


def n_in_frame(corners):
    c = np.asarray(corners, float).reshape(4, 2)
    return int(np.sum((c[:, 0] >= -MARGIN) & (c[:, 0] <= L.W + MARGIN) &
                      (c[:, 1] >= -MARGIN) & (c[:, 1] <= L.H + MARGIN)))


def enrich(items, trunk):
    """residgate2.gather rows + n_corners_in_frame of the GT quad.

    gather() drops rows whose GT pose fails, so re-derive the alignment by asking it for
    the same items one batch at a time is overkill -- instead recompute n_in for every
    item and let gather's own 'size' field key the join. Simpler and exact: gather()
    preserves item order for every row it emits EXCEPT the pg-is-None drops, which are
    vanishingly rare; to stay exact we tag each item with its n_in first and read it back
    through the 'clipped' field's neighbour by re-running the same loader ordering.
    """
    rows = gather(items, trunk)
    # gather() iterates the DataLoader with shuffle=False and appends one row per
    # prediction in loader order, skipping only items whose GT pose fails. Recompute the
    # same loader order and drop in lockstep.
    order = []
    from torch.utils.data import DataLoader
    loader = DataLoader(G.GateCrops(items, False), batch_size=64, shuffle=False,
                        num_workers=0)
    with torch.no_grad():
        for _x, _y, _g, i in loader:
            order.extend(int(v) for v in i.numpy())
    if len(order) != len(rows):
        # a GT-pose drop happened; fall back to matching on size, which is item-unique
        # enough for this diagnostic
        by_size = {}
        for k in order:
            by_size.setdefault(round(float(items[k]['size_px']), 3), []).append(k)
        for r in rows:
            cand = by_size.get(round(float(r['size']), 3))
            r['n_in'] = n_in_frame(items[cand[0]]['corners']) if cand else -1
        return rows
    for r, k in zip(rows, order):
        r['n_in'] = n_in_frame(items[k]['corners'])
    return rows


def table(tag, rows):
    n_in = np.array([r['n_in'] for r in rows])
    size = np.array([r['size'] for r in rows])
    resid = np.array([r['resid'] for r in rows])
    rng = np.array([r['rng_err'] for r in rows])
    bad = np.array([r['rng_err'] > 0.30 or r['dir_err'] > 10.0 for r in rows])
    good = np.array([r['rng_err'] < 0.10 and r['dir_err'] < 3.0 for r in rows])
    print(f'\n== {tag}: pose quality by N CORNERS IN FRAME ==')
    print('| n_in | n | size med | resid med/p90 | rng_err med/p90 | good % | '
          'CATASTROPHE % |')
    print('|---:|---:|---:|---:|---:|---:|---:|')
    for k in (0, 1, 2, 3, 4):
        m = n_in == k
        if not m.any():
            continue
        print(f'| {k} | {int(m.sum())} | {q(size[m],50):.0f} | {q(resid[m],50):.3f}/'
              f'{q(resid[m],90):.2f} | {q(rng[m],50):.3f}/{q(rng[m],90):.2f} | '
              f'{100*good[m].mean():.1f} | {100*bad[m].mean():.1f} |')
    # the close-range regime specifically: this is where Bug 1 lives
    print(f'\n-- restricted to size >= 120 px (the close regime) --')
    for k in (0, 1, 2, 3, 4):
        m = (n_in == k) & (size >= 120)
        if m.sum() < 3:
            continue
        print(f'  n_in={k}: n={int(m.sum())}  resid med {q(resid[m],50):.2f}  '
              f'good {100*good[m].mean():.1f}%  catastrophe {100*bad[m].mean():.1f}%')
    return n_in, size, resid, bad, good


def rules(tag, n_in, size, resid, bad, good):
    print(f'\n== {tag}: candidate trust rules ==')

    def ev(name, keep):
        catch = 100 * (~keep)[bad].mean() if bad.any() else float('nan')
        kg = 100 * keep[good].mean() if good.any() else float('nan')
        gbig = good & (size >= 120)
        kb = 100 * keep[gbig].mean() if gbig.any() else float('nan')
        print(f'  {name:44s} catch {catch:5.1f}%  keep good {kg:5.1f}%  '
              f'keep good>=120px {kb:5.1f}% (n={int(gbig.sum())})')

    fin = np.isfinite(resid)
    ev('size-only max(0.3, 0.015*size) [deployed]',
       fin & (resid <= np.maximum(0.3, 0.015 * size)))
    ev('size-only max(6, 0.05*size) [pre-fix producer]',
       fin & (resid <= np.maximum(6.0, 0.05 * size)))
    ev('corners>=3 only', n_in >= 3)
    ev('corners>=2 only', n_in >= 2)
    # the measured cliff sits between 1 and 2 in-frame corners, so pair corners>=2 with
    # a residual gate and look for the combination that adds catch without paying the
    # big-gate coverage that corners>=3 costs
    for floor, k in ((6.0, 0.05), (1.0, 0.05), (1.0, 0.03), (0.3, 0.05), (0.3, 0.03),
                     (0.3, 0.015)):
        ev(f'corners>=2 AND max({floor}, {k}*size)',
           (n_in >= 2) & fin & (resid <= np.maximum(floor, k * size)))
    for k in (0.015, 0.03, 0.05):
        ev(f'corners>=3 AND max(0.3, {k}*size)',
           (n_in >= 3) & fin & (resid <= np.maximum(0.3, k * size)))
    for k in (0.015, 0.03):
        ev(f'corners>=3 AND max(1.0, {k}*size)',
           (n_in >= 3) & fin & (resid <= np.maximum(1.0, k * size)))
    # complementary form: size floor for far gates, corner count for near ones
    for k in (0.03, 0.05):
        ev(f'(size<120 -> max(0.3,{k}s)) AND (size>=120 -> corners>=3)',
           np.where(size < 120,
                    fin & (resid <= np.maximum(0.3, k * size)),
                    (n_in >= 3) & fin & (resid <= np.maximum(1.0, k * size))))


def main():
    trunk = GC.load_trunk(TRUNK, torch.device('cpu'))
    items = G.load_index()
    _, va = G.split_index(items, 'block')
    rows = enrich(va, trunk)
    a = table('NET, VQ1 val', rows)
    rules('VQ1 val', *a)

    hi = hand_items()
    rows_h = enrich(hi, trunk)
    b = table('NET, VQ2 hand (Claire labels: big/clipped/upward)', rows_h)
    rules('VQ2 hand', *b)

    pooled = tuple(np.concatenate([x, y]) for x, y in zip(a, b))
    rules('POOLED', *pooled)


if __name__ == '__main__':
    main()
