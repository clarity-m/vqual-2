"""Per-session histogram of the contour rows for contested pairs (7-8, 8-9, 6-7, 9-10).

Which SESSION contributes which mode? The bimodality guard can only referee what it can
separate; this shows the raw material per flight.
"""
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mapvq2 as M           # noqa: E402
from mapedges import TARGETING  # noqa: E402

PAIRS = [(7, 8), (8, 9), (6, 7), (2, 3), (9, 10), (12, 13), (13, 14)]


def main():
    sessions = [M.PRIMARY, M.SECOND, M.PAUSING, TARGETING]
    pooled = M.run_many(sessions, verbose=False)
    rows = pooled['rows']
    for p in PAIRS:
        key = (('g', p[0]), ('g', p[1]))
        sel = [r for r in rows if r['pair'] == key]
        print(f'\npair {p[0]}-{p[1]}: {len(sel)} contour rows')
        bysess = collections.defaultdict(list)
        for r in sel:
            bysess[r['session'][-12:]].append(r)
        for s, rs in sorted(bysess.items()):
            ds = np.array([r['d'] for r in rs])
            mk = sum(bool(r.get('mk')) for r in rs)
            hist, edges = np.histogram(ds, bins=np.arange(0, 45, 3))
            hs = ' '.join(f'{int(e)}m:{c}' for e, c in zip(edges, hist) if c)
            print(f'  {s:>14s} n={len(ds):3d} mk={mk:2d} med={np.median(ds):6.2f} '
                  f'p10-p90 {np.percentile(ds,10):5.1f}-{np.percentile(ds,90):5.1f}  [{hs}]')


if __name__ == '__main__':
    main()
