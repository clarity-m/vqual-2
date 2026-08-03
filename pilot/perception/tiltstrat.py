"""tiltstrat.py -- stratify the PnP-FREE edge tilt by measurement quality.

gatetilt.py's pooled edge-tilt medians said every gate is plumb, gate 9 included (3.4 deg).
The coordinator reports the opposite from a PnP-normal channel once stratified, and calls
the pooled reading a far-view artefact. Both cannot be right, so re-cut MY channel -- which
shares no failure mode with theirs (image edge line + gravity only, no PnP, no compass) --
by apparent size and by range.

If gate 9 is genuinely tilted, its edge lean must RISE as the view improves and stay put
across sessions; if the 3.4 deg pooled median was a far-view artefact, the far bucket is
what produced it.

    python3 pilot/perception/tiltstrat.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gatetilt import edge_tilt, ROWS      # noqa: E402

SIZE_BANDS = ((0, 25), (25, 60), (60, 120), (120, 1e9))
RANGE_BANDS = ((0, 10), (10, 15), (15, 25), (25, 1e9))


def med(v):
    return float(np.median(v)) if len(v) else float('nan')


def main():
    rows = {int(k): v for k, v in json.load(open(ROWS)).items()}
    feats = {}
    for g, rr in rows.items():
        out = []
        for r in rr:
            e = edge_tilt(r)
            if e is None:
                continue
            out.append((e['lean_abs'], e['lean'], float(r['size_px']),
                        float(r['range']), r['s']))
        feats[g] = out

    print('=== PnP-free edge lean, by APPARENT SIZE (deg; n in parens)')
    hdr = '  '.join('%-14s' % ('%d-%d px' % (lo, hi) if hi < 1e8 else '>=%d px' % lo)
                    for lo, hi in SIZE_BANDS)
    print('%-5s %s' % ('gate', hdr))
    for g in sorted(feats):
        cells = []
        for lo, hi in SIZE_BANDS:
            v = [f[0] for f in feats[g] if lo <= f[2] < hi]
            cells.append('%-14s' % ('%.1f (n%d)' % (med(v), len(v)) if len(v) >= 5
                                    else '-  (n%d)' % len(v)))
        print('%-5d %s' % (g, '  '.join(cells)))

    print('\n=== PnP-free edge lean, by RANGE (deg)')
    hdr = '  '.join('%-14s' % ('%d-%d m' % (lo, hi) if hi < 1e8 else '>=%d m' % lo)
                    for lo, hi in RANGE_BANDS)
    print('%-5s %s' % ('gate', hdr))
    for g in sorted(feats):
        cells = []
        for lo, hi in RANGE_BANDS:
            v = [f[0] for f in feats[g] if lo <= f[3] < hi]
            cells.append('%-14s' % ('%.1f (n%d)' % (med(v), len(v)) if len(v) >= 5
                                    else '-  (n%d)' % len(v)))
        print('%-5d %s' % (g, '  '.join(cells)))

    print('\n=== BEST-VIEW bucket (size >= 60 px): per-session medians + signed lean')
    print('%-5s %6s %8s %8s %8s   %s' %
          ('gate', 'n', 'lean_med', 'IQR', 'sgn_med', 'per-session'))
    summary = {}
    for g in sorted(feats):
        sel = [f for f in feats[g] if f[2] >= 60.0]
        if len(sel) < 8:
            print('%-5d %6d  (too few best-view rows)' % (g, len(sel)))
            continue
        v = np.array([f[0] for f in sel])
        s = np.array([f[1] for f in sel])
        by = collections.defaultdict(list)
        for f in sel:
            by[f[4]].append(f[0])
        per = ' '.join('%s:%.1f(n%d)' % (k.split('-')[1], med(x), len(x))
                       for k, x in sorted(by.items()) if len(x) >= 5)
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        summary[g] = {'n': len(v), 'lean_median_deg': round(med(v), 2),
                      'iqr_deg': round(iqr, 2),
                      'signed_median_deg': round(med(s), 2),
                      'per_session': {k.split('-')[1]: round(med(x), 2)
                                      for k, x in sorted(by.items()) if len(x) >= 5}}
        print('%-5d %6d %8.2f %8.2f %8.2f   %s' % (g, len(v), med(v), iqr, med(s), per))

    json.dump(summary, open(os.path.join(HERE, 'tiltstrat_bestview.json'), 'w'), indent=1)
    print('\nwrote tiltstrat_bestview.json')


if __name__ == '__main__':
    main()
