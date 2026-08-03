"""Contact sheets of what the detector actually returned on the VQ2 lap, colour-coded by
which filter would reject it. Diagnostic only -- nothing here feeds the map.

Exists because this project's four measurement failures were all caught by LOOKING at the
frames, never by staring at the aggregate. `mapvq2.py`'s filter cascade throws away 85% of
detections and the aggregate cannot say whether it is throwing away decoration or gates.

    python3 pilot/perception/vq2look.py <session> --frames 103500,104000 --out sheet.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vq2cache  # noqa: E402

GREEN = (0, 230, 0)      # survives every filter
YELLOW = (0, 220, 255)   # rejected only by quality()
RED = (0, 80, 255)       # rejected as decoration
BLUE = (255, 160, 0)     # outer-boundary fallback


def draw(session, fids, out, scale=2.0):
    det = vq2cache.load(session)
    tiles = []
    for fid in fids:
        name = '%08d.jpg' % fid
        img = cv2.imread(os.path.join(session, 'frames', name))
        if img is None:
            print('missing', name)
            continue
        for d in det.get(name, []):
            if d['deco']:
                col, tag = RED, 'deco'
            elif d['source'] == 'outer':
                col, tag = BLUE, 'outer'
            elif d['bad']:
                col, tag = YELLOW, ','.join(b.split()[0] for b in d['bad'])
            else:
                col, tag = GREEN, 'ok'
            q = np.asarray(d['quad']).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [q], True, col, 1)
            c = d['centre'].astype(int)
            cv2.putText(img, '%.0fm %s' % (d['range_m'], tag), (c[0] - 20, c[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
        cv2.putText(img, name, (5, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(cv2.resize(img, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_NEAREST))
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    cv2.imwrite(out, sheet)
    print('wrote', out, sheet.shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', default='sheet.png')
    ap.add_argument('--scale', type=float, default=2.0)
    a = ap.parse_args()
    draw(a.session, [int(x) for x in a.frames.split(',')], a.out, a.scale)
