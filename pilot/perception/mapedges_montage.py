"""Render accepted gatenet edge measurements onto real frames -- the LOOK step.

Every real bug in this map was found by rendering identities/quads on frames and
looking, never by aggregates. So before any minted edge is believed, this draws:

    green   = gatenet quad (the accepted, PnP-gated measurement)
    orange  = the contour detection that seeded the crop (what clean() refused)

one tile per accepted measurement, prioritising the target linking edges, and writes
pilot/perception/vercheck/gatenet_edges_montage.png.

Reads mapedges_accepted.json (written by mapedges.py).

    python3 pilot/perception/mapedges_montage.py
"""

from __future__ import annotations

import collections
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.path.join(HERE, 'mapedges_accepted.json')
OUT = os.path.join(HERE, 'vercheck', 'gatenet_edges_montage.png')

NET_COL = (90, 230, 90)      # green
SEED_COL = (60, 170, 255)    # orange

TARGET_GATES = {2, 3, 6, 7, 8, 9, 10, 12, 13, 14}


def main():
    acc = json.load(open(ACC))
    print(f'{len(acc)} accepted measurements')
    bygate = collections.defaultdict(list)
    for m in acc:
        bygate[m['g']].append(m)

    tiles = []
    for g in sorted(bygate, key=lambda g: (g not in TARGET_GATES, g)):
        ms = sorted(bygate[g], key=lambda m: m['resid'])
        # spread the samples over the gate's frames rather than taking one burst
        sel = [ms[int(k)] for k in np.linspace(0, len(ms) - 1, min(4, len(ms)))]
        seen = set()
        for m in sel:
            if m['fid'] in seen:
                continue
            seen.add(m['fid'])
            img = cv2.imread(m['path'], cv2.IMREAD_COLOR)
            if img is None:
                continue
            pad = 70
            big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                     value=(20, 20, 20))
            q = np.asarray(m['quad'], np.float32) + pad
            sq = np.asarray(m['seed_quad'], np.float32).reshape(-1, 2) + pad
            cv2.polylines(big, [sq.astype(np.int32)], True, SEED_COL, 1, cv2.LINE_AA)
            cv2.polylines(big, [q.astype(np.int32)], True, NET_COL, 2, cv2.LINE_AA)
            c = q.mean(0)
            S = max(m['size_px'] * 2.2, 170.0)
            x0 = int(np.clip(c[0] - S / 2, 0, big.shape[1] - S))
            y0 = int(np.clip(c[1] - S / 2, 0, big.shape[0] - S))
            tile = big[y0:y0 + int(S), x0:x0 + int(S)]
            tile = cv2.resize(tile, (250, 250), interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, 'g%d f%d r%.1fm res%.2f %s%s' % (
                m['g'], m['fid'], m['range_m'], m['resid'],
                m['seed_source'], '' if m.get('cont_checked') else ' UNCHK'),
                (3, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (249, 249), (70, 70, 70), 1)
            tiles.append(tile)

    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 250, cols * 250, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 250:(r + 1) * 250, c * 250:(c + 1) * 250] = t
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, sheet)
    print(f'wrote {OUT} ({len(tiles)} tiles; green=gatenet, orange=seed contour det)')


if __name__ == '__main__':
    main()
