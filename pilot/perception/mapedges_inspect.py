"""Render every gatenet pair row of ONE pair, full frame, for identity judgement.

    python3 pilot/perception/mapedges_inspect.py --pair 1-5

Draws, on each full frame that produced a row for the pair: all accepted gatenet quads
in that frame (green, labelled g<k> r<range>), the seed contour quads (orange, thin).
The partner side of a row may be a clean contour detection, which is not in the
accepted dump -- the frame itself shows where it is. Output: vercheck/inspect_<pair>.png
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--max', type=int, default=8)
    a = ap.parse_args()
    ga, gb = (int(x) for x in a.pair.split('-'))

    rows = json.load(open(os.path.join(HERE, 'mapedges_rows.json')))
    acc = json.load(open(os.path.join(HERE, 'mapedges_accepted.json')))
    acc_by = collections.defaultdict(list)
    for m in acc:
        acc_by[(m['session'], m['i'])].append(m)

    sel = [r for r in rows if r['pair'][0][1] == ga and r['pair'][1][1] == gb]
    print(f'{len(sel)} rows for pair {ga}-{gb}')
    tiles = []
    for r in sel[:a.max]:
        ms = acc_by.get((r['session'], r['i']), [])
        if not ms:
            continue
        img = cv2.imread(ms[0]['path'], cv2.IMREAD_COLOR)
        if img is None:
            continue
        pad = 80
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                 value=(20, 20, 20))
        for m in ms:
            sq = np.asarray(m['seed_quad'], np.float32).reshape(-1, 2) + pad
            q = np.asarray(m['quad'], np.float32) + pad
            cv2.polylines(big, [sq.astype(np.int32)], True, (60, 170, 255), 1)
            cv2.polylines(big, [q.astype(np.int32)], True, (90, 230, 90), 2)
            c = q.mean(0)
            cv2.putText(big, 'g%d %.0fm' % (m['g'], m['range_m']),
                        (int(c[0]) - 20, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 230, 90), 1, cv2.LINE_AA)
        cv2.putText(big, '%s f%d  d=%.1fm ra=%.1f rb=%.1f%s' % (
            r['session'][-14:], r['fid'], r['d'], r['ra'], r['rb'],
            ' MARKER' if r['mk'] else ''),
            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(big)
        print('  %s f%d d=%.2f ra=%.1f rb=%.1f mk=%s' %
              (r['session'], r['fid'], r['d'], r['ra'], r['rb'], r['mk']))
    if not tiles:
        print('nothing to render')
        return
    sheet = np.vstack(tiles)
    out = os.path.join(HERE, 'vercheck', f'inspect_{ga}-{gb}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, sheet)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
