"""Render example frames for the largest label-vs-detection offsets.

Green = projected auto-label quad, orange = matched inner detection quad. Zoomed about
the label centre. Answers by eye which of the two is sitting on the real gate.

    python3 pilot/perception/gatebias_sheet.py [--gate 3] [--n 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detect as D  # noqa: E402

SESSIONS = os.path.join(os.path.dirname(HERE), 'sessions')
GREEN, ORANGE = (90, 230, 90), (60, 170, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', type=int, default=3)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--csv', default=os.path.join(HERE, 'gatebias_offsets.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'gatebias_sheet.png'))
    args = ap.parse_args()

    labels = json.load(open(os.path.join(HERE, 'autolabels_vq1.json')))
    rows = [r for r in csv.DictReader(open(args.csv)) if int(r['gate']) == args.gate]
    for r in rows:
        r['off'] = float(np.hypot(float(r['dx']), float(r['dy'])))
    rows.sort(key=lambda r: -r['off'])
    # spread the picks: take worst, then skip near-in-time duplicates
    picks, seen = [], []
    for r in rows:
        fr = int(r['key'].split('/')[1].split('.')[0])
        if any(abs(fr - s) < 5 for s in seen):
            continue
        picks.append(r)
        seen.append(fr)
        if len(picks) >= args.n:
            break

    side = 360
    tiles = []
    for r in picks:
        sess, fname = r['key'].split('/', 1)
        img = cv2.imread(os.path.join(SESSIONS, sess, 'frames', fname))
        if img is None:
            continue
        d = labels[r['key']][int(r['idx'])]
        uv = np.asarray(d['corners'], np.float32)
        c = uv.mean(axis=0)
        dets = D.detections(img)
        # re-find the same match
        best, bd = None, 1e18
        for dd in dets:
            if dd['source'] != 'inner':
                continue
            dist = float(np.linalg.norm(np.asarray(dd['centre']) - c))
            ratio = dd['size_px'] / max(float(d['size_px']), 1e-6)
            if dist > float(d['size_px']) or ratio > 2.0 or ratio < 0.5:
                continue
            if dist < bd:
                best, bd = dd, dist
        S = max(float(d['size_px']) * 4.0, 90.0)
        a = side / S
        M = np.array([[a, 0, side / 2 - c[0] * a], [0, a, side / 2 - c[1] * a]], np.float32)
        t = cv2.warpAffine(img, M, (side, side), flags=cv2.INTER_LINEAR)
        tf = lambda q: (np.asarray(q, np.float32) * a + M[:, 2]).astype(np.int32)  # noqa
        cv2.polylines(t, [tf(uv).reshape(-1, 1, 2)], True, GREEN, 2)
        if best is not None:
            cv2.polylines(t, [tf(best['quad']).reshape(-1, 1, 2)], True, ORANGE, 2)
        cv2.putText(t, f'{sess[-6:]}/{fname} g{args.gate} {float(d["size_px"]):.0f}px '
                       f'off {r["off"]:.1f}px', (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tiles.append(t)

    cols = 4
    nr = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((nr * side, cols * side, 3), np.uint8)
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sheet[rr * side:(rr + 1) * side, cc * side:(cc + 1) * side] = t
    cv2.imwrite(args.out, sheet)
    print(f'wrote {args.out} ({len(tiles)} tiles)')


if __name__ == '__main__':
    main()
