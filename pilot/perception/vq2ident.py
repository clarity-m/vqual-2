"""Look at the identities mapvq2.py assigned, on the real frames.

The whole map rests on one claim -- "this detection is gate k" -- and that claim is made by
code that has never been looked at. This project's four measurement failures were every one
of them caught by looking at frames, so this is not optional polish.

Two views:
  --frames   full frames with every identified detection labelled by race index.
  --gate K   a montage of ONE gate across time. If a gate's node has absorbed a fragment of
             a different gate, the montage shows two different objects under one number,
             which no aggregate statistic will tell you.

    python3 pilot/perception/vq2ident.py --frames 8 --out ident.png
    python3 pilot/perception/vq2ident.py --gate 13 --out g13.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mapvq2 as M  # noqa: E402

COLS = [(60, 180, 255), (0, 255, 255), (0, 255, 120), (255, 200, 0), (255, 120, 255),
        (120, 120, 255), (255, 255, 255), (0, 160, 255)]


def identified(session):
    # Go through run() rather than rebuilding the tracker here, so that what is drawn is
    # exactly what the map is built from -- including the backward extensions, which are the
    # part most likely to be wrong and therefore the part most worth looking at.
    res = M.run(session, verbose=False)
    S, good, ident = res['S'], res['good'], res['ident']
    byframe = {}
    for tid, obs in good.items():
        if tid not in ident:
            continue
        for i, d in obs:
            if M.clean(d):
                byframe.setdefault(i, []).append((ident[tid], d))
    return S, byframe


def sheet_frames(session, n, out):
    S, byframe = identified(session)
    keys = sorted(k for k, v in byframe.items() if len(v) >= 2)
    picks = [keys[int(round(x))] for x in np.linspace(0, len(keys) - 1, n)]
    tiles = []
    for i in picks:
        img = cv2.imread(os.path.join(session, 'frames', S.frames[i]['file']))
        if img is None:
            continue
        for g, d in sorted(byframe[i], key=lambda x: x[0]):
            col = COLS[g % len(COLS)]
            q = np.asarray(d['quad']).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [q], True, col, 2)
            c = d['centre'].astype(int)
            cv2.putText(img, '%d  %.0fm' % (g, d['range_m']), (c[0] - 14, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(img, '%s  active=%d' % (S.frames[i]['file'], S.frame_gate[i]),
                    (5, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(cv2.resize(img, None, fx=1.6, fy=1.6))
    grid(tiles, 2, out)


def sheet_gate(session, g, out, n=12):
    S, byframe = identified(session)
    hits = [(i, d) for i, v in byframe.items() for gg, d in v if gg == g]
    hits.sort()
    if not hits:
        print('no identified detections for gate', g)
        return
    picks = [hits[int(round(x))] for x in np.linspace(0, len(hits) - 1, min(n, len(hits)))]
    tiles = []
    for i, d in picks:
        img = cv2.imread(os.path.join(session, 'frames', S.frames[i]['file']))
        if img is None:
            continue
        c = d['centre'].astype(int)
        s = int(max(45, d['size_px'] * 1.3))
        q = np.asarray(d['quad']).reshape(-1, 2).astype(np.int32)
        cv2.polylines(img, [q], True, (0, 255, 255), 1)
        x0, y0 = max(0, c[0] - s), max(0, c[1] - s)
        crop = img[y0:y0 + 2 * s, x0:x0 + 2 * s]
        pad = np.zeros((2 * s, 2 * s, 3), np.uint8)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        t = cv2.resize(pad, (240, 240), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, '%d %.0fm' % (S.fid[i], d['range_m']), (4, 232),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(t)
    grid(tiles, 4, out)


def grid(tiles, cols, out):
    if not tiles:
        print('nothing to draw')
        return
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
    cv2.imwrite(out, sheet)
    print('wrote', out, sheet.shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default=M.PRIMARY)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--gate', type=int, default=-1)
    ap.add_argument('--out', default='ident.png')
    a = ap.parse_args()
    if a.frames:
        sheet_frames(a.session, a.frames, a.out)
    if a.gate >= 0:
        sheet_gate(a.session, a.gate, a.out)
