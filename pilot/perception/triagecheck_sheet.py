"""Render the 6 lowest-rate N slides with their auto-label quads, zoomed.

    python3 pilot/perception/triagecheck_sheet.py
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(os.path.dirname(HERE), 'sessions')

TILE = 480


def main():
    picks = json.load(open(os.path.join(HERE, 'triagecheck_picks.json')))
    labels = json.load(open(os.path.join(HERE, 'autolabels_vq1.json')))

    tiles = []
    for pid in picks:
        key, inst = pid.rsplit('#', 1)
        inst = int(inst)
        sess, fname = key.split('/', 1)
        img = cv2.imread(os.path.join(SESSIONS, sess, 'frames', fname), cv2.IMREAD_COLOR)
        d = labels[key][inst]
        corners = np.asarray(d['corners'], np.float64)  # 4x2
        c = corners.mean(axis=0)
        size = float(d['size_px'])
        half = max(size * 1.1, 40.0)
        h, w = img.shape[:2]
        x0 = int(np.clip(c[0] - half, 0, w - 1)); x1 = int(np.clip(c[0] + half, 1, w))
        y0 = int(np.clip(c[1] - half, 0, h - 1)); y1 = int(np.clip(c[1] + half, 1, h))
        crop = img[y0:y1, x0:x1].copy()
        s = TILE / max(crop.shape[0], crop.shape[1])
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                          interpolation=cv2.INTER_CUBIC)
        pts = ((corners - [x0, y0]) * s).astype(np.int32)
        cv2.polylines(crop, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(crop, tuple(p), 3, (0, 0, 255), -1, cv2.LINE_AA)
        canvas = np.zeros((TILE + 28, TILE, 3), np.uint8)
        oy = (TILE - crop.shape[0]) // 2
        ox = (TILE - crop.shape[1]) // 2
        canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        cv2.putText(canvas, f'{pid}  g{d["gate"]} {size:.0f}px', (4, TILE + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    sheet = np.vstack(rows)
    out = os.path.join(HERE, 'triagecheck_sheet.png')
    cv2.imwrite(out, sheet)
    print('wrote', out, sheet.shape)


if __name__ == '__main__':
    main()
