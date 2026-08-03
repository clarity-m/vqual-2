"""bugfix2_strips.py -- verification strips for Bug 2 (upward-view gates against the
lit ceiling): frames the OLD bright-only rule rejected as decoration, that the NEW
dark-gap rule accepts as real gates. Read back visually against labels_gates_all.json.

    python3 pilot/perception/bugfix2_strips.py
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HANDFRAMES = os.path.join(HERE, 'vq2_label', 'frames')

IN_WHITE_MAX, IN_V_MAX, IN_DARK_V, IN_DARK_FRAC = 0.15, 150.0, 90.0, 0.05


def features(hsv, quad):
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    ctr = q.mean(0)
    poly = (ctr + (q - ctr) * 0.7).astype(np.int32)
    msk = np.zeros(hsv.shape[:2], np.uint8)
    cv2.fillPoly(msk, [poly], 1)
    px = hsv[msk.astype(bool)]
    if len(px) < 4:
        return 0.0, 0.0, 1.0
    S, V = px[:, 1].astype(np.float32), px[:, 2].astype(np.float32)
    return (float(np.mean((S < 70) & (V > 150))), float(V.mean()),
            float(np.mean(V < IN_DARK_V)))


def old_reject(w, v):
    return w > IN_WHITE_MAX or v > IN_V_MAX


def new_reject(w, v, d):
    return old_reject(w, v) and d < IN_DARK_FRAC


def main():
    raw = json.load(open(os.path.join(HERE, 'labels_gates_all.json')))
    tiles = []
    for key, insts in sorted(raw.items()):
        img = cv2.imread(os.path.join(HANDFRAMES, key))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for inst in insts:
            if inst.get('unsure'):
                continue
            w, v, d = features(hsv, inst['corners'])
            if old_reject(w, v) and not new_reject(w, v, d):
                tile = img.copy()
                q = np.asarray(inst['corners'], np.int32)
                cv2.polylines(tile, [q.reshape(-1, 1, 2)], True, (0, 230, 230), 2)
                cv2.putText(tile, f'{key} w{w:.2f} v{v:.0f} d{d:.2f}: '
                            'OLD reject -> NEW accept', (6, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
    print(f'{len(tiles)} frames flip old-reject -> new-accept')
    tiles = tiles[:9]
    if not tiles:
        return
    H, W = tiles[0].shape[:2]
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = t
    out = os.path.join(HERE, 'bugfix2_upward_strip.png')
    cv2.imwrite(out, sheet)
    print('wrote', out)


if __name__ == '__main__':
    main()
