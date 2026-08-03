"""tiltlook.py -- LOOK at the gates the tilt channels disagree about.

gatetilt.py's PnP-free edge_tilt says every gate's side edges are plumb to ~1-5 deg,
including gate 9 -- contradicting Claire's visual report of one laterally tilted gate.
Before believing either, render the evidence: for each requested gate, tile the frames
with the LARGEST apparent size and the frames with the HIGHEST measured lean, draw the
detected quad, and draw a true PLUMB line through the quad centre (gravity projected
into the image) so the comparison is visible rather than numerical.

    python3 pilot/perception/tiltlook.py 8 9 12
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L                       # noqa: E402
from gatetilt import edge_tilt, ROWS    # noqa: E402

SESSIONS = os.path.join(os.path.dirname(HERE), 'sessions')


def plumb_endpoints(row, ctr, length=140.0):
    """Project the gravity direction into the image at the quad centre.

    A short 3-D segment along gravity, placed at the gate's bearing, projected through K.
    Using the ray through the quad centre keeps the drawn line where the gate is.
    """
    g = np.asarray(row['g_body'], float)
    g = g / max(np.linalg.norm(g), 1e-9)
    g_cam = L.body_to_cam() @ g
    K = np.array([[L.FX, 0.0, L.CX], [0.0, L.FY, L.CY], [0.0, 0.0, 1.0]])
    # back-project the centre to a unit ray, step along gravity, reproject
    ray = np.linalg.inv(K) @ np.array([ctr[0], ctr[1], 1.0])
    ray /= np.linalg.norm(ray)
    P = ray * 10.0                       # arbitrary depth; direction is what matters
    out = []
    for s in (-1.0, 1.0):
        Q = P + g_cam * s * 1.5
        u = K @ Q
        out.append((u[0] / u[2], u[1] / u[2]))
    d = np.array(out[1]) - np.array(out[0])
    d = d / max(np.linalg.norm(d), 1e-9) * length
    return (tuple((np.array(ctr) - d).astype(int)), tuple((np.array(ctr) + d).astype(int)))


def main():
    gates = [int(a) for a in sys.argv[1:]] or [8, 9]
    rows = {int(k): v for k, v in json.load(open(ROWS)).items()}
    for g in gates:
        rr = []
        for r in rows.get(g, []):
            e = edge_tilt(r)
            if e is None or r['size_px'] < 25.0:
                continue
            rr.append((e['lean_abs'], e['lean'], r))
        if not rr:
            print(f'gate {g}: no rows')
            continue
        rr.sort(key=lambda x: -x[2]['size_px'])
        picks = rr[:4]                                    # biggest apparent size
        rr.sort(key=lambda x: -x[0])
        picks += rr[:4]                                   # highest measured lean
        tiles = []
        for lean_abs, lean, r in picks:
            path = os.path.join(SESSIONS, r['s'], 'frames',
                                _frame_file(r['s'], r['t']))
            img = cv2.imread(path) if path else None
            if img is None:
                continue
            q = np.asarray(r['quad'], float).reshape(4, 2)
            ctr = q.mean(0)
            cv2.polylines(img, [q.astype(np.int32).reshape(-1, 1, 2)], True,
                          (0, 230, 230), 2)
            a, b = plumb_endpoints(r, ctr)
            cv2.line(img, a, b, (255, 120, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f'g{g} lean {lean:+.1f} sz{r["size_px"]:.0f} '
                        f'rng{r["range"]:.1f} (magenta = plumb)', (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        if not tiles:
            print(f'gate {g}: no frames resolved')
            continue
        H, W = tiles[0].shape[:2]
        cols = 2
        nr = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((nr * H, cols * W, 3), np.uint8)
        for i, t in enumerate(tiles):
            r_, c_ = divmod(i, cols)
            sheet[r_ * H:(r_ + 1) * H, c_ * W:(c_ + 1) * W] = t
        out = os.path.join(HERE, f'tiltlook_gate{g}.png')
        cv2.imwrite(out, sheet)
        print('wrote', out, len(tiles), 'tiles')


_FRAME_CACHE: dict = {}


def _frame_file(sess, t):
    """Map a row's time back to its frame filename.

    gatetilt.collect() stores `float(S.t[i])`, and mapvq2's S.t is the RAW
    t_recv_wall_ns column -- absolute nanoseconds, NOT seconds from session start.
    Matching against a session-relative seconds axis silently returns frame 0-ish for
    every row, which renders each quad over an unrelated image.
    """
    if sess not in _FRAME_CACHE:
        import csv
        p = os.path.join(SESSIONS, sess, 'frames.csv')
        with open(p, newline='') as f:
            rd = list(csv.DictReader(f))
        rd = [r for r in rd if r['file']]
        _FRAME_CACHE[sess] = (
            np.array([float(r['t_recv_wall_ns']) for r in rd]),
            [r['file'] for r in rd])
    ts, files = _FRAME_CACHE[sess]
    i = int(np.argmin(np.abs(ts - t)))
    return files[i]


if __name__ == '__main__':
    main()
