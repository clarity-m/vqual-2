"""bugfix3_strips.py -- verification strips for Claire's three named regimes.

Reads back the BUGFIX3 replay (diag.csv + obs.npy) and renders the frames that decide
whether the 2026-08-02 fixes behave, drawing the producer's OWN current-gate output
(position and, when asserted, the approach normal) projected back into the image:

  A. BIG NEAR GATE with a FRESH updating normal -- Bug 1's target. The old size-only
     residual gate refused these; the corner-count rule should keep them, and the
     normal's staleness should stay small (it is being re-measured, not coasted).
  B. UPWARD-LOOKING gate accepted -- Bug 2. A real aperture against the lit ceiling that
     the bright-only interior test would have thrown away.
  C. act=12 HOVER -- Bug 3. Either a correct normal or an honest no-normal; what must NOT
     appear is a confident normal pointing at the wrong side.

    python3 pilot/perception/bugfix3_strips.py
"""
from __future__ import annotations

import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PILOT)

import producer as P    # noqa: E402

RUN = os.path.join(HERE, 'producer_runs', 'lap-121520-BUGFIX3')
LAP = os.path.join(PILOT, 'sessions', '20260801-121520-vq2-lap-0-15')


def draw(img, pos, nrm, nvalid, txt):
    uv = P.project_body(pos)
    if uv is not None:
        c = (int(uv[0]), int(uv[1]))
        cv2.circle(img, c, 7, (0, 255, 0), 2, cv2.LINE_AA)
        if nvalid and np.linalg.norm(nrm) > 1e-6:
            d = float(np.clip(np.linalg.norm(pos) * 0.35, 0.8, 4.0))
            uv2 = P.project_body(pos + d * nrm)
            if uv2 is not None:
                cv2.arrowedLine(img, c, (int(uv2[0]), int(uv2[1])), (255, 200, 0), 2,
                                cv2.LINE_AA, tipLength=0.3)
    cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1,
                cv2.LINE_AA)
    return img


def main():
    dg = np.loadtxt(os.path.join(RUN, 'diag.csv'), delimiter=',')
    v = np.load(os.path.join(RUN, 'obs.npy'))
    with open(os.path.join(LAP, 'frames.csv'), newline='') as f:
        frames = [r['file'] for r in csv.DictReader(f) if r['file']]

    t, act, valid = dg[:, 0], dg[:, 1], dg[:, 2] > 0
    rng, stale = dg[:, 4], dg[:, 5]
    pos, nrm = v[:, 0:3], v[:, 3:6]
    nvalid = v[:, 7] > 0.5

    picks = []

    # A. big near gate, fresh normal
    m = np.where(valid & nvalid & (rng < 6.0) & (stale < 0.10))[0]
    for i in m[:: max(1, len(m) // 3)][:3]:
        picks.append((i, 'A big-near rng%.1fm stale%.2fs NORMAL' % (rng[i], stale[i])))

    # B. upward-looking accepted: brightest-ceiling frames among valid ones
    cand = np.where(valid)[0]
    scored = []
    for i in cand[::7]:
        if i >= len(frames):
            continue
        img = cv2.imread(os.path.join(LAP, 'frames', frames[i]))
        if img is None:
            continue
        top = cv2.cvtColor(img[:P.H // 3], cv2.COLOR_BGR2HSV)
        scored.append((float(np.mean(top[:, :, 2] > 150)), int(i)))
    scored.sort(key=lambda x: -x[0])
    for _b, i in scored[:3]:
        picks.append((i, 'B upward ceil%.2f rng%.1fm %s'
                      % (_b, rng[i], 'NORMAL' if nvalid[i] else 'no-normal')))

    # C. act=12 hover
    m = np.where((act == 12) & valid)[0]
    for i in m[:: max(1, len(m) // 3)][:3]:
        picks.append((i, 'C act=12 rng%.1fm %s'
                      % (rng[i], 'NORMAL' if nvalid[i] else 'no-normal (honest)')))

    tiles = []
    for i, txt in picks:
        if i >= len(frames):
            continue
        img = cv2.imread(os.path.join(LAP, 'frames', frames[i]))
        if img is None:
            continue
        tiles.append(draw(img, pos[i], nrm[i], nvalid[i], 't=%.1fs %s' % (t[i], txt)))
    if not tiles:
        print('no tiles')
        return
    H, W = tiles[0].shape[:2]
    cols = 3
    nr = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((nr * H, cols * W, 3), np.uint8)
    for k, tl in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = tl
    out = os.path.join(HERE, 'bugfix3_regimes.png')
    cv2.imwrite(out, sheet)
    print('wrote', out, len(tiles), 'tiles')


if __name__ == '__main__':
    main()
