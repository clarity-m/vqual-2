"""Detection cache for the VQ2 lap sessions.

Running detect.py over 6303 frames costs minutes, and the map is going to be rebuilt many
times while the identity/tracking logic is tuned. So detections are computed ONCE per
session and pickled; every later pass reads the pickle.

Nothing here changes what detect.py returns. The only additions are two flags stored
alongside each detection, both computed from code that already exists:

  * `deco`  -- labelgates.drop_decorations_by_parent() rejected it (a checkerboard square
    or wordmark inside a near gate's orange blob, implying an impossible range).
  * `bad`   -- labelgates.quality() reasons: outer-fallback, clipped, oblique, or PnP vs
    apparent-size range disagreement.
  * `in_white`, `in_v` -- what the fitted quad's INTERIOR looks like. A real gate aperture
    is a hole you see the dark hangar through; the AI-GP wordmark and the checkerboard
    strips are white paint on the orange frame, and their contours pass the same quad fit.
    Measured on four frames: real apertures read mean V 17-57 with 0-2% white pixels, every
    decoration reads mean V ~245 with 29-100% white. That is not a threshold that needs
    tuning, it is two populations that do not touch. See mapvq2.clean().

They are stored rather than applied, because TRACKING wants every detection it can get
(continuity) while METRIC MEASUREMENT wants only the clean ones. Filtering at cache time
would force one policy on both.

    python3 pilot/perception/vq2cache.py <session-dir> [<session-dir> ...]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402
import detect as D  # noqa: E402
import labelgates as LG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')


def cache_path(session):
    return os.path.join(CACHE, os.path.basename(os.path.normpath(session)) + '.det.pkl')


def build(session, force=False):
    out = cache_path(session)
    if os.path.exists(out) and not force:
        return out
    os.makedirs(CACHE, exist_ok=True)
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    res = {}
    t0 = time.time()
    for n, r in enumerate(frames):
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is None:
            continue
        det = D.detections(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        keep = LG.drop_decorations_by_parent(img, det)
        keep_ids = {id(d) for d in keep}
        recs = []
        for d in det:
            q = np.asarray(d['quad']).reshape(-1, 2).astype(np.float32)
            ctr = q.mean(0)
            # Shrink to 70% so the orange frame itself never enters the sample.
            poly = (ctr + (q - ctr) * 0.7).astype(np.int32)
            msk = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(msk, [poly], 1)
            px = hsv[msk.astype(bool)]
            if len(px) >= 4:
                in_white = float(np.mean((px[:, 1] < 70) & (px[:, 2] > 150)))
                in_v = float(px[:, 2].mean())
            else:
                in_white, in_v = 0.0, 0.0
            recs.append({
                'in_white': in_white,
                'in_v': in_v,
                'centre': np.asarray(d['centre'], float),
                'quad': np.asarray(d['quad'], float),
                'size_px': float(d['size_px']),
                'pos_body': np.asarray(d['pos_body'], float),
                'normal_body': np.asarray(d['normal_body'], float),
                'range_m': float(d['range_m']),
                'range_size_m': float(d['range_size_m']),
                'area': float(d['area']),
                'source': d['source'],
                'deco': id(d) not in keep_ids,
                'bad': LG.quality(d),
            })
        res[r['file']] = recs
        if n % 500 == 0:
            print(f'  {n}/{len(frames)}  {time.time()-t0:.0f}s', flush=True)
    with open(out, 'wb') as fh:
        pickle.dump(res, fh, protocol=4)
    print(f'wrote {out}  ({len(res)} frames, {time.time()-t0:.0f}s)')
    return out


def load(session):
    p = cache_path(session)
    if not os.path.exists(p):
        raise SystemExit(f'no cache for {session}; run vq2cache.py first')
    with open(p, 'rb') as fh:
        return pickle.load(fh)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    for s in a.sessions:
        print(s)
        build(s, a.force)
