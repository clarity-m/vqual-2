"""Gate detection: orange mask -> contour -> quadrilateral -> solvePnP.

Fills the geometry half of interface.GateObs from a single frame. No temporal state, no
pose input -- deliberately. This is the layer that must keep working under VQ2, where
there is no pose stream to lean on and the only external spatial fact is
active_gate_index. Tracking and attention sit ABOVE this.

Thresholds come from a measurement on the VQ2 start frame (NOTES.md): the three bright
classes separate on SATURATION alone -- orange gate 202, cyan ribbon 197, white lights 3 --
and 66.5% of the frame sits below V=40. So the mask is generous on hue and strict on
saturation, which is what keeps ceiling lights out.

WHAT IS DETECTED IS THE FRAME, NOT THE APERTURE. A gate renders as an orange annulus: an
outer border with a hole. The hole is what we need -- interface.py's range relation
480/gate_px is defined on the 1500 mm INNER aperture -- so the quad is fitted to the
contour's HOLE (child in the hierarchy), never to its outer boundary. Fitting the outer
edge instead yields a range biased short by the border width, which grows as you close in
and therefore corrupts exactly the approach it is meant to guide.

Scoring mode compares against label.py's projected corners:

    python3 pilot/perception/detect.py <session-dir> --score --limit 400
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402

# Mask. Hue wraps at red, so orange is two intervals in OpenCV's 0..179 hue.
S_MIN, V_MIN = 110, 110
HUE_LO, HUE_HI = 12, 170

MIN_HOLE_AREA = 120.0      # px^2. Below this the aperture is under ~11 px across.
MAX_ASPECT = 3.0           # a gate seen very obliquely, past which PnP is worthless
MIN_FILL = 0.55            # hole area / fitted quad area; rejects ragged blobs

# SOLVEPNP_IPPE_SQUARE does NOT accept an arbitrary winding: OpenCV fixes the order as
# TL, TR, BR, BL with +y UP in object space, and silently returns a garbage pose for any
# other ordering. Caught by measurement, not by reading -- with the winding below wrong,
# PnP range was off by -18.6 m median while size-only range on the SAME quad was off by
# -0.64 m. Two range estimates from one quad disagreeing is the signature; keep both
# wired up in score() so a future regression here cannot hide.
OBJ = np.array([[-L.HALF, +L.HALF, 0.0],
                [+L.HALF, +L.HALF, 0.0],
                [+L.HALF, -L.HALF, 0.0],
                [-L.HALF, -L.HALF, 0.0]], dtype=np.float64)
K = np.array([[L.FX, 0.0, L.CX], [0.0, L.FY, L.CY], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)


def orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (((h <= HUE_LO) | (h >= HUE_HI)) & (s > S_MIN) & (v > V_MIN))
    m = m.astype(np.uint8)
    # Close only. Opening would erode thin gate borders at range, which is where the
    # detector is most needed and least able to spare pixels.
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def order_quad(pts):
    """Corners as TL, TR, BR, BL in image coords, so PnP sees a consistent winding."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def fit_quad(contour):
    """Contour -> 4 corners, or None. minAreaRect is the fallback for rounded corners."""
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.04, 0.06, 0.08):
        ap = cv2.approxPolyDP(contour, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            return ap.reshape(4, 2).astype(np.float64)
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return box.astype(np.float64)


def detections(bgr):
    """All plausible gate apertures in one frame, nearest (largest) first."""
    m = orange_mask(bgr)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(cnts):
        if hier[i][3] < 0:          # outer boundary of the frame body -- not the aperture
            continue
        area = cv2.contourArea(c)
        if area < MIN_HOLE_AREA:
            continue
        quad = fit_quad(c)
        if quad is None:
            continue
        qa = cv2.contourArea(quad.astype(np.float32))
        if qa <= 1.0 or area / qa < MIN_FILL:
            continue
        e = [np.linalg.norm(quad[(j + 1) % 4] - quad[j]) for j in range(4)]
        if min(e) < 4.0 or max(e) / max(min(e), 1e-6) > MAX_ASPECT:
            continue
        ok, rvec, tvec = cv2.solvePnP(OBJ, order_quad(quad), K, DIST,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        # Camera frame -> body frame. body_to_cam() is orthonormal, so transpose inverts.
        pos_body = L.body_to_cam().T @ t
        normal_cam = R[:, 2]
        # Sign the normal toward the camera: for a gate not yet crossed we are on its
        # approach side by construction (interface.py). PnP alone gives an UNDIRECTED
        # normal, so this is a choice the geometry cannot make for us.
        if float(normal_cam @ t) > 0:
            normal_cam = -normal_cam
        out.append({
            'quad': order_quad(quad),
            'centre': quad.mean(axis=0),
            'size_px': float(max(e)),
            'pos_body': pos_body,
            'normal_body': L.body_to_cam().T @ normal_cam,
            'range_m': float(np.linalg.norm(t)),
            'range_size_m': 480.0 / max(float(max(e)), 1e-6),
            'area': float(area),
        })
    out.sort(key=lambda d: -d['area'])
    return out


def score(session, limit):
    """Detections vs label.py's projections. Reports recall and metric error."""
    track = L.PoseTrack(session)
    import json
    gt = json.load(open(L.GATE_TRUTH))['consensus']
    gates = {int(k): np.array([v['x'], v['y'], v['z']]) for k, v in gt.items()}
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    ft = np.array([float(r['sim_time_ns']) for r in frames]) / 1e9

    n_vis = n_hit = 0
    px_err, rng_err, rng_size_err = [], [], []
    step = max(1, len(frames) // limit)
    for idx in range(0, len(frames), step):
        pose = track.at_time(ft[idx])
        if pose is None:
            continue
        img = cv2.imread(os.path.join(session, 'frames', frames[idx]['file']))
        if img is None:
            continue
        # Which gates are genuinely visible, per the labels?
        vis = []
        for gi, g in gates.items():
            uv, z = L.project(L.gate_corners_world(g), pose)
            if np.any(z <= 0.5):
                continue
            c = uv.mean(axis=0)
            size = np.linalg.norm(uv[0] - uv[2]) / math.sqrt(2)
            if 0 <= c[0] < L.W and 0 <= c[1] < L.H and size >= 12:
                vis.append((gi, c, float(np.mean(z)), size))
        if not vis:
            continue
        det = detections(img)
        for gi, c, zt, size in vis:
            n_vis += 1
            if not det:
                continue
            d = min(det, key=lambda dd: np.linalg.norm(dd['centre'] - c))
            e = float(np.linalg.norm(d['centre'] - c))
            if e < max(0.5 * size, 15.0):
                n_hit += 1
                px_err.append(e)
                rng_err.append(d['range_m'] - zt)
                rng_size_err.append(d['range_size_m'] - zt)

    print(f'visible gate instances : {n_vis}')
    print(f'detected               : {n_hit}   recall {n_hit/max(n_vis,1):.2f}')
    if px_err:
        print(f'centre error   median {np.median(px_err):5.1f} px')
        print(f'PnP range err  median {np.median(rng_err):+5.2f} m   '
              f'p90 |err| {np.percentile(np.abs(rng_err), 90):.2f} m')
        print(f'size range err median {np.median(rng_size_err):+5.2f} m   '
              f'p90 |err| {np.percentile(np.abs(rng_size_err), 90):.2f} m')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--limit', type=int, default=400)
    args = ap.parse_args()
    if args.score:
        score(args.session, args.limit)


if __name__ == '__main__':
    main()
