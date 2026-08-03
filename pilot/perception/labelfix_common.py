"""Shared machinery for the labelfix_* passes (2026-08-01).

Three established auto-label error mechanisms (triage 2026-08-01):
  A. attitude interpolation aliasing at body rate > ~1 rad/s  -> annotate, down-weight;
  B. labels projected onto the BACK of a nearer gate           -> z-order solid coverage;
  C. gate-1 shear at zero body rate                            -> orientation refit.

This module holds only geometry + data plumbing reused by labelfix_gate1fit.py,
labelfix_occlusion.py and labelfix_build_v2.py. Projection, pose interpolation and the
yaw negation all come from label.py -- nothing here re-derives a sign.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402

SESSIONS = os.path.join(os.path.dirname(HERE), 'sessions')
OUTER_HALF = 2.7 / 2.0        # spec 3.7: full outer board is 2.7 m square
INNER_HALF = L.HALF           # 0.75 m

# Base corner offsets in the gate's own frame, BEFORE any orientation: plane spans
# world y-z (normal = world x), the survey assumption every VQ1 label was built on.
BASE_INNER = np.array([[0.0, -INNER_HALF, -INNER_HALF],
                       [0.0, +INNER_HALF, -INNER_HALF],
                       [0.0, +INNER_HALF, +INNER_HALF],
                       [0.0, -INNER_HALF, +INNER_HALF]])
BASE_OUTER = BASE_INNER * (OUTER_HALF / INNER_HALF)


def gate_R(rvec):
    """Orientation as a Rodrigues vector; rvec = 0 is the surveyed world-x-normal model."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    return R


def gate_corners(centre, rvec=None, outer=False):
    """4x3 world corners of the inner aperture (or the 2.7 m outer board edge)."""
    base = BASE_OUTER if outer else BASE_INNER
    c = np.asarray(centre, np.float64).reshape(3)
    if rvec is None:
        return c + base
    return c + base @ gate_R(rvec).T


def solid_mask(uv_inner, uv_outer, shape=(L.H, L.W)):
    """Raster mask of a gate's SOLID region: the full outer board minus the aperture
    opening. This is what physically blocks the view; the aperture does not."""
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(uv_outer).astype(np.int32).reshape(-1, 1, 2)], 1)
    cv2.fillPoly(m, [np.round(uv_inner).astype(np.int32).reshape(-1, 1, 2)], 0)
    return m


def band_mask(uv_inner, shape=(L.H, L.W)):
    """Raster mask of the target's frame band (annulus between inner quad and the
    weak-perspective outer quad) -- autolabel.py's _scale_quad rule, restated so the
    coverage is measured on the same region the orange-band visibility test used."""
    q = np.asarray(uv_inner, np.float32)
    c = q.mean(axis=0)
    outer = (q - c) * (OUTER_HALF / INNER_HALF) + c
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(outer).astype(np.int32).reshape(-1, 1, 2)], 1)
    cv2.fillPoly(m, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 0)
    return m


def quad_mask(uv, shape=(L.H, L.W)):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(uv, np.float32)).astype(np.int32)
                     .reshape(-1, 1, 2)], 1)
    return m


def behind_gate_fraction(pose, gates, target_gi, target_uv):
    """Fraction of the target instance's on-screen extent (band + quad) covered by the
    SOLID region of any gate NEARER to the camera.

    `gates` maps race index -> (centre, rvec|None). Nearer = smaller camera range to
    centre. Only solid board counts: a gate seen THROUGH another gate's open aperture is
    legitimate and must not flag. Returns (fraction, which_gate | None).
    """
    tgt = (band_mask(target_uv) | quad_mask(target_uv)).astype(bool)
    tot = float(tgt.sum())
    if tot == 0:
        return 0.0, None
    trange = float(np.linalg.norm(np.asarray(gates[target_gi][0]) - pose[0]))
    best_f, best_g = 0.0, None
    cover = np.zeros(tgt.shape, bool)
    for gi, (c, rv) in gates.items():
        if gi == target_gi:
            continue
        if float(np.linalg.norm(np.asarray(c) - pose[0])) >= trange:
            continue
        uv_i, z_i = L.project(gate_corners(c, rv), pose)
        uv_o, z_o = L.project(gate_corners(c, rv, outer=True), pose)
        if np.any(z_i <= 0.1) or np.any(z_o <= 0.1):
            continue
        sm = solid_mask(uv_i, uv_o).astype(bool)
        f = float((tgt & sm).sum()) / tot
        if f > best_f:
            best_f, best_g = f, gi
        cover |= sm
    return float((tgt & cover).sum()) / tot, best_g


# ---------------------------------------------------------------------------------------
# body rate at frame time -- triagecheck_rates.py's matching path, verbatim:
# frames.csv t_recv_wall_ns -> imu.csv t_wall_ns, |gyro| median over +/-50 ms.


class RateLookup:
    def __init__(self):
        self._imu, self._frames = {}, {}

    def _load(self, sess):
        ts, gy = [], []
        with open(os.path.join(SESSIONS, sess, 'imu.csv')) as fh:
            for r in csv.DictReader(fh):
                ts.append(float(r['t_wall_ns']))
                gy.append((float(r['xgyro']), float(r['ygyro']), float(r['zgyro'])))
        self._imu[sess] = (np.asarray(ts), np.linalg.norm(np.asarray(gy), axis=1))
        fr = {}
        with open(os.path.join(SESSIONS, sess, 'frames.csv')) as fh:
            for r in csv.DictReader(fh):
                if r['file']:
                    fr[r['file']] = float(r['t_recv_wall_ns'])
        self._frames[sess] = fr

    def rate(self, key):
        """key = '<session>/<file>' -> |body rate| rad/s, or None if the frame row is
        missing."""
        sess, fname = key.split('/', 1)
        if sess not in self._imu:
            self._load(sess)
        t = self._frames[sess].get(fname)
        if t is None:
            return None
        ts, mag = self._imu[sess]
        k = np.abs(ts - t) <= 50e6
        if k.any():
            return float(np.median(mag[k]))
        return float(np.median(mag[int(np.argmin(np.abs(ts - t)))]))


def load_refined_centres():
    """race index -> surveyed centre (gates_refined.json is one-based)."""
    d = json.load(open(os.path.join(HERE, 'gates_refined.json')))['consensus']
    return {int(k) - 1: np.array([v['x'], v['y'], v['z']]) for k, v in d.items()}


def frame_times(session_dir):
    """{file: sim_time_s} -- label.py's alignment rule (sim_time_ns, NOT recv wall)."""
    return {r['file']: float(r['sim_time_ns']) / 1e9
            for r in L.load_csv(os.path.join(session_dir, 'frames.csv')) if r['file']}
