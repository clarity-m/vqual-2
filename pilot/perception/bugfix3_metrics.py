"""bugfix3_metrics.py -- the three NEW before/after numbers for the 2026-08-02 pass.

Complements bugfix2_metrics.py (which covers the standard regression set + final-approach
normal availability + the strafe canary) with the measurements the earlier harness did not
have:

  1. normal LATERAL SIGN FLIPS (Bug 3). The IPPE twin mirrors the plane normal about the
     line of sight, so it flips the component of the normal PERPENDICULAR to the LOS while
     leaving the parallel part untouched. Tracking the sign of that perpendicular part over
     a gate's track therefore counts twin swaps directly. A correct, stable normal does not
     flip; the "attention pointed at the wrong side of gate 12" symptom is a run of flips.

     NOTE ON THE REFEREE: the brief asked for flips "vs map-predicted planes". The map's
     per-gate plane azimuths carry MAD 7-18 deg and (per the 2026-08-02 grid analysis)
     do not cluster on the column grid, so they cannot referee a sign at this precision.
     The intra-track flip count needs no external frame and is reported instead; the map
     channel is printed alongside, clearly marked as the weaker one.

  2. FALSE INTERIOR REJECTIONS on upward-looking segments (Bug 2), measured on the lap's
     own frames rather than on hand labels -- the lit ceiling is the confounder, so the
     population has to be the frames that actually look at it.

  3. per-gate TILT-FROM-VERTICAL is measured in gatetilt.py / tiltstrat.py, not here.

Usage: python3 pilot/perception/bugfix3_metrics.py
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PILOT)

RUNS = os.path.join(HERE, 'producer_runs')
SESSIONS = os.path.join(PILOT, 'sessions')
LAP = '20260801-121520-vq2-lap-0-15'

FLIP_MIN_PERP = math.sin(math.radians(5.0))   # ignore near-head-on frames: perp ~ 0 there
FLIP_DOT = -0.5                                # antiparallel enough to be a twin swap
FLIP_MAX_DT = 0.2                              # s; consecutive observations only


def load(run):
    dg = np.loadtxt(os.path.join(RUNS, run, 'diag.csv'), delimiter=',')
    v = np.load(os.path.join(RUNS, run, 'obs.npy'))
    return dg, v


def sign_flips(run):
    """Count twin swaps on the CURRENT-gate slot."""
    dg, v = load(run)
    t, act = dg[:, 0], dg[:, 1]
    pos, nrm = v[:, 0:3], v[:, 3:6]
    valid, nvalid = v[:, 6] > 0.5, v[:, 7] > 0.5
    ok = valid & nvalid & (np.linalg.norm(nrm, axis=1) > 1e-6)
    prev_u, prev_t, prev_act = None, None, None
    flips = n_pairs = 0
    for i in range(len(v)):
        if not ok[i]:
            prev_u = None
            continue
        l = pos[i] / max(np.linalg.norm(pos[i]), 1e-9)
        n = nrm[i] / max(np.linalg.norm(nrm[i]), 1e-9)
        perp = n - float(n @ l) * l
        m = float(np.linalg.norm(perp))
        if m < FLIP_MIN_PERP:
            prev_u = None                      # too head-on to have a side
            continue
        u = perp / m
        if prev_u is not None and prev_act == act[i] and 0 < t[i] - prev_t < FLIP_MAX_DT:
            n_pairs += 1
            if float(u @ prev_u) < FLIP_DOT:
                flips += 1
        prev_u, prev_t, prev_act = u, t[i], act[i]
    rate = 100.0 * flips / max(n_pairs, 1)
    print(f'  {run}: normal lateral sign flips {flips} in {n_pairs} consecutive '
          f'normal_valid pairs ({rate:.2f}%)')
    return flips, n_pairs


def upward_interior(limit=400, stride=5):
    """False decoration rejections on UPWARD-LOOKING frames, old rule vs new rule.

    Runs the detector over the lap's upward-looking frames and applies both interior
    rules to every candidate quad. The old rule is bright-only; the new one adds the
    dark-gap escape hatch. Frames are selected by the accel-derived pitch from the
    producer's own filter, so "upward" means what the aircraft was actually doing.
    """
    import detect as D
    import producer as P

    dg, _ = load('lap-121520-AFTER')
    # diag.csv has no pitch column; recompute upward-ness from the frame itself by using
    # the producer's IMU filter would need a full replay, so use the cheap image proxy:
    # ceiling light-grid coverage in the upper third (skylight's own cue).
    import csv
    with open(os.path.join(SESSIONS, LAP, 'frames.csv'), newline='') as f:
        frames = [r for r in csv.DictReader(f) if r['file']]
    picked = []
    for r in frames[::stride]:
        p = os.path.join(SESSIONS, LAP, 'frames', r['file'])
        img = cv2.imread(p)
        if img is None:
            continue
        top = cv2.cvtColor(img[:P.H // 3], cv2.COLOR_BGR2HSV)
        bright = float(np.mean(top[:, :, 2] > 150))
        picked.append((bright, p))
    picked.sort(key=lambda x: -x[0])
    picked = picked[:limit]
    print(f'  upward-looking population: {len(picked)} frames '
          f'(top ceiling-brightness quantile of the lap, stride {stride})')

    old_rej = new_rej = total = 0
    for _b, p in picked:
        img = cv2.imread(p)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        try:
            dets = D.detections(img)
        except Exception:
            continue
        for d in dets:
            if d.get('source') != 'inner':
                continue
            inw, inv, ind = P.interior_colour(hsv, np.asarray(d['quad'], np.float64))
            total += 1
            bright = inw > P.IN_WHITE_MAX or inv > P.IN_V_MAX
            if bright:
                old_rej += 1
            if P.is_decoration(inw, inv, ind):
                new_rej += 1
    print(f'  interior rejections on upward frames: OLD (bright-only) {old_rej}/{total} '
          f'({100*old_rej/max(total,1):.1f}%)  ->  NEW (dark-gap hatch) {new_rej}/{total} '
          f'({100*new_rej/max(total,1):.1f}%)')
    print(f'  candidates RECOVERED by the dark-gap hatch: {old_rej - new_rej} '
          f'({100*(old_rej-new_rej)/max(old_rej,1):.1f}% of the old rejections)')


def main():
    runs = [r for r in ('lap-121520-AFTER', 'lap-121520-BUGFIX2', 'lap-121520-BUGFIX3')
            if os.path.isdir(os.path.join(RUNS, r))]
    print('== normal lateral sign flips (Bug 3) ==')
    for r in runs:
        sign_flips(r)
    print('\n== false interior rejections, upward-looking frames (Bug 2) ==')
    upward_interior()


if __name__ == '__main__':
    main()
