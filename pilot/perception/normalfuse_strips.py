"""normalfuse_strips.py -- visual verification of the per-track normal accumulator.

Renders contact sheets for chosen approach windows with BOTH twin hypotheses drawn
(winner solid green, rejected twin thin magenta) plus the likelihood margin and the LOS
baseline behind the verdict, so a wrong-side normal is readable as a wrong-side normal
rather than as a confident one.

The producer is stepped from the beginning of the session every time -- the accumulator
is stateful and a window started mid-lap would show an accumulator that never saw the
approach it is supposed to have accumulated over.

Usage:
    python3 pilot/perception/normalfuse_strips.py [--session NAME] [--acts 12,3,6,15]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PILOT)

import producer as P    # noqa: E402

SESSIONS = os.path.join(PILOT, 'sessions')
OUT = os.path.join(HERE, 'producer_runs')
LAP = '20260801-121520-vq2-lap-0-15'


def sheet(tiles, path, cols=3):
    if not tiles:
        print('  (no tiles for', path, ')')
        return
    rows = (len(tiles) + cols - 1) // cols
    im = np.zeros((rows * P.H, cols * P.W, 3), np.uint8)
    for i, (cap, t) in enumerate(tiles):
        cv2.putText(t, cap, (6, P.H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        im[r * P.H:(r + 1) * P.H, c * P.W:(c + 1) * P.W] = t
    cv2.imwrite(path, im)
    print('  wrote', path, f'({len(tiles)} tiles)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default=LAP)
    ap.add_argument('--acts', default='12,3,6,15')
    ap.add_argument('--per-act', type=int, default=6)
    ap.add_argument('--gap-s', type=float, default=0.8)
    args = ap.parse_args()
    acts = [int(x) for x in args.acts.split(',')]

    sdir = os.path.join(SESSIONS, args.session)
    with open(os.path.join(sdir, 'frames.csv'), newline='') as f:
        frames = [r for r in csv.DictReader(f) if r['file']]
    with open(os.path.join(sdir, 'imu.csv'), newline='') as f:
        imu = list(csv.DictReader(f))
    with open(os.path.join(sdir, 'race.csv'), newline='') as f:
        race = list(csv.DictReader(f))
    imu_t = np.array([float(r['t_wall_ns']) for r in imu])
    race_t = np.array([float(r['t_wall_ns']) for r in race])

    prod = P.Producer()
    t0 = float(frames[0]['t_recv_wall_ns'])
    imu_i = 0
    per_act = {a: [] for a in acts}
    for fr in frames:
        t_ns = float(fr['t_recv_wall_ns'])
        t_s = (t_ns - t0) / 1e9
        j = int(np.searchsorted(imu_t, t_ns))
        rows, imu_i = imu[imu_i:j], j
        ri = max(0, int(np.searchsorted(race_t, t_ns)) - 1)
        active = int(race[ri]['active_gate_index'])
        img = cv2.imread(os.path.join(sdir, 'frames', fr['file']))
        obs = prod.step(img, rows, {'active_gate_index': active, 'armed': True,
                                    'n_gates_total': 17}, t_s)
        if img is None or active not in per_act:
            continue
        keep = per_act[active]
        g0 = obs.gates[0]
        if not g0.valid:
            continue
        if len(keep) < args.per_act and (not keep or t_s - keep[-1][0] >= args.gap_s):
            tr = prod.tracks.get(g0.index)
            cap = (f'act={active} t={t_s:.1f}s r={g0.range_m:.1f}m '
                   f'nv={int(g0.normal_valid)} '
                   f'{tr.normal_src if tr is not None else "-"}')
            keep.append((t_s, cap, P.annotate(img.copy(), obs, prod)))
    for a in acts:
        sheet([(c, t) for _ts, c, t in per_act[a]],
              os.path.join(OUT, f'normalfuse_act{a}.png'))


if __name__ == '__main__':
    main()
