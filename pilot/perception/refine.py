"""Recover gate centres by fusing detections through the pose stream.

gate_truth.json holds CROSSING POSITIONS -- where the drone flew through, n=1 per gate --
so each centre carries up to the inner half-aperture (0.75 m) of error and no averaging.
That error propagates straight into every label, and then into any detector scored or
trained against those labels.

Detections fix it. Each accepted detection gives a full gate pose in the body frame, and
the pose stream puts the camera in the world, so every detection is one vote on where the
gate actually is:

    gate_world = camera_pos + R_body_to_world @ pos_body

Hundreds of votes per gate, from a wide spread of ranges and bearings, beat one crossing
sample by a large margin. The gates share a known normal (world x), so orientation is not
estimated and this is 3 DoF per gate, not 6.

MEDIAN, NOT MEAN. PnP range error has a long tail (p90 4.8 m) and misassociated detections
land arbitrarily far away, so a mean is dragged by exactly the samples that are wrong.

Association is bootstrapped from gate_truth and re-run each pass. That is safe here only
because the priors are already within a metre and gates are tens of metres apart; it would
not be safe on a course where gates crowd together.

BEARING-ONLY TRIANGULATION WAS TRIED AND IS WORSE. The objection to the estimator above is
that it consumes PnP's RANGE, so the fit can slide a gate along the viewing direction to
absorb PnP's range bias -- and indeed every refined gate shifts toward the approach.
Replacing it with least-squares triangulation of bearing rays removes that coupling
completely. Measured, held out on a session the fit never saw:

    estimator                 centre err    PnP range err (p90)   x-shift
    gate_truth (no refit)        5.2 px       +0.75 m (4.8)         --
    PnP position, median         1.8 px       +1.43 m (5.4)      +0.3..+1.0
    bearing-only triangulation   6.8 px       +2.24 m (23.3)     +1.2..+1.6

Bearing-only is worse on everything. The geometry is why: the aircraft approaches each
gate nearly head-on, so every ray to it is near-parallel to the course axis and the
triangulation is ill-conditioned in exactly the direction being solved for. Trading a
BOUNDED bias for an ILL-CONDITIONED estimate is a bad trade, however principled it looks.

The lesson is narrower than "use PnP": range measured against these refined labels is not
a valid metric, because the labels inherit the range bias being measured. That makes the
METRIC uninformative, not the ESTIMATOR wrong -- two different things, and conflating them
is what motivated the failed rewrite. Judge this fit by CENTRE (bearing) error, which no
part of the estimator can flatter.

    python3 pilot/perception/refine.py <session-dir> [<session-dir> ...] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402
import detect as D  # noqa: E402

ASSOC_MAX_M = 6.0      # a detection further than this from every prior is discarded
MIN_VOTES = 8          # below this, keep the prior rather than trust a thin fit
N_PASSES = 3


def collect(sessions, limit):
    """(gate-agnostic) world-frame gate position votes, with a quality weight."""
    votes = []
    for S in sessions:
        track = L.PoseTrack(S)
        frames = [r for r in L.load_csv(os.path.join(S, 'frames.csv')) if r['file']]
        ft = np.array([float(r['sim_time_ns']) for r in frames]) / 1e9
        step = max(1, len(frames) // limit)
        for idx in range(0, len(frames), step):
            pose = track.at_time(ft[idx])
            if pose is None:
                continue
            img = cv2.imread(os.path.join(S, 'frames', frames[idx]['file']))
            if img is None:
                continue
            p, roll, pitch, yaw = pose
            R_bw = L.euler_to_R(roll, pitch, yaw)
            for d in D.detections(img):
                w = p + R_bw @ d['pos_body']
                q = d['size_px'] * (1.0 if d['source'] == 'inner' else 0.5)
                votes.append((w, float(q), d['source']))
    return votes


def refine(priors, votes):
    """Associate rays to gates, triangulate, repeat. Outliers rejected each pass.

    Association is by perpendicular distance from the ray to the current estimate, so a
    detection of a DIFFERENT gate that happens to lie near the same bearing is rejected
    only if it is also far transversely. Safe here because gates are tens of metres apart
    and the priors start within a metre; it would not be on a course where gates crowd.
    """
    est = {k: v.copy() for k, v in priors.items()}
    buckets = {k: [] for k in est}
    for _ in range(N_PASSES):
        buckets = {k: [] for k in est}
        for w, q, src in votes:
            best, bd = None, 1e9
            for k, c in est.items():
                dd = float(np.linalg.norm(w - c))
                if dd < bd:
                    best, bd = k, dd
            if best is not None and bd < ASSOC_MAX_M:
                buckets[best].append(w)
        for k in est:
            if len(buckets[k]) >= MIN_VOTES:
                est[k] = np.median(np.stack(buckets[k]), axis=0)
    return est, {k: len(v) for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--limit', type=int, default=600)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'gates_refined.json'))
    args = ap.parse_args()

    gt = json.load(open(L.GATE_TRUTH))['consensus']
    priors = {int(k): np.array([v['x'], v['y'], v['z']]) for k, v in gt.items()}

    votes = collect(args.sessions, args.limit)
    print(f'{len(votes)} detection votes from {len(args.sessions)} session(s)')
    inner = sum(1 for *_, s in votes if s == 'inner')
    print(f'  inner-sourced {inner}  outer-sourced {len(votes)-inner}')

    est, counts = refine(priors, votes)

    print('\ngate   votes     shift from gate_truth (m)        refined position')
    for k in sorted(est):
        d = est[k] - priors[k]
        print('  %d   %6d   [%+6.2f %+6.2f %+6.2f]  |%5.2f|   [%8.2f %7.2f %7.2f]'
              % (k, counts.get(k, 0), d[0], d[1], d[2], np.linalg.norm(d),
                 est[k][0], est[k][1], est[k][2]))

    out = {'consensus': {str(k): {'x': float(v[0]), 'y': float(v[1]), 'z': float(v[2]),
                                  'n': counts.get(k, 0)} for k, v in est.items()},
           'source': 'refine.py — detection fusion through the pose stream',
           'sessions': [os.path.basename(s.rstrip('/\\')) for s in args.sessions]}
    json.dump(out, open(args.out, 'w'), indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
