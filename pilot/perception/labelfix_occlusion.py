"""Mechanism B: z-order occlusion test -- labels projected onto the BACK of a nearer gate.

WHY THE EXISTING CHECKS MISSED IT. autolabel.py's occl_frac only counts overlap between
APERTURE quads of labelled instances, so a label landing on the solid BOARD of a nearer
gate (outside that gate's aperture quad) scores 0.0; and the orange-band visibility filter
passes because the nearer gate supplies orange at exactly the projected location. Example
slides 20260731-195307/00008843.jpg#0 and .../00012502.jpg#0 show both failures at once.

THE TEST. For each labelled instance, project every NEARER gate's SOLID region -- the full
2.7 m outer board minus the aperture opening -- into the image and measure the fraction of
the target's band+quad on-screen extent it covers. Solid coverage only: a gate seen
THROUGH another gate's open aperture is legitimate and common, and does not count.
Geometry only, no JPEG is read.

Occluder set is all six gates (gate 5 from gate5_recovered.json -- it has no labels but
absolutely has a board), with --gate1-pose optionally substituting the mechanism-C refit
pose for gate 1 as an occluder.

    python3 pilot/perception/labelfix_occlusion.py [--thresh 0.5] [--sheet]

Writes labelfix_occl_flags.json: {"<key>#<idx>": {"behind_frac": f, "behind": gate}}
for every instance with behind_frac > 0, plus a summary. --sheet renders
labelfix_occl_sheet.png (random flagged + near-miss + through-aperture tiles) for the
read-back-before-trusting step. Modifies no existing file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402
import labelfix_common as C  # noqa: E402

THRESH = 0.5


def load_gate_set(gate1_pose=None):
    gates = {gi: (c, None) for gi, c in C.load_refined_centres().items()}
    g5 = os.path.join(HERE, 'gate5_recovered.json')
    if os.path.exists(g5):
        rec = json.load(open(g5))
        c = rec['centre']
        gates[int(rec['gate'])] = (np.array([c['x'], c['y'], c['z']]), None)
    if gate1_pose:
        rec = json.load(open(gate1_pose))
        gates[int(rec['gate'])] = (np.asarray(rec['centre']),
                                   np.asarray(rec['rvec']))
    return gates


def scan(labels, gates):
    """-> {key#idx: {'behind_frac': f, 'behind': gi}} for every instance with f > 0."""
    tracks, ftimes = {}, {}
    flags = {}
    keys = sorted(labels)
    for n, key in enumerate(keys):
        sess, fname = key.split('/', 1)
        sdir = os.path.join(C.SESSIONS, sess)
        if sess not in tracks:
            tracks[sess] = L.PoseTrack(sdir)
            ftimes[sess] = C.frame_times(sdir)
        t = ftimes[sess].get(fname)
        pose = tracks[sess].at_time(t) if t is not None else None
        if pose is None:
            continue
        for i, d in enumerate(labels[key]):
            frac, gi = C.behind_gate_fraction(pose, gates, d['gate'],
                                              np.asarray(d['corners'], np.float64))
            if frac > 0.0:
                flags[f'{key}#{i}'] = {'behind_frac': round(frac, 4), 'behind': gi}
        if n % 1000 == 0:
            print(f'  {n}/{len(keys)} frames, {len(flags)} nonzero', flush=True)
    return flags


def tile(labels, key, idx, note, side=180):
    sess, fname = key.split('/', 1)
    img = cv2.imread(os.path.join(C.SESSIONS, sess, 'frames', fname))
    if img is None:
        return None
    uv = np.asarray(labels[key][idx]['corners'], np.float64)
    cv2.polylines(img, [uv.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
    c = uv.mean(axis=0)
    s = max(float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1]))) * 3.0
    s = max(s, 90.0)
    a = side / s
    M = np.array([[a, 0, side / 2 - c[0] * a], [0, a, side / 2 - c[1] * a]], np.float32)
    t = cv2.warpAffine(img, M, (side, side))
    cv2.putText(t, note, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
    cv2.putText(t, fname + f'#{idx}', (2, side - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                (200, 200, 255), 1)
    return t


def sheet(labels, flags, thresh, path, n_each=12, seed=0):
    """Three rows: flagged (must look hidden), near-miss 0.15..thresh (judgement zone),
    and instances with a NEARER gate in frame but behind_frac == 0 -- the
    seen-through-the-aperture population that must NOT be flagged."""
    rng = np.random.default_rng(seed)
    flagged = [k for k, v in flags.items() if v['behind_frac'] >= thresh]
    near = [k for k, v in flags.items() if 0.15 <= v['behind_frac'] < thresh]
    thru = [k for k, v in flags.items() if 0.0 < v['behind_frac'] < 0.05]
    rows = []
    for name, pool in (('FLAG', flagged), ('near', near), ('thru', thru)):
        picks = rng.permutation(len(pool))[:n_each]
        tiles = []
        for j in picks:
            key, idx = pool[int(j)].rsplit('#', 1)
            v = flags[pool[int(j)]]
            t = tile(labels, key, int(idx),
                     f'{name} {v["behind_frac"]:.2f} behind g{v["behind"]}')
            if t is not None:
                tiles.append(t)
        while len(tiles) < n_each:
            tiles.append(np.zeros((180, 180, 3), np.uint8))
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))
    print(f'wrote {path}  (rows: flagged >= {thresh} / near-miss / trace-overlap)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=os.path.join(HERE, 'autolabels_vq1.json'))
    ap.add_argument('--gate1-pose', default=None,
                    help='labelfix_gate1_pose.json: use the refit gate-1 pose as occluder')
    ap.add_argument('--thresh', type=float, default=THRESH)
    ap.add_argument('--out', default=os.path.join(HERE, 'labelfix_occl_flags.json'))
    ap.add_argument('--sheet', action='store_true')
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    n_inst = sum(len(v) for v in labels.values())
    gates = load_gate_set(args.gate1_pose)
    print(f'{n_inst} instances over {len(labels)} frames; occluder gates {sorted(gates)}')
    flags = scan(labels, gates)

    fr = np.array([v['behind_frac'] for v in flags.values()])
    print(f'\nnonzero behind_frac: {len(flags)} instances '
          f'({100 * len(flags) / n_inst:.2f}%)')
    for t in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f'  behind_frac >= {t:4.2f}: {int((fr >= t).sum())}')
    flagged = {k: v for k, v in flags.items() if v['behind_frac'] >= args.thresh}
    from collections import Counter
    cg = Counter(k.split('#')[0].split('/')[0] for k in flagged)
    print(f'flagged (>= {args.thresh}): {len(flagged)}  by session {dict(cg)}')

    # the two triage examples MUST flag
    for ex in ('20260731-195307/00008843.jpg#0', '20260731-195307/00012502.jpg#0'):
        v = flags.get(ex)
        print(f'  example {ex}: behind_frac='
              f'{v["behind_frac"] if v else 0.0} behind='
              f'{v["behind"] if v else None}  -> '
              f'{"FLAGGED" if v and v["behind_frac"] >= args.thresh else "NOT FLAGGED"}')

    json.dump({'thresh': args.thresh, 'gate1_pose': args.gate1_pose,
               'flags': flags}, open(args.out, 'w'))
    print(f'wrote {args.out}')

    if args.sheet:
        sheet(labels, flags, args.thresh,
              os.path.join(HERE, 'labelfix_occl_sheet.png'))


if __name__ == '__main__':
    main()
