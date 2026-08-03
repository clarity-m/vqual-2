"""Build autolabels_vq1_v3.json: v2 plus the same full-pose refit applied to more gates.

v2 handled mechanisms A (body_rate field), B (behind-gate removals) and C for gate 1.
The v2 acceptance test left 10 low-rate residual N slides, all of them mechanism C again
on gates 2 and 4. This pass regenerates corners for whichever additional gates PASSED
the held-out test (--apply, decided by measurement in labelfix_gate{N}_pose.json, never
by default), exactly as v2 did for gate 1:

  * corners reprojected from the refit pose through the frame's pose stream;
  * clipped / size_px recomputed; orange/occl numbers left stale (they described the old
    quad) with 'gate<N>_refit': true marking the instance;
  * behind-gate fraction recomputed on the NEW quad with ALL refit poses as occluders;
    >= thresh moves the instance to labelfix_negatives_v3.json (reason 'behind-gate').

v2's negatives are carried over unchanged; v2 itself is left in place. (key, gate)
remains the join key across v1/v2/v3.

    python3 pilot/perception/labelfix_build_v3.py --apply 2,4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402
from autolabel import canon  # noqa: E402
import labelfix_common as C  # noqa: E402

THRESH = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', default='2,4',
                    help='comma list of gates whose refit passed held-out')
    args = ap.parse_args()
    apply_gates = sorted(int(g) for g in args.apply.split(','))

    v2 = json.load(open(os.path.join(HERE, 'autolabels_vq1_v2.json')))
    neg = json.load(open(os.path.join(HERE, 'labelfix_negatives_v2.json')))

    # occluder set: surveyed centres, gate 5 recovered, refit poses for gate 1 + applied
    gates = {gi: (c, None) for gi, c in C.load_refined_centres().items()}
    g5 = json.load(open(os.path.join(HERE, 'gate5_recovered.json')))
    gates[int(g5['gate'])] = (np.array([g5['centre'][k] for k in 'xyz']), None)
    pts, poses = {}, {}
    for g in [1] + apply_gates:
        rec = json.load(open(os.path.join(HERE, f'labelfix_gate{g}_pose.json')))
        gates[g] = (np.asarray(rec['centre']), np.asarray(rec['rvec']))
        pts[g] = C.gate_corners(rec['centre'], rec['rvec'])
        poses[g] = rec
    print(f'applying refit poses for gates {apply_gates} '
          f'(deltas: ' + ', '.join(
          f'{g}: |{np.linalg.norm(poses[g]["centre_delta_m"]):.2f}| m '
          f'{poses[g]["rotation_deg"]:.1f} deg' for g in apply_gates) + ')')

    tracks, ftimes = {}, {}
    out = {}
    n_regen = {g: 0 for g in apply_gates}
    n_removed = 0
    moves = {g: [] for g in apply_gates}
    for key in sorted(v2):
        sess, fname = key.split('/', 1)
        sdir = os.path.join(C.SESSIONS, sess)
        if sess not in tracks:
            tracks[sess] = L.PoseTrack(sdir)
            ftimes[sess] = C.frame_times(sdir)
        pose = tracks[sess].at_time(ftimes[sess][fname])
        keep = []
        for d in v2[key]:
            d = dict(d)
            g = d['gate']
            if g in n_regen and pose is not None:
                old = np.asarray(d['corners'], np.float64)
                uv, z = L.project(pts[g], pose)
                if np.all(z > 0.1) and np.all(np.isfinite(uv)):
                    uv = canon(uv)
                    moves[g].append(float(np.linalg.norm(uv - canon(old),
                                                         axis=1).mean()))
                    e = [float(np.linalg.norm(uv[(j + 1) % 4] - uv[j]))
                         for j in range(4)]
                    d['corners'] = [[round(float(u), 2), round(float(v), 2)]
                                    for u, v in uv]
                    d['size_px'] = round(max(e), 2)
                    d['clipped'] = bool(np.any(uv[:, 0] < 0) or
                                        np.any(uv[:, 0] > L.W - 1) or
                                        np.any(uv[:, 1] < 0) or
                                        np.any(uv[:, 1] > L.H - 1))
                    d[f'gate{g}_refit'] = True
                    n_regen[g] += 1
                    f2, gi2 = C.behind_gate_fraction(pose, gates, g, uv)
                    if f2 >= THRESH:
                        d['visibility'] = 'hidden'
                        d['reason'] = 'behind-gate'
                        d['behind_frac'] = round(f2, 4)
                        d['behind_gate'] = gi2
                        neg.setdefault(key, []).append(d)
                        n_removed += 1
                        continue
            keep.append(d)
        if keep:
            for z, d in enumerate(keep):
                d['z'] = z
            out[key] = keep
    for key in neg:
        for z, d in enumerate(neg[key]):
            d['z'] = z

    with open(os.path.join(HERE, 'autolabels_vq1_v3.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(HERE, 'labelfix_negatives_v3.json'), 'w') as fh:
        json.dump(neg, fh, indent=1)
    for g in apply_gates:
        m = np.asarray(moves[g]) if moves[g] else np.zeros(1)
        print(f'gate {g}: {n_regen[g]} regenerated; corner move median '
              f'{np.median(m):.2f} px  p90 {np.percentile(m, 90):.2f}  max {m.max():.1f}')
    print(f'{sum(len(v) for v in out.values())} kept, '
          f'{sum(len(v) for v in neg.values())} negatives '
          f'({n_removed} newly behind-gate)')
    print('wrote autolabels_vq1_v3.json + labelfix_negatives_v3.json')


if __name__ == '__main__':
    main()
