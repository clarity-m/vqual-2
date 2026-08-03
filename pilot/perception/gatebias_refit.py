"""Refit ONE VQ1 gate's 3D centre from detections + pose, gate5.py's method verbatim.

This is gate5.py's --validate path pointed at an arbitrary race gate: hide the gate's
(suspect) surveyed centre from the known set, collect world-frame votes during the
windows where race.csv says it is the active gate, exclude votes near the REMAINING
known gates, densest-cluster seed + iterated median, pick the NEAREST surviving cluster.
Nothing here modifies gates_refined.json or autolabel.py.

    python3 pilot/perception/gatebias_refit.py <session> ... --target N [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate5 as G5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--target', type=int, required=True)
    ap.add_argument('--limit', type=int, default=4000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    known = G5.load_known()
    old = known[args.target]
    hidden = {k: v for k, v in known.items() if k != args.target}
    print(f'refit race gate {args.target}; surveyed centre '
          f'[{old[0]:8.2f} {old[1]:7.2f} {old[2]:7.2f}] HIDDEN from the known set')

    est, sc, st, votes, mask, meta = G5.fit(args.sessions, args.target, hidden,
                                            args.limit)
    if est is None:
        print('no fit')
        return
    delta = est - old
    print(f'\nOLD  [{old[0]:8.2f} {old[1]:7.2f} {old[2]:7.2f}]')
    print(f'NEW  [{est[0]:8.2f} {est[1]:7.2f} {est[2]:7.2f}]')
    print('DELTA [%+7.2f %+7.2f %+7.2f]  |%.2f| m' % (*delta, np.linalg.norm(delta)))
    rr = [m[4] for m, k in zip(meta, mask) if k]
    print(f'inlier ranges: min {min(rr):.1f} med {np.median(rr):.1f} max {max(rr):.1f} m'
          f'   sources: '
          f'{sum(1 for m, k in zip(meta, mask) if k and m[3] == "inner")} inner / '
          f'{sum(1 for m, k in zip(meta, mask) if k and m[3] == "outer")} outer')

    out = args.out or os.path.join(HERE, f'gatebias_gate{args.target}_refit.json')
    unc = {'why': 'bootstrap SE understates: correlated frames + systematic PnP bias',
           'within_cluster_mad_m': sc['mad'],
           'gate5_leave_one_out_gate4_m': 0.2074865445991895,
           'gate5_use_this_m': 0.31}
    json.dump({
        'source': 'gatebias_refit.py - gate5.py estimator, target hidden from known set',
        'sessions': [os.path.basename(os.path.normpath(s)) for s in args.sessions],
        'gate': args.target,
        'gates_refined_key': str(args.target + 1),
        'old_centre': old.tolist(),
        'centre': {'x': float(est[0]), 'y': float(est[1]), 'z': float(est[2])},
        'delta_m': delta.tolist(),
        'delta_norm_m': float(np.linalg.norm(delta)),
        'scatter': sc, 'stats': st, 'uncertainty': unc,
    }, open(out, 'w'), indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
