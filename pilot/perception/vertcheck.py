"""vertcheck.py -- measure per-gate tilt-from-vertical, PRIOR-INDEPENDENT.

Claire's verticality claim (relayed 2026-08-02, in two versions naming gate 8 then
correcting to gate 9 as the one non-vertical gate) is not something to encode from the
verbal claim alone -- producer.pnp_pose's VERT_DEBUG hook logs (gate_idx, tilt0, tilt1,
rms0, rms1) for EVERY ambiguous-pose frame with gravity available, where tilt0/tilt1 are
computed from cv2.solvePnPGeneric's own rms-sorted candidates BEFORE any exemption or
resolution logic runs -- so this measurement cannot be biased by which gate the code
currently exempts.

tilt0 = tilt-from-vertical of the LOWER-residual IPPE twin (the one pnp_pose would pick
absent any tilt-ambiguity logic at all). If verticality is real, tilt0 should cluster
near 0 deg for every gate except whichever one is actually tilted.

    python3 pilot/perception/vertcheck.py <session>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PILOT)

import producer as P    # noqa: E402


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else \
        '20260801-121520-vq2-lap-0-15'
    P.VERT_DEBUG = True
    P._VERT_LOG.clear()
    P.replay(session, outdir=os.path.join(HERE, 'producer_runs', 'vertcheck_tmp'))

    log = P._VERT_LOG
    print(f'\n{len(log)} ambiguous-pose observations with gravity available')
    by_gate = {}
    for g, t0, t1, r0, r1 in log:
        by_gate.setdefault(g, []).append((t0, t1))

    print('\n| gate | n | tilt0 med | tilt0 p90 | tilt0 max | n tilt0>10deg |')
    print('|---:|---:|---:|---:|---:|---:|')
    for g in sorted(by_gate):
        t0 = np.array([r[0] for r in by_gate[g]])
        n_big = int((t0 > 10.0).sum())
        print(f'| {g} | {len(t0)} | {np.median(t0):.2f} | {np.percentile(t0,90):.2f} | '
              f'{t0.max():.2f} | {n_big} ({100*n_big/len(t0):.0f}%) |')


if __name__ == '__main__':
    main()
