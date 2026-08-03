"""bugfix2_metrics.py -- before/after checks for the 2026-08-02 residual-gate (Bug 1)
and interior-colour (Bug 2) fixes, run on top of the already-fixed association/attention
producer (lap-121520-AFTER is the BEFORE baseline for this pass).

  BEFORE = producer_runs/lap-121520-AFTER      (association fix only; old residual/interior)
  AFTER  = producer_runs/lap-121520-BUGFIX2    (+ size-relative residual, + dark-gap interior)

Checks:
  1. normal_valid availability on the FINAL 15 m of approach into each crossing (the
     number Bug 1 should move: PnP was starving the normal exactly there).
  2. Standard regression set (valid/pose_valid/staleness/monotonicity/over-bound),
     reprinted for both runs so nothing regressed.
  3. Strafe 6-7 canary (separation stability), both runs.

Usage: python3 pilot/perception/bugfix2_metrics.py
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'producer_runs')


def load(run):
    dg = np.loadtxt(os.path.join(RUNS, run, 'diag.csv'), delimiter=',')
    v = np.load(os.path.join(RUNS, run, 'obs.npy'))
    return dg, v


def final_approach_normal(run, window_m=15.0):
    dg, v = load(run)
    t, act, valid, rng = dg[:, 0], dg[:, 1], dg[:, 2], dg[:, 4]
    normal_valid = v[:, 7] > 0.5
    adv = np.where(np.diff(act) > 0)[0]           # frame just before each crossing
    n_win = n_nv = n_valid_win = 0
    per_crossing = []
    for a in adv:
        # walk back from the crossing while active gate is constant and range < window
        j = a
        while j > 0 and act[j - 1] == act[a] and valid[j - 1] > 0 and rng[j - 1] < window_m:
            j -= 1
        w = slice(j, a + 1)
        vw = valid[w] > 0
        if vw.sum() == 0:
            continue
        nv = normal_valid[w] & vw
        per_crossing.append((int(act[a]), int(vw.sum()), int(nv.sum())))
        n_win += vw.sum()
        n_nv += nv.sum()
        n_valid_win += vw.sum()
    print(f'  {run}: final-{window_m:.0f}m-approach normal_valid: {n_nv}/{n_valid_win} '
          f'frames ({100 * n_nv / max(n_valid_win, 1):.1f}%) across {len(per_crossing)} '
          f'crossings')
    return per_crossing


def standard(run):
    dg, v = load(run)
    t, act, valid, pose_valid = dg[:, 0], dg[:, 1], dg[:, 2], dg[:, 3]
    rng, stale = dg[:, 4], dg[:, 5]
    print(f'  {run}: valid {100*valid.mean():.1f}%  pose_valid {100*pose_valid.mean():.1f}%  '
          f'mean staleness(valid) {stale[valid>0].mean():.3f}s')
    if dg.shape[1] > 10:
        bound = dg[:, 10]
        chk = (valid > 0) & np.isfinite(rng) & np.isfinite(bound)
        over = chk & (rng > bound)
        print(f'    over-bound (ratcheted, in-code): {100*over.sum()/max(chk.sum(),1):.2f}% '
              f'({int(over.sum())}/{int(chk.sum())})')
    adv = np.where(np.diff(act) > 0)[0]
    mono = []
    for a in adv:
        w = (t >= t[a] - 2.0) & (t <= t[a]) & (valid > 0) & np.isfinite(rng)
        if w.sum() >= 10:
            mono.append(float(np.mean(np.diff(rng[w]) < 0.05)))
    if mono:
        print(f'    approach monotonicity: median {np.median(mono):.2f} min {min(mono):.2f} '
              f'({len(mono)} crossings)')
    print(f'    crossings (unique active gate values seen): {len(set(act.astype(int)))}')


def pairs(run):
    p = os.path.join(RUNS, run, 'pairs.csv')
    if not os.path.exists(p):
        print(f'  {run}: no pairs.csv (run without --paircheck)')
        return
    pr = np.loadtxt(p, delimiter=',')
    sep = pr[:, 1]
    med = float(np.median(sep))
    print(f'  {run}: separation median {med:.2f} m  MAD '
          f'{float(np.median(np.abs(sep - med))):.2f}  n={len(pr)}')


def main():
    # BUGFIX2 is kept in the table as the REJECTED intermediate: it tightened the
    # producer's residual gate to a size-only rule and regressed every headline number.
    runs = ['lap-121520-AFTER', 'lap-121520-BUGFIX2', 'lap-121520-BUGFIX3']
    runs = [r for r in runs if os.path.isdir(os.path.join(RUNS, r))]
    print('== standard regression set ==')
    for r in runs:
        standard(r)
    print('\n== final-15m-approach normal_valid availability (Bug 1 target) ==')
    for r in runs:
        final_approach_normal(r)
    print('\n== strafe 6-7 canary ==')
    pairs('20260802-005431-vq2-strafe-6-7')       # original AFTER-fix pair-check run
    for r in ('strafe67-BUGFIX2', 'strafe67-BUGFIX3'):
        if os.path.isdir(os.path.join(RUNS, r)):
            pairs(r)


if __name__ == '__main__':
    main()
