"""normalfuse_metrics.py -- acceptance measurements for the per-track IPPE-twin
accumulator (normalfuse.py).

BEFORE = producer_runs/lap-121520-BUGFIX3   (per-frame temporal+verticality resolver)
AFTER  = producer_runs/lap-121520-NORMFUSE  (per-track accumulation)

Reported:
  1. normal LATERAL SIGN-FLIP rate within tracks -- the headline (baseline 4.02%). Same
     estimator as bugfix3_metrics.sign_flips so the numbers are comparable.
  2. AGREEMENT with the map / race-line-bisector prior, per gate: median |err| and MAD of
     the accumulated normal's levelled azimuth against course_vq2.json.
  3. normal_valid AVAILABILITY overall and in the final 15 m before each crossing -- the
     "did we buy consistency by refusing everything" check.
  4. SEPARATION-VS-BASELINE curve, from the per-frame twin log: how much LOS motion a
     track needs before the hypotheses actually separate, and whether the early decision
     agrees with the same track's well-supported late decision.
  5. Standard regression set + strafe canary.

Usage: python3 pilot/perception/normalfuse_metrics.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PILOT)

import normalfuse as NF   # noqa: E402

RUNS = os.path.join(HERE, 'producer_runs')
BEFORE = 'lap-121520-BUGFIX3'
AFTER = 'lap-121520-NORMFUSE'

FLIP_MIN_PERP = math.sin(math.radians(5.0))
FLIP_DOT = -0.5
FLIP_MAX_DT = 0.2

# normal_log.npy column layout (producer._fuse_normal)
C_T, C_G, C_SZ, C_NS, C_TWIN, C_BASE, C_NSAMP, C_LLR, C_LLRP, C_WIN, C_DEC, C_SRC, \
    C_UNAMB = range(13)
C_C0, C_C1, C_LOS, C_NA, C_NL, C_MAPAZ, C_MAPERR, C_NRESET = 13, 16, 19, 22, 25, 28, 29, 30


def load(run):
    dg = np.loadtxt(os.path.join(RUNS, run, 'diag.csv'), delimiter=',')
    v = np.load(os.path.join(RUNS, run, 'obs.npy'))
    return dg, v


# ---------------------------------------------------------------- 1. sign flips
def sign_flips(run, quiet=False):
    dg, v = load(run)
    t, act = dg[:, 0], dg[:, 1]
    pos, nrm = v[:, 0:3], v[:, 3:6]
    valid, nvalid = v[:, 6] > 0.5, v[:, 7] > 0.5
    ok = valid & nvalid & (np.linalg.norm(nrm, axis=1) > 1e-6)
    prev_u = prev_t = prev_act = None
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
            prev_u = None
            continue
        u = perp / m
        if prev_u is not None and prev_act == act[i] and 0 < t[i] - prev_t < FLIP_MAX_DT:
            n_pairs += 1
            if float(u @ prev_u) < FLIP_DOT:
                flips += 1
        prev_u, prev_t, prev_act = u, t[i], act[i]
    rate = 100.0 * flips / max(n_pairs, 1)
    if not quiet:
        print(f'  {run:26s} lateral sign flips {flips:4d} / {n_pairs:5d} consecutive '
              f'normal_valid pairs = {rate:5.2f}%')
    return rate, flips, n_pairs


# ---------------------------------------------------- 2. agreement with the map prior
def map_agreement(run):
    """|accumulated normal azimuth - bisector| per gate, from the producer's own
    per-frame log (which already carries the levelled-frame error it computed live)."""
    p = os.path.join(RUNS, run, 'normal_log.npy')
    if not os.path.exists(p):
        print(f'  {run}: no normal_log.npy (run with --normal-debug)')
        return
    a = np.load(p)
    err = a[:, C_MAPERR]
    ok = np.isfinite(err) & (a[:, C_DEC] > 0.5)
    print(f'  {run}: {int(ok.sum())} decided frames with a trusted map azimuth '
          f'({100*ok.mean():.1f}% of logged normal frames)')
    if ok.sum() == 0:
        return
    e = np.abs(err[ok])
    print(f'    ALL GATES: median |err| {np.median(e):5.1f} deg   MAD '
          f'{np.median(np.abs(e - np.median(e))):4.1f}   p90 {np.percentile(e, 90):5.1f}')
    print('    per gate:  gate    n   median|err|   MAD    signed median')
    for g in sorted(set(a[ok, C_G].astype(int))):
        m = ok & (a[:, C_G].astype(int) == g)
        if m.sum() < 20:
            continue
        eg = np.abs(err[m])
        print(f'               {g:3d} {int(m.sum()):5d}   {np.median(eg):8.1f}   '
              f'{np.median(np.abs(eg - np.median(eg))):5.1f}   '
              f'{np.median(err[m]):+8.1f}')


# ---------------------------------------------------------- 3. availability
def availability(run, window_m=15.0):
    dg, v = load(run)
    act, valid, rng = dg[:, 1], dg[:, 2], dg[:, 4]
    nv = v[:, 7] > 0.5
    vv = v[:, 6] > 0.5
    overall = 100.0 * (nv & vv).sum() / max(vv.sum(), 1)
    adv = np.where(np.diff(act) > 0)[0]
    n_win = n_nv = 0
    for a in adv:
        j = a
        while j > 0 and act[j - 1] == act[a] and valid[j - 1] > 0 \
                and rng[j - 1] < window_m:
            j -= 1
        w = slice(j, a + 1)
        m = vv[w]
        n_win += int(m.sum())
        n_nv += int((nv[w] & m).sum())
    print(f'  {run:26s} normal_valid: overall {overall:5.1f}% of valid frames | '
          f'final-{window_m:.0f}m {100*n_nv/max(n_win,1):5.1f}% ({n_nv}/{n_win}, '
          f'{len(adv)} crossings)')
    if dg.shape[1] > 15:
        src = dg[:, 15].astype(int)
        names = {0: 'refused/none', 1: 'unambiguous', 2: 'accum-consistency',
                 3: 'accum-cons+prior', 4: 'accum-prior', 5: 'accum-coast'}
        tot = max((vv).sum(), 1)
        parts = ', '.join(f'{names[k]} {100*np.sum((src == k) & vv)/tot:.1f}%'
                          for k in sorted(names) if np.any((src == k) & vv))
        print(f'      source mix (current slot, valid frames): {parts}')


# ------------------------------------------ 4. separation vs angular baseline
def _episodes(a):
    """Contiguous stretches of one track's accumulator with no reset and no re-seed.
    Within an episode the two hypothesis SLOTS (mu[0], mu[1]) are the same two physical
    directions, so the logged winner index is directly comparable across frames -- which
    is what makes 'did the early decision survive' answerable without a common frame."""
    eps, cur = [], []
    for i in range(len(a)):
        if cur and (a[i, C_G] != a[cur[-1], C_G]
                    or a[i, C_T] - a[cur[-1], C_T] > 2.5
                    or a[i, C_NRESET] != a[cur[-1], C_NRESET]
                    or a[i, C_NSAMP] < a[cur[-1], C_NSAMP]):
            eps.append(cur)
            cur = []
        cur.append(i)
    if cur:
        eps.append(cur)
    return eps


BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 35), (35, 60), (60, 1e9)]


def baseline_curve(run):
    """SEPARATION VS ANGULAR BASELINE -- the question that decides whether this fix helps
    in the regime the race actually flies.

    Two curves, per bin of accumulated LOS baseline (degrees, measured in the track-common
    frame so body rotation contributes nothing):
      * |LLR_consistency| -- the raw evidence the geometry supplies, decision rule aside;
      * self-agreement    -- P(winner at this frame == winner at the end of the episode)
        over episodes that END confidently decided (baseline >= 25 deg, LLR >= 8). The
        referee is the track's own best-supported verdict, not external truth (VQ2 has no
        gate ground truth); it answers exactly the failure mode at issue -- an early lock
        that the later evidence contradicts.
    """
    p = os.path.join(RUNS, run, 'normal_log.npy')
    if not os.path.exists(p):
        print(f'  {run}: no normal_log.npy')
        return
    a = np.load(p)
    llr_cons = np.abs(a[:, C_LLR] - a[:, C_LLRP])
    eps = _episodes(a)
    stat = {b: [0, 0, 0, 0, []] for b in BINS}   # n, n_dec, n_agree, n_dec_ref, llrs
    n_ref = 0
    for ep in eps:
        for i in ep:
            for lo, hi in BINS:
                if lo <= a[i, C_BASE] < hi:
                    s = stat[(lo, hi)]
                    s[0] += 1
                    s[4].append(llr_cons[i])
                    if a[i, C_DEC] > 0.5:
                        s[1] += 1
        last = ep[-1]
        if not (a[last, C_DEC] > 0.5 and a[last, C_BASE] >= 25.0
                and a[last, C_LLR] >= 8.0):
            continue
        n_ref += 1
        wref = a[last, C_WIN]
        for i in ep:
            if a[i, C_DEC] < 0.5:
                continue
            for lo, hi in BINS:
                if lo <= a[i, C_BASE] < hi:
                    s = stat[(lo, hi)]
                    s[3] += 1
                    s[2] += int(a[i, C_WIN] == wref)
    print(f'  {run}: {len(eps)} track episodes, {n_ref} end confidently decided '
          f'(the self-agreement referee set)')
    print('    baseline(deg)       n   median|LLR_cons|   p90    decided%   '
          'self-agree% (n)')
    for lo, hi in BINS:
        s = stat[(lo, hi)]
        if s[0] == 0:
            continue
        hs = 'inf' if hi > 1e8 else f'{hi:.0f}'
        ag = f'{100*s[2]/s[3]:5.1f} ({s[3]})' if s[3] else '    -'
        print(f'    [{lo:5.0f},{hs:>5s})  {s[0]:6d}   {np.median(s[4]):14.2f}  '
              f'{np.percentile(s[4], 90):6.2f}   {100*s[1]/s[0]:7.1f}    {ag}')


def approach_baseline(run):
    """How much baseline does a track actually HAVE on final approach?  This is the
    honest limit: a straight-in approach translates along the LOS and generates none."""
    p = os.path.join(RUNS, run, 'normal_log.npy')
    if not os.path.exists(p):
        return
    a = np.load(p)
    dg, _ = load(run)
    act = np.interp(a[:, C_T], dg[:, 0], dg[:, 1])
    cur = np.abs(act - a[:, C_G]) < 0.5             # current-slot rows only
    rng = np.interp(a[:, C_T], dg[:, 0], dg[:, 4])
    for lab, m in (('all current-slot frames', cur),
                   ('final 15 m', cur & (rng < 15.0)),
                   ('final 8 m', cur & (rng < 8.0))):
        if m.sum() < 10:
            continue
        b = a[m, C_BASE]
        print(f'    {lab:24s} n={int(m.sum()):5d}  baseline p10/median/p90 = '
              f'{np.percentile(b,10):5.1f} / {np.median(b):5.1f} / '
              f'{np.percentile(b,90):5.1f} deg   >= {NF.BASELINE_MIN_DEG:.0f} deg: '
              f'{100*np.mean(b >= NF.BASELINE_MIN_DEG):4.1f}%')


# --------------------------------------------------------- 5. regressions
def standard(run):
    dg, v = load(run)
    t, act, valid, pose_valid = dg[:, 0], dg[:, 1], dg[:, 2], dg[:, 3]
    rng, stale = dg[:, 4], dg[:, 5]
    print(f'  {run:26s} valid {100*valid.mean():5.1f}%  pose_valid '
          f'{100*pose_valid.mean():5.1f}%  staleness {stale[valid>0].mean():.3f}s')
    bound = dg[:, 10]
    chk = (valid > 0) & np.isfinite(rng) & np.isfinite(bound)
    over = chk & (rng > bound)
    mono = []
    for a in np.where(np.diff(act) > 0)[0]:
        w = (t >= t[a] - 2.0) & (t <= t[a]) & (valid > 0) & np.isfinite(rng)
        if w.sum() >= 10:
            mono.append(float(np.mean(np.diff(rng[w]) < 0.05)))
    print(f'      over-bound {100*over.sum()/max(chk.sum(),1):.2f}%  crossings '
          f'{len(set(act.astype(int)))}  monotonicity median '
          f'{np.median(mono):.2f} min {min(mono):.2f}')


def pairs(run):
    p = os.path.join(RUNS, run, 'pairs.csv')
    if not os.path.exists(p):
        print(f'  {run}: no pairs.csv')
        return
    pr = np.loadtxt(p, delimiter=',')
    sep = pr[:, 1]
    med = float(np.median(sep))
    print(f'  {run:26s} separation median {med:.2f} m  MAD '
          f'{float(np.median(np.abs(sep - med))):.2f}  n={len(pr)}')


def main():
    runs = [r for r in (BEFORE, AFTER, 'lap-121520-NOMAP', 'lap-121520-NOPRIOR')
            if os.path.exists(os.path.join(RUNS, r, 'obs.npy'))]
    print('== 1. normal lateral sign flips within tracks ==')
    for r in runs:
        sign_flips(r)
    print('\n== 2. agreement with the map / race-bisector azimuth ==')
    for r in runs:
        map_agreement(r)
    print('\n== 3. normal_valid availability ==')
    for r in runs:
        availability(r)
    print('\n== 4. separation vs angular baseline ==')
    for r in runs:
        baseline_curve(r)
        approach_baseline(r)
    print('\n== 5. standard regression set ==')
    for r in runs:
        standard(r)
    print('\n== strafe 6-7 canary ==')
    for r in ('strafe67-BUGFIX3', 'strafe67-NORMFUSE'):
        if os.path.isdir(os.path.join(RUNS, r)):
            pairs(r)


if __name__ == '__main__':
    main()
