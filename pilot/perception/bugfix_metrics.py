"""bugfix_metrics.py -- before/after checks for the 2026-08-02 wrong-gate attention bug.

Compares two full-lap replay runs of producer.py on 20260801-121520-vq2-lap-0-15:

  producer_runs/lap-121520-BEFORE   (pre-fix snapshot)
  producer_runs/lap-121520-AFTER    (fixed producer)

Checks:
  1. Over-bound metric, SAME static rule for both runs (the in-code ratcheted bound only
     exists after the fix): current-slot range vs the map-consistent bound
     (active==0 -> the 0-1 edge plain; active==k -> 1.3 x edge(k-1, k)).
  2. Ribbon anti-parallel metric from obs.npy: whenever attention == RIBBON, its target
     bearing must not be >120 deg (levelled-yaw corrected) from the bearing at which the
     current gate is next actually measured within 1.5 s.
  3. Strip renders from replay.mp4: opening (act=0), act=3 (wordmark regime), act=7
     (tiny-far-lock regime), plus handoffs into gates 5 and 12.

Usage: python3 pilot/perception/bugfix_metrics.py
"""

import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'producer_runs')
MAP = json.load(open(os.path.join(HERE, 'map_vq2.json')))

EDGE = {}
for k, v in MAP['measured_pairs'].items():
    i, j = sorted(int(x) for x in k.split('-'))
    EDGE[(i, j)] = float(v['dist_m'])
POS = MAP['layout_directions_2026_08_02']['positions_m']


def edge(i, j):
    key = (min(i, j), max(i, j))
    if key in EDGE:
        return EDGE[key]
    a, b = POS[str(i)], POS[str(j)]
    return math.dist((a['x'], a['y'], a['z_up_m']), (b['x'], b['y'], b['z_up_m']))


def static_bound(active):
    a = int(active)
    if a == 0:
        return edge(0, 1)              # start: farther than the whole 0-1 edge = gate 1
    return 1.3 * edge(a - 1, a)


def load_diag(run):
    d = np.loadtxt(os.path.join(RUNS, run, 'diag.csv'), delimiter=',')
    return d


def overbound(run):
    dg = load_diag(run)
    t, act, valid, rng = dg[:, 0], dg[:, 1], dg[:, 2], dg[:, 4]
    b = np.array([static_bound(a) for a in act])
    chk = (valid > 0) & np.isfinite(rng)
    over = chk & (rng > b)
    print(f'  {run}: static-rule over-bound {100 * over.sum() / max(chk.sum(), 1):.2f}% '
          f'({int(over.sum())}/{int(chk.sum())} checkable frames)')
    # where
    if over.sum():
        acts = sorted(set(int(a) for a in act[over]))
        print(f'    offending active indices: {acts}')
        i0 = int(np.argmax(over))
        print(f'    first: t={t[i0]:.2f}s act={int(act[i0])} range={rng[i0]:.1f} '
              f'bound={b[i0]:.1f}')
    return over


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ribbon_metric(run, dt=1.0 / 30.0, horizon_s=1.5):
    """obs layout: gate0 pos v[0:3], valid v[6], staleness v[10]; gyro v[48:51];
    roll v[54] pitch v[55]; attn dir v[66:69]; kind onehot v[69:73] (RIBBON = v[72])."""
    v = np.load(os.path.join(RUNS, run, 'obs.npy'))
    t = load_diag(run)[:, 0]
    n = len(v)
    # levelled yaw rate per frame: psi_dot = g_unit . omega
    g = np.stack([-np.sin(v[:, 55]),
                  np.sin(v[:, 54]) * np.cos(v[:, 55]),
                  np.cos(v[:, 54]) * np.cos(v[:, 55])], 1)
    psid = np.einsum('ij,ij->i', g, v[:, 48:51])
    dts = np.diff(t, prepend=t[0])
    psi = np.cumsum(psid * dts)         # rad, nose-right positive
    ribbon = v[:, 72] > 0.5
    fresh = (v[:, 6] > 0.5) & (v[:, 10] == 0.0)
    n_rib = int(ribbon.sum())
    n_chk = n_bad = 0
    H = int(round(horizon_s / dt))
    for i in np.nonzero(ribbon)[0]:
        b_rib = math.atan2(v[i, 67], v[i, 66])
        j_ok = None
        for j in range(i + 1, min(i + H, n)):
            if fresh[j]:
                j_ok = j
                break
        if j_ok is None:
            continue
        b_gate = math.atan2(v[j_ok, 1], v[j_ok, 0])
        b_gate_at_i = b_gate + (psi[j_ok] - psi[i])   # undo body yaw between i and j
        n_chk += 1
        if abs(wrap(b_rib - b_gate_at_i)) > math.radians(120.0):
            n_bad += 1
    print(f'  {run}: RIBBON frames {n_rib}; checkable (gate re-measured within '
          f'{horizon_s}s) {n_chk}; anti-parallel (>120 deg) {n_bad}'
          f'  ({100.0 * n_bad / max(n_chk, 1):.1f}%)')


def normal_on_tiny(run):
    v = np.load(os.path.join(RUNS, run, 'obs.npy'))
    dg = load_diag(run)
    rng = dg[:, 4]
    nv = v[:, 7] > 0.5                      # gate0 normal_valid flag
    tiny = np.isfinite(rng) & (rng > 24.0)  # 480/20px = 24 m: sub-20 px apparent
    both = nv & tiny
    print(f'  {run}: current-slot normal_valid on sub-20px (range>24 m): '
          f'{int(both.sum())} frames')


def tiles(run, picks, out_png):
    cap = cv2.VideoCapture(os.path.join(RUNS, run, 'replay.mp4'))
    dg = load_diag(run)
    ims = []
    for label, idx in picks:
        if idx is None or idx >= len(dg):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, im = cap.read()
        if not ok:
            continue
        cv2.putText(im, label, (6, im.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
        r = dg[idx]
        cv2.putText(im, f't={r[0]:.1f}s act={int(r[1])} rng={r[4]:.1f}m',
                    (6, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        ims.append(im)
    cap.release()
    if not ims:
        print(f'  !! no tiles for {out_png}')
        return
    H, W = ims[0].shape[:2]
    cols = min(3, len(ims))
    rows = (len(ims) + cols - 1) // cols
    sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = im
    cv2.imwrite(os.path.join(RUNS, out_png), sheet)
    print(f'  wrote {out_png} ({len(ims)} tiles)')


def block_picks(dg, act_value, offsets):
    """Frame indices at the start of the first contiguous active==act_value block."""
    idx = np.nonzero(dg[:, 1] == act_value)[0]
    if len(idx) == 0:
        return []
    start = idx[0]
    gaps = np.nonzero(np.diff(idx) > 1)[0]
    end = idx[gaps[0]] if len(gaps) else idx[-1]
    span = max(int(end - start), 1)
    return [(f'act={act_value}', int(start + min(o, span))) for o in offsets]


def render(run, tag):
    dg = load_diag(run)
    picks = block_picks(dg, 0, [0, 10, 25, 45])
    tiles(run, picks, f'bugfix_opening_{tag}.png')
    picks = block_picks(dg, 3, [2, 12, 30, 60]) + block_picks(dg, 7, [2, 12, 30, 60])
    tiles(run, picks, f'bugfix_act3_act7_{tag}.png')
    picks = block_picks(dg, 5, [1, 8]) + block_picks(dg, 12, [1, 8]) \
        + block_picks(dg, 15, [1, 8])
    tiles(run, picks, f'bugfix_handoffs_{tag}.png')


def main():
    before, after = 'lap-121520-BEFORE', 'lap-121520-AFTER'
    print('== over-bound metric (identical static rule both runs) ==')
    overbound(before)
    overbound(after)
    print('== ribbon anti-parallel metric ==')
    ribbon_metric(before)
    ribbon_metric(after)
    print('== normal_valid on tiny/far current gate ==')
    normal_on_tiny(before)
    normal_on_tiny(after)
    print('== strips ==')
    render(before, 'before')
    render(after, 'after')


if __name__ == '__main__':
    main()
