"""interiorgate2.py -- Bug 2, second pass: finer dark thresholds + unsure-aware loss.

interiorgate.py showed dark_frac(V<60) separates lit-ceiling real gates from solid deco
fills, but 18/57 of the bug population still fail -- nearly all `unsure` hand quads
(guessed corners on marginal ~20 px gates, per LABEL_POLICY). Here: (a) recompute with
dark cut at V<60/90/120, (b) report loss over CONFIDENT labels (unsure excluded) as the
headline, unsure separately, (c) sanity-check the chosen rule per size band.

    python3 pilot/perception/interiorgate2.py
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from interiorgate import interior_mask, collect_neg, HERE, HANDFRAMES, arr


def features2(hsv, quad):
    msk = interior_mask(hsv.shape, quad)
    n = int(msk.sum())
    if n < 12:
        return None
    px = hsv[msk.astype(bool)]
    S, V = px[:, 1].astype(np.float32), px[:, 2].astype(np.float32)
    return {
        'white_frac': float(np.mean((S < 70) & (V > 150))),
        'mean_v': float(V.mean()),
        'dark60': float(np.mean(V < 60)),
        'dark90': float(np.mean(V < 90)),
        'dark120': float(np.mean(V < 120)),
        'n_px': n,
    }


def collect_true2():
    raw = json.load(open(os.path.join(HERE, 'labels_gates_all.json')))
    rows = []
    for key, insts in sorted(raw.items()):
        img = cv2.imread(os.path.join(HANDFRAMES, key))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for inst in insts:
            f = features2(hsv, inst['corners'])
            if f is None:
                continue
            c = np.asarray(inst['corners'], np.float32)
            f.update(key=key, unsure=bool(inst.get('unsure')),
                     occluded=bool(inst.get('occluded')),
                     size_px=float(max(np.linalg.norm(c[(j + 1) % 4] - c[j])
                                       for j in range(4))))
            rows.append(f)
    return rows


def collect_neg2(reason='deco'):
    raw = json.load(open(os.path.join(HERE, 'conf_negatives_vq2.json')))
    sessions = os.path.join(os.path.dirname(HERE), 'sessions')
    rows = []
    for key, insts in sorted(raw.items()):
        sess, fname = key.split('/', 1)
        img = cv2.imread(os.path.join(sessions, sess, 'frames', fname))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for inst in insts:
            if inst.get('reason') != reason:
                continue
            f = features2(hsv, inst['corners'])
            if f is None:
                continue
            f.update(key=key, size_px=float(inst.get('size_px', 0.0)))
            rows.append(f)
    return rows


def confusion(name, sure, unsure, deco, fn):
    rs = np.array([fn(r) for r in sure]) if sure else np.array([])
    ru = np.array([fn(r) for r in unsure]) if unsure else np.array([])
    rd = np.array([fn(r) for r in deco])
    print(f'  {name:52s} deco rej {100*rd.mean():5.1f}%  sure lost '
          f'{100*rs.mean():5.2f}% ({int(rs.sum())}/{len(rs)})  unsure lost '
          f'{100*ru.mean():5.1f}% ({int(ru.sum())}/{len(ru)})')


def main():
    true_rows = collect_true2()
    deco = collect_neg2('deco')
    sure = [r for r in true_rows if not r['unsure']]
    unsure = [r for r in true_rows if r['unsure']]
    print(f'true: {len(sure)} sure + {len(unsure)} unsure; deco: {len(deco)}')

    bright = lambda r: r['white_frac'] > 0.15 or r['mean_v'] > 150.0     # noqa: E731
    print('\n== confusion (headline = SURE labels) ==')
    confusion('TODAY: bright', sure, unsure, deco, bright)
    for dk in ('dark60', 'dark90', 'dark120'):
        for b in (0.02, 0.05, 0.10):
            confusion(f'bright & {dk}<{b}', sure, unsure, deco,
                      lambda r, dk=dk, b=b: bright(r) and r[dk] < b)

    # chosen-rule candidates by size band (sure only)
    for name, fn in (
        ('bright & dark60<0.05', lambda r: bright(r) and r['dark60'] < 0.05),
        ('bright & dark90<0.05', lambda r: bright(r) and r['dark90'] < 0.05),
        ('bright & dark120<0.10', lambda r: bright(r) and r['dark120'] < 0.10),
    ):
        print(f'\n== {name}: loss by size band (sure) / deco rejection by size ==')
        for lo, hi in ((0, 20), (20, 40), (40, 80), (80, 1e9)):
            s = [r for r in sure if lo <= r['size_px'] < hi]
            d = [r for r in deco if lo <= r['size_px'] < hi]
            ls = 100 * np.mean([fn(r) for r in s]) if s else float('nan')
            rd = 100 * np.mean([fn(r) for r in d]) if d else float('nan')
            print(f'  {lo:.0f}-{hi:.0f}px: sure lost {ls:5.1f}% (n={len(s)})   '
                  f'deco rej {rd:5.1f}% (n={len(d)})')
        lost = [r for r in sure if fn(r)]
        for r in lost[:10]:
            print(f'    LOST {r["key"]} sz{r["size_px"]:.0f} w{r["white_frac"]:.2f} '
                  f'v{r["mean_v"]:.0f} d60={r["dark60"]:.3f} d90={r["dark90"]:.3f} '
                  f'd120={r["dark120"]:.3f}{" occl" if r["occluded"] else ""}')
        surv = [r for r in deco if not fn(r)]
        print(f'  deco survivors: {len(surv)}')
        for r in surv[:6]:
            print(f'    SURV {r["key"]} sz{r["size_px"]:.0f} w{r["white_frac"]:.2f} '
                  f'v{r["mean_v"]:.0f} d60={r["dark60"]:.3f} d120={r["dark120"]:.3f}')


if __name__ == '__main__':
    main()
