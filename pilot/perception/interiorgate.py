"""interiorgate.py -- derive an interior-colour decoration discriminator that survives
the LIT-CEILING background (Bug 2).

The deployed strict test (producer.IN_WHITE_MAX=0.15 / IN_V_MAX=150 on the 70%-shrunk
quad interior) assumes a real aperture opens onto something DARK. Claire's live frame
falsified that: a real gate seen from below opens onto bright trusses/lights/signage
("interior w0.17 v250" -> rejected). Decoration interiors are UNIFORMLY bright (solid
white wordmark fill, V~245); real apertures against a lit background still contain dark
patches (sky between trusses).

Populations:
  * TRUE gates: labels_gates_all.json (Claire's VQ2 hand labels, incl. upward views).
  * DECO: conf_negatives_vq2.json, reason 'deco' (438 quads across the same sessions).

Features on the shrunk interior (producer.interior_colour's 0.7 polygon):
  white_frac  mean(S<70 & V>150)          [today's]
  mean_v      mean(V)                      [today's]
  dark_frac   mean(V<60)                   candidate: trusses/sky gaps survive lighting
  blob_frac   largest connected bright component (V>=200) / interior area
              candidate: uniformity -- a solid fill is ONE big blob

Target (stricter than before, we now know the failure mode):
  reject >= 95% of deco, lose <= 2% of true gates.

    python3 pilot/perception/interiorgate.py
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(os.path.dirname(HERE), 'sessions')
HANDFRAMES = os.path.join(HERE, 'vq2_label', 'frames')


def interior_mask(shape, quad, shrink=0.7):
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    ctr = q.mean(0)
    poly = (ctr + (q - ctr) * shrink).astype(np.int32)
    msk = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(msk, [poly], 1)
    return msk


def features(hsv, quad):
    msk = interior_mask(hsv.shape, quad)
    n = int(msk.sum())
    if n < 12:
        return None
    px = hsv[msk.astype(bool)]
    S, V = px[:, 1].astype(np.float32), px[:, 2].astype(np.float32)
    white_frac = float(np.mean((S < 70) & (V > 150)))
    mean_v = float(V.mean())
    dark_frac = float(np.mean(V < 60))
    bright = ((hsv[:, :, 2] >= 200) & msk.astype(bool)).astype(np.uint8)
    nb = int(bright.sum())
    if nb == 0:
        blob_frac = 0.0
    else:
        ncc, lab = cv2.connectedComponents(bright)
        blob = max(np.bincount(lab[lab > 0]).max() if ncc > 1 else 0, 0)
        blob_frac = float(blob) / n
    return {'white_frac': white_frac, 'mean_v': mean_v, 'dark_frac': dark_frac,
            'blob_frac': blob_frac, 'n_px': n}


def collect_true():
    raw = json.load(open(os.path.join(HERE, 'labels_gates_all.json')))
    rows = []
    for key, insts in sorted(raw.items()):
        img = cv2.imread(os.path.join(HANDFRAMES, key))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for inst in insts:
            f = features(hsv, inst['corners'])
            if f is None:
                continue
            f.update(key=key, unsure=bool(inst.get('unsure')),
                     occluded=bool(inst.get('occluded')),
                     clipped=bool(inst.get('clipped')))
            c = np.asarray(inst['corners'], np.float32)
            f['size_px'] = float(max(np.linalg.norm(c[(j + 1) % 4] - c[j])
                                     for j in range(4)))
            rows.append(f)
    return rows


def collect_neg(reason='deco'):
    raw = json.load(open(os.path.join(HERE, 'conf_negatives_vq2.json')))
    rows = []
    for key, insts in sorted(raw.items()):
        sess, fname = key.split('/', 1)
        img = cv2.imread(os.path.join(SESSIONS, sess, 'frames', fname))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for inst in insts:
            if inst.get('reason') != reason:
                continue
            f = features(hsv, inst['corners'])
            if f is None:
                continue
            f.update(key=key, size_px=float(inst.get('size_px', 0.0)))
            rows.append(f)
    return rows


def arr(rows, k):
    return np.array([r[k] for r in rows])


def show_dist(tag, rows):
    print(f'\n== {tag} (n={len(rows)}) ==')
    for k in ('white_frac', 'mean_v', 'dark_frac', 'blob_frac'):
        a = arr(rows, k)
        print(f'  {k:11s} p10 {np.percentile(a,10):7.3f}  med {np.median(a):7.3f}  '
              f'p90 {np.percentile(a,90):7.3f}  p98 {np.percentile(a,98):7.3f}')


def confusion(name, rows_true, rows_deco, fn):
    rt = np.array([fn(r) for r in rows_true])
    rd = np.array([fn(r) for r in rows_deco])
    print(f'  {name:58s} deco rejected {100*rd.mean():5.1f}%   '
          f'true lost {100*rt.mean():5.2f}% ({rt.sum()}/{len(rt)})')
    return rt, rd


def main():
    true_rows = collect_true()
    deco_rows = collect_neg('deco')
    empty_rows = collect_neg('empty-random')
    json.dump({'true': true_rows, 'deco': deco_rows, 'empty': empty_rows},
              open(os.path.join(HERE, 'interiorgate_features.json'), 'w'))

    show_dist('TRUE gates (hand labels, all)', true_rows)
    bright_true = [r for r in true_rows if r['mean_v'] > 150 or r['white_frac'] > 0.15]
    show_dist('TRUE gates REJECTED by today\'s strict rule (the bug population)',
              bright_true)
    show_dist('DECO negatives', deco_rows)
    show_dist('empty-random negatives', empty_rows)

    print('\n== rules: reject-as-deco confusion (true gates vs deco) ==')
    today = lambda r: r['white_frac'] > 0.15 or r['mean_v'] > 150.0        # noqa: E731
    confusion("TODAY: white>0.15 or v>150", true_rows, deco_rows, today)
    confusion("TODAY current-slot variant: white>0.60", true_rows, deco_rows,
              lambda r: r['white_frac'] > 0.60)
    for b in (0.02, 0.03, 0.05, 0.08):
        confusion(f'dark_frac < {b} & (white>0.15 or v>150)', true_rows, deco_rows,
                  lambda r, b=b: r['dark_frac'] < b and
                  (r['white_frac'] > 0.15 or r['mean_v'] > 150.0))
    for c in (0.5, 0.6, 0.7, 0.8):
        confusion(f'blob_frac > {c}', true_rows, deco_rows,
                  lambda r, c=c: r['blob_frac'] > c)
    for b in (0.02, 0.03, 0.05):
        for c in (0.5, 0.6, 0.7):
            confusion(f'dark<{b} & blob>{c}', true_rows, deco_rows,
                      lambda r, b=b, c=c: r['dark_frac'] < b and r['blob_frac'] > c)
    for b in (0.02, 0.03, 0.05, 0.08, 0.10):
        confusion(f'dark_frac < {b} & white>0.15', true_rows, deco_rows,
                  lambda r, b=b: r['dark_frac'] < b and r['white_frac'] > 0.15)

    # what the shortlisted rules do to the bug population specifically
    print('\n== shortlist on the BUG population (true gates today\'s rule rejects) ==')
    for name, fn in (
        ("dark<0.03 & (white>0.15 or v>150)",
         lambda r: r['dark_frac'] < 0.03 and (r['white_frac'] > 0.15
                                              or r['mean_v'] > 150.0)),
        ("dark<0.05 & blob>0.6",
         lambda r: r['dark_frac'] < 0.05 and r['blob_frac'] > 0.6),
    ):
        lost = [r for r in bright_true if fn(r)]
        print(f'  {name:44s} still lost: {len(lost)}/{len(bright_true)}')
        for r in lost[:8]:
            print(f'    {r["key"]} size {r["size_px"]:.0f} w{r["white_frac"]:.2f} '
                  f'v{r["mean_v"]:.0f} d{r["dark_frac"]:.3f} b{r["blob_frac"]:.2f}'
                  f'{" unsure" if r["unsure"] else ""}{" occl" if r["occluded"] else ""}')
    # false-accept side: which deco survive
    print('\n== deco that SURVIVE the shortlisted rules ==')
    for name, fn in (
        ("dark<0.03 & (white>0.15 or v>150)",
         lambda r: r['dark_frac'] < 0.03 and (r['white_frac'] > 0.15
                                              or r['mean_v'] > 150.0)),
    ):
        surv = [r for r in deco_rows if not fn(r)]
        print(f'  {name}: {len(surv)} survivors')
        for r in surv[:10]:
            print(f'    {r["key"]} size {r["size_px"]:.0f} w{r["white_frac"]:.2f} '
                  f'v{r["mean_v"]:.0f} d{r["dark_frac"]:.3f} b{r["blob_frac"]:.2f}')


if __name__ == '__main__':
    main()
