"""Per-gate image-space bias between auto-labels and independent detections.

Claire's triage: ONE gate's projected truth quad sits consistently off the real gate.
Hypothesis: that gate's surveyed 3D centre is wrong by a constant offset. Signature to
separate: a constant 3D error Delta at distance d projects to ~f*Delta/d px while apparent
size ~ f*1.5/d px, so offset/size = Delta/1.5 = CONSTANT and raw px offset SCALES WITH
SIZE. A constant raw-px offset instead points at something else (camera model, timing).

Method: for every instance in autolabels_vq1.json that is unclipped, visibility=="visible"
and 12-80 px, match detect.py's best INNER-source detection under the exact rule
compare_net_vs_detector.detector_match uses (dist <= size_px, 0.5 <= size ratio <= 2.0,
nearest). Offset = label centre - detection centre. Report per gate: n, median dx/dy,
median |offset|, IQR, direction consistency (resultant length of unit offset vectors),
median offset/size, and the offset-vs-size regression slope.

    python3 pilot/perception/gatebias_measure.py [--labels FILE] [--csv FILE]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detect as D  # noqa: E402

SESSIONS = os.path.join(os.path.dirname(HERE), 'sessions')
SIZE_LO, SIZE_HI = 12.0, 80.0


def match_inner(dets, centre, size_px):
    """compare_net_vs_detector.detector_match's rule, restricted to source=='inner'."""
    best, bd = None, 1e18
    for d in dets:
        if d['source'] != 'inner':
            continue
        dist = float(np.linalg.norm(np.asarray(d['centre'], np.float64) - centre))
        ratio = d['size_px'] / max(size_px, 1e-6)
        if dist > size_px or ratio > 2.0 or ratio < 0.5:
            continue
        if dist < bd:
            best, bd = d, dist
    return best


def collect(labels_path, csv_path):
    labels = json.load(open(labels_path))
    # group candidate instances by frame so each JPEG is decoded once
    frames = {}
    for key in sorted(labels):
        cand = []
        for i, d in enumerate(labels[key]):
            if d.get('clipped') or d.get('visibility') != 'visible':
                continue
            s = float(d.get('size_px', 0.0))
            if not (SIZE_LO <= s <= SIZE_HI):
                continue
            cand.append((i, d))
        if cand:
            frames[key] = cand

    rows = []
    n_img_missing = n_no_match = 0
    for k, key in enumerate(sorted(frames)):
        sess, fname = key.split('/', 1)
        path = os.path.join(SESSIONS, sess, 'frames', fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            n_img_missing += 1
            continue
        dets = D.detections(img)
        for i, d in frames[key]:
            c = np.asarray(d['corners'], np.float64).mean(axis=0)
            m = match_inner(dets, c, float(d['size_px']))
            if m is None:
                n_no_match += 1
                continue
            dc = np.asarray(m['centre'], np.float64)
            rows.append({
                'key': key, 'idx': i, 'session': sess, 'gate': int(d['gate']),
                'size_px': float(d['size_px']),
                'dx': float(c[0] - dc[0]), 'dy': float(c[1] - dc[1]),
                'det_size': float(m['size_px']),
            })
        if k % 500 == 0:
            print(f'  {k}/{len(frames)} frames, {len(rows)} matches', flush=True)

    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} matched instances over {len(frames)} candidate frames '
          f'({n_img_missing} unreadable frames, {n_no_match} instances with no '
          f'inner-source match); wrote {csv_path}')
    return rows


def table(rows, by=('gate',)):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    hdr = (f'{"group":>16} {"n":>5} {"med dx":>7} {"med dy":>7} {"med |off|":>9} '
           f'{"IQR":>7} {"dircons":>7} {"off/size":>8} {"slope":>7}')
    print(hdr)
    print('-' * len(hdr))
    for g in sorted(groups):
        rs = groups[g]
        dx = np.array([r['dx'] for r in rs])
        dy = np.array([r['dy'] for r in rs])
        sz = np.array([r['size_px'] for r in rs])
        mag = np.hypot(dx, dy)
        # direction consistency: resultant length of unit offset vectors, 0..1
        u = np.stack([dx, dy], 1) / np.maximum(mag, 1e-9)[:, None]
        dircons = float(np.linalg.norm(u.mean(axis=0)))
        # does |offset| scale with size? robust slope through origin
        slope = float(np.median(mag / sz))
        print(f'{str(g):>16} {len(rs):>5d} {np.median(dx):>+7.2f} {np.median(dy):>+7.2f} '
              f'{np.median(mag):>9.2f} '
              f'{np.percentile(mag, 75) - np.percentile(mag, 25):>7.2f} '
              f'{dircons:>7.2f} {np.median(mag / sz):>8.3f} {slope:>7.3f}')
    return groups


def scaling_check(rows, gate):
    """For the suspect gate: is the offset constant in px, or proportional to size?"""
    rs = [r for r in rows if r['gate'] == gate]
    sz = np.array([r['size_px'] for r in rs])
    mag = np.hypot([r['dx'] for r in rs], [r['dy'] for r in rs])
    print(f'\ngate {gate} offset vs size (n={len(rs)}):')
    for lo, hi in ((12, 20), (20, 30), (30, 45), (45, 60), (60, 80)):
        k = (sz >= lo) & (sz < hi)
        if k.sum() >= 5:
            print(f'  size {lo:>2d}-{hi:<2d}  n={int(k.sum()):>4d}  '
                  f'median |off| {np.median(mag[k]):5.2f} px   '
                  f'median |off|/size {np.median(mag[k] / sz[k]):.3f}')
    r = np.corrcoef(sz, mag)[0, 1] if len(rs) > 2 else float('nan')
    print(f'  corr(|off|, size) = {r:+.3f}  '
          f'(positive & off/size flat => constant 3D error; '
          f'flat |off| => constant px error)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=os.path.join(HERE, 'autolabels_vq1.json'))
    ap.add_argument('--csv', default=os.path.join(HERE, 'gatebias_offsets.csv'))
    ap.add_argument('--reuse', action='store_true', help='re-read --csv, skip the scan')
    args = ap.parse_args()

    if args.reuse and os.path.exists(args.csv):
        rows = []
        with open(args.csv) as fh:
            for r in csv.DictReader(fh):
                r['gate'] = int(r['gate'])
                for k in ('size_px', 'dx', 'dy', 'det_size'):
                    r[k] = float(r[k])
                rows.append(r)
        print(f'reusing {args.csv}: {len(rows)} rows')
    else:
        rows = collect(args.labels, args.csv)

    print('\nPER GATE (all sessions):')
    table(rows, ('gate',))
    print('\nPER GATE x SESSION:')
    table(rows, ('gate', 'session'))

    # worst gate by median |offset|
    gates = sorted({r['gate'] for r in rows})
    med = {g: np.median(np.hypot([r['dx'] for r in rows if r['gate'] == g],
                                 [r['dy'] for r in rows if r['gate'] == g]))
           for g in gates}
    worst = max(med, key=med.get)
    print(f'\nworst gate by median |offset|: {worst} ({med[worst]:.2f} px)')
    scaling_check(rows, worst)


if __name__ == '__main__':
    main()
