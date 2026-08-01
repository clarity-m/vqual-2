"""Pick the most informative frames for hand-labelling gate identity, then solve for
METRIC SCALE and the VERTICAL PROFILE the top-down sketch cannot carry.

WHY THIS IS POSSIBLE AT ALL. readmap.py's assignment-free scale fit is degenerate: it never
learns which gate is which, so 136 sketch pair-distances absorb any scale. That is a
limitation of THAT estimator, not of the data. Every detection already carries a metric 3-D
position from the known 1500 mm aperture, so a co-visible pair IS a metric distance
measurement -- it only lacks identity. Hand-labelling a few pairs supplies exactly the
missing ingredient, and one labelled pair already pins the scale.

VERTICAL COMES FREE WITH THE SAME LABELS, and does not need a new flight either. The height
difference between two co-visible gates needs only the direction of gravity, which
HIGHRES_IMU gives directly. It is invariant to yaw -- which is the whole reason the frame
problem does not bite here -- so no attitude estimate and no pose stream is involved.

    # 1. choose frames worth labelling, and write a stub to fill in
    python3 labelgates.py <session> --pick 12 --outdir label/
    # 2. fill label/labels.json:  "<frame>": {"<det index>": <race index>, ...}
    # 3. solve
    python3 labelgates.py <session> --solve label/labels.json --map map_approx.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402
import detect as D  # noqa: E402

MIN_SIZE_PX = 26.0   # below this the PnP range is too soft to anchor a scale


def drop_decorations_by_parent(img, det, range_ratio=1.6):
    """Reject decoration false positives using the ORANGE BLOB each hole sits in.

    Supersedes drop_decorations(), which compared a detection against other DETECTIONS and
    so needed the near gate to have been detected. It often is not: a gate clipped by the
    image edge has a broken orange ring, its aperture stops being an enclosed hole, and
    detect.py rejects it as 'not-a-hole' -- leaving the decorations with no near anchor to
    be measured against. That is exactly the frame Claire flagged.

    The parent blob is always there, detected or not. A hole's implied range comes from its
    own size; the parent's comes from the parent's, via the 2700 mm outer boundary
    (range = 864 / parent_px). A checkerboard square implying 68 m inside a blob implying
    5 m is decoration, and the contradiction does not care whether the gate itself passed.
    """
    m = D.orange_mask(img)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return det
    hier = hier[0]
    parents = [(c, i) for i, c in enumerate(cnts) if hier[i][3] < 0]
    children = {}
    for i, c in enumerate(cnts):
        if hier[i][3] >= 0:
            children.setdefault(hier[i][3], []).append(c)
    keep = []
    for d in det:
        r = float(np.linalg.norm(d['pos_body']))
        pt = (float(d['centre'][0]), float(d['centre'][1]))
        drop = False
        for c, i in parents:
            if cv2.pointPolygonTest(c, pt, False) < 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            p_px = float(max(w, h))
            if p_px < 1:
                continue
            r_parent = 864.0 / p_px       # 480 * (2700/1500), the outer-boundary range
            if r <= r_parent * range_ratio:
                continue
            # SEEN THROUGH THE APERTURE, not painted on the frame. A far gate visible
            # through a near gate is geometrically inside that gate's OUTER contour, so the
            # range contradiction fires on it too -- and those are precisely the
            # long-baseline pairs worth labelling. A hole big enough to contain the
            # detection, and not the detection itself, means we are looking through.
            through = False
            for ch in children.get(i, []):
                if cv2.pointPolygonTest(ch, pt, False) < 0:
                    continue
                _cx, _cy, cw, chh = cv2.boundingRect(ch)
                if max(cw, chh) > 1.8 * d['size_px']:
                    through = True
                    break
            if not through:
                drop = True
                break
        if not drop:
            keep.append(d)
    return keep


def drop_decorations(det, range_ratio=1.25):
    """Reject false positives raised by a near gate's own graphics.

    SUPERSEDED by drop_decorations_by_parent(); kept because it is the cheaper test and
    still correct when the near gate was itself detected.

    A gate filling the frame shows white-on-orange decoration -- the AI-GP wordmark, the
    checkerboard strips -- whose contours pass the quad fit. They are small, so `range =
    480 / gate_px` places them FAR, while they sit inside the image footprint of a gate that
    is near. That is physically impossible: nothing 16 m away can be inside a gate 5 m away.

    So the test is the contradiction itself, not appearance: a detection whose centre lies
    within another detection's box AND whose range is meaningfully greater is discarded.
    Kept here rather than pushed into detect.py, which the whole pipeline depends on and
    which deserves its own held-out check before being changed.
    """
    keep = []
    for i, d in enumerate(det):
        ri = float(np.linalg.norm(d['pos_body']))
        nested = False
        for j, o in enumerate(det):
            if i == j:
                continue
            ro = float(np.linalg.norm(o['pos_body']))
            # Decoration sits on the gate's FRAME, outside the aperture, so the footprint
            # to test against is the 2700 mm outer boundary -- 1.8x the 1500 mm inner span
            # that size_px measures, hence a half-extent of 0.9 * size_px.
            half = o['size_px'] * 0.9
            inside = (abs(d['centre'][0] - o['centre'][0]) < half and
                      abs(d['centre'][1] - o['centre'][1]) < half)
            if inside and ri > ro * range_ratio:
                nested = True
                break
        if not nested:
            keep.append(d)
    return keep


def gravity_at(imu_rows, t_ns):
    """Unit gravity in BODY frame at the given time, from the accelerometer.

    Valid because the aircraft is not accelerating hard during a slow pass; the residual
    specific force is dominated by gravity. Checked by magnitude: a sample far from 9.8
    m/s^2 is manoeuvring and gets rejected by the caller rather than quietly biasing a
    height.
    """
    ts = np.array([float(r['t_wall_ns']) for r in imu_rows])
    i = int(np.argmin(np.abs(ts - t_ns)))
    r = imu_rows[i]
    a = np.array([float(r['xacc']), float(r['yacc']), float(r['zacc'])])
    n = float(np.linalg.norm(a))
    # Accelerometer measures specific force: at rest it reads -g (upward). Gravity DOWN is
    # therefore -a, normalised.
    return -a / max(n, 1e-9), n, abs(ts[i] - t_ns) / 1e9


def score_frames(session, npick, stride):
    """Frames where labelling buys the most: two-plus big gates, widely separated.

    Baseline length is what pins a scale -- two gates 40 m apart constrain it far better
    than two 8 m apart at the same relative error -- so the score rewards separation, and
    requires both gates be near enough for PnP to be trusted.
    """
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    out = []
    for r in frames[::stride]:
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is None:
            continue
        det = drop_decorations_by_parent(img, [d for d in D.detections(img)
                                             if d['size_px'] >= MIN_SIZE_PX])
        if len(det) < 2:
            continue
        P = np.array([d['pos_body'] for d in det])
        sep = float(np.max(np.linalg.norm(P[:, None] - P[None, :], axis=2)))
        small = float(min(d['size_px'] for d in det))
        out.append({'file': r['file'], 't_ns': float(r['t_recv_wall_ns']),
                    'n': len(det), 'sep_m': sep, 'min_size_px': small,
                    'score': sep * np.sqrt(small)})
    out.sort(key=lambda x: -x['score'])
    return out[:npick]


def pick(session, npick, stride, outdir):
    os.makedirs(outdir, exist_ok=True)
    picks = score_frames(session, npick, stride)
    stub = {}
    for p in picks:
        img = cv2.imread(os.path.join(session, 'frames', p['file']))
        det = drop_decorations_by_parent(img, [d for d in D.detections(img)
                                             if d['size_px'] >= MIN_SIZE_PX])
        det.sort(key=lambda d: -d['size_px'])
        for k, d in enumerate(det):
            c = d['centre'].astype(int)
            s = int(max(8, d['size_px'] / 2))
            cv2.rectangle(img, (c[0] - s, c[1] - s), (c[0] + s, c[1] + s), (0, 220, 255), 2)
            cv2.putText(img, str(k), (c[0] - s, c[1] - s - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(img, '%.0fm' % np.linalg.norm(d['pos_body']), (c[0] - s, c[1] + s + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(outdir, p['file'].replace('.jpg', '.png')),
                    cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST))
        stub[p['file']] = {str(k): None for k in range(len(det))}
        print('%s  %d gates, baseline %.1f m, smallest %.0f px'
              % (p['file'], p['n'], p['sep_m'], p['min_size_px']))
    # NEVER CLOBBER HAND WORK. An earlier version overwrote labels.json on every --pick,
    # and re-running the picker destroyed labels Claire had already entered -- the one file
    # here that cannot be regenerated from the recordings. Existing entries win over the
    # fresh stub, always, and anything already labelled survives even if the frame is no
    # longer among the picks.
    path = os.path.join(outdir, 'labels.json')
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception as e:
            backup = path + '.corrupt'
            os.replace(path, backup)
            print(f'existing {path} is not valid JSON ({e}); moved to {backup} rather than '
                  'overwriting it')
            old = {}
        for fname, mapping in old.items():
            kept = {k: v for k, v in mapping.items() if v is not None}
            if not kept:
                continue
            stub.setdefault(fname, {})
            stub[fname].update(kept)
        n = sum(len(v) for v in old.values() if isinstance(v, dict))
        print(f'merged {n} existing entries from {path}')
    json.dump(stub, open(path, 'w'), indent=1)
    print(f'\n{len(picks)} frames -> {outdir}')
    print(f'fill {path}: map each detection index to its RACE INDEX (0-16), null to skip.')
    print('Only pairs where BOTH are labelled are used, so partial labelling is fine.')


def solve(session, labels_json, map_json):
    m = json.load(open(map_json))
    k = m['row_separation_px'] / m['station_pitch_px']
    by_race = {g['race_index']: g for g in m['gates']}
    U = {}
    for a in by_race:
        for b in by_race:
            if a < b:
                pa, pb = by_race[a], by_race[b]
                U[(a, b)] = float(np.hypot(pa['along_station'] - pb['along_station'],
                                           (pa['across'] - pb['across']) * k))

    labels = json.load(open(labels_json))
    imu = list(L.load_csv(os.path.join(session, 'imu.csv')))
    rows, heights = [], {}
    for fname, mapping in labels.items():
        used = {int(i): v for i, v in mapping.items() if v is not None}
        if len(used) < 2:
            continue
        img = cv2.imread(os.path.join(session, 'frames', fname))
        if img is None:
            continue
        det = drop_decorations_by_parent(img, [d for d in D.detections(img)
                                             if d['size_px'] >= MIN_SIZE_PX])
        det.sort(key=lambda d: -d['size_px'])
        frow = next((r for r in L.load_csv(os.path.join(session, 'frames.csv'))
                     if r['file'] == fname), None)
        g_body, gmag, dt = gravity_at(imu, float(frow['t_recv_wall_ns'])) if frow else (None, 0, 9)
        idxs = sorted(used)
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                i, j = idxs[ii], idxs[jj]
                if i >= len(det) or j >= len(det):
                    continue
                ra, rb = used[i], used[j]
                if ra == rb:
                    continue
                pa, pb = np.array(det[i]['pos_body']), np.array(det[j]['pos_body'])
                d_m = float(np.linalg.norm(pb - pa))
                u = U[(min(ra, rb), max(ra, rb))]
                rows.append({'frame': fname, 'pair': (ra, rb), 'd_m': d_m, 'u': u,
                             'scale': d_m / u if u > 1e-6 else np.nan})
                # Height: project the inter-gate vector onto gravity. Yaw-free.
                if g_body is not None and abs(gmag - 9.81) < 1.5 and dt < 0.2:
                    dh = -float((pb - pa) @ g_body)     # +ve => rb is HIGHER than ra
                    heights.setdefault((min(ra, rb), max(ra, rb)), []).append(
                        dh if ra < rb else -dh)

    if not rows:
        raise SystemExit('no labelled pairs found — fill in labels.json first')
    s = np.array([r['scale'] for r in rows if np.isfinite(r['scale'])])
    print(f'{len(rows)} labelled pairs from {len(set(r["frame"] for r in rows))} frames\n')
    for r in sorted(rows, key=lambda r: -r['d_m']):
        print('  gates %2d-%-2d  measured %6.1f m   sketch %5.2f u   -> %5.2f m/station   %s'
              % (r['pair'][0], r['pair'][1], r['d_m'], r['u'], r['scale'], r['frame']))
    print(f'\nSCALE  median {np.median(s):.2f} m/station   '
          f'p10 {np.percentile(s, 10):.2f}  p90 {np.percentile(s, 90):.2f}   n={s.size}')
    print('  spread across pairs is the honest error bar: these are independent '
          'measurements\n  of one constant, so their disagreement IS the uncertainty.')
    if heights:
        print('\nHEIGHT DIFFERENCES (metres, +ve = second gate higher), gravity-referenced:')
        for (a, b), v in sorted(heights.items()):
            print('  gates %2d-%-2d  %+6.2f m  (n=%d, spread %.2f)'
                  % (a, b, float(np.median(v)), len(v), float(np.std(v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--pick', type=int, default=0)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--outdir', default='label')
    ap.add_argument('--solve', default='')
    ap.add_argument('--map', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'map_approx.json'))
    args = ap.parse_args()
    if args.pick:
        pick(args.session, args.pick, args.stride, args.outdir)
    if args.solve:
        solve(args.session, args.solve, args.map)


if __name__ == '__main__':
    main()
