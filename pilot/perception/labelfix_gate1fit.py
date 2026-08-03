"""Mechanism C: fit ONE gate's FULL pose (position AND orientation) from pixel evidence.

WHY ORIENTATION. Triage finds gate 1 60% bad with quads SHEARED off a plainly visible
aperture at zero body rate, yet gatebias_measure.py declared it healthy (median centre
offset 1.51 px). Those two are consistent only if the error is an ORIENTATION error: a
sheared quad keeps its centre, and the bias measurement matched detector CENTRES of
detector-matched instances only -- exactly the statistic a rotation cannot move. So this
fit uses ALL FOUR CORNERS of detector inner quads across many viewpoints and solves for
the gate pose minimizing corner reprojection error through the pose stream.

CONTAMINATION GUARDS, because the fit input passes through the other two mechanisms:
  * body rate > RATE_MAX rad/s is skipped (mechanism A puts ~3.4 px of pose-interpolation
    error on every corner; the fit must not absorb it into orientation);
  * frames where a NEARER gate's solid board covers > OCCL_MAX of the instance are
    skipped (mechanism B would hand the fit the nearer gate's quad).

CORRESPONDENCE. Both the projected aperture and the detected inner quad are put through
autolabel.canon (image-space clockwise, index 0 = min(x+y)), the same deterministic rule
the labels themselves use, so corner i matches corner i with no search.

CONTROL. --target 0 runs the identical fit on gate 0 (triage N-rate 11%): its orientation
must come back essentially unchanged (< ~2 deg), else the fit itself is broken.

    python3 pilot/perception/labelfix_gate1fit.py --target 1
    python3 pilot/perception/labelfix_gate1fit.py --target 0     # control
    python3 pilot/perception/labelfix_gate1fit.py --target 1 --render

Writes labelfix_gate<N>_pose.json (+ labelfix_gate1_valid.png with --render).
Nothing here modifies gates_refined.json or any label file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402
import detect as D  # noqa: E402
from autolabel import canon  # noqa: E402
import labelfix_common as C  # noqa: E402

RATE_MAX = 0.5      # rad/s; mechanism A's error is ~1 px below 0.2, ~3.4 px above 1
OCCL_MAX = 0.20     # behind-gate solid coverage above this -> frame excluded from fit
BLOCK = 15          # consecutive-frame block size for the interleaved holdout

# ROTATION REGULARIZER, forced by the gate-0 control. Unregularized, gate 0 (triage
# N-rate 11%, no shear) came back rotated 12.4 deg with ZERO held-out improvement
# (1.36 -> 1.52 px median): near-head-on approach viewpoints leave the normal-tilt mode
# flat, and the optimizer wanders in the flat valley. The penalty (REG * rvec, px per
# rad) zeroes exactly those directions. Sized so it cannot eat a real signal: gate 1's
# measured improvement is worth ~2800 px^2 of cost while 7.7 deg of rotation costs
# (100 * 0.133)^2 = 177 px^2 -- a 16x margin; gate 0's flat-valley rotation buys ~nothing
# and now stays at zero.
REG = 100.0


def match_inner(dets, centre, size_px):
    """gatebias_measure.py's rule verbatim: nearest inner-source detection within
    size_px, size ratio in [0.5, 2]."""
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


def collect(target, centres, size_lo=12.0, size_hi=250.0):
    """(pose, det_quad_canon, key, t) per usable frame.

    Candidates come from scanning autolabels_vq1.json for the target gate directly, NOT
    from gatebias_offsets.csv: that CSV kept only unclipped 12-80 px instances, and the
    shear is worst exactly on the large oblique views whose (wrong) label runs off the
    image edge -- gate 1 has 345 clipped instances at 12-80 px plus 369 at 80-250 px that
    the CSV never saw. Selecting on the old label's clipped flag would condition the fit
    on the very error being fitted."""
    labels = json.load(open(os.path.join(HERE, 'autolabels_vq1.json')))
    cand = {}
    for key in sorted(labels):
        for i, d in enumerate(labels[key]):
            if d['gate'] == target and size_lo <= d['size_px'] <= size_hi:
                cand.setdefault(key, i)
    print(f'gate {target}: {len(cand)} candidate frames ({size_lo:.0f}-{size_hi:.0f} px, '
          f'clipped included)')

    rates = C.RateLookup()
    tracks, ftimes = {}, {}
    gates_pose = {gi: (c, None) for gi, c in centres.items()}
    obs = []
    n_rate = n_occl = n_nomatch = n_nopose = 0
    for key in sorted(cand):
        sess, fname = key.split('/', 1)
        sdir = os.path.join(C.SESSIONS, sess)
        if sess not in tracks:
            tracks[sess] = L.PoseTrack(sdir)
            ftimes[sess] = C.frame_times(sdir)
        t = ftimes[sess].get(fname)
        pose = tracks[sess].at_time(t) if t is not None else None
        if pose is None:
            n_nopose += 1
            continue
        r = rates.rate(key)
        if r is None or r > RATE_MAX:
            n_rate += 1
            continue
        inst = labels[key][cand[key]]
        uv_lbl = np.asarray(inst['corners'], np.float64)
        frac, _ = C.behind_gate_fraction(pose, gates_pose, target, uv_lbl)
        if frac > OCCL_MAX:
            n_occl += 1
            continue
        img = cv2.imread(os.path.join(sdir, 'frames', fname))
        if img is None:
            continue
        m = match_inner(D.detections(img), uv_lbl.mean(axis=0),
                        float(inst['size_px']))
        if m is None:
            n_nomatch += 1
            continue
        obs.append({'pose': pose, 'det': canon(m['quad']), 'key': key, 't': t})
    print(f'  usable {len(obs)}  (skipped: rate>{RATE_MAX} {n_rate}, behind-gate {n_occl},'
          f' no-match {n_nomatch}, no-pose {n_nopose})')
    return obs


def residuals(p, c0, obs):
    c = c0 + p[:3]
    pts = C.gate_corners(c, p[3:])
    res = []
    for o in obs:
        uv, z = L.project(pts, o['pose'])
        if np.any(z <= 0.1) or not np.all(np.isfinite(uv)):
            res.append(np.full(8, 50.0))
            continue
        res.append((canon(uv) - o['det']).ravel())
    res.append(REG * p[3:])
    return np.concatenate(res)


def corner_err(p, c0, obs):
    """Per-frame mean corner error, px."""
    out = []
    c = c0 + p[:3]
    pts = C.gate_corners(c, p[3:])
    for o in obs:
        uv, z = L.project(pts, o['pose'])
        if np.any(z <= 0.1):
            out.append(np.nan)
            continue
        out.append(float(np.linalg.norm(canon(uv) - o['det'], axis=1).mean()))
    return np.asarray(out)


def solve(obs, c0):
    r = least_squares(residuals, np.zeros(6), args=(c0, obs),
                      loss='soft_l1', f_scale=3.0, x_scale=[1, 1, 1, 0.05, 0.05, 0.05])
    return r.x


def describe(p, c0):
    rv = p[3:]
    Rg = C.gate_R(rv)
    ang = float(np.degrees(np.linalg.norm(rv)))
    n_new = Rg @ np.array([1.0, 0.0, 0.0])
    tilt = float(np.degrees(np.arccos(np.clip(n_new[0], -1, 1))))
    # in-plane: where the old +y (right) axis lands, projected back into the old plane
    y_new = Rg @ np.array([0.0, 1.0, 0.0])
    inplane = float(np.degrees(np.arctan2(y_new[2], y_new[1])))
    print(f'  centre delta [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}] m  '
          f'|{np.linalg.norm(p[:3]):.2f}| m')
    print(f'  rotation {ang:.2f} deg total; normal tilt {tilt:.2f} deg '
          f'(new normal [{n_new[0]:+.3f} {n_new[1]:+.3f} {n_new[2]:+.3f}]); '
          f'in-plane {inplane:+.2f} deg')
    return {'centre': (c0 + p[:3]).tolist(), 'centre_delta_m': p[:3].tolist(),
            'rvec': p[3:].tolist(), 'rotation_deg': ang, 'normal_tilt_deg': tilt,
            'normal_new': n_new.tolist(), 'inplane_deg': inplane}


def render_examples(target, c0, p, examples, outpath):
    labels = json.load(open(os.path.join(HERE, 'autolabels_vq1.json')))
    tracks, ftimes = {}, {}
    tiles = []
    pts = C.gate_corners(c0 + p[:3], p[3:])
    for key, idx in examples:
        sess, fname = key.split('/', 1)
        sdir = os.path.join(C.SESSIONS, sess)
        if sess not in tracks:
            tracks[sess] = L.PoseTrack(sdir)
            ftimes[sess] = C.frame_times(sdir)
        pose = tracks[sess].at_time(ftimes[sess][fname])
        img = cv2.imread(os.path.join(sdir, 'frames', fname))
        old = np.asarray(labels[key][idx]['corners'], np.float64)
        new, z = L.project(pts, pose)
        cv2.polylines(img, [old.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        cv2.polylines(img, [new.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 1)
        c = old.mean(axis=0)
        s = max(float(np.ptp(old[:, 0])), float(np.ptp(old[:, 1]))) * 3.0
        s = max(s, 100.0)
        a = 220.0 / s
        M = np.array([[a, 0, 110 - c[0] * a], [0, a, 110 - c[1] * a]], np.float32)
        t = cv2.warpAffine(img, M, (220, 220))
        cv2.putText(t, f'{fname}#{idx}', (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)
        tiles.append(cv2.resize(t, (440, 440), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(outpath, np.hstack(tiles))
    print(f'wrote {outpath}  (old label GREEN, refit projection YELLOW)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=1)
    ap.add_argument('--render', action='store_true')
    args = ap.parse_args()

    centres = C.load_refined_centres()
    c0 = centres[args.target]
    obs = collect(args.target, centres)
    if len(obs) < 30:
        print('too few observations; stopping')
        return

    # interleaved holdout: consecutive-time blocks, even -> fit, odd -> held out.
    obs.sort(key=lambda o: o['t'])
    train = [o for i, o in enumerate(obs) if (i // BLOCK) % 2 == 0]
    test = [o for i, o in enumerate(obs) if (i // BLOCK) % 2 == 1]
    p_tr = solve(train, c0)
    e_old = corner_err(np.zeros(6), c0, test)
    e_new = corner_err(p_tr, c0, test)
    print(f'\nHELD-OUT corner error (n={len(test)} frames, blocks of {BLOCK}):')
    print(f'  old pose: median {np.nanmedian(e_old):.2f} px  p90 '
          f'{np.nanpercentile(e_old, 90):.2f}')
    print(f'  new pose: median {np.nanmedian(e_new):.2f} px  p90 '
          f'{np.nanpercentile(e_new, 90):.2f}')

    # final pose from ALL usable frames
    p = solve(obs, c0)
    e_all_old = corner_err(np.zeros(6), c0, obs)
    e_all_new = corner_err(p, c0, obs)
    print(f'\nFINAL fit on all {len(obs)} frames:')
    print(f'  corner error old {np.nanmedian(e_all_old):.2f} -> new '
          f'{np.nanmedian(e_all_new):.2f} px (median)')
    desc = describe(p, c0)

    out = {
        'source': 'labelfix_gate1fit.py - full-pose fit from detector inner-quad corners',
        'gate': args.target,
        'old_centre': c0.tolist(),
        'n_frames': len(obs),
        'rate_max': RATE_MAX, 'occl_max': OCCL_MAX,
        'holdout': {'n_test': len(test),
                    'old_median_px': float(np.nanmedian(e_old)),
                    'new_median_px': float(np.nanmedian(e_new)),
                    'old_p90_px': float(np.nanpercentile(e_old, 90)),
                    'new_p90_px': float(np.nanpercentile(e_new, 90))},
        'all_frames': {'old_median_px': float(np.nanmedian(e_all_old)),
                       'new_median_px': float(np.nanmedian(e_all_new))},
        **desc,
    }
    path = os.path.join(HERE, f'labelfix_gate{args.target}_pose.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'wrote {path}')

    if args.render:
        render_examples(args.target, c0, p,
                        [('20260731-195307/00010658.jpg', 2),
                         ('20260731-195307/00010507.jpg', 0)],
                        os.path.join(HERE, 'labelfix_gate1_valid.png'))


if __name__ == '__main__':
    main()
