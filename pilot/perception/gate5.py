"""Recover the world centre of VQ1 race gate 5 -- the one gate with no truth.

THE HOLE. `vqual-1/pilot/gate_truth.json` and `gates_refined.json` both hold five centres,
keyed ONE-BASED, so key "1".."5" are race gates 0..4. Race gate 5 is the last gate of the
VQ1 course and was never CROSSED in either recording: both sessions end with
`active_gate_index == 5` still pending. `gate_truth.json` is built from crossing positions,
so gate 5 has none, and `refine.py` -- which bootstraps association from those priors --
had nothing to seed with and silently skipped it.

The cost is not cosmetic. `autolabel.py` emits 13008 instances from these two sessions and
labels gate 5 in none of them, so every frame that shows gate 5 carries a REAL, UNLABELLED
gate. Train a confidence head on that and it is taught that a genuine gate is background.

WHY IT IS RECOVERABLE WITHOUT A CROSSING. The aircraft flew TOWARD gate 5 with it in view
for ~15 s across the two sessions, and VQ1 streams full pose. Each detection gives a metric
gate pose in the body frame (PnP against the known 1500 mm aperture / 2700 mm outer), and
pose puts the camera in the world, so each detection is one vote:

    gate_world = camera_pos + R_body_to_world @ pos_body

which is exactly `refine.py`'s estimator. Nothing here re-derives a sign; the pose,
projection and the yaw negation all come from `label.py` / `CONVENTIONS.md`.

IDENTITY WITHOUT A PRIOR. `refine.py` associates a vote to the nearest prior centre. Gate 5
has no prior, so that door is shut. Instead identity comes from the sim itself:

  * only frames where `active_gate_index == 5` are used, so the sim is stating that gates
    0-4 are done and gate 5 is the target being flown at;
  * any vote landing within EXCLUDE_M of a KNOWN centre is thrown away, so an old gate
    still in view cannot masquerade as gate 5;
  * what survives is mode-sought, not averaged over: the remaining votes still contain
    ceiling lights, signage and merged-blob detections, and a mean would chase them.

The seed is the vote with the most neighbours inside SEED_R, then MEDIAN over votes inside
CLUSTER_R, three passes -- median for the same reason `refine.py` gives, PnP range has a
long tail and a mean is dragged by exactly the samples that are wrong.

SCATTER IS THE ANSWER, NOT A FOOTNOTE. The per-axis MAD of the surviving cluster and the
bootstrap SE of the median are both reported and both written to the JSON. A single number
with no spread would hide that the along-course axis (world x) is the ill-conditioned one:
the approach is near head-on, so range error projects almost entirely onto x.

THE HONEST TEST is `--validate`: run the identical pipeline on race gate 4, whose centre IS
known (`gates_refined.json` key "5"), pretending it is not -- same active-gate windows, same
exclusions against gates 0-3 only, same estimator -- and report the error against the known
value. Anything the method gets wrong on gate 5 it should also get wrong on gate 4.

    python3 pilot/perception/gate5.py <session-dir> [<session-dir> ...] [--out FILE]
    python3 pilot/perception/gate5.py <session-dir> ... --validate
    python3 pilot/perception/gate5.py <session-dir> ... --overlay DIR [--n 10]

Writes new files only. Nothing here modifies gates_refined.json, gate_truth.json,
autolabel.py, detect.py, label.py or refine.py.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REFINED = os.path.join(HERE, 'gates_refined.json')

EXCLUDE_M = 8.0     # a vote this close to a KNOWN centre is that gate, not the target
SEED_R = 3.0        # neighbourhood for picking the densest vote as the cluster seed
CLUSTER_R = 5.0     # votes inside this of the current estimate feed the median
N_PASSES = 3
MIN_VOTES = 8       # refine.py's floor; below it the fit is not worth reporting


# ---------------------------------------------------------------------------------------
# data


def load_known():
    """race index -> centre, from gates_refined.json. Keys there are ONE-BASED."""
    d = json.load(open(REFINED))['consensus']
    return {int(k) - 1: np.array([v['x'], v['y'], v['z']]) for k, v in d.items()}


def active_windows(session, target):
    """[(t0, t1), ...] wall-clock seconds where race.csv says `target` is the active gate.

    This is the sim's own statement of which gate is being flown at -- the same
    independent-referee role gate_truth.json played, but supplied in-band. Contiguous runs
    are merged; a session may contain more than one (195307 opens with gate 5 pending from
    an earlier race, then resets).
    """
    rows = L.load_csv(os.path.join(session, 'race.csv'))
    t = np.array([float(r['t_wall_ns']) for r in rows]) / 1e9
    g = np.array([int(r['active_gate_index']) for r in rows])
    wins, start = [], None
    for i in range(len(g)):
        if g[i] == target and start is None:
            start = t[i]
        elif g[i] != target and start is not None:
            wins.append((start, t[i - 1]))
            start = None
    if start is not None:
        wins.append((start, t[-1]))
    return [w for w in wins if w[1] > w[0]]


def collect(sessions, target, known, limit=4000):
    """World-frame votes for `target`, with the frames they came from.

    Returns (votes, meta) where meta[i] = (session, file, size_px, source, range_m).
    """
    votes, meta = [], []
    kn = list(known.values())
    n_frames = n_det = n_excl = 0
    for S in sessions:
        wins = active_windows(S, target)
        if not wins:
            continue
        track = L.PoseTrack(S)
        frames = [r for r in L.load_csv(os.path.join(S, 'frames.csv')) if r['file']]
        ft = np.array([float(r['sim_time_ns']) for r in frames]) / 1e9
        sel = [i for i in range(len(frames))
               if any(a <= ft[i] <= b for a, b in wins)]
        step = max(1, len(sel) // limit)
        for idx in sel[::step]:
            pose = track.at_time(ft[idx])
            if pose is None:
                continue
            img = cv2.imread(os.path.join(S, 'frames', frames[idx]['file']))
            if img is None:
                continue
            n_frames += 1
            p, roll, pitch, yaw = pose
            R_bw = L.euler_to_R(roll, pitch, yaw)
            for d in D.detections(img):
                w = p + R_bw @ d['pos_body']
                n_det += 1
                if any(np.linalg.norm(w - c) < EXCLUDE_M for c in kn):
                    n_excl += 1
                    continue
                votes.append(w)
                meta.append((S, frames[idx]['file'], d['size_px'], d['source'],
                             d['range_m']))
    stats = {'frames': n_frames, 'detections': n_det, 'excluded_known': n_excl,
             'votes': len(votes)}
    return (np.array(votes) if votes else np.zeros((0, 3))), meta, stats


# ---------------------------------------------------------------------------------------
# estimator


def mode_median(votes, avail, seed_r=SEED_R, cluster_r=CLUSTER_R, n_passes=N_PASSES):
    """Densest-neighbourhood seed within `avail`, then iterated median.

    -> (centre, inlier mask). A plain median over all surviving votes is not safe here:
    unlike refine.py there is no prior to associate against, so the vote set still contains
    ceiling lights, signage, merged-blob detections and OTHER GATES scattered arbitrarily.
    Seeding on density finds a place many independent viewpoints agree on, which is what a
    gate looks like in this vote set.
    """
    idx = np.flatnonzero(avail)
    if len(idx) == 0:
        return None, np.zeros(len(votes), bool)
    sub = votes[idx]
    dm = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    est = sub[int(np.argmax((dm < seed_r).sum(axis=1)))].copy()
    mask = np.zeros(len(votes), bool)
    for _ in range(n_passes):
        mask = avail & (np.linalg.norm(votes - est, axis=1) < cluster_r)
        if mask.sum() == 0:
            break
        est = np.median(votes[mask], axis=0)
    return est, mask


def clusters(votes, meta, max_k=6):
    """Greedy peel: densest cluster, remove it, repeat. -> list of dicts, densest first.

    THIS IS THE FIX THE VALIDATION FORCED. A single densest-cluster fit run on the gate-4
    window returns GATE 5, because gate 5 sits 24 m beyond gate 4 and stays small and
    well-inside the frame for the whole approach while gate 4 clips out of frame as it is
    neared. Density alone therefore does not identify the ACTIVE gate; it identifies the
    most-detected gate, which is a different thing. Selection below.
    """
    avail = np.ones(len(votes), bool)
    out = []
    for _ in range(max_k):
        est, mask = mode_median(votes, avail)
        if est is None or mask.sum() < MIN_VOTES:
            break
        rng = np.array([meta[i][4] for i in np.flatnonzero(mask)])
        out.append({'centre': est, 'mask': mask, 'n': int(mask.sum()),
                    'median_range': float(np.median(rng))})
        avail &= ~mask
    return out


def pick_active(cl):
    """Which cluster is the ACTIVE gate: the NEAREST one, not the most-voted one.

    The active gate is by definition the next gate to be flown THROUGH, so among the gates
    visible during its window it is the nearest. Density ranks the far ones first (they
    stay whole in frame for longer), which is exactly the failure the gate-4 leave-one-out
    caught. Clusters below FRAC of the largest are dropped first so a handful of stray
    votes on a near ceiling light cannot win on proximity alone.
    """
    if not cl:
        return None
    keep = [c for c in cl if c['n'] >= max(MIN_VOTES, 0.15 * max(x['n'] for x in cl))]
    return min(keep or cl, key=lambda c: c['median_range'])


def scatter(votes):
    """Per-axis spread of the inlier cluster, plus the bootstrap SE of the median."""
    med = np.median(votes, axis=0)
    mad = np.median(np.abs(votes - med), axis=0)
    rng = np.random.default_rng(0)
    boot = np.array([np.median(votes[rng.integers(0, len(votes), len(votes))], axis=0)
                     for _ in range(400)])
    return {
        'n': int(len(votes)),
        'median': med.tolist(),
        'mad': mad.tolist(),
        'std': votes.std(axis=0).tolist(),
        'p10': np.percentile(votes, 10, axis=0).tolist(),
        'p90': np.percentile(votes, 90, axis=0).tolist(),
        'bootstrap_se': boot.std(axis=0).tolist(),
        'radial_mad_m': float(np.median(np.linalg.norm(votes - med, axis=1))),
    }


def fit(sessions, target, known, limit=4000, verbose=True):
    votes, meta, st = collect(sessions, target, known, limit)
    if verbose:
        print(f'gate {target}: {st["frames"]} active-gate frames, {st["detections"]} '
              f'detections, {st["excluded_known"]} dropped as a known gate, '
              f'{st["votes"]} candidate votes')
    cl = clusters(votes, meta)
    if verbose:
        for i, c in enumerate(cl):
            print('    cluster %d  n=%4d  median range %5.1f m  '
                  '[%8.2f %7.2f %7.2f]' % (i, c['n'], c['median_range'], *c['centre']))
    best = pick_active(cl)
    if best is None or best['n'] < MIN_VOTES:
        if verbose:
            print('  no cluster above the vote floor')
        return None, None, st, votes, np.zeros(len(votes), bool), meta
    est, mask = best['centre'], best['mask']
    sc = scatter(votes[mask])
    if verbose:
        print(f'  centre [{est[0]:8.2f} {est[1]:7.2f} {est[2]:7.2f}]  '
              f'inliers {int(mask.sum())}/{len(votes)}')
        print('  MAD    [%8.2f %7.2f %7.2f] m   bootstrap SE [%.2f %.2f %.2f] m'
              % (*sc['mad'], *sc['bootstrap_se']))
    return est, sc, st, votes, mask, meta


# ---------------------------------------------------------------------------------------
# overlays


def overlay(sessions, centre, known, outdir, n=10, target=5):
    """Project the recovered centre back into real frames and write a contact sheet.

    Known gates are drawn too, in a different colour: if the recovered quad lands on a gate
    that a known quad is also on, the recovery is a duplicate, not a new gate.
    """
    os.makedirs(outdir, exist_ok=True)
    tiles = []
    per = max(1, n // max(len(sessions), 1))
    for S in sessions:
        wins = active_windows(S, target)
        if not wins:
            continue
        name = os.path.basename(os.path.normpath(S))
        track = L.PoseTrack(S)
        frames = [r for r in L.load_csv(os.path.join(S, 'frames.csv')) if r['file']]
        ft = np.array([float(r['sim_time_ns']) for r in frames]) / 1e9
        sel = [i for i in range(len(frames)) if any(a <= ft[i] <= b for a, b in wins)]
        if not sel:
            continue
        picks = [sel[int(round(k))] for k in np.linspace(0, len(sel) - 1, min(per, len(sel)))]
        for idx in picks:
            pose = track.at_time(ft[idx])
            img = cv2.imread(os.path.join(S, 'frames', frames[idx]['file']))
            if pose is None or img is None:
                continue
            for gi, c in sorted(known.items()):
                uv, z = L.project(corners(c, pose[0]), pose)
                if np.any(z <= 0.3):
                    continue
                cv2.polylines(img, [uv.astype(np.int32).reshape(-1, 1, 2)], True,
                              (255, 160, 0), 1)
            uv, z = L.project(corners(centre, pose[0]), pose)
            rng = float(np.linalg.norm(np.asarray(centre) - pose[0]))
            if np.all(z > 0.3):
                cv2.polylines(img, [uv.astype(np.int32).reshape(-1, 1, 2)], True,
                              (0, 0, 255), 2)
                c2 = uv.mean(axis=0).astype(int)
                cv2.putText(img, f'{target} {rng:.0f}m', (c2[0] + 6, c2[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(img, f'{name}/{frames[idx]["file"]}  r={rng:.1f}m', (5, 352),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(img)
    if not tiles:
        print('no overlay tiles')
        return None
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * L.H, cols * L.W, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * L.H:(r + 1) * L.H, c * L.W:(c + 1) * L.W] = t
    path = os.path.join(outdir, f'gate{target}_overlay.png')
    cv2.imwrite(path, sheet)
    print(f'wrote {path}  ({len(tiles)} tiles)')
    return path


def check(sessions, centre, outdir, target=5, crop=None):
    """Projected centre vs DETECTED centre, over every active-gate frame.

    The same cross-check `autolabel.verify` runs, and for the same reason: `detect.py` uses
    orange pixels alone -- no pose, no gate truth -- so agreement is evidence from outside
    every telemetry convention. The 2x size gate is not a fudge; without it a frame where
    the detector missed contributes the distance to some OTHER gate and the statistic
    measures recall instead of projection.
    """
    err, sizes, miss, tiles = [], [], 0, []
    for S in sessions:
        wins = active_windows(S, target)
        track = L.PoseTrack(S)
        frames = [r for r in L.load_csv(os.path.join(S, 'frames.csv')) if r['file']]
        ft = np.array([float(r['sim_time_ns']) for r in frames]) / 1e9
        for i in range(len(frames)):
            if not any(a <= ft[i] <= b for a, b in wins):
                continue
            pose = track.at_time(ft[i])
            if pose is None:
                continue
            uv, z = L.project(corners(centre, pose[0]), pose)
            if np.any(z <= 0.3):
                continue
            c = uv.mean(axis=0)
            sz = max(float(np.linalg.norm(uv[(j + 1) % 4] - uv[j])) for j in range(4))
            if not (0 <= c[0] < L.W and 0 <= c[1] < L.H) or sz < 12:
                continue
            img = cv2.imread(os.path.join(S, 'frames', frames[i]['file']))
            if img is None:
                continue
            det = D.detections(img)
            cand = [d for d in det if sz / 2 < d['size_px'] < sz * 2]
            if not cand:
                miss += 1
                continue
            d = min(cand, key=lambda dd: np.linalg.norm(dd['centre'] - c))
            err.append(float(np.linalg.norm(d['centre'] - c)))
            sizes.append(sz)
            if crop and len(tiles) < crop:
                x0, y0 = int(max(0, c[0] - 70)), int(max(0, c[1] - 70))
                cv2.polylines(img, [uv.astype(np.int32).reshape(-1, 1, 2)], True,
                              (0, 0, 255), 1)
                t = img[y0:y0 + 140, x0:x0 + 140]
                pad = np.zeros((140, 140, 3), np.uint8)
                pad[:t.shape[0], :t.shape[1]] = t
                tiles.append(cv2.resize(pad, (280, 280), interpolation=cv2.INTER_NEAREST))
    if not err:
        print('cross-check: no comparable frames')
        return
    e, s = np.array(err), np.array(sizes)
    print(f'detector cross-check on gate {target}: n={len(e)} size-consistent, '
          f'{miss} frames with no size-consistent detection')
    print(f'  centre error median {np.median(e):.1f} px  p75 {np.percentile(e, 75):.1f}  '
          f'p90 {np.percentile(e, 90):.1f}   relative {np.median(e / s):.3f} of gate width')
    if tiles and outdir:
        os.makedirs(outdir, exist_ok=True)
        cols = min(5, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * 280, cols * 280, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c2 = divmod(i, cols)
            sheet[r * 280:(r + 1) * 280, c2 * 280:(c2 + 1) * 280] = t
        p = os.path.join(outdir, f'gate{target}_zoom.png')
        cv2.imwrite(p, sheet)
        print(f'  wrote {p}')


def corners(centre, cam_pos):
    """Inner-aperture corners, view-dependent winding -- autolabel.py's rule, restated
    rather than imported so this file cannot perturb the labelling path."""
    cx, cy, cz = (float(v) for v in centre)
    s = 1.0 if (cx - float(cam_pos[0])) >= 0.0 else -1.0
    h = L.HALF
    return np.array([[cx, cy - s * h, cz - h], [cx, cy + s * h, cz - h],
                     [cx, cy + s * h, cz + h], [cx, cy - s * h, cz + h]])


# ---------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--target', type=int, default=5, help='race index to recover')
    ap.add_argument('--limit', type=int, default=4000)
    ap.add_argument('--out', default=os.path.join(HERE, 'gate5_recovered.json'))
    ap.add_argument('--validate', action='store_true',
                    help='also re-run on gate 4, whose centre is known, and report error')
    ap.add_argument('--overlay', metavar='DIR')
    ap.add_argument('--n', type=int, default=10)
    args = ap.parse_args()

    known = load_known()
    print(f'known centres (race indices): {sorted(known)}')

    result = {'source': 'gate5.py - active-gate detection fusion through the pose stream',
              'sessions': [os.path.basename(os.path.normpath(s)) for s in args.sessions],
              'note': ('centre for VQ1 race gate 5, which has no crossing position. '
                       'RACE-INDEXED, unlike gates_refined.json whose keys are one-based.')}

    if args.validate:
        # leave-one-out on gate 4: hide its centre, recover it the same way, compare.
        truth4 = known[4]
        k3 = {k: v for k, v in known.items() if k != 4}
        e4, sc4, st4, v4, m4, meta4 = fit(args.sessions, 4, k3, args.limit)
        if e4 is not None:
            err = e4 - truth4
            print(f'  KNOWN  [{truth4[0]:8.2f} {truth4[1]:7.2f} {truth4[2]:7.2f}]')
            print('  ERROR  [%+8.2f %+7.2f %+7.2f]  |%.2f| m'
                  % (*err, np.linalg.norm(err)))
            result['validation_gate4'] = {
                'recovered': e4.tolist(), 'known': truth4.tolist(),
                'error': err.tolist(), 'error_norm_m': float(np.linalg.norm(err)),
                'scatter': sc4, 'stats': st4}
            # SECOND, INDEPENDENT MEASUREMENT OF GATE 5. The gate-4 window's densest
            # cluster is gate 5 itself -- it sits 24 m beyond and stays whole in frame
            # while gate 4 clips out. Different frames, different ranges, no overlap with
            # the gate-5 window, so the disagreement between the two is a real error bar
            # in a way the within-cluster scatter is not.
            cl4 = clusters(v4, meta4)
            far = max(cl4, key=lambda c: c['n']) if cl4 else None
            if far is not None:
                result['corroboration_from_gate4_window'] = {
                    'centre': far['centre'].tolist(), 'n': far['n'],
                    'median_range_m': far['median_range']}
                print(f'  gate-4 window\'s densest cluster (= gate 5, seen ahead): '
                      f'[{far["centre"][0]:8.2f} {far["centre"][1]:7.2f} '
                      f'{far["centre"][2]:7.2f}]  n={far["n"]}')

    est, sc, st, votes, mask, meta = fit(args.sessions, args.target, known, args.limit)
    if est is None:
        print('no fit')
        return

    # sanity: where does it sit relative to the flown course?
    d4 = est - known[4]
    print(f'\nrelative to race gate 4: [{d4[0]:+.2f} {d4[1]:+.2f} {d4[2]:+.2f}]  '
          f'|{np.linalg.norm(d4):.2f}| m')
    spac = [float(np.linalg.norm(known[i + 1] - known[i])) for i in range(4)]
    print('gate-to-gate spacing 0..4: ' + ' '.join(f'{s:.1f}' for s in spac) +
          f'   -> 4..{args.target}: {np.linalg.norm(d4):.1f} m')
    rr = [m[4] for m, k in zip(meta, mask) if k]
    print(f'inlier PnP ranges: min {min(rr):.1f}  median {np.median(rr):.1f}  '
          f'max {max(rr):.1f} m   sources: '
          f'{sum(1 for m, k in zip(meta, mask) if k and m[3] == "inner")} inner / '
          f'{sum(1 for m, k in zip(meta, mask) if k and m[3] == "outer")} outer')

    result['gate'] = args.target
    # gates_refined.json / gate_truth.json are ONE-BASED; merging this in means key 6.
    result['gates_refined_key'] = str(args.target + 1)
    result['centre'] = {'x': float(est[0]), 'y': float(est[1]), 'z': float(est[2])}
    result['scatter'] = sc
    result['stats'] = st
    result['relative_to_gate4'] = d4.tolist()
    # HONEST UNCERTAINTY. `scatter` is NOT it. Votes come from consecutive frames of one
    # approach, so they are heavily correlated, and PnP range bias is systematic rather
    # than random -- both make the bootstrap SE (centimetres) an underestimate by roughly
    # an order of magnitude. The two figures that are not self-referential are the gate-4
    # leave-one-out error and the disagreement between two independent windows.
    unc = {'why': 'bootstrap SE understates: correlated frames + systematic PnP range bias',
           'within_cluster_mad_m': sc['mad']}
    if 'validation_gate4' in result:
        unc['leave_one_out_gate4_m'] = result['validation_gate4']['error_norm_m']
    if 'corroboration_from_gate4_window' in result:
        c2 = np.array(result['corroboration_from_gate4_window']['centre'])
        unc['independent_window_disagreement_m'] = float(np.linalg.norm(c2 - est))
    unc['use_this_m'] = max([v for k, v in unc.items() if k.endswith('_m')
                             and isinstance(v, float)] or [0.0])
    result['uncertainty'] = unc
    print(f'\nhonest uncertainty: {unc["use_this_m"]:.2f} m '
          f'(NOT the {np.linalg.norm(sc["bootstrap_se"]):.2f} m bootstrap SE)')
    with open(args.out, 'w') as fh:
        json.dump(result, fh, indent=1)
    print(f'\nwrote {args.out}')

    if args.overlay:
        overlay(args.sessions, est, known, args.overlay, args.n, args.target)
        check(args.sessions, est, args.overlay, args.target, crop=10)


if __name__ == '__main__':
    main()
