"""Intent-anchored pair measurement for the targeting session -- identity-free.

WHY THIS EXISTS. The targeting session (20260801-155701) crosses gates 0-13, so the
crossing-anchored identity machinery runs -- and rendering its marker frames shows it
FAILING there in the pausing-lap way, but worse: long drifting hovers let a named
fragment migrate to a neighbouring gate without ever co-occurring with a true-named
track, so the per-gate contradiction pruning cannot see it (marker 6: a 'g7' fragment
sits on a 19 m gate while the real gate 7 fills the frame; marker 1: 'g2' at 17 m with
the 2->3 crossing 7 s out). Track identity from THIS session is therefore refused for
map rows.

What the session carries instead is Claire's WORD per marker (1 = 2+3, 2-4 = 6+7,
5-8 = 7+8, 9-12 = 9+10, 13-15 = 13+14) -- and the FIRST version of this file learned,
by rendering, that her word alone is not enough either: "both gates fully in frame"
does not mean both gates MEASURABLE. At the 7+8 markers the aircraft hovers metres from
gate 7, which fails every measurement path (too big, clipped, unstable), so the naive
"exactly two measurements = the intended pair" rule silently measured gates 8+9 and
called it 7-8. The tell that unravelled it: 11.04 (pausing 7-8) + 15.2 ("7-8" here)
= 26.2 (fast-lap "7-8") -- gates 7, 8, 9 are nearly collinear and every mode was the
same line under a different naming.

So measurement identity here is CLUSTER RANK, with refusals:

  * within one marker window, measurements chain into mini-tracks (3D continuity);
  * mini-tracks sorted by range; the nearest is the ACTIVE gate k, then k+1, k+2 --
    UNLESS a huge (>=120 px, i.e. under ~4 m) raw detection sits unmeasured in the
    window, in which case the active gate is that blob and labels start at k+1;
  * a window is REFUSED outright if it yields a labeled pair that contradicts an
    already-measured pair by more than 3 m (the components referee the labels), if it
    has more than three mini-tracks (ambiguous), or if a mini-track's range wobbles
    (MAD > 2.5 m -- a static gate seen from a hover does not wobble);
  * rows carry |dz| only (which gate is which within the pair adds no sign here);
    dz is NaN like strafe rows.

Measurements themselves: contour detections passing mapvq2.clean() PLUS gatenet on
refused (clipped/outer) detections, gated by PnP residual, jitter stability, size,
range agreement, non-decoration interior.

    python3 pilot/perception/mapedges_intent.py          # report + montage tiles
"""

from __future__ import annotations

import collections
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mapvq2 as M                                    # noqa: E402
import mapedges as ME                                 # noqa: E402

PAD_S = 1.5
MAX_R = 45.0
MIN_SEED_PX = 8.0
HUGE_PX = 120.0            # a raw det this big is a gate the aircraft is basically at
CHAIN_M = 2.5              # 3D continuity for mini-tracks inside a hover window
TRACK_MIN_OBS = 3
TRACK_MAX_MAD = 2.5
KNOWN_TOL = 3.0

ACTIVE = {1: 2}
ACTIVE.update({k: 6 for k in (2, 3, 4)})
ACTIVE.update({k: 7 for k in (5, 6, 7, 8)})
ACTIVE.update({k: 9 for k in (9, 10, 11, 12)})
ACTIVE.update({k: 13 for k in (13, 14, 15)})

# PER-WINDOW AUDIT, from reading the rendered frames (vercheck/intent_pairs_montage.png,
# vercheck/targeting_markers.png) against ribbon order and the known pair medians. The
# rank heuristic alone got two windows wrong and could not know it; these entries record
# what the pictures settled, exactly as mapvq2.EXCLUDE records session facts.
#   'refuse'          window's labeling is unverifiable -> contributes nothing
#   {'labels': [...]} explicit gate label per mini-track, in increasing-range order;
#                     a REPEATED label marks range-split fragments of ONE gate
AUDIT = {
    1: 'refuse',    # 5 mini-tracks; near gate identity (2 vs 3) unresolved on frames
    5: 'refuse',    # 4 mini-tracks, huge gate 7 partially measured -- mixed labels
    6: 'refuse',
    7: 'refuse',
    # marker 8 render (ip_1 bottom): aircraft hovers AT gate 7 (huge frame, top,
    # ribbon), measured tracks are the 16-17 m ribbon gate (= 8) and the ~30 m one
    # (= 9), which the tracker split into two range fragments. The rank heuristic
    # labeled them 7/8/9 because the huge-blob test missed (gate 7's fragments each
    # fall under the size threshold when clipped).
    8: {'labels': [8, 9, 9]},
    # marker 12 render (ip_2 bottom): a huge clipped gate (9) sits unmeasured mid-left;
    # the measured 16/24/31 m tracks would be 10/11/12, but labeling them by rank about
    # an invisible anchor is guesswork -- and (10,11)=12.2 m contradicts the known
    # 10-11 = 33.9 m under EITHER labeling. Unverifiable.
    12: 'refuse',
    # marker 14 render (ip_3 top): gate 13 at ~13 m (ribbon loops around Station 13),
    # ONE far gate at 21-22 m split into two range fragments. 14-15 = 32.9 m rules out
    # the two far tracks being distinct gates 14 and 15.
    14: {'labels': [13, 14, 14]},
    15: 'refuse',   # 10 mini-tracks, mixed vantages (ip_3 mid/bottom show mixed pairs)
}


def frame_measurements(S, i, net, dev, img, stats):
    """All trustworthy gate measurements in one frame: clean contour + gated net."""
    meas = []
    for d in S.det.get(S.frames[i]['file'], []):
        if M.clean(d):
            meas.append({'pos': d['pos_body'], 'r': d['range_m'],
                         'size': d['size_px'], 'src': 'contour',
                         'quad': np.asarray(d['quad'], float).reshape(4, 2),
                         'seed': None})
            continue
        if d.get('in_white', 0.0) > 0.15 or d.get('in_v', 0.0) > 150.0:
            continue
        if d['size_px'] < MIN_SEED_PX:
            continue
        m = ME.net_measure(net, dev, img, d)
        if not m['ok']:
            stats['net-' + m['why']] += 1
            continue
        if m['resid'] > ME.RESID_PX:
            stats['net-resid'] += 1
            continue
        meas.append({'pos': m['pos_body'], 'r': m['range_m'], 'size': m['size_px'],
                     'src': 'gatenet', 'quad': m['quad'], 'resid': m['resid'],
                     'seed': np.asarray(d['quad'], float).reshape(4, 2)})
    meas.sort(key=lambda m: (m['src'] != 'contour', -m['size']))
    kept = []
    for m in meas:
        c = m['quad'].mean(0)
        dup = False
        for k in kept:
            if (np.linalg.norm(c - k['quad'].mean(0)) < 0.5 * max(m['size'], k['size'])
                    and abs(m['r'] - k['r']) < 5.0):
                dup = True
                break
        if not dup:
            kept.append(m)
    return kept


def window_tracks(frames_meas):
    """[(i, [meas])] -> mini-tracks by 3D continuity. Body-frame positions drift as the
    aircraft yaws, so continuity is frame-to-frame, not global."""
    tracks = []
    for i, ms in frames_meas:
        for m in ms:
            best, bd = None, CHAIN_M
            for t in tracks:
                if i - t['last_i'] > 15:
                    continue
                d3 = float(np.linalg.norm(m['pos'] - t['last_pos']))
                if d3 < bd:
                    best, bd = t, d3
            if best is None:
                best = {'obs': [], 'last_i': i, 'last_pos': m['pos']}
                tracks.append(best)
            best['obs'].append((i, m))
            best['last_i'], best['last_pos'] = i, m['pos']
    return [t for t in tracks if len(t['obs']) >= TRACK_MIN_OBS]


def huge_unmeasured(S, frames_meas):
    """Is there, in a good share of the window, a near-huge raw detection that no
    measured track accounts for? That blob is the ACTIVE gate, hovered at and
    unmeasurable, and rank labels must start one later."""
    n_frames = max(len(frames_meas), 1)
    n_huge = 0
    for i, ms in frames_meas:
        for d in S.det.get(S.frames[i]['file'], []):
            if d['size_px'] < HUGE_PX:
                continue
            c = np.asarray(d['quad'], float).reshape(4, 2).mean(0)
            if any(np.linalg.norm(c - m['quad'].mean(0)) < 0.6 * d['size_px']
                   for m in ms):
                continue
            n_huge += 1
            break
    return n_huge / n_frames > 0.3


def collect(net=None, dev=None, known=None, verbose=True):
    """-> (rows, tiles, stats). rows are mapvq2-shaped; pair identity = cluster rank
    anchored on Claire's per-marker active gate, refereed by `known` pair medians."""
    dev = dev or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net or ME.load_net(dev)
    S = M.Session(ME.TARGETING)
    known = dict(known or {})
    stats = collections.Counter()
    rows, tiles = [], []
    for mi, (t, _g) in enumerate(S.markers, 1):
        k0 = ACTIVE.get(mi)
        if k0 is None:
            continue
        lo = int(np.searchsorted(S.t, t - PAD_S * 1e9))
        hi = int(np.searchsorted(S.t, t + PAD_S * 1e9))
        frames_meas = []
        for i in range(lo, min(hi, len(S.frames))):
            fpath = os.path.join(S.path, 'frames', S.frames[i]['file'])
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                continue
            ms = [m for m in frame_measurements(S, i, net, dev, img, stats)
                  if m['r'] <= MAX_R]
            frames_meas.append((i, ms))

        tracks = window_tracks(frames_meas)
        for t2 in tracks:
            rr = np.array([m['r'] for _i, m in t2['obs']])
            t2['r_med'] = float(np.median(rr))
            t2['r_mad'] = float(np.median(np.abs(rr - np.median(rr))))
        tracks = [t2 for t2 in tracks if t2['r_mad'] <= TRACK_MAX_MAD]
        tracks.sort(key=lambda t2: t2['r_med'])
        if not tracks:
            stats['window-no-tracks'] += 1
            continue
        audit = AUDIT.get(mi)
        if audit == 'refuse':
            stats['window-audit-refused'] += 1
            if verbose:
                print(f'  marker {mi:2d}: AUDIT-REFUSED ({len(tracks)} mini-tracks at '
                      + ', '.join(f'{t2["r_med"]:.0f}m' for t2 in tracks) + ')')
            continue
        if isinstance(audit, dict):
            if len(audit['labels']) != len(tracks):
                stats['window-audit-mismatch'] += 1
                if verbose:
                    print(f'  marker {mi:2d}: AUDIT-MISMATCH, expected '
                          f'{len(audit["labels"])} tracks, found {len(tracks)} at '
                          + ', '.join(f'{t2["r_med"]:.0f}m' for t2 in tracks))
                continue
            label = {id(t2): audit['labels'][j] for j, t2 in enumerate(tracks)}
            shift = 'audit'
        else:
            if len(tracks) > 3:
                stats['window-too-many-tracks'] += 1
                if verbose:
                    print(f'  marker {mi:2d}: REFUSED, {len(tracks)} mini-tracks')
                continue
            shift = 1 if huge_unmeasured(S, frames_meas) else 0
            label = {id(t2): k0 + shift + j for j, t2 in enumerate(tracks)}
        # candidate rows
        byobs = {}
        for t2 in tracks:
            for i, m in t2['obs']:
                byobs.setdefault(i, []).append((label[id(t2)], m))
        cand = []
        for i, lm in byobs.items():
            gh = S.gravity(i)
            for x in range(len(lm)):
                for y in range(x + 1, len(lm)):
                    (ga, ma), (gb, mb) = lm[x], lm[y]
                    if ga == gb:
                        continue
                    if ga > gb:
                        ga, gb, ma, mb = gb, ga, mb, ma
                    v = mb['pos'] - ma['pos']
                    d = float(np.linalg.norm(v))
                    if d < 3.0:
                        stats['pair-sub-3m'] += 1
                        continue
                    adz = abs(float(v @ gh)) if gh is not None else np.nan
                    h = float(np.sqrt(max(d * d - adz * adz, 0.0))) \
                        if gh is not None else np.nan
                    cand.append({'i': i, 'fid': int(S.fid[i]),
                                 'pair': (('g', ga), ('g', gb)), 'd': d,
                                 'dz': np.nan, 'h': h, 'abs_dz': adz, 'mk': True,
                                 'ra': float(ma['r']), 'rb': float(mb['r']),
                                 'src': 'intent-' + ('net' if 'gatenet' in
                                                     (ma['src'], mb['src'])
                                                     else 'contour'),
                                 'marker': mi, 'session': S.name,
                                 '_meas': (ma, mb)})
        # the components referee the labels: any labeled pair (n>=3) contradicting a
        # known median by > KNOWN_TOL condemns the WHOLE window's labeling
        med = collections.defaultdict(list)
        for r in cand:
            med[(r['pair'][0][1], r['pair'][1][1])].append(r['d'])
        bad = None
        for p, ds in med.items():
            if len(ds) >= 3 and p in known and \
                    abs(float(np.median(ds)) - known[p]) > KNOWN_TOL:
                bad = (p, float(np.median(ds)), known[p])
                break
        if bad is not None:
            stats['window-refused-known'] += 1
            if verbose:
                print(f'  marker {mi:2d}: REFUSED, labeled pair {bad[0][0]}-{bad[0][1]} '
                      f'reads {bad[1]:.1f} m vs known {bad[2]:.1f} m '
                      f'(shift={shift}, tracks at '
                      + ', '.join(f'{t2["r_med"]:.0f}m' for t2 in tracks) + ')')
            continue
        if verbose:
            print(f'  marker {mi:2d} (active {k0}, shift {shift}): tracks at '
                  + ', '.join(f'{t2["r_med"]:.0f}m->g{label[id(t2)]}' for t2 in tracks)
                  + f'  rows {len(cand)}')
        for r in cand:
            ma, mb = r.pop('_meas')
            rows.append(r)
            tiles.append({'i': r['i'], 'fid': r['fid'],
                          'pair': (r['pair'][0][1], r['pair'][1][1]), 'd': r['d'],
                          'meas': [ma, mb], 'marker': mi,
                          'path': os.path.join(S.path, 'frames',
                                               S.frames[r['i']]['file'])})
        stats['rows'] += len(cand)
    if verbose:
        print('[intent] ' + '  '.join(f'{k}={v}' for k, v in sorted(stats.items())))
        per = collections.defaultdict(list)
        for r in rows:
            per[(r['pair'][0][1], r['pair'][1][1])].append(r)
        for p in sorted(per):
            ds = np.array([r['d'] for r in per[p]])
            hh = np.array([r['h'] for r in per[p] if np.isfinite(r['h'])])
            zz = np.array([r['abs_dz'] for r in per[p] if np.isfinite(r['abs_dz'])])
            nn = sum(1 for r in per[p] if r['src'] == 'intent-net')
            print('  %2d-%-2d n=%3d (net-involved %3d)  d med %6.2f MAD %5.2f  '
                  'p10-p90 %5.1f-%-5.1f  horiz %6.2f  |dz| %5.2f'
                  % (p[0], p[1], len(ds), nn, np.median(ds),
                     np.median(np.abs(ds - np.median(ds))),
                     np.percentile(ds, 10), np.percentile(ds, 90),
                     np.median(hh) if len(hh) else float('nan'),
                     np.median(zz) if len(zz) else float('nan')))
    return rows, tiles, stats


def montage(tiles, out=os.path.join(HERE, 'vercheck', 'intent_pairs_montage.png')):
    """A few full frames per labeled pair: green = the two accepted quads, orange =
    seed contour where the measurement came from gatenet."""
    bypair = collections.defaultdict(list)
    for t in tiles:
        bypair[t['pair']].append(t)
    outs = []
    for p, ts in sorted(bypair.items()):
        ts.sort(key=lambda t: t['fid'])
        sel = [ts[int(k)] for k in np.linspace(0, len(ts) - 1, min(3, len(ts)))]
        seen = set()
        for t in sel:
            if t['fid'] in seen:
                continue
            seen.add(t['fid'])
            img = cv2.imread(t['path'], cv2.IMREAD_COLOR)
            if img is None:
                continue
            big = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_NEAREST)
            for g_lab, m in zip(t['pair'], t['meas']):
                if m['seed'] is not None:
                    cv2.polylines(big, [(m['seed'] * 2).astype(np.int32)], True,
                                  (60, 170, 255), 1, cv2.LINE_AA)
                q = (m['quad'] * 2).astype(np.int32)
                cv2.polylines(big, [q], True, (90, 230, 90), 2, cv2.LINE_AA)
                c = q.mean(0)
                cv2.putText(big, 'g%s %.0fm %s' % (g_lab, m['r'], m['src'][:3]),
                            (int(c[0]) - 40, int(np.clip(c[1], 16, 700))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 230, 90), 2,
                            cv2.LINE_AA)
            cv2.putText(big, 'pair %s-%s  marker %s  fid %d  d=%.2f m'
                        % (*t['pair'], t['marker'], t['fid'], t['d']),
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            outs.append(big)
    if outs:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, np.vstack(outs))
        print('wrote %s (%d frames)' % (out, len(outs)))


if __name__ == '__main__':
    # standalone run uses the already-established pair medians as the referee
    KNOWN = {(0, 1): 20.12, (1, 2): 8.32, (3, 4): 21.72, (4, 5): 22.46, (4, 6): 22.74,
             (5, 6): 15.44, (10, 11): 33.93, (11, 12): 19.48, (14, 15): 32.89,
             (15, 16): 20.93}
    rows, tiles, _ = collect(known=KNOWN)
    json.dump([{**r, 'pair': [list(r['pair'][0]), list(r['pair'][1])]} for r in rows],
              open(os.path.join(HERE, 'mapedges_intent_rows.json'), 'w'), indent=1)
    print('wrote mapedges_intent_rows.json (%d rows)' % len(rows))
    montage(tiles)
