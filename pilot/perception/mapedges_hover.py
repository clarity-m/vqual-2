"""Intent-anchored pair rows for the 2026-08-01 evening link laps' marker hovers.

WHY A THIRD IDENTITY PATH. Both link laps have crossings, so mapvq2's machinery runs on
them natively -- but the staged pairs sit exactly where that machinery is weakest: at the
hover moments the near gate is clipped (never votes, never measures) and the far gate is
named, if at all, by a BACKWARD EXTENSION that this evening was caught sliding between
gates four separate times (mapvq2.REFUSE_ROWS). So, exactly as mapedges_intent.py did for
the targeting session: measurement is identity-free (contour clean() + gatenet on refused
detections, chained into mini-tracks inside each window), and IDENTITY per mini-track is
Claire's stated intent per marker, cross-checked on rendered frames and refereed by the
known pair medians. Windows whose scene contradicts the intent are AUDIT-REFUSED.

WHAT THE RENDERS SETTLED (vercheck/hover_windows.png, hover_pairs_montage.png,
markers_202110*.png, markers_202923*.png, zoom_*.png):

  * 202110 markers 1-4 staged "6+7" -- but gate 7 was ~40 m out and out of frame at every
    one of them (its verified approach track reads 36.2 m ONE second after the gate-6
    crossing, entering from the right image edge: rowcheck_g7approach.png). The 21-31 m
    partner behind gate 6 is GATE 5 -- their measured separation reproduces the known
    5-6 to half a metre. All four windows REFUSED for 6-7; no 6-7 measurement exists.
  * 202110 markers 5-6 staged "13+14" and the scene agrees: gate 13 (crossing-anchored,
    ribbon-threaded, at the Station 13 column) plus one far gate on the ribbon = 14.
  * 202923 markers staged "12+13" or "12+14". The crossing-anchored g12 fragments in the
    hover windows are vote-based with a funnel made toothless by the 165 s to the
    crossing, so g12 is treated as intent like everything else and every window's
    labeling is settled by its render.

    python3 pilot/perception/mapedges_hover.py           # survey: tracks + renders only
    python3 pilot/perception/mapedges_hover.py --rows    # apply AUDIT, emit + dump rows
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mapvq2 as M                                     # noqa: E402
import mapedges as ME                                  # noqa: E402
import mapedges_intent as MI                           # noqa: E402

PAD_S = 2.0            # stable-ish hovers with slight drift (Claire): wide windows
MAX_R = 50.0
KNOWN_TOL = 3.0

# (session, marker index 1-based) -> staged pair, from Claire's session notes.
INTENT = {
    ('20260801-202110-vq2-23-67-1314', 1): (6, 7),
    ('20260801-202110-vq2-23-67-1314', 2): (6, 7),
    ('20260801-202110-vq2-23-67-1314', 3): (6, 7),
    ('20260801-202110-vq2-23-67-1314', 4): (6, 7),
    ('20260801-202110-vq2-23-67-1314', 5): (13, 14),
    ('20260801-202110-vq2-23-67-1314', 6): (13, 14),
    # 202923: "either only 12 and 13 are visible or 12 and 14" -- which is which is a
    # per-window fact settled by the renders, so the intent here is the union.
    ('20260801-202923-vq2-1213-1214', 1): (12, None),
    ('20260801-202923-vq2-1213-1214', 2): (12, None),
    ('20260801-202923-vq2-1213-1214', 3): (12, None),
    ('20260801-202923-vq2-1213-1214', 4): (12, None),
    ('20260801-202923-vq2-1213-1214', 5): (12, None),
    ('20260801-202923-vq2-1213-1214', 6): (12, None),
}

# PER-WINDOW AUDIT, written from the rendered frames (survey mode), exactly as
# mapedges_intent.AUDIT records what the pictures settled.
#   'refuse'            scene contradicts or cannot verify the staged labels
#   {'labels': [...]}   gate per mini-track in increasing-range order; repeated label =
#                       range-split fragments of ONE gate; None = track skipped (its
#                       identity could not be verified, so it contributes nothing)
#
# THE CONSTELLATION THE RENDERS SETTLED (all of it cross-checked in
# precross12_tracks.png, cross12_dense.png, hover_windows.png, markers_202110_p2.png):
# gate 12 is the big gate at the Station 13 column (the 202923 lap flies THROUGH its
# aperture at the fid-645531 crossing); the ribbon exits 12 straight to gate 13 at
# 13.6 m (net-only-visible -- clipped for contour -- which is why every vote-based
# channel misnamed it); 13 -> 14 = 12.3 (reproduces the intent channel's 12.54);
# 14 -> 15 = 13.1; 13-14-15 near-collinear (13-15 = 25.1). Both laps' hover windows
# show the SAME receding 13/14/15 chain from different vantages.
AUDIT = {
    # 202110 markers 1-4 staged "6+7", but gate 7 was ~40 m out and out of frame at
    # every one (verified: its approach track enters at 36.2 m from the right edge one
    # second after the gate-6 crossing, rowcheck_g7approach.png). The 21-31 m partner
    # behind gate 6 is GATE 5 (separation reproduces the known 5-6). No 6-7 exists here.
    ('20260801-202110-vq2-23-67-1314', 1): 'refuse',
    ('20260801-202110-vq2-23-67-1314', 2): 'refuse',
    ('20260801-202110-vq2-23-67-1314', 3): 'refuse',
    ('20260801-202110-vq2-23-67-1314', 4): 'refuse',
    # markers 5-6: staged "13+14"; the scene delivers the 13/14/15 chain.
    ('20260801-202110-vq2-23-67-1314', 5): {'labels': [13, 14, 14, 14]},
    ('20260801-202110-vq2-23-67-1314', 6): {'labels': [13, 14, 15, 15]},
    # 202923 markers: staged "12+13 or 12+14"; gate 12 itself is the huge clipped gate
    # in every window and never measurable, so the measured pairs are the 13/14/15
    # chain seen past it. Track order is by median range.
    ('20260801-202923-vq2-1213-1214', 1): {'labels': [13, 14]},
    ('20260801-202923-vq2-1213-1214', 2): {'labels': [13, 14, 14]},
    ('20260801-202923-vq2-1213-1214', 3): {'labels': [13, 13, 14, 14, 14]},
    ('20260801-202923-vq2-1213-1214', 4): 'refuse',   # single mini-track, no pair
    ('20260801-202923-vq2-1213-1214', 5): 'refuse',   # 27/28/38 m tracks: 13/14 vs
                                                      # 14/15 unresolvable on frames
    ('20260801-202923-vq2-1213-1214', 6): {'labels': [13, 14, 14, 14, 15, 15, 15]},
}

# AUDITED NON-MARKER WINDOWS. Same discipline, anchored on a fid range instead of a
# marker. The one entry is the 202110 pre-crossing-12 pause, where gate 12 is CLEAN
# (3.6 m, crossed one second later -- identity by crossing, not vote) and the ribbon
# order labels what is seen through and beyond its aperture. This window is what
# settles 12-13 (13.57 m) and re-mints the refused "12-13"=20.61 rows as 12-14
# (mapvq2.REFUSE_ROWS). T3/T4 (27.8/38.2 m) stay None: 14-duplicate vs 15 vs 16 could
# not be settled on the frames.
EXTRA_WINDOWS = [
    ('20260801-202110-vq2-23-67-1314', 627500, 627611,
     [12, 13, 14, None, None], 'pre-crossing-12 pause'),
]


def window_tracks_range(S, net, dev, lo, hi, stats):
    frames_meas = []
    for i in range(lo, min(hi, len(S.frames))):
        fpath = os.path.join(S.path, 'frames', S.frames[i]['file'])
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            continue
        ms = [m for m in MI.frame_measurements(S, i, net, dev, img, stats)
              if m['r'] <= MAX_R]
        frames_meas.append((i, ms))
    tracks = MI.window_tracks(frames_meas)
    for t2 in tracks:
        rr = np.array([m['r'] for _i, m in t2['obs']])
        t2['r_med'] = float(np.median(rr))
        t2['r_mad'] = float(np.median(np.abs(rr - np.median(rr))))
        t2['n_net'] = sum(1 for _i, m in t2['obs'] if m['src'] == 'gatenet')
    tracks = [t2 for t2 in tracks if t2['r_mad'] <= MI.TRACK_MAX_MAD]
    tracks.sort(key=lambda t2: t2['r_med'])
    return tracks, frames_meas


def window_tracks_for(S, net, dev, mi, t, stats):
    lo = int(np.searchsorted(S.t, t - PAD_S * 1e9))
    hi = int(np.searchsorted(S.t, t + PAD_S * 1e9))
    return window_tracks_range(S, net, dev, lo, hi, stats)


def pair_rows(S, mi, tracks, label):
    """Same row construction as mapedges_intent.collect: per-frame labeled pairs."""
    byobs = {}
    for t2 in tracks:
        for i, m in t2['obs']:
            byobs.setdefault(i, []).append((label[id(t2)], m))
    rows, tiles = [], []
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
                    continue
                adz = abs(float(v @ gh)) if gh is not None else np.nan
                h = float(np.sqrt(max(d * d - adz * adz, 0.0))) \
                    if gh is not None else np.nan
                rows.append({'i': i, 'fid': int(S.fid[i]),
                             'pair': (('g', ga), ('g', gb)), 'd': d,
                             'dz': np.nan, 'h': h, 'abs_dz': adz, 'mk': True,
                             'ra': float(ma['r']), 'rb': float(mb['r']),
                             'src': 'hover-' + ('net' if 'gatenet' in
                                                (ma['src'], mb['src']) else 'contour'),
                             'marker': mi, 'session': S.name})
                tiles.append({'i': i, 'fid': int(S.fid[i]), 'pair': (ga, gb), 'd': d,
                              'meas': [ma, mb], 'marker': mi,
                              'path': os.path.join(S.path, 'frames',
                                                   S.frames[i]['file'])})
    return rows, tiles


def survey_render(S, mi, tracks, out_dir):
    """One frame per window with every mini-track's quad + index + range -- the sheet
    the AUDIT labels are read from."""
    if not tracks:
        return None
    mid = tracks[0]['obs'][len(tracks[0]['obs']) // 2][0]
    img = cv2.imread(os.path.join(S.path, 'frames', S.frames[mid]['file']),
                     cv2.IMREAD_COLOR)
    if img is None:
        return None
    big = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_NEAREST)
    cols = [(90, 230, 90), (60, 170, 255), (255, 200, 0), (255, 120, 255),
            (0, 255, 255), (120, 120, 255), (255, 255, 255)]
    for j, t2 in enumerate(tracks):
        near = min(t2['obs'], key=lambda o: abs(o[0] - mid))
        q = (np.asarray(near[1]['quad'], np.float32) * 2.0)
        col = cols[j % len(cols)]
        cv2.polylines(big, [np.clip(q, -2000, 4000).astype(np.int32)], True, col, 2,
                      cv2.LINE_AA)
        c = q.mean(0)
        cv2.putText(big, 'T%d %.0fm %s' % (j, t2['r_med'],
                                           'net' if t2['n_net'] else 'ctr'),
                    (int(np.clip(c[0] - 30, 0, 1200)), int(np.clip(c[1], 16, 700))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(big, '%s marker %d  fid %d' % (S.name[-9:], mi, S.fid[mid]),
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return big


def _process(S, tag, tracks, audit, known, stats, rows, tiles, verbose):
    """Apply one window's audit verdict; append surviving rows/tiles."""
    if audit is None or audit == 'refuse':
        stats['window-audit-refused'] += 1
        if verbose:
            print('    AUDIT-REFUSED' if audit == 'refuse'
                  else '    no audit entry -> contributes nothing')
        return
    if len(audit['labels']) != len(tracks):
        stats['window-audit-mismatch'] += 1
        if verbose:
            print('    AUDIT-MISMATCH: expected %d tracks, found %d'
                  % (len(audit['labels']), len(tracks)))
        return
    kept = [(t2, lab) for t2, lab in zip(tracks, audit['labels']) if lab is not None]
    label = {id(t2): lab for t2, lab in kept}
    cand, ctiles = pair_rows(S, tag, [t2 for t2, _lab in kept], label)
    # the known medians referee the labels, as in mapedges_intent
    med = collections.defaultdict(list)
    for r in cand:
        med[(r['pair'][0][1], r['pair'][1][1])].append(r['d'])
    for p, ds in med.items():
        if len(ds) >= 3 and p in known and \
                abs(float(np.median(ds)) - known[p]) > KNOWN_TOL:
            stats['window-refused-known'] += 1
            if verbose:
                print('    REFUSED: labeled %d-%d reads %.1f m vs known %.1f m'
                      % (p[0], p[1], float(np.median(ds)), known[p]))
            return
    if verbose and cand:
        print('    rows %d  labels %s' % (len(cand),
              ['%s->g%s' % (round(t2['r_med']), lab) for t2, lab in kept]))
    rows += cand
    tiles += ctiles
    stats['rows'] += len(cand)


def collect(net=None, dev=None, known=None, apply_audit=True, verbose=True):
    dev = dev or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net or ME.load_net(dev)
    known = dict(known or {})
    stats = collections.Counter()
    rows, tiles, sheets = [], [], []
    for sess in (M.LINKLAP, M.TIEBREAK):
        S = M.Session(sess)
        for mi, (t, _g) in enumerate(S.markers, 1):
            key = (S.name, mi)
            if key not in INTENT:
                continue
            tracks, _fm = window_tracks_for(S, net, dev, mi, t, stats)
            if verbose:
                print('  %s marker %d: %d mini-tracks at %s' %
                      (S.name[-9:], mi, len(tracks),
                       ', '.join('%.0fm(%s)' % (t2['r_med'],
                                                'net' if t2['n_net'] else 'ctr')
                                 for t2 in tracks)))
            sheet = survey_render(S, mi, tracks, None)
            if sheet is not None:
                sheets.append(sheet)
            if not apply_audit:
                continue
            _process(S, mi, tracks, AUDIT.get(key), known, stats, rows, tiles,
                     verbose)
        for sname, flo, fhi, labels, why in EXTRA_WINDOWS:
            if sname != S.name or not apply_audit:
                continue
            lo = int(np.searchsorted(S.fid, flo))
            hi = int(np.searchsorted(S.fid, fhi))
            tracks, _fm = window_tracks_range(S, net, dev, lo, hi, stats)
            if verbose:
                print('  %s extra window %s (fid %d-%d): %d mini-tracks at %s' %
                      (S.name[-9:], why, flo, fhi, len(tracks),
                       ', '.join('%.0fm(%s)' % (t2['r_med'],
                                                'net' if t2['n_net'] else 'ctr')
                                 for t2 in tracks)))
            _process(S, why, tracks, {'labels': labels}, known, stats, rows, tiles,
                     verbose)
    if sheets:
        out = os.path.join(HERE, 'vercheck', 'hover_windows.png')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, np.vstack(sheets))
        print('wrote %s (%d windows)' % (out, len(sheets)))
    if verbose:
        print('[hover] ' + '  '.join(f'{k}={v}' for k, v in sorted(stats.items())))
        per = collections.defaultdict(list)
        for r in rows:
            per[(r['pair'][0][1], r['pair'][1][1])].append(r)
        for p in sorted(per):
            ds = np.array([r['d'] for r in per[p]])
            zz = np.array([r['abs_dz'] for r in per[p] if np.isfinite(r['abs_dz'])])
            nn = sum(1 for r in per[p] if r['src'] == 'hover-net')
            print('  %2d-%-2d n=%3d (net-involved %3d)  d med %6.2f MAD %5.2f '
                  'p10-p90 %5.1f-%-5.1f  |dz| %5.2f'
                  % (p[0], p[1], len(ds), nn, np.median(ds),
                     np.median(np.abs(ds - np.median(ds))),
                     np.percentile(ds, 10), np.percentile(ds, 90),
                     np.median(zz) if len(zz) else float('nan')))
    return rows, tiles, stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', action='store_true',
                    help='apply AUDIT and emit rows (default: survey only)')
    a = ap.parse_args()
    KNOWN = {(0, 1): 20.12, (1, 2): 8.32, (3, 4): 21.72, (4, 5): 22.46,
             (4, 6): 22.74, (5, 6): 15.44, (7, 8): 11.37, (9, 10): 10.10,
             (10, 11): 33.93, (11, 12): 19.48, (13, 14): 12.54, (14, 15): 32.89,
             (15, 16): 20.93}
    rows, tiles, _ = collect(known=KNOWN, apply_audit=a.rows)
    if a.rows:
        json.dump([{**r, 'pair': [list(r['pair'][0]), list(r['pair'][1])]}
                   for r in rows],
                  open(os.path.join(HERE, 'mapedges_hover_rows.json'), 'w'), indent=1)
        print('wrote mapedges_hover_rows.json (%d rows)' % len(rows))
        MI.montage(tiles, out=os.path.join(HERE, 'vercheck',
                                           'hover_pairs_montage.png'))
