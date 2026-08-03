"""Intent-anchored pair rows for the 2026-08-02 staged strafe sessions.

Claire flew five sessions on 2026-08-02, each staged at a NAMED pair (the session name
is her word): strafe-12-13-14 (004153), strafe-2-3 (005208), strafe-6-7 (005431),
2-3-b (005725), strafe-2-3-c (010914). None of them has usable crossing-anchored
identity at the staged gates (2-3-c crosses gate 2 only at t=136 s, at the very end),
so identity here follows mapedges_hover.py's discipline exactly:

  * measurement is identity-free (contour clean() + gatenet on refused detections,
    chained into mini-tracks inside each window, MI.frame_measurements unchanged);
  * IDENTITY per mini-track is Claire's staged pair, cross-checked on rendered survey
    sheets (vercheck/strafe0802_windows_*.png) and refereed by the known pair medians;
  * windows whose scene contradicts the intent are AUDIT-REFUSED; a window with no
    audit entry contributes NOTHING (survey mode exists to write the audit from
    pictures, not from the rank heuristic).

Within a staged two-gate pair the rows carry the PAIR separation and |dz| only (which
gate is which adds nothing to an unsigned edge), exactly as mapvq2.strafe_rows() did
for 15/16. For the three-gate session the render + the pair triangle (12-14 must be
the LONG side) settle the per-track labels.

Known decoration hazard (validator note, strafe23c marker 4): the AI-GP wordmark
renders as a fake second gate. MI.frame_measurements already refuses net seeds with
in_white > 0.15 or in_v > 150 and clean() applies the same interior test to contour
quads; the survey renders are read with that failure mode in mind anyway.

    python3 pilot/perception/mapedges_strafe.py           # survey: tracks + renders
    python3 pilot/perception/mapedges_strafe.py --rows    # apply AUDIT, emit + dump rows
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
import mapedges_hover as MH                            # noqa: E402

PAD_S = 2.0
MAX_R = 50.0

S1213 = '20260802-004153-vq2-strafe-12-13-14'
S23A = '20260802-005208-vq2-strafe-2-3'
S67 = '20260802-005431-vq2-strafe-6-7'
S23B = '20260802-005725-vq2-2-3-b'
S23C = '20260802-010914-vq2-strafe-2-3-c'
SESSIONS = [S1213, S23A, S67, S23B, S23C]

# (session, marker index 1-based) -> staged gates, from the session names (Claire's
# word). Markers are the moments she pressed at a staged viewpoint.
INTENT = {}
INTENT.update({(S1213, k): (12, 13, 14) for k in (1, 2, 3)})
INTENT.update({(S23A, k): (2, 3) for k in (1, 2, 3)})
INTENT.update({(S67, k): (6, 7) for k in (1, 2, 3, 4, 5)})
INTENT.update({(S23B, k): (2, 3) for k in range(1, 8)})
INTENT.update({(S23C, k): (2, 3) for k in (1, 2, 3, 4)})

# NON-MARKER WINDOWS, fid ranges. strafe-2-3-c's cleanest evidence is BETWEEN markers
# (strafe23c_fullscan.log: GOLD two-clean-gate clusters at t~22-33, 43-44, 57-61,
# 100-104 and 131-135 s), so the gold clusters get their own windows. The last one ends
# short of the gate-2 crossing at t~136 s; nothing here spans it.
EXTRA_WINDOWS = [
    (S23C, 1142020, 1142170, (2, 3), 'gold cluster t~22-26s'),
    (S23C, 1142260, 1142370, (2, 3), 'gold cluster t~30-33s'),
    (S23C, 1142660, 1142860, (2, 3), 'gold cluster t~43-49s'),
    (S23C, 1143080, 1143220, (2, 3), 'gold cluster t~57-61s'),
    (S23C, 1144320, 1144500, (2, 3), 'gold cluster t~98-104s'),
    (S23C, 1145300, 1145440, (2, 3), 'pre-crossing t~131-135s'),
]

# PER-WINDOW AUDIT, written from the rendered survey sheets
# (vercheck/strafe0802_windows_*.png), exactly as mapedges_hover.AUDIT records what the
# pictures settled. 'refuse' = scene contradicts or cannot verify the staged labels;
# {'labels': [...]} = gate per mini-track in increasing-range order, repeated label =
# range-split fragments of ONE gate, None = track skipped.
# WHAT THE RENDERS SETTLED (strafe0802_windows_*.png, read window by window):
#
#   * The 1/2/3 constellation: gate 1 is the BIG BANKED ELEVATED gate (ribbon
#     corkscrews through it; consistent with 0-1 dz = +3.3 m), gate 2 the small floor
#     gate ~8 m below/behind it (1-2 = 8.32 known), gate 3 the next ribbon-threaded
#     floor gate. Every "hover at the big gate" window shows fragments of gate 2
#     below it; the clean side view (010914 t~98-104) shows gates 2 and 3 as two
#     similar floor gates, both ~14 m, ribbon looping around the Station 22 column
#     between them, gate 2 anchored by its own crossing 30 s later.
#   * strafe-6-7: gate 6 is the big floor gate AT the Station 16 column (sketch:
#     along 16.3, left row); the partner is the small elevated gate down-course. All
#     five markers show exactly this pair, 2 tracks (m3 adds 42-45 m far tracks).
#     The 202110 "partner was gate 5" trap is answered by the measured separation:
#     ~19-20 m, inconsistent with 5-6 = 15.44 and with 4-6 = 22.74 (see the printed
#     per-pair medians; refuse if it ever reads ~15.4).
#   * strafe-12-13-14 is REFUSED WHOLE: no ribbon renders that far down-course (the
#     ribbon only draws near the active gate, 0 here), and 12-13 (13.45) vs 13-14
#     (12.37) differ by ~1 m, inside the 3 m referee tolerance -- a 12<->14 swap is
#     undetectable on these renders, and both edges are already measured (n=14/212).
#     Refusal, not correction.
#   * 005208 (2-3-a): every window has at most ONE measurable staged gate (partner
#     clipped/oblique/far); the far 36-46 m clusters cannot be named. Nothing usable
#     (matches the validator: "no usable gate track" on most steady segments).
AUDIT = {
    # ---- strafe-6-7: near = 6 (Station 16 anchor), partner = 7
    (S67, 1): {'labels': [6, 7]},
    (S67, 2): {'labels': [6, 7]},
    (S67, 3): {'labels': [6, 7, None, None, None]},
    (S67, 4): {'labels': [6, 7]},
    (S67, 5): {'labels': [6, 7]},
    # ---- strafe-12-13-14: refused whole (see block comment)
    (S1213, 1): 'refuse',
    (S1213, 2): 'refuse',
    (S1213, 3): 'refuse',
    # ---- 2-3-a: no window with both staged gates measurable
    (S23A, 1): 'refuse',
    (S23A, 2): 'refuse',
    (S23A, 3): 'refuse',
    # ---- 2-3-b
    # m1: gate 2 at 9 m right (ribbon descends into it from the elevated gate 1
    #     off-frame), gate 3 = the 16 m ribbon-threaded middle gate (two range
    #     fragments), far 37/38 m unverifiable.
    (S23B, 1): {'labels': [2, 3, 3, None, None]},
    (S23B, 2): 'refuse',   # only gate 3 measured; 36/46 m unnameable
    (S23B, 3): 'refuse',   # fragments of gate 2 below gate 1; 34-39 m unnameable
    (S23B, 4): 'refuse',   # 4 m gate + suspect wordmark boxes (16/19 m ON the frame)
    (S23B, 5): 'refuse',   # three fragments of ONE small gate; no second gate
    (S23B, 6): 'refuse',   # gate 2 fragments only; 37-40 m unnameable
    (S23B, 7): 'refuse',   # single mini-track
    # ---- 2-3-c markers
    (S23C, 1): 'refuse',   # gate 2 fragments seen through clipped gate 1; no pair
    (S23C, 2): 'refuse',   # overlapping fragments at one small gate; 33/38 unnameable
    # m3: the two 17 m net tracks are gate 1 (banked, net-only) and gate 2 below it;
    #     if they are really fragments of one gate the <3 m pair guard drops them,
    #     and the (1,2)=8.32 known median referees the labeling either way.
    (S23C, 3): {'labels': [1, 2]},
    (S23C, 4): 'refuse',   # validator's wordmark-FP window; only gate 2 measurable
    # ---- 2-3-c gold clusters (fid windows)
    # t~22-26: hover AT gate 1 (huge, clipped); 9-12 m fragments = gate 2; the 22 m
    #     ribbon-threaded small gate = 3; 41/42 m unverifiable.
    (S23C, 'gold cluster t~22-26s'): {'labels': [2, 2, 2, 2, 3, None, None]},
    # t~30-33: same vantage, cleaner: gate 2 at 10 m, gate 3 at 22 m (ribbon visibly
    #     threads it), 30 m track unverifiable.
    (S23C, 'gold cluster t~30-33s'): {'labels': [2, 3, None]},
    # t~43-49: gate 1 measured CLEAN at 7 m (aperture quad), gate 2 fragments at
    #     15 m; the 18/20 m boxes sit on gate 1's own wordmark/checkerboard strip
    #     (decoration FPs) and the 37-40 m cluster is unnameable. (1,2) refereed.
    (S23C, 'gold cluster t~43-49s'): {'labels': [1, 2, 2, None, None, None, None,
                                                 None, None]},
    (S23C, 'gold cluster t~57-61s'): 'refuse',   # middle cluster 16-21 m unresolvable
    # t~98-104: the clean side view of gates 2 and 3 -- but the montage tile
    #     (strafe0802_pairs_montage.png, fid 1144434, d=9.05) shows the g3 NET quad
    #     straddling TWO OVERLAPPING gates behind the column: the merged-gate failure
    #     mode from NOTES.md, and this window only ever yielded 3 rows. REFUSED --
    #     2-3 keeps n~79 from three other vantages that agree without it.
    (S23C, 'gold cluster t~98-104s'): 'refuse',
    (S23C, 'pre-crossing t~131-135s'): 'refuse',  # single mini-track (gate 2 at 4 m)
}


def survey_key(S, tag):
    return (S.name, tag)


def collect(net=None, dev=None, known=None, apply_audit=True, verbose=True):
    dev = dev or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net or ME.load_net(dev)
    known = dict(known or {})
    stats = collections.Counter()
    rows, tiles = [], []
    for sname in SESSIONS:
        S = M.Session(os.path.join(M.SESS, sname))
        sheets = []
        for mi, (t, _g) in enumerate(S.markers, 1):
            key = (S.name, mi)
            if key not in INTENT:
                continue
            lo = int(np.searchsorted(S.t, t - PAD_S * 1e9))
            hi = int(np.searchsorted(S.t, t + PAD_S * 1e9))
            tracks, _fm = MH.window_tracks_range(S, net, dev, lo, hi, stats)
            tracks = [t2 for t2 in tracks if t2['r_med'] <= MAX_R]
            if verbose:
                print('  %s marker %d (staged %s): %d mini-tracks at %s' %
                      (S.name[-14:], mi, INTENT[key], len(tracks),
                       ', '.join('%.0fm(%s)' % (t2['r_med'],
                                                'net' if t2['n_net'] else 'ctr')
                                 for t2 in tracks)))
            sheet = MH.survey_render(S, mi, tracks, None)
            if sheet is not None:
                sheets.append(sheet)
            if apply_audit:
                MH._process(S, mi, tracks, AUDIT.get(key), known, stats, rows,
                            tiles, verbose)
        for wj, (sname2, flo, fhi, _staged, why) in enumerate(EXTRA_WINDOWS, 100):
            if sname2 != S.name:
                continue
            lo = int(np.searchsorted(S.fid, flo))
            hi = int(np.searchsorted(S.fid, fhi))
            tracks, _fm = MH.window_tracks_range(S, net, dev, lo, hi, stats)
            tracks = [t2 for t2 in tracks if t2['r_med'] <= MAX_R]
            if verbose:
                print('  %s extra %s (fid %d-%d): %d mini-tracks at %s' %
                      (S.name[-14:], why, flo, fhi, len(tracks),
                       ', '.join('%.0fm(%s)' % (t2['r_med'],
                                                'net' if t2['n_net'] else 'ctr')
                                 for t2 in tracks)))
            sheet = MH.survey_render(S, wj, tracks, None)  # numeric tag for the label;
                                                           # wj-100 = EXTRA_WINDOWS index
            if sheet is not None:
                sheets.append(sheet)
            if apply_audit:
                MH._process(S, why, tracks, AUDIT.get((S.name, why)), known, stats,
                            rows, tiles, verbose)
        if sheets:
            out = os.path.join(HERE, 'vercheck',
                               'strafe0802_windows_%s.png' % S.name[9:15])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cv2.imwrite(out, np.vstack(sheets))
            print('wrote %s (%d windows)' % (out, len(sheets)))
    # provenance: retag the rows this module owns (MH.pair_rows stamps 'hover-*')
    for r in rows:
        r['src'] = r['src'].replace('hover-', 'strafe-')
    if verbose:
        print('[strafe0802] ' + '  '.join(f'{k}={v}' for k, v in sorted(stats.items())))
        per = collections.defaultdict(list)
        for r in rows:
            per[(r['pair'][0][1], r['pair'][1][1])].append(r)
        for p in sorted(per):
            ds = np.array([r['d'] for r in per[p]])
            zz = np.array([r['abs_dz'] for r in per[p] if np.isfinite(r['abs_dz'])])
            nn = sum(1 for r in per[p] if r['src'] == 'strafe-net')
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
    KNOWN = {(0, 1): 20.12, (1, 2): 8.32, (3, 4): 21.82, (4, 5): 22.45,
             (4, 6): 22.74, (5, 6): 15.44, (7, 8): 11.37, (7, 9): 25.75,
             (8, 9): 15.05, (9, 10): 10.10, (10, 11): 33.93, (11, 12): 19.48,
             (12, 13): 13.55, (12, 14): 20.61, (13, 14): 12.37, (13, 15): 26.52,
             (14, 15): 13.09, (15, 16): 20.93}
    rows, tiles, _ = collect(known=KNOWN, apply_audit=a.rows)
    if a.rows:
        json.dump([{**r, 'pair': [list(r['pair'][0]), list(r['pair'][1])]}
                   for r in rows],
                  open(os.path.join(HERE, 'mapedges_strafe_rows.json'), 'w'), indent=1)
        MI.montage(tiles, out=os.path.join(HERE, 'vercheck',
                                           'strafe0802_pairs_montage.png'))
