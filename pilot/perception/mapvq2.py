"""The MEASURED VQ2 gate map: 17 gates, metric, with vertical, from vision + gravity.

WHAT MAKES THIS POSSIBLE NOW. Until 2026-08-01 no VQ2 recording had crossed more than one
gate, so `active_gate_index` never advanced and no detection could be named. `NOTES.md`
called that "THE DATA WALL -- the wall is ORDER, not coverage". Claire then flew
`20260801-121520-vq2-lap-0-15`: gates 0-15 crossed in order, gate 16 lit and approached but
deliberately not crossed (completing the course auto-submits).

THE IDENTITY MECHANISM. At each `active_gate_index` k -> k+1 transition the gate being
crossed is BY DEFINITION gate k, and at that instant it is the nearest gate in frame --
the aircraft is inside its 1500 mm aperture. So:

  1. track detections frame-to-frame (nearest-neighbour, as mapbuild.py does);
  2. at each crossing, the track whose range bottoms out is gate k;
  3. that identity propagates BACKWARD along the whole track, to every earlier frame in
     which the gate was visible -- including frames where it was 40 m away and shared the
     image with four other gates.

Sixteen crossings give sixteen identified tracks. Gate 16 has no crossing and is recovered
the way `gate5.py` recovered VQ1's uncrossed last gate: from the window where the sim says
it is the active gate, taking the NEAREST cluster rather than the most-detected one.

WHY THE MAP IS DISTANCES AND HEIGHTS, NOT POSES. VQ2 blocks every pose stream, so the
camera's orientation is unknown -- except that GRAVITY is not. `HIGHRES_IMU` is permitted
and its accelerometer is canonical (CONVENTIONS.md), so at low acceleration it gives the
vertical in the body frame directly. That fixes two of the camera's three rotational
degrees of freedom. Heading stays unknown, so for a co-visible pair of gates:

    height difference   dz = -(p_b - p_a) . g_hat     KNOWN, exactly, yaw-free
    horizontal distance  h = sqrt(|p_b - p_a|^2 - dz^2)   KNOWN
    horizontal BEARING                                    unknown (that is the missing yaw)

A pair therefore measures a distance and a height, and nothing else. Distances plus one
known axis is exactly a 2-D embedding problem: heights solve as a linear system, and the
horizontal layout comes from weighted MDS over the measured horizontal distances. The
layout is fixed only up to rotation and REFLECTION, which is all guidance ever needs and
which the sketch resolves by comparison.

THE MEASUREMENT SETUP IS THE ENEMY. This project has been bitten four times by a setup that
silently decided its own answer, most recently when two overlapping gates merged into one
orange blob, the 2700 mm outer-boundary fallback measured the pair as one gate, and the far
gate was placed 20 m away where geometry demanded 26 -- faking a 1.65x inconsistency in
Claire's sketch. Guards here, all of them refusals rather than corrections:

  * metric measurements use `source == 'inner'` ONLY. The merged-blob failure and the
    clipped-gate failure both live in the outer fallback; refusing it removes both classes
    outright rather than trying to detect them. Tracking still sees every detection.
  * `labelgates.quality()` must be clean: no clipping, no extreme obliquity, and PnP range
    must agree with apparent-size range. Two independent range estimates from one quad.
  * decoration is rejected on the INTERIOR COLOUR of the fitted quad, not on
    `drop_decorations_by_parent()`, which was measured here rejecting 73% of detections
    including real gates -- see clean().
  * a pair is only used if BOTH detections pass, and the per-pair scatter across many
    viewpoints is reported. Two static gates whose measured separation changes with
    viewpoint is a MEASUREMENT fault -- the check that was missing on 2026-08-01.

WHAT THIS ACTUALLY DELIVERED, stated up front because a docstring that describes the
intention rather than the result is how a project talks itself into a map it cannot check:

  * identity for ALL 17 gates, from crossings alone, no hand labelling -- verified by
    drawing the assigned race index on real frames and looking at them (vq2ident.py);
  * metric inter-gate distances and gravity-referenced height differences for 12 gate
    pairs (was 8 before the 2026-08-01 pausing lap), each with the scatter of one fixed
    quantity measured from many viewpoints; pair 0-1 repeats across three flights to
    0.10 m;
  * NOT a 17-gate layout. All 17 gates now sit in five internally-measured components --
    [0-2] [3-6] [7-9] [10-13] [14-16] -- but nothing measured links one component to the
    next: at exactly the moments the linking gate was in frame it was clipped, and clean()
    refuses clipped quads. `map_vq2.json` carries the components in separate local frames
    and its `status` block says what may and may not be read from them.

    python3 pilot/perception/mapvq2.py            # full: pairs, scale, LOO, lap-vs-lap
    python3 pilot/perception/mapvq2.py --quick    # skip leave-one-out and jackknife
    python3 pilot/perception/vq2ident.py --frames 6 --out ident.png   # LOOK at identity
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402
import vq2cache  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.abspath(os.path.join(HERE, '..', 'sessions'))

PRIMARY = os.path.join(SESS, '20260801-121520-vq2-lap-0-15')
SECOND = os.path.join(SESS, '20260801-115735-vq2-lap-0-11')
STRAFE = os.path.join(SESS, '20260801-114948-vq2-strafing-15-16')
# The lap flown 2026-08-01 SPECIFICALLY to fix the bracing problem: pause wherever two or
# three gates are fully visible at once, and press a MARKER at each such point (recorded as
# kind == 'marker' in events.jsonl, with t_wall_ns). 25 markers over gates 1-14. The pauses
# are what let identity and co-visibility overlap, which is what the first two laps lacked.
PAUSING = os.path.join(SESS, '20260801-144858-vqual2-lap-pausing')
# The 2026-08-01 evening link laps. BOTH have crossings (202110 reaches active 14, 202923
# reaches 13), so the ordinary crossing-anchored identity runs on them; the markers are
# stable-ish hovers at the staged pairs (202110: 4 @active 6 for 6+7, 2 @active 13 for
# 13+14, plus an unmarked 19 s pause in the active-3 window for 2+3; 202923: 6 @active 12,
# each viewing 12+13 or 12+14 -- which is which is settled per marker by rendering, in
# mapedges_hover.py).
LINKLAP = os.path.join(SESS, '20260801-202110-vq2-23-67-1314')
TIEBREAK = os.path.join(SESS, '20260801-202923-vq2-1213-1214')

MARKER_PAD_S = 0.75     # half-window around a marker, ~0.5 s guaranteed by Claire + margin
# The link laps' markers were pressed at "stable-ish" hovers with slight translational
# drift and near-zero rotation (Claire's description), so their windows are wider. Only
# these two sessions -- widening the old sessions' pads would change the marker-referee
# populations that already survived frame verification.
MARKER_PAD_OVERRIDE_S = {
    '20260801-202110-vq2-23-67-1314': 2.0,
    '20260801-202923-vq2-1213-1214': 2.0,
}

# Regions to exclude from TRACK BUILDING, from session-notes.txt. Both are reversals: the
# aircraft approaches one gate twice, and naive nearest-neighbour association would splice
# the two passes into a single track and then anchor it to the wrong crossing. The frames
# are not merely skipped -- every live track is flushed at both boundaries, so nothing can
# span the gap. Same class of error as the post-run reset that mapbuild.py drops the tail
# for.
EXCLUDE = {
    '20260801-121520-vq2-lap-0-15': [
        (102799, 102987),    # real index reversal 1 -> 0 -> 1
        (105273, 106140),    # Claire's 32 s backtrack between gates 9 and 10. Ends short
    ],                       # of the 106229 crossing so gate 10 still has an approach.
    '20260801-155701-vq2-targeting-pairs': [
        (147783, 147917),    # sim_reset ~16 s in (teleport to start, index 1 -> 0);
    ],                       # +/- 2 s so no track and no vote spans the teleport.
    '20260801-202110-vq2-23-67-1314': [
        (622855, 622979),    # sim_reset 2 s in (teleport to start), +/- 2 s from t=0
        (628441, 628576),    # trailing sim_reset (14 -> 0), through end of recording
    ],
    '20260801-202923-vq2-1213-1214': [
        (645741, 645820),    # trailing sim_reset (13 -> 0), through end of recording
    ],
}

# ROWS REFUSED AFTER FRAME VERIFICATION (2026-08-01 evening). Every entry here was a
# pair the pipeline measured and a render then contradicted -- the same LOOK step that
# caught every earlier identity fault. Refusal, not correction: the rows are dropped and
# reported, never re-labeled. All four are BACKWARD-EXTENSION slides: extend_back()'s
# continuity allowance (3 m + 0.7 m/frame of coasting) lets a long walk hand off between
# gates during fast yaws/dashes, and the walk then carries its name onto a different
# physical gate. Renders: vercheck/rowcheck_*.png.
REFUSE_ROWS = {
    ('20260801-202923-vq2-1213-1214', (0, 2)):
        'g2-EXT sat on gate 1 during the opening dash; rows re-measure 0-1 '
        '(20.59 vs known 20.12). rowcheck_tiebreak_open.png',
    ('20260801-202923-vq2-1213-1214', (1, 3)):
        'g1-EXT sat on gate 0 (the lit, yellow-glow active gate) and g3-EXT on gate 1; '
        'rows re-measure 0-1 (21.61 vs known 20.12). rowcheck_tiebreak_open.png',
    ('20260801-202923-vq2-1213-1214', (0, 3)):
        'two far objects 7.6 m apart under wrong names during the same opening dash '
        '(n=2, below MIN_PAIR_OBS anyway).',
    ('20260801-202110-vq2-23-67-1314', (6, 8)):
        'g8-EXT slid onto GATE 5, seen nearly behind gate 6 from the 6+7 hover '
        '(22->47->67->12 m range discontinuity in one walk); the 14.96 m rows '
        'reproduce the known 5-6 = 15.44. zoom_625868.png',
    ('20260801-202110-vq2-23-67-1314', (12, 13)):
        'the rows\' far object chains (EXT -> trk1123, junction verified) to the gate '
        'the RIBBON threads SECOND after gate 12 -- gate 14, not 13. The real 13 is the '
        'gate the ribbon threads FIRST, dead behind 12\'s own aperture, net-only-visible '
        '(clipped for contour) at 20 m; the vote went to 14 because only 14 passed '
        'clean(). These rows measure 12-14 = 20.61 and re-enter under that name via the '
        'audited pre-crossing window in mapedges_hover.py. precross12_tracks.png',
    ('20260801-144858-vqual2-lap-pausing', (14, 15)):
        'the "15" here is gate 16: 15 was never crossed in this lap (kmax naming, '
        'nearest-cluster rule) and the link-lap windows measure 14-15 = 13.1 with '
        'ribbon-ordered identity; 13.09 + 20.93 (strafe 15-16) = 34.0 vs these rows\' '
        '32.89 -- a 175-degree near-collinear closure, so the rows are 14-16 under a '
        'wrong name. Reread of 14-15_f34722_d32.8.png: gate 14 at 3 m, the labeled '
        '"15" at 36 m, and an unlabeled nearer gate at bottom right.',
    ('20260801-115735-vq2-lap-0-11', (8, 9)):
        'g9-EXT has two slide signatures (48->70 m jump with 160 px centre move; '
        '18->13.6 m with 170 px move) and leans on outer-fallback quads; its 22.64 m '
        'contradicts the render-verified 8-9 = 15.0 of the link lap '
        '(rowcheck_linklap.png) and the 7-9 = 26.0/26.1 collinearity closure.',
}

# ---------------------------------------------------------------------------------------
# THE ONE RELABEL (2026-08-02). Everywhere else in this file identity faults are REFUSED,
# never corrected -- refusal is safe because a dropped row cannot lie. This single entry is
# the exception, and it is here rather than in REFUSE_ROWS because the rows are a perfectly
# good measurement of a REAL pair; only the NAME on the far end was wrong. Dropping them
# would delete the only link across a graph bridge and leave gates 11-16 unplaceable, which
# is a worse answer than the one the evidence actually supports.
#
# WHAT WAS WRONG. The accepted 10-11 edge read 33.88 m. Three independent channels say no:
#   (1) DRAG PATH. Integrating the validated drag speed model between the gate-10 and
#       gate-11 crossings gives path/chord = 0.53 (fast lap 20260802-161838) and 0.67
#       (pausing lap) against this chord, while every other leg of the same laps sits at
#       1.02-1.08 median. A flown path SHORTER than the straight-line chord is impossible,
#       and dz is only -1.75 m so altitude cannot rescue it (a climb pushes the ratio UP).
#   (2) PROVENANCE. All 76 rows come from ONE session, ONE contiguous 171-frame window
#       (pausing lap, fid 32235-32406), bearing spread 1.5 deg -- a single vantage. The
#       static-pair viewpoint referee that caught every other misidentification in this
#       file NEVER FIRED, because it needs two viewpoints to fire. The far gate is 11-15 px
#       against MIN_SIZE_PX = 14. Re-detecting the window (xfer_leg1011.py,
#       vercheck/xfer_leg1011.png) shows gates strung at ~8 / ~22-25 / ~32 / ~41 m of
#       range, and the accepted rows pair the 8 m gate with the 41 m one, skipping two.
#   (3) STATION-NUMBER RULER. Regressing map along-course x on Claire's sketch station
#       numbers, FIT ON GATES 0-10 ONLY, gives 22.59 m/station (independent column-pitch
#       estimate 25.4 +- 2 m) with gates 0-10 residuals mean 0.00 m, sd 3.13 m -- and gates
#       11-16 all at -17.8..-25.5 m, mean -19.79. A clean RIGID STEP at gate 11: the tail
#       is translated, so exactly one edge is wrong and it is this one.
#
# WHAT IT ACTUALLY MEASURES. The far detection was GATE 12. Tested against the alternatives
# by re-solving the layout with the vector relabelled and asking the ruler (the only channel
# outside this measurement) which one it accepts:
#       far = 12 -> implied 10-11 = 15.24 m @ 188.9 deg, gates 11-16 residual mean -4.21 m
#                   (sd 2.90) -- INSIDE the gates 0-10 band
#       far = 13 -> implied 10-11 =  8.13 m @ 128.7 deg, residual mean +5.76 m, and the leg
#                   points BACKWARDS across the course
#       far = 14 -> implied 10-11 =  5.04 m @ 351.0 deg, residual mean +15.82 m
# far = 12 also puts the 10-11 bearing (188.9) between its neighbours 9-10 (184.2) and 11-12
# (217.6), i.e. a smoothly turning course rather than a kink.
RELABEL_ROWS = [
    ('20260801-144858-vqual2-lap-pausing', (10, 11), 32235, 32406, (10, 12),
     'the far (11-15 px, ~41 m) detection is GATE 12, not 11: see the block above. '
     'Distance and bearing are unchanged, only the name; 10-11 becomes a DERIVED edge '
     '(15.24 m, from this vector minus the independently measured 11-12 vector).'),
]


def relabel_rows(rows, session, verbose=False):
    """Apply RELABEL_ROWS in place-ish. Window-scoped on purpose: a global (10,11)->(10,12)
    rule would silently eat a future genuine 10-11 measurement from another vantage, which
    is exactly the measurement this correction is asking for."""
    n = 0
    for sname, old, lo, hi, new, _why in RELABEL_ROWS:
        if sname != session:
            continue
        for r in rows:
            a, b = r['pair']
            if a[0] != 'g' or b[0] != 'g' or (a[1], b[1]) != old:
                continue
            if not (lo <= r['fid'] <= hi):
                continue
            r['pair'] = (('g', new[0]), ('g', new[1]))
            n += 1
        if verbose and n:
            print('  RELABEL_ROWS: %d rows %d-%d -> %d-%d (see mapvq2.RELABEL_ROWS)'
                  % (n, old[0], old[1], new[0], new[1]))
    return rows


# Tracking. MAX_GAP is 30 frames (1 s) rather than mapbuild.py's 4 (0.13 s): there the short
# gap was itself named as the prime suspect for fragmentation, and fragmentation is fatal
# here in a way it was not there -- a track that breaks loses the identity its crossing
# anchor gave it. See build_tracks() for why the association radius must NOT simply scale
# with the gap.
MATCH_PX_BASE = 40.0
MATCH_SIZE_RATIO = 2.2
CLOSING_M_PER_FRAME = 0.7  # 21 m/s: how fast a real gate's range can change
MAX_GAP = 30
MIN_TRACK_LEN = 4

ANCHOR_BACK = 90        # frames before a crossing searched for the gate being crossed
ANCHOR_MAX_R = 8.0      # the crossed gate passes within 0.75 m; 8 m is a generous floor

MIN_SIZE_PX = 14.0      # below this PnP is too soft to anchor a metric distance
MAX_ASPECT = 2.5        # fitted-quad edge ratio; past this a corner error swings the pose
MAX_RANGE_DISAGREE = 0.30   # PnP range vs apparent-size range
NEAREST_VOTES = 4
NEAREST_VOTES_DOC = 'frames a track must lead its window to inherit that identity'
CROSS_MAX_R = 12.0      # a gate that was CROSSED must have been seen at least this close
GRAV_TOL = 0.35         # |a| - 9.81 tolerance for the accelerometer to be reading gravity
MIN_PAIR_OBS = 3        # a pair measured fewer times than this is not reported


STATUS = {
    'headline': 'COMPLETE AS A RELATIVE MAP (2026-08-02). Identity for all 17 gates, '
                'verified on frames; 20 gate pairs measured -- ALL 16 consecutive '
                'links plus skips 4-6, 7-9, 12-14, 13-15 -- and the gates sit in ONE '
                'connected component. The last two links came from the 2026-08-02 '
                'staged strafes: 2-3 = 13.3 m (n=79, three vantages, two sessions) '
                'and 6-7 = 16.4 m (n=556, five hover viewpoints). Positions outside '
                'the braced clusters are still chain topology, not surveyed geometry '
                '-- read what_is_NOT_measured before trusting a coordinate.',
    'what_is_measured': 'Direct inter-gate distances and gravity-referenced height '
                        'differences for the gate pairs under measured_pairs -- metric, '
                        'anchored on the spec 1500 mm aperture, each with the scatter of '
                        'one fixed quantity measured from many viewpoints. Best cross-lap '
                        'repeat: pair 0-1 = 20.12/20.02/20.06/20.96 m on four flights. '
                        'The 12/13/14/15 cluster is now measured as a braced quad: '
                        '12-13, 12-14, 13-14, 13-15, 14-15, 15-16 all direct.',
    'what_is_NOT_measured': 'Honest limits, weakest first. (1) 1-2 rests on n=5 contour '
                            'rows (8.32 m) while a REFUSED 34-row gatenet channel reads '
                            '13.0 m and the sketch scale mildly favours the larger value '
                            '-- the tension is unresolved and 1-2 is the least-trusted '
                            'edge in the map. (2) Thin edges by n: 14-15 (n=9), 9-10 '
                            '(n=12, single channel), 12-13 (n=14). (3) Heights: strafe/'
                            'hover/intent rows carry |dz| only, so 2-3, 6-7 and most of '
                            '12-16 have NO signed dz in the map (6-7\'s step is +5.3 m, '
                            'gate 7 above, measured but held outside the solve; see '
                            'NOTES.md 2026-08-02). (4) The single component is chain-'
                            'like between braced clusters (12-13-14-15 quad, 4-5-6 '
                            'triangle, 7-8-9): leave-one-out on bridge edges is weak '
                            '(median 9.5 m) BY CONSTRUCTION, and local_positions are '
                            'topology there, not geometry. (5) The 10-11 sketch-scale '
                            'outlier (57.6 m/station) persists: the sketch compresses '
                            'across-row distances, so metres_per_station is a spread, '
                            'not a constant.',
    'why': 'Vote-based naming fails exactly at staged hovers (the active gate clips out '
           'and the nearest CLEAN detection inherits the name) and backward extensions '
           'slide between gates during fast yaws -- four render-verified cases of each '
           'class are recorded in REFUSE_ROWS and mapedges_hover.AUDIT. Every pair that '
           'survives is either crossing-funnel-anchored, ribbon-order-verified on frames, '
           'or triangle-closed against an independently measured pair.',
    'what_would_fix_it': 'DONE 2026-08-02 for the links themselves (staged strafes; '
                         'see status.staged_strafes_2026_08_02). Still open: the 1-2 '
                         'tension wants one slow pass with gates 1 and 2 co-visible '
                         'and UNCLIPPED from ~15-20 m out; signed dz for 2-3/6-7/'
                         '12-16 wants any window where the pair identity is ordered '
                         'on frames. Original notes, kept for the record: for 6-7 a '
                         'hover ~20 m out on the 6->7 leg past gate 6 (this is exactly '
                         'the flight that worked); for 2-3 a vantage outside the '
                         'Station-22 loop keeping 2 and 3 co-visible; '
                         'the natural racing line never does.',
    'pausing_lap_2026_08_01': {
        'session': '20260801-144858-vqual2-lap-pausing',
        'markers': '25 in events.jsonl (kind=marker, t_wall_ns), gates 1-14; marker '
                   'windows contributed 485 pair rows and refereed two bimodal pairs.',
        'lessons_added': [
            'Long hover windows let SEVERAL tracks win nearest-in-window votes for the '
            'same gate (11 tracks claimed gate 13, ranges 2-44 m, one with range RISING '
            'through the approach). Fixed by three refusals: cross_consistent() binds the '
            'observation nearest the crossing into a closing-speed funnel; per-gate '
            'pruning un-names any track contradicting a closer-approach track in-frame by '
            '>3 m; and measure() refuses a gate in any frame where its claimed detections '
            'disagree by >3 m, checked on ALL observations because impostor rows survive '
            'exactly where the true gate fails clean().',
            'measure() kept same-gate duplicates in a plain dict, so whichever fragment '
            'was inserted last silently won the frame; that made 0-1 bimodal (20/34 m). '
            'Found by rendering identities on frames, again.',
            'aggregate() now splits a bimodal pair (two static gates have ONE separation) '
            'and keeps the cluster containing Claire\'s marker rows -- an external referee '
            '-- else the larger; the loser is dropped, not averaged, and reported as '
            'bimodal_rows_dropped.',
        ],
    },
    'bugs_found_and_fixed_while_building_this': [
        'Association radius scaled with the coasting gap without a cap (inherited from '
        'mapbuild.py where MAX_GAP was 4). At MAX_GAP 60 the radius exceeded the image and '
        'tracks spliced onto decoration and onto other gates. Caught by drawing assigned '
        'identities on frames, invisible in every aggregate.',
        'labelgates.drop_decorations_by_parent() rejects 73% of detections on this lap '
        'INCLUDING REAL GATES: merged gates share a parent contour and produce the same '
        'range contradiction it tests for. Replaced by an interior-colour test.',
        'Decoration inherited gate identity when the real gate was too close to pass the '
        'quality filters, so the nearest surviving detection was a checkerboard square on '
        'the gate being flown through. Fixed by the physical close-approach test.',
        'Backward extensions of gates 6, 7, 8 and 9 all coasted onto the SAME object, '
        'giving pairwise distances of 0.00 m with MAD 0.00 and n=151. Fixed by mutual '
        'exclusion; a hard refusal of sub-metre gate separations is now in measure().',
    ],
    'resolved_5_6': 'The 15.6 vs 37.0 m lap disagreement is settled: 5-6 = 16.3 m. The '
                    '0-11 lap (115735) carried the misidentification, and its own rows '
                    'contain both modes (29 rows at ~15 m, 35 at ~38 m) -- a fragment of '
                    'its gate-6 identity had slid onto a farther gate. The primary lap is '
                    'unimodal at 15.6 m and the bridge-graph LOO predicts 14.1 m.',
    'link_laps_2026_08_01_evening': {
        'sessions': ['20260801-202110-vq2-23-67-1314 (crossings 0-13, markers 4x "6+7", '
                     '2x "13+14")',
                     '20260801-202923-vq2-1213-1214 (crossings 0-12, 6 markers at '
                     'active 12, 164 s hover)'],
        'what_they_settled': [
            '12-13 = 13.6 m. Neither prior candidate was right: the old thin contour '
            'rows (~10-12 m) underread it and the gatenet channel (17.15) carries the '
            'net\'s +3 m range bias on far gates. Settled in the audited pre-crossing '
            'window: gate 12 is CLEAN at 3.6 m (crossed one second later), the ribbon '
            'exits 12 straight to gate 13 at 20 m (net-only-visible), and 13-14 closes '
            'at 12.3 vs the independently measured 12.54. precross12_tracks.png.',
            '12-14 = 20.6 m -- the very rows the pipeline had labeled "12-13"; their '
            'far object chains to the gate the ribbon threads SECOND (= 14).',
            '8-9 = 15.0 m, replacing 22.64 (a g9-extension double-slide in the 0-11 '
            'lap). Render-verified on the link lap and closed by 7-8 + 8-9 = 26.4 vs '
            'the twice-measured 7-9 = 26.0/26.1 -- gates 7-8-9 are nearly collinear, '
            'as the targeting session first suggested.',
            '14-15 = 13.1 m, replacing 32.89, which was 14-16 under a wrong name: '
            '13.09 + 20.93 (strafe 15-16) = 34.0, a 175-degree collinear closure. '
            'Gates 14-15-16 are nearly collinear; 13-14-15 nearly collinear too '
            '(13-15 = 25.1 measured directly).',
            '6-7 was NOT delivered: at every "6+7" hover gate 7 was out of frame '
            '(~40 m out); the partner behind gate 6 was gate 5. 2-3 was NOT delivered: '
            'the 2+3 pause shows 3+4. Both stagings need different vantages.',
        ],
        'method': 'Identity-free mini-track measurement (contour clean() + gatenet on '
                  'refused detections) inside marker/audited windows; identity per '
                  'mini-track from Claire\'s stated intent, cross-checked on rendered '
                  'frames (mapedges_hover.AUDIT) and refereed by known pair medians. '
                  'Vote/extension identity from the hovers themselves was demonstrably '
                  'contaminated and is refused wholesale (REFUSE_ROWS).',
    },
    'known_open_disagreement': '7-8: the two fast laps read 26.6-28.5 m, the pausing lap '
                               '11.04 m with MAD 0.32. Sided with 11.04 (marker-vouched, '
                               'crossing-funnel-anchored; the 26.6 mode is the 7-9 '
                               'collinear line under a wrong name). Remaining thin '
                               'spots: 12-13 rests on one audited window (n~12, one '
                               'lap); 14-15 = 13.1 rests on hover rows from two '
                               'windows. The "10-11 = 34.0 repeats across two laps" '
                               'claim was WRONG and is retracted: both repeats are the '
                               'SAME single vantage, and the pair is 10-12. See '
                               'RELABEL_ROWS.',
}



# ---------------------------------------------------------------------------------------
# session data


class Session:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.frames = [r for r in L.load_csv(os.path.join(path, 'frames.csv')) if r['file']]
        self.fid = np.array([int(r['frame_id']) for r in self.frames])
        self.t = np.array([float(r['t_recv_wall_ns']) for r in self.frames])
        self.det = vq2cache.load(path)
        imu = L.load_csv(os.path.join(path, 'imu.csv'))
        self.imu_t = np.array([float(r['t_wall_ns']) for r in imu])
        self.imu_a = np.array([[float(r['xacc']), float(r['yacc']), float(r['zacc'])]
                               for r in imu])
        race = L.load_csv(os.path.join(path, 'race.csv'))
        self.race_t = np.array([float(r['t_wall_ns']) for r in race])
        self.race_g = np.array([int(r['active_gate_index']) for r in race])
        self.excl = EXCLUDE.get(self.name, [])
        self.frame_gate = self._frame_gate()
        self.markers = self._markers()

    def _markers(self):
        """Claire's in-flight markers from events.jsonl: [(t_wall_ns, active_gate), ...].

        Pressed at moments where she judged all visible gates FULLY in frame -- exactly the
        co-visibility instants the map needs. Sessions recorded before the marker key exist
        too; they simply return []."""
        p = os.path.join(self.path, 'events.jsonl')
        out = []
        if os.path.exists(p):
            with open(p) as fh:
                for line in fh:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get('kind') == 'marker':
                        out.append((float(e['t_wall_ns']), int(e.get('active_gate', -1))))
        return out

    def marker_frames(self, pad_s=None):
        """Set of frame indices within pad_s of any marker (per-session default pad)."""
        if pad_s is None:
            pad_s = MARKER_PAD_OVERRIDE_S.get(self.name, MARKER_PAD_S)
        out = set()
        for t, _g in self.markers:
            lo = int(np.searchsorted(self.t, t - pad_s * 1e9))
            hi = int(np.searchsorted(self.t, t + pad_s * 1e9))
            out.update(range(lo, min(hi, len(self.frames))))
        return out

    def _frame_gate(self):
        """Active gate index per FRAME, -1 after the run ends.

        Truncated at the last row that reads the highest index the session ever reached,
        because every one of these recordings ends with a sim_reset that snaps the index
        back to 0. Without the cut, `active_gate_index == 0` would cover both the opening
        approach and the post-run reset, and the reset frames -- taken from the teleported
        start position -- would vote as if they were gate 0's approach. That is mapbuild.py's
        drop-the-tail trap in a different costume.
        """
        g = np.full(len(self.frames), -1, dtype=int)
        if len(self.race_g) == 0:
            return g
        kmax = int(self.race_g.max())
        last = int(np.max(np.where(self.race_g == kmax)))
        t_end = self.race_t[last]
        j = np.searchsorted(self.race_t, self.t, side='right') - 1
        ok = (j >= 0) & (self.t <= t_end)
        g[ok] = self.race_g[j[ok]]
        return g

    def gravity(self, i):
        """Unit gravity DOWN in the body frame at frame i, or None if manoeuvring.

        The accelerometer measures specific force, reading -g at rest, so gravity-down is
        -a normalised (labelgates.gravity_at, same convention). CONVENTIONS.md settles that
        the accelerometer is NOT mirrored, so no sign is applied here.
        """
        j = int(np.argmin(np.abs(self.imu_t - self.t[i])))
        if abs(self.imu_t[j] - self.t[i]) > 0.1e9:
            return None
        a = self.imu_a[j]
        n = float(np.linalg.norm(a))
        if abs(n - 9.81) > GRAV_TOL:
            return None
        return -a / n

    def crossings(self):
        """{k: frame index} for every active_gate_index k -> k+1 advance.

        Derived from race.csv rather than copied from session-notes.txt. Reversals (k -> k-1)
        are reported separately by report(); they are the reason EXCLUDE exists.
        """
        out, rev = {}, []
        prev = None
        for t, g in zip(self.race_t, self.race_g):
            if prev is None:
                prev = g
                continue
            if g == prev + 1:
                out[prev] = int(np.argmin(np.abs(self.t - t)))
            elif g != prev:
                rev.append((prev, g, int(np.argmin(np.abs(self.t - t)))))
            prev = g
        return out, rev

    def active_window(self, k):
        """Frame-index range over which the sim says gate k is the gate being flown at."""
        m = self.race_g == k
        if not m.any():
            return None
        t0, t1 = self.race_t[m].min(), self.race_t[m].max()
        i0 = int(np.searchsorted(self.t, t0))
        i1 = int(np.searchsorted(self.t, t1))
        return (i0, min(i1, len(self.frames) - 1))

    def excluded(self, i):
        f = self.fid[i]
        return any(a <= f <= b for a, b in self.excl)


def clean(d, min_size=MIN_SIZE_PX):
    """Is this detection fit to carry a metric distance?

    Recomputed here from the cached quad rather than reusing `labelgates.quality()`'s stored
    verdict, because two of that function's defaults are wrong for this data and one of its
    companions is actively harmful:

    * `source == 'inner'` is REQUIRED, and it is the merged-gate guard. Two gates that
      overlap in image merge into a single orange blob, and the 2700 mm outer-boundary
      fallback then measures the pair as one gate -- the failure that faked a 1.65x
      inconsistency in Claire's sketch. The apertures do NOT merge: they stay separate hole
      contours inside the shared blob. Refusing the outer fallback removes the merged-gate
      failure AND the clipped-gate failure at once, because both live in that path.

    * `drop_decorations_by_parent()` IS NOT USED, and that is a change of position. Measured
      on this lap it rejects 22142 of 30161 detections, and inspection of the frames shows
      it is rejecting real gates: its test is "inside a parent orange blob that implies a
      much nearer range", and MERGED GATES produce exactly that signature. Frame 00104000 is
      the case -- a 25 m gate overlapping a 6 m one shares its parent contour, so the far
      gate is called decoration. It is safe to drop here because identity does the same job
      better: a detection only ever enters the map if its track was anchored by a CROSSING,
      and no wordmark or checkerboard square is ever the thing the aircraft flew through.

    * obliquity is allowed to 2.5 rather than quality()'s 1.8. Gates on this course are
      genuinely seen at an angle; the PnP-versus-apparent-size agreement below is the real
      test of whether the pose survived it, and it is kept strict.
    """
    if d['source'] != 'inner' or d['size_px'] < min_size:
        return False
    # DECORATION, rejected on what is INSIDE the quad. A gate's aperture is a hole with the
    # dark hangar behind it; the AI-GP wordmark and the checkerboard strips are white paint
    # on the frame, and their contours pass the same quad fit and the same PnP. Two
    # populations that do not overlap: real apertures mean V 17-57 and 0-2% white, every
    # decoration mean V ~245 and 29-100% white.
    #
    # This replaces drop_decorations_by_parent(), which rejected 73% of all detections here
    # including real gates, and it is strictly better evidence: that function inferred
    # decoration from a range contradiction against a parent blob, which merged gates
    # produce too. This looks at the thing itself.
    if d.get('in_white', 0.0) > 0.15 or d.get('in_v', 0.0) > 150.0:
        return False
    q = np.asarray(d['quad']).reshape(-1, 2)
    if q[:, 0].min() < 0 or q[:, 1].min() < 0 or q[:, 0].max() > L.W or q[:, 1].max() > L.H:
        return False                                     # clipped: corners never seen
    e = [float(np.linalg.norm(q[(k + 1) % 4] - q[k])) for k in range(4)]
    if max(e) / max(min(e), 1e-6) > MAX_ASPECT:
        return False
    r1, r2 = d['range_m'], d['range_size_m']
    # Two independent range estimates from ONE quad. Their disagreement is a free confidence
    # measure, and it is what catches a quad fitted to the cyan ribbon's silhouette instead
    # of the orange frame.
    return abs(r1 - r2) / max(r1, r2, 1e-6) <= MAX_RANGE_DISAGREE


# ---------------------------------------------------------------------------------------
# tracking


def build_tracks(S, verbose=False):
    """Nearest-neighbour association over CONSECUTIVE frames -> {tid: [(i, det), ...]}.

    Consecutive is not optional: association assumes a detection moves a little between
    samples, and mapbuild.py records that striding a session produced ZERO tracks. Excluded
    regions flush every live track rather than being skipped over.
    """
    tracks = collections.defaultdict(list)
    live = {}
    next_tid = 0
    per_frame = []
    for i, r in enumerate(S.frames):
        det = S.det.get(r['file'], [])
        assigned = {}
        used = set()
        for d in sorted(det, key=lambda x: -x['size_px']):
            best, bd = None, 1e18
            for tid, t in live.items():
                if tid in used or i - t['last'] > MAX_GAP:
                    continue
                dist = float(np.linalg.norm(t['centre'] - d['centre']))
                ratio = d['size_px'] / max(t['size'], 1e-6)
                if not (1 / MATCH_SIZE_RATIO < ratio < MATCH_SIZE_RATIO):
                    continue
                # THE GAP MULTIPLIER MUST BE CAPPED. mapbuild.py wrote this as
                # `* (i - last)` with MAX_GAP = 4, so the radius could never exceed 4x.
                # Raising MAX_GAP to 60 to fight fragmentation silently turned the radius
                # into 2400 px -- larger than the image -- and every track then matched
                # every detection. The damage was not subtle and it was not visible in any
                # aggregate: tracks spliced onto checkerboard decoration and onto entirely
                # different gates, identity followed the splice, and the map came out with
                # gate 13 and gate 15 measured 4.1 m apart. Caught by drawing the assigned
                # identities on six frames and looking at them.
                lim = max(MATCH_PX_BASE, 1.2 * t['size']) * min(i - t['last'], 3)
                # Range continuity. Two gates that overlap in image are still metres apart
                # in depth, so a jump in implied range is a splice even when the pixels
                # line up.
                gap = i - t['last']
                if abs(d['range_m'] - t['range']) > 3.0 + CLOSING_M_PER_FRAME * gap:
                    continue
                if dist < lim and dist < bd:
                    best, bd = tid, dist
            if best is None:
                best = next_tid
                next_tid += 1
                live[best] = {'range': d['range_m']}
            used.add(best)
            live[best].update(last=i, centre=d['centre'], size=d['size_px'],
                              range=d['range_m'])
            tracks[best].append((i, d))
            assigned[best] = d
        for tid in [t for t, v in live.items() if i - v['last'] > MAX_GAP]:
            del live[tid]
        per_frame.append((i, assigned))
    good = {t: v for t, v in tracks.items() if len(v) >= MIN_TRACK_LEN}
    if verbose:
        lens = sorted((len(v) for v in good.values()), reverse=True)
        print(f'  tracks >= {MIN_TRACK_LEN} obs: {len(good)}   longest {lens[:10]}')
    return good, per_frame


def anchor(S, good, per_frame, verbose=False, drop=()):
    """Active-gate window -> track identity, by NEAREST-IN-WINDOW voting. {tid: race_index}.

    THE FIRST VERSION OF THIS ANCHORED AT THE CROSSING INSTANT and produced ZERO usable
    pairs. The reason is worth keeping, because it is a property of the data and not of the
    code: at the moment of crossing the gate fills the frame and is clipped, so the track
    that owns it is the LAST and SHORTEST fragment of the approach -- 6 to 68 observations,
    0.2 to 3.8 s. Identity propagated backward along that fragment reaches almost no frames,
    and the map needs frames where the named gate shares the image with another named gate.

    The fix uses the window rather than the instant. Between crossing k-1 and crossing k the
    sim says gate k is the gate being flown at, and gates 0..k-1 are behind, so throughout
    that window gate k is the NEAREST gate in view -- not just at the end. Every frame of the
    window therefore votes, and a track inherits the identity of the window it most often
    leads. That picks the LONGEST fragment of the approach instead of the last, and several
    fragments of one gate can each be identified, which is exactly what is wanted.

    The crossing is still what makes any of it true: without a k -> k+1 advance the window
    is not evidence of anything, which is why `trusted` below is built from observed
    crossings only. On the strafing session, where the index never advances, this returns
    nothing at all rather than confidently labelling the nearest blob "gate 0".

    `drop` removes gates from the trusted set -- the leave-one-out hook.
    """
    cross, rev = S.crossings()
    trusted = set(cross)
    kmax = int(S.frame_gate.max())
    if kmax > 0:
        trusted.add(kmax)         # the last gate approached but never crossed (gate 16)
    trusted -= set(drop)

    votes = collections.defaultdict(collections.Counter)
    for i, assigned in per_frame:
        k = int(S.frame_gate[i])
        if k not in trusted:
            continue
        if S.excluded(i):
            # EXCLUDED FROM VOTING, NOT FROM MEASUREMENT. Both excluded regions are
            # reversals, and what a reversal breaks is the "the active gate is the nearest
            # gate" premise: flying backwards brings ALREADY-PASSED gates in front of the
            # camera while the index still names the gate ahead, so the nearest detection
            # is no longer the active one and a vote taken there would be a lie.
            #
            # It does not break the measurements. A pair of gates identified from clean
            # windows measures the same fixed distance from inside the backtrack as from
            # anywhere else, and those frames are the ONLY place gates 9 and 10 are ever
            # co-visible -- dropping them outright split the map into two components that
            # nothing could bridge.
            continue
        cand = [(d['range_m'], tid) for tid, d in assigned.items()
                if tid in good and clean(d)]
        if not cand:
            continue
        votes[min(cand)[1]][k] += 1

    ident, detail = {}, collections.defaultdict(list)
    for tid, c in votes.items():
        k, n = c.most_common(1)[0]
        if n < NEAREST_VOTES or n < 0.6 * sum(c.values()):
            continue
        # THE CLOSE-APPROACH TEST, and it is the one that makes the vote honest.
        #
        # "The active gate is the nearest thing in view" fails in exactly one place: when
        # the active gate is SO near that it clips the image and stops passing clean(). The
        # nearest surviving detection is then a checkerboard square on that same gate's
        # frame, implying 30 m, and it collects the window's votes. Drawn on the frames this
        # is unmistakable -- gate 15 labelled twice on two decorations of the gate in front
        # of it -- and it is invisible in every aggregate.
        #
        # The fix is physical rather than statistical: the aircraft flew THROUGH gate k, so
        # gate k must at some point have been metres away. Decoration on a near gate's frame
        # and a gate 30 m down the hangar both fail that outright. Measured on the raw
        # detections, not the clean ones, because at 2 m the real gate is clipped by
        # construction.
        rmin = min(d['range_m'] for _i, d in good[tid])
        if k in cross and not cross_consistent(good[tid], cross[k], rmin):
            continue
        ident[tid] = k
        detail[k].append((n, tid, len(good[tid]), rmin))
    # PER-GATE CONTRADICTION PRUNING, added for the pausing lap. Long active-gate windows
    # (20-30 s of hovering) let several tracks each accumulate enough nearest-in-window
    # votes to be named the same gate -- eleven tracks carried "gate 13" at ranges from
    # 2 m to 44 m, one of them with range INCREASING through the approach. A gate is one
    # object: two tracks named g that are both observed in the same frame at ranges more
    # than 3 m apart cannot both be right. The track with the CLOSEST approach wins, because
    # the aircraft demonstrably flew to the gate; the contradicting fragment loses its name
    # entirely (its frames where the two never co-occur would otherwise keep poisoning the
    # pair table -- frame-level refusal alone is not enough).
    bygate = collections.defaultdict(list)
    for tid, k in ident.items():
        bygate[k].append(tid)
    for k, tids in sorted(bygate.items()):
        tids.sort(key=lambda t: min(d['range_m'] for _i, d in good[t]))
        kept = []
        for t in tids:
            fr = {i: d['range_m'] for i, d in good[t]}
            clash = None
            for t2 in kept:
                fr2 = {i: d['range_m'] for i, d in good[t2]}
                bad = [i for i in fr.keys() & fr2.keys() if abs(fr[i] - fr2[i]) > 3.0]
                if bad:
                    clash = t2
                    break
            if clash is None:
                kept.append(t)
            else:
                del ident[t]
                detail[k] = [c for c in detail[k] if c[1] != t]
                if verbose:
                    print(f'  gate {k:2d}: track {t} un-named, contradicts track {clash} '
                          f'in-frame by >3 m')
    # The LAST gate was never crossed, so no close approach is available to test it with.
    # gate5.py's rule stands in: among the candidates take the NEAREST, not the most-voted,
    # and take only one.
    if kmax in trusted and kmax not in cross:
        cands = detail.get(kmax, [])
        if cands:
            best = min(cands, key=lambda c: min(d['range_m'] for _i, d in good[c[1]]))
            for _n, tid, _o, _r in cands:
                if tid != best[1]:
                    del ident[tid]
            detail[kmax] = [best]
    if verbose:
        for k in sorted(trusted):
            fr = [tid for tid, g in ident.items() if g == k]
            obs = sum(len(good[t]) for t in fr)
            spans = []
            for t in sorted(fr, key=lambda t: -len(good[t]))[:3]:
                oo = [i for i, _ in good[t]]
                spans.append(f'{S.fid[min(oo)]}-{S.fid[max(oo)]}')
            tag = '' if k in cross else '   <- NO CROSSING (last gate, approach only)'
            print(f'  gate {k:2d}: {len(fr):2d} track(s), {obs:5d} obs   '
                  f'{" ".join(spans)}{tag}')
    return ident, cross, rev


def cross_consistent(obs, ic, rmin):
    """Is this track physically consistent with being the gate crossed at frame ic?

    The original test was a flat `rmin <= CROSS_MAX_R`, and the pausing lap broke it for
    gates 2, 3, 8 and 9: pausing fragments the approach into a long hover fragment (which
    wins the window vote) and a short final dash (which is clipped and never forms a clean
    track), so the vote-winning fragment's own rmin sits at 13-20 m and the flat test threw
    the identity away. All four losses were majority-vote, correct-window tracks.

    The physics the flat test was standing in for is the CROSSING TIME: the aircraft is at
    gate k's position at frame ic, and range can close no faster than the aircraft flies
    (CLOSING_M_PER_FRAME, same constant tracking uses). So:

      * a track still observed near ic must actually BE near: min range inside the last
        second before the crossing <= CROSS_MAX_R, else it is some other gate watched while
        crossing this one -- the failure the flat test existed to kill, still killed;
      * a track that ENDED g frames before ic is consistent iff its final range could still
        shrink to ~0 by ic: rmin <= CROSS_MAX_R + CLOSING_M_PER_FRAME * g.

    Decoration no longer needs this test at all -- clean() rejects it on interior colour,
    and votes only come from clean detections.
    """
    # The BINDING observation is the one nearest the crossing instant, and it must sit
    # inside the closing funnel r <= 5 + CLOSING_M_PER_FRAME * (ic - i). Using the track's
    # min range instead lets a track that once read 11 m but reads 15 m six frames before
    # the crossing pass -- 15 m in 0.2 s is 75 m/s, and that track (1845, window 12) was a
    # DIFFERENT gate watched while gate 12 was being crossed.
    pre = [(i, d['range_m']) for i, d in obs if i <= ic + 5]
    if not pre:
        return False       # exists only after the crossing: cannot be the gate crossed
    i0, r0 = max(pre)
    if ic - i0 <= 90:      # observed within 3 s of the crossing: the funnel binds
        return r0 <= 5.0 + CLOSING_M_PER_FRAME * max(ic - i0, 0)
    # ended long before the crossing (a pause, then the dash): consistent iff the final
    # range could still shrink to ~0 in the time that remained
    return r0 <= CROSS_MAX_R + CLOSING_M_PER_FRAME * (ic - i0)


def anchor_active(S, good, k, exclude_tids=()):
    """Identity for a gate with NO crossing, from the sim's own active-gate window.

    gate5.py's method, and its lesson: within that window the NEAREST cluster is the active
    gate, not the most-detected one. Density ranks far gates first because they stay whole in
    frame while the near one clips out -- the exact failure its gate-4 leave-one-out caught.
    """
    win = S.active_window(k)
    if win is None:
        return None
    i0, i1 = win
    cands = []
    for tid, obs in good.items():
        if tid in exclude_tids:
            continue
        rr = [d['range_m'] for i, d in obs if i0 <= i <= i1 and clean(d)]
        if len(rr) < 6:
            continue
        cands.append((float(np.median(rr)), tid, len(rr)))
    if not cands:
        return None
    cands.sort()
    return cands[0][1]


def extend_back(S, obs, ident_at, gate, max_miss=90):
    """Follow ONE known gate backwards in time from the earliest frame it was named in.

    THE BOTTLENECK THIS EXISTS TO BREAK. A crossing names gate k, and that name reaches back
    only as far as the multi-target tracker's chain survives -- typically 4 to 10 s, which
    ends inside gate k's own approach. Gate k-1's name reaches back a similar distance from
    its own crossing. The two intervals barely overlap, so the number of frames in which TWO
    NAMED gates share the image came out at three pairs across both laps. A map needs
    hundreds.

    A single-target tracker is a much easier problem than the multi-target one and it is the
    right tool once identity is known: there is exactly one object, its identity is not in
    doubt, and every rival detection can be rejected rather than assigned. So the greedy
    association that has to be conservative when it is deciding WHAT something is can be
    generous when it already knows.

    Three guards, all continuity rather than appearance:
      * range may change no faster than a drone flies (CLOSING_M_PER_FRAME);
      * apparent size may not jump by more than MATCH_SIZE_RATIO;
      * the search radius scales with the coasting gap but is CAPPED, for the reason
        build_tracks() records at length.
    `max_miss` is 90 frames (3 s) of coasting. Swept: 45 frames recovers 7 gate-to-gate
    pairs across both laps, 90 recovers 18, and 150 recovers no more -- so 3 s is where the
    method saturates rather than a number chosen to make the answer look better.
    And one identity guard: if the candidate is a detection already named as a DIFFERENT
    gate in that frame, stop. Walking backwards onto the neighbouring gate is the failure
    mode with teeth, and it announces itself exactly there.
    """
    obs = sorted(obs, key=lambda o: o[0])
    i0, d0 = obs[0]
    cur = {'centre': d0['centre'], 'size': d0['size_px'], 'range': d0['range_m']}
    out, miss = [], 0
    for i in range(i0 - 1, -1, -1):
        if S.excluded(i):
            break
        gap = miss + 1
        best, bd = None, 1e18
        for d in S.det.get(S.frames[i]['file'], []):
            if abs(d['range_m'] - cur['range']) > 3.0 + CLOSING_M_PER_FRAME * gap:
                continue
            ratio = d['size_px'] / max(cur['size'], 1e-6)
            if not (1 / MATCH_SIZE_RATIO < ratio < MATCH_SIZE_RATIO):
                continue
            dist = float(np.linalg.norm(d['centre'] - cur['centre']))
            lim = max(MATCH_PX_BASE, 1.2 * cur['size']) * min(gap, 3)
            if dist < lim and dist < bd:
                best, bd = d, dist
        if best is None:
            miss += 1
            if miss > max_miss:
                break
            continue
        other = ident_at.get(i, {})
        if any(g != gate and o is best for g, o in other.items()):
            break
        miss = 0
        cur.update(centre=best['centre'], size=best['size_px'], range=best['range_m'])
        out.append((i, best))
    return out


# ---------------------------------------------------------------------------------------
# pair measurements


def measure(S, good, ident, verbose=False):
    """Co-visible identified pairs -> distance and height difference, one row per frame.

    Rows are kept individually rather than averaged on the fly so that the SPREAD of one
    gate pair measured from many viewpoints is available. That spread is the map's real
    precision, and it is also the tell for a merged-gate or clipped-gate failure: two static
    gates whose measured separation changes with viewpoint is a measurement fault.
    """
    rows = []
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, obs in good.items():
        # NODE, not gate. Every track is a node in the distance graph; an IDENTIFIED track
        # collapses onto its gate's node (so all fragments of gate 5 are one node), and an
        # UNIDENTIFIED track keeps its own.
        #
        # Keeping the unidentified ones is the whole reason the map has a shape. Identity
        # only ever reaches backward as far as a track survives, so the pairs it produces
        # are almost entirely CONSECUTIVE gates -- and a chain of consecutive distances is a
        # string of free hinges, with no shape at all. A track that is never nearest and so
        # never named is still a fixed point in the hangar seen from many viewpoints, and it
        # cross-braces the chain: gate 3 -- (unnamed) -- gate 7 constrains the angle between
        # them without anyone ever having to say which gate the bridge is.
        node = ('g', ident[tid]) if tid in ident else ('t', tid)
        for i, d in obs:
            if clean(d):
                raw[i][node].append(d)
    # The contradiction check below must see EVERY observation a named track has, clean or
    # not: on the pausing lap the surviving impostor rows were exactly the frames where the
    # true gate's detection failed clean() (clipped, oblique) while the impostor's passed,
    # so a clean-only check waved them through.
    claims = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, obs in good.items():
        if tid in ident:
            for i, d in obs:
                claims[i][ident[tid]].append(d['range_m'])
    # A GATE APPEARS ONCE PER FRAME. Two detections in one frame carrying the same gate
    # identity is a contradiction -- one of the fragments is misnamed -- and the previous
    # version of this function hid it: byframe[i][node] was a plain dict, so whichever
    # fragment happened to be inserted last won, silently. On the pausing lap that poisoned
    # pair 0-1 with a bimodal 20 m / 34 m distribution (a far gate through gate 0's aperture
    # had inherited "gate 1"). Verified on frame 25398: two detections both labelled gate 0,
    # at 5 m and at 24 m. The refusal, not an arbitration: if a gate's claimed detections in
    # a frame disagree by more than 3 m of range, that gate is AMBIGUOUS in that frame and
    # contributes nothing there. Same-object duplicates (a crossing track and a backward
    # extension sharing the detection) agree in range and collapse harmlessly.
    byframe = {}
    n_amb = 0
    for i, gd in raw.items():
        row = {}
        for node, ds in gd.items():
            rr = [d['range_m'] for d in ds]
            if node[0] == 'g':
                cl = claims[i][node[1]]
                if max(cl) - min(cl) > 3.0:
                    n_amb += 1
                    continue
            row[node] = ds[int(np.argmin(rr))]
        byframe[i] = row
    if verbose and n_amb:
        print(f'  ambiguous gate-frames refused (same gate claimed at >3 m apart): {n_amb}')
    mk = S.marker_frames()
    for i, gd in byframe.items():
        if len(gd) < 2:
            continue
        gh = S.gravity(i)
        ks = sorted(gd)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                ga, gb = ks[a], ks[b]
                pa, pb = gd[ga]['pos_body'], gd[gb]['pos_body']
                v = pb - pa
                d = float(np.linalg.norm(v))
                # Two distinct race gates are never a metre apart. A near-zero reading means
                # one detection has been claimed by two identities, which is a data
                # association failure, not a measurement -- refuse it rather than average it
                # in. See the mutual-exclusion note in run().
                if d < 1.0:
                    continue
                dz = float(-(v @ gh)) if gh is not None else np.nan
                h = float(np.sqrt(max(d * d - dz * dz, 0.0))) if gh is not None else np.nan
                rows.append({'i': i, 'fid': int(S.fid[i]), 'pair': (ga, gb), 'd': d,
                             'dz': dz, 'h': h, 'mk': i in mk,
                             'ra': float(gd[ga]['range_m']), 'rb': float(gd[gb]['range_m'])})
    if verbose:
        print(f'  identified co-visible pair measurements: {len(rows)} '
              f'from {len(byframe)} frames')
    return rows


def aggregate(rows):
    """Per gate pair: robust distance, height difference, and the scatter of both."""
    acc = collections.defaultdict(list)
    for r in rows:
        acc[r['pair']].append(r)
    out = {}
    for p, rs in acc.items():
        # TWO STATIC GATES HAVE ONE SEPARATION. A pair whose measurements split into two
        # well-separated modes is a data-association fault by definition -- on the pausing
        # lap, gate 8's backward extension coasted through a yaw sweep and re-attached to a
        # farther gate, so pair 7-8 read 11 m in 44 rows and 27 m in 18. When that happens
        # the cluster containing Claire's MARKER moments wins (she pressed the key while
        # LOOKING at the gates -- an external referee); with no marker rows on either side,
        # the larger cluster wins. Either way the losing mode is dropped, not averaged in,
        # and the pair is flagged bimodal.
        bimodal = 0
        if len(rs) >= 6:
            o = sorted(range(len(rs)), key=lambda j: rs[j]['d'])
            ds = [rs[j]['d'] for j in o]
            gaps = [ds[j + 1] - ds[j] for j in range(len(ds) - 1)]
            j = int(np.argmax(gaps))
            if gaps[j] > 5.0 and j + 1 >= 3 and len(ds) - j - 1 >= 3:
                lo, hi = [rs[t] for t in o[:j + 1]], [rs[t] for t in o[j + 1:]]
                key = lambda c: (sum(bool(r.get('mk')) for r in c), len(c))
                win = lo if key(lo) >= key(hi) else hi
                bimodal = len(rs) - len(win)
                rs = win
        d = np.array([r['d'] for r in rs])
        hz = np.array([r['dz'] for r in rs if np.isfinite(r['dz'])])
        hh = np.array([r['h'] for r in rs if np.isfinite(r['h'])])
        if len(d) < MIN_PAIR_OBS:
            continue
        med = float(np.median(d))
        out[p] = {
            'n': int(len(d)), 'd': med,
            'd_mad': float(np.median(np.abs(d - med))),
            'd_p10': float(np.percentile(d, 10)), 'd_p90': float(np.percentile(d, 90)),
            'n_h': int(len(hz)),
            'dz': float(np.median(hz)) if len(hz) >= MIN_PAIR_OBS else None,
            'dz_mad': float(np.median(np.abs(hz - np.median(hz)))) if len(hz) >= MIN_PAIR_OBS else None,
            'h': float(np.median(hh)) if len(hh) >= MIN_PAIR_OBS else None,
            'range_min': float(min(min(r['ra'], r['rb']) for r in rs)),
            'range_med': float(np.median([max(r['ra'], r['rb']) for r in rs])),
            'bimodal_dropped': bimodal,
            'n_marker': int(sum(bool(r.get('mk')) for r in rs)),
        }
    return out


# ---------------------------------------------------------------------------------------
# solve


def node_list(agg):
    """Every node with at least one measured edge, gates first and in race order."""
    ns = sorted({n for p in agg for n in p})
    gates = [n for n in ns if n[0] == 'g']
    return gates + [n for n in ns if n[0] != 'g']


def solve_heights(agg, nodes):
    """Least squares over the pairwise height differences -> one height per node.

    Yaw-free and over-determined: every co-visible pair contributes one equation, and the
    residual is what says whether they agree. This is the only part of the map that needs no
    embedding at all -- gravity gives the axis directly, so heights are a linear system, not
    an optimisation with local minima.

    Weighted by 1/(1 + MAD): a pair whose height difference wobbles across viewpoints is
    measuring badly and should not drag the solution.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    m = len(nodes)
    A, y, w, used = [], [], [], []
    for (a, b), v in sorted(agg.items(), key=lambda kv: str(kv[0])):
        if v['dz'] is None or a not in idx or b not in idx:
            continue
        row = np.zeros(m)
        row[idx[b]] += 1.0
        row[idx[a]] -= 1.0
        A.append(row)
        y.append(v['dz'])
        w.append(1.0 / (1.0 + v['dz_mad']))
        used.append((a, b))
    if not A:
        return None, None, []
    A, y = np.array(A), np.array(y)
    W = np.sqrt(np.array(w))
    # Gauge. The system is singular by construction (only differences are observed), so one
    # node is pinned: gate 0 if it is present, else the first node.
    g0 = idx.get(('g', 0), 0)
    Afix = np.vstack([A * W[:, None], np.eye(m)[g0] * 100.0])
    yfix = np.concatenate([y * W, [0.0]])
    z, *_ = np.linalg.lstsq(Afix, yfix, rcond=None)
    return z, A @ z - y, used


def smacof(Dobs, Wt, X0, iters=800):
    """Weighted MDS by stress majorisation, on OBSERVED pairs only.

    Unobserved pairs get weight ZERO rather than a shortest-path guess. mapbuild.py used a
    Floyd-Warshall completion and NOTES.md records what that cost: nine nodes came out
    exactly collinear across 25 m, positioned by the completion rather than by any distance
    that was ever measured, and their coordinates were an artifact. Leaving a pair
    unconstrained is honest; inventing a distance for it is not. The completion survives
    here only as the SEED, which the iteration is free to leave.
    """
    X = X0.copy()
    Wsum = Wt.sum(axis=1)
    for _ in range(iters):
        Dc = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(Dc > 1e-9, Dobs / Dc, 0.0)
        B = -Wt * ratio
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        Xn = (B @ X) / np.maximum(Wsum, 1e-9)[:, None]
        if np.max(np.abs(Xn - X)) < 1e-10:
            return Xn
        X = Xn
    return X


def edge_weight(v):
    """How much one measured edge should count.

    Three penalties, each for a failure this project has actually seen: scatter across
    viewpoints (the merged-gate signature), long range (PnP range error grows with it), and
    few observations.
    """
    s = v['d_mad'] + 0.05 * v['range_med'] + 0.5
    return float(min(v['n'], 200) ** 0.5 / s ** 2)


def solve_layout(agg, nodes, seed=None, iters=1200):
    """Horizontal layout from measured horizontal distances. Rotation- and reflection-free.

    Distances alone cannot fix a handedness, so the answer is a shape, not an orientation.
    That is not a shortcoming for guidance -- "how far and which way round is gate k+1 from
    gate k" is a relative query -- but it does mean the comparison against Claire's sketch
    has to allow a reflection and say which one it chose.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    m = len(nodes)
    Dobs = np.zeros((m, m))
    Wt = np.zeros((m, m))
    for (a, b), v in agg.items():
        if a not in idx or b not in idx:
            continue
        h = v['h'] if v['h'] is not None else v['d']
        i, j = idx[a], idx[b]
        Dobs[i, j] = Dobs[j, i] = h
        Wt[i, j] = Wt[j, i] = edge_weight(v)
    # Keep the largest connected component: a node reachable from nothing gets its position
    # from nothing, and mapbuild.py's collinear-artifact nodes were exactly that case.
    seen, comp = set(), []
    adj = {i: np.flatnonzero(Wt[i] > 0).tolist() for i in range(m)}
    for s0 in range(m):
        if s0 in seen:
            continue
        stack, cur = [s0], []
        seen.add(s0)
        while stack:
            u = stack.pop()
            cur.append(u)
            for v2 in adj[u]:
                if v2 not in seen:
                    seen.add(v2)
                    stack.append(v2)
        comp.append(cur)
    keep = sorted(max(comp, key=len))
    nodes = [nodes[i] for i in keep]
    Dobs = Dobs[np.ix_(keep, keep)]
    Wt = Wt[np.ix_(keep, keep)]
    m = len(keep)

    if seed is not None:
        X0 = seed
    else:
        Df = np.where(Wt > 0, Dobs, np.inf)
        np.fill_diagonal(Df, 0.0)
        for k in range(m):
            Df = np.minimum(Df, Df[:, k, None] + Df[None, k, :])
        D2 = Df ** 2
        J = np.eye(m) - np.ones((m, m)) / m
        B = -0.5 * J @ D2 @ J
        w, V = np.linalg.eigh(B)
        o = np.argsort(w)[::-1][:2]
        X0 = V[:, o] * np.sqrt(np.clip(w[o], 0, None))
    X = smacof(Dobs, Wt, X0, iters)
    Dc = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    msk = np.triu(Wt > 0, 1)
    resid = (Dc - Dobs)[msk]
    return nodes, X, {'n_nodes': m, 'n_edges': int(msk.sum()),
                      'stress_median_m': float(np.median(np.abs(resid))),
                      'stress_p90_m': float(np.percentile(np.abs(resid), 90))}


# ---------------------------------------------------------------------------------------
# the sketch


def sketch_xy(mapjson=None):
    """map_approx.json -> (17, 2) array in STATION UNITS, race-indexed.

    `across` is rescaled by row_separation/station_pitch so both axes are station units,
    which is what readmap.map_pair_distances() does and the only way a distance in the
    sketch means anything.
    """
    m = json.load(open(mapjson or os.path.join(HERE, 'map_approx.json')))
    k = m['row_separation_px'] / m['station_pitch_px']
    P = np.zeros((17, 2))
    for g in m['gates']:
        P[g['race_index']] = [g['along_station'], g['across'] * k]
    return P, m


def procrustes(A, B, allow_reflect=True):
    """Similarity transform taking A onto B. -> (A_fitted, scale, rms, reflected).

    Reflection is ALLOWED and reported. A map built from distances alone is fixed only up to
    reflection -- there is no handedness in a set of scalars -- so refusing reflection would
    not make the answer more certain, it would just hide which of the two the sketch prefers.
    """
    best = None
    for sgn in ((1, -1) if allow_reflect else (1,)):
        A2 = A * np.array([1.0, sgn])
        ca, cb = A2.mean(0), B.mean(0)
        X, Y = A2 - ca, B - cb
        U, s, Vt = np.linalg.svd(X.T @ Y)
        R = U @ Vt
        scale = s.sum() / max((X ** 2).sum(), 1e-12)
        F = scale * (X @ R) + cb
        rms = float(np.sqrt(((F - B) ** 2).sum(1).mean()))
        if best is None or rms < best[2]:
            best = (F, scale, rms, sgn < 0)
    return best


# ---------------------------------------------------------------------------------------
# pipeline


def run(session, verbose=True, drop=()):
    """Full identity + measurement pass on one session. -> dict of everything downstream
    needs, so that a second lap can be run through the identical code and compared."""
    S = Session(session)
    if verbose:
        print(f'\n=== {S.name}  {len(S.frames)} frames')
    good, per_frame = build_tracks(S, verbose)
    ident, cross, rev = anchor(S, good, per_frame, verbose, drop=drop)
    if verbose and rev:
        print(f'  index reversals in race.csv: '
              + ', '.join(f'{a}->{b}@{S.fid[i]}' for a, b, i in rev))
    # Reach every named gate backwards past where the multi-target tracker gave up.
    ident_at = collections.defaultdict(dict)
    for tid, obs in good.items():
        if tid in ident:
            for i, d in obs:
                ident_at[i][ident[tid]] = d
    n_add = 0
    for g in sorted(set(ident.values())):
        seeds = [obs for tid, obs in good.items() if ident.get(tid) == g]
        if not seeds:
            continue
        earliest = min(seeds, key=lambda o: min(i for i, _d in o))
        ext = extend_back(S, earliest, ident_at, g)
        if ext:
            n_add += len(ext)
            tid = -1 - g                     # synthetic track id for the extension
            good[tid] = ext
            ident[tid] = g
            # MUTUAL EXCLUSION, and it is not a nicety. Without it every gate's backward
            # walk is blind to every other's, and gates 6, 7, 8 and 9 all coasted onto the
            # SAME physical gate: their pairwise distances came out 0.00 m with n=151 and a
            # MAD of zero, and 4-6, 4-7, 4-8, 4-9 were byte-identical. A distance of zero
            # between two race gates is impossible, which is what made it visible -- but
            # only because the pair table was printed. Registering each extension as it is
            # made lets the next walk see it and stop.
            for i, d in ext:
                ident_at[i][g] = d
    if verbose:
        print('  backward extension added %d observations' % n_add)
    rows = measure(S, good, ident, verbose)
    # rows refused after frame verification (see REFUSE_ROWS) -- refusal, not correction
    refuse = {p for (sname, p), _why in REFUSE_ROWS.items() if sname == S.name}
    if refuse:
        n0 = len(rows)
        rows = [row for row in rows
                if not (row['pair'][0][0] == 'g' and row['pair'][1][0] == 'g'
                        and (row['pair'][0][1], row['pair'][1][1]) in refuse)]
        if verbose and n0 != len(rows):
            print(f'  REFUSE_ROWS dropped {n0 - len(rows)} rows for pairs '
                  f'{sorted(refuse)} (render-contradicted; see mapvq2.REFUSE_ROWS)')
    rows = relabel_rows(rows, S.name, verbose)
    agg = aggregate(rows)
    seen = sorted({g for p in agg for g in p})
    if verbose:
        print(f'  gates with at least one measured pair: {len(seen)}  {seen}')
    if verbose and S.markers:
        # Did the markers deliver? Each one was pressed at a moment Claire judged two or
        # three gates fully visible, so each should yield NAMED co-visible pairs. A marker
        # that yields none is a marker whose gates the tracker failed to name -- worth
        # seeing per marker, not as an average.
        mf = S.marker_frames()
        byfr = collections.defaultdict(set)
        for tid, obs in good.items():
            if tid in ident:
                for i, d in obs:
                    if clean(d):
                        byfr[i].add(ident[tid])
        inwin = [r for r in rows if r['i'] in mf]
        gg = [r for r in inwin if r['pair'][0][0] == 'g' and r['pair'][1][0] == 'g']
        print(f'  markers: {len(S.markers)}   pair rows in marker windows: {len(inwin)} '
              f'({len(gg)} gate-gate)')
        for mi, (t, g) in enumerate(S.markers, 1):
            lo = int(np.searchsorted(S.t, t - MARKER_PAD_S * 1e9))
            hi = int(np.searchsorted(S.t, t + MARKER_PAD_S * 1e9))
            named = sorted(set().union(*[byfr.get(i, set())
                                         for i in range(lo, min(hi, len(S.frames)))] or [set()]))
            print(f'    marker {mi:2d} (active {g:2d}): named gates co-visible {named}')
    return {'S': S, 'good': good, 'ident': ident, 'cross': cross,
            'rows': rows, 'agg': agg}


def gate_report(agg):
    """Only the GATE-to-GATE edges, which are the ones a human can check."""
    ge = {(a[1], b[1]): v for (a, b), v in agg.items() if a[0] == 'g' and b[0] == 'g'}
    print(f'\n--- {len(ge)} directly measured gate-to-gate pairs (of 136 possible)')
    print('  pair      n    dist m   MAD   p10-p90       horiz     dz m   n_h  nearest'
          '  n_mark  dropped')
    for (a, b), v in sorted(ge.items()):
        print('  %2d-%-2d %5d %8.2f %6.2f  %5.1f-%-5.1f %8.2f %+7.2f %5d %7.1f %6d %7d'
              % (a, b, v['n'], v['d'], v['d_mad'], v['d_p10'], v['d_p90'],
                 v['h'] if v['h'] is not None else float('nan'),
                 v['dz'] if v['dz'] is not None else float('nan'),
                 v['n_h'], v['range_min'], v.get('n_marker', 0),
                 v.get('bimodal_dropped', 0)))
    return ge


def solve(res, verbose=True):
    """agg -> 17 gate positions (metres, gravity-aligned) with per-gate uncertainty."""
    agg = res['agg']
    nodes = node_list(agg)
    z, zres, zused = solve_heights(agg, nodes)
    nodes2, X, st = solve_layout(agg, nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    idx2 = {n: i for i, n in enumerate(nodes2)}
    out = {}
    for k in range(17):
        n = ('g', k)
        if n not in idx2:
            continue
        out[k] = np.array([X[idx2[n]][0], X[idx2[n]][1],
                           z[idx[n]] if z is not None else np.nan])
    if verbose:
        print(f'\n--- layout: {st["n_nodes"]} nodes ({sum(1 for n in nodes2 if n[0]=="g")} '
              f'gates + {sum(1 for n in nodes2 if n[0]!="g")} unnamed bridges), '
              f'{st["n_edges"]} edges')
        print(f'    MDS stress on OBSERVED edges: median {st["stress_median_m"]:.2f} m  '
              f'p90 {st["stress_p90_m"]:.2f} m')
        if zres is not None:
            print(f'    height residual: median |r| {np.median(np.abs(zres)):.2f} m  '
                  f'p90 {np.percentile(np.abs(zres), 90):.2f} m  over {len(zres)} pairs')
    return out, {'layout': st, 'nodes': nodes2, 'X': X,
                 'height_resid': None if zres is None else zres.tolist()}


def report(res):

    S, agg = res['S'], res['agg']
    print(f'\n--- {len(agg)} measured gate pairs (of 136 possible)')
    print('  pair      n    dist m   MAD   p10-p90        horiz    dz m   n_h   nearest')
    for (a, b), v in sorted(agg.items()):
        print('  %2d-%-2d %5d %8.2f %6.2f  %5.1f-%-5.1f %8.2f %+7.2f %5d %7.1f'
              % (a, b, v['n'], v['d'], v['d_mad'], v['d_p10'], v['d_p90'],
                 v['h'] if v['h'] is not None else float('nan'),
                 v['dz'] if v['dz'] is not None else float('nan'),
                 v['n_h'], v['range_min']))
    cons = [(k, agg[(k, k + 1)]['d']) for k in range(16) if (k, k + 1) in agg]
    print(f'\n  consecutive-gate distances measured: {len(cons)} of 16')
    for k, d in cons:
        print(f'    {k}->{k+1}: {d:.2f} m')


STRAFE_WINDOW = (59217, 59975)   # session-notes.txt: only gates 15 and 16 in view


def strafe_rows(session=None, window=STRAFE_WINDOW):
    """The 15/16 separation from the strafing session, on Claire's word rather than the index.

    `checksession.py` FAILS this recording -- `active_gate_index` never advances, so nothing
    in it can name a gate. What it has instead is an external observation: Claire flew it
    deliberately with only gates 15 and 16 in view, over a stated frame range. That is a
    referee outside the sim, which is the standing rule's own definition of what settles
    something, and it is the only source here for the far end of the course, where both laps
    are thinnest.

    What her note does NOT say is which of the two is which. So this contributes the PAIR
    separation and the magnitude of the height step, and nothing else -- frames with exactly
    two clean detections, no more, because a third would make even the pair ambiguous. The
    orientation of the step is left to the laps, which do know.

    Lateral motion is the reason this is worth having at all: a head-on approach leaves range
    error almost entirely along the approach axis (gate5.py's finding), and strafing is
    precisely the geometry that breaks that degeneracy.
    """
    S = Session(session or STRAFE)
    rows = []
    for i, r in enumerate(S.frames):
        if not (window[0] <= S.fid[i] <= window[1]):
            continue
        det = [d for d in S.det.get(r['file'], []) if clean(d)]
        if len(det) != 2:
            continue
        gh = S.gravity(i)
        v = det[1]['pos_body'] - det[0]['pos_body']
        d = float(np.linalg.norm(v))
        dz = abs(float(v @ gh)) if gh is not None else np.nan
        rows.append({'i': i, 'fid': int(S.fid[i]), 'pair': (('g', 15), ('g', 16)),
                     'd': d, 'dz': np.nan, 'h': float(np.sqrt(max(d * d - dz * dz, 0.0)))
                     if gh is not None else np.nan,
                     'abs_dz': dz,
                     'ra': float(det[0]['range_m']), 'rb': float(det[1]['range_m']),
                     'session': S.name})
    return rows


def run_many(sessions, verbose=True):
    """Pool several recordings into ONE distance graph.

    Needed, not merely nicer. The primary lap's graph splits into two components at gates
    9|10: Claire's 32 s backtrack sits exactly there, and across it no identified gate is
    ever co-visible with another, so nothing bridges 9 to 10. The second lap flies the same
    stretch without a backtrack and measures 9-10 and 10-11 directly.

    Unnamed bridge tracks are namespaced by session -- track 42 of one lap is not track 42
    of another -- while GATE nodes are shared, which is the whole point: gate 7 is gate 7 in
    both recordings, so pooling adds edges without adding unknowns.
    """
    rows = []
    per = {}
    for sess in sessions:
        r = run(sess, verbose=verbose)
        per[r['S'].name] = r
        tag = r['S'].name[-4:]
        for row in r['rows']:
            a, b = row['pair']
            a = a if a[0] == 'g' else ('t', f'{tag}:{a[1]}')
            b = b if b[0] == 'g' else ('t', f'{tag}:{b[1]}')
            rows.append({**row, 'pair': (a, b) if a <= b else (b, a),
                         'session': r['S'].name})
    return {'rows': rows, 'agg': aggregate(rows), 'per': per}


# ---------------------------------------------------------------------------------------
# checks


def scale_check(agg, mapjson=None):
    """metres per station, straight from identified pairs. No embedding involved.

    This is the cleanest test of the 15.97 m/station in map_approx.json, because every row
    is an independent measurement of one constant: a measured metre distance between two
    gates whose separation the sketch states in station units. The SPREAD across pairs is
    the error bar -- they cannot all be right if they disagree.
    """
    P, m = sketch_xy(mapjson)
    out = []
    for (a, b), v in sorted(agg.items()):
        if a[0] != 'g' or b[0] != 'g':
            continue
        u = float(np.linalg.norm(P[a[1]] - P[b[1]]))
        if u < 1e-6:
            continue
        h = v['h'] if v['h'] is not None else v['d']
        out.append({'pair': (a[1], b[1]), 'u': u, 'h': h, 'scale': h / u,
                    'n': v['n'], 'mad': v['d_mad']})
    return out, m


def loo_edges(agg, nodes):
    """Leave-one-edge-out: hide a measured gate-gate distance, rebuild, predict it back.

    gate5.py's discipline applied to a map instead of a point. A map that merely fits every
    number it was given says nothing; a map that predicts a distance it never saw is being
    tested. Cheap -- one re-solve per held-out edge -- and it is the only figure here that
    is not, in some way, scoring the fit against its own input.
    """
    ge = [(k, v) for k, v in agg.items() if k[0][0] == 'g' and k[1][0] == 'g']
    out = []
    for (a, b), v in ge:
        sub = {k: vv for k, vv in agg.items() if k != (a, b)}
        n2, X, _st = solve_layout(sub, node_list(sub))
        idx = {n: i for i, n in enumerate(n2)}
        if a not in idx or b not in idx:
            out.append({'pair': (a[1], b[1]), 'measured': v['h'] or v['d'],
                        'predicted': None, 'err': None})
            continue
        pred = float(np.linalg.norm(X[idx[a]] - X[idx[b]]))
        meas = v['h'] if v['h'] is not None else v['d']
        out.append({'pair': (a[1], b[1]), 'measured': meas, 'predicted': pred,
                    'err': pred - meas, 'n': v['n']})
    return out


def jackknife(rows, blocks=8, seed=0):
    """Per-gate position uncertainty by deleting time blocks of the raw measurements.

    Not a bootstrap over pair AVERAGES: consecutive frames of one approach are heavily
    correlated, so resampling them independently would report centimetres and mean nothing
    -- gate5.py's note on exactly this. Deleting a whole contiguous block of frames removes
    a correlated chunk at once, so the scatter reflects "how much does this gate move if a
    viewpoint is taken away", which is the question.
    """
    fids = np.array([r['fid'] for r in rows])
    edges = np.percentile(fids, np.linspace(0, 100, blocks + 1))
    ests = []
    for k in range(blocks):
        keep = [r for r, f in zip(rows, fids) if not (edges[k] <= f <= edges[k + 1])]
        if len(keep) < 100:
            continue
        agg = aggregate(keep)
        nodes = node_list(agg)
        z, _r, _u = solve_heights(agg, nodes)
        n2, X, _st = solve_layout(agg, nodes)
        idx, idz = {n: i for i, n in enumerate(n2)}, {n: i for i, n in enumerate(nodes)}
        ests.append({g: np.array([X[idx[('g', g)]][0], X[idx[('g', g)]][1],
                                  z[idz[('g', g)]]])
                     for g in range(17) if ('g', g) in idx})
    return ests
def align_rigid(A, B):
    """Rotate/reflect/translate A onto B WITHOUT rescaling. -> (A_fitted, rms, reflected).

    Scale is deliberately not free here: both arguments are already in metres, from the same
    1500 mm aperture, so allowing a scale would let a real metric disagreement be absorbed as
    "different units" -- which is precisely the disagreement the lap-to-lap check exists to
    expose. Rotation and reflection ARE free, because a map built from distances alone has no
    orientation and no handedness.
    """
    best = None
    for sgn in (1, -1):
        A2 = A * np.array([1.0, sgn])
        ca, cb = A2.mean(0), B.mean(0)
        X, Y = A2 - ca, B - cb
        U, _s, Vt = np.linalg.svd(X.T @ Y)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        F = (X @ R) + cb
        rms = float(np.sqrt(((F - B) ** 2).sum(1).mean()))
        if best is None or rms < best[1]:
            best = (F, rms, sgn < 0)
    return best


def components(agg):
    """Connected components of the measured distance graph, largest-gate-count first."""
    adj = collections.defaultdict(set)
    for a, b in agg:
        adj[a].add(b)
        adj[b].add(a)
    seen, out = set(), []
    for n in node_list(agg):
        if n in seen:
            continue
        stack, cur = [n], set()
        seen.add(n)
        while stack:
            u = stack.pop()
            cur.add(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        out.append(cur)
    out.sort(key=lambda c: (-sum(1 for n in c if n[0] == 'g'), -len(c)))
    return out


def build(sessions, verbose=True, strafe=False):
    """Pool sessions, then solve EACH CONNECTED COMPONENT separately.

    A single global solve was wrong twice over: solve_layout() silently kept only the
    largest component (48 nodes but just gates 0-2), and coordinates across disconnected
    components are not comparable anyway -- nothing measures one against another. So the
    map is a LIST of local maps, each in its own frame, and it says so.
    """
    pooled = run_many(sessions, verbose)
    if strafe:
        sr = strafe_rows()
        if verbose:
            hh = np.array([r['h'] for r in sr if np.isfinite(r['h'])])
            dz = np.array([r['abs_dz'] for r in sr if np.isfinite(r['abs_dz'])])
            print('\n=== strafing session, gates 15/16 by Claire\'s note (no race index)')
            print('    %d frames with exactly two clean detections' % len(sr))
            if len(hh):
                print('    horizontal separation  median %.2f m  p10 %.2f  p90 %.2f'
                      % (np.median(hh), np.percentile(hh, 10), np.percentile(hh, 90)))
                print('    |height step|          median %.2f m' % np.median(dz))
        pooled['rows'] = pooled['rows'] + sr
        pooled['strafe'] = sr
        pooled['agg'] = aggregate(pooled['rows'])
    agg = pooled['agg']
    comps = []
    for c in components(agg):
        gs = sorted(n[1] for n in c if n[0] == 'g')
        if len(gs) < 2:
            continue
        sub = {k: v for k, v in agg.items() if k[0] in c and k[1] in c}
        nodes = node_list(sub)
        z, zres, _zu = solve_heights(sub, nodes)
        n2, X, st = solve_layout(sub, nodes)
        idx = {n: i for i, n in enumerate(n2)}
        idz = {n: i for i, n in enumerate(nodes)}
        pos = {g: np.array([X[idx[('g', g)]][0], X[idx[('g', g)]][1], z[idz[('g', g)]]])
               for g in gs if ('g', g) in idx}
        comps.append({'gates': gs, 'sub': sub, 'pos': pos, 'stats': st,
                      'zres': zres, 'nodes': n2})
    pooled['components'] = comps
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', default=','.join(
        [PRIMARY, SECOND, PAUSING, LINKLAP, TIEBREAK]))
    ap.add_argument('--out', default=os.path.join(HERE, 'map_vq2.json'))
    ap.add_argument('--quick', action='store_true', help='skip leave-one-out + jackknife')
    ap.add_argument('--no-strafe', action='store_true')
    a = ap.parse_args()
    sessions = a.sessions.split(',')

    P = build(sessions, strafe=not a.no_strafe)
    agg = P['agg']
    gate_report(agg)
    comp_json = []
    print('\n--- %d connected components with 2+ gates (each in ITS OWN frame; '
          'coordinates are NOT comparable across components)' % len(P['components']))
    for c in P['components']:
        st = c['stats']
        zres = c['zres']
        print('    gates %-16s %2d nodes %2d edges   MDS stress med %5.2f m  p90 %5.2f m'
              % (c['gates'], st['n_nodes'], st['n_edges'],
                 st['stress_median_m'], st['stress_p90_m']))
        comp_json.append({
            'gates': c['gates'],
            'local_positions': {str(g): {'x': float(p[0]), 'y': float(p[1]),
                                         'z_up_m': float(p[2])}
                                for g, p in c['pos'].items()},
            'layout_stats': st,
            'height_residual_median_m': None if zres is None else
                float(np.median(np.abs(zres)))})

    # ---- metric scale, straight from identified pairs --------------------------------
    sc, mj = scale_check(agg)
    ss = np.array([r['scale'] for r in sc])
    print('\n--- METRIC SCALE from %d identified gate pairs' % len(sc))
    for r in sorted(sc, key=lambda r: -r['h']):
        print('    gates %2d-%-2d  measured %6.2f m   sketch %5.2f u  -> %6.2f m/station  '
              '(n=%d, MAD %.2f)'
              % (r['pair'][0], r['pair'][1], r['h'], r['u'], r['scale'], r['n'], r['mad']))
    print('    median %.2f m/station   p10 %.2f  p90 %.2f'
          % (np.median(ss), np.percentile(ss, 10), np.percentile(ss, 90)))
    print('    map_approx.json says %s (21 hand-labelled pairs, p10 13.38 p90 18.49)'
          % mj.get('metres_per_station'))

    result = {
        'status': STATUS,
        'source': 'mapvq2.py -- measured VQ2 gate map from vision + gravity, no pose stream',
        'sessions': [os.path.basename(os.path.normpath(s)) for s in sessions],
        'frame': {'note': 'Each component has ITS OWN frame -- coordinates are NOT '
                          'comparable across components. Within one: x/y metres, '
                          'horizontal, arbitrary rotation and reflection; z metres UP '
                          'from gravity, lowest-indexed gate of the component = 0.'},
        'components': comp_json,
        'metres_per_station': {'median': float(np.median(ss)),
                               'p10': float(np.percentile(ss, 10)),
                               'p90': float(np.percentile(ss, 90)),
                               'n_pairs': len(sc),
                               'per_pair': {'%d-%d' % r['pair']: round(r['scale'], 2)
                                            for r in sc},
                               'map_approx_says': mj.get('metres_per_station')},
        'measured_pairs': {'%d-%d' % (a2[1], b2[1]): {
            'n': v['n'], 'dist_m': v['d'], 'horiz_m': v['h'], 'dz_m': v['dz'],
            'mad_m': v['d_mad'], 'n_height': v['n_h'],
            'n_marker_rows': v.get('n_marker', 0),
            'bimodal_rows_dropped': v.get('bimodal_dropped', 0)}
            for (a2, b2), v in sorted(agg.items()) if a2[0] == 'g' and b2[0] == 'g'},
    }

    if not a.quick:
        print('\n--- LEAVE-ONE-EDGE-OUT, per component (held-out gate-gate distances)')
        loo_json = []
        for c in P['components']:
            loo = loo_edges(c['sub'], node_list(c['sub']))
            for r in sorted(loo, key=lambda r: -abs(r['err'] or 0)):
                if r['err'] is None:
                    print('    %2d-%-2d  held out -> gate unplaceable without it (hinge)'
                          % r['pair'])
                else:
                    print('    %2d-%-2d  measured %6.2f m   predicted %6.2f m   err %+6.2f m'
                          % (r['pair'][0], r['pair'][1], r['measured'], r['predicted'],
                             r['err']))
                loo_json.append(dict(r, pair=list(r['pair'])))
        e = np.array([abs(r['err']) for r in loo_json if r['err'] is not None])
        if len(e):
            print('    |err| median %.2f m   p90 %.2f m'
                  % (np.median(e), np.percentile(e, 90)))
            result['leave_one_out'] = {'median_abs_err_m': float(np.median(e)),
                                       'p90_abs_err_m': float(np.percentile(e, 90)),
                                       'edges': loo_json}

    # ---- lap against lap, at the level the data actually supports ---------------------
    # A position-level comparison needs the two laps to place the same gates, and neither
    # lap on its own places enough. The DISTANCES do overlap, and a distance measured twice
    # from two separate flights is an independent repeat of the same fixed quantity -- which
    # is what an error bar is made of. Reported whether or not the embedding comparison
    # below finds anything.
    if len(sessions) > 1:
        pl = {}
        for sess in sessions:
            name = os.path.basename(os.path.normpath(sess))
            r = P['per'][name]
            pl[name] = {(a[1], b[1]): v for (a, b), v in r['agg'].items()
                        if a[0] == 'g' and b[0] == 'g'}
        names = list(pl)
        shared = sorted(k for k in set().union(*pl.values())
                        if sum(k in pl[n] for n in names) >= 2)
        print('\n--- LAP AGAINST LAP on measured DISTANCES '
              '(%d pairs measured in 2+ sessions of %d)' % (len(shared), len(names)))
        rep = {}
        for k in shared:
            vals = {}
            for n in names:
                if k in pl[n]:
                    v = pl[n][k]
                    vals[n] = (v['h'] if v['h'] is not None else v['d'], v['n'])
            spread = max(v[0] for v in vals.values()) - min(v[0] for v in vals.values())
            rep['%d-%d' % k] = {'per_session_m': {n: v[0] for n, v in vals.items()},
                                'per_session_n': {n: v[1] for n, v in vals.items()},
                                'spread_m': spread}
            print('    gates %2d-%-2d   %s   spread %5.2f m'
                  % (k[0], k[1],
                     '   '.join('%s %6.2f m (n=%3d)' % (n[-12:], v[0], v[1])
                                for n, v in vals.items()), spread))
        if rep:
            dv = np.array([v['spread_m'] for v in rep.values()])
            print('    spread median %.2f m   max %.2f m   THIS IS THE UNCERTAINTY'
                  % (np.median(dv), dv.max()))
            result['lap_vs_lap_distances'] = {
                'sessions': names, 'pairs': rep,
                'median_spread_m': float(np.median(dv)),
                'max_spread_m': float(dv.max())}

    # The position-level lap-against-lap comparison was removed with the move to
    # per-component solving: each lap alone yields small hinge-prone components, and
    # rigidly aligning two under-braced embeddings compares artifacts, not measurements.
    # The distance-level table above is the honest cross-lap check.

    with open(a.out, 'w') as fh:
        json.dump(result, fh, indent=1)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
