"""build_tilt.py -- per-gate TILT-FROM-VERTICAL (magnitude AND direction) for the export.

    python3 pilot/course/build_tilt.py      # -> pilot/course/tilt_measured.json

REVISED 2026-08-02. The first version of this file concluded "no gate reads consistently
tilted" and exported gates 8 and 9 with a uniform 0-20 deg prior. THAT WAS A POOLING
ARTEFACT and is overturned here. What changed is not the data -- it is the SAME
gatetilt_rows.json -- but how it is reduced. See the METHODOLOGY note below; it is the
reusable part.

WHAT IS MEASURED NOW
  gate 9  is TILTED. Magnitude 21 +- 5 deg (deliberately wide -- see ADJUDICATION), and
          its top leans toward azimuth 130 deg in the export frame: mostly toward +y (the
          Station 21-29 column row), with a component along the race direction (-x). The
          DIRECTION is the better-determined half of this measurement.
  every other gate is VERTICAL to within 1.5-4 deg, which is the estimator's noise floor
          on close views. Gate 8 -- the previous version's suspect -- is 3.6 deg.

METHODOLOGY (the reusable lesson)
  The first pass took the MEDIAN over all ~8500 detections per gate. Most detections are
  FAR and SMALL, and at 30-40 px a 20 deg lean moves the quad by a couple of pixels: the
  tilt is geometrically almost invisible there and the estimator returns noise centred a
  few degrees above zero. Pooling therefore pulls a genuinely tilted gate DOWN toward the
  noise floor while pushing every vertical gate UP to it, and the two become
  indistinguishable -- which is exactly the "every gate reads 1.3-4.9 deg" result that
  was reported.

  The fix is to stratify by MEASUREMENT QUALITY and read the TREND rather than the level:

      a tilt is a property of the gate, so it must SURVIVE as views get closer and
      larger; estimator noise must SHRINK.

  Stratified (pooled median, degrees):

      gate |  all  | >=60px | >=120px | <=15m
        9  | 14.6  |  24.0  |  21.4   | 23.9     <-- RISES with proximity
        8  | 11.3  |   3.0  |   2.6   |  4.2
      others  2.6-10.6 | 0.9-6.1 | 0.9-7.3 | 1.9-11.9   <-- all FALL toward vertical

  Gate 9 is the only gate that diverges upward. That divergence is the signal, and NO
  single pooled number can show it.

ACCEPTED REDUCTION RULE (frozen before the direction was looked at)
  rows with size_px >= 60 (roughly 25 m or nearer for the 2.7 m frame); per-session
  medians first, then the median OVER sessions so one long flight cannot outvote three
  others; sigma from the robust spread of the session medians, floored at 1 deg.
  Gate 9 per-session near-row medians: 24.9 (n=86), 22.5 (n=20), 20.6 (n=20), IQRs ~3 deg.

SIZE/RANGE IS THE WRONG STRATIFIER FOR THE MAGNITUDE, and this nearly caused a second
error in the opposite direction. Both estimators' blind spots are functions of the same
geometric variable -- theta, the angle between the line of sight and the gate's plane
normal (0 = head-on, 90 = edge-on):

  * PnP normal elevation is worst conditioned and biased UP at theta ~ 0 (that is the
    same IPPE head-on valley that makes yaw unmeasurable at gates 8/12/13);
  * edge_tilt is BLIND at theta ~ 0 -- a gate leaning straight toward or away from the
    camera still projects its side edges as vertical lines.

Near, large detections on this course are overwhelmingly HEAD-ON APPROACHES. Selecting
them selects rows where BOTH channels are at their worst, in opposite directions -- which
is why one reduction of this row set read 24 deg for gate 9 and another read 3 deg.
gatetilt_obliquity.py re-cuts the same rows by theta.

THE OBLIQUITY EVIDENCE (gatetilt_obliquity.py; theta buckets, gate 9 vs controls)

  gate 9   theta 0-10 : PnP 24.5   edge 3.1   (n=114, 3 sessions)
           theta 10-20: PnP 22.0   edge 6.4   (n= 30, 4 sessions)
           theta > 20 : NO ROWS AT ALL -- max theta over the whole row set is 19.2 deg.

  * PnP does NOT decay as theta grows (24.5 -> 22.0). If the 24 deg were a head-on
    conditioning artefact it should fall away from theta = 0. It does not.
  * edge RISES with theta (3.1 -> 6.4), which is what a real tilt must do: the channel
    obeys sin(psi) = sin(phi) sin(theta), so a small non-zero reading at small theta is
    evidence FOR a large phi, not against it. A vertical gate stays at its noise floor.
  * at MATCHED obliquity, gate 9 is the outlier by a wide margin: in the theta 0-10
    bucket the control gates read PnP 1.8-9.5 deg and gate 9 reads 24.5.

ADJUDICATION OF THE MAGNITUDE -- and why sigma is 5 deg and not 3
  Three estimates of phi, each stated with the regime it is valid in:
    21-25 deg  PnP normal elevation / resolved 3-D axis (23.1 deg, residual median
               2.7 deg), from head-on rows -- the regime where PnP is weakest, but
               control gates at the SAME obliquity read 2-9 deg.
    19.7 deg   the independent screenshot gate8.png, PnP-free, gate seen EDGE-ON, i.e.
               the one observation actually in the edge channel's valid regime; two
               hangar columns in the same frame read 1.0-1.2 deg as in-frame controls.
    13.2 deg   the PnP-free slope fit sin(psi) = a sin(theta) + b over gate 9's flight
               rows [90% CI 11.8-16.1]. TREATED AS A WEAK LOWER BOUND, not as a rival:
               gate 9's theta lever is only 19 deg wide, and control gates with equally
               short levers return spurious 8-19 deg from the same fit (gate 3 returns
               19.0 deg and is vertical). Short-lever slope fits are biased up because
               quad corner noise itself grows with obliquity.
  Every channel lands in 13-25 deg and NONE supports a vertical or near-vertical gate.
  Exported: 21 deg with sigma 5, clipped to 12-30 -- wide on purpose, because the
  measurement is obliquity-limited rather than count-limited.

  HOW TO CLOSE IT PROPERLY: fly one lateral pass at gate 9 (a strafe, as was done at
  pairs 15-16, 6-7 and 2-3) to get theta > 45 deg rows. There the edge channel measures
  phi directly with no model and no twin, and the answer stops depending on any of this.
  The detector's aspect-ratio filter also discards the most oblique views, so that pass
  wants the filter loosened.

DIRECTION -- the better-determined half, and NOT taken on trust
The plane normal is an AXIS (its overall sign is unobservable from a square) but the
gate's TOP-LEAN direction is invariant under n -> -n:  d ~ -n_z * n_horiz, since both
factors flip together. What is NOT invariant is the IPPE TWIN, which mirrors the normal
about the line of sight and therefore flips the lean by ~180 deg. Resolved by RANSAC vote
over both candidates of all 130 near rows, in the export frame:

  selected branch : one tight cluster -- axis [0.599, -0.698, +0.392], angular residual
                    median 2.7 deg, p90 7.5 deg, 91.5% inliers, and 95% of rows had IPPE's
                    own preferred solution selected (the vote is not fighting the solver).
  rejected branch : does not form an axis at all -- residual median 24.1 deg.

A mirror artefact cannot produce that asymmetry: the twin moves with the line of sight,
so the wrong branch cannot be coherent across vantages, and it is not. Corroboration:
(1) three flights with different compass branches agree on the lean azimuth to 0.3 deg
(129.5 / 129.7 / 129.8) while their vantages differ by 15 deg -- a mirror artefact would
have moved by ~30 deg; (2) the lean must be perpendicular to the gate plane, and the
map's independently measured plane azimuth for gate 9 (132.4 deg mod 180, n=382) agrees
to 2.7 deg.

DISCARDED CHANNEL, recorded so nobody rebuilds it. gatetilt_foreshorten.py tests the
top/bottom edge-length ratio (a gate leaning away projects a shorter top edge). It FAILED
its own built-in control: every gate, including known-vertical ones, fits 15-35 deg of
tilt with the lean pointing away from the camera, so the detector's quad has a systematic
top/bottom asymmetry that swamps the signal. Its gate-9 numbers (32.5 deg at azimuth 110)
are NOT used for anything. The file is kept only because the control failure is the
result.

INHERITED CAVEAT: the export frame is only known mod 90 deg with respect to the sim
world (README section 2), so the lean azimuth inherits that same k*90 ambiguity. It is
exact RELATIVE to the course's own axes and to the race line, which is what a policy
needs.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERC = os.path.normpath(os.path.join(HERE, '..', 'perception'))
sys.path.insert(0, PERC)
import gatetilt_strat as GS      # noqa: E402

OUT = os.path.join(HERE, 'tilt_measured.json')
MAP = os.path.join(PERC, 'map_vq2.json')

TILTED_GATE = 9
# A gate is called TILTED only if its near-row estimate clears this AND its session
# medians agree AND its resolved lean direction is concentrated. Gate 9 is the only one.
TILT_ACCEPT_DEG = 12.0
DIR_ACCEPT_R = 0.85          # circular concentration of the per-row lean direction
# The adjudicated magnitude (see ADJUDICATION in the module docstring). Written as
# constants rather than taken from one channel, because no single channel is in its
# valid regime -- the value is the reconciliation of three, and the sigma is wide
# because the limit is obliquity coverage, not row count.
TILT_DEG, TILT_SIG_DEG, TILT_CLIP_DEG = 21.0, 5.0, [12.0, 30.0]
# Vertical gates: exported sigma is the measured near-row noise floor, clamped. Not a
# prior -- these are the residual readings of gates we believe are exactly vertical.
SIG_MIN, SIG_MAX = 1.5, 4.0


def main():
    R = GS.load_rows()
    mapj = json.load(open(MAP))

    strata = GS.strata_table(R)
    acc = GS.magnitudes(R)
    dirs = {g: GS.direction(R, g, mapj) for g in range(GS.N_GATES)}

    per = {}
    print('\n%-5s %8s %8s %10s %10s  %s'
          % ('gate', 'tilt', 'sigma', 'lean_az', 'lean_R', 'verdict'))
    for g in range(GS.N_GATES):
        A = acc[g]['pnp'] if acc[g] and acc[g]['pnp'] else None
        E = acc[g]['edge'] if acc[g] and acc[g]['edge'] else None
        D = dirs[g]
        all_med = strata[g]['all'][0]
        near_med = A['deg'] if A else None
        rec = {
            'tilt_median_all_rows_deg': all_med,
            'tilt_near_rows_median_deg': (round(near_med, 2) if near_med is not None
                                          else None),
            'tilt_edge_channel_deg': round(E['deg'], 2) if E else None,
            'tilt_n_near_rows': acc[g]['n_near_rows'] if acc[g] else 0,
            'tilt_n_sessions': A['n_sessions'] if A else 0,
            'tilt_per_session_deg': ([d['median_deg'] for d in A['per_session']]
                                     if A else []),
            'tilt_lean_azimuth_deg': None,
            'tilt_lean_azimuth_sigma_deg': None,
            'tilt_lean_unit_xy': None,
        }
        tilted = (A is not None and A['deg'] >= TILT_ACCEPT_DEG
                  and D is not None and D['lean_concentration_R'] >= DIR_ACCEPT_R)
        if tilted:
            rec['tilt_from_vertical_deg'] = TILT_DEG
            rec['tilt_sigma_deg'] = TILT_SIG_DEG
            rec['tilt_clip_deg'] = list(TILT_CLIP_DEG)
            rec['tilt_channels_deg'] = {
                'pnp_head_on_rows': round(A['deg'], 1),
                'pnp_resolved_3d_axis': (round(D['axis_tilt_deg'], 1) if D else None),
                'screenshot_gate8png_edge_on_pnp_free': 19.7,
                'edge_slope_fit_pnp_free_weak_lower_bound': 13.2,
            }
            rec['tilt_limited_by'] = (
                'OBLIQUITY COVERAGE, not row count: every gate-9 row is a head-on view '
                '(theta <= 19.2 deg, no strafe session passes gate 9). One lateral pass '
                'at gate 9 would measure this directly with the PnP-free edge channel '
                'and remove the model dependence entirely.')
            az = D['lean_azimuth_deg']
            rec['tilt_lean_azimuth_deg'] = round(az, 1)
            rec['tilt_lean_azimuth_sigma_deg'] = 5.0
            rec['tilt_lean_unit_xy'] = [round(math.cos(math.radians(az)), 3),
                                        round(math.sin(math.radians(az)), 3)]
            rec['tilt_lean_concentration_R'] = D['lean_concentration_R']
            rec['tilt_status'] = (
                'TILTED (measured). Magnitude %.0f +- %.0f deg, clipped to %.0f-%.0f -- '
                'WIDE ON PURPOSE: three channels give 13.2 / 19.7 / 21-25 deg and none '
                'is in its ideal regime, because every gate-9 row is a head-on view '
                '(theta <= 19 deg). None supports a vertical gate. DIRECTION is the '
                'better-determined half: the gate TOP leans toward azimuth %.0f deg in '
                'the export frame (unit xy %s) -- mostly +y, the Station 21-29 column '
                'row, with a component along the race direction. Twin-resolved (the '
                'selected branch clusters to 2.7 deg residual, the mirror branch to '
                '24.1 deg), three flights agree to 0.3 deg, and it matches the map\'s '
                'independent plane azimuth 132.4 deg to 2.7 deg. Azimuth inherits the '
                'frame\'s k*90 ambiguity w.r.t. the sim world but is exact relative to '
                'the course. Per-session head-on PnP medians: %s deg.'
                % (TILT_DEG, TILT_SIG_DEG, TILT_CLIP_DEG[0], TILT_CLIP_DEG[1],
                   az, rec['tilt_lean_unit_xy'],
                   ', '.join('%.1f' % x for x in rec['tilt_per_session_deg'])))
        else:
            rec['tilt_from_vertical_deg'] = 0.0
            if near_med is None:
                rec['tilt_sigma_deg'] = SIG_MAX
                rec['tilt_clip_deg'] = [0.0, 3.0 * SIG_MAX]
                rec['tilt_status'] = ('vertical -- assumed: no rows at >= 60 px, so the '
                                      'quality-stratified test could not run here. '
                                      'sigma is the population noise floor.')
            else:
                rec['tilt_sigma_deg'] = round(
                    float(min(SIG_MAX, max(SIG_MIN, near_med))), 1)
                # cap the Gaussian tail at 3 sigma: a gate we call vertical should never
                # be handed to a policy leaning 15 deg because of a 4-sigma draw
                rec['tilt_clip_deg'] = [0.0, round(3.0 * rec['tilt_sigma_deg'], 1)]
                rec['tilt_status'] = (
                    'VERTICAL (measured). Near-row (>= 60 px) tilt %.1f deg over %d '
                    'session(s) -- the estimator noise floor, and it FALLS from %.1f '
                    'deg on all rows as views get closer, which is what noise does and '
                    'a real tilt does not. sigma is that residual, not a prior.'
                    % (near_med, A['n_sessions'], all_med))
        per[g] = rec
        print('%-5d %8.1f %8.1f %10s %10s  %s'
              % (g, rec['tilt_from_vertical_deg'], rec['tilt_sigma_deg'],
                 '%.1f' % rec['tilt_lean_azimuth_deg']
                 if rec['tilt_lean_azimuth_deg'] is not None else '-',
                 '%.2f' % D['lean_concentration_R'] if D else '-',
                 rec['tilt_status'].split('.')[0]))

    doc = {
        'method': __doc__.strip(),
        'revision': ('2026-08-02: OVERTURNS the previous "no gate is tilted / gates 8 '
                     'and 9 carry a uniform 0-20 deg prior" verdict. Same rows, '
                     'quality-stratified reduction.'),
        'rule': {'min_size_px': GS.NEAR_PX,
                 'pooling': 'per-session medians, then median over sessions',
                 'sigma': 'robust spread of the session medians, floored at 1 deg; '
                          'vertical gates export the near-row residual clamped to '
                          '%.1f-%.1f deg' % (SIG_MIN, SIG_MAX)},
        'per_gate': {str(g): per[g] for g in range(GS.N_GATES)},
        'structure': {
            'gates_are_vertical': (
                'SIXTEEN of the seventeen gates are vertical planes: on close views '
                '(>= 60 px) they read 1.5-4.0 deg out of gravity-vertical, and that '
                'reading FALLS as views get closer, which is the signature of estimator '
                'noise. They are exported as tilt 0 with that residual as sigma. This '
                'is a usable PRIOR for a pose estimator: it resolves the two-fold IPPE '
                'tilt ambiguity, the same ambiguity that made gate YAW unmeasurable at '
                '8/12/13.'),
            'tilted_gate': (
                'GATE 9 IS TILTED. DIRECTION is solid: the gate top leans toward '
                'azimuth 130 deg in the export frame (unit xy [-0.64, +0.77]) -- mostly '
                'toward +y, the Station 21-29 column row, with a component along the '
                'race direction. The IPPE twin was resolved by RANSAC over both '
                'candidates of all 130 near rows: the selected branch forms one cluster '
                '(2.7 deg residual), the mirror branch does not form an axis at all '
                '(24.1 deg), and three flights whose vantages differ by 15 deg agree on '
                'the azimuth to 0.3 deg -- which a mirror artefact cannot do. It also '
                'matches the map\'s independent gate-plane azimuth (132.4 deg mod 180) '
                'to 2.7 deg, as a lean perpendicular to the plane must. MAGNITUDE is '
                '21 +- 5 deg, deliberately wide: gate 9 is seen only head-on (theta <= '
                '19 deg in every row), which is the worst regime for BOTH estimators in '
                'opposite directions, so the value reconciles three channels -- 21-25 '
                'deg (PnP elevation / resolved axis), 19.7 deg (the edge-on gate8.png '
                'screenshot, PnP-free and the only observation in the edge channel\'s '
                'valid regime, with in-frame column controls at 1.0-1.2 deg), and 13.2 '
                'deg (a short-lever slope fit that reads 19.0 deg on a gate known to be '
                'vertical, so a weak lower bound only). No channel supports a vertical '
                'gate. GATE 8 IS NOT TILTED (3.6 deg on near rows); the earlier '
                'suspicion came from its pooled far-row tail. TO CLOSE THE MAGNITUDE: '
                'one lateral strafe past gate 9 gives theta > 45 deg rows, where the '
                'PnP-free edge channel measures the lean directly.'),
            'why_the_first_pass_missed_it': (
                'TWO reduction mistakes, in opposite directions, on the SAME rows -- '
                'both worth knowing before re-deriving anything from this file. '
                '(1) POOLING. The first pass took the median over all ~8500 detections '
                'per gate. Most rows are far and small, where a 20 deg lean is '
                'geometrically invisible and the estimator returns noise; pooling drags '
                'a tilted gate DOWN to the noise floor and pushes vertical gates UP to '
                'it, and the two become indistinguishable -- which is exactly the '
                '"every gate reads 1.3-4.9 deg" non-result that was reported. '
                '(2) STRATIFYING ON THE WRONG VARIABLE. Fixing (1) by selecting big, '
                'close rows finds the tilted gate, but it cannot size it, because on '
                'this course near+large means HEAD-ON, and head-on is where the PnP '
                'estimator is biased up AND the PnP-free edge estimator is blind. The '
                'variable both blind spots depend on is OBLIQUITY (angle between the '
                'viewing ray and the gate normal), and gate 9 has no rows above 19 deg '
                'of it. General form: stratify by the variable the ESTIMATOR\'s error '
                'depends on, not by the one that looks like data quality, and check '
                'every reduction against control gates carried through the identical '
                'cut. Both errors here were invisible in the pooled number and obvious '
                'in the stratified one.'),
            'gate_faces_and_the_grid': (
                'The pilot reports that gate faces are parallel to the hangar columns, '
                'i.e. gate yaw is quantized to the ceiling/floor/column grid. OUR '
                'MEASURED YAWS DO NOT REPRODUCE THAT: the 14 accepted plane azimuths, '
                'taken mod 90, spread over 11-88 deg with no cluster at the grid bin '
                '(deviations -37 to +44 deg). They agree far better with the RACE-LINE '
                'BISECTOR (median |error| 10.9 deg, under 6 deg at gates 1, 3, 6, 10, '
                '14, 16). Both facts are exported per gate -- yaw_deg, yaw_grid_bin_deg '
                '+ yaw_grid_dev_deg, yaw_race_bisector_deg -- and the conflict is '
                'flagged rather than smoothed. Given how noisy IPPE azimuth proved on '
                'this course, the visual claim may well be right and our azimuths '
                'wrong; do not build anything that needs an exact gate yaw. NOTE the '
                'gate-9 tilt direction is a separate and much better-conditioned '
                'measurement: its 130 deg agrees with the map\'s independent plane '
                'azimuth (132.4 deg mod 180) to 2.7 deg, as it must, since a lean is '
                'perpendicular to the gate plane.'),
        },
    }
    json.dump(doc, open(OUT, 'w'), indent=1)
    print('\nwrote %s' % OUT)


if __name__ == '__main__':
    main()
