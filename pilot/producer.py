"""producer.py -- flight-time perception: frames + HIGHRES_IMU + race status -> Observation.

This is perception's half of the frozen contract in pilot/interface.py. Control consumes
the Observation this file fills and returns an Action; nothing above the interface knows
how the fields were made. Every quantity is body-referenced or an elapsed time -- there is
no world frame, no absolute position and no yaw anywhere in the OUTPUT (the compass below
is internal to attention/lookahead only, per the interface's design note).

PIPELINE per frame (all CPU, budget ~15 ms median; measured by --replay):

  1. IMU ingest (every sample since the last frame): gyro mirrored x-1 at this read
     boundary (CONVENTIONS.md: the mirror is applied "when reading HIGHRES_IMU's gyro" --
     this class IS that read), accel canonical. Streaming gravity complementary filter
     (same alpha schedule skylight.py validated: roll/pitch ~1.1/1.5 deg vs VQ1 truth),
     plus the levelled yaw integral psi_int that carries heading between compass fixes.
     Tracks are rotation-coasted with the same increment, so a coasted gate stays where
     the world is while the body turns.

  2. Contour detections (detect.py, unchanged) + the interior-colour decoration test
     (vq2cache.py's in_white/in_v on the shrunken quad interior).

  3. Identity: active_gate_index names the current gate (the one sim-supplied spatial
     fact); slots k, k+1, k+2 are tracked by association against the track's predicted
     centre, with a map-lookahead prediction (map_vq2.json layout + skylight compass)
     when no track exists yet. A race-index RESET (index decreases) clears all tracks.

  4. gatenet (colab-v3-rot) on seeded crops -- always for the current gate when a seed
     exists, and for lookahead gates whose detection is clipped/outer/large or whose
     track is coasting. Paired confidence head (colab-conf-rot, operating point
     p = 0.082, TRAINING.md pairing note: this head ONLY with this trunk).

  5. Three cheap rejection layers on every accepted quad, per the flight-stack design:
     learned confidence (p < CONF_P rejects), PnP reprojection residual, and the
     interior-colour test. Plus the PnP-vs-apparent-size range agreement that caught the
     ribbon-silhouette quads (mapvq2.clean()).

  6. IPPE_SQUARE pose (detect.OBJ winding -- the order is load-bearing, see detect.py)
     via solvePnPGeneric so BOTH tilt solutions are visible: direction is signed toward
     the camera (approach side by construction for an uncrossed gate). The TILT branch is
     NOT decided per frame (2026-08-02): both candidates go to the track's
     normalfuse.NormalAccumulator, which rotates them into a track-common frame through
     the gyro integral and scores the two hypotheses across views -- the true branch is a
     fixed world direction, the false one swings with the line of sight. normal_valid
     goes True only on a stated likelihood margin WITH real angular baseline (or when the
     reprojection alone separates the branches); otherwise False, per the contract.

  7. Degradation ladder per GateObs: PnP (pose_valid) -> centroid + apparent size
     (range biased LONG when oblique; range_sigma_m carries that honestly) -> coasting
     with staleness_s growing -> invalid after COAST_MAX_S.

  8. Ribbon: cyan mask -> N_RIBBON direction samples binned by image row (bottom = near,
     a floor-ribbon distance proxy); pixel_fraction is the dropout detector. Not
     load-bearing, by interface design.

  9. Attention + AUTO_ATTENTION yaw servo: track the current gate until range <
     R_COMMIT, then hand off to next gate / ribbon; SEARCH is an informed sweep toward
     where the map + compass say the target will appear, not a blind spin. The servo
     drives AZIMUTH to zero only (yaw cannot change elevation; interface YawMode note).

CAMERA -> BODY. Camera shares the body origin, tilted 20 deg UP (label.body_to_cam,
sign derived AND sweep-confirmed there). pos_body = R_cb.T @ t_cam. Check, stated as the
mission demands: a gate dead-centre in the image at range R lies on the camera boresight,
which is 20 deg ABOVE body-forward, so it must come out at bearing 0, elevation +20 deg.
    R_cb.T @ (0,0,R) = R * (cos20, 0, -sin20)  ->  bearing atan2(0, cos20) = 0,
    elevation atan2(+sin20, cos20) = +20 deg.                      (--selftest asserts it)

USAGE
    python3 pilot/producer.py --selftest
    python3 pilot/producer.py --replay <session-dir-or-name> [--video] [--limit N]
    python3 pilot/producer.py --replay <session> --paircheck    # strafe geometry check

Live wiring: see pilot/PRODUCER.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERC = os.path.join(HERE, 'perception')
sys.path.insert(0, HERE)
sys.path.insert(0, PERC)

import interface as I           # noqa: E402  (the frozen contract)
import label as L               # noqa: E402  (camera model, body_to_cam)
import detect as D              # noqa: E402  (contour detector, OBJ winding)
import skylight as SKY          # noqa: E402  (compass: heading())
import normalfuse as NF
import dragmodel as DRAG   # noqa: E402  (drag speed model)         # noqa: E402  (per-track IPPE twin accumulator)

# torch + gatenet are imported lazily so --no-net / --selftest never pay for them
_G = _GC = _torch = None

MAP_JSON = os.path.join(PERC, 'map_vq2.json')
COURSE_JSON = os.path.join(HERE, 'course', 'course_vq2.json')   # read-only, other owner
TRUNK_CKPT = os.path.join(PERC, 'gatenet_runs', 'colab-v3-rot', 'best.pt')
HEAD_CKPT = os.path.join(PERC, 'gatenet_runs', 'colab-conf-rot', 'best.pt')

# ---- operating points and thresholds ----------------------------------------------------
CONF_P = 0.082            # confidence-head reject threshold (TRAINING.md 2026-08-02 ACCEPT)
N_GATES_MAP = 17
R_COMMIT = 4.0            # m. Map consecutive edges run 8.3-22.7 m, so 4 m is "inside the
                          # final approach" on every edge without being inside PnP noise.
COAST_MAX_S = 2.5         # track invalid after this much coasting (consumer sees 5.0 cap)
NET_MAX_CROPS = 3         # hard batch cap per frame (one per slot)
NET_MIN_SEED_PX = 14.0    # smallest seed box worth a crop
NET_LARGE_PX = 90.0       # detector quality degrades above this -> prefer the net
                          # (60-90 px unclipped detector quads still pass the PnP
                          # residual gate; the net's own >=120 px band is its weakest)
PNP_RESID_REL = 0.05      # residual gate is SIZE-RELATIVE (residgate.py 2026-08-02:
                          # residual scales with apparent size; the absolute-tight 0.3 px
                          # HUD gate refused r=0.87 px on a ~500 px gate -- 0.17% -- and
                          # starved final approach). Paired with MIN_CORNERS_IN_FRAME
                          # below; the pair was chosen on the pose-level referee
                          # (cornergate.py), NOT on corner-pixel error:
                          #   rule                       catch  keep good  keep good>=120
                          #   max(6, 0.05s)   (pre-fix)   0.0%     100.0%      100.0%
                          #   max(0.3, 0.015s)           29.4%      98.5%       78.1%
                          #   corners>=2                 37.3%      99.5%       95.8%
                          #   corners>=2 & max(0.3,.05s) 49.0%      99.5%       95.8%   <-
                          #   corners>=3 & max(0.3,.05s) 62.7%      89.2%       35.4%
                          # The chosen row dominates the size-only rules on ALL THREE
                          # columns. A tighter slope (0.03) buys 2 points of catch for 4
                          # points of big-gate coverage -- the wrong trade, since
                          # big-gate coverage is precisely what Bug 1 is about.
MIN_CORNERS_IN_FRAME = 2  # Claire's 2026-08-02 insight, measured in cornergate.py: for a
                          # CLOSE gate, size is the wrong trust variable -- what decides
                          # whether the pose is constrained is how much of the gate is
                          # actually OBSERVED. Catastrophe rate by in-frame corner count
                          # (VQ1 val, pose-level referee): 0 in -> 100%, 1 -> 44.4%,
                          # 2 -> 3.4%, 3 -> 0.0%, 4 -> 1.2%. The cliff is between ONE and
                          # TWO, not at three: two in-frame corners of a known 1.5 m
                          # square already pin the pose. Requiring three would cost 60
                          # points of big-gate coverage for 14 points of catch.
PNP_RESID_ABS_NET = 0.3   # floor for gatenet quads (pnpsnap population: good net quads
                          # sit at 0.01-0.5 px; catastrophe med >=0.4 px at small sizes)
PNP_RESID_ABS_CTR = 1.0   # floor for contour quads: their GOOD poses measure rms
                          # 0.16-0.49 px (corner noise), a 0.3 floor loses ~20% of them;
                          # 1.0 keeps 100% (residgate.py contour arm). Residual cannot
                          # catch contour catastrophes anyway (consistent-but-wrong
                          # quads, med rms 0.18): range-agreement + interior do that.
EMISSIVE_V_MIN = 100.0    # gates are EMISSIVE: their frame pixels read V>=185 (p10)
EMISSIVE_S_MIN = 80.0     # at any range, while lit-by-gate reflections on hangar
EMISSIVE_FRAC = 0.25      # columns read V~42-48 (measured, 20260802-204323 crash:
                          # gatenet minted a track on a column's glow band and flew the
                          # servo policy into the pillar). The net regresses corners on
                          # whatever crop it is given, so its acceptances -- unlike the
                          # contour path, which inherits detect.py's V_MIN mask -- need
                          # this check: a ring just outside the accepted quad must
                          # contain actual emissive-orange pixels.
NET_PRED_ERR_ABS = 6.0    # conf-head predicted-corner-error acceptance floor (was the
                          # old PNP_RESID_ABS; role unchanged)
RANGE_DISAGREE = 0.30     # PnP vs apparent-size range disagreement (mapvq2.clean())
TILT_AMBIG_RATIO = 1.6    # 2nd IPPE solution within this error ratio = ambiguous tilt
TILT_CONSIST_DEG = 25.0   # temporal consistency window for picking a tilt branch.
                          # 2026-08-02 (normalfuse): temporal consistency is NO LONGER a
                          # decision channel -- it compared the new candidates against the
                          # producer's OWN previous pick, so a wrong first pick was
                          # self-confirming and the flip rate sat at ~4% regardless of what
                          # the tiebreak did. The per-track accumulator in normalfuse.py
                          # replaces it; these two constants survive only for pnp_pose's
                          # legacy branch (LEGACY_TILT, kept for A/B measurement).
LEGACY_TILT = False       # True restores the per-frame temporal+verticality resolver
NORMAL_TRUST_UNAMBIG = True   # a frame whose 2nd IPPE solution is >TILT_AMBIG_RATIO worse
                          # is decided by the REPROJECTION, not by a guess, so it may carry
                          # normal_valid on its own. Measured: turning this off costs most
                          # of the final-approach availability and buys little (see
                          # normalfuse_metrics.py).
VERTICALITY_TOL_DEG = 20.0    # gravity-referenced tilt disambiguator (2026-08-02).
                              # Every VQ2 gate face is a VERTICAL plane -- rotated only
                              # about the up axis -- except GATE 9, which leans (Claire's
                              # sim inspection; corroborated by the stratified tilt
                              # analysis, where gate 9 is the standout in every channel
                              # and every other gate falls to 1-4 deg as the view
                              # improves). A candidate's plane normal should therefore be
                              # near HORIZONTAL; applied only as a FALLBACK when temporal
                              # consistency does not resolve the branch.
VERTICALITY_MARGIN_DEG = 8.0  # required tilt gap between the two IPPE twins before
                              # the prior is trusted to pick one (else genuinely
                              # ambiguous under the prior too -> normal_valid False)
VERTICALITY_CONF_MIN = 0.3    # only trust gravity when attitude_conf clears this
VERTICALITY_EXEMPT_GATES = frozenset({9})   # the one gate this prior does not apply to.
                              # NOT gate 8: an earlier note named 8, from a screenshot
                              # that was misfiled -- gate 8 measures vertical (1.9 deg
                              # best-view edge lean) in both the PnP-normal and the
                              # PnP-free channel. Exempting the wrong gate is doubly
                              # wrong: it withholds the prior where it works AND applies
                              # it where the plane really is tilted, forcing the tilted
                              # gate's correct twin to be scored as the outlier.
                              # NOTE (2026-08-02): the two tilt channels agree gate 9 is
                              # the exception but NOT on magnitude -- the PnP-normal
                              # elevation reads ~21-24 deg on close views while the
                              # PnP-free edge-line channel reads 3.3 deg at best view.
                              # The exemption only needs "gate 9 is not reliably
                              # vertical", which both support, so it does not wait on
                              # that being settled. See NOTES.md 2026-08-02 addendum.
USE_MAP_NORMAL_PRIOR = True   # feed course_vq2.json's per-gate race-line bisector to the
MAP_PRIOR_MIN_ANCHORS = 8     # normal accumulator, once the map-frame anchor has this many
                              # co-measured pairs behind it AND the compass has anchored
                              # psi_int. WHAT DEGRADES IF THE COMPASS IS WRONG: psi_int
                              # error rotates the expected azimuth one-for-one, and a wrong
                              # MIRROR (MapModel.s, known unstable run-to-run) REFLECTS it
                              # -- in which case the prior actively favours the wrong twin.
                              # normalfuse clamps the priors' net LLR contribution
                              # (PRIOR_LLR_CAP) so that failure can at worst turn a
                              # no-baseline refusal into a wrong answer; it can never
                              # override a consistency verdict that has baseline behind it.
NORMAL_DEBUG = False          # log per-frame twin candidates for the offline
                              # separation-vs-baseline analysis (normalfuse_metrics.py)
# The azimuth GRID SNAP (quantise the plane azimuth to grid_angle + k*90) was designed
# and then DROPPED, not merely deferred: the 14 map-accepted gate azimuths spread 11-88
# deg mod 90 with deviations up to +-44 deg from any grid bin and show no clustering, so
# the quantiser has no support in our data. A snap that always snaps converts a wrong
# pose into a confidently wrong one, which is the exact failure mode Bug 3 is about.
SLOT_TRI_TOL = 4.0        # m slack on the range triangle inequality between two
                          # slots (PnP range noise + map edge error + the size-range
                          # fallback's long bias). Deliberately loose: the test only has
                          # to catch SWAPS, which are wrong by a whole edge length
                          # (8-34 m on this course), not to measure anything.
SLOT_SEP_TOL = 4.0        # m slack on the 3-D separation test between two measured
                          # detections and the map distance for the gates they claim
                          # to be. Used only when both carry a PnP position.
SLOT_ORDER_TOL = 4.0      # m. Slot k must not measure FARTHER than slot k+1 by more
                          # than this. Deliberately loose -- Claire's two frames are
                          # inverted by 8.2 m and 20.0 m, while the tightest genuine
                          # margin on the course (gate 4) is 0.28 m, so a metres-wide
                          # tolerance separates real swaps from degenerate geometry
                          # instead of gambling on it.
CUR_ORDER_EVICT_S = 1.0   # a current track that stays FARTHER than a fresh lookahead
                          # detection for this long is a re-association error, not
                          # motion: the aircraft cannot recede from the gate it is
                          # flying at. Second line of defence for when the joint pass
                          # has no admissible candidate to swap the current slot to.
JOINT_SLOTS = True        # joint (vs greedy) slot assignment; off restores pre-fix
SLOT_JOINT_MARGIN = 0.15  # the joint assignment must beat the greedy one by this much
                          # relative cost before it is allowed to override it -- a tie
                          # keeps the incumbent, so this can only fix a swap, never
                          # churn a good assignment.
SLOT_BANDS = True         # TWO-SIDED per-slot range bands + in-order slot filling +
                          # slot exclusivity (2026-08-02 late). The three slots are a
                          # MONOTONE, INJECTIVE assignment onto gates k, k+1, k+2 at map-
                          # consistent ranges; every earlier fix here constrained the case
                          # last seen instead of that invariant. Off (--no-slot-bands)
                          # restores the pre-fix behaviour for a one-binary A/B.
SLOT_BAND_TOL_M = 5.0     # metres of absolute slack on a lookahead slot's expected-range
SLOT_BAND_TOL_K = 0.55    # band, plus this fraction of the MEASURED current-gate range.
                          # The band is built by placing the aircraft r0 short of the
                          # current gate ALONG the incoming map edge, so its error scales
                          # with r0 (how far short, and how much the real approach can
                          # deviate from that edge). At r0 = 7.5 m the band is +-5.0 m,
                          # which separates gate 6 (22.5 m) from gate 7 (38.1 m) with
                          # 10 m of margin; at r0 = 20 m it is +-16 m, i.e. barely
                          # binding -- deliberately, since the placement assumption is
                          # only exact at the two ends of a leg.
SLOT_DUP_TOL_M = 2.0      # two slots whose tracked positions sit closer than this (or
SLOT_DUP_TOL_K = 0.15     # than this fraction of the nearer range) are the SAME physical
                          # object, and the FARTHER slot loses -- the object cannot be a
                          # later gate than the nearest slot claiming it.
ASSOC_RADIUS_MIN = 40.0   # px association radius floor
ASSOC_RANGE_JUMP = 0.35   # relative range jump that breaks association (closing-speed cap)
RANGE_PRIOR_K = 1.3       # map-range prior (2026-08-02 wrong-gate bug): at crossing
                          # k -> k+1 the current gate STARTS near the k->k+1 edge length
                          # and its range can only shrink; a candidate beyond K x the
                          # remaining expected range is a lookahead gate, never current
CUR_EVICT_S = 0.5         # evict a current track that stays beyond that bound this long
NORMAL_MIN_PX = 20.0      # below this apparent size the IPPE tilt pair is noise-ranked;
                          # never assert normal_valid on a sub-20 px quad
RIBBON_PRIOR_DEG = 90.0   # ribbon is an UNDIRECTED line: accept a sample only within
                          # this angle of the course-progress prior, else SEARCH
IN_WHITE_MAX = 0.15       # interior-colour decoration test (mapvq2.clean thresholds)
IN_V_MAX = 150.0
IN_DARK_V = 90.0          # interiorgate.py 2026-08-02: a dark-pixel escape hatch. Claire's
IN_DARK_FRAC = 0.05       # live catch -- real apertures against the LIT CEILING (bright
                          # trusses/lights) still contain dark gaps (sky between trusses);
                          # decoration is a SOLID fill with none. On hand labels: bright&
                          # dark90<0.05 cuts true-gate loss 9.7%->1.4% (sure labels) at
                          # unchanged deco rejection (96.6%).
CYAN_MIN_FRAC = 0.0015    # ribbon dropout threshold (share of frame)
COMPASS_STRIDE = 10       # run skylight every Nth frame (~15 ms/call on this CPU; the
                          # sim gyro is near-noiseless so 0.33 s of pure integration
                          # costs well under a degree between fixes)
SEARCH_SWEEP_DPS = 60.0   # SEARCH-mode sweep rate when even the map has no opinion
K_YAW = 2.5               # AUTO_ATTENTION azimuth->zero P gain, rad/s per rad
HOVER_ACC_MIN = 0.35      # m/s^2 horizontal specific force below which vel_bearing is noise
COAST_TRANSLATE = True    # translate coasted tracks by the drag-derived own velocity, not
                          # only rotate them by the gyro. Measured 2026-08-02 on the VM
                          # race-pace session: rotation-only coasting left a median 3.26 m
                          # reacquire miss, and the miss correlates with gap x speed, which
                          # is the signature of un-modelled own TRANSLATION rather than
                          # rotation error. Off restores the previous behaviour for A/B.
COAST_SPEED_CONF_MIN = 0.55   # below this the speed estimate is at the accelerometer
                          # noise floor; translating on it would add noise, not signal.
                          # RAISED from 0.25 after measurement: on the gentle standing lap
                          # (20260801-121520, median speed 1.5 m/s) translating on a
                          # marginal estimate perturbed the predicted track position enough
                          # to lose associations, costing 2.2 points of current-gate
                          # validity -- the fix actively hurting in the regime it was never
                          # meant for. 0.55 corresponds to |a_xy| ~ 0.30 m/s^2, i.e.
                          # |v_xy| ~ 2.6 m/s: below race pace the coast stays rotation-only,
                          # which is what the gentle laps were already good at.
COAST_SIGMA_K = 0.35      # fraction of the translated distance added to range_sigma_m.
                          # The in-plane speed itself measures 7.2% median error out of
                          # sample, but the BODY-Z component is not observable at all
                          # (thrust sits on that axis, and the barometer is nan in this
                          # sim), so the honest per-step uncertainty is dominated by the
                          # un-modelled vertical term, not by the speed error.

# ---- speed_est validity: the two ways sqrt(|a_xy|/k) lies ------------------------------
# (1) IT HAS NO ZERO. A stationary airframe resting on any tilted surface reads |a_xy| =
#     3.0 m/s^2 on the VQ2 pad, i.e. 8.4 m/s at speed_conf 1.00, and COAST_TRANSLATE then
#     flies every coasted track at that phantom speed. Measured over the 42 s in which
#     20260802-163548 did not move at all: 8.06 / 8.08 / 8.10 m/s (p10/p50/p90) at conf
#     1.00. NOTHING IN THE IMU CAN CATCH THIS -- a park on a tilted pad and steady flight
#     at constant velocity are the same constant gravity vector to three decimals
#     (staticdet.py's docstring has both). The camera settles it in one number, and at
#     flight time we HAVE frames, so the offline detector's mechanism is ported below.
# (2) It degrades during transients, but NOT for the reason one would guess -- see
#     TRANSIENT_* below.
SPEED_VETO = True         # master switch for BOTH vetoes below. Off (--no-speed-veto)
                          # restores the pre-2026-08-02 estimator exactly -- phantom speed
                          # included -- so the A/B is one flag on one binary, not a diff
                          # against an older run whose other inputs may have moved.
IMG_SIZE_STATIC = (160, 120)  # staticdet.SIZE -- downscale kills JPEG block noise
IMG_M_STATIC = 0.25       # staticdet.M_STATIC. Mean |dI| between consecutive greyscale
                          # frames below this = the image is not changing. Calibrated
                          # offline, NOT re-invented here: frozen p99 0.125, moving p10
                          # ~1.7, so any threshold in 0.15-0.30 gives identical verdicts.
IMG_SMOOTH_S = 0.5        # staticdet.SMOOTH_S, but TRAILING rather than centred: online
                          # there is no future. Same sample count, ~0.25 s more lag.
IMG_STATIC_MIN_S = 2.0    # staticdet.STATIC_MIN_S. Parked is a SUSTAINED state; one
                          # glance at an untextured wall, or one stalled frame pair, is
                          # not. This run-length rule is what makes the low threshold
                          # safe (VQ1 truth-moving frames called parked: 1.16%).
                          # Cost of being causal: the first IMG_STATIC_MIN_S of a park
                          # still reports speed. That is the honest price of not
                          # pre-judging a low-texture glance, and it is 2 s of a 42 s bug.

# ---- transient degradation, and what was MEASURED rather than assumed ------------------
# THE STATED HYPOTHESIS WAS "in steady flight the horizontal specific force is drag alone,
# but while ACCELERATING it is drag + the linear acceleration, so speed_est is biased
# during exactly the transients a race consists of". IT IS FALSE, and the reason is worth
# keeping: an accelerometer measures SPECIFIC FORCE (non-gravitational forces / m), never
# coordinate acceleration. Thrust is body -z by definition, so the horizontal body pair is
# drag ALONE whether or not the aircraft is accelerating -- the acceleration is the
# CONSEQUENCE of those forces, not a further additive term in what the sensor reads.
# Measured directly at the force level on VQ1 truth (dragtransient.py, _dragforce path):
# regressing the drag residual (a_meas_xy + k|v_xy|v_xy) on the body-horizontal coordinate
# acceleration gives slope +0.007 (x) / +0.001 (y) pooled, +0.041 / +0.017 restricted to
# |gyro| < 0.2 rad/s. A full leak would be 1.000.
#
# WHAT IS REAL is a milder, differently-caused degradation: the drag residual grows from
# 0.38 to 1.19 m/s^2 median as |dv/dt| goes 0-2 -> 20-40 m/s^2, but it is only 4-6% of the
# acceleration and is NOT aligned with it (corr +0.05..+0.18) -- noise amplification, not a
# leak. In speed terms, med|e| 0.51 -> 1.00 m/s and p90 0.82 -> 2.12 m/s between steady and
# |dv/dt| >= 25 m/s^2, while the RELATIVE error is flat or better (7.2% -> 5.2%) because
# the aircraft is faster there. A timing-offset sweep of the truth stream does not remove
# it (flat minimum within +-16 ms), so it is not a truth-alignment artefact either.
#
# So the transient term is a PRECISION statement, deliberately bounded, not a refusal.
TRANSIENT_JERK_LO = 10.0  # m/s^3. |d|a_xy|/dt|. Bands (VQ1, n=17463): med|e| 0.51 below
TRANSIENT_JERK_HI = 50.0  # 2, 0.46 at 2-5, 0.45 at 5-10, 0.66 at 10-25, 0.88 at 25-50,
                          # 1.15 above 50. Degradation starts at ~10 and doubles by ~50.
TRANSIENT_GYRO_LO = 1.0   # rad/s. Same shape on the other proxy: med|e| 0.51 / p90 0.77
TRANSIENT_GYRO_HI = 3.0   # below 0.2 rad/s, 0.63 / 1.51 at 2-4.
TRANSIENT_FLOOR = 0.6     # the factor never goes below this, for two reasons. It encodes
                          # "about twice as noisy", which is the whole measured effect --
                          # claiming more would be inventing. And COAST_TRANSLATE gates on
                          # the NOISE part of the confidence, not on this factor, precisely
                          # so that aggressive rotation -- when detections are lost and
                          # coasting matters most -- cannot switch translation off over a
                          # 2x precision change worth centimetres per frame.

W, H = L.W, L.H
FX, CX, CY = L.FX, L.CX, L.CY
R_CB = L.body_to_cam()          # body FRD -> camera (right, down, forward)
R_BC = R_CB.T
K_CAM = np.array([[L.FX, 0, L.CX], [0, L.FY, L.CY], [0, 0, 1.0]])


def _lazy_net():
    global _G, _GC, _torch
    if _G is None:
        import torch as _t
        _t.set_num_threads(3)     # perception shares the CPU with control at 30 Hz
        import gatenet as g
        import gatenet_conf as gc
        _torch, _G, _GC = _t, g, gc
    return _G, _GC, _torch


def pixel_ray_body(u, v):
    """Unit ray in the BODY frame through pixel (u, v)."""
    r = np.array([(u - L.CX) / L.FX, (v - L.CY) / L.FY, 1.0])
    r /= np.linalg.norm(r)
    return R_BC @ r


def project_body(p_body):
    """Body-frame point -> (u, v) pixel, or None if behind the camera."""
    c = R_CB @ np.asarray(p_body, float)
    if c[2] <= 0.25:
        return None
    return (L.CX + L.FX * c[0] / c[2], L.CY + L.FY * c[1] / c[2])


# =========================================================================================
# IMU: gravity filter + levelled yaw integral (streaming port of skylight.Imu)


class ImuFilter:
    """Consumes RAW HIGHRES_IMU rows; owns the one legal gyro sign flip (the read side of
    the link layer, CONVENTIONS.md). State: unit gravity DOWN in body, levelled yaw
    integral psi_int (deg, positive nose-right), latest canonical gyro/accel."""

    def __init__(self):
        self.g = None                  # unit gravity down, body frame
        self.psi_int = None            # deg; None until the compass first anchors it
        self._psi_acc = 0.0            # gyro integral before the first anchor
        self.gyro = np.zeros(3)        # canonical rad/s
        self.accel = np.zeros(3)       # specific force, canonical (no sign flip)
        self.amag = 9.81
        self.jerk = 0.0                # |d|a_xy|/dt|, m/s^3 -- the online transient proxy
        self._ah = []                  # (t_s, |a_xy|) ring of 3, for a central difference
        self.t_ns = None
        self._Rstep = np.eye(3)        # body rotation since last consume() reset

    def push(self, t_ns, accel_raw, gyro_raw):
        a = np.asarray(accel_raw, float)
        w = -np.asarray(gyro_raw, float)          # THE mirror, applied here only
        dt = 0.0 if self.t_ns is None else min(max((t_ns - self.t_ns) / 1e9, 0.0), 0.1)
        self.t_ns = t_ns
        n = float(np.linalg.norm(a))
        if self.g is None:
            self.g = -a / max(n, 1e-9)
        else:
            self.g = self.g - np.cross(w, self.g) * dt
            self.g /= max(np.linalg.norm(self.g), 1e-9)
            err = abs(n - 9.81)
            alpha = 0.008 if err < 0.35 else (0.001 if err < 1.5 else 0.0001)
            self.g = (1.0 - alpha) * self.g + alpha * (-a / max(n, 1e-9))
            self.g /= max(np.linalg.norm(self.g), 1e-9)
        dpsi = math.degrees(float(self.g @ w) * dt)
        if self.psi_int is not None:
            self.psi_int += dpsi
        else:
            self._psi_acc += dpsi
        # incremental body rotation for track coasting: p' = p - w x p dt (small angle)
        wx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        self._Rstep = (np.eye(3) - wx * dt) @ self._Rstep
        # |a_xy| jerk by CENTRAL difference over the raw 61 Hz rows -- the identical
        # estimator dragtransient.py binned the error against, so the thresholds above
        # mean here what they meant there. Unsmoothed on purpose: smoothing it would
        # move the calibration.
        self._ah.append((t_ns / 1e9, float(math.hypot(a[0], a[1]))))
        if len(self._ah) > 3:
            del self._ah[0]
        if len(self._ah) == 3:
            dtj = self._ah[2][0] - self._ah[0][0]
            if dtj > 1e-6:
                self.jerk = abs(self._ah[2][1] - self._ah[0][1]) / dtj
        self.gyro, self.accel, self.amag = w, a, n

    def consume_rotation(self):
        """Body rotation accumulated since the last call; re-orthonormalised."""
        u, _, vt = np.linalg.svd(self._Rstep)
        R = u @ vt
        self._Rstep = np.eye(3)
        return R

    def anchor_psi(self, psi_mod90):
        """Compass fix: pull psi_int to the measurement on its mod-90 branch."""
        if self.psi_int is None:
            self.psi_int = psi_mod90
        else:
            self.psi_int += (psi_mod90 - self.psi_int + 45.0) % 90.0 - 45.0

    def roll_pitch(self):
        if self.g is None:
            return 0.0, 0.0, 0.0
        g = self.g
        roll = math.atan2(g[1], g[2])
        pitch = math.atan2(-g[0], math.hypot(g[1], g[2]))
        # Two degradation terms, both needed. |a|-g mismatch catches translational
        # bursts; it does NOT catch a coordinated banked turn, where |a| stays near g
        # while apparent gravity tilts with the body (the contract's own warning, and
        # skylight.Imu's docstring). Sustained body rate is the observable proxy for
        # that regime, so confidence also decays with |gyro|.
        conf_mag = max(0.0, 1.0 - abs(self.amag - 9.81) / 4.0)
        conf_turn = 1.0 / (1.0 + 0.8 * max(0.0, float(np.linalg.norm(self.gyro)) - 0.3))
        return roll, pitch, conf_mag * conf_turn

    def level_rotation(self):
        return SKY.level_rotation(self.g) if self.g is not None else None


# =========================================================================================
# Online static/parked detector: the referee the accelerometer cannot be


class ImageMotion:
    """Live port of staticdet.py -- mean |dI| between consecutive downscaled greyscale
    frames, the one measurement that separates a park on a tilted pad from steady flight.

    Nothing about the drone is assumed, only that the world outside is static and textured:
    a moving camera changes its image, a stationary one does not. Its claim is deliberately
    NEGATIVE -- "the image is not changing, therefore the aircraft is not moving through the
    scene" -- and it ABSTAINS (reports moving) whenever it has not been fed frames, so it
    can only ever remove a speed it has positive evidence against.

    Differences from the offline version, both forced by causality and both stated because
    they are the only places the calibration does not transfer verbatim:
      * the 0.5 s median is TRAILING, not centred (same sample count, ~0.25 s more lag);
      * the run-length rule can only look backwards, so a park is declared
        IMG_STATIC_MIN_S after it starts rather than from its first frame.
    Thresholds are staticdet's, unchanged: any value in 0.15-0.30 gives the same verdicts,
    the frozen-image p99 is 0.125 and the moving p10 is ~1.7.
    """

    def __init__(self):
        self.prev = None
        self.hist = []          # trailing (t_s, m) over IMG_SMOOTH_S
        self.m = float('nan')   # median-smoothed mean |dI|, latest
        self.low_since = None   # start of the current sub-threshold run
        self.static = False
        self.n = 0

    def push(self, frame_bgr, t_s):
        """-> True when the image has been static for IMG_STATIC_MIN_S."""
        if frame_bgr is None:
            return self.static
        g = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                       IMG_SIZE_STATIC).astype(np.float32)
        if self.prev is not None:
            self.hist.append((t_s, float(np.mean(np.abs(g - self.prev)))))
            while self.hist and t_s - self.hist[0][0] > IMG_SMOOTH_S:
                del self.hist[0]
            self.m = float(np.median([v for _, v in self.hist]))
            self.n += 1
            if self.m < IMG_M_STATIC:
                if self.low_since is None:
                    self.low_since = t_s
                self.static = (t_s - self.low_since) >= IMG_STATIC_MIN_S
            else:
                self.low_since = None
                self.static = False
        self.prev = g
        return self.static


def transient_factor(jerk, gyro_norm):
    """Multiplicative precision factor on speed_conf, in [TRANSIENT_FLOOR, 1].

    Both proxies are ramps read off the measured error bands (constants above), and the
    weaker of the two wins. This does NOT encode the falsified "linear acceleration leaks
    into the accelerometer" story -- it encodes the measured fact that the estimate is
    about twice as noisy when |a_xy| or the body rate is changing fast.
    """
    span = 1.0 - TRANSIENT_FLOOR
    fj = 1.0 - span * (jerk - TRANSIENT_JERK_LO) / (TRANSIENT_JERK_HI - TRANSIENT_JERK_LO)
    fg = 1.0 - span * (gyro_norm - TRANSIENT_GYRO_LO) / (TRANSIENT_GYRO_HI - TRANSIENT_GYRO_LO)
    return float(np.clip(min(fj, fg), TRANSIENT_FLOOR, 1.0))


# =========================================================================================
# Map lookahead: layout positions + online frame anchoring


def _seed_clash(box, seeds, frac=0.5):
    """Would this net seed box crop (mostly) the same object as one already seeded?
    Intersection over the SMALLER box, so a small box inside a big one counts."""
    x0, y0, x1, y1 = box
    aw, ah = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    for _g, (u0, v0, u1, v1) in seeds:
        iw = min(x1, u1) - max(x0, u0)
        ih = min(y1, v1) - max(y0, v0)
        if iw <= 0 or ih <= 0:
            continue
        small = min(aw * ah, max(u1 - u0, 1e-6) * max(v1 - v0, 1e-6))
        if iw * ih / small > frac:
            return True
    return False


class MapModel:
    """map_vq2.json layout_directions positions, plus the ONLINE frame anchor.

    The layout is stored in sketch-aligned axes; the sketch frame may be mirrored
    relative to the physical levelled frame (grid_to_sketch_reflection). Rather than
    trusting a composition of stored transforms, the anchor is FIT online: every time two
    gates are measured in the same frame, (bearing_map, bearing_internal) is recorded,
    and s (+1/-1 mirror) and c (offset) are refit. Until >=2 distinct pairs are seen the
    map can only say DISTANCES, and SEARCH falls back to a plain sweep."""

    def __init__(self, path=MAP_JSON):
        self.pos = None
        self.n = N_GATES_MAP
        self.pairs = {}                    # (i, j) -> measured dist_m (i < j)
        try:
            m = json.load(open(path))
            ld = m['layout_directions_2026_08_02']
            self.pos = np.array([[ld['positions_m'][str(i)]['x'],
                                  ld['positions_m'][str(i)]['y'],
                                  ld['positions_m'][str(i)]['z_up_m']]
                                 for i in range(self.n)])
            for k, v in m.get('measured_pairs', {}).items():
                i, j = sorted(int(x) for x in k.split('-'))
                self.pairs[(i, j)] = float(v['dist_m'])
        except Exception as e:                               # map optional, never fatal
            print(f'[producer] map unavailable ({e}); lookahead disabled', file=sys.stderr)
        self.obs = deque(maxlen=64)        # (b_map_deg, b_int_deg) anchor observations
        self.s = None                      # +1/-1 once resolved
        self.c = 0.0

    def map_bearing(self, i, j):
        v = self.pos[j] - self.pos[i]
        return math.degrees(math.atan2(v[1], v[0]))

    def dist(self, i, j):
        """Inter-gate distance in metres: measured pair if we have one, else the solved
        layout, else None. Distances are mirror- and compass-free."""
        if not (0 <= i < self.n and 0 <= j < self.n):
            return None
        key = (min(i, j), max(i, j))
        if key in self.pairs:
            return self.pairs[key]
        if self.pos is not None:
            return float(np.linalg.norm(self.pos[j] - self.pos[i]))
        return None

    def add_anchor(self, i, j, b_int_deg):
        if self.pos is None:
            return
        self.obs.append((self.map_bearing(i, j), b_int_deg))
        self._fit()

    def _fit(self):
        if len(self.obs) < 2:
            return
        bm = np.radians([o[0] for o in self.obs])
        bi = np.radians([o[1] for o in self.obs])
        fits = []
        for s in (1.0, -1.0):
            d = bi - s * bm
            c = math.atan2(np.mean(np.sin(d)), np.mean(np.cos(d)))
            res = np.abs((d - c + np.pi) % (2 * np.pi) - np.pi)
            fits.append((float(np.mean(res)), s, math.degrees(c)))
        fits.sort()
        # s is only observable from >=2 DISTINCT map bearings, and only lockable with
        # a real margin between the mirror hypotheses
        db = (np.degrees(bm) - np.degrees(bm)[0] + 180.0) % 360.0 - 180.0
        if float(np.ptp(db)) > 15.0 and fits[1][0] - fits[0][0] > math.radians(10.0):
            self.s, self.c = fits[0][1], fits[0][2]

    def predict_vector_lev(self, i, j, psi_int_deg):
        """Levelled-body-frame vector gate i -> gate j (x fwd, y right, z down), or None."""
        if self.pos is None or self.s is None or psi_int_deg is None:
            return None
        v = self.pos[j] - self.pos[i]
        d_h = math.hypot(v[0], v[1])
        b_int = self.s * self.map_bearing(i, j) + self.c
        ang = math.radians(b_int - psi_int_deg)
        return np.array([d_h * math.cos(ang), d_h * math.sin(ang), -v[2]])


class CourseNormals:
    """Per-gate expected PLANE-NORMAL azimuth, from pilot/course/course_vq2.json.

    Uses `yaw_race_bisector_deg` -- the bisector of the incoming and outgoing racing-line
    directions -- and NOT `yaw_deg`. Two reasons, and the second is the load-bearing one:
      * the bisector exists for all 17 gates; yaw_deg is null at 8, 12 and 13 (the map
        refused them, IPPE head-on valley, MAD > 20 deg);
      * yaw_deg was MEASURED from PnP normals + the skylight compass, i.e. from the very
        channel this prior is supposed to referee. Using it would close the loop again --
        the same self-confirming structure the accumulator exists to break. The bisector
        comes from gate POSITIONS (the racing line), so it is independent of tilt.
    Its accuracy is known and modest: median |bisector - measured| = 10.9 deg over the 14
    gates where both exist, with a 48 deg tail (gate 7). SIGMA_MAP_DEG is set from that.

    The azimuths live in the map/sketch frame, so they only become usable once
    MapModel's online (s, c) anchor has resolved AND the compass has anchored psi_int.
    IF EITHER IS WRONG the prior points at a wrong azimuth (a wrong mirror s reflects it),
    which is why normalfuse clamps the priors' total contribution to the likelihood
    ratio -- see PRIOR_LLR_CAP there.
    """

    def __init__(self, path=COURSE_JSON):
        self.bisector = {}
        try:
            c = json.load(open(path))
            for g in c['gates']:
                y = g.get('yaw_race_bisector_deg')
                if y is not None:
                    self.bisector[int(g['id'])] = float(y)
        except Exception as e:
            print(f'[producer] course normals unavailable ({e}); map prior off',
                  file=sys.stderr)


# =========================================================================================
# Gate track


class Track:
    __slots__ = ('gate', 'pos', 'normal', 'normal_valid', 'quad', 'size_px', 'conf',
                 't_obs', 'pose_valid', 'range_sigma', 'n_obs', 'nrm', 'normal_src',
                 'coast_dist')

    def __init__(self, gate):
        self.gate = gate
        self.pos = None
        self.normal = None
        self.normal_valid = False
        self.nrm = NF.NormalAccumulator(gate)     # per-track IPPE twin accumulator
        self.normal_src = 'none'
        self.quad = None
        self.size_px = 0.0
        self.conf = 0.0
        self.t_obs = -1e9
        self.pose_valid = False
        self.range_sigma = 0.0
        self.n_obs = 0
        self.coast_dist = 0.0      # metres of own displacement applied since last fix

    def translate(self, d):
        """Subtract own displacement. A gate is static in the world, so own motion moves
        it in the BODY frame by exactly -d. Rotation alone keeps the bearing right and
        lets the RANGE go stale, which at race pace is metres per coast step."""
        if self.pos is not None:
            self.pos = self.pos - d

    def rotate(self, R):
        if self.pos is not None:
            self.pos = R @ self.pos
        if self.normal is not None:
            self.normal = R @ self.normal
        self.nrm.rotate(R)      # the accumulator's common frame un-does the same step


def interior_colour(hsv, quad):
    """(in_white, in_v, in_dark) of the quad interior shrunk to 70% -- vq2cache.py's
    test, plus the dark-pixel escape hatch (interiorgate.py 2026-08-02): in_dark is the
    fraction of the interior with V < IN_DARK_V. A real aperture against a lit ceiling
    still has dark gaps (sky between trusses); a decoration fill does not -- see
    is_decoration()."""
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    ctr = q.mean(0)
    poly = (ctr + (q - ctr) * 0.7).astype(np.int32)
    msk = np.zeros(hsv.shape[:2], np.uint8)
    cv2.fillPoly(msk, [poly], 1)
    px = hsv[msk.astype(bool)]
    if len(px) < 4:
        return 0.0, 0.0, 1.0
    return (float(np.mean((px[:, 1] < 70) & (px[:, 2] > 150))), float(px[:, 2].mean()),
            float(np.mean(px[:, 2] < IN_DARK_V)))


def n_corners_in_frame(quad, margin=2.0):
    """How many of the quad's four corners actually lie inside the image.

    The trust variable for close gates (cornergate.py 2026-08-02). Corners outside the
    rectangle are the fitter's EXTRAPOLATION, not observation, so a residual computed
    against them scores invented geometry. Catastrophe rate falls off a cliff between one
    and two in-frame corners (44.4% -> 3.4%), which is where MIN_CORNERS_IN_FRAME sits.
    """
    q = np.asarray(quad, np.float64).reshape(-1, 2)
    return int(np.sum((q[:, 0] >= -margin) & (q[:, 0] <= W + margin) &
                      (q[:, 1] >= -margin) & (q[:, 1] <= H + margin)))


def is_decoration(in_white, in_v, in_dark):
    """True -> reject as decoration. Bright-interior test with the dark-gap escape
    hatch: a bright reading (white fill OR high mean V) is overridden if the interior
    still contains real dark patches (IN_DARK_FRAC of pixels below IN_DARK_V) -- exactly
    the lit-ceiling-background signature a solid decoration fill cannot produce."""
    bright = in_white > IN_WHITE_MAX or in_v > IN_V_MAX
    return bright and in_dark < IN_DARK_FRAC


VERT_DEBUG = False    # set True by an analysis script to log (gate_idx, tilt0, tilt1,
                      # resolved_by) for every ambiguous pose, PRIOR-INDEPENDENT (logged
                      # before any exemption/resolution is applied) -- residgate/vert
                      # measurement scripts use this, never the live flight path.
_VERT_LOG = []


def pnp_pose(quad_img, prev_normal=None, gravity_body=None, attitude_conf=0.0,
             gate_idx=None):
    """Ordered quad (TL,TR,BR,BL image px) -> dict or None.

    solvePnPGeneric(IPPE_SQUARE) returns both tilt solutions. Direction: normal signed
    toward the camera. TILT IS NOT DECIDED HERE any more (2026-08-02): both solutions are
    returned under 'sols' with an 'ambiguous' flag, and _fuse_normal hands them to the
    track's normalfuse accumulator. 'normal_valid' in the returned dict therefore means
    only "the reprojection itself separated the branches"; the accumulator has the final
    say. LEGACY_TILT=True restores the old per-frame temporal+verticality resolver for
    A/B measurement -- it is retained as a measurement tool, not as a fallback."""
    imgp = np.ascontiguousarray(quad_img, np.float64).reshape(4, 1, 2)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            D.OBJ, imgp, K_CAM, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    if n == 0:
        return None
    sols = []
    for k in range(n):
        R, _ = cv2.Rodrigues(rvecs[k])
        t = tvecs[k].reshape(3)
        if t[2] <= 0.1:
            continue
        nc = R[:, 2]
        if float(nc @ t) > 0:
            nc = -nc
        proj, _ = cv2.projectPoints(D.OBJ, rvecs[k], tvecs[k], K_CAM, None)
        proj = proj.reshape(4, 2)
        # residual MODULO the square's 4-fold symmetry: near head-on, OpenCV may return
        # a pose whose in-plane 90-deg branch differs from our winding. That branch is
        # unobservable for a square (t and the normal are identical), so scoring it as
        # residual would reject perfectly good poses (measured: rms 33.9 px on an EXACT
        # synthetic head-on quad, t error zero).
        rms = min(float(np.sqrt(np.mean((np.roll(pp, r, axis=0) - quad_img) ** 2)))
                  for pp in (proj, proj[::-1]) for r in range(4))
        sols.append({'pos': R_BC @ t, 'normal': R_BC @ nc, 'rms': rms,
                     'range': float(np.linalg.norm(t))})
    if not sols:
        return None
    sols.sort(key=lambda s: s['rms'])
    best = sols[0]
    normal_valid = True
    ambiguous = len(sols) > 1 and sols[1]['rms'] < best['rms'] * TILT_AMBIG_RATIO
    extra = {'sols': sols[:2], 'ambiguous': ambiguous}
    if not LEGACY_TILT:
        # The per-frame resolver is retired: the two candidates and their residuals are
        # handed UP to the per-track accumulator (normalfuse.NormalAccumulator), which
        # decides across views instead of against the producer's own previous pick.
        # normal_valid here means only "the reprojection itself settled the branch".
        return {**best, 'normal_valid': not ambiguous, **extra}
    if VERT_DEBUG and gravity_body is not None and len(sols) > 1:
        up0 = -np.asarray(gravity_body, float)
        up0 /= max(float(np.linalg.norm(up0)), 1e-9)
        t0 = 90.0 - math.degrees(math.acos(np.clip(abs(float(sols[0]['normal'] @ up0)),
                                                    0.0, 1.0)))
        t1 = 90.0 - math.degrees(math.acos(np.clip(abs(float(sols[1]['normal'] @ up0)),
                                                    0.0, 1.0)))
        _VERT_LOG.append((gate_idx, t0, t1, sols[0]['rms'], sols[1]['rms']))
    if len(sols) > 1 and sols[1]['rms'] < best['rms'] * TILT_AMBIG_RATIO:
        # near-degenerate tilt pair: temporal consistency first, verticality fallback
        resolved = False
        if prev_normal is not None:
            angs = [math.degrees(math.acos(np.clip(s['normal'] @ prev_normal, -1, 1)))
                    for s in sols[:2]]
            if min(angs) < TILT_CONSIST_DEG and abs(angs[0] - angs[1]) > 15.0:
                best = sols[int(np.argmin(angs))]
                resolved = True
        if not resolved and gravity_body is not None \
                and attitude_conf >= VERTICALITY_CONF_MIN \
                and gate_idx not in VERTICALITY_EXEMPT_GATES:
            up = -np.asarray(gravity_body, float)
            up /= max(float(np.linalg.norm(up)), 1e-9)
            # tilt-from-vertical: 0 deg = normal exactly horizontal (a vertical gate
            # face); unsigned dot since the normal's camera-facing sign is irrelevant
            # to the PLANE's orientation.
            tilts = [90.0 - math.degrees(math.acos(np.clip(abs(float(s['normal'] @ up)),
                                                           0.0, 1.0)))
                     for s in sols[:2]]
            if min(tilts) <= VERTICALITY_TOL_DEG \
                    and abs(tilts[0] - tilts[1]) >= VERTICALITY_MARGIN_DEG:
                best = sols[int(np.argmin(tilts))]
                resolved = True
        normal_valid = resolved
    return {**best, 'normal_valid': normal_valid, **extra}


# =========================================================================================
# The producer


class Producer:
    def __init__(self, use_net=True, use_compass=True, map_path=MAP_JSON, n_gates=17):
        self.imu = ImuFilter()
        self.map = MapModel(map_path)
        self.course = CourseNormals()
        self.normal_log = []       # NORMAL_DEBUG rows (analysis only; off by default)
        self.tracks: dict[int, Track] = {}
        self.n_gates = n_gates
        self.use_compass = use_compass
        self.prev_active = -1
        self.t_last_gate = None
        self.cur_bound0 = None     # map-range prior, edge-based: set at each crossing,
                                   # stable within a leg -- the EVICTION referee
        self.cur_rmax = None       # ratcheted (only shrinks) copy: candidate ADMISSION
                                   # only, where a too-tight frame costs one association,
                                   # not a good track
        self.cur_over_t = None     # when the current track first exceeded cur_bound0
        self.cur_order_t = None    # when the current track first fell BEHIND a lookahead
        self.prog_dir = None       # course-progress unit vector, body frame, gyro-coasted
                                   # (last known current-gate direction: compass-free)
        self.frame_i = 0
        self.ribbon_stale_t = None
        self.ribbon_last = None
        self.search_phase = 0.0
        self.reacq = []            # (staleness_s, miss_m, miss_px, speed_mps,
                                   #  coast_dist_m) at re-acquisition: the coasting
                                   # verification channel. speed and coast distance are
                                   # carried so the miss can be regressed on them -- the
                                   # test that separates a rotation error (miss ~ gap x
                                   # rate) from a translation error (miss ~ gap x speed).
        self.last_speed = 0.0
        self.last_speed_conf = 0.0
        self.last_speed_conf_noise = 0.0   # the |a_xy| noise-floor part ONLY: the coast
                                           # gate reads this, not the transient factor
        self.imgmot = ImageMotion()
        self.n_static = 0                  # frames the camera called parked
        self.t_prev = None
        self.slot_fixes = 0        # joint-assignment repairs of an infeasible assignment
        self.slot_evicts = 0       # current tracks evicted for being behind a lookahead
        self.slot_moves = 0        # lookahead matches re-homed to a NEARER empty slot
        self.slot_band_drops = 0   # slot occupants dropped for violating their map band
        self.slot_dupes = 0        # frames where one object was claimed by two slots
        self._bands = {}           # per-frame lookahead slot -> (lo, hi) range band
        self.slot_viol = [0, 0]    # [frames checkable, frames violating map geometry]
        self.slot_viol_fresh = [0, 0]   # same, but only slots measured THIS frame
        self.slot_order = [0, 0]   # [frames with >=2 slots, frames with range ordering
                                   #  inverted by more than 1 m]
        self.timing = {'detect': [], 'net': [], 'compass': [], 'pnp': [], 'imgmot': [],
                       'total': []}
        self.trunk = self.head = None
        if use_net:
            G, GC, torch = _lazy_net()
            self.trunk = GC.load_trunk(TRUNK_CKPT, torch.device('cpu'))
            self.head = GC.ConfHead(160)
            ck = torch.load(HEAD_CKPT, map_location='cpu', weights_only=False)
            self.head.load_state_dict(ck['head'])
            self.head.eval()

    # ---- gatenet on a batch of seed boxes ------------------------------------------
    def _net_quads(self, img, seeds):
        """seeds: list of (slot_gate, (x0,y0,x1,y1)). -> {gate: (quad, p, pred_px)}"""
        if self.trunk is None or not seeds:
            return {}
        G, GC, torch = _lazy_net()
        crops, geos = [], []
        for _g, (x0, y0, x1, y1) in seeds:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            S = max(x1 - x0, y1 - y0, G.MIN_VIS_PX) * G.MARGIN
            crop = G.make_crop(img, cx, cy, S)
            x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
            crops.append(x.div_(255.0).sub_(0.45).div_(0.25))
            geos.append((cx, cy, S))
        with torch.no_grad():
            xb = torch.stack(crops)
            feat = self.trunk.body(self.trunk.stem(xb))
            pred8 = self.trunk.head(feat)
            logit, logerr = self.head(feat)
        out = {}
        for k, (g, _) in enumerate(seeds):
            cx, cy, S = geos[k]
            quad = G.from_norm(pred8[k].numpy(), cx, cy, S).astype(np.float64)
            p = float(1.0 / (1.0 + math.exp(-float(logit[k]))))
            pred_px = float(10.0 ** float(logerr[k])) * S / 2.0
            out[g] = (D.order_quad(quad), p, pred_px)
        return out

    # ---- one frame -----------------------------------------------------------------
    def step(self, frame_bgr, imu_rows, race, t_s):
        """frame_bgr: 640x360 BGR or None. imu_rows: iterable of raw HIGHRES_IMU rows
        (dicts with t_wall_ns/xacc..zgyro, or (t_ns, ax..gz) tuples) since last step.
        race: dict with active_gate_index / race_time_s / armed / t_since_collision_s /
        collision_episodes (missing keys default). t_s: seconds (monotonic or wall)."""
        t_total = time.perf_counter()
        for r in imu_rows:
            if isinstance(r, dict):
                self.imu.push(float(r['t_wall_ns']),
                              (float(r['xacc']), float(r['yacc']), float(r['zacc'])),
                              (float(r['xgyro']), float(r['ygyro']), float(r['zgyro'])))
            else:
                self.imu.push(r[0], r[1:4], r[4:7])
        # The camera BEFORE the speed, because it is the only thing that can veto it.
        t_im = time.perf_counter()
        if self.imgmot.push(frame_bgr, t_s):
            self.n_static += 1
        self.timing['imgmot'].append(time.perf_counter() - t_im)
        Rstep = self.imu.consume_rotation()
        for tr in self.tracks.values():
            tr.rotate(Rstep)
        if self.prog_dir is not None:
            self.prog_dir = Rstep @ self.prog_dir
        # Own translation since the last step. Rotation alone is not enough at race pace:
        # a gate is static in the world, so flying 3 m toward it moves it 3 m closer in
        # the body frame, and a rotation-only coast keeps reporting the old range.
        # v_body comes from the drag inversion (dragmodel.py) -- an algebraic read of the
        # accelerometer, no integration -- and only its IN-PLANE part is trusted, because
        # the body-z component is not observable under VQ2.
        self.last_speed, self.last_speed_conf, v_body = self._speed()
        # The gate is the NOISE-FLOOR confidence, not the published one. The published
        # confidence also carries the transient precision factor, and letting that switch
        # translation off would lose the coast exactly during the aggressive rotation that
        # makes tracks coast in the first place -- over a 2x change in a per-frame
        # displacement of a few centimetres. v_body is None whenever the estimate is
        # REFUSED (below the noise floor, above the contact guard, or the camera says
        # parked), so the phantom-speed case cannot reach this branch by any route.
        if COAST_TRANSLATE and v_body is not None \
                and not (SPEED_VETO and self.imgmot.static) \
                and self.last_speed_conf_noise >= COAST_SPEED_CONF_MIN \
                and self.t_prev is not None:
            dt = min(max(t_s - self.t_prev, 0.0), 0.2)
            d = v_body * dt
            dn = float(np.linalg.norm(d))
            for tr in self.tracks.values():
                if tr.pos is not None and (t_s - tr.t_obs) > 0.0:
                    tr.translate(d)
                    tr.coast_dist += dn
        self.t_prev = t_s

        active = int(race.get('active_gate_index', 0))
        if active < self.prev_active:                    # race reset
            self.tracks.clear()
            self.t_last_gate = t_s
            self.cur_bound0 = self.cur_rmax = self._reset_bound(active)
            self.cur_over_t = None
            self.prog_dir = None
        elif active > self.prev_active:
            self.t_last_gate = t_s
            for g in list(self.tracks):                  # drop everything behind us
                if g < active:
                    del self.tracks[g]
            # Map-range prior. At a real crossing we are AT gate prev_active, so the
            # new current gate starts near the crossed edge's length. At the very
            # first step (prev_active == -1: the start pad) there is no crossed edge;
            # a "gate 0" farther away than the whole 0-1 edge is more plausibly gate 1
            # -- exactly the observed wrong-gate lock -- so bound by that edge, plain.
            d = self.map.dist(self.prev_active, active) if self.prev_active >= 0 else None
            self.cur_bound0 = self.cur_rmax = RANGE_PRIOR_K * d if d is not None \
                else self._reset_bound(active)
            self.cur_over_t = None
        self.prev_active = active
        slots = [active + k for k in range(I.N_GATES) if active + k < self.n_gates]

        ribbon = I.RibbonObs()
        if frame_bgr is not None:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)   # shared: detect's own
            self._vision(frame_bgr, slots, t_s, hsv)           # cvtColor is internal to
            ribbon = self._ribbon(hsv, t_s)                    # it, but interior test +
            self.frame_i += 1                                  # ribbon reuse this one

        obs = I.Observation(t_s=t_s, dt_s=1.0 / 30.0)
        obs.ribbon = ribbon
        obs.own = self._selfobs()
        obs.race = I.RaceObs(
            active_gate_index=active,
            n_gates_total=int(race.get('n_gates_total', self.n_gates)),
            t_since_gate_s=(t_s - self.t_last_gate) if self.t_last_gate is not None
            else float(race.get('t_since_gate_s', 0.0)),
            race_time_s=float(race.get('race_time_s', 0.0)),
            armed=bool(race.get('armed', False)),
            t_since_collision_s=float(race.get('t_since_collision_s', 1e3)),
            collision_episodes=int(race.get('collision_episodes', 0)))
        for k, g in enumerate(slots):
            obs.gates[k] = self._gateobs(g, t_s)
        self._slot_audit(slots, obs)
        obs.attention = self._attention(obs, slots, t_s)
        self.timing['total'].append(time.perf_counter() - t_total)
        return obs

    # ---- vision core -----------------------------------------------------------------
    def _vision(self, img, slots, t_s, hsv):
        t0 = time.perf_counter()
        dets = D.detections(img)
        for d in dets:
            d['in_white'], d['in_v'], d['in_dark'] = interior_colour(hsv, d['quad'])
        self.timing['detect'].append(time.perf_counter() - t0)

        # compass (internal only)
        if self.use_compass and self.frame_i % COMPASS_STRIDE == 0 \
                and self.imu.g is not None:
            t0 = time.perf_counter()
            theta, conf = SKY.heading(img, self.imu.g)
            if (theta is not None and conf['n_seg'] >= SKY.CONF_MIN_SEG
                    and conf['spread_deg'] is not None
                    and conf['spread_deg'] <= SKY.CONF_MAX_SPREAD_DEG):
                self.imu.anchor_psi(conf['psi_mod90'])
            self.timing['compass'].append(time.perf_counter() - t0)

        # ---- map-range eviction of an implausible CURRENT track ----------------------
        # Once a wrong track owns the current slot, association stickiness reinforces it
        # forever (a far tiny gate measures consistently). The map bound is the referee:
        # a current track persistently beyond the EDGE-BASED bound is some LOOKAHEAD
        # gate holding the wrong slot -- drop it and re-acquire under the bound.
        # (Eviction deliberately uses cur_bound0, not the ratcheted cur_rmax: clipped/
        # oblique near-gate ranges wobble by metres, and losing a good track costs far
        # more than admitting a candidate one frame late.)
        tr0 = self.tracks.get(slots[0]) if slots else None
        if (tr0 is not None and tr0.pos is not None and self.cur_bound0 is not None
                and float(np.linalg.norm(tr0.pos)) > self.cur_bound0):
            if self.cur_over_t is None:
                self.cur_over_t = t_s
            elif t_s - self.cur_over_t > CUR_EVICT_S:
                del self.tracks[slots[0]]
                self.cur_over_t = None
        else:
            self.cur_over_t = None

        # ---- ORDERING eviction: the current gate cannot be behind a lookahead one -----
        # Claire's 2026-08-02 second frame: an inherited far track (gate 11 at 28 m) kept
        # the current slot while the gate filling the frame at 8 m was labelled k+1. The
        # edge-based bound could not see it -- 1.3 x edge(10,11) is 44 m and both ranges
        # pass. What is impossible is the ORDER: the aircraft is flying AT the current
        # gate, so it cannot be receding from it while a later gate sits nearer. This is
        # the approach-monotonicity metric used as a live guard rather than a report.
        tr0 = self.tracks.get(slots[0]) if slots else None
        if JOINT_SLOTS and tr0 is not None and tr0.pos is not None and len(slots) > 1:
            r0 = float(np.linalg.norm(tr0.pos))
            # Only a POSE-BEARING, freshly measured lookahead track may convict the
            # current one. Evicting is expensive -- it forces a re-acquisition, and two
            # spurious evictions cost 2.4 points of current-gate validity on the gentle
            # standing lap -- so the evidence has to be the best the producer has, not
            # merely the nearest thing it happens to be holding.
            nearer = [float(np.linalg.norm(self.tracks[g].pos))
                      for g in slots[1:]
                      if g in self.tracks and self.tracks[g].pos is not None
                      and self.tracks[g].pose_valid
                      and (t_s - self.tracks[g].t_obs) < 0.15]
            if nearer and r0 > min(nearer) + SLOT_ORDER_TOL:
                if self.cur_order_t is None:
                    self.cur_order_t = t_s
                elif t_s - self.cur_order_t > CUR_ORDER_EVICT_S:
                    del self.tracks[slots[0]]
                    self.cur_order_t = None
                    self.slot_evicts += 1
            else:
                self.cur_order_t = None
        else:
            self.cur_order_t = None

        # ---- TWO-SIDED map bands per lookahead slot ---------------------------------
        # The current gate's MEASURED range r0 pins the aircraft on the incoming map edge
        # (exact at both ends of a leg, interpolating in between), which turns the map
        # into a two-sided expected range for every later gate. The old prior had only an
        # UPPER bound, so a candidate far too NEAR for its slot was still admitted -- that
        # is how gate 6 at 23 m came to sit in the +2 slot while +1 stayed empty. Bands
        # need no compass and no frame anchor: distances and one edge direction only.
        tr0 = self.tracks.get(slots[0]) if slots else None
        r0_ref = None
        if tr0 is not None and tr0.pos is not None and (t_s - tr0.t_obs) < 0.7:
            r0_ref = float(np.linalg.norm(tr0.pos))
        self._bands = {g: self._slot_band(slots[0], r0_ref, g) for g in slots[1:]} \
            if (SLOT_BANDS and r0_ref is not None) else {}
        self._bands = {g: b for g, b in self._bands.items() if b is not None}

        # ---- predictions per slot -------------------------------------------------
        preds = {}
        for g in slots:
            tr = self.tracks.get(g)
            if tr is not None and tr.pos is not None:
                uv = project_body(tr.pos)
                preds[g] = (uv, tr, float(np.linalg.norm(tr.pos)))
            else:
                p = self._map_predict(g)
                uv = project_body(p) if p is not None else None
                preds[g] = (uv, None, float(np.linalg.norm(p)) if p is not None else None)

        # ---- association (current slot first, greedy; then a joint pass) -------------
        used = set()
        matched = {}
        matched_j = {}              # slot -> detection index, for the joint re-check
        cands = {}                  # slot -> [(det_index, px_dist, radius)] admissible
        for g in slots:
            uv, tr, rng = preds[g]
            cand = None
            radius = ASSOC_RADIUS_MIN
            if uv is not None:
                # radius grows with staleness (a coasted prediction is soft) and is
                # generous for pure map predictions (solve residuals + psi drift)
                stale = (t_s - tr.t_obs) if tr is not None else 1.0
                radius = max(ASSOC_RADIUS_MIN,
                             1.2 * (tr.size_px if tr is not None else 0.0),
                             40.0 + 150.0 * min(stale, 1.5))
                tol = max(3.0, (ASSOC_RANGE_JUMP + 0.5 * min(stale, 1.5)) * (rng or 0))
                best_d = radius
                for j, d in enumerate(dets):
                    if j in used or self._is_deco(d, current=(g == slots[0])):
                        continue
                    r_d = d['range_size_m'] if d['source'] == 'outer' else d['range_m']
                    if g == slots[0] and self.cur_rmax is not None \
                            and r_d > self.cur_rmax:
                        continue          # map says: that is a lookahead gate
                    bnd = self._bands.get(g)
                    if bnd is not None and not (bnd[0] <= r_d <= bnd[1]):
                        continue          # two-sided map band: wrong SLOT for that range
                    dist = float(np.hypot(*(d['centre'] - np.array(uv))))
                    if rng is not None and abs(r_d - rng) > tol:
                        continue
                    # keep EVERY admissible candidate, not just the nearest: a swap is a
                    # property of the assignment, so the joint pass below needs the
                    # alternatives that greedy would have thrown away
                    cands.setdefault(g, []).append((j, dist, radius))
                    if dist < best_d:
                        best_d, cand = dist, j
            if cand is None and g == slots[0]:
                # CURRENT-gate fallback bootstrap: the race index is the identity
                # referee here, so never let a soft prediction block acquisition --
                # but a FRESH track's prediction is hard evidence, and overriding it
                # with "largest detection anywhere" is how identity swaps happen.
                # (After a crossing the k+1 lookahead track is INHERITED -- tracks are
                # keyed by absolute gate index -- so this bootstrap only runs when no
                # inherited track matched.)
                # The current gate is by construction the NEAREST upcoming gate, and
                # exactly when it is huge/clipped the contour detector only has an
                # OUTER (or no inner) quad for it while the NEXT gate offers a small
                # clean inner one. So: largest-apparent-size first (dets order),
                # OUTER detections admitted, and the map-range bound applied -- the
                # 2026-08-02 wrong-gate lock was a small clean 23.8 m quad taken as
                # gate 0 while gate 0 filled the frame (0-1 edge: 20.4 m).
                fresh_pred = (tr is not None and (t_s - tr.t_obs) < 0.4
                              and uv is not None)
                for j, d in enumerate(dets):
                    if j in used or self._is_deco(d, True):
                        continue
                    r_d = d['range_size_m'] if d['source'] == 'outer' else d['range_m']
                    if r_d > 60.0:
                        continue
                    if self.cur_rmax is not None and r_d > self.cur_rmax:
                        continue          # beyond the map bound: lookahead, not current
                    if fresh_pred and float(np.hypot(*(d['centre'] - np.array(uv)))) \
                            > max(150.0, 3.0 * tr.size_px):
                        continue
                    if rng is not None and abs(r_d - rng) > max(10.0, 0.6 * rng):
                        continue
                    cand = j
                    break                                   # dets sorted largest-first
            if cand is not None:
                used.add(cand)
                matched[g] = dets[cand]
                matched_j[g] = cand
                cands.setdefault(g, []).append((cand, 0.0, max(radius, 1.0)))

        # ---- JOINT SLOT ASSIGNMENT: reject label swaps -------------------------------
        # Greedy picked each slot's nearest prediction independently, which is exactly
        # what a swap defeats. Re-choose over the three slots together, scored on map
        # geometry, and take the result only if it beats the greedy one clearly.
        # All three slots, including the current one: Claire's second frame swaps the
        # CURRENT gate with k+1, so a lookahead-only repair cannot reach it. Safe to
        # widen because the pass now only runs on frames that are already infeasible.
        alt = self._joint_slots(slots, dets, cands, matched_j) \
            if (JOINT_SLOTS and len(slots) > 1) else None
        if alt is not None:
            self.slot_fixes += 1
            matched = {g: dets[j] for g, j in alt.items()}
            matched_j = dict(alt)
            used = set(alt.values())

        # ---- lookahead bootstrap: map DISTANCE consistency ---------------------------
        # A lookahead slot with no track and no frame-anchored map prediction can still
        # be identified from the map's pair DISTANCES (branch- and compass-free): a
        # clean inner detection whose PnP separation from the measured current gate
        # matches d_map(k, g) within tolerance, with a margin against the other slot.
        cur_tr = self.tracks.get(slots[0])
        if self.map.pos is not None and cur_tr is not None and cur_tr.pose_valid:
            for g in slots[1:]:
                if g in matched or self.tracks.get(g) is not None:
                    continue
                d_map = float(np.linalg.norm(self.map.pos[g] - self.map.pos[slots[0]]))
                others = [float(np.linalg.norm(self.map.pos[o] - self.map.pos[slots[0]]))
                          for o in slots[1:] if o != g]
                best = None
                for j, d in enumerate(dets):
                    if j in used or d['source'] != 'inner' or d['size_px'] < 18.0 \
                            or self._is_deco(d, False):
                        continue
                    bnd = self._bands.get(g)
                    if bnd is not None \
                            and not (bnd[0] <= float(d['range_m']) <= bnd[1]):
                        continue
                    sep = float(np.linalg.norm(np.asarray(d['pos_body'])
                                               - cur_tr.pos))
                    err = abs(sep - d_map)
                    if err > max(3.0, 0.30 * d_map):
                        continue
                    if others and min(abs(sep - o) for o in others) < err + 2.0:
                        continue                     # ambiguous with the other slot
                    if best is None or err < best[0]:
                        best = (err, j)
                if best is not None:
                    used.add(best[1])
                    matched[g] = dets[best[1]]

        # ---- IN-ORDER slot filling + band enforcement ---------------------------------
        self._reslot(slots, dets, matched, matched_j, used)

        # ---- which slots get a net crop ----------------------------------------------
        seeds = []
        for g in slots:
            if len(seeds) >= NET_MAX_CROPS:
                break
            d = matched.get(g)
            tr = self.tracks.get(g)
            if d is not None:
                q = np.asarray(d['quad'])
                clipped = (q[:, 0].min() < 1 or q[:, 1].min() < 1
                           or q[:, 0].max() > W - 1 or q[:, 1].max() > H - 1)
                # net only where the contour detector is weak (clipped / outer /
                # large / ribbon-suspect): a clean mid-size inner quad is already
                # 1.5-2 px and not worth 4-9 ms of CPU at 30 Hz
                ribbon_suspect = abs(d['range_m'] - d['range_size_m']) \
                    / max(d['range_m'], d['range_size_m'], 1e-6) > 0.15
                want = (clipped or d['source'] == 'outer'
                        or d['size_px'] >= NET_LARGE_PX
                        or (g == slots[0] and ribbon_suspect))
                # lookahead gates: net only where the crop is worth the ms -- a small
                # far detection's contour quad is already at the net's own accuracy
                if g != slots[0] and d['size_px'] < 30.0:
                    want = False
                if want:
                    x0, y0 = q.min(0)
                    x1, y1 = q.max(0)
                    box = (float(x0), float(y0), float(x1), float(y1))
                    if not _seed_clash(box, seeds):
                        seeds.append((g, box))
            elif tr is not None and tr.quad is not None \
                    and (t_s - tr.t_obs) < 0.7 and tr.size_px >= NET_MIN_SEED_PX:
                uv = project_body(tr.pos) if tr.pos is not None else None
                if uv is not None and -40 < uv[0] < W + 40 and -40 < uv[1] < H + 40:
                    s = tr.size_px
                    box = (uv[0] - s / 2, uv[1] - s / 2, uv[0] + s / 2, uv[1] + s / 2)
                    # a COASTED seed box overlapping a box already seeded for another
                    # slot is the second half of the duplicate-assignment bug: the net
                    # happily regresses the SAME physical gate out of both crops, and
                    # detection-level exclusivity (`used`) never sees it.
                    if not _seed_clash(box, seeds):
                        seeds.append((g, box))
        t0 = time.perf_counter()
        net_out = self._net_quads(img, seeds)
        if seeds:
            self.timing['net'].append(time.perf_counter() - t0)

        # ---- measure + update tracks ---------------------------------------------------
        t0 = time.perf_counter()
        for g in slots:
            self._update_track(g, matched.get(g), net_out.get(g), hsv, t_s,
                               is_current=(g == slots[0]))
        self.timing['pnp'].append(time.perf_counter() - t0)
        self._dedupe_slots(slots, t_s)

        # ---- map-range prior ratchet + course-progress anchor -------------------------
        # The current gate's range can only DECREASE between crossings, so each accepted
        # measurement pulls the plausible bound down (with slack: 1.3x, floored at +3 m
        # so a near gate can still be re-acquired after a coast gap). n_obs >= 2 keeps a
        # single spurious quad from strangling the bound.
        tr0 = self.tracks.get(slots[0]) if slots else None
        if tr0 is not None and tr0.pos is not None:
            if tr0.t_obs == t_s and tr0.n_obs >= 2:
                r = float(np.linalg.norm(tr0.pos))
                # +5 m floor: near-field clipped-gate ranges wobble by metres
                b = max(RANGE_PRIOR_K * r, r + 5.0)
                self.cur_rmax = b if self.cur_rmax is None else min(self.cur_rmax, b)
            # progress prior for the (undirected) ribbon: the last known direction of
            # the current gate IS the course direction, and it gyro-coasts with the
            # body, so it survives losing the gate. Compass-free by construction.
            self.prog_dir = tr0.pos / max(float(np.linalg.norm(tr0.pos)), 1e-6)

        # ---- map-frame anchoring from co-measured pairs -------------------------------
        fresh = [g for g in slots if g in self.tracks
                 and self.tracks[g].pose_valid and self.tracks[g].t_obs == t_s]
        R_lb = self.imu.level_rotation()
        if len(fresh) >= 2 and R_lb is not None and self.imu.psi_int is not None:
            for a in range(len(fresh)):
                for b in range(a + 1, len(fresh)):
                    i, j = fresh[a], fresh[b]
                    v = R_lb @ (self.tracks[j].pos - self.tracks[i].pos)
                    b_int = self.imu.psi_int + math.degrees(math.atan2(v[1], v[0]))
                    self.map.add_anchor(i, j, b_int)

    def _slot_band(self, cur, r0, g):
        """Two-sided plausible RANGE band (lo, hi) for lookahead slot g, or None.

        Given a measured current-gate range r0, the aircraft sits (approximately) r0 short
        of gate `cur` along the cur-1 -> cur map edge. That placement is EXACT at both ends
        of the leg -- at the crossing of cur-1 (r0 = the edge length) and at gate cur
        (r0 = 0) -- and interpolates the straight line between, so the residual error is
        the aircraft's lateral deviation from the edge, which scales with r0. Hence the
        r0-proportional tolerance.

        WHY THIS AND NOT THE TRIANGLE INEQUALITY: |d - r0| <= r <= d + r0 is exact but
        vacuous on a near-collinear course -- for gates 5/6/7 with r0 = 7.5 m it admits
        gate 7 anywhere from 23.1 m, which is precisely where gate 6 is. The band uses the
        map TURN ANGLE as well as its distances, and that is what separates them.

        Compass-free and anchor-free: map distances and one map edge direction used only
        RELATIVE TO ITSELF (both endpoints in map coordinates), so neither the mirror s nor
        psi_int enters."""
        if not SLOT_BANDS or r0 is None or self.map.pos is None or g == cur:
            return None
        if not (0 <= cur < self.map.n and 0 <= g < self.map.n):
            return None
        tol = SLOT_BAND_TOL_M + SLOT_BAND_TOL_K * r0
        u = None
        if cur - 1 >= 0:
            v = self.map.pos[cur] - self.map.pos[cur - 1]
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                u = v / nv
        if u is not None:
            p_ac = self.map.pos[cur] - r0 * u
            pred = float(np.linalg.norm(self.map.pos[g] - p_ac))
            return (max(0.0, pred - tol), pred + tol)
        d = self.map.dist(cur, g)                # start / reset: no incoming edge
        if d is None:
            return None
        return (max(0.0, d - r0 - SLOT_TRI_TOL), d + r0 + SLOT_TRI_TOL)

    def _reslot(self, slots, dets, matched, matched_j, used):
        """Enforce the slot invariant on the lookahead slots: fill IN ORDER, at map-
        consistent ranges, one object per slot.

        A lookahead occupant whose range falls outside its own band is in the wrong slot.
        If some NEARER, EMPTY lookahead slot band does contain that range, the occupant is
        moved there -- an empty intermediate slot must never let a nearer gate skip
        outward, and a range that matches the empty slot band is positive evidence that it
        IS that slot. Otherwise the occupant is dropped: a labelled wrong gate is worse
        than an empty slot, because attention and the policy both read the label.

        Mutates matched / matched_j / used in place and re-keys nothing -- a moved
        DETECTION is re-measured from scratch by _update_track under its new index, and the
        misplaced track is deleted rather than re-keyed, so no accumulated per-gate normal
        evidence follows a label across gates."""
        if not SLOT_BANDS or not self._bands or len(slots) < 2:
            return
        for g in slots[1:]:
            bnd = self._bands.get(g)
            if bnd is None:
                continue
            d = matched.get(g)
            tr = self.tracks.get(g)
            if d is not None:
                r = float(d['range_size_m'] if d['source'] == 'outer' else d['range_m'])
            elif tr is not None and tr.pos is not None:
                r = float(np.linalg.norm(tr.pos))
            else:
                continue
            if bnd[0] <= r <= bnd[1]:
                continue
            # wrong slot. Look for a nearer EMPTY lookahead slot whose band owns it.
            home = None
            for g2 in slots[1:]:
                if g2 >= g or g2 in matched or self.tracks.get(g2) is not None:
                    continue
                b2 = self._bands.get(g2)
                if b2 is not None and b2[0] <= r <= b2[1]:
                    home = g2
                    break
            if tr is not None:
                del self.tracks[g]
            if d is not None:
                del matched[g]
                j = matched_j.pop(g, None)
                if home is not None:
                    matched[home] = d
                    if j is not None:
                        matched_j[home] = j
                    self.slot_moves += 1
                else:
                    if j is not None:
                        used.discard(j)
                    self.slot_band_drops += 1
            else:
                self.slot_band_drops += 1

    def _dedupe_slots(self, slots, t_s):
        """No physical object may occupy two slots.

        Detection-level exclusivity is already enforced by `used`, but two slots can still
        converge on the same object without ever sharing a detection: the net regresses the
        same gate out of two overlapping seed crops, or one slot simply coasts onto the
        other. The map settles which slot is entitled to it -- the object cannot be a LATER
        gate than the nearest slot claiming it -- so the farther slot always loses."""
        if not SLOT_BANDS:
            return
        for a in range(len(slots)):
            for b in range(a + 1, len(slots)):
                ga, gb = slots[a], slots[b]
                ta, tb = self.tracks.get(ga), self.tracks.get(gb)
                if ta is None or tb is None or ta.pos is None or tb.pos is None:
                    continue
                ra = float(np.linalg.norm(ta.pos))
                rb = float(np.linalg.norm(tb.pos))
                tol = max(SLOT_DUP_TOL_M, SLOT_DUP_TOL_K * min(ra, rb))
                if float(np.linalg.norm(ta.pos - tb.pos)) > tol:
                    continue
                dm = self.map.dist(ga, gb)
                if dm is not None and dm <= tol:
                    continue      # the map says these two gates really are that close
                self.slot_dupes += 1
                del self.tracks[gb]

    def _slot_pair_ok(self, ga, gb, ra, rb):
        """Range-only triangle inequality between two slots, against the map distance.

        Exact geometry: two points at ranges ra, rb from the same origin and known to be
        d apart must satisfy |ra - rb| <= d <= ra + rb. Needs no compass, no frame anchor
        and no pose, and MapModel.dist is mirror-free, so nothing unresolved feeds it.

        NOTE ITS BLIND SPOT: both sides are symmetric in ra and rb, so this test cannot
        see a pure SWAP of two gates' labels. That is what _slot_order_ok is for.
        Returns True when the map cannot say."""
        d = self.map.dist(ga, gb)
        if d is None or ra is None or rb is None:
            return True
        return abs(ra - rb) <= d + SLOT_TRI_TOL and (ra + rb) >= d - SLOT_TRI_TOL

    @staticmethod
    def _slot_order_ok(ranked):
        """ranked: [(gate, range)] in RACE ORDER. A gate is flown before its successor,
        so it cannot measure materially farther than one."""
        for i in range(len(ranked) - 1):
            ra, rb = ranked[i][1], ranked[i + 1][1]
            if ra is not None and rb is not None and ra > rb + SLOT_ORDER_TOL:
                return False
        return True

    def _assign_ok(self, assign, rng_of):
        """Is a slot -> detection assignment geometrically possible?

        Only INNER, pose-bearing detections get a vote. An outer detection's range is the
        fronto-parallel apparent-size estimate, biased long and carried at range_sigma
        0.40 for that reason; convicting an assignment on it produced measurable false
        repairs (see this method's patch note). rng_of returns None for a detection that
        cannot testify, and every test below treats None as 'no opinion'."""
        gs = sorted(assign)
        ranked = [(g, rng_of(assign[g])) for g in gs]
        if not self._slot_order_ok(ranked):
            return False
        for g, r in ranked:                      # two-sided map band per lookahead slot
            bnd = self._bands.get(g)
            if bnd is not None and r is not None and not (bnd[0] <= r <= bnd[1]):
                return False
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                if not self._slot_pair_ok(gs[a], gs[b], ranked[a][1], ranked[b][1]):
                    return False
        return True

    def _joint_slots(self, slots, dets, cands, greedy):
        """REPAIR an infeasible slot assignment. Returns a better one, or None.

        Runs only when the greedy assignment is geometrically impossible, so it cannot
        churn a frame that was already right -- see this patch's header for the measured
        reason that matters. Cost among feasible repairs is pixel distance from each
        slot's own prediction (in units of its association radius, so a slot with a soft
        prediction is not punished for being soft) plus the 3-D separation error against
        the map where two detections both carry a PnP position.
        """
        import itertools
        if len(slots) < 2 or not cands:
            return None

        def rng_of(j):
            """Range for the FEASIBILITY verdict, or None if this detection cannot
            testify. Deliberately stricter than the association's own range use."""
            d = dets[j]
            return d['range_m'] if d['source'] == 'inner' else None

        def judgeable(assign):
            return sum(1 for j in assign.values() if rng_of(j) is not None) >= 2

        def pos_of(j):
            d = dets[j]
            p = d.get('pos_body')
            return np.asarray(p, float) if (p is not None and d['source'] == 'inner') \
                else None

        if len(greedy) < 2 or not judgeable(greedy) \
                or self._assign_ok(greedy, rng_of):
            return None      # nothing wrong, or nothing able to say that anything is

        opts = []
        for g in slots:
            seen, uniq = set(), []
            for j, dpx, rad in cands.get(g, []):
                if j in seen:
                    continue
                seen.add(j)
                uniq.append((g, j, dpx, rad))
            opts.append([(g, None, 0.0, 1.0)] + uniq)

        best = None
        for combo in itertools.product(*opts):
            picked = [(g, j, dpx, rad) for g, j, dpx, rad in combo if j is not None]
            js = [j for _, j, _, _ in picked]
            if len(set(js)) != len(js) or len(picked) < 2:
                continue
            assign = {g: j for g, j, _, _ in picked}
            if not judgeable(assign) or not self._assign_ok(assign, rng_of):
                continue
            cost = 0.60 * (len(slots) - len(picked))
            for g, j, dpx, rad in picked:
                cost += min(dpx / max(rad, 1.0), 2.0)
            for a in range(len(picked)):
                for b in range(a + 1, len(picked)):
                    dm = self.map.dist(picked[a][0], picked[b][0])
                    pa, pb = pos_of(picked[a][1]), pos_of(picked[b][1])
                    if dm is not None and pa is not None and pb is not None:
                        err = abs(float(np.linalg.norm(pa - pb)) - dm)
                        cost += min(max(err - SLOT_SEP_TOL, 0.0) / max(dm, 1.0), 2.0)
            if best is None or cost < best[0]:
                best = (cost, assign)
        if best is None or best[1] == dict(greedy):
            return None
        return best[1]

    def _slot_audit(self, slots, obs):
        """Per-frame slot-geometry audit -- the regression channel for label swaps.

        Two counts, because they answer different questions:
          * map geometry -- the triangle inequality against the map distance, over all
            valid slots and again over only those measured THIS frame. A heavily coasted
            range is the least trustworthy number in the system, so mixing it in measures
            coasting rather than labelling.
          * range ordering -- what a human reads off the HUD, and the only test that can
            see a pure two-gate SWAP (the triangle test is symmetric in the two ranges and
            is structurally blind to it -- see _slot_pair_ok).
        """
        rs, fresh = {}, {}
        for k, g in enumerate(slots):
            if k < len(obs.gates) and obs.gates[k].valid:
                rs[g] = float(obs.gates[k].range_m)
                if obs.gates[k].staleness_s == 0.0 and obs.gates[k].pose_valid:
                    fresh[g] = rs[g]
        self._audit_pairs(fresh, self.slot_viol_fresh)
        if len(rs) < 2:
            return
        self._audit_pairs(rs, self.slot_viol)
        seq = [(g, rs[g]) for g in sorted(rs)]
        self.slot_order[0] += 1
        if not self._slot_order_ok(seq):
            self.slot_order[1] += 1

    def _audit_pairs(self, rs, ctr):
        """Count frames whose slot ranges violate the map triangle inequality."""
        if len(rs) < 2:
            return
        gs = sorted(rs)
        checkable = bad = 0
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                if self.map.dist(gs[a], gs[b]) is None:
                    continue
                checkable += 1
                if not self._slot_pair_ok(gs[a], gs[b], rs[gs[a]], rs[gs[b]]):
                    bad += 1
        if checkable:
            ctr[0] += 1
            if bad:
                ctr[1] += 1

    def _is_deco(self, d, current):
        """Interior-colour decoration test. The ACTIVE gate renders with a bright
        interior fill (NOTES.md lit-gate cue), so the current slot is exempted from the
        brightness half and relies on the confidence head + PnP residual instead.
        Lookahead slots get is_decoration()'s dark-gap escape hatch (interiorgate.py
        2026-08-02): a real aperture against a lit ceiling still has dark patches that
        a solid decoration fill cannot produce."""
        if current:
            return d.get('in_white', 0.0) > 0.60
        return is_decoration(d.get('in_white', 0.0), d.get('in_v', 0.0),
                             d.get('in_dark', 1.0))

    def _reset_bound(self, a):
        """Current-gate range bound with no crossed edge to anchor it (start / reset):
        the next consecutive edge, plain (no K: this is already the loose reading)."""
        d = self.map.dist(a, a + 1)
        if d is None:
            d = self.map.dist(a - 1, a)
        return d

    def _map_predict(self, g):
        """Predicted body-frame position of gate g, chained from the nearest tracked
        gate through map vectors. None when the frame anchor is unresolved."""
        R_lb = self.imu.level_rotation()
        if R_lb is None:
            return None
        for ref in sorted(self.tracks, key=lambda x: abs(x - g)):
            tr = self.tracks[ref]
            if tr.pos is None or ref == g:
                continue
            v_lev = self.map.predict_vector_lev(ref, g, self.imu.psi_int)
            if v_lev is None:
                return None
            return tr.pos + R_lb.T @ v_lev
        return None

    @staticmethod
    def _solid_fill_frac(hsv, quad):
        """Fraction of the quad interior (shrunk to 70%, interior_colour's convention)
        that is saturated orange. Real gate outer boundary: ring + dark aperture,
        <= ~0.4 even point-blank. Solid decoration (start pedestals): ~1.0."""
        if hsv is None:
            return 0.0
        q = quad.reshape(-1, 2).astype(np.float32)
        ctr = q.mean(0)
        poly = (ctr + (q - ctr) * 0.7).astype(np.int32)
        msk = np.zeros(hsv.shape[:2], np.uint8)
        cv2.fillPoly(msk, [poly], 1)
        px = hsv[msk.astype(bool)]
        if len(px) < 8:
            return 0.0
        orange = (((px[:, 0] <= 12) | (px[:, 0] >= 170))
                  & (px[:, 1] >= 80) & (px[:, 2] >= 100))
        return float(np.mean(orange))

    @staticmethod
    def _emissive_frac(hsv, quad):
        """Fraction of a ring just OUTSIDE the quad (the frame material, x1.15 about
        the centre) that is emissive orange. See EMISSIVE_* provenance."""
        if hsv is None:
            return 1.0
        c = quad.mean(axis=0)
        ring = c + (quad - c) * 1.15
        pts = []
        for k in range(4):
            a, b = ring[k], ring[(k + 1) % 4]
            for f in (0.0, 0.25, 0.5, 0.75):
                pts.append(a + (b - a) * f)
        h, w = hsv.shape[:2]
        good = tot = 0
        for p in pts:
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h:
                tot += 1
                H, S, V = hsv[y, x]
                if ((H <= 12 or H >= 170) and S >= EMISSIVE_S_MIN
                        and V >= EMISSIVE_V_MIN):
                    good += 1
        return (good / tot) if tot else 0.0

    def _update_track(self, g, det, net, hsv, t_s, is_current=False):
        tr = self.tracks.get(g)
        quad = None
        conf = 0.5
        src = None
        if net is not None:
            nquad, p, pred_px = net
            size_n = float(max(np.linalg.norm(nquad[(k + 1) % 4] - nquad[k])
                               for k in range(4)))
            if (p >= CONF_P and pred_px <= max(NET_PRED_ERR_ABS, 0.08 * size_n)
                    and self._emissive_frac(hsv, nquad) >= EMISSIVE_FRAC):
                quad, conf, src = nquad, p, 'net'
        if quad is None and det is not None and det['source'] == 'inner':
            quad, conf, src = np.asarray(det['quad'], np.float64), 0.6, 'inner'
        if quad is None and det is not None and det['source'] == 'outer':
            # keep the outer detection only as a centroid+size observation -- but NOT
            # when the region is a SOLID orange fill. An outer quad on a real gate is a
            # ring around a mostly-dark aperture; the start pedestals are solid emissive
            # orange, and one of them captured the current-gate track and flew the servo
            # policy into itself (20260802-204323, t=12.3-14.4: outer sizes walking
            # 46->320 px were the pedestal, range 10.5->1.07 m all false). The quadless
            # outer path skipped every interior referee; this is that referee.
            if self._solid_fill_frac(hsv, np.asarray(det['quad'], np.float64)) > 0.60:
                return
            quad, conf, src = None, 0.4, 'outer'

        if quad is None and src != 'outer':
            return                                           # no observation: coast

        if tr is None:
            tr = self.tracks[g] = Track(g)

        if quad is not None:
            size_px = float(max(np.linalg.norm(quad[(k + 1) % 4] - quad[k])
                                for k in range(4)))
            # rotation-coasting verification: when a track is re-acquired after a coast
            # gap, the coasted (gyro-rotated) prediction should land near the new
            # measurement. Recorded, and reported by --replay.
            gap = t_s - tr.t_obs
            if tr.n_obs > 0 and tr.pos is not None and 0.15 < gap < COAST_MAX_S:
                ctr_new = quad.mean(0)
                uv_pred = project_body(tr.pos)
                miss_px = float(np.hypot(*(np.array(uv_pred) - ctr_new))) \
                    if uv_pred is not None else float('nan')
                rng_new = 480.0 / max(size_px, 1e-6)
                p_new = pixel_ray_body(*ctr_new) * rng_new
                self.reacq.append((gap, float(np.linalg.norm(tr.pos - p_new)), miss_px,
                                   float(self.last_speed), float(tr.coast_dist)))
            inw, inv, ind = interior_colour(hsv, quad)
            if src == 'net':
                # third rejection layer. The CURRENT gate renders with a bright
                # interior fill (lit-gate cue), so it only gets the lax test; a
                # LOOKAHEAD net quad gets the same strict decoration test as contour
                # candidates in _is_deco -- this path had silently skipped it, which
                # is the hole a confident wordmark quad would need. Both get the
                # dark-gap escape hatch (interiorgate.py 2026-08-02): a real aperture
                # against a lit ceiling still has dark patches; decoration does not.
                if is_current:
                    if inw > 0.60 and ind < IN_DARK_FRAC:
                        return
                elif is_decoration(inw, inv, ind):
                    return
            _, _, att_conf = self.imu.roll_pitch()
            # RE-ASSOCIATION RESET (anti-lock): past the validity horizon the slot may
            # have been re-acquired on a different physical gate, and the map/eviction
            # referee can hand the same index to a different object. Accumulated twin
            # evidence must not survive that -- see normalfuse's "NO LOCKING" note.
            if tr.n_obs > 0 and (t_s - tr.t_obs) > COAST_MAX_S:
                tr.nrm.reset()
            pose = pnp_pose(quad, tr.normal if tr.n_obs else None,
                            gravity_body=self.imu.g, attitude_conf=att_conf, gate_idx=g)
            n_in = n_corners_in_frame(quad)
            in_frame = n_in == 4
            if pose is not None:
                r_size = 480.0 / max(size_px, 1e-6)
                # SIZE-RELATIVE gate (residgate.py 2026-08-02): a fixed absolute px
                # threshold rejects excellent poses on big gates (residual scales with
                # apparent size) and admits mediocre ones on tiny gates. Floor differs
                # by source: net quads' good-pose residuals cluster far tighter
                # (pnpsnap) than contour quads', whose corner noise alone sits at
                # 0.16-0.49 px rms even when the pose is right.
                resid_floor = PNP_RESID_ABS_NET if src == 'net' else PNP_RESID_ABS_CTR
                # OBSERVED-EXTENT gate first: a quad with fewer than two corners inside
                # the image is mostly extrapolation, and its residual is measured against
                # invented corners -- so the residual cannot referee it. This is the
                # close-range half of Bug 1 (cornergate.py).
                resid_ok = (n_in >= MIN_CORNERS_IN_FRAME
                            and pose['rms'] <= max(resid_floor,
                                                   PNP_RESID_REL * size_px))
                agree_ok = (abs(pose['range'] - r_size)
                            / max(pose['range'], r_size, 1e-6) <= RANGE_DISAGREE) \
                    if in_frame else True   # size range is meaningless on a clipped quad
                if resid_ok and agree_ok:
                    tr.pos = pose['pos']
                    tr.normal = pose['normal']
                    self._fuse_normal(tr, pose, size_px, att_conf, t_s)
                    tr.pose_valid = True
                    tr.range_sigma = max(0.03 * pose['range'], pose['rms'] / max(size_px, 1)
                                         * pose['range'])
                    tr.quad, tr.size_px = quad, size_px
                    tr.conf, tr.t_obs = conf, t_s
                    tr.n_obs += 1
                    tr.coast_dist = 0.0      # a fix lands: coast budget spent
                    return
            # centroid + size fallback (fronto-parallel assumption; oblique bias LONG)
            ctr = quad.mean(0)
            rng = 480.0 / max(size_px, 1e-6)
            tr.pos = pixel_ray_body(*ctr) * rng
            # The normal is a WORLD-fixed direction held in a gyro-coasted common frame,
            # so an already-decided accumulator still knows it even on a frame whose pose
            # fit failed. Position quality and normal availability are separate facts --
            # the contract says so explicitly ("normal_valid False does NOT mean the
            # position is unusable", and the converse holds here).
            n_acc = tr.nrm.normal_body(tr.pos / max(float(np.linalg.norm(tr.pos)), 1e-9)) \
                if tr.nrm.decided else None
            if n_acc is not None and size_px >= NORMAL_MIN_PX:
                tr.normal, tr.normal_valid, tr.normal_src = n_acc, True, 'accum-coast'
            else:
                tr.normal_valid = False
                tr.normal_src = 'none'
            tr.pose_valid = False
            tr.range_sigma = 0.30 * rng          # honest: oblique gates read long
            tr.quad, tr.size_px = quad, size_px
            tr.conf, tr.t_obs = min(conf, 0.5), t_s
            tr.n_obs += 1
            tr.coast_dist = 0.0      # a fix lands: coast budget spent
            return

        # outer-fallback centroid observation
        ctr = np.asarray(det['centre'], float)
        rng = det['range_size_m']
        tr.pos = pixel_ray_body(*ctr) * rng
        tr.normal_valid = False
        tr.pose_valid = False
        tr.range_sigma = 0.40 * rng              # outer edge: bloom + 1.8x pixel leverage
        tr.quad = None
        tr.size_px = det['size_px']
        tr.conf, tr.t_obs = 0.4, t_s
        tr.n_obs += 1
        tr.coast_dist = 0.0      # a fix lands: coast budget spent

    # ---- normal fusion ----------------------------------------------------------------
    def _map_normal_az(self, g):
        """Expected PLANE-NORMAL azimuth of gate g in the LEVELLED BODY frame, degrees,
        or None when the anchor/compass cannot support it.

        Chain: bisector azimuth (map frame) -> internal levelled bearing via the online
        mirror/offset anchor (s, c) -> minus psi_int to get a body-relative azimuth. Same
        composition MapModel.predict_vector_lev uses for positions, so if the SEARCH
        lookahead is pointing at the right place this azimuth is in the same frame.
        """
        if not USE_MAP_NORMAL_PRIOR or self.map.s is None or self.imu.psi_int is None:
            return None
        if len(self.map.obs) < MAP_PRIOR_MIN_ANCHORS:
            return None
        y = self.course.bisector.get(g)
        if y is None:
            return None
        return self.map.s * y + self.map.c - self.imu.psi_int

    def _fuse_normal(self, tr, pose, size_px, att_conf, t_s):
        """Hand this frame's IPPE twin PAIR to the track's accumulator and read back the
        verdict. The per-frame branch pick is no longer a decision -- see normalfuse.py.
        """
        g = tr.gate
        los = pose['pos'] / max(float(np.linalg.norm(pose['pos'])), 1e-9)
        sols = pose.get('sols') or [pose]
        cands = [s['normal'] for s in sols]
        rms = [s['rms'] for s in sols]
        up = None
        if self.imu.g is not None:
            up = -self.imu.g / max(float(np.linalg.norm(self.imu.g)), 1e-9)
        R_lb = self.imu.level_rotation()
        map_az = self._map_normal_az(g)
        # observation weight: a small quad ranks the pair on noise (the sub-20 px
        # normal_valid-on-an-8 px-gate failure), so it contributes proportionally less
        quality = float(np.clip((size_px - NORMAL_MIN_PX) / 40.0, 0.15, 1.0))
        win_before = tr.nrm.win
        tr.nrm.update(cands, rms, los, quality, up, att_conf,
                      g in VERTICALITY_EXEMPT_GATES, map_az, R_lb)
        # anti-lock: the accumulated leader changed hands while the track was already
        # confident. Decay rather than let the old evidence out-vote the new geometry.
        if tr.nrm.win != win_before and tr.nrm.n_samp > 2 * NF.MIN_SAMPLES:
            tr.nrm.contradiction_reset()

        n_acc = tr.nrm.normal_body(los) if tr.nrm.decided else None
        if n_acc is not None and size_px >= NORMAL_MIN_PX:
            tr.normal = n_acc
            tr.normal_valid = True
            tr.normal_src = 'accum-' + tr.nrm.decided_by
        elif (NORMAL_TRUST_UNAMBIG and not pose.get('ambiguous', True)
              and size_px >= NORMAL_MIN_PX):
            # the reprojection itself separated the branches by >TILT_AMBIG_RATIO: that
            # is measurement, not a guess, and the contract only forbids guessing
            tr.normal = pose['normal']
            tr.normal_valid = True
            tr.normal_src = 'unambiguous'
        else:
            tr.normal_valid = False
            tr.normal_src = 'refused'

        if NORMAL_DEBUG:
            c0 = cands[0]
            c1 = cands[1] if len(cands) > 1 else cands[0]
            na = tr.nrm.normal_body(los)
            nl = tr.nrm.loser_body(los)
            map_err = np.nan
            if map_az is not None and R_lb is not None and na is not None:
                v = R_lb @ na
                map_err = NF.wrap90(math.degrees(math.atan2(float(v[1]), float(v[0])))
                                    - map_az)
            self.normal_log.append([
                t_s, g, size_px, len(sols),
                NF._ang_deg(NF._unit(c0), NF._unit(c1)),
                tr.nrm.baseline_deg, tr.nrm.n_samp, tr.nrm.llr, tr.nrm.llr_prior,
                float(tr.nrm.win), float(tr.nrm.decided),
                {'refused': 0.0, 'unambiguous': 1.0}.get(tr.normal_src, 2.0),
                float(not pose.get('ambiguous', True)),
                *c0, *c1, *los,
                *(na if na is not None else [np.nan] * 3),
                *(nl if nl is not None else [np.nan] * 3),
                map_az if map_az is not None else np.nan, map_err,
                float(tr.nrm.n_reset)])

    # ---- observation assembly --------------------------------------------------------
    def _gateobs(self, g, t_s):
        go = I.GateObs(index=g)
        tr = self.tracks.get(g)
        if tr is None or tr.pos is None:
            return go
        stale = max(0.0, t_s - tr.t_obs)
        if stale > COAST_MAX_S:
            return go
        go.valid = True
        go.pos_body = tr.pos.copy()
        go.staleness_s = stale
        go.confidence = tr.conf * max(0.2, 1.0 - stale / COAST_MAX_S)
        go.pose_valid = tr.pose_valid and stale == 0.0
        # Coasting is rotation AND translation (COAST_TRANSLATE). The bearing was
        # already right under rotation-only coasting; what was missing was RANGE, which
        # cannot update without a speed estimate. With one, the residual uncertainty is
        # no longer "the whole distance flown" but the error in the distance flown --
        # dominated by the un-modelled body-z component, so it is charged as a fraction
        # of the translated distance rather than a flat rate per second.
        if stale == 0.0:
            go.range_sigma_m = tr.range_sigma
        elif COAST_TRANSLATE:
            go.range_sigma_m = tr.range_sigma + COAST_SIGMA_K * tr.coast_dist \
                + 1.0 * stale
        else:
            go.range_sigma_m = tr.range_sigma + 10.0 * stale
        go.size_px = tr.size_px
        if tr.normal is not None and tr.normal_valid:
            go.normal_body = tr.normal.copy()
            go.normal_valid = True
        return go

    def _ribbon(self, hsv, t_s):
        rb = I.RibbonObs()
        cyan = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 100) &
                (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 110))
        frac = float(cyan.mean())
        rb.pixel_fraction = frac
        if frac < CYAN_MIN_FRAC:
            if self.ribbon_last is not None and self.ribbon_stale_t is not None:
                stale = t_s - self.ribbon_stale_t
                if stale < COAST_MAX_S:
                    rb.valid = True
                    rb.bearings_rad, rb.elevs_rad = self.ribbon_last
                    rb.staleness_s = stale
            return rb
        ys, xs = np.nonzero(cyan)
        # bottom of frame = near (floor ribbon); bin by row into N_RIBBON quantiles
        order = np.argsort(-ys)                     # near -> far
        bins = np.array_split(order, I.N_RIBBON)
        bearings = np.zeros(I.N_RIBBON)
        elevs = np.zeros(I.N_RIBBON)
        for k, b in enumerate(bins):
            if len(b) == 0:
                continue
            u, v = float(xs[b].mean()), float(ys[b].mean())
            r = pixel_ray_body(u, v)
            bearings[k] = math.atan2(r[1], r[0])
            elevs[k] = math.atan2(-r[2], math.hypot(r[0], r[1]))
        rb.valid = True
        rb.bearings_rad, rb.elevs_rad = bearings, elevs
        rb.staleness_s = 0.0
        self.ribbon_last = (bearings, elevs)
        self.ribbon_stale_t = t_s
        return rb

    def _selfobs(self):
        so = I.SelfObs()
        so.gyro = self.imu.gyro.copy()
        so.accel = self.imu.accel.copy()
        so.roll_rad, so.pitch_rad, so.attitude_conf = self.imu.roll_pitch()
        ax, ay = float(so.accel[0]), float(so.accel[1])
        h = math.hypot(ax, ay)
        # vel_bearing is the SAME phantom as speed_est and needs the same veto: parked on
        # the VQ2 pad reads |a_xy| = 3.0 m/s^2, eight times HOVER_ACC_MIN, pointing along
        # the pad's downhill direction. Left ungated it would publish a confident heading
        # of travel for an aircraft that is not travelling.
        if h > HOVER_ACC_MIN and not (SPEED_VETO and self.imgmot.static):
            # horizontal body specific force is drag, antiparallel to airspeed
            so.vel_bearing_rad = math.atan2(-ay, -ax)
            so.vel_valid = True
        so.speed_est_mps, so.speed_conf = self.last_speed, self.last_speed_conf
        return so

    def _speed(self):
        """(speed_mps, conf, v_body_inplane) from the drag inversion.

        speed_est_mps is the IN-PLANE (body xy) speed, which is the magnitude that pairs
        with vel_bearing_rad's direction -- together they are the body-horizontal velocity
        vector. It is NOT the total speed: the body-z component sits under the thrust axis
        and is not observable, and the barometer that would supply it reads nan in this
        sim. Fitting it anyway (via a world-horizontal-velocity assumption) was tried and
        REFUSED -- it validated at p90 15 m/s out of sample against VQ1 truth, because the
        recorded flight-path angle runs to a 26-34 deg median. See NOTES.md 2026-08-02.

        THE INVERSION HAS NO ZERO, so two vetoes sit on top of it, in this order:

        1. THE CAMERA. sqrt(|a_xy|/k) cannot distinguish a stationary airframe on a tilted
           surface from flight, because no function of the IMU can -- both are one constant
           gravity vector at the trim tilt. When the image has not changed for
           IMG_STATIC_MIN_S the speed is REFUSED outright: 0.0 at confidence 0.0, with
           v_body None so no coasted track is translated. Measured cost of not having it:
           8.06-8.10 m/s at conf 1.00 across 42 s of 20260802-163548 in which the aircraft
           did not move at all.
        2. THE TRANSIENT FACTOR, a precision statement bounded at TRANSIENT_FLOOR. It
           multiplies the PUBLISHED confidence only; the coast gate reads
           last_speed_conf_noise. The mechanism originally proposed for it -- linear
           acceleration adding to the horizontal specific force -- was measured against VQ1
           truth and FALSIFIED (regression slope 0.007-0.041 where a full leak is 1.000);
           what remains is noise amplification. Constants block has the numbers.
        """
        if SPEED_VETO and self.imgmot.static:
            self.last_speed_conf_noise = 0.0
            return 0.0, 0.0, None
        s, conf, v = DRAG.inplane(self.imu.accel)
        self.last_speed_conf_noise = conf
        if v is None:
            return s, conf, None
        if SPEED_VETO:
            conf *= transient_factor(self.imu.jerk, float(np.linalg.norm(self.imu.gyro)))
        return s, conf, np.array([v[0], v[1], 0.0])

    def _attention(self, obs, slots, t_s):
        at = I.Attention()
        cur = obs.gates[0] if slots else None
        nxt = obs.gates[1] if len(slots) > 1 else None
        if cur is not None and cur.valid and cur.range_m > R_COMMIT:
            at.kind = I.Attn.GATE_CURRENT
            at.target_gate_index = slots[0]
            at.target_dir_body = cur.pos_body / max(cur.range_m, 1e-6)
            return at
        if nxt is not None and nxt.valid:
            at.kind = I.Attn.GATE_NEXT
            at.target_gate_index = slots[1]
            at.target_dir_body = nxt.pos_body / max(nxt.range_m, 1e-6)
            return at
        if len(slots) > 1:
            p = self._map_predict(slots[1])
            if p is not None:
                at.kind = I.Attn.GATE_NEXT
                at.target_gate_index = slots[1]
                at.target_dir_body = p / max(float(np.linalg.norm(p)), 1e-6)
                return at
        if cur is not None and cur.valid:                    # inside R_COMMIT, nothing next
            at.kind = I.Attn.GATE_CURRENT
            at.target_gate_index = slots[0]
            at.target_dir_body = cur.pos_body / max(cur.range_m, 1e-6)
            return at
        if obs.ribbon.valid and obs.ribbon.staleness_s == 0.0:
            # The ribbon is an UNDIRECTED line: a centreline sample cannot tell
            # forward-along-course from backward (or a switchback segment re-entering
            # the frame). Disambiguate with the course-progress prior: prog_dir (last
            # known current-gate direction, gyro-coasted, compass-free) or, failing
            # that, the map prediction. Take the farthest sample within RIBBON_PRIOR_DEG
            # of the prior; if every sample fights the prior, REFUSE ribbon attention
            # and fall through to SEARCH -- pointing backward is worse than sweeping.
            prior = self.prog_dir
            if prior is None and slots:
                p = self._map_predict(slots[0])
                if p is not None:
                    prior = p / max(float(np.linalg.norm(p)), 1e-6)
            pick = I.N_RIBBON - 1
            if prior is not None:
                bp = math.atan2(float(prior[1]), float(prior[0]))
                pick = None
                for k in range(I.N_RIBBON - 1, -1, -1):          # far -> near
                    db = (float(obs.ribbon.bearings_rad[k]) - bp + math.pi) \
                        % (2 * math.pi) - math.pi
                    if abs(db) <= math.radians(RIBBON_PRIOR_DEG):
                        pick = k
                        break
            if pick is not None:
                at.kind = I.Attn.RIBBON
                b = float(obs.ribbon.bearings_rad[pick])
                e = float(obs.ribbon.elevs_rad[pick])
                at.target_dir_body = np.array([math.cos(e) * math.cos(b),
                                               math.cos(e) * math.sin(b), -math.sin(e)])
                return at
        # SEARCH: informed by the map if a prediction exists, else a level sweep
        p = self._map_predict(slots[0]) if slots else None
        if p is not None:
            at.kind = I.Attn.SEARCH
            at.target_gate_index = slots[0]
            at.target_dir_body = p / max(float(np.linalg.norm(p)), 1e-6)
            return at
        self.search_phase += math.radians(SEARCH_SWEEP_DPS) / 30.0
        at.kind = I.Attn.SEARCH
        at.target_dir_body = np.array([math.cos(self.search_phase),
                                       math.sin(self.search_phase), 0.0])
        return at

    def auto_yaw_rate(self, obs):
        """AUTO_ATTENTION servo: azimuth -> zero, canonical signs (positive = nose
        right; a target right of the nose has bearing > 0 and needs positive yaw)."""
        d = obs.attention.target_dir_body
        az = math.atan2(float(d[1]), float(d[0]))
        return float(np.clip(K_YAW * az, -2.5, 2.5))

    def timing_report(self):
        out = {}
        for k, v in self.timing.items():
            if v:
                a = np.array(v) * 1000.0
                out[k] = (float(np.median(a)), float(np.percentile(a, 90)), len(a))
        return out


# =========================================================================================
# Self test


def selftest():
    # 1. camera->body sign check: image-centre target at range R
    p = R_BC @ np.array([0.0, 0.0, 10.0])
    bearing = math.degrees(math.atan2(p[1], p[0]))
    elev = math.degrees(math.atan2(-p[2], math.hypot(p[0], p[1])))
    print(f'image-centre target: bearing {bearing:+.3f} deg, elevation {elev:+.3f} deg '
          f'(want 0, +20)')
    assert abs(bearing) < 1e-9 and abs(elev - 20.0) < 1e-6

    # 2. synthetic gate: fronto-parallel square 10 m down the BORESIGHT
    obj_cam = np.array([[-0.75, 0.75, 10.0], [0.75, 0.75, 10.0],
                        [0.75, -0.75, 10.0], [-0.75, -0.75, 10.0]])
    uv = np.stack([L.CX + L.FX * obj_cam[:, 0] / obj_cam[:, 2],
                   L.CY + L.FY * obj_cam[:, 1] / obj_cam[:, 2]], 1)
    pose = pnp_pose(D.order_quad(uv))
    r = float(np.linalg.norm(pose['pos']))
    el = math.degrees(math.atan2(-pose['pos'][2], math.hypot(*pose['pos'][:2])))
    print(f'synthetic gate at 10 m boresight: PnP range {r:.3f} m, elev {el:+.2f} deg, '
          f'rms {pose["rms"]:.4f} px (mod square symmetry)')
    assert abs(r - 10.0) < 0.05 and abs(el - 20.0) < 0.2
    assert pose['rms'] < 0.01, 'exact synthetic quad must have ~zero symmetric residual'

    # 3. to_vector round trip + masking
    obs = I.Observation()
    obs.gates[0].valid = True
    obs.gates[0].pos_body = np.array([5.0, 1.0, -0.5])
    obs.gates[0].confidence = 0.9
    v = obs.to_vector()
    assert v.size == I.OBS_DIM
    obs.gates[0].valid = False
    v2 = obs.to_vector()
    blk = v2[0:11]
    assert np.all(blk[:10] == 0.0) and blk[10] == 5.0, blk    # zeros + staleness cap
    print(f'to_vector: {v.size} dims; invalidated gate block zeroed correctly')
    print('SELFTEST PASS')


# =========================================================================================
# Replay driver


def load_csv(path):
    import csv
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


MIN_SEGMENT_S = 5.0       # a post-reset segment shorter than this is not a run


def find_last_reset(race, frames):
    """(t_wall_ns, active_index) of the last usable reset, or None.

    A reset is active_gate_index DROPPING. Only resets leaving >= MIN_SEGMENT_S of frames
    behind them count, so a session that ends on a reset does not select an empty tail."""
    if not race or not frames:
        return None
    rt = [float(r['t_wall_ns']) for r in race]
    ai = [int(r['active_gate_index']) for r in race]
    t_end = float(frames[-1]['t_recv_wall_ns'])
    best = None
    for i in range(1, len(ai)):
        if ai[i] < ai[i - 1] and (t_end - rt[i]) / 1e9 >= MIN_SEGMENT_S:
            best = (rt[i], ai[i - 1], ai[i])
    return best


def replay(session, limit=None, video=False, paircheck=False, no_net=False,
           stride=1, outdir=None, from_last_reset=False):
    if not os.path.isdir(session):
        session = os.path.join(HERE, 'sessions', session)
    name = os.path.basename(os.path.normpath(session))
    outdir = outdir or os.path.join(HERE, 'perception', 'producer_runs', name)
    os.makedirs(outdir, exist_ok=True)

    frames = [r for r in load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    imu = load_csv(os.path.join(session, 'imu.csv'))
    race = load_csv(os.path.join(session, 'race.csv'))
    coll = load_csv(os.path.join(session, 'collisions.csv'))
    if from_last_reset:
        r = find_last_reset(race, frames)
        if r is None:
            print('  --from-last-reset: no reset found; using the whole session')
        else:
            t_cut, a_from, a_to = r
            n0 = len(frames)
            keep = [f for f in frames if float(f['t_recv_wall_ns']) >= t_cut]
            if keep:
                t_first = float(frames[0]['t_recv_wall_ns'])
                print('  --from-last-reset: cut at t=%.2f s (gate %d -> %d), '
                      'frame %s, dropping %d of %d frames; %.1f s remain'
                      % ((t_cut - t_first) / 1e9, a_from, a_to, keep[0]['file'],
                         n0 - len(keep), n0,
                         (float(keep[-1]['t_recv_wall_ns'])
                          - float(keep[0]['t_recv_wall_ns'])) / 1e9))
                frames = keep
    frames = frames[::stride]
    if limit:
        frames = frames[:limit]
    imu_t = np.array([float(r['t_wall_ns']) for r in imu])
    race_t = np.array([float(r['t_wall_ns']) for r in race]) if race else np.array([0.0])
    coll_t = np.array([float(r['t_wall_ns']) for r in coll]) if coll else None

    prod = Producer(use_net=not no_net)
    t0_ns = float(frames[0]['t_recv_wall_ns'])
    imu_i = 0
    vecs, diags = [], []
    vw = None
    if video:
        vw = cv2.VideoWriter(os.path.join(outdir, 'replay.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 30 // stride, (W, H))
    episodes = 0
    last_coll_ns = -1e18
    pair_rows = []
    strip = {}          # regime -> list of annotated tiles (validation d)

    for n, fr in enumerate(frames):
        t_ns = float(fr['t_recv_wall_ns'])
        t_s = (t_ns - t0_ns) / 1e9
        j = int(np.searchsorted(imu_t, t_ns))
        rows = imu[imu_i:j]
        imu_i = j
        ri = max(0, int(np.searchsorted(race_t, t_ns)) - 1)
        active = int(race[ri]['active_gate_index']) if race else 0
        race_time = 0.0
        if race:
            rs = float(race[ri]['race_start_boot_time_ms'])
            sb = float(race[ri]['sim_boot_time_ms'])
            race_time = max(0.0, (sb - rs) / 1000.0) if rs >= 0 else 0.0
        t_since_coll, episodes_now = 1e3, episodes
        if coll_t is not None and len(coll_t):
            k = int(np.searchsorted(coll_t, t_ns))
            if k > 0:
                if coll_t[k - 1] - last_coll_ns > 0.5e9:
                    episodes += 1
                last_coll_ns = coll_t[k - 1]
                t_since_coll = (t_ns - coll_t[k - 1]) / 1e9
        img = cv2.imread(os.path.join(session, 'frames', fr['file']))
        obs = prod.step(img, rows, {
            'active_gate_index': active, 'race_time_s': race_time, 'armed': True,
            'n_gates_total': 17, 't_since_collision_s': t_since_coll,
            'collision_episodes': episodes_now}, t_s)
        vecs.append(obs.to_vector())
        g0 = obs.gates[0]
        tr0 = prod.tracks.get(g0.index)
        acc = tr0.nrm if tr0 is not None else None
        diags.append((t_s, active, int(g0.valid), int(g0.pose_valid),
                      g0.range_m if g0.valid else np.nan, g0.staleness_s,
                      math.degrees(g0.bearing_rad) if g0.valid else np.nan,
                      int(obs.attention.kind), obs.own.attitude_conf,
                      float(np.linalg.norm(obs.own.gyro)),
                      prod.cur_bound0 if prod.cur_bound0 is not None else np.nan,
                      # ---- normalfuse diagnostics (columns 11-15) ----
                      acc.llr if acc is not None else np.nan,
                      acc.baseline_deg if acc is not None else np.nan,
                      acc.n_samp if acc is not None else np.nan,
                      acc.llr_prior if acc is not None else np.nan,
                      {'none': 0, 'refused': 0, 'unambiguous': 1, 'accum-consistency': 2,
                       'accum-consistency+prior': 3, 'accum-prior': 4,
                       'accum-coast': 5}.get(
                          tr0.normal_src if tr0 is not None else 'none', 0),
                      math.degrees(g0.elev_rad) if g0.valid else np.nan,
                      obs.own.speed_est_mps, obs.own.speed_conf,
                      # ---- speed-validity channel (columns 19-21) ----
                      prod.imgmot.m, float(prod.imgmot.static),
                      prod.last_speed_conf_noise))
        if paircheck and img is not None:
            # Track-independent static-pair geometry: the two nearest CLEAN inner
            # detections through the SAME PnP path. Two static gates must keep a
            # constant separation from every viewpoint; drift = frame/sign bug.
            row = measure_pair(img)
            if row is not None:
                pair_rows.append((t_s,) + row)
        if img is not None:
            regime = _regime(obs)
            keep = strip.setdefault(regime, [])
            if len(keep) < 2 and (not keep or t_s - keep[-1][0] > 8.0):
                keep.append((t_s, annotate(img.copy(), obs, prod), regime))
        if vw is not None and img is not None:
            vw.write(annotate(img.copy(), obs, prod))
        if n % 300 == 0:
            print(f'  {n}/{len(frames)}  t={t_s:.1f}s gate={active} '
                  f'valid={g0.valid}', flush=True)
    if vw is not None:
        vw.release()

    tiles = [t for v in strip.values() for t in v][:12]
    if tiles:
        rows_n = (len(tiles) + 2) // 3
        sheet = np.zeros((rows_n * H, 3 * W, 3), np.uint8)
        for i, (ts, im, rg) in enumerate(tiles):
            cv2.putText(im, f'{rg} t={ts:.1f}s', (6, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            r, c = divmod(i, 3)
            sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = im
        cv2.imwrite(os.path.join(outdir, 'strip.png'), sheet)

    vecs = np.array(vecs, np.float32)
    np.save(os.path.join(outdir, 'obs.npy'), vecs)
    dg = np.array(diags, float)
    np.savetxt(os.path.join(outdir, 'diag.csv'), dg, delimiter=',', fmt='%.4f',
               header='t_s,active,valid,pose_valid,range_m,staleness_s,bearing_deg,attn,'
                      'att_conf,gyro_norm,cur_bound0_m,'
                      'nrm_llr,nrm_baseline_deg,nrm_nsamp,nrm_llr_prior,nrm_src,'
                      'elev_deg,speed_mps,speed_conf,img_motion,img_static,speed_conf_noise')
    if prod.normal_log:
        np.save(os.path.join(outdir, 'normal_log.npy'),
                np.array(prod.normal_log, np.float32))
    report(dg, prod, outdir, name)
    if paircheck and pair_rows:
        pr = np.array(pair_rows)
        np.savetxt(os.path.join(outdir, 'pairs.csv'), pr, delimiter=',', fmt='%.3f',
                   header='t_s,sep_m,range_a,range_b')
        sep, ra, rb = pr[:, 1], pr[:, 2], pr[:, 3]
        med = float(np.median(sep))
        print(f'\n== PAIR CHECK (two nearest clean PnP detections/frame, n={len(pr)}) ==')
        print(f'  separation median {med:.2f} m  MAD '
              f'{float(np.median(np.abs(sep - med))):.2f}  p10 '
              f'{np.percentile(sep, 10):.2f}  p90 {np.percentile(sep, 90):.2f}')
        # drift test: separation must not track viewpoint (range to nearer gate)
        if len(pr) > 20:
            c = float(np.corrcoef(ra, sep)[0, 1])
            print(f'  corr(sep, nearer-gate range) = {c:+.2f}  '
                  f'(|corr| >> 0 with large spread = frame/sign bug)')
    return outdir


def measure_pair(img):
    """(sep_m, range_a, range_b) for the two nearest clean inner detections, or None."""
    dets = D.detections(img)
    hsv = None
    good = []
    for d in dets:
        if d['source'] != 'inner' or d['size_px'] < 26.0:
            continue
        q = np.asarray(d['quad'])
        if q[:, 0].min() < 0 or q[:, 1].min() < 0 \
                or q[:, 0].max() > W or q[:, 1].max() > H:
            continue
        if hsv is None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        inw, inv, ind = interior_colour(hsv, q)
        if is_decoration(inw, inv, ind):
            continue
        if abs(d['range_m'] - d['range_size_m']) / max(d['range_m'],
                                                       d['range_size_m']) > RANGE_DISAGREE:
            continue
        pose = pnp_pose(D.order_quad(q.astype(np.float64)))
        if pose is None or pose['rms'] > max(PNP_RESID_ABS_CTR,
                                             PNP_RESID_REL * d['size_px']):
            continue
        good.append(pose)
    if len(good) < 2:
        return None
    good.sort(key=lambda p: p['range'])
    a, b = good[0], good[1]
    return (float(np.linalg.norm(a['pos'] - b['pos'])), a['range'], b['range'])


def _regime(obs):
    g0 = obs.gates[0]
    if not g0.valid:
        return 'ribbon-only' if (obs.ribbon.valid and obs.ribbon.staleness_s == 0) \
            else 'nothing-visible'
    if g0.staleness_s > 0.3:
        return 'coasting'
    if g0.range_m < 8.0:
        return 'near'
    if not g0.pose_valid:
        return 'fallback-centroid'
    if g0.range_m > 25.0:
        return 'far'
    return 'mid'


def annotate(img, obs, prod):
    for k, g in enumerate(obs.gates):
        if not g.valid:
            continue
        tr = prod.tracks.get(g.index)
        col = (0, 255, 0) if g.pose_valid else ((0, 200, 255) if g.staleness_s == 0
                                                else (0, 100, 255))
        uv = project_body(g.pos_body)
        if tr is not None and tr.quad is not None and g.staleness_s == 0:
            cv2.polylines(img, [tr.quad.astype(np.int32).reshape(-1, 1, 2)], True, col, 2)
        if uv is not None:
            cv2.circle(img, (int(uv[0]), int(uv[1])), 5, col, 1)
            cv2.putText(img, f'g{g.index} {g.range_m:.1f}m'
                        f'{"" if g.staleness_s == 0 else f" st{g.staleness_s:.1f}"}'
                        f'{" N" if g.normal_valid else ""}',
                        (int(uv[0]) + 6, int(uv[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        # ---- normalfuse: BOTH twin hypotheses + the accumulated winner --------------
        # Winner solid green, loser thin magenta. If the winner ever points at the far
        # side of a gate the arrow says so directly -- that is the symptom Claire saw.
        if tr is not None and uv is not None and tr.nrm.mu[0] is not None:
            los = tr.pos / max(float(np.linalg.norm(tr.pos)), 1e-9)
            d = float(np.clip(g.range_m * 0.35, 0.8, 4.0))
            for nvec, ccol, th in ((tr.nrm.loser_body(los), (255, 0, 255), 1),
                                   (tr.nrm.normal_body(los), (0, 255, 0), 2)):
                if nvec is None:
                    continue
                tip = project_body(tr.pos + d * nvec)
                if tip is not None:
                    cv2.arrowedLine(img, (int(uv[0]), int(uv[1])),
                                    (int(tip[0]), int(tip[1])), ccol, th,
                                    cv2.LINE_AA, tipLength=0.25)
            cv2.putText(img, f'L{tr.nrm.llr:.1f} b{tr.nrm.baseline_deg:.0f} '
                        f'{tr.normal_src[:6]}', (int(uv[0]) + 6, int(uv[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (0, 255, 0) if g.normal_valid else (120, 120, 120), 1,
                        cv2.LINE_AA)
    d = obs.attention.target_dir_body
    uv = project_body(d * 8.0)
    if uv is not None:
        cv2.arrowedLine(img, (W // 2, H - 20), (int(uv[0]), int(uv[1])),
                        (255, 200, 0), 2, tipLength=0.15)
    cv2.putText(img, f'attn={obs.attention.kind.name} act={obs.race.active_gate_index} '
                f'rp={math.degrees(obs.own.roll_rad):+.0f}/'
                f'{math.degrees(obs.own.pitch_rad):+.0f}',
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    rb = obs.ribbon
    if rb.valid and rb.staleness_s == 0:
        for b, e in zip(rb.bearings_rad, rb.elevs_rad):
            v = np.array([math.cos(e) * math.cos(b), math.cos(e) * math.sin(b),
                          -math.sin(e)])
            uv = project_body(v * 10.0)
            if uv is not None:
                cv2.circle(img, (int(uv[0]), int(uv[1])), 3, (255, 255, 0), -1)
    return img


def report(dg, prod, outdir, name):
    t, active, valid, pose_valid = dg[:, 0], dg[:, 1], dg[:, 2], dg[:, 3]
    rng, stale = dg[:, 4], dg[:, 5]
    print(f'\n== REPLAY {name}: {len(dg)} frames ==')
    print(f'current gate valid: {100 * valid.mean():.1f}%   pose_valid: '
          f'{100 * pose_valid.mean():.1f}%   mean staleness (valid frames): '
          f'{stale[valid > 0].mean():.3f} s   max coast gap: {stale.max():.2f} s')
    # map-range prior consistency: frames whose current-slot range exceeds the bound
    if dg.shape[1] > 10:
        bound = dg[:, 10]
        chk = (valid > 0) & np.isfinite(rng) & np.isfinite(bound)
        over = chk & (rng > bound)
        print(f'current range > map bound: {100 * over.sum() / max(chk.sum(), 1):.2f}% '
              f'of {int(chk.sum())} checkable frames ({int(over.sum())} frames)')
    # range monotonicity into each crossing
    adv = np.where(np.diff(active) > 0)[0]
    mono = []
    for a in adv:
        w = (t >= t[a] - 2.0) & (t <= t[a]) & (valid > 0) & np.isfinite(rng)
        if w.sum() >= 10:
            r = rng[w]
            dec = float(np.mean(np.diff(r) < 0.05))
            mono.append(dec)
    if mono:
        print(f'approach monotonicity (frac decreasing steps, 2 s pre-crossing, '
              f'{len(mono)} crossings): median {np.median(mono):.2f} '
              f'min {min(mono):.2f}')
    # rotation-coasting verification: coasted prediction vs re-acquired measurement
    if prod.reacq:
        ra = np.array(prod.reacq)
        print(f'coast->reacquire misses (n={len(ra)}, gap median '
              f'{np.median(ra[:, 0]):.2f} s): pos median {np.median(ra[:, 1]):.2f} m '
              f'p90 {np.percentile(ra[:, 1], 90):.2f} m | image median '
              f'{np.nanmedian(ra[:, 2]):.0f} px p90 {np.nanpercentile(ra[:, 2], 90):.0f} px')
        np.savetxt(os.path.join(outdir, 'reacq.csv'), ra, delimiter=',', fmt='%.4f',
                   header='gap_s,miss_m,miss_px,speed_mps,coast_dist_m')
        if ra.shape[1] >= 5 and len(ra) >= 8:
            # WHICH coasting term is missing? A rotation error grows with gap alone; a
            # TRANSLATION error grows with the distance flown during the gap. Correlating
            # the miss against both separates them, and the separation is the whole test.
            gap, miss, spd, cd = ra[:, 0], ra[:, 1], ra[:, 3], ra[:, 4]
            def _c(a, b):
                return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 \
                    and np.std(b) > 1e-9 else float('nan')
            print(f'  miss vs gap {_c(gap, miss):+.2f} | vs speed {_c(spd, miss):+.2f} '
                  f'| vs gap*speed {_c(gap * spd, miss):+.2f} | vs coast dist '
                  f'{_c(cd, miss):+.2f}   (speed med {np.median(spd):.1f} m/s, '
                  f'coast dist med {np.median(cd):.2f} m)')
    # slot geometry: the label-swap regression channel
    if prod.slot_viol[0]:
        slot_geom_bad = 100.0 * prod.slot_viol[1] / prod.slot_viol[0]
        print(f'slot map-geometry violations: {slot_geom_bad:.2f}% of '
              f'{prod.slot_viol[0]} multi-slot frames '
              f'(triangle inequality vs map distance)')
    if prod.slot_viol_fresh[0]:
        print(f'  ... restricted to FRESH slots (staleness 0): '
              f'{100.0 * prod.slot_viol_fresh[1] / prod.slot_viol_fresh[0]:.2f}% of '
              f'{prod.slot_viol_fresh[0]} frames')
    if prod.slot_order[0]:
        print(f'slot range-ordering inversions: '
              f'{100.0 * prod.slot_order[1] / prod.slot_order[0]:.2f}% of '
              f'{prod.slot_order[0]} multi-slot frames | joint repairs: '
              f'{prod.slot_fixes} | current-slot ordering evictions: {prod.slot_evicts}')
    print('slot bands (%s): re-homed to a nearer empty slot %d | dropped for band '
          'violation %d | duplicate-claim frames %d'
          % ('on' if SLOT_BANDS else 'OFF', prod.slot_moves, prod.slot_band_drops,
             prod.slot_dupes))
    # attitude_conf must collapse under sustained non-gravitational acceleration
    ac, gy = dg[:, 8], dg[:, 9]
    hi = gy > 1.0
    if hi.any():
        print(f'attitude_conf: overall median {np.median(ac):.2f} | during '
              f'|gyro|>1 rad/s ({100 * hi.mean():.0f}% of frames) median '
              f'{np.median(ac[hi]):.2f} p10 {np.percentile(ac[hi], 10):.2f}')
    # speed validity: the phantom-speed channel. A parked airframe must not be able to
    # publish a speed at confidence, and the camera is the only referee that can say so.
    if dg.shape[1] > 20:
        spd, sconf, imgm, ist = dg[:, 17], dg[:, 18], dg[:, 19], dg[:, 20]
        mm = np.nanmedian(imgm) if np.isfinite(imgm).any() else float('nan')
        print(f'image motion: median mean|dI| {mm:.2f} | camera calls '
              f'PARKED {100 * ist.mean():.1f}% of frames ({int(ist.sum())})')
        if ist.sum():
            print(f'  while parked: speed_est max {spd[ist > 0].max():.2f} m/s, '
                  f'speed_conf max {sconf[ist > 0].max():.2f}   (want 0.00 / 0.00)')
        mv = ist < 1
        if mv.sum():
            print(f'  while moving: speed_est median {np.median(spd[mv]):.2f} m/s '
                  f'p90 {np.percentile(spd[mv], 90):.2f} | speed_conf median '
                  f'{np.median(sconf[mv]):.2f} p10 {np.percentile(sconf[mv], 10):.2f} | '
                  f'transient-degraded (conf < noise conf) '
                  f'{100 * np.mean(sconf[mv] < dg[mv, 21] - 1e-6):.1f}%')
    tm = prod.timing_report()
    print('latency ms (median / p90 / n): ' +
          '  '.join(f'{k} {v[0]:.1f}/{v[1]:.1f}/{v[2]}' for k, v in tm.items()))
    if prod.map.s is not None:
        print(f'map frame anchor: s={prod.map.s:+.0f} c={prod.map.c:+.1f} deg '
              f'({len(prod.map.obs)} anchor obs)')
    else:
        print('map frame anchor: UNRESOLVED (lookahead prediction was unavailable)')
    with open(os.path.join(outdir, 'report.txt'), 'w') as fh:
        fh.write(f'{name}: valid {100 * valid.mean():.1f}% pose {100 * pose_valid.mean():.1f}%\n')
        fh.write('timing: ' + json.dumps(tm) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--replay', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--paircheck', action='store_true')
    ap.add_argument('--no-net', action='store_true')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--normal-debug', action='store_true',
                    help='log per-frame IPPE twin candidates -> normal_log.npy')
    ap.add_argument('--legacy-tilt', action='store_true',
                    help='restore the per-frame temporal+verticality tilt resolver')
    ap.add_argument('--no-map-normal-prior', action='store_true')
    ap.add_argument('--no-prior-only', action='store_true',
                    help='forbid the prior-only fallback: refuse instead')
    ap.add_argument('--no-speed-veto', action='store_true',
                    help='disable the static-image and transient vetoes on speed_est '
                         '(the pre-2026-08-02 behaviour, phantom speed included)')
    ap.add_argument('--no-coast-translate', action='store_true',
                    help='rotation-only track coasting (the pre-2026-08-02 behaviour)')
    ap.add_argument('--no-joint-slots', action='store_true',
                    help='greedy per-slot association only (the pre-fix behaviour)')
    ap.add_argument('--no-slot-bands', action='store_true',
                    help='disable the two-sided map range bands, in-order slot filling '
                         'and slot exclusivity (the pre-2026-08-02-late behaviour)')
    ap.add_argument('--from-last-reset', action='store_true',
                    help='replay only the data after the final sim reset')
    args = ap.parse_args()
    global NORMAL_DEBUG, LEGACY_TILT, USE_MAP_NORMAL_PRIOR, COAST_TRANSLATE, JOINT_SLOTS
    global SPEED_VETO
    global SLOT_BANDS
    SPEED_VETO = not args.no_speed_veto
    NORMAL_DEBUG = args.normal_debug
    LEGACY_TILT = args.legacy_tilt
    USE_MAP_NORMAL_PRIOR = not args.no_map_normal_prior
    COAST_TRANSLATE = not args.no_coast_translate
    JOINT_SLOTS = not args.no_joint_slots
    SLOT_BANDS = not args.no_slot_bands
    if args.no_prior_only:
        NF.ALLOW_PRIOR_ONLY = False
    if args.selftest:
        selftest()
        return
    if args.replay:
        replay(args.replay, limit=args.limit, video=args.video, outdir=args.outdir,
               paircheck=args.paircheck, no_net=args.no_net, stride=args.stride,
               from_last_reset=args.from_last_reset)
        return
    print('nothing to do: --selftest or --replay <session>')


if __name__ == '__main__':
    main()
