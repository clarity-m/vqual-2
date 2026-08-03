"""normalfuse.py -- per-TRACK accumulation of the IPPE tilt (twin) hypothesis.

THE PROBLEM THIS REPLACES.  IPPE on a planar square returns two poses that reproject
identically.  They are related by a reflection of the plane about the line of sight:

        n_false(t) = 2 (n_true . l(t)) l(t)  -  n_true                       (1)

with l the unit LOS to the gate centre.  producer.py used to pick one PER FRAME, by
comparing against *its own previous pick* (temporal consistency) with a gravity
verticality tiebreak.  That is a feedback loop with no external referee: a wrong first
pick is self-confirming, and the measured intra-track lateral sign-flip rate stayed at
~4% no matter what was done to the per-frame tiebreak.

THE STRUCTURE THAT BREAKS IT.  A gate is a static world object, so n_true is a FIXED
direction.  Rotate every frame's candidate pair into a track-common frame (the body
frame at track birth, carried forward by the gyro-integrated body rotation the producer
already maintains) and equation (1) says:

    * the TRUE branch is the same vector every frame -- it does not move;
    * the FALSE branch swings, because l swings, at roughly 2x the LOS rate.

So the hypotheses separate as soon as the LOS to the gate has moved.  Two hypotheses are
carried, each a running mean direction in the common frame plus a predictive
log-likelihood; the winner is the one whose assigned candidate keeps landing where its
own mean already was.  Nothing here consults the previous PICK -- only the accumulated
geometry -- which is what removes the feedback.

WHAT "ANGULAR BASELINE" MEANS AND WHERE THIS FAILS.  Body ROTATION contributes nothing:
it is removed by the common frame, and (1) depends only on l expressed in the world.  The
LOS moves only through TRANSLATION relative to the gate, and a perfectly straight-in
approach translates ALONG l, which leaves l unchanged.  **On a dead-straight approach this
mechanism has no signal at all**, by construction, not by tuning.  It works when the gate
is approached off-axis, passed laterally, or held while the aircraft flies a corner --
which is most of a lap, but not the last seconds of a straight final.  For that regime the
only evidence left is the priors below, or normal_valid=False.

PRIORS (evidence, not verdict).  Both are accumulated into the same log-likelihood so
they trade off against consistency instead of overriding it, and their net contribution
to the likelihood ratio is HARD-CLAMPED at PRIOR_LLR_CAP:

  verticality -- every VQ2 gate face is a vertical plane except gate 9, so the plane
      normal should be near horizontal.  NOTE THE BLIND SPOT, which explains why the
      per-frame version of this prior "rarely cleared its margin": if l is horizontal and
      n_true is horizontal, (1) keeps n_false horizontal too.  The prior discriminates
      only when the LOS has an elevation component.

  map azimuth -- course_vq2.json's per-gate yaw_race_bisector_deg, rotated into the
      levelled body frame through the producer's map-frame anchor (s, c) and the
      compass-anchored yaw integral psi_int.  This one DOES discriminate for a horizontal
      LOS, which is why it matters.  IF THE COMPASS IS WRONG the expected azimuth is
      wrong by that error; if the map MIRROR s is wrong the expected azimuth is reflected
      and the prior actively favours the WRONG twin.  That is why (a) the caller must
      only pass an azimuth when the anchor is resolved and attitude confidence is up,
      (b) SIGMA_MAP_DEG is set from the bisector's own median disagreement with the
      measured azimuths (10.9 deg, tail to 48 deg) rather than optimistically, and
      (c) the clamp exists.  With the clamp, a fully wrong compass can at worst push a
      no-baseline track from "refuse" to "wrong", never override an accumulated
      consistency verdict that has real baseline behind it.

NO LOCKING.  The accumulator is reset by the caller on track re-association (a coast gap
longer than the track's validity horizon, i.e. identity may have been reassigned) and
dies with the track on a crossing or a race reset.  Within a track, a persistent
contradiction is handled by contradiction_reset(): when the accumulated LEADER changes
hands on an already-confident track the producer halves all evidence, so a genuine
re-approach from the far side costs a few tenths of a second, not a permanent wrong lock.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

# ---- likelihood model ------------------------------------------------------------------
SIGMA_N_DEG = 10.0        # per-observation normal noise (deg). gatenet corner error puts
                          # the well-conditioned normal at ~3-6 deg and the oblique/clipped
                          # tail well past 15; 10 is deliberately pessimistic so a single
                          # bad frame cannot swing the ratio.
LL_CLIP = 4.0             # max per-sample penalty (nats). Bounds the damage from one
                          # catastrophic pose; without it a single 90 deg outlier is worth
                          # 40 nats and decides the track by itself.
SIGMA_VERT_DEG = 12.0     # verticality prior width. Measured per-gate tilt spreads sit at
                          # 2-8 deg at good views (course_vq2 tilt_sigma_deg), so 12 is
                          # loose enough to not fight gate 9's neighbours.
SIGMA_MAP_DEG = 22.0      # bisector-vs-measured azimuth disagreement is median 10.9 deg
                          # with a 48 deg tail (gate 7) -- 22 covers the bulk without
                          # making the tail a 5-sigma event.
PRIOR_PER_FRAME = 1.0 / 30.0   # priors accumulate per FRAME but are worth 1 s of frames
                               # per effective observation: consecutive frames share the
                               # same systematic map/compass error, so they are not
                               # independent samples and must not compound at 30 Hz.
PRIOR_LLR_CAP = 6.0       # hard clamp on the priors' net contribution to the LLR (nats)

# ---- what counts as evidence -----------------------------------------------------------
LOS_STEP_DEG = 1.0        # a frame contributes to the CONSISTENCY channel only when the
                          # LOS in the common frame has moved this much since the last
                          # contributing frame. Dwelling adds correlated noise, not
                          # information, and at 30 Hz it would otherwise dominate.
TWIN_MIN_SEP_DEG = 3.0    # below this the two IPPE branches are the same vector; there is
                          # nothing to decide and nothing to learn.
LAMBDA = 0.99             # forgetting per contributing sample (~100-sample window). Slow
                          # on purpose: the hypothesis MEANS must not chase a swinging
                          # false twin, which would re-create the self-confirming loop at
                          # the hypothesis level.

# ---- decision ---------------------------------------------------------------------------
LLR_MIN = 4.0             # winner must lead by this (nats) -- ~e^4 = 55:1
BASELINE_MIN_DEG = 8.0    # ...AND the track must have this much LOS spread behind it
MIN_SAMPLES = 4           # ...over at least this many contributing frames
PRIOR_DECIDE_MIN = 4.0    # prior-only fallback threshold when baseline is insufficient
ALLOW_PRIOR_ONLY = True   # the no-baseline fallback: map prior alone may decide. Set
                          # False to fall back to normal_valid=False instead (measured
                          # both ways -- see normalfuse_metrics.py).
MAX_LOS_SAMPLES = 64      # baseline window (max pairwise angle over the recent LOS set)
REORTHO_EVERY = 120       # re-orthonormalise the common-frame rotation this often


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _ang_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def wrap90(x):
    """Fold an angle difference into +-90 deg: a PLANE normal is undirected, so an
    azimuth and azimuth+180 describe the same plane."""
    return (x + 90.0) % 180.0 - 90.0


def reflect_about(n, l):
    """The IPPE twin of n about the line of sight l -- equation (1). Exported so the
    analysis can verify the twin relation on real data rather than assume it."""
    l = _unit(l)
    return 2.0 * float(np.dot(n, l)) * l - np.asarray(n, float)


class NormalAccumulator:
    """Two competing hypotheses for one gate's plane normal, in a track-common frame."""

    __slots__ = ('gate', 'R_ct', 'mu', 'w', 'll_cons', 'll_prior', 'los', 'last_los',
                 'baseline_deg', 'n_samp', 'n_frames', 'llr', 'llr_prior', 'win',
                 'decided', 'decided_by', '_rot_count', 'n_reset')

    def __init__(self, gate=-1):
        self.gate = gate
        self.n_reset = 0
        self.reset()

    # -- lifecycle ----------------------------------------------------------------------
    def reset(self):
        self.R_ct = np.eye(3)          # common <- current body
        self.mu = [None, None]
        self.w = [0.0, 0.0]
        self.ll_cons = [0.0, 0.0]
        self.ll_prior = [0.0, 0.0]
        self.los = deque(maxlen=MAX_LOS_SAMPLES)
        self.last_los = None
        self.baseline_deg = 0.0
        self.n_samp = 0
        self.n_frames = 0
        self.llr = 0.0
        self.llr_prior = 0.0
        self.win = 0
        self.decided = False
        self.decided_by = 'none'
        self._rot_count = 0

    def rotate(self, R):
        """R maps a world-fixed direction from the previous body frame to the new one
        (producer's ImuFilter.consume_rotation). The common frame must un-do it."""
        self.R_ct = self.R_ct @ R.T
        self._rot_count += 1
        if self._rot_count % REORTHO_EVERY == 0:
            u, _, vt = np.linalg.svd(self.R_ct)
            self.R_ct = u @ vt

    def _halve(self):
        """Decay, not erase: keeps the track pointing somewhere sane while new evidence
        takes over."""
        for h in (0, 1):
            self.ll_cons[h] *= 0.5
            self.ll_prior[h] *= 0.5
            self.w[h] *= 0.5

    # -- evidence -----------------------------------------------------------------------
    def update(self, cands_body, rms, los_body, quality, up_body, att_conf,
               vert_exempt, map_az_deg, R_lb):
        """One frame of evidence.

        cands_body : 1 or 2 unit normals in the CURRENT body frame, camera-signed.
        rms        : matching reprojection residuals (px), same order.
        los_body   : unit direction to the gate centre, current body frame.
        quality    : 0..1 observation weight (apparent size / source).
        up_body    : unit up (-gravity) in body, or None.
        att_conf   : gravity-filter confidence, 0..1.
        vert_exempt: True for gates whose face is not vertical (gate 9).
        map_az_deg : expected PLANE-NORMAL azimuth in the LEVELLED body frame (deg), or
                     None when the map anchor / compass is not trustworthy.
        R_lb       : levelled <- body rotation, or None.
        """
        self.n_frames += 1
        cands_body = [_unit(c) for c in cands_body]
        cands = [_unit(self.R_ct @ c) for c in cands_body]
        los_c = _unit(self.R_ct @ los_body)

        # ---- baseline: angular spread of the LOS in the common frame -------------------
        novel = self.last_los is None or _ang_deg(los_c, self.last_los) >= LOS_STEP_DEG
        if novel:
            self.last_los = los_c
            self.los.append(los_c)
            if len(self.los) >= 2:
                Lm = np.array(self.los)
                self.baseline_deg = math.degrees(
                    math.acos(float(np.clip((Lm @ Lm.T).min(), -1.0, 1.0))))

        twin_sep = _ang_deg(cands[0], cands[1]) if len(cands) > 1 else 0.0

        # ---- seed ---------------------------------------------------------------------
        if self.mu[0] is None:
            if len(cands) > 1 and twin_sep >= TWIN_MIN_SEP_DEG:
                self.mu = [cands[0].copy(), cands[1].copy()]
                self.w = [quality, quality]
                self.n_samp = 1
            return self._decide(twin_sep)

        # ---- assignment: bijection maximising alignment with the two means -------------
        if len(cands) > 1 and twin_sep >= TWIN_MIN_SEP_DEG:
            direct = float(cands[0] @ self.mu[0]) + float(cands[1] @ self.mu[1])
            swap = float(cands[1] @ self.mu[0]) + float(cands[0] @ self.mu[1])
            asg = (cands[0], cands[1]) if direct >= swap else (cands[1], cands[0])
        else:
            # single (or degenerate) solution: the frame asserts ONE direction, and a
            # hypothesis far from it is penalised by exactly that distance. This is the
            # PnP-decisive case and it is legitimate evidence for both hypotheses.
            asg = (cands[0], cands[0])

        # ---- consistency channel: predictive likelihood BEFORE the mean is updated -----
        if novel and twin_sep >= TWIN_MIN_SEP_DEG:
            for h in (0, 1):
                a = _ang_deg(asg[h], self.mu[h])
                self.ll_cons[h] = LAMBDA * self.ll_cons[h] - min(
                    LL_CLIP, 0.5 * (a / SIGMA_N_DEG) ** 2)
            for h in (0, 1):
                self.w[h] = LAMBDA * self.w[h] + quality
                # the mean is a slow average, NOT a tracker: it must stay put so that a
                # swinging false twin accumulates residual instead of being followed.
                self.mu[h] = _unit(self.mu[h] * (self.w[h] - quality) + asg[h] * quality)
            self.n_samp += 1

        # ---- residual channel (small): the frame's own reprojection preference ---------
        if len(cands) > 1 and rms is not None and len(rms) > 1:
            r_min = max(min(rms), 1e-3)
            order = (0, 1) if (asg[0] is cands[0]) else (1, 0)
            for h in (0, 1):
                extra = float(rms[order[h]]) / r_min - 1.0
                self.ll_prior[h] += PRIOR_PER_FRAME * (-min(2.0, 2.0 * extra))

        # ---- verticality prior --------------------------------------------------------
        if up_body is not None and att_conf >= 0.3 and not vert_exempt:
            for h in (0, 1):
                nb = self.R_ct.T @ asg[h]
                tilt = 90.0 - math.degrees(
                    math.acos(float(np.clip(abs(float(nb @ up_body)), 0.0, 1.0))))
                self.ll_prior[h] += PRIOR_PER_FRAME * (
                    -min(LL_CLIP, 0.5 * (tilt / SIGMA_VERT_DEG) ** 2))

        # ---- map / race-bisector azimuth prior ----------------------------------------
        if map_az_deg is not None and R_lb is not None:
            for h in (0, 1):
                nb = self.R_ct.T @ asg[h]
                nl = R_lb @ nb
                az = math.degrees(math.atan2(float(nl[1]), float(nl[0])))
                err = wrap90(az - map_az_deg)
                self.ll_prior[h] += PRIOR_PER_FRAME * (
                    -min(LL_CLIP, 0.5 * (err / SIGMA_MAP_DEG) ** 2))

        return self._decide(twin_sep)

    # -- verdict ------------------------------------------------------------------------
    def _decide(self, twin_sep=0.0):
        if self.mu[0] is None:
            self.decided, self.decided_by, self.llr, self.llr_prior = \
                False, 'none', 0.0, 0.0
            return
        d_prior = float(np.clip(self.ll_prior[0] - self.ll_prior[1],
                                -PRIOR_LLR_CAP, PRIOR_LLR_CAP))
        d_cons = self.ll_cons[0] - self.ll_cons[1]
        d_tot = d_cons + d_prior
        self.win = 0 if d_tot >= 0 else 1
        s = 1.0 if self.win == 0 else -1.0
        self.llr = abs(d_tot)
        self.llr_prior = s * d_prior
        llr_cons = s * d_cons
        by_cons = (llr_cons >= LLR_MIN and self.baseline_deg >= BASELINE_MIN_DEG
                   and self.n_samp >= MIN_SAMPLES)
        by_prior = ALLOW_PRIOR_ONLY and self.llr_prior >= PRIOR_DECIDE_MIN
        by_both = self.llr >= LLR_MIN and self.baseline_deg >= BASELINE_MIN_DEG \
            and self.n_samp >= MIN_SAMPLES
        self.decided = bool(by_cons or by_both or by_prior)
        self.decided_by = ('consistency' if by_cons else
                           ('consistency+prior' if by_both else
                            ('prior' if by_prior else 'none')))

    def contradiction_reset(self):
        """Called by the producer when the accumulated winner is contradicted hard and
        persistently (re-approach from the far side, identity reassignment)."""
        self._halve()
        self.n_reset += 1
        self._decide()

    # -- output -------------------------------------------------------------------------
    def normal_body(self, los_body=None):
        """Accumulated winner in the CURRENT body frame, signed toward the camera."""
        if self.mu[self.win] is None:
            return None
        n = _unit(self.R_ct.T @ self.mu[self.win])
        if los_body is not None and float(n @ los_body) > 0:
            n = -n
        return n

    def loser_body(self, los_body=None):
        if self.mu[1 - self.win] is None:
            return None
        n = _unit(self.R_ct.T @ self.mu[1 - self.win])
        if los_body is not None and float(n @ los_body) > 0:
            n = -n
        return n


# =========================================================================================
# self test: synthetic gate, known truth, exercises the whole mechanism


def _selftest():
    rng = np.random.default_rng(7)
    n_true = _unit([-1.0, 0.15, -0.05])          # gate normal, pointing back at us
    ok = 0
    for trial, sweep_deg in enumerate((0.0, 4.0, 10.0, 25.0, 45.0)):
        acc = NormalAccumulator(0)
        for k in range(60):
            a = math.radians(sweep_deg) * k / 59.0
            los = _unit([math.cos(a), math.sin(a), 0.02])     # LOS sweeps in the world
            c0 = _unit(n_true + rng.normal(0, 0.06, 3))
            c1 = _unit(reflect_about(c0, los))
            order = [c0, c1] if rng.random() < 0.5 else [c1, c0]
            acc.update(order, [0.2, 0.2], los, 1.0, np.array([0.0, 0.0, -1.0]), 0.0,
                       True, None, None)
        pick = acc.normal_body(los)
        err = _ang_deg(pick, n_true) if pick is not None else float('nan')
        print(f'  sweep {sweep_deg:5.1f} deg -> baseline {acc.baseline_deg:5.1f} '
              f'n_samp {acc.n_samp:3d}  LLR {acc.llr:7.2f}  decided '
              f'{str(acc.decided):5s} ({acc.decided_by})  err {err:5.1f} deg')
        if sweep_deg >= 25.0:
            assert acc.decided and err < 20.0, 'must resolve with real baseline'
            ok += 1
        if sweep_deg == 0.0:
            assert not acc.decided, 'must refuse with zero baseline and no prior'
            ok += 1
    print(f'NORMALFUSE SELFTEST PASS ({ok} assertions)')


if __name__ == '__main__':
    _selftest()
