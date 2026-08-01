"""Stage 2 and 3: fit the plant. Writes `plant.json`, which `sysid_replay.py` then judges.

    python3 pilot/control/sysid_frames.py pilot/sessions/2026*/      # must PASS first
    python3 pilot/control/sysid_fit.py                               # writes plant.json
    python3 pilot/control/sysid_replay.py                            # the gate

## Why the first cut got R^2 0.05 and this gets 0.92

Nothing was wrong with the regressor, the data, or the noise level. One term was
missing. `-a_z` is not thrust; it is thrust *plus vertical drag*, and in these
recordings the two nearly cancel: the throttle goes up exactly when the drone is moving
fast along body z, so the drag it buys grows with the thrust it commands. Mean |w| in
the top throttle bin is 29 m/s against 1 m/s at hover. Regressing `-a_z` on throttle
alone therefore measures the *difference* of two large numbers that track each other,
which is why both the throttle command and the motor sum -- the "physically correct
regressor" -- returned the same R^2 0.05. The regressor was never the problem.

Add `kz * w|w|` to the same regression on the same data and it goes to 0.92.

## What identifies what, and in which order

The order matters, because it is what removes the collinearity that made the first cut
ill-conditioned:

1. **Drag from body x and y, where thrust contributes nothing.** Thrust acts along body
   -z by definition, so `accelerometer_x` and `accelerometer_y` are pure drag. Fitting
   `-k * v_i|v_i|` there gives R^2 0.999 in-sample and 0.99 on a held-out session, with
   no thrust parameter involved and therefore nothing to trade off against.
2. **kz from the card 2 apex arcs, before any thrust parameter exists** (`sysid_apex.py`).
   Throttle is a constant through an arc, so `kz` is a slope against `w|w|` with thrust
   as a free intercept. This replaces the earlier joint fit, which was the one number in
   the model with something to trade against, and it moves kz from 0.0364 to 0.0436 --
   a 20% change, in the direction that says the joint fit was absorbing thrust.
3. **The thrust curve from body z with kz now known**, so `-a_z - kz*w|w|` is thrust and
   nothing else. Quadratic rather than affine, because the direct reads say so: the
   local slope rises from 34 m/s^2 per unit throttle at 0.05 to 57 at full throttle, and
   no straight line passes through both ends. The affine curve was the reason
   `plant.thrust()` needed a hand clamp.
4. **The rate loop separately, from the gyro**, on the epochs that actually commanded
   rotation. Epochs where the commanded rate never left zero produce a meaningless R^2
   -- the denominator is noise -- so they are excluded rather than averaged in.

Delays are *not* fitted here. They are measured in `sysid_latency.py` and taken as
given, because the force model turns out to be insensitive to them: sweeping the thrust
delay from 0 to 30 ms moves held-out R^2 by 0.003, which is well inside the noise, so a
delay fitted here would be a fitted number that the data does not actually constrain.
The delays that *are* resolvable live on the command path (thrust cmd -> motor +15 ms,
rate cmd -> gyro +10 ms) and those are what go into `plant.json`.

## The split

The fit set is **discovered, not listed**: every session directory on disk that is not
held out and not explicitly excluded. Recordings land in `pilot/sessions` and a hardcoded
list is one more thing to forget to update. What gets *removed* stays explicit, because
removing data is a judgement that has to be written down and defended; adding it is not.

`20260731-131305` is held out -- nine minutes of ordinary flying, so it tests
generalisation past the excitation the fit was built from. It is unchanged across all
three fits so their numbers are directly comparable.

Only one of the two terminal sessions is in the fit. They are the same open-loop script
run twice from the same reset and they agree to 0.1 m/s on every read, so the second is
a repeatability check, not a second sample.

## Ordering, which is the whole argument

Each stage is fitted only where the previous stage's parameter cannot hide in it:

    kx, ky   body x and y, where thrust and lift contribute nothing at all
    kz       the apex arcs -- near-vertical, so |u| ~ 0 and lift is not there to absorb
    T/m      samples chosen for having no drag term in the way (`thrust_mask`)
    c_lift   what body z has left over once the curve and kz are both fixed
    T/m      again, with lift now removed from the residual

Fitted jointly instead, these trade against each other and the estimate returns whatever
the collinearity happens to be: `corr(throttle, w|w|)` is -0.91 over the pooled set, and
the joint fit's `kz` comes out 22% low as a result. That is not a hypothetical -- it is
what the superseded fit did, and `main` still prints it for comparison.

`20260731-130744` and `20260731-233219` are excluded from both. They are the two epochs
where the kinematic referee does not close (median residual 1.45 and 2.01 m/s^2 against
0.01..0.22 everywhere else). What they have in common is a `LOCAL_POSITION_NED` velocity
stream that repeats bit-identically in 53% and 43% of its rows while the drone maneuvers
hard, so the referee is differentiating a held signal -- a recording artefact rather than
a fact about the simulator, and not one a fit set should be asked to absorb.
"""

import json
import os
import sys

import numpy as np

from plant import G, Plant
from sysid_apex import measure as apex_measure
from sysid_data import load_epochs, zoh
from sysid_frames import truth_body_rates, v_body

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(HERE, "..", "sessions")

CARD1 = ["20260731-150712", "20260731-143025", "20260731-144815"]
CARD2 = ["20260801-004843", "20260801-005059"]
HOLDOUT = [
    # Nine minutes of ordinary flying. Held out across every fit so far, which is what
    # makes their numbers comparable -- do not swap it out.
    "20260731-131305",
    # The only completed 6/6 lap ever recorded, and the only thing on disk shaped like a
    # race course: winding, slow, and flown near hover rather than in a straight line.
    # `sessions/README.md` has always said "do not fit on it"; it belongs here because
    # free-flight drift and course drift turn out to be very different numbers, and the
    # one that matters is the one measured on a course.
    "20260731-204841-vq1-lap-slow",
]
EXCLUDED = {
    "20260731-130744": "referee does not close (median 1.45 m/s^2); position velocity "
                       "stream repeats in 53% of rows",
    "20260731-233219": "referee does not close (median 2.01 m/s^2); position velocity "
                       "stream repeats in 43% of rows",
    "20260801-005518": "identical rerun of 20260801-005059; kept as a repeatability "
                       "check rather than counted twice",
    "20260801-004337": "never left the pad (0 usable samples) and the wall->sim clock "
                       "fit is degenerate: +-1382 s of spread over 20.5 s of recording",
}

# Sessions flown with the levelling assist ON. Their commands are still what went out
# over MAVLink, so they are valid force data -- but the assist generates those commands
# from the state through an outer P loop clamped at 3.0 rad/s, so the rate loop cannot
# be identified from them. They are coverage, not excitation.
#
# These three are named because they predate the recorder emitting a `level_assist`
# event and there is nothing in their files to detect it from; `sessions/README.md` is
# the only record. Anything flown since says so in `events.jsonl`, which `assist_on()`
# reads -- so a new card does not need this list edited, and forgetting to edit it does
# not silently poison the rate loop.
ASSIST_ON_LEGACY = {"20260731-195307", "20260731-203428",
                    "20260731-204841-vq1-lap-slow"}


def assist_on(name, root=SESSIONS):
    """True if this session flew any part of itself under the levelling assist."""
    if name in ASSIST_ON_LEGACY:
        return True
    path = os.path.join(root, name, "events.jsonl")
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or '"level_assist"' not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("kind") == "level_assist" and ev.get("on"):
                return True
    return False


def discover(root=SESSIONS):
    """Every session directory on disk that is neither held out nor excluded.

    Discovery rather than a hard-coded list, because new recordings land in
    `pilot/sessions` and a list in this file is one more thing to forget to update.
    The two curated sets stay explicit: what is *removed* from a fit is a judgement
    that has to be written down and justified, while what is added is just data.
    """
    names = sorted(d for d in os.listdir(root)
                   if d[:4].isdigit() and os.path.isdir(os.path.join(root, d)))
    return [n for n in names if n not in EXCLUDED and n not in HOLDOUT]

RATE_DELAY = 0.010      # s, measured: rate cmd -> gyro
THRUST_DELAY = 0.015    # s, measured: thrust cmd -> motor outputs
RATE_TAU_MAX = 0.010    # s, upper bound: replay R^2 is flat below this, falls above it
MIN_RATE_R2 = 0.90      # below this the epoch did not command enough rotation to count
MIN_RATE_FOR_MIRROR = 0.5   # rad/s on the axis being refereed, per sample


def force_samples(eps, thrust_delay=THRUST_DELAY, min_thrust=0.0):
    """Pooled (specific force, body velocity, delayed throttle, weight) over epochs.

    Weights are the IMU sample intervals. The rate is load-dependent (47.7 Hz on the
    VQ1 build against 62.9 on VQ2, and it drifts inside a session), so equal weight per
    sample silently over-weights whatever the recorder happened to sample densely.

    `min_thrust=0.0` keeps the card 2 samples: the apex holds park at exactly 0.05 and
    the terminal descent at 0.00, both of which the default gate in `sysid_data` throws
    away. The speed gate still excludes the pad.
    """
    f, vb, thr, wt = [], [], [], []
    for ep in eps:
        t = ep.imu["t"]
        m = ep.usable(t, min_thrust=min_thrust)
        if m.sum() < 100:
            continue
        dt = np.gradient(t)
        f.append(ep.imu["acc"][m])
        vb.append(v_body(ep, t[m]))
        thr.append(zoh(t[m] - thrust_delay, ep.cmd["t"], ep.cmd["thrust"]))
        wt.append(dt[m])
    if not f:
        raise SystemExit("no usable samples in %s" % [e.name for e in eps])
    return (np.concatenate(f), np.concatenate(vb), np.concatenate(thr),
            np.concatenate(wt))


def wls(X, y, w):
    """Weighted least squares, plus the weighted R^2 it achieves."""
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef, weighted_r2(y, X @ coef, w)


def weighted_r2(y, pred, w):
    mean = np.average(y, weights=w)
    return float(1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - mean) ** 2))


def fit_drag_axis(f, vb, wt, axis):
    """-k * v|v| on body x or y, where thrust contributes nothing by construction."""
    q = vb[:, axis] * np.abs(vb[:, axis])
    coef, r2 = wls(q[:, None], -f[:, axis], wt)
    return float(coef[0]), r2


def fit_vertical(f, vb, thr, wt):
    """The superseded joint fit: -f_z = c0 + c1*throttle + kz*w|w|, all three at once.

    Kept because `main` prints it beside the new one. This is the fit whose kz had a
    thrust parameter to trade against, and the comparison is the evidence that it did.
    """
    w = vb[:, 2]
    X = np.stack([np.ones_like(thr), thr, w * np.abs(w)], 1)
    coef, r2 = wls(X, -f[:, 2], wt)
    return tuple(float(c) for c in coef), r2


QUASI_W = 2.0     # m/s; below this the whole drag term is under 0.18 m/s^2
VERT_UV = 5.0     # m/s; below this the flight is near-vertical, as the arcs were


def thrust_mask(vb):
    """Samples where `-a_z` is a thrust read rather than a thrust-minus-drag tangle.

    Racing flight cannot identify this curve and never could: throttle and `w|w|`
    correlate at -0.90 over the whole pooled set and at -0.95 in the fast bins, which is
    gap 2 of card 2 restated as a number. Fitting there returns whatever the
    collinearity produces -- kz reads 0.091 on the samples with no lever arm and 0.035
    on the ones with plenty.

    So the curve is built only where the drag term is not in the way. Two ways for that
    to be true, and card 2 manufactured both:

    * `|w| < QUASI_W` -- the drag term is under 0.18 m/s^2 whatever kz is, so `-a_z` is
      thrust directly. This is the apex reads and every hover, and it covers throttle
      0.00 to 0.50.
    * near-vertical flight -- the terminal runs, where `w` reaches 31 m/s but `|u|` is
      2 m/s. The drag correction is large there, but it is the same coefficient the arcs
      measured in the same near-vertical regime, rather than an extrapolation of it into
      racing flight where `|u|` reaches 25 m/s and the component-wise form is untested.
      This is the only thing anchoring the top of the curve.
    """
    u, v, w = vb[:, 0], vb[:, 1], vb[:, 2]
    return (np.abs(w) < QUASI_W) | ((np.abs(u) < VERT_UV) & (np.abs(v) < VERT_UV))


BIN_W = 0.025     # throttle bin width for the curve fit
BIN_MIN = 20      # samples a bin needs before it is an observation of the curve
MODEL_SE = 0.20   # m/s^2; see thrust_bins


def fit_body_lift(f, vb, thr, wt, kz, knots):
    """`c` in `-a_z = T/m + kz*w|w| + c*u^2`, with the curve and `kz` already fixed.

    Ordering is the whole argument for trusting this number. `kz` comes from the apex
    arcs, which are near-vertical -- `|u|` is small there, so lift is ~0 and cannot be
    absorbed into `kz`. The thrust curve comes from samples chosen for having no drag
    term in the way. Only then is the residual asked what is left, and what is left
    correlates with forward speed squared.

    Fitted the other way round -- `c` and `kz` together on racing flight -- the two are
    not separable: `u` and `w` covary through every turn, and the joint estimate returns
    whatever that covariance happens to be, which is the same trap that gave `kz` a 20%
    error when it was fitted alongside thrust.

    Returns (c, r2, alternatives) where `alternatives` scores the rival forms on the
    same residual so the choice of `u^2` is visible rather than asserted.
    """
    u, w = vb[:, 0], vb[:, 2]
    kx_ = np.asarray([k[0] for k in knots], dtype=float)
    ky_ = np.asarray([k[1] for k in knots], dtype=float)
    resid = -f[:, 2] - np.interp(thr, kx_, ky_) - kz * w * np.abs(w)

    forms = {"c*u^2": u ** 2, "c*u|u|": u * np.abs(u), "c*|u|w": np.abs(u) * w}
    scored = {}
    for name, q in forms.items():
        coef, r2 = wls(q[:, None], resid, wt)
        scored[name] = (float(coef[0]), r2)
    c, r2 = scored["c*u^2"]
    return c, r2, scored


def thrust_bins(f, vb, thr, wt, kz, c_lift=0.0):
    """`T/m` per throttle bin: (centre, mean, standard error, n) for each populated bin.

    The curve is fitted to these rather than to the samples, because the two carry
    different information. 65% of the recordings sit in throttle 0.25..0.30, and 3000
    samples there pin *where hover is* to a hundredth of a m/s^2 while saying nothing
    about the curve's shape anywhere else. Fitting the samples directly lets that one
    throttle level set the curvature for the whole range, and it showed: the fit came
    out 3 m/s^2 above the measured full-throttle read with 240 samples sitting on it.

    Weights downstream are `1 / (SE^2 + MODEL_SE^2)`. The floor matters more than the
    standard error does -- no polynomial passes through every read exactly, so the
    binding error is misspecification, not sampling, and without the floor the densest
    bin dominates again through its vanishing SE.
    """
    m = thrust_mask(vb)
    w = vb[:, 2]
    # Body lift has to come out here too, and it is not a small correction: the mask
    # admits samples on `|w| < QUASI_W`, which includes fast level flight, so `u` can be
    # 25 m/s in a bin that is supposed to be reading thrust alone. Leaving lift in the
    # residual would push the middle of the curve up and then the curve would explain
    # the lift, which is the collinearity this whole ordering exists to avoid. On the
    # first pass `c_lift` is 0 because it is not known yet; `main` runs the pass again
    # once it is.
    y = (-f[:, 2] - kz * w * np.abs(w) - c_lift * vb[:, 0] ** 2)[m]
    t, weight = thr[m], wt[m]
    out = []
    edges = np.arange(0.0, 1.0 + BIN_W / 2, BIN_W)
    for i, lo in enumerate(edges[:-1]):
        hi = edges[i + 1]
        # The top bin has to include its upper edge: full throttle is recorded as
        # exactly 1.0, and it is the only sample anchoring that end of the curve.
        k = (t >= lo) & ((t <= hi) if i == len(edges) - 2 else (t < hi))
        if k.sum() < BIN_MIN:
            continue
        mean = float(np.average(y[k], weights=weight[k]))
        sd = float(np.sqrt(np.average((y[k] - mean) ** 2, weights=weight[k])))
        out.append((float(np.average(t[k], weights=weight[k])), mean,
                    sd / np.sqrt(k.sum()), int(k.sum())))
    return out


def fit_thrust_curve(bins, degree=2):
    """A polynomial through the bins. Kept only so `main` can show what it costs.

    Returns (coefficients padded to (c0, c1, c2), weighted R^2).
    """
    t = np.array([b[0] for b in bins])
    y = np.array([b[1] for b in bins])
    w = 1.0 / (np.array([b[2] for b in bins]) ** 2 + MODEL_SE ** 2)
    X = np.stack([t ** k for k in range(degree + 1)], 1)
    coef, r2 = wls(X, y, w)
    return tuple(float(c) for c in coef) + (0.0,) * (2 - degree), r2


def build_knots(bins, apex_reads):
    """The thrust table: the best available read at each throttle, sorted and monotone.

    Where card 2 parked the throttle and held it, the arc read wins over the bin at the
    same throttle. They disagree by up to 0.6 m/s^2 and the arc is right: a bin pools
    every sample that passed through that throttle, including the ramp on the way to the
    hold, and thrust lags the command by 15 ms, so a ramp sample reads the thrust of a
    throttle the drone has already left. The arcs use only the parked segment, which is
    the entire reason the maneuver was flown that way.
    """
    by_thr = {}
    for t, tm in apex_reads:
        by_thr.setdefault(round(t, 3), []).append(tm)
    reads = {t: float(np.mean(v)) for t, v in by_thr.items()}
    anchors = sorted(reads.items())

    # A bin that contradicts the direct reads bracketing it is dropped, not averaged in
    # and not allowed to push them around. This is the half that used to be missing, and
    # it cost 0.2 m/s^2 at hover: the monotone repair below is a running maximum from the
    # bottom, so a single low bin reading high dragged every read above it up with it --
    # the measured 0.05 read (0.595) came out of the table at 0.762 because one bin near
    # zero throttle sat above it. A measurement losing to a regression bin is backwards.
    def bracket(t):
        lo = max([v for rt, v in anchors if rt < t], default=-np.inf)
        hi = min([v for rt, v in anchors if rt > t], default=np.inf)
        return lo, hi

    kept = []
    for b in bins:
        if any(abs(b[0] - t) <= BIN_W for t in reads):
            continue                      # a read already covers this throttle
        lo, hi = bracket(b[0])
        if lo <= b[1] <= hi:
            kept.append((b[0], b[1]))

    knots = sorted(list(reads.items()) + kept)

    # A non-monotone table would let a policy find a throttle band where pushing harder
    # produces less thrust, which is a fit artefact and exactly the kind of thing an
    # optimiser goes looking for. After the bracket filter above this can only ever be
    # repairing bins against each other, never a bin against a measurement.
    out = [list(knots[0])]
    for t, tm in knots[1:]:
        out.append([t, max(tm, out[-1][1])])
    return out


def zero_crossing(coef):
    """Throttle at which the fitted curve reaches zero thrust -- where a clamp would bite."""
    c0, c1, c2 = coef
    roots = np.roots([c2, c1, c0]) if abs(c2) > 1e-12 else np.array([-c0 / c1])
    real = [float(r.real) for r in np.atleast_1d(roots)
            if abs(np.imag(r)) < 1e-9 and 0.0 <= r.real <= 1.0]
    return min(real) if real else float("nan")


def predict_force(plant, vb, thr):
    return np.stack([plant.specific_force(vb[i], thr[i]) for i in range(len(thr))])


def fit_rates(eps, delay=RATE_DELAY):
    """Per-axis gain from commanded rate to gyro, on the epochs that commanded rotation.

    Fitted in the simulator's own convention -- command and gyro carry the same mirror,
    so the gain is clean and the mirror is applied once, in `plant.rates_ned()`.
    """
    out = []
    for ax in range(3):
        key = ("roll_rate", "pitch_rate", "yaw_rate")[ax]
        used, gains = [], []
        for ep in eps:
            t = ep.imu["t"]
            # Flying only, same mask as the force fit: on the pad the ground resists the
            # rotation the command asked for, which biases the gain down (by 1% here).
            m = ep.usable(t)
            c = zoh(t[m] - delay, ep.cmd["t"], ep.cmd[key])
            g = ep.imu["gyro"][m, ax]
            if np.dot(c, c) < 1e-6:
                continue
            k = float(np.dot(c, g) / np.dot(c, c))
            r2 = float(1 - np.sum((g - k * c) ** 2) / np.sum((g - g.mean()) ** 2))
            if r2 > MIN_RATE_R2:
                used.append((ep.name, k, r2))
                gains.append(k)
        out.append((np.mean(gains) if gains else np.nan, used))
    return out


def rate_mirror(eps):
    """Per-axis ratio of true NED body rate to reported gyro. Expect -1 on all three.

    A third referee for the mirror, independent of the camera one in `camreferee.py`
    and of anything in this pipeline: differentiate the sign-corrected truth attitude,
    convert Euler rates to body rates, and compare to the gyro. The camera settled the
    mirror from outside the convention; this settles it again from the pose stream, and
    also says whether the magnitude is exactly 1 -- which the camera could not.
    """
    out = []
    for ax in range(3):
        num = den = 0.0
        for ep in eps:
            ta, rates = truth_body_rates(ep)
            t = ep.imu["t"]
            # Per axis, not per sample: a ratio whose denominator is one axis' noise is
            # not a measurement of that axis. Pooling those in is what made this read
            # -0.48 on roll and +119 on yaw before the gate was per-axis.
            m = ep.usable(t) & (np.abs(ep.imu["gyro"][:, ax]) > MIN_RATE_FOR_MIRROR)
            if m.sum() < 100:
                continue
            body = np.interp(t[m], ta, rates[:, ax])
            g = ep.imu["gyro"][m, ax]
            num += float(np.dot(g, body))
            den += float(np.dot(g, g))
        out.append(num / den if den else np.nan)
    return out


def report_kz_regime(plant, f, vb, thr, wt, fh, vh, th, wh):
    """What kz the recordings want, now that the thrust curve no longer trades with it.

    This is the check that says whether the arcs and racing flight actually agree, and
    it could only be run once the curve was pinned independently. They do agree -- but
    not on a constant: kz falls about 7% from near-vertical flight to 15 m/s of forward
    speed, which is a real cross-axis effect the component-wise model has no term for.
    A single number has to sit somewhere in that range, and the arcs measure the top of
    it. What that costs is printed here rather than hidden.
    """
    def refit(fx, vx, tx, wx, mask=None):
        w = vx[:, 2]
        q = w * np.abs(w)
        y = -fx[:, 2] - plant.thrust(tx)
        m = np.ones(len(y), bool) if mask is None else mask
        if m.sum() < 200:
            return None
        coef, _ = wls(q[m][:, None], y[m], wx[m])
        return float(coef[0])

    print("\nkz refitted with the thrust curve held fixed -- the collinearity that made")
    print("the joint fit unusable is gone, so this is now a real second opinion:")
    print("    pooled fit set   %.5f        held-out session %.5f"
          % (refit(f, vb, thr, wt), refit(fh, vh, th, wh)))
    print("    by forward speed, fit set:", end="")
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 15)):
        u = np.abs(vb[:, 0])
        k = refit(f, vb, thr, wt, (u >= lo) & (u < hi) & (np.abs(vb[:, 2]) > 3))
        if k is not None:
            print("  |u| %d-%d %.4f" % (lo, hi, k), end="")
    print("\n    the arcs measure the |u| ~ 0 end (%.5f); a compromise near 0.0420 buys"
          % plant.kz)
    p2 = Plant(kx=plant.kx, ky=plant.ky, kz=0.0420, thrust_knots=plant.thrust_knots,
               rate_gain=plant.rate_gain, rate_delay=plant.rate_delay,
               thrust_delay=plant.thrust_delay, rate_tau_max=plant.rate_tau_max)
    print("    %+.4f on held-out body z, at the cost of contradicting the measurement"
          % (weighted_r2(fh[:, 2], predict_force(p2, vh, th)[:, 2], wh)
             - weighted_r2(fh[:, 2], predict_force(plant, vh, th)[:, 2], wh)))


def report_axis(name, y, pred, wt):
    print("    %-14s R2 %6.4f   rms %5.2f m/s^2   range %+6.1f..%+6.1f"
          % (name, weighted_r2(y, pred, wt),
             np.sqrt(np.average((y - pred) ** 2, weights=wt)), y.min(), y.max()))


def main(argv):
    out_path = argv[0] if argv else os.path.join(HERE, "plant.json")
    fit_names = discover()
    fit_eps = load_epochs([os.path.join(SESSIONS, s) for s in fit_names])
    hold_eps = load_epochs([os.path.join(SESSIONS, s) for s in HOLDOUT])
    print("fit      %d sessions, %d epochs: %s"
          % (len(fit_names), len(fit_eps), ", ".join(fit_names)))
    print("held out %s" % ", ".join(ep.name for ep in hold_eps))
    for name, why in EXCLUDED.items():
        print("excluded %s -- %s" % (name, why))

    assisted = sorted(n for n in fit_names if assist_on(n))
    rate_eps = [ep for ep in fit_eps if ep.name.split("#")[0] not in set(assisted)]
    print("rate loop from %d of %d epochs (assist-off only; %s flew under the assist,"
          " so their commands come from an outer loop rather than from a pilot)"
          % (len(rate_eps), len(fit_eps), ", ".join(assisted) or "none"))

    f, vb, thr, wt = force_samples(fit_eps)
    print("\n%d fit samples, %.1f s of flight, throttle %.2f..%.2f, "
          "body speed |u| to %.1f, |v| to %.1f, |w| to %.1f m/s"
          % (len(f), wt.sum(), thr.min(), thr.max(),
             np.abs(vb[:, 0]).max(), np.abs(vb[:, 1]).max(), np.abs(vb[:, 2]).max()))

    kx, r2x = fit_drag_axis(f, vb, wt, 0)
    ky, r2y = fit_drag_axis(f, vb, wt, 1)
    print("\nstep 1  drag off body x and y (no thrust term involved)")
    print("    kx %.5f  R2 %.4f        ky %.5f  R2 %.4f" % (kx, r2x, ky, r2y))

    apex = apex_measure([os.path.join(SESSIONS, s) for s in CARD2])
    if apex["kz"] is None:
        raise SystemExit("no usable apex arc -- run sysid_apex.py to see why")
    kz = apex["kz"]
    print("step 2  kz off the card 2 apex arcs, before any thrust parameter exists")
    print("    kz %.5f   spread %.5f over %d arcs   terminal descent says %s"
          % (kz, apex["kz_spread"], len(apex["wide"]),
             ", ".join("%.5f" % k for k in apex["kz_terminal"]) or "-"))
    (jc0, jc1, jkz), r2j = fit_vertical(f, vb, thr, wt)
    print("    the superseded joint fit gave kz %.5f, %+.0f%% -- that gap is what it"
          % (jkz, 100 * (jkz - kz) / kz))
    print("    was absorbing from the thrust term it was fitted alongside")

    tm = thrust_mask(vb)
    bins = thrust_bins(f, vb, thr, wt, kz)
    print("step 3  thrust curve off body z with kz known")
    print("    %d of %d samples read thrust cleanly, throttle %.2f..%.2f  "
          "(corr(thr, w|w|) over the rest is %+.2f -- that is why they are dropped)"
          % (tm.sum(), len(tm), thr[tm].min(), thr[tm].max(),
             np.corrcoef(thr[~tm], (vb[~tm, 2] * np.abs(vb[~tm, 2])))[0, 1]))
    print("    %d throttle bins carry the curve: %s"
          % (len(bins), " ".join("%.2f" % b[0] for b in bins)))
    knots = build_knots(bins, apex["thrust_reads"])
    print("    %d knots after the direct reads replace the bins they overlap"
          % len(knots))

    c_lift, r2_lift, lift_forms = fit_body_lift(f, vb, thr, wt, kz, knots)
    print("step 3b body lift off what body z has left over, curve and kz already fixed")
    for nm in ("c*u^2", "c*u|u|", "c*|u|w"):
        coef, r2 = lift_forms[nm]
        print("    %-7s  c %+.6f   R2 %.4f%s"
              % (nm, coef, r2, "   <- kept" if nm == "c*u^2" else ""))
    print("    at |u| 20 m/s that is %.2f m/s^2, %.0f%% of hover thrust -- not a "
          "rounding error" % (c_lift * 400.0, 100 * c_lift * 400.0 / G))

    # Second pass. The first curve was built with lift still in the residual, so it
    # absorbed some of it; now that lift is known, take it out and read the curve again.
    # One pass, not iterated to convergence: the correction is small enough that a
    # second round moves the knots by less than the direct reads' own spread, and an
    # iteration that keeps going would only be fitting its own last answer.
    before = np.interp(np.linspace(0, 1, 101),
                       [k[0] for k in knots], [k[1] for k in knots])
    bins = thrust_bins(f, vb, thr, wt, kz, c_lift)
    knots = build_knots(bins, apex["thrust_reads"])
    after = np.interp(np.linspace(0, 1, 101),
                      [k[0] for k in knots], [k[1] for k in knots])
    print("    curve rebuilt with lift removed: moves by at most %.2f m/s^2 "
          "(%.2f at hover), %d knots" % (np.abs(after - before).max(),
                                         abs(float(np.interp(0.27, np.linspace(0, 1, 101),
                                                             after - before))),
                                         len(knots)))
    curves = {}
    for degree, label in ((1, "affine   ", ), (2, "quadratic")):
        coef, r2 = fit_thrust_curve(bins, degree)
        curves[degree] = coef
        print("    for comparison, %s T/m = %+.2f %+.2f*thr %+.2f*thr^2   "
              "zero at thr %.3f" % (label, coef[0], coef[1], coef[2],
                                    zero_crossing(coef)))

    thrust_only = np.stack([np.ones_like(thr), thr], 1)
    _, r2_naive = wls(thrust_only, -f[:, 2], wt)
    print("    the same regression without any drag term: R2 %.4f  <- the dead end"
          % r2_naive)

    c0, c1, c2 = curves[2]

    rates = fit_rates(rate_eps)
    print("\nstep 4  rate loop, per axis, in the simulator's convention")
    for ax, (gain, used) in enumerate(rates):
        label = ("roll", "pitch", "yaw")[ax]
        if not used:
            print("    %-6s no epoch commanded this axis hard enough to fit" % label)
            continue
        print("    %-6s gain %.3f   from %d epoch(s): %s"
              % (label, gain, len(used),
                 ", ".join("%s %.2f (R2 %.3f)" % u for u in used)))

    mirror = rate_mirror(fit_eps)
    print("    mirror referee: true NED body rate / gyro = %s"
          % "  ".join("%s %+.3f" % (n, m)
                      for n, m in zip(("roll", "pitch", "yaw"), mirror)))
    print("        %s"
          % ("all three -1.00 +-0.02: the mirror is exact and global, as camreferee.py"
             " found" if all(abs(m + 1) < 0.02 for m in mirror)
             else "NOT a clean -1 on every axis -- plant.SIGN_RATE is wrong for this data"))

    def build(kn, c=None):
        return Plant(kx=kx, ky=ky, kz=kz,
                     c_lift=c_lift if c is None else c, thrust_knots=kn,
                     rate_gain=[float(g) for g, _ in rates],
                     rate_delay=RATE_DELAY, thrust_delay=THRUST_DELAY,
                     rate_tau_max=RATE_TAU_MAX,
                     meta=dict(fit_sessions=fit_names, holdout=HOLDOUT,
                               excluded=EXCLUDED, n_samples=int(len(f)),
                               kz_source="sysid_apex arcs", kz_spread=apex["kz_spread"],
                               kz_terminal=apex["kz_terminal"],
                               c_lift_source="body-z residual after curve and kz, "
                                             "u^2 chosen over u|u| and |u|w on R^2",
                               c_lift_r2=r2_lift,
                               rate_sessions=[e.name for e in rate_eps],
                               r2_fit=[r2x, r2y, None]))

    fh, vh, th, wh = force_samples(hold_eps)

    # Decided on held-out body z, not in sample. A polynomial can always be made to fit
    # the bins; the question is whether its shape is the flight's shape, and it is not:
    # the measured curve has a deadband at the bottom and flattens at the top, and both
    # ends are where a racing policy lives.
    print("\nchoosing the curve shape on held-out body z (%d samples)" % len(fh))
    grid = np.linspace(0.0, 1.0, 101)
    for label, kn in (("affine   ", [[t, curves[1][0] + curves[1][1] * t]
                                     for t in grid]),
                      ("quadratic", [[t, curves[2][0] + curves[2][1] * t
                                      + curves[2][2] * t * t] for t in grid]),
                      ("table    ", knots)):
        kn = [[t, max(0.0, y)] for t, y in kn]
        p = build(kn)
        r2 = weighted_r2(fh[:, 2], predict_force(p, vh, th)[:, 2], wh)
        err = max(abs(float(p.thrust(t)) - tm) for t, tm in apex["thrust_reads"])
        print("    %s  held-out body z R2 %.4f   hover %.3f   full %.1f   "
              "worst miss on a direct read %.2f m/s^2"
              % (label, r2, p.hover_throttle(), p.thrust(1.0), err))

    plant = build(knots)

    print("\nspecific force, predicted vs recorded:")
    pred = predict_force(plant, vb, thr)
    print("  in sample")
    for ax, nm in enumerate(("body x", "body y", "body z")):
        report_axis(nm, f[:, ax], pred[:, ax], wt)
    ph = predict_force(plant, vh, th)
    print("  held out (%d samples)" % len(fh))
    r2_out = []
    for ax, nm in enumerate(("body x", "body y", "body z")):
        report_axis(nm, fh[:, ax], ph[:, ax], wh)
        r2_out.append(weighted_r2(fh[:, ax], ph[:, ax], wh))
    plant.meta["r2_holdout"] = r2_out
    plant.meta["r2_fit"] = [r2x, r2y,
                            weighted_r2(f[:, 2], pred[:, 2], wt)]

    # What the lift term actually bought, measured where it counts: on data the fit
    # never saw, against the same curve and the same kz. In sample it can only help.
    r2_nolift = weighted_r2(fh[:, 2], predict_force(build(knots, c=0.0),
                                                    vh, th)[:, 2], wh)
    plant.meta["r2_holdout_no_lift_z"] = r2_nolift
    print("    body z with the lift term set to zero: R2 %.4f  (%+.4f from keeping it)"
          % (r2_nolift, r2_out[2] - r2_nolift))

    print("\nthrust curve against the direct reads it was built to honour:")
    for t_read, tm in sorted(apex["thrust_reads"]):
        print("    throttle %.2f   measured %7.3f   model %7.3f   %+6.3f m/s^2"
              % (t_read, tm, plant.thrust(t_read), plant.thrust(t_read) - tm))

    report_kz_regime(plant, f, vb, thr, wt, fh, vh, th, wh)

    print("\n%s" % plant.describe())
    print("terminal speed at 20 deg tilt: %.1f m/s forward, %.1f m/s sideways"
          % (plant.terminal_speed(0), plant.terminal_speed(1)))
    print("hover throttle %.3f  (flight-measured ~0.27, NOTES.md)"
          % plant.hover_throttle())

    plant.to_json(out_path)
    print("\nwrote %s" % out_path)
    print("NOT VALIDATED YET -- run sysid_replay.py before tuning anything on this.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
