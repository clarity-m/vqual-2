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
2. **Thrust from body z, with the vertical drag term carried along.** kz is fitted
   jointly here rather than assumed equal to kx: it comes out at 0.036 against 0.049
   horizontal, which is a real difference (a quad presents a different area vertically),
   not fit noise.
3. **The rate loop separately, from the gyro**, on the epochs that actually commanded
   rotation. Epochs where the commanded rate never left zero produce a meaningless R^2
   -- the denominator is noise -- so they are excluded rather than averaged in.

Delays are *not* fitted here. They are measured in `sysid_latency.py` and taken as
given, because the force model turns out to be insensitive to them: sweeping the thrust
delay from 0 to 30 ms moves held-out R^2 by 0.003, which is well inside the noise, so a
delay fitted here would be a fitted number that the data does not actually constrain.
The delays that *are* resolvable live on the command path (thrust cmd -> motor +15 ms,
rate cmd -> gyro +10 ms) and those are what go into `plant.json`.

## The split

Fit on the three deliberate system-ID cards, hold out `20260731-131305` -- nine minutes
of ordinary flying, so it tests generalisation past the excitation the fit was built
from, as `sessions/README.md` suggests.

`20260731-130744` is excluded from both. It is the one session where the kinematic
referee does not close (median residual 1.45 m/s^2 against 0.02..0.22 everywhere else),
so something is wrong with it that is not understood yet, and a fit set is the wrong
place to find out.
"""

import os
import sys

import numpy as np

from plant import Plant
from sysid_data import load_epochs, zoh
from sysid_frames import truth_body_rates, v_body

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(HERE, "..", "sessions")

FIT = ["20260731-150712", "20260731-143025", "20260731-144815"]
HOLDOUT = ["20260731-131305"]
EXCLUDED = {"20260731-130744": "kinematic referee does not close (median 1.45 m/s^2)"}

RATE_DELAY = 0.010      # s, measured: rate cmd -> gyro
THRUST_DELAY = 0.015    # s, measured: thrust cmd -> motor outputs
RATE_TAU_MAX = 0.010    # s, upper bound: replay R^2 is flat below this, falls above it
MIN_RATE_R2 = 0.90      # below this the epoch did not command enough rotation to count
MIN_RATE_FOR_MIRROR = 0.5   # rad/s on the axis being refereed, per sample


def force_samples(eps, thrust_delay=THRUST_DELAY):
    """Pooled (specific force, body velocity, delayed throttle, weight) over epochs.

    Weights are the IMU sample intervals. The rate is load-dependent (47.7 Hz on the
    VQ1 build against 62.9 on VQ2, and it drifts inside a session), so equal weight per
    sample silently over-weights whatever the recorder happened to sample densely.
    """
    f, vb, thr, wt = [], [], [], []
    for ep in eps:
        t = ep.imu["t"]
        m = ep.usable(t)
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
    """Thrust curve and vertical drag together: -f_z = c0 + c1*throttle + kz*w|w|."""
    w = vb[:, 2]
    X = np.stack([np.ones_like(thr), thr, w * np.abs(w)], 1)
    coef, r2 = wls(X, -f[:, 2], wt)
    return tuple(float(c) for c in coef), r2


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


def report_axis(name, y, pred, wt):
    print("    %-14s R2 %6.4f   rms %5.2f m/s^2   range %+6.1f..%+6.1f"
          % (name, weighted_r2(y, pred, wt),
             np.sqrt(np.average((y - pred) ** 2, weights=wt)), y.min(), y.max()))


def main(argv):
    out_path = argv[0] if argv else os.path.join(HERE, "plant.json")
    fit_eps = load_epochs([os.path.join(SESSIONS, s) for s in FIT])
    hold_eps = load_epochs([os.path.join(SESSIONS, s) for s in HOLDOUT])
    print("fit      %s" % ", ".join(ep.name for ep in fit_eps))
    print("held out %s" % ", ".join(ep.name for ep in hold_eps))
    for name, why in EXCLUDED.items():
        print("excluded %s -- %s" % (name, why))

    f, vb, thr, wt = force_samples(fit_eps)
    print("\n%d fit samples, %.1f s of flight, throttle %.2f..%.2f, "
          "body speed |u| to %.1f, |v| to %.1f, |w| to %.1f m/s"
          % (len(f), wt.sum(), thr.min(), thr.max(),
             np.abs(vb[:, 0]).max(), np.abs(vb[:, 1]).max(), np.abs(vb[:, 2]).max()))

    kx, r2x = fit_drag_axis(f, vb, wt, 0)
    ky, r2y = fit_drag_axis(f, vb, wt, 1)
    (c0, c1, kz), r2z = fit_vertical(f, vb, thr, wt)
    print("\nstep 1  drag off body x and y (no thrust term involved)")
    print("    kx %.5f  R2 %.4f        ky %.5f  R2 %.4f" % (kx, r2x, ky, r2y))
    print("step 2  thrust and vertical drag together off body z")
    print("    T/m = %+.2f %+.2f * throttle,  kz %.5f   R2 %.4f" % (c0, c1, kz, r2z))

    thrust_only = np.stack([np.ones_like(thr), thr], 1)
    _, r2_naive = wls(thrust_only, -f[:, 2], wt)
    print("    the same regression without the drag term: R2 %.4f  <- the dead end"
          % r2_naive)

    rates = fit_rates(fit_eps)
    print("\nstep 3  rate loop, per axis, in the simulator's convention")
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

    plant = Plant(kx=kx, ky=ky, kz=kz, thrust_c0=c0, thrust_c1=c1,
                  rate_gain=[float(g) for g, _ in rates],
                  rate_delay=RATE_DELAY, thrust_delay=THRUST_DELAY,
                  rate_tau_max=RATE_TAU_MAX,
                  meta=dict(fit_sessions=FIT, holdout=HOLDOUT,
                            excluded=EXCLUDED, n_samples=int(len(f)),
                            r2_fit=[r2x, r2y, r2z]))

    print("\nspecific force, predicted vs recorded:")
    pred = predict_force(plant, vb, thr)
    print("  in sample")
    for ax, nm in enumerate(("body x", "body y", "body z")):
        report_axis(nm, f[:, ax], pred[:, ax], wt)
    fh, vh, th, wh = force_samples(hold_eps)
    ph = predict_force(plant, vh, th)
    print("  held out (%d samples)" % len(fh))
    r2_out = []
    for ax, nm in enumerate(("body x", "body y", "body z")):
        report_axis(nm, fh[:, ax], ph[:, ax], wh)
        r2_out.append(weighted_r2(fh[:, ax], ph[:, ax], wh))
    plant.meta["r2_holdout"] = r2_out

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
