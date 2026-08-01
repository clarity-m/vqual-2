"""Stage 1: bracket the command-to-response latency, over a window wide enough to hold it.

The first-cut fit regressed command against response at an unknown misalignment: its
cross-correlation peaked at the *edge* of a +-15 sample window, which means the delay was
never bracketed, only bounded from below. A real relationship smears to nothing under
that, so the lag has to be measured before anything is fitted.

Three separate paths, because they fail differently:

    thrust command -> motor outputs     what the flight controller did with the command
    motor outputs  -> -a_z              what the airframe did with the motors
    rate command   -> gyro              the inner loop

`ACTUATOR_OUTPUT_STATUS` at ~95 Hz is what makes the split possible, and it is permitted
under VQ2. The window here is +-0.6 s, forty times the width that failed, and every peak
reported below is interior to it.

    python3 pilot/control/sysid_latency.py pilot/sessions/2026*/

Result (2026-07-31, epochs whose clock spread is small enough for a 10 ms lag to mean
anything). Every peak is interior, and every lag is *positive* -- the response follows
the command, which is the first thing a latency measurement has to get right:

    thrust cmd -> motor    median +15 ms, r >= 0.88      clean, interior
    rate cmd   -> gyro     median +10 ms, r >= 0.68      clean, interior
    motor      -> -a_z     no usable peak at any lag     NOT a latency problem

The third line is the interesting one, and it is why this stage is not the fix. The
motor-to-vertical-force correlation is weak at *every* lag because `-a_z` is not thrust:
it is thrust plus vertical drag, and in this dataset the two very nearly cancel. High
throttle happens exactly when the drone is moving fast along body z (mean |w| = 29 m/s
in the top throttle bin, against 1 m/s at hover), so the drag term grows with the thrust
term and the sum barely moves. No alignment of a single-regressor correlation can see
through that; only adding the missing term can. That is stage 2.
"""

import sys

import numpy as np

from sysid_data import SETTLE, interp_cols, load_epochs, zoh

FS = 200.0        # Hz, resampling grid; the fastest stream is ~95 Hz
MAX_LAG = 0.6     # s, each way
MIN_EXCITATION = 0.01   # std below which a channel was not commanded at all


def xcorr(a, b, fs=FS, max_lag=MAX_LAG):
    """Normalised cross-correlation of two mean-removed signals.

    Positive lag means b follows a, which is the sign convention a delay should have.
    """
    a = a - a.mean()
    b = b - b.mean()
    n = int(max_lag * fs)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den == 0:
        return np.array([0.0]), np.array([0.0])
    lags = np.arange(-n, n + 1)
    out = np.empty(len(lags))
    for i, k in enumerate(lags):
        x, y = (a[:len(a) - k], b[k:]) if k >= 0 else (a[-k:], b[:len(b) + k])
        out[i] = (x * y).sum() / den
    return lags / fs, out


def peak(lags, corr):
    i = int(np.argmax(np.abs(corr)))
    return dict(lag=float(lags[i]), r=float(corr[i]),
                edge=(i == 0 or i == len(corr) - 1))


def grid(ep):
    t0, t1 = ep.span()
    return np.arange(t0 + SETTLE, t1, 1 / FS)


def measure(ep):
    """Every path, for one epoch. Channels that were never commanded are skipped."""
    t = grid(ep)
    if len(t) < FS * 4:
        return {}
    thrust = zoh(t, ep.cmd["t"], ep.cmd["thrust"])
    out = {}
    if ep.act is not None and thrust.std() > MIN_EXCITATION:
        motor = interp_cols(t, ep.act["t"], ep.act["m"].mean(1))
        neg_az = -interp_cols(t, ep.imu["t"], ep.imu["acc"][:, 2])
        out["thrust cmd -> motor"] = peak(*xcorr(thrust, motor))
        out["motor -> -a_z"] = peak(*xcorr(motor, neg_az))
        out["thrust cmd -> -a_z"] = peak(*xcorr(thrust, neg_az))
    for ax, key in enumerate(("roll_rate", "pitch_rate", "yaw_rate")):
        cmd = zoh(t, ep.cmd["t"], ep.cmd[key])
        if cmd.std() <= MIN_EXCITATION:
            continue
        gyro = interp_cols(t, ep.imu["t"], ep.imu["gyro"][:, ax])
        out["%s -> gyro" % key] = peak(*xcorr(cmd, gyro))
    return out


MAX_SPREAD_MS = 30.0   # a lag is only meaningful within the epoch's own clock spread


def main(paths):
    eps = load_epochs(paths, truth_only=False)
    rows = {}
    for ep in eps:
        got = measure(ep)
        if not got:
            continue
        print("%-22s  clock spread +-%.0f ms" % (ep.name, ep.clock.spread_ms))
        for path, r in got.items():
            print("    %-22s lag %+7.1f ms   r %+.3f%s"
                  % (path, r["lag"] * 1e3, r["r"],
                     "   AT WINDOW EDGE -- not bracketed" if r["edge"] else ""))
            rows.setdefault(path, []).append((ep, r))

    print("\nsummary over epochs with a strong interior peak (|r| > 0.5) and a clock")
    print("spread under %.0f ms -- a lag smaller than the transport jitter of the epoch"
          % MAX_SPREAD_MS)
    print("that produced it is not a measurement:")
    for path, rs in rows.items():
        good = [r for ep, r in rs
                if not r["edge"] and abs(r["r"]) > 0.5
                and ep.clock.spread_ms < MAX_SPREAD_MS]
        if not good:
            print("  %-22s none of %d epochs -- see the header of this file"
                  % (path, len(rs)))
            continue
        lags = np.array([r["lag"] for r in good]) * 1e3
        print("  %-22s %2d/%2d epochs   lag %+.1f..%+.1f ms (median %+.1f)   |r| >= %.3f"
              % (path, len(good), len(rs), lags.min(), lags.max(), np.median(lags),
                 min(abs(r["r"]) for r in good)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
