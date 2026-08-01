"""Does teleop's TiltEstimator actually recover roll and pitch? No sim needed.

This is the strongest offline check available on this project, because it has a
referee OUTSIDE the thing being tested: it replays recorded HIGHRES_IMU through
the SHIPPED estimator and scores the result against the VQ1 build's truth pose,
which is not derived from the IMU. Same rig argument as everywhere else - VQ1 and
VQ2 physics are identical, only telemetry differs.

    truth_roll  = ATTITUDE.roll        truth_pitch = ODOMETRY.pitch    (CONVENTIONS.md)

It also settles a sign. CONVENTIONS.md lists "HIGHRES_IMU accelerometer axis signs"
under Still Unverified; sweeping all four combinations against truth separates them
by two orders of magnitude, and the winner is gyro mirrored / accelerometer canonical.

What this CANNOT prove: that the levelling assist built on top FLIES well. A P loop
closed on a 1-2 degree estimate should, but only a flight says so.

    python3 pilot/test_imu_level.py [--sessions DIR]
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teleop  # noqa: E402

DEFAULT_SESSIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")

# Recorded flying, ordered from lap-like to deliberately violent. The point of the
# spread is that accuracy is a function of how hard the aircraft is flown, and the
# slow mapping lap this was built for sits at the top of the list.
CASES = [
    # session,                        max median deg, max p90 deg, what it was
    ("20260731-204841-vq1-lap-slow",  2.5,  6.0, "clean 6/6 slow lap - the target profile"),
    ("20260731-195307",               2.5,  6.0, "lap with resets"),
    ("20260731-131305",               1.5, 10.0, "long mixed flying"),
    ("20260731-150712",               2.5, 35.0, "thrust excursions, cruise ladder, skids"),
    ("20260731-143025",               5.0, 35.0, "rate doublets - the WORST case, by design"),
]

FAILS = []


def check(name, ok, detail=""):
    print("%-4s %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def interp(t, ts, vs):
    """Linear interpolation onto t. Plain python: this must not need numpy."""
    out = []
    j = 0
    for x in t:
        while j + 1 < len(ts) - 1 and ts[j + 1] < x:
            j += 1
        if ts[j + 1] == ts[j]:
            out.append(vs[j])
        else:
            f = (x - ts[j]) / (ts[j + 1] - ts[j])
            out.append(vs[j] + f * (vs[j + 1] - vs[j]))
    return out


def replay(session, gyro_sign=-1.0, accel_sign=1.0):
    """Run the SHIPPED estimator over a session; return per-sample errors in degrees.

    gyro_sign/accel_sign exist only for the sweep below. At (-1, +1) the inputs are
    passed through untouched, which is exactly what teleop does in flight.
    """
    imu = load(os.path.join(session, "imu.csv"))
    att = os.path.join(session, "attitude.csv")
    odo = os.path.join(session, "odometry.csv")
    if len(imu) < 500 or not (os.path.exists(att) and os.path.exists(odo)):
        return None
    a, o = load(att), load(odo)
    if len(a) < 100 or len(o) < 100:
        return None

    t = [float(r["t_wall_ns"]) / 1e9 for r in imu]
    ta = [float(r["t_wall_ns"]) / 1e9 for r in a]
    tr = [float(r["roll"]) for r in a]
    to = [float(r["t_wall_ns"]) / 1e9 for r in o]
    tp = []
    for r in o:
        qw, qx, qy, qz = (float(r["qw"]), float(r["qx"]),
                          float(r["qy"]), float(r["qz"]))
        tp.append(math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))))
    true_roll = interp(t, ta, tr)
    true_pitch = interp(t, to, tp)

    est = teleop.TiltEstimator()
    er, ep = [], []
    last_us = None
    for k, row in enumerate(imu):
        us = float(row["time_usec"])
        dt = 0.0
        if last_us is not None and us > last_us:
            d = (us - last_us) / 1e6
            if d <= teleop.HEADING_MAX_GAP_S:
                dt = d
        last_us = us
        # The signs the sweep is over are applied to the INPUTS, so the estimator
        # itself is unmodified and it really is the shipped code under test.
        gyro = tuple(-gyro_sign * float(row[k2])
                     for k2 in ("xgyro", "ygyro", "zgyro"))
        accel = tuple(accel_sign * float(row[k2])
                      for k2 in ("xacc", "yacc", "zacc"))
        roll, pitch = est.update(gyro, accel, dt)
        er.append(abs(math.degrees(teleop.wrap_pi(roll - true_roll[k]))))
        ep.append(abs(math.degrees(teleop.wrap_pi(pitch - true_pitch[k]))))
    er.sort()
    ep.sort()
    return er, ep, est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=DEFAULT_SESSIONS)
    args = ap.parse_args()

    have = [(s, mm, mp, what) for s, mm, mp, what in CASES
            if os.path.isdir(os.path.join(args.sessions, s))]
    if not have:
        print("no VQ1 truth sessions under %s - nothing to referee against" % args.sessions)
        return 0

    # -- 1. the sign sweep: is (gyro -1, accel +1) actually the right hypothesis? -------
    print("sign hypotheses, pooled median |error| over %d sessions" % len(have))
    scores = {}
    for gs in (-1.0, 1.0):
        for asg in (-1.0, 1.0):
            rs, ps = [], []
            for s, _, _, _ in have:
                got = replay(os.path.join(args.sessions, s), gs, asg)
                if got:
                    rs.append(pct(got[0], 0.5))
                    ps.append(pct(got[1], 0.5))
            med = (sum(rs) / len(rs), sum(ps) / len(ps))
            scores[(gs, asg)] = med
            print("     gyro %+.0f  accel %+.0f :  roll %7.2f   pitch %7.2f  deg"
                  % (gs, asg, med[0], med[1]))
    best = min(scores, key=lambda k: scores[k][0] + scores[k][1])
    check("the measured signs are gyro MIRRORED, accelerometer CANONICAL",
          best == (-1.0, 1.0),
          "best hypothesis %s" % (best,))
    # The two signs are NOT settled equally well, and saying so is the point. Flipping
    # the accelerometer is catastrophic; flipping the gyro is merely bad, because the
    # accelerometer trim keeps dragging a wrong-signed integration back toward truth.
    # So the accelerometer sign - the one CONVENTIONS.md lists as unverified - is what
    # this test settles hardest, and the gyro sign it merely corroborates.
    total = lambda k: scores[k][0] + scores[k][1]
    acc_margin = total((-1.0, -1.0)) / max(1e-9, total((-1.0, 1.0)))
    gyro_margin = total((1.0, 1.0)) / max(1e-9, total((-1.0, 1.0)))
    check("flipping the ACCELEROMETER sign is catastrophic (>50x)",
          acc_margin > 50.0, "%.0fx worse" % acc_margin)
    check("flipping the GYRO sign is clearly worse, but only ~4x - the accelerometer "
          "trim partly hides it", gyro_margin > 3.0, "%.1fx worse" % gyro_margin)

    # -- 2. accuracy per session, against thresholds set from the measurement ----------
    print("\nestimator vs VQ1 truth (tau = %.1f s, accel gate %.1f m/s^2)"
          % (teleop.LEVEL_IMU_TAU, teleop.LEVEL_IMU_ACCEL_TOL))
    for s, max_med, max_p90, what in have:
        er, ep, est = replay(os.path.join(args.sessions, s))
        med = max(pct(er, 0.5), pct(ep, 0.5))
        p90 = max(pct(er, 0.9), pct(ep, 0.9))
        print("     %-30s roll %5.2f/%5.2f  pitch %5.2f/%5.2f  (median/p90 deg)  %s"
              % (s, pct(er, 0.5), pct(er, 0.9), pct(ep, 0.5), pct(ep, 0.9), what))
        check("  %s within its measured envelope" % s,
              med <= max_med and p90 <= max_p90,
              "median %.2f (<= %.1f), p90 %.2f (<= %.1f)" % (med, max_med, p90, max_p90))
        check("  %s: the accelerometer gate accepts a usable fraction" % s,
              est.seeded and est.accepted > 0.1 * est.samples,
              "%d of %d samples (%.0f%%) - the gyro carries the rest"
              % (est.accepted, est.samples, 100.0 * est.accepted / est.samples))

    # -- 3. properties that must hold regardless of any recording ----------------------
    print("\nproperties")
    e = teleop.TiltEstimator()
    check("unseeded until the accelerometer says something trustworthy",
          not e.seeded and e.update((0.0, 0.0, 0.0), (30.0, 0.0, 0.0), 0.02) == (0.0, 0.0),
          "a 30 m/s^2 sample is not gravity")

    e = teleop.TiltEstimator()
    # The pad is inclined 17.8 deg nose-down: seeding must land there, not at level.
    pad = (teleop.G * math.sin(math.radians(-17.8)), 0.0,
           -teleop.G * math.cos(math.radians(-17.8)))
    r, p = e.update((0.0, 0.0, 0.0), pad, 0.02)
    check("seeds to the inclined pad instead of walking to it",
          abs(math.degrees(p) + 17.8) < 0.1 and abs(r) < 1e-9,
          "pitch %.2f deg on the first sample" % math.degrees(p))

    e = teleop.TiltEstimator()
    e.update((0.0, 0.0, 0.0), (0.0, 0.0, -teleop.G), 0.02)
    # A physical +0.5 rad/s roll rate arrives mirrored, so xgyro = -0.5.
    for _ in range(50):
        e.update((-0.5, 0.0, 0.0), (5.0, 5.0, -5.0), 0.02)   # accel rejected: manoeuvring
    check("the gyro carries attitude while the accelerometer is rejected",
          abs(e.roll - 0.5) < 0.02,
          "1.0 s at 0.5 rad/s -> %.3f rad (want 0.500)" % e.roll)
    check("a rejected accelerometer does not corrupt the estimate",
          e.accepted == 1, "%d accepted of %d" % (e.accepted, e.samples))

    e = teleop.TiltEstimator()
    for _ in range(2000):
        e.update((0.0, 0.0, 0.0), (0.0, 0.0, -teleop.G), 0.02)
    check("level and still stays level (no drift, no wind-up)",
          abs(e.roll) < 1e-9 and abs(e.pitch) < 1e-9)

    e = teleop.TiltEstimator()
    e.roll, e.pitch, e.seeded = 1.0, 0.0, True
    for _ in range(int(5 * teleop.LEVEL_IMU_TAU / 0.02)):
        e.update((0.0, 0.0, 0.0), (0.0, 0.0, -teleop.G), 0.02)
    check("a wrong estimate is trimmed out by the accelerometer, monotonically",
          abs(e.roll) < 0.02, "1.00 rad -> %.4f after 5 tau" % e.roll)

    e = teleop.TiltEstimator()
    e.update((0.0, 0.0, 0.0), (0.0, 0.0, -teleop.G), 0.02)
    before = (e.roll, e.pitch)
    e.update((1.0, 1.0, 1.0), (0.0, 0.0, -teleop.G), 0.0)
    check("dt = 0 (a stall or reconnect) integrates nothing",
          (e.roll, e.pitch) == before)

    print()
    print("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
