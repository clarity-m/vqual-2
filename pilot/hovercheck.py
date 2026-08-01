"""
Re-measure teleop's THRUST_HOVER independently, from every recorded session.

Run it if the hover point ever looks wrong in flight, or after any change that could
move it:  python3 pilot/hovercheck.py

Method: an IMU sample is "steady and level" when |a| is within TOL of g (so the
drone is not accelerating and drag is small) AND gravity lies along body -z
(-az/|a| close to 1, so there is no bank). At that instant thrust must be exactly
balancing weight, so the concurrently commanded thrust IS the hover point.

The trap this script exists to check: a drone SITTING ON THE PAD also reads
steady and level, at whatever thrust the stick happens to be at. Hence the
armed + thrust-floor + moving filters below.
"""
import csv
import math
import os
import sys

G = 9.81
TOL = 0.35
COS_MIN = math.cos(math.radians(10.0))


def load(path, cols):
    out = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or any(c not in rd.fieldnames for c in cols):
            return []
        for r in rd:
            try:
                out.append([float(r[c]) for c in cols])
            except (ValueError, TypeError, KeyError):
                pass
    return out


def pct(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    return v[max(0, min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1)))))]


def analyse(d, require_moving=True):
    imu = load(os.path.join(d, "imu.csv"),
               ["t_wall_ns", "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro"])
    cmd = load(os.path.join(d, "cmd.csv"), ["t_wall_ns", "armed", "thrust"])
    if not imu or not cmd:
        return None
    cmd.sort()
    ts = [c[0] for c in cmd]
    import bisect
    hits = []
    for t, ax, ay, az, gx, gy, gz in imu:
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if abs(mag - G) > TOL:
            continue
        if -az / mag < COS_MIN:
            continue
        if require_moving and max(abs(gx), abs(gy), abs(gz)) < 1e-4:
            # Identical repeated sample: the sim streams the parked pose
            # verbatim before the physics starts. Not flight.
            continue
        i = bisect.bisect_left(ts, t) - 1
        if i < 0 or t - ts[i] > 200_000_000:
            continue
        _, armed, thr = cmd[i]
        if armed < 0.5 or thr < 0.05:
            continue
        hits.append(thr)
    return hits


root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "imu.csv")):
        continue
    h = analyse(d)
    if not h:
        continue
    print("%-34s n=%4d  mean %.3f  med %.3f  p10 %.3f  p90 %.3f" % (
        name, len(h), sum(h) / len(h), pct(h, 50), pct(h, 10), pct(h, 90)))
