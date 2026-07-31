"""Settle the body-rate sign conventions from a doublet recording.

Compares what was COMMANDED (cmd.csv, i.e. what went out over MAVLink) against what
the airframe actually DID, refereed by HIGHRES_IMU's gyro -- a physical sensor in the
body frame (spec 3.8: body -> IMU is identity) and, unlike ATTITUDE/ODOMETRY, a stream
VQ2 actually permits. If a commanded rate and the resulting gyro rate have opposite
signs, the simulator's convention for that axis is inverted relative to body NED.

Also cross-checks ATTITUDE and ODOMETRY body rates against the same gyro, since those
two are each known to be sign-inverted on a different attitude axis and their RATE
fields need checking separately.

    python3 pilot/signcheck.py pilot/sessions/<session>
"""

import csv
import json
import os
import sys

import numpy as np

AXES = [("roll", "roll_rate", "xgyro", "rollspeed"),
        ("pitch", "pitch_rate", "ygyro", "pitchspeed"),
        ("yaw", "yaw_rate", "zgyro", "yawspeed")]

DEADBAND = 0.15  # rad/s; ignore samples where nothing meaningful was commanded


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def series(rows, tkey, vkeys):
    t = np.array([float(r[tkey]) for r in rows]) / 1e9
    return t, {k: np.array([float(r[k]) for r in rows]) for k in vkeys}


def resample(t_src, v_src, t_dst):
    return np.interp(t_dst, t_src, v_src)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    cmd = load(os.path.join(d, "cmd.csv"))
    imu = load(os.path.join(d, "imu.csv"))

    tc, cv = series(cmd, "t_wall_ns", ["roll_rate", "pitch_rate", "yaw_rate", "thrust"])
    ti, iv = series(imu, "t_wall_ns", ["xgyro", "ygyro", "zgyro"])

    # data quality first -- a laptop that is not keeping up distorts command timing
    imu_sim = np.array([float(r["time_usec"]) for r in imu]) / 1e6
    wall = ti[-1] - ti[0]
    sim = imu_sim[-1] - imu_sim[0]
    print("=== data quality ===")
    print("wall span %.2f s | sim span %.2f s | realtime factor %.4f" % (wall, sim, sim / wall))
    print("IMU %.1f Hz | cmd %.1f Hz" % (len(ti) / wall, len(tc) / (tc[-1] - tc[0])))

    evp = os.path.join(d, "events.jsonl")
    if os.path.exists(evp):
        ev = [json.loads(l) for l in open(evp)]
        marks = [e for e in ev if e.get("kind") == "marker"]
        print("markers: %d" % len(marks))

    print("\n=== commanded vs achieved (referee: HIGHRES_IMU gyro) ===")
    print("%-6s %7s %9s %9s %9s  %s" % ("axis", "n", "corr", "slope", "achieved", "verdict"))
    for name, ckey, gkey, _ in AXES:
        c = cv[ckey]
        g = resample(ti, iv[gkey], tc)
        m = np.abs(c) > DEADBAND
        if m.sum() < 20:
            print("%-6s %7d  -- not commanded --" % (name, m.sum()))
            continue
        cc, gg = c[m], g[m]
        corr = float(np.corrcoef(cc, gg)[0, 1])
        slope = float(np.polyfit(cc, gg, 1)[0])
        verdict = "TO SPEC" if slope > 0 else "INVERTED"
        print("%-6s %7d %9.3f %9.3f %9.2f  %s"
              % (name, int(m.sum()), corr, slope, np.abs(gg).max(), verdict))

    # ATTITUDE / ODOMETRY rate fields vs the same gyro
    for fname, label in (("attitude.csv", "ATTITUDE"), ("odometry.csv", "ODOMETRY")):
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            continue
        rows = load(p)
        if len(rows) < 10:
            continue
        keys = [k for _, _, _, k in AXES]
        tt, tv = series(rows, "t_wall_ns", keys)
        print("\n=== %s rate fields vs gyro ===" % label)
        for name, _, gkey, rkey in AXES:
            g = resample(ti, iv[gkey], tt)
            v = tv[rkey]
            m = np.abs(g) > DEADBAND
            if m.sum() < 20:
                print("  %-6s -- too little motion --" % name)
                continue
            slope = float(np.polyfit(g[m], v[m], 1)[0])
            print("  %-6s slope %+.3f  -> %s" % (name, slope,
                                                 "agrees" if slope > 0 else "INVERTED"))


if __name__ == "__main__":
    main()
