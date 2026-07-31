"""First-cut plant fit: drag and thrust, from steady-state segments.

Deliberately sign-free. At constant velocity the net force is zero, so

    thrust + drag + gravity = 0

and the accelerometer (which measures specific force, i.e. everything except gravity)
reads exactly -g expressed in body axes. Two consequences, both needing only the
accelerometer and the magnitude of the true velocity:

    tilt      = atan2(hypot(ax, ay), -az)      angle of the thrust axis from vertical
    drag/m    = g * tan(tilt)                  horizontal balance
    thrust/m  = g / cos(tilt)                  vertical balance

No attitude, no quaternion, no rotation matrix -- so none of the simulator's mirrored
sign conventions can contaminate the result. Speed comes from LOCAL_POSITION_NED, whose
MAGNITUDE is frame-independent.

    python3 pilot/control/plantfit.py <session-dir> [more-sessions...]
"""

import csv
import os
import sys

import numpy as np

G = 9.81
STEADY_ACC = 1.2   # m/s^2, world-frame |dv/dt| below which we call it steady
MIN_SPEED = 2.0


def load(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def gather(d):
    pos = load(os.path.join(d, "position.csv"))
    imu = load(os.path.join(d, "imu.csv"))
    cmd = load(os.path.join(d, "cmd.csv"))
    if not pos or not imu:
        return None
    tp = np.array([float(x["t_wall_ns"]) for x in pos]) / 1e9
    v = np.stack([[float(x[k]) for k in ("vx", "vy", "vz")] for x in pos])
    ti = np.array([float(x["t_wall_ns"]) for x in imu]) / 1e9
    a = np.stack([[float(x[k]) for k in ("xacc", "yacc", "zacc")] for x in imu])
    tc = np.array([float(x["t_wall_ns"]) for x in cmd]) / 1e9
    th = np.array([float(x["thrust"]) for x in cmd])

    # world-frame acceleration, to find genuinely steady stretches
    dv = np.gradient(v, axis=0) / np.maximum(np.gradient(tp), 1e-3)[:, None]
    acc = np.linalg.norm(dv, axis=1)
    speed = np.linalg.norm(v, axis=1)

    # resample everything onto the IMU clock
    sp = np.interp(ti, tp, speed)
    ac = np.interp(ti, tp, acc)
    tc_i = np.interp(ti, tc, th)

    tilt = np.arctan2(np.hypot(a[:, 0], a[:, 1]), -a[:, 2])
    amag = np.linalg.norm(a, axis=1)

    steady = (ac < STEADY_ACC) & (sp > MIN_SPEED) & (np.abs(amag - G) < 1.5)
    return dict(tilt=tilt[steady], speed=sp[steady], thr=tc_i[steady], n=int(steady.sum()),
                total=len(ti))


def main():
    parts = [gather(d) for d in sys.argv[1:]]
    parts = [p for p in parts if p and p["n"] > 50]
    if not parts:
        sys.exit("no usable steady-state samples")
    tilt = np.concatenate([p["tilt"] for p in parts])
    speed = np.concatenate([p["speed"] for p in parts])
    thr = np.concatenate([p["thr"] for p in parts])
    print("steady samples: %d (of %d)" % (sum(p["n"] for p in parts),
                                          sum(p["total"] for p in parts)))
    print("speed %.1f..%.1f m/s | tilt %.1f..%.1f deg | thrust cmd %.2f..%.2f\n"
          % (speed.min(), speed.max(), np.degrees(tilt).min(), np.degrees(tilt).max(),
             thr.min(), thr.max()))

    # --- drag: g*tan(tilt) = (k/m) * speed^n ---------------------------------------
    drag = G * np.tan(tilt)
    m = (speed > 3) & (drag > 0.05)
    n, logk = np.polyfit(np.log(speed[m]), np.log(drag[m]), 1)
    k = np.exp(logk)
    pred = k * speed[m] ** n
    r2 = 1 - np.sum((drag[m] - pred) ** 2) / np.sum((drag[m] - drag[m].mean()) ** 2)
    print("DRAG   drag/m = %.5f * v^%.3f   (R^2 %.3f, n=%d)" % (k, n, r2, m.sum()))
    n2 = 2.0
    k2 = np.sum(drag[m] * speed[m] ** n2) / np.sum(speed[m] ** (2 * n2))
    p2 = k2 * speed[m] ** n2
    r2b = 1 - np.sum((drag[m] - p2) ** 2) / np.sum((drag[m] - drag[m].mean()) ** 2)
    print("       forced v^2: drag/m = %.5f * v^2   (R^2 %.3f)" % (k2, r2b))
    print("       terminal speed at 20 deg tilt: %.1f m/s"
          % ((G * np.tan(np.radians(20)) / k2) ** 0.5))

    # --- thrust: g/cos(tilt) vs commanded ------------------------------------------
    tam = G / np.cos(tilt)
    mt = thr > 0.05
    s, b = np.polyfit(thr[mt], tam[mt], 1)
    pr = s * thr[mt] + b
    r2t = 1 - np.sum((tam[mt] - pr) ** 2) / np.sum((tam[mt] - tam[mt].mean()) ** 2)
    print("\nTHRUST thrust/m = %.2f * cmd + %.2f  m/s^2   (R^2 %.3f, n=%d)"
          % (s, b, r2t, mt.sum()))
    print("       hover cmd (thrust/m = g): %.3f" % ((G - b) / s))
    print("       max accel at cmd=1.0: %.1f m/s^2 = %.2f g" % (s + b, (s + b) / G))


if __name__ == "__main__":
    main()
