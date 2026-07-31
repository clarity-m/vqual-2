"""Settle body-rate signs against the CAMERA -- the one referee that shares no
convention with any telemetry stream.

The camera is rigidly bolted to the airframe (spec 3.8), so how the world moves in
the image is a physical fact. Comparing a commanded rate against the sim's own gyro
is self-consistent by construction and cannot detect a mirrored convention; this can.

Method: measure the apparent rotation and translation of image content across a
control input, using ORB features and a partial-affine fit.

CONVENTION, derived rather than assumed. Image coords are x right, y DOWN, and the
rotation is read as theta = atan2(M[1,0], M[0,0]) from the fitted matrix
[[cos,-sin],[sin,cos]].

Roll the camera RIGHT by 90 deg (right wing down, positive roll in body NED). The
camera's up-vector then points along world-right, so world-up -- the ceiling --
appears at image LEFT. In image coords that maps up=(0,-1) to left=(-1,0), which is
theta = -90 deg. So:

    theta_image  =  -roll_angle           roll RIGHT  -> theta NEGATIVE
    horizontal shift: yaw RIGHT  -> content moves LEFT  (dx negative)
    vertical shift:   pitch UP   -> content moves DOWN  (dy positive)

    python3 pilot/camreferee.py <session-dir>
"""

import csv
import json
import os
import sys

import cv2
import numpy as np

AXES = [("roll", "xgyro", "rotation"),
        ("pitch", "ygyro", "dy"),
        ("yaw", "zgyro", "dx")]


def load(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def fit(img_a, img_b):
    """Apparent motion of content from a to b: (theta_deg, dx, dy, n_inliers)."""
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(img_a, None)
    kb, db = orb.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:400]
    if len(m) < 20:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0)
    if M is None:
        return None
    # Translation must be measured at the IMAGE CENTRE, not the origin. The fitted
    # matrix expresses translation about (0,0), so a pure rotation about the centre
    # appears as a large spurious dx/dy -- for theta = -16.5 deg that is (-38, +98) px,
    # which is exactly the "translation" the roll taps first appeared to show.
    cx, cy = img_a.shape[1] / 2.0, img_a.shape[0] / 2.0
    mapped = M @ np.array([cx, cy, 1.0])
    return (np.degrees(np.arctan2(M[1, 0], M[0, 0])),
            float(mapped[0] - cx), float(mapped[1] - cy), int(inl.sum()))


def main():
    d = sys.argv[1]
    fr = [f for f in load(os.path.join(d, "frames.csv")) if f["file"]]
    tf = np.array([float(x["t_recv_wall_ns"]) for x in fr]) / 1e9
    imu = load(os.path.join(d, "imu.csv"))
    ti = np.array([float(x["t_wall_ns"]) for x in imu]) / 1e9
    gyro = {k: np.array([float(x[k]) for x in imu]) for k in ("xgyro", "ygyro", "zgyro")}
    cmd = load(os.path.join(d, "cmd.csv"))
    tc = np.array([float(x["t_wall_ns"]) for x in cmd]) / 1e9
    cv_ = {k: np.array([float(x[k]) for x in cmd])
           for k in ("roll_rate", "pitch_rate", "yaw_rate")}

    ev = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
    mk = sorted(float(e["t_wall_ns"]) / 1e9 for e in ev if e.get("kind") == "marker")
    print("%d markers" % len(mk))

    def frame(t):
        i = int(np.argmin(np.abs(tf - t)))
        return cv2.imread(os.path.join(d, "frames", fr[i]["file"]), cv2.IMREAD_GRAYSCALE)

    print("\n%-6s %-8s %9s %9s %8s %8s %7s  %s"
          % ("axis", "marker", "cmd", "gyro_int", "theta", "dx", "dy", "verdict"))

    for n, t0 in enumerate(mk):
        axis, gkey, cue = AXES[n % 3]
        ckey = {"roll": "roll_rate", "pitch": "pitch_rate", "yaw": "yaw_rate"}[axis]
        w = (ti > t0 - 0.05) & (ti < t0 + 0.55)
        wc = (tc > t0 - 0.05) & (tc < t0 + 0.55)
        if w.sum() < 5 or wc.sum() < 5:
            continue
        g = gyro[gkey][w]
        gint = float(np.trapezoid(g, ti[w]))          # rad actually rotated (sim frame)
        cpk = float(cv_[ckey][wc][np.argmax(np.abs(cv_[ckey][wc]))])

        a, b = frame(t0 - 0.15), frame(t0 + 0.45)
        if a is None or b is None:
            continue
        r = fit(a, b)
        if r is None:
            print("%-6s %-8d  -- no fit --" % (axis, n))
            continue
        th, dx, dy, ninl = r

        # measured physical motion, from the convention derived in the docstring
        if cue == "rotation":
            phys = -np.radians(th)                      # rad, + = rolled RIGHT
            obs = "rolled RIGHT" if phys > 0 else "rolled LEFT"
        elif cue == "dx":
            phys = -dx                                  # + = yawed RIGHT
            obs = "yawed RIGHT" if phys > 0 else "yawed LEFT"
        else:
            phys = dy                                   # + = pitched UP
            obs = "pitched UP" if phys > 0 else "pitched DOWN"

        print("%-6s %-8d %9.2f %9.3f %8.2f %8.1f %7.1f  %s (n=%d)"
              % (axis, n, cpk, gint, th, dx, dy, obs, ninl))


if __name__ == "__main__":
    main()
