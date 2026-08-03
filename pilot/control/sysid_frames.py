"""Stage 0: settle the frame, then prove it. Nothing downstream runs until this passes.

The surrogate is the only place in this project where a world frame appears, so it is
the only place a sign error is silent instead of self-announcing. This file is the
referee, and it is deliberately a brute force rather than an argument: build every
candidate attitude convention, and score each one against a kinematic identity that
cannot be satisfied by a mirrored frame.

The identity is that the accelerometer measures specific force -- everything except
gravity -- so for the *correct* rotation

    R_wb . a_body  +  g_world  ==  dv_world/dt

with the left side from HIGHRES_IMU plus ATTITUDE and the right side from
LOCAL_POSITION_NED velocity, which are three independent streams. A mirrored axis
breaks it immediately, and the failure is loud: the margin between the best and second
best candidate is a factor of six.

Candidates are the eight sign combinations on (roll, pitch, yaw) times both signs of
world gravity -- sixteen in all, which also settles whether the position stream is NED
(z down) as MAVLink claims.

    python3 pilot/control/sysid_frames.py pilot/sessions/2026*/

Result (2026-07-31, all eight sessions, 24 731 flying samples):

    truth_roll  =  +ATTITUDE.roll
    truth_pitch =  -ATTITUDE.pitch     (== +ODOMETRY.pitch; the streams correlate -1.000)
    truth_yaw   =  -ATTITUDE.yaw
    g_world     =  +9.81 on z          (NED, z down, as documented)

reproducing the gravity-refereed corrections in NOTES.md from a completely different
observation -- and additionally **settling the yaw sign**, which the parked-drone
referee could not do because both streams sat at the degenerate -179.9 deg heading.
This data covers the full -180..+180 deg range, and yaw+ scores 6x worse than yaw-.
"""

import sys

import numpy as np

from sysid_data import interp_cols, load_epochs

G = 9.81

# Settled by the brute force below. Applied to the ATTITUDE stream only; ODOMETRY
# carries the same information mirrored on roll and pitch, so it is used for nothing
# here beyond the cross-check printed by main().
SIGN_ROLL = +1.0
SIGN_PITCH = -1.0
SIGN_YAW = -1.0

MAX_ACCEL = 40.0     # m/s^2; above this the velocity stream is aliasing an impact
MAX_SMEAR = 0.10     # rad of rotation between truth samples, above which R is stale
SMOOTH = 5           # samples of moving average before differentiating velocity


def euler_from_quat(q):
    """ODOMETRY quaternion (w, x, y, z) -> roll, pitch, yaw. Raw, uncorrected."""
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def rot_wb(roll, pitch, yaw):
    """Body -> world (NED) rotation, Z-Y-X. Returns (N, 3, 3)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.empty((len(np.atleast_1d(roll)), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def truth_euler(ep, t):
    """Sign-corrected truth attitude, interpolated onto sim times t.

    Roll and yaw are unwrapped *before* interpolating. Both wrap at +-180 deg, and this
    drone flies inverted, so interpolating the wrapped signal sweeps an angle through
    zero that never went near it -- and differentiating it produces a rate spike of
    thousands of rad/s. Unwrapping leaves every rotation matrix unchanged (sin and cos
    do not care about 2*pi) and makes the derivative usable.
    """
    roll = SIGN_ROLL * np.interp(t, ep.att["t"], np.unwrap(ep.att["roll"]))
    pitch = SIGN_PITCH * np.interp(t, ep.att["t"], ep.att["pitch"])
    yaw = SIGN_YAW * np.interp(t, ep.att["t"], np.unwrap(ep.att["yaw"]))
    return roll, pitch, yaw


def truth_body_rates(ep):
    """Body rates in NED from the truth attitude, on the ATTITUDE stream's own clock.

    Differentiated at 114 Hz rather than at the IMU's 61 Hz: the pose stream is the
    faster one (NOTES.md), and downsampling before differentiating throws away exactly
    the transients a rate measurement is about.
    """
    t = ep.att["t"]
    roll, pitch, yaw = truth_euler(ep, t)
    dt = np.gradient(t)
    dr, dp, dy = (np.gradient(a) / dt for a in (roll, pitch, yaw))
    return t, np.stack([
        dr - dy * np.sin(pitch),
        dp * np.cos(roll) + dy * np.sin(roll) * np.cos(pitch),
        -dp * np.sin(roll) + dy * np.cos(roll) * np.cos(pitch)], 1)


def rotation(ep, t):
    return rot_wb(*truth_euler(ep, t))


def v_body(ep, t):
    """True velocity in body axes -- the variable drag actually depends on.

    This is the whole reason the truth streams are needed: `|v_world|` is frame-free
    and therefore safe, but it is also the wrong regressor, because a quad's drag is
    anisotropic in body axes and these recordings contain deliberate sideslip.
    """
    v_world = interp_cols(t, ep.pos["t"], ep.pos["v"])
    R = rotation(ep, t)
    return np.einsum("nji,nj->ni", R, v_world)      # R^T v == world -> body


def world_accel(ep, t):
    """dv_world/dt at times t, from the velocity stream.

    Smoothed before differentiating: LOCAL_POSITION_NED velocity is quantised, and
    the epoch splitting in sysid_data has already removed the reset seams that would
    otherwise dominate any derivative.
    """
    tp, v = ep.pos["t"], ep.pos["v"]
    ker = np.ones(SMOOTH) / SMOOTH
    vs = np.stack([np.convolve(v[:, i], ker, mode="same") for i in range(3)], 1)
    dv = np.gradient(vs, axis=0) / np.gradient(tp)[:, None]
    edge = SMOOTH  # the convolution ramps in at both ends
    return interp_cols(t, tp[edge:-edge], dv[edge:-edge])


def sample_mask(ep, t):
    """Samples where the identity is testable at all.

    Beyond flying-and-not-touching-anything, one more exclusion: attitude arrives at a
    finite rate, so interpolating it across a fast rotation is itself an error. At
    32 Hz and 5 rad/s the drone turns 9 deg between truth samples, and the resulting
    rotation error shows up as residual that has nothing to do with the convention
    under test. `|omega| * dt_attitude` bounds that smear directly.
    """
    d = world_accel(ep, t)
    dt_att = np.interp(t, ep.att["t"][:-1], np.diff(ep.att["t"]))
    smear = np.linalg.norm(ep.imu["gyro"], axis=1) * dt_att
    return (ep.usable(t) & np.all(np.isfinite(d), 1)
            & (np.linalg.norm(d, axis=1) < MAX_ACCEL) & (smear < MAX_SMEAR)), d


def _samples(eps):
    """Testable samples pooled over epochs: accelerometer, dv/dt, raw ATTITUDE."""
    acc, dv, ang = [], [], []
    for ep in eps:
        t = ep.imu["t"]
        m, d = sample_mask(ep, t)
        if not m.any():
            continue
        acc.append(ep.imu["acc"][m])
        dv.append(d[m])
        ang.append(np.stack([np.interp(t[m], ep.att["t"], ep.att[k])
                             for k in ("roll", "pitch", "yaw")], 1))
    if not acc:
        raise SystemExit("no testable samples with truth in the given sessions")
    return np.concatenate(acc), np.concatenate(dv), np.concatenate(ang)


def score(acc, dv, ang, signs, gz):
    """Residual of the identity for one candidate convention.

    Ranked on the *median* residual magnitude, not the mean square. Even after cutting
    contact windows a handful of samples carry a velocity-stream glitch, and squaring
    lets a dozen of those out of ten thousand outrank a genuine mirror. The median
    answers the question actually being asked -- does this convention describe the
    flight -- and the reported p95 keeps the tail visible rather than hidden.
    """
    R = rot_wb(signs[0] * ang[:, 0], signs[1] * ang[:, 1], signs[2] * ang[:, 2])
    res = np.einsum("nij,nj->ni", R, acc) + np.array([0, 0, gz]) - dv
    mag = np.linalg.norm(res, axis=1)
    return dict(signs=tuple(signs), gz=gz,
                median=float(np.median(mag)), p95=float(np.percentile(mag, 95)),
                rms=float(np.sqrt((mag ** 2).mean())),
                r2=[float(1 - np.sum(res[:, i] ** 2)
                          / np.sum((dv[:, i] - dv[:, i].mean()) ** 2))
                    for i in range(3)])


def referee(eps):
    """Score every candidate convention. Returns rows sorted best first."""
    acc, dv, ang = _samples(eps)
    rows = [score(acc, dv, ang, (sr, sp, sy), gz)
            for sr in (+1, -1) for sp in (+1, -1) for sy in (+1, -1)
            for gz in (+G, -G)]
    rows.sort(key=lambda r: r["median"])
    return rows, len(acc)


def main(paths):
    eps = load_epochs(paths)
    if not eps:
        raise SystemExit("no epochs with truth + commands in %s" % (paths,))
    print("epochs: %s\n" % ", ".join(ep.name for ep in eps))

    rows, n = referee(eps)
    print("kinematic referee:  R_wb . a_body + g == dv_world/dt      (%d samples)" % n)
    print("  rank  roll pitch  yaw   g_z   median   p95    R2 north   east   down")
    for i, r in enumerate(rows[:5]):
        print("  %2d    %+4d %+5d %+5d  %+5.1f  %7.3f %7.3f    %6.3f %6.3f %6.3f"
              % (i + 1, r["signs"][0], r["signs"][1], r["signs"][2], r["gz"],
                 r["median"], r["p95"], *r["r2"]))
    best, second = rows[0], rows[1]
    print("  ...")
    print("  %2d    %+4d %+5d %+5d  %+5.1f  %7.3f %7.3f  (worst)"
          % (len(rows), *rows[-1]["signs"], rows[-1]["gz"],
             rows[-1]["median"], rows[-1]["p95"]))

    print("\nper epoch, under the winning convention:")
    for ep in eps:
        t = ep.imu["t"]
        m, d = sample_mask(ep, t)
        if m.sum() < 50:
            print("  %-22s  too few testable samples" % ep.name)
            continue
        ang = np.stack([np.interp(t[m], ep.att["t"], ep.att[k])
                        for k in ("roll", "pitch", "yaw")], 1)
        s = score(ep.imu["acc"][m], d[m], ang, best["signs"], best["gz"])
        print("  %-22s  n %5d  median %6.3f  p95 %7.3f  imu %4.1f Hz"
              % (ep.name, m.sum(), s["median"], s["p95"],
                 (len(t) - 1) / (t[-1] - t[0])))

    print("\nATTITUDE vs ODOMETRY, re-derived rather than inherited:")
    for ep in eps[:1]:
        r_o, p_o, _ = euler_from_quat(ep.odo["q"])
        r_a = np.interp(ep.odo["t"], ep.att["t"], ep.att["roll"])
        p_a = np.interp(ep.odo["t"], ep.att["t"], ep.att["pitch"])
        print("  corr(ATTITUDE.roll,  ODOMETRY.roll)  = %+.4f" % np.corrcoef(r_a, r_o)[0, 1])
        print("  corr(ATTITUDE.pitch, ODOMETRY.pitch) = %+.4f" % np.corrcoef(p_a, p_o)[0, 1])

    margin = second["median"] / best["median"]
    print("\nbest: roll %+d, pitch %+d, yaw %+d on ATTITUDE, g_z %+.2f"
          % (*best["signs"], best["gz"]))
    print("margin over the runner-up: %.1fx on median residual" % margin)
    ok = (best["signs"] == (SIGN_ROLL, SIGN_PITCH, SIGN_YAW) and best["gz"] > 0
          and margin > 2.0 and best["median"] < 0.5)
    print("\n%s" % ("PASS -- the convention in this file is the one the data chose."
                    if ok else
                    "FAIL -- do not fit anything until this is understood."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
