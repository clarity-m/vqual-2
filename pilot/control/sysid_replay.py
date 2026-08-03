"""Stage 4: the gate. Replay recorded commands through the fitted plant, compare to truth.

    python3 pilot/control/sysid_replay.py [plant.json]

A force-level R^2 is not enough to trust a model with. It is computed with the *true*
attitude and the *true* velocity supplied at every sample, so it never asks the model to
propagate its own state, and a mirrored rotation can hide inside it: flip a sign and the
per-axis force fit barely notices, while the trajectory it implies bends the other way.
That is the failure the handoff describes as "great on the surrogate, bad in the sim",
which reads like a sim-to-real gap and sends you off fixing the noise model.

So this replays the whole thing open loop. Start from truth position, velocity and
attitude, feed in nothing but the recorded commands, integrate, and measure how far the
prediction has drifted after a fixed horizon. Windows are short (2 s and 5 s) and
restarted from truth, because an eight-minute open-loop integration of any model diverges
and tells you nothing.

**The corruption controls are the point.** The same replay is run with the rate mirror
inverted, with each rate axis flipped one at a time, and with the drag terms removed. If
a corrupted model scored the same as the fitted one, this test would have no teeth and
the numbers above it would be decoration. The gate is therefore the *closest* corrupted
variant, not the worst -- and each control is scored only on windows that excite the axis
it corrupts, since "indistinguishable" and "untested" are different answers.

Result (2026-07-31, held out on `20260731-131305`, which the fit never saw):

    horizon   fitted drift        nearest corrupted variant
    2 s       0.54 m,  0.4 deg    vertical drag removed, 2x
    5 s       2.60 m,  0.9 deg    vertical drag removed, 3x

over a median 35 m travelled per 5 s window, so the fitted model holds position to about
7% of distance flown open loop. Every sign corruption is caught by 17x to 60x.

Two things the table teaches beyond pass/fail:

**A yaw sign error does not move the drone.** Flipping the yaw rate sign leaves the
position error at 1.0x and puts 80 deg into the attitude error. That is not a weakness of
the test, it is the free-gimbal property the control architecture is built on -- yaw
costs nothing in path -- seen from the other side. It is also why the gate reads both
channels: a position-only gate would have passed a yaw-mirrored model.

**Removing vertical drag costs 3x.** That single term is the entire difference between
the R^2 0.05 dead end and a working fit, and here it is again as trajectory error.
"""

import os
import sys

import numpy as np

from plant import Plant, Sim, quat_from_euler, quat_to_rot
from sysid_data import interp_cols, load_epochs, zoh
from sysid_frames import truth_euler

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(HERE, "..", "sessions")

DT = 1 / 100.0
HORIZONS = (2.0, 5.0)
STRIDE = 1.0


def variants(plant):
    """The fitted plant plus deliberately corrupted copies of it.

    The last field is the command axis a variant depends on. A yaw-sign error cannot
    show up in a window that never commanded yaw, so each control is scored only on
    windows that excite the axis it corrupts. Scoring it on all of them would report
    "indistinguishable" and mean "untested".
    """
    def corrupt(**kw):
        blob = {f: getattr(plant, f) for f in Plant.FIELDS}
        blob.update(kw)
        return Plant(**blob)

    return [
        ("fitted", plant, 1.0, None),
        ("rate mirror inverted", plant, -1.0, None),
        ("roll rate sign flipped",
         corrupt(rate_gain=plant.rate_gain * np.array([-1, 1, 1])), 1.0, 0),
        ("pitch rate sign flipped",
         corrupt(rate_gain=plant.rate_gain * np.array([1, -1, 1])), 1.0, 1),
        ("yaw rate sign flipped",
         corrupt(rate_gain=plant.rate_gain * np.array([1, 1, -1])), 1.0, 2),
        ("drag removed", corrupt(kx=0.0, ky=0.0, kz=0.0), 1.0, None),
        ("vertical drag removed", corrupt(kz=0.0), 1.0, None),
    ]


MIN_EXCITATION = 0.5    # rad/s rms on an axis for a window to test that axis' sign


def windows(ep, horizon):
    """Usable windows as (start, per-axis command rms).

    Usable means flying and contact-free from end to end -- an impact inside the window
    is a force the plant has no term for, so a replay through it measures nothing.
    """
    t0, t1 = ep.span()
    out = []
    for s in np.arange(t0 + 1.0, t1 - horizon, STRIDE):
        probe = np.arange(s, s + horizon, 0.05)
        if not ep.usable(probe).all():
            continue
        rms = np.array([np.sqrt(np.mean(zoh(probe, ep.cmd["t"], ep.cmd[k]) ** 2))
                        for k in ("roll_rate", "pitch_rate", "yaw_rate")])
        out.append((s, rms))
    return out


def replay(ep, plant, horizon, start, rate_sign):
    """Integrate from truth at `start` for `horizon` seconds on commands alone."""
    t = np.arange(start, start + horizon, DT)
    roll, pitch, yaw = truth_euler(ep, t)
    p_true = interp_cols(t, ep.pos["t"], ep.pos["p"])
    v_true = interp_cols(t, ep.pos["t"], ep.pos["v"])

    sim = Sim(plant, dt=DT)
    sim.reset(p=p_true[0], v=v_true[0], q=quat_from_euler(roll[0], pitch[0], yaw[0]))
    cmd = np.stack([zoh(t, ep.cmd["t"], ep.cmd[k])
                    for k in ("roll_rate", "pitch_rate", "yaw_rate")], 1)
    thr = zoh(t, ep.cmd["t"], ep.cmd["thrust"])

    for i in range(len(t) - 1):
        sim.step(rate_sign * cmd[i], thr[i])

    R_pred = quat_to_rot(sim.q)
    R_true = quat_to_rot(quat_from_euler(roll[-1], pitch[-1], yaw[-1]))
    tilt = float(np.degrees(np.arccos(np.clip(
        (np.trace(R_true.T @ R_pred) - 1) / 2, -1, 1))))
    return dict(speed=float(np.linalg.norm(sim.v - v_true[-1])),
                position=float(np.linalg.norm(sim.p - p_true[-1])),
                attitude=tilt,
                travelled=float(np.linalg.norm(p_true[-1] - p_true[0])))


def main(argv):
    path = argv[0] if argv else os.path.join(HERE, "plant.json")
    plant = Plant.load(path)
    print("plant   %s" % path)
    print("%s\n" % plant.describe())

    eps = load_epochs([os.path.join(SESSIONS, s) for s in plant.meta["holdout"]])
    print("held-out epochs: %s" % ", ".join(ep.name for ep in eps))

    margins = []
    for horizon in HORIZONS:
        all_windows = [(ep, s, rms) for ep in eps for s, rms in windows(ep, horizon)]
        if not all_windows:
            print("\nhorizon %.0f s: no contact-free flying window that long" % horizon)
            continue
        travelled = np.median([replay(ep, plant, horizon, s, 1.0)["travelled"]
                               for ep, s, _ in all_windows])
        print("\nhorizon %.0f s   %d windows   median %.0f m travelled per window"
              % (horizon, len(all_windows), travelled))
        print("    variant                    n   speed err  position err  attitude err")

        for name, p, sign, axis in variants(plant):
            sel = all_windows if axis is None else \
                [w for w in all_windows if w[2][axis] > MIN_EXCITATION]
            if len(sel) < 10:
                print("    %-24s  --   only %d window(s) command that axis above"
                      " %.1f rad/s: untested here, not passed"
                      % (name, len(sel), MIN_EXCITATION))
                continue
            rs = [replay(ep, p, horizon, s, sign) for ep, s, _ in sel]
            med = {k: float(np.median([r[k] for r in rs]))
                   for k in ("speed", "position", "attitude")}
            base = [replay(ep, plant, horizon, s, 1.0) for ep, s, _ in sel]
            ref = {k: float(np.median([r[k] for r in base]))
                   for k in ("position", "attitude")}
            # A corruption counts as caught if it shows up in *either* channel. Yaw is
            # the reason: rotating the camera does not move the drone, so a yaw sign
            # error leaves the flight path untouched and only the attitude betrays it.
            # That is the same property the control architecture leans on -- yaw is a
            # free gimbal and costs nothing in path -- seen from the other side.
            ratio = max(med["position"] / max(ref["position"], 1e-6),
                        med["attitude"] / max(ref["attitude"], 1e-6))
            print("    %-24s %4d  %6.2f m/s   %7.2f m    %7.1f deg%s"
                  % (name, len(sel), med["speed"], med["position"], med["attitude"],
                     "" if name == "fitted" else "   (caught %.0fx)" % ratio))
            if name != "fitted":
                margins.append((horizon, name, ratio))

    if not margins:
        raise SystemExit("nothing replayed -- no usable windows")
    print("\nthe gate is the *closest* corrupted variant, not the worst:")
    for horizon in sorted({h for h, _, _ in margins}):
        near = min((r, n) for h, n, r in margins if h == horizon)
        print("  horizon %.0f s   nearest miss: %s, caught %.0fx over"
              % (horizon, near[1], near[0]))
    ok = min(r for _, _, r in margins) > 2.0
    print("\n%s" % ("PASS -- every corruption this data can see is clearly worse than the"
                    " fit.\nThe absolute drift is what it is; read it before trusting a"
                    " gain tuned on this."
                    if ok else
                    "FAIL -- a corrupted model scores nearly as well as the fit, so this"
                    "\nreplay cannot tell them apart and neither can the numbers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
