"""Offline exercise of teleop's input shaping. No sim, no keyboard, no MAVLink.

What this can prove: that the `full` preset reproduces the pre-shaping command
stream EXACTLY - not approximately, and not by calling the new code twice, but
against the old algebra written out again below; that expo softens the centre
without touching the endpoints; that the rate cap cannot be escaped by boost, by
the align loop or by the levelling assist; that the throttle channel is untouched
by any of it; and that the shaping in force is recorded.

What it CANNOT prove: whether the aircraft is nicer to fly. That needs a flight.

    python3 pilot/test_shaping.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teleop  # noqa: E402


# Local doubles rather than an import from test_thrust_snap: that module runs its
# checks at import time and calls sys.exit, so importing it would run the wrong suite.
class FakeConn:
    target_system = 1
    target_component = 1

    def __init__(self):
        self.sent = []
        self.mav = self

    def set_attitude_target_send(self, _t, _sys, _comp, _mask, _q,
                                 roll, pitch, yaw, thrust):
        self.sent.append((roll, pitch, yaw, thrust))


class FakeRec:
    def __init__(self):
        self.events = []

    def row(self, *a):
        pass

    def event(self, kind, **p):
        self.events.append((kind, p))


FAILS = []


def check(name, ok, detail=""):
    print("%-4s %s%s" % ("PASS" if ok else "FAIL", name,
                         ("   " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def make(shaping=None):
    conn, rec = FakeConn(), FakeRec()
    return teleop.Pilot(conn, rec, 0, shaping=shaping), conn, rec


# --------------------------------------------------------------------------------------
# The pre-shaping code, written out so the comparison is against the OLD ALGEBRA
# rather than against the new code calling itself. Mirrors teleop.Sticks.poll and
# the stick half of teleop.Pilot.send as they stood at commit dba8ad6.
# --------------------------------------------------------------------------------------

def legacy_axes(key_sequence, dt=0.02):
    """Old Sticks.poll: four axes slewed at SLEW_PER_S toward the key targets."""
    axes = [0.0, 0.0, 0.0, 0.0]
    out = []
    for target in key_sequence:
        step = teleop.SLEW_PER_S * dt
        for i, want in enumerate(target):
            delta = want - axes[i]
            if delta > step:
                delta = step
            elif delta < -step:
                delta = -step
            axes[i] += delta
        out.append(list(axes))
    return out


def legacy_rates(axes, scale):
    """Old Pilot.send: axis -> body rate, no shaping anywhere."""
    pitch_ax, roll_ax, _thr_ax, yaw_ax = axes
    return (roll_ax * teleop.ACRO_ROLL * scale,
            -pitch_ax * teleop.ACRO_PITCH * scale,
            yaw_ax * teleop.ACRO_YAW * scale)


# Synthetic key sequences. Each entry is (pitch, roll, throttle, yaw) key demand.
def seq_taps(n=400):
    """Taps, holds, reversals and diagonals - the awkward inputs, not smooth ones."""
    out = []
    for k in range(n):
        p = 1.0 if 20 <= k < 26 else -1.0 if 60 <= k < 130 else 0.0
        r = 1.0 if 40 <= k < 43 else -1.0 if 200 <= k < 260 else 0.0
        t = 1.0 if 100 <= k < 140 else -1.0 if 300 <= k < 310 else 0.0
        y = -1.0 if 150 <= k < 155 else 1.0 if 250 <= k < 350 else 0.0
        out.append((p, r, t, y))
    return out


KEYS = seq_taps()


# -- 1. `full` is byte-identical to the pre-shaping command stream ----------------------
sticks = teleop.Sticks(slew=teleop.SHAPING_PRESETS["full"]["stick_slew"])
full = teleop.Shaping.preset("full")
p, conn, _ = make(full)

# Drive the real Sticks slew law with the same key sequence the legacy one sees, by
# substituting the key reader. This exercises poll() itself, not a copy of it.
held = {"keys": (0.0, 0.0, 0.0, 0.0)}
name_to_idx = {}
for i, ax in enumerate(("pitch", "roll", "throttle", "yaw")):
    pos, neg = teleop.KEYS_AXIS[ax]
    name_to_idx[pos] = (i, +1)
    name_to_idx[neg] = (i, -1)
teleop.Sticks._held = staticmethod(
    lambda name: (name in name_to_idx
                  and held["keys"][name_to_idx[name][0]] * name_to_idx[name][1] > 0.5))

got_axes = []
for keys in KEYS:
    held["keys"] = keys
    axes, scale, _actions = sticks.poll(0.02)
    got_axes.append(axes)

want_axes = legacy_axes(KEYS)
check("Sticks slew under `full` matches the old law exactly",
      got_axes == want_axes,
      "%d ticks" % len(got_axes))

for scale in (1.0, teleop.BOOST, teleop.PRECISION):
    p, conn, _ = make(teleop.Shaping.preset("full"))
    for axes in want_axes:
        p.send(axes, scale, 0.02, True)
    got = [s[:3] for s in conn.sent]
    want = [legacy_rates(a, scale) for a in want_axes]
    check("`full` rates identical to pre-shaping algebra at scale %.2f" % scale,
          got == want,
          "%d ticks, max |diff| %.3g" % (
              len(got),
              max(max(abs(g[i] - w[i]) for i in range(3))
                  for g, w in zip(got, want))))

# thrust must be untouched by shaping, under both presets
thr_full, thr_slow = [], []
for shp, sink in ((teleop.Shaping.preset("full"), thr_full),
                  (teleop.Shaping.preset("slow"), thr_slow)):
    p, conn, _ = make(shp)
    for axes in want_axes:
        p.send(axes, 1.0, 0.02, True)
    sink.extend(s[3] for s in conn.sent)
check("the throttle channel is identical under `slow` and `full`",
      thr_full == thr_slow, "%d ticks" % len(thr_full))

s_slow = teleop.Sticks(slew=teleop.SHAPING_PRESETS["slow"]["stick_slew"])
slow_axes = []
for keys in KEYS:
    held["keys"] = keys
    slow_axes.append(s_slow.poll(0.02)[0])
check("throttle axis rides SLEW_PER_S regardless of preset",
      [a[2] for a in slow_axes] == [a[2] for a in want_axes])
check("rate axes ride the preset's slower ramp",
      max(abs(a[1]) for a in slow_axes) <= max(abs(a[1]) for a in want_axes)
      and [a[0] for a in slow_axes] != [a[0] for a in want_axes])


# -- 2. the expo curve -----------------------------------------------------------------
s = teleop.Shaping("t", expo=0.55)
check("expo is identity at 0 and at full deflection",
      s.curve(0.0) == 0.0 and abs(s.curve(1.0) - 1.0) < 1e-12
      and abs(s.curve(-1.0) + 1.0) < 1e-12,
      "curve(1) = %.6f" % s.curve(1.0))
check("expo softens the centre", s.curve(0.5) < 0.5 and s.curve(0.5) > 0.0,
      "half stick -> %.3f (%.0f%% of linear)" % (s.curve(0.5), 200 * s.curve(0.5)))
check("expo is odd (symmetric about centre)",
      all(abs(s.curve(-x) + s.curve(x)) < 1e-12
          for x in (0.1, 0.25, 0.5, 0.75, 1.0)))
check("expo is monotonic", all(s.curve(x / 100.0) < s.curve((x + 1) / 100.0)
                               for x in range(0, 100)))
check("expo 0 is an exact early return, not a multiply",
      all(teleop.Shaping("z", expo=0.0).curve(x) == x
          for x in (-1.0, -0.37, 0.0, 0.37, 1.0)))


# -- 3. the rate cap is inescapable ----------------------------------------------------
cap = teleop.SHAPING_PRESETS["slow"]["max_rate"]
p, conn, _ = make(teleop.Shaping.preset("slow"))
for _ in range(20):
    p.send([1.0, -1.0, 0.0, 1.0], teleop.BOOST, 0.02, True)
check("LCTRL boost cannot exceed the cap",
      all(max(abs(v) for v in s[:3]) <= cap + 1e-12 for s in conn.sent),
      "peak %.3f vs cap %.2f" % (max(max(abs(v) for v in s[:3])
                                     for s in conn.sent), cap))

# align-to-velocity drives yaw straight from the accelerometer
p, conn, _ = make(teleop.Shaping.preset("slow"))
for _ in range(20):
    p.send([0.0, 0.0, 0.0, 0.0], 1.0, 0.02, True, align=True, accel=(-3.0, 3.0, -9.0))
check("the align loop cannot exceed the cap",
      p.aligning and all(abs(s[2]) <= cap + 1e-12 for s in conn.sent),
      "peak yaw %.3f" % max(abs(s[2]) for s in conn.sent))

# the levelling assist's outer P loop is clamped at LEVEL_MAX_RATE = 3.0, above the cap
p, conn, _ = make(teleop.Shaping.preset("slow"))
p.send([0.0, 0.0, 0.0, 0.0], 1.0, 0.02, True, level=True,
       truth_att=(math.radians(60), math.radians(-45), time.time()))
check("the levelling assist cannot exceed the cap",
      p.levelling and max(abs(v) for v in conn.sent[-1][:3]) <= cap + 1e-12,
      "sent %s" % (tuple(round(v, 3) for v in conn.sent[-1][:3]),))

p, conn, _ = make(teleop.Shaping.preset("full"))
p.send([1.0, -1.0, 0.0, 1.0], teleop.BOOST, 0.02, True)
check("`full` applies no cap at all",
      abs(conn.sent[-1][0]) > cap and abs(conn.sent[-1][1]) > cap,
      "roll %.2f pitch %.2f" % conn.sent[-1][:2])

check("a cap of 0 disables the clamp",
      teleop.Shaping("z", max_rate=0.0).clamp(99.0) == 99.0)


# -- 4. sign is never changed by shaping -----------------------------------------------
for preset in ("full", "slow"):
    p, conn, _ = make(teleop.Shaping.preset(preset))
    p.send([1.0, 1.0, 0.0, 1.0], 1.0, 0.02, True)
    a = conn.sent[-1]
    p2, conn2, _ = make(teleop.Shaping.preset(preset))
    p2.send([-1.0, -1.0, 0.0, -1.0], 1.0, 0.02, True)
    b = conn2.sent[-1]
    check("`%s` preserves the sign convention" % preset,
          all(math.copysign(1, a[i]) == -math.copysign(1, b[i]) for i in range(3))
          and math.copysign(1, a[0]) == math.copysign(1, teleop.ACRO_ROLL),
          "+full %s / -full %s" % (tuple(round(v, 2) for v in a[:3]),
                                   tuple(round(v, 2) for v in b[:3])))


# -- 5. the shaping in force is recorded -----------------------------------------------
d = teleop.Shaping.preset("slow").as_dict()
check("as_dict carries everything that scales a command",
      set(d) == {"preset", "expo", "max_rate_rad_s", "stick_slew_per_s",
                 "acro_roll", "acro_pitch", "acro_yaw", "boost", "precision"},
      str(sorted(d)))
import json  # noqa: E402
check("as_dict is JSON-serialisable (it goes to events.jsonl)",
      json.loads(json.dumps(d)) == d)
check("`full` describes itself as unshaped",
      "unshaped" in teleop.Shaping.preset("full").describe(),
      teleop.Shaping.preset("full").describe())
check("`slow` names its numbers",
      "expo" in teleop.Shaping.preset("slow").describe(),
      teleop.Shaping.preset("slow").describe())


# -- 6. the drag speed estimate reproduces the VQ1 truth medians -----------------------
# Calibration data at teleop.DRAG_K: true horizontal speed vs measured |(ax,ay)|,
# 35 203 in-flight IMU samples on the VQ1 build. Reproduced here as a regression on
# the constant -- if someone retunes DRAG_K, this says what it has to keep matching.
BANDS = [(2, 0.19), (3, 0.38), (4, 0.68), (5, 1.04), (6, 1.65), (7, 1.99), (8, 2.60)]
worst = 0.0
for v_true, a_med in BANDS:
    v_est = teleop.speed_from_drag((a_med, 0.0, -teleop.G))
    worst = max(worst, abs(v_est - v_true) / v_true)
check("drag speed estimate is within 15% of VQ1 truth from 2 to 8 m/s",
      worst < 0.15, "worst %.1f%%" % (100 * worst))
check("no accelerometer sample -> no speed", teleop.speed_from_drag(None) is None)
check("the parked pad reads as fast, which is why the HUD gates on armed",
      teleop.speed_from_drag((teleop.G * math.sin(math.radians(17.8)), 0.0, 0.0))
      > teleop.SPEED_MAP_BAD - 1.0,
      "%.1f m/s parked" % teleop.speed_from_drag(
          (teleop.G * math.sin(math.radians(17.8)), 0.0, 0.0)))
check("flight_quality is blank while disarmed",
      teleop.flight_quality({"armed": False, "accel": (3.0, 0.0, -9.0)}, 0.0)
      == ("", ""))

# -- 7. steadiness gate ----------------------------------------------------------------
check("level and still is steady", teleop.is_steady((0.0, 0.0, -teleop.G)))
check("accelerating is not steady", not teleop.is_steady((4.0, 0.0, -teleop.G)))
check("a hard pull is not steady", not teleop.is_steady((0.0, 0.0, -2 * teleop.G)))
check("banked but not accelerating is steady",
      teleop.is_steady((0.0, teleop.G * 0.5, -teleop.G * math.sqrt(0.75))),
      "|a| = g at any attitude - this measures acceleration, not level")

print()
print("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
