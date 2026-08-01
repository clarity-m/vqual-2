"""
Offline exercise of teleop's thrust channel. No sim, no keyboard, no MAVLink.

What this can prove: the snap fires on release, the held-key ramp is bit-identical
to the pre-snap code, --no-thrust-snap restores the old behaviour exactly, and the
optional IMU tilt compensation falls back to flat hover when the accelerometer is
not reading gravity. What it CANNOT prove: anything about how the aircraft responds.
That needs a flight.

    python3 pilot/test_thrust_snap.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teleop  # noqa: E402


class FakeConn:
    """Stands in for the MAVLink connection: records what would go on the wire."""
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
        self.rows = []

    def row(self, key, *f):
        self.rows.append((key, f))

    def event(self, kind, **p):
        self.events.append((kind, p))


def make(**kw):
    conn = FakeConn()
    rec = FakeRec()
    return teleop.Pilot(conn, rec, 0, **kw), conn, rec


def run(pilot, throttle_axis, ticks, dt=0.02, accel=None):
    """Drive send() with a throttle axis and nothing else. Returns thrust history."""
    out = []
    for _ in range(ticks):
        pilot.send([0.0, 0.0, throttle_axis, 0.0], 1.0, dt, True, accel=accel)
        out.append(pilot.thrust)
    return out


def legacy_thrust(start, axis, ticks, dt=0.02):
    """The pre-snap throttle law, written out so the comparison is against the
    old ALGEBRA rather than against the new code calling itself."""
    t = start
    out = []
    for _ in range(ticks):
        t = min(1.0, max(0.0, t + axis * teleop.THRUST_SLEW * dt))
        out.append(t)
    return out


FAILS = []


def check(name, ok, detail=""):
    print("%-4s %s%s" % ("PASS" if ok else "FAIL", name,
                         ("   " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


# -- 1. held key: identical to the old law, bit for bit ---------------------------
p, _, _ = make()
got = run(p, 1.0, 25)
want = legacy_thrust(teleop.THRUST_START, 1.0, 25)
check("held UP matches the pre-snap law exactly", got == want,
      "%d ticks, end %.6f" % (len(got), got[-1]))

p, _, _ = make()
p.thrust = 0.5
got = run(p, -1.0, 25)
want = legacy_thrust(0.5, -1.0, 25)
check("held DOWN matches the pre-snap law exactly", got == want,
      "end %.6f" % got[-1])

# -- 2. release snaps to hover ----------------------------------------------------
p, _, rec = make()
run(p, 1.0, 30)                       # climb away from hover
high = p.thrust
rel = run(p, 0.0, 3)
check("release snaps to hover on the very next tick",
      rel[0] == p.hover and high > p.hover,
      "%.3f -> %.3f (hover %.3f)" % (high, rel[0], p.hover))
check("snap holds while released", all(v == p.hover for v in rel))
check("one thrust_snap event per release edge",
      [e for e in rec.events if e[0] == "thrust_snap"].__len__() == 1)

# a release from BELOW hover must snap up, not just stop falling
p, _, _ = make()
p.thrust = 0.05
rel = run(p, 0.0, 2)
check("release from below hover snaps UP", rel[0] == p.hover,
      "0.050 -> %.3f" % rel[0])

# -- 3. the throttle is now an offset: press, release, press again ----------------
p, _, rec = make()
run(p, 0.0, 2)                        # settle at hover
run(p, 1.0, 10)
mid = p.thrust
run(p, 0.0, 2)
run(p, 1.0, 10)
check("second press starts again from hover", abs(p.thrust - mid) < 1e-12,
      "%.4f then %.4f" % (mid, p.thrust))
check("two release edges -> two events",
      len([e for e in rec.events if e[0] == "thrust_snap"]) == 2)

# -- 4. --no-thrust-snap is the old behaviour -------------------------------------
p, _, rec = make(thrust_snap=False)
run(p, 1.0, 30)
held = p.thrust
run(p, 0.0, 50)
check("--no-thrust-snap holds where you left it", p.thrust == held,
      "%.4f" % held)
check("--no-thrust-snap emits no snap events",
      not [e for e in rec.events if e[0] == "thrust_snap"])

# -- 5. the dead zone: a barely-moving axis still counts as held ------------------
p, _, _ = make()
p.thrust = 0.6
run(p, teleop.THRUST_STICK_EPS * 0.5, 1)
check("axis inside the dead zone snaps", p.thrust == p.hover)
p, _, _ = make()
p.thrust = 0.6
run(p, teleop.THRUST_STICK_EPS * 2.0, 1)
check("axis outside the dead zone integrates", p.thrust > 0.6)

# -- 6. auto-takeoff still owns the channel ---------------------------------------
p, _, _ = make()
p.arm = lambda *a, **k: None
import time as _t
p.takeoff_until = _t.time() + 1.0
run(p, 0.0, 1)
check("takeoff burst is not overridden by the snap",
      p.thrust == teleop.TAKEOFF_THRUST, "%.2f" % p.thrust)

# -- 7. IMU tilt compensation -----------------------------------------------------
G = teleop.G
level = (0.0, 0.0, -G)
bank30 = (0.0, G * math.sin(math.radians(30)), -G * math.cos(math.radians(30)))
manoeuvre = (6.0, 0.0, -G)            # |a| = 11.5: not gravity, must be rejected

p, _, _ = make(tilt_comp_imu=True)
run(p, 0.0, 100, accel=level)
check("tilt comp OFF-by-default is respected when enabled+level",
      abs(p.hover_ref - p.hover) < 1e-9, "hover_ref %.4f" % p.hover_ref)

p, _, _ = make(tilt_comp_imu=True)
run(p, 0.0, 200, accel=bank30)         # 4 s: well past TILT_IMU_TAU
want = p.hover / math.cos(math.radians(30))
check("30 deg of steady bank raises the target by 1/cos", abs(p.hover_ref - want) < 1e-3,
      "%.4f vs %.4f" % (p.hover_ref, want))

p, _, _ = make(tilt_comp_imu=True)
run(p, 0.0, 200, accel=bank30)
before = p.hover_ref
run(p, 0.0, int(teleop.TILT_IMU_MAX_AGE / 0.02) + 5, accel=manoeuvre)
check("a manoeuvring accelerometer decays back to FLAT hover",
      abs(p.hover_ref - p.hover) < 1e-9,
      "%.4f -> %.4f (flat %.3f)" % (before, p.hover_ref, p.hover))

p, _, _ = make(tilt_comp_imu=True)
run(p, 0.0, 200, accel=(0.0, G * math.sin(math.radians(80)),
                        -G * math.cos(math.radians(80))))
check("extreme tilt is clamped at TILT_COMP_MAX",
      abs(p.hover_ref - p.hover * teleop.TILT_COMP_MAX) < 1e-6,
      "%.4f" % p.hover_ref)

p, _, _ = make()                       # flag off
run(p, 0.0, 200, accel=bank30)
check("tilt comp does nothing unless the flag is set", p.hover_ref == p.hover)

# -- 8. hover_ref does not latch after the levelling assist drops out -------------
p, _, _ = make()
truth = (math.radians(35), 0.0, 0.0)
p.send([0, 0, 0, 0], 1.0, 0.02, True, level=True,
       truth_att=(truth[0], truth[1], _t.time()))
banked = p.hover_ref
p.send([0, 0, 0, 0], 1.0, 0.02, True, level=False, truth_att=None)
check("hover_ref returns to flat when levelling ends",
      banked > p.hover and p.hover_ref == p.hover,
      "%.4f -> %.4f" % (banked, p.hover_ref))

# -- 9. what actually goes on the wire --------------------------------------------
p, conn, _ = make()
run(p, 0.0, 3)
check("the snapped value is what is transmitted",
      all(s[3] == p.hover for s in conn.sent),
      "last %.4f" % conn.sent[-1][3])
check("thrust stays inside 0..1 for every tick",
      all(0.0 <= s[3] <= 1.0 for s in conn.sent))

print()
print("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
