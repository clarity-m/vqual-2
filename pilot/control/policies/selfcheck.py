"""Unit checks for the baseline and the recovery supervisor. No surrogate, no torch.

Hand-built `interface.Observation`s in, signs and clamps out. This exists because the
one failure this project cannot afford is a mirrored control law: it flies, it looks
plausible, and it steers away from every gate. The canonical convention being enforced
here is `interface.py`'s, unmirrored -- the link layer owns the sim's mirror and nothing
above it may contain a sign flip.

    python pilot/control/policies/selfcheck.py

The comparative checks (frame guard, coasting, approach point) assert an ORDERING
between two situations rather than a value, so they survive gain tuning: change
`k_frame` and the number moves, but nose-down-when-low had better still be true.
"""

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot import interface                                         # noqa: E402
from pilot.control.policies import envelope                         # noqa: E402
from pilot.control.policies.baseline import BaselinePolicy          # noqa: E402
from pilot.control.policies.supervisor import RecoverySupervisor    # noqa: E402

HOVER = interface.HOVER_THRUST
SEEN = []          # every action any check produced, re-audited at the end


class Checks:
    def __init__(self):
        self.n = 0
        self.bad = []

    def __call__(self, ok, name, detail=""):
        self.n += 1
        ok = bool(ok)
        if not ok:
            self.bad.append(name)
        print("  %-4s  %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
        return ok

    def head(self, title):
        print("\n=== %s ===" % title)


# -- observation builders ------------------------------------------------------
def gate(pos, normal=None, index=0, valid=True, conf=1.0, stale=0.0,
         pose_valid=True, sigma=0.0):
    g = interface.GateObs()
    g.valid = valid
    g.index = index
    g.pos_body = np.asarray(pos, dtype=float)
    if normal is not None:
        n = np.asarray(normal, dtype=float)
        g.normal_body = n / max(np.linalg.norm(n), 1e-9)
        g.normal_valid = True
    g.pose_valid = pose_valid
    g.range_sigma_m = sigma
    g.confidence = conf
    g.staleness_s = stale
    g.size_px = 60.0
    return g


def ribbon(bearing=0.0, elev=0.0, valid=True, stale=0.0):
    rb = interface.RibbonObs()
    rb.valid = valid
    rb.bearings_rad = np.full(interface.N_RIBBON, float(bearing))
    rb.elevs_rad = np.full(interface.N_RIBBON, float(elev))
    rb.pixel_fraction = 0.015
    rb.staleness_s = stale
    return rb


def obs(gates=(), rb=None, roll=0.0, pitch=0.0, att_conf=1.0, speed=9.0,
        speed_conf=1.0, vel_bearing=None, active=0, t_coll=1e3, episodes=0, dt=0.02):
    o = interface.Observation()
    o.dt_s = dt
    gs = list(gates)
    while len(gs) < interface.N_GATES:
        gs.append(interface.GateObs())
    o.gates = gs[:interface.N_GATES]
    o.ribbon = rb if rb is not None else interface.RibbonObs()
    o.own.roll_rad = roll
    o.own.pitch_rad = pitch
    o.own.attitude_conf = att_conf
    o.own.speed_est_mps = speed
    o.own.speed_conf = speed_conf
    if vel_bearing is not None:
        o.own.vel_bearing_rad = vel_bearing
        o.own.vel_valid = True
    o.race.active_gate_index = active
    o.race.n_gates_total = 20
    o.race.t_since_collision_s = t_coll
    o.race.collision_episodes = episodes
    o.attention.kind = interface.Attn.GATE_CURRENT
    return o


def act(policy, o):
    a = policy(o)
    SEEN.append(a)
    return a


class StubInner(interface.Policy):
    """A distinctive, already-legal action, plus call/reset bookkeeping."""

    FIXED = (0.5, -0.4, 0.6)

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, _o):
        self.calls += 1
        return interface.Action(self.FIXED[0], self.FIXED[1], 0.0, self.FIXED[2],
                                interface.YawMode.AUTO_ATTENTION)


# -- the checks ----------------------------------------------------------------
def check_baseline_steering(C):
    C.head("baseline: horizontal steering (bearing > 0 == target RIGHT of nose)")
    p = BaselinePolicy()

    right = act(p, obs([gate((10.0, 3.0, 0.0))]))
    C(right.roll_rate > 0.1, "gate right -> roll right (positive)",
      "roll_rate %+.3f" % right.roll_rate)

    left = act(p, obs([gate((10.0, -3.0, 0.0))]))
    C(left.roll_rate < -0.1, "gate left -> roll left (negative)",
      "roll_rate %+.3f" % left.roll_rate)

    C(abs(right.roll_rate + left.roll_rate) < 1e-9, "left/right are mirror images",
      "%+.3f vs %+.3f" % (right.roll_rate, left.roll_rate))

    ahead = act(p, obs([gate((10.0, 0.0, 0.0))]))
    C(abs(ahead.roll_rate) < 1e-9, "gate dead ahead -> wings level",
      "roll_rate %+.3f" % ahead.roll_rate)

    # Already banked into the turn: the loop must stop rolling once the bank matches.
    o = obs([gate((10.0, 3.0, 0.0))])
    bank = BaselinePolicy().k_bearing * math.atan2(3.0, 10.0)
    settled = act(p, obs([gate((10.0, 3.0, 0.0))], roll=bank))
    C(abs(settled.roll_rate) < 1e-9, "bank already at the demand -> zero roll rate",
      "bank %.3f rad, roll_rate %+.3f" % (bank, settled.roll_rate))
    C(act(p, obs([gate((10.0, 3.0, 0.0))], roll=2 * bank)).roll_rate < 0,
      "over-banked -> roll back the other way")
    del o


def check_baseline_velocity_referenced(C):
    C.head("baseline: steer on the DRAG bearing, not the nose")
    p = BaselinePolicy()
    # Yaw is a free gimbal and AUTO_ATTENTION drives the nose onto the gate, so a
    # target at bearing 0 says nothing about whether we are translating past it.
    drift = act(p, obs([gate((10.0, 0.0, 0.0))], vel_bearing=+0.30))
    C(drift.roll_rate < -0.1, "target ahead, drifting RIGHT -> bank left",
      "roll_rate %+.3f" % drift.roll_rate)
    drift_l = act(p, obs([gate((10.0, 0.0, 0.0))], vel_bearing=-0.30))
    C(drift_l.roll_rate > 0.1, "target ahead, drifting LEFT -> bank right",
      "roll_rate %+.3f" % drift_l.roll_rate)
    aligned = act(p, obs([gate((10.0, 3.0, 0.0))],
                         vel_bearing=math.atan2(3.0, 10.0)))
    C(abs(aligned.roll_rate) < 1e-9, "already flying at the target -> no bank demand",
      "roll_rate %+.3f" % aligned.roll_rate)
    # vel_valid False (near hover, drag -> 0) falls back to nose-referenced pursuit.
    o = obs([gate((10.0, 3.0, 0.0))])
    o.own.vel_bearing_rad = 1.2      # garbage, and flagged invalid
    C(act(p, o).roll_rate > 0.1, "vel_valid False -> ignore the drag bearing")


def check_baseline_vertical(C):
    C.head("baseline: vertical (thrust climbs, pitch frames)")
    p = BaselinePolicy(k_trim=0.0)     # the trim integrator gets its own section
    level = act(p, obs([gate((10.0, 0.0, 0.0))]))
    above = act(p, obs([gate((10.0, 0.0, -3.0))]))
    below = act(p, obs([gate((10.0, 0.0, 3.0))]))
    C(abs(level.thrust - HOVER) < 1e-9, "level target, level attitude -> HOVER_THRUST",
      "%.4f" % level.thrust)
    C(above.thrust > level.thrust > below.thrust,
      "thrust orders above > level > below",
      "%.3f / %.3f / %.3f" % (above.thrust, level.thrust, below.thrust))

    # elev_rad > 0 means ABOVE the nose; the encoder's own definition.
    g_above = gate((10.0, 0.0, -3.0))
    C(g_above.elev_rad > 0, "interface elev_rad sign (above -> positive)",
      "%.3f rad" % g_above.elev_rad)

    # Horizon referencing: at a nose-down cruise trim a LEVEL target reads high in the
    # body frame. Thrust must not read that as a climb demand.
    th = -0.35
    tilted = act(p, obs([gate((10.0 * math.cos(th), 0.0, 10.0 * math.sin(th)))],
                        pitch=th))
    C(abs(p.last["e_tgt"] - -th) < 1e-9 and abs(p.last["e_horizon"]) < 1e-9
      and abs(p.last["climb"]) < 1e-9,
      "nose-down trim, level target -> no climb demand",
      "body elev %+.2f rad, horizon %+.2f rad, thrust %.4f"
      % (p.last["e_tgt"], p.last["e_horizon"], tilted.thrust))
    C(abs(tilted.thrust - HOVER * p.last["comp"]) < 1e-9,
      "the 1/cos term is all that is left of it",
      "%.4f = %.4f * %.4f" % (tilted.thrust, HOVER, p.last["comp"]))

    # Attitude compensation: banked, the vertical component of thrust drops.
    banked = act(p, obs([gate((10.0, 0.0, 0.0))], roll=0.6))
    C(banked.thrust > level.thrust, "banked -> thrust compensates 1/cos",
      "%.4f vs %.4f" % (banked.thrust, level.thrust))
    unsure = act(p, obs([gate((10.0, 0.0, 0.0))], roll=0.6, att_conf=0.0))
    C(abs(unsure.thrust - HOVER) < 1e-9,
      "attitude_conf 0 -> no compensation from an unusable gravity read",
      "%.4f" % unsure.thrust)


def check_baseline_frame_guard(C):
    C.head("baseline: the -9.4 deg lower frame edge (nose UP is what loses a gate)")
    p = BaselinePolicy()
    mid = act(p, obs([gate((10.0, 0.0, 0.0))]))                 # elev  0.00 rad
    low = act(p, obs([gate((10.0, 0.0, 3.0))]))                 # elev -0.29 rad
    high = act(p, obs([gate((10.0, 0.0, -10.3))]))              # elev +0.80 rad
    C(low.pitch_rate < mid.pitch_rate - 1e-6,
      "target near the low edge -> extra NOSE DOWN",
      "%+.3f vs %+.3f rad/s" % (low.pitch_rate, mid.pitch_rate))
    C(high.pitch_rate > mid.pitch_rate + 1e-6,
      "target near the +49.4 edge -> nose up",
      "%+.3f vs %+.3f rad/s" % (high.pitch_rate, mid.pitch_rate))

    # The identity the guard is built on: elev = -pitch for a target level and ahead.
    th = 0.20
    g = gate((10.0 * math.cos(th), 0.0, 10.0 * math.sin(th)))
    C(abs(g.elev_rad + th) < 1e-9,
      "nose UP by th puts a level target at elev = -th",
      "pitch %+.2f -> elev %+.2f rad" % (th, g.elev_rad))
    edge = math.radians(-9.4)
    C(g.elev_rad < edge, "20 deg nose up drops a level target out of frame",
      "%.3f rad < %.3f rad" % (g.elev_rad, edge))


def check_baseline_speed(C):
    C.head("baseline: speed regulation through pitch")
    p = BaselinePolicy()
    hi = act(p, obs([gate((10.0, 0.0, -2.0))], speed=25.0))
    lo = act(p, obs([gate((10.0, 0.0, -2.0))], speed=0.0))
    C(hi.pitch_rate > 0, "too fast -> nose up (brake)", "%+.3f rad/s" % hi.pitch_rate)
    C(lo.pitch_rate < 0, "too slow -> nose down (accelerate)", "%+.3f rad/s" % lo.pitch_rate)
    C(p.last["v_tgt"] > 0, "speed target positive", "%.2f m/s" % p.last["v_tgt"])

    straight = BaselinePolicy()
    act(straight, obs([gate((10.0, 0.0, 0.0))]))
    v_straight = straight.last["v_tgt"]
    act(straight, obs([gate((10.0, 8.0, 0.0))]))
    v_turn = straight.last["v_tgt"]
    C(v_turn < v_straight, "slow down into a turn", "%.2f vs %.2f m/s" % (v_turn, v_straight))


def check_baseline_approach_point(C):
    C.head("baseline: approach point = pos_body + d * normal_body")
    p = BaselinePolicy()
    centred = act(p, obs([gate((10.0, 0.0, 0.0))]))
    C(abs(centred.roll_rate) < 1e-9, "normal_valid False -> steer at the gate centre")

    # Gate dead ahead but its axis runs off to our left: the approach point (2 m up the
    # camera-facing normal) sits left of centre, so we must swing left to line up.
    left_axis = act(p, obs([gate((10.0, 0.0, 0.0), normal=(-1.0, -0.5, 0.0))]))
    C(left_axis.roll_rate < -0.05, "approach point offsets the steering",
      "roll_rate %+.3f" % left_axis.roll_rate)
    right_axis = act(p, obs([gate((10.0, 0.0, 0.0), normal=(-1.0, 0.5, 0.0))]))
    C(abs(left_axis.roll_rate + right_axis.roll_rate) < 1e-9,
      "and does so symmetrically")

    # A normal pointing AWAY from the camera violates the producer's contract; the
    # approach point would land behind the gate. Ignore it, do not flip it.
    bad = act(p, obs([gate((10.0, 0.0, 0.0), normal=(1.0, 0.5, 0.0))]))
    C(abs(bad.roll_rate) < 1e-9, "normal pointing away is ignored, not flipped")

    # On short final the offset must collapse or it points backwards.
    near = gate((1.0, 0.0, 0.0), normal=(-1.0, -0.5, 0.0))
    tgt = p._approach_point(near)
    C(float(np.linalg.norm(tgt - near.pos_body)) < 1e-9,
      "inside commit range -> d shrinks to zero",
      "|offset| %.3f m" % float(np.linalg.norm(tgt - near.pos_body)))

    # pose_valid False means the range is biased LONG; commit earlier, not later.
    far = gate((4.0, 0.0, 0.0), normal=(-1.0, -0.5, 0.0), pose_valid=False, sigma=2.0)
    d_unsure = float(np.linalg.norm(p._approach_point(far) - far.pos_body))
    sure = gate((4.0, 0.0, 0.0), normal=(-1.0, -0.5, 0.0))
    d_sure = float(np.linalg.norm(p._approach_point(sure) - sure.pos_body))
    C(d_unsure < d_sure, "pose_valid False -> shorter approach offset",
      "%.2f m vs %.2f m" % (d_unsure, d_sure))


def check_baseline_degraded(C):
    C.head("baseline: degraded inputs")
    p = BaselinePolicy()
    fresh = act(p, obs([gate((10.0, 3.0, 0.0), stale=0.0)]))
    coast = act(p, obs([gate((10.0, 3.0, 0.0), stale=0.6)]))
    C(0 < coast.roll_rate < fresh.roll_rate,
      "coasting track -> same sign, less authority",
      "%+.3f vs %+.3f" % (coast.roll_rate, fresh.roll_rate))

    dead = act(p, obs([gate((10.0, 3.0, 0.0), stale=1.5)], roll=0.30))
    C(p.last["cue"] == "blind", "gate past stale_max is not used", p.last["cue"])
    C(dead.roll_rate < 0, "and the wings are levelled", "%+.3f" % dead.roll_rate)

    lowconf = act(p, obs([gate((10.0, 3.0, 0.0), conf=0.0)]))
    C(p.last["cue"] == "blind", "zero-confidence detection is not used")
    del lowconf

    rb = act(p, obs([gate((10.0, 3.0, 0.0), valid=False)], rb=ribbon(bearing=+0.4)))
    C(p.last["cue"] == "ribbon" and rb.roll_rate > 0.1,
      "no gate, ribbon valid -> steer the ribbon", "roll_rate %+.3f" % rb.roll_rate)
    v_ribbon = p.last["v_tgt"]
    act(p, obs([gate((10.0, 0.0, 0.0))]))
    C(v_ribbon < p.last["v_tgt"], "and fly it slower than a gate",
      "%.2f vs %.2f m/s" % (v_ribbon, p.last["v_tgt"]))

    blind = act(p, obs([], roll=0.35, pitch=-0.20, speed=15.0))
    C(p.last["cue"] == "blind" and p.last["v_tgt"] == p.v_min,
      "nothing valid -> slow to v_min", "%.2f m/s" % p.last["v_tgt"])
    C(blind.roll_rate < 0 and blind.pitch_rate > 0,
      "and level out (roll toward 0, nose up from -0.20)",
      "roll %+.3f pitch %+.3f" % (blind.roll_rate, blind.pitch_rate))
    C(abs(blind.thrust - p.last["trim"] * p.last["comp"]) < 1e-9,
      "blind thrust carries no climb term", "%.4f" % blind.thrust)


def check_baseline_trim(C):
    C.head("baseline: the hover-trim integrator")
    above = obs([gate((10.0, 0.0, -3.0))])
    below = obs([gate((10.0, 0.0, 3.0))])
    blind = obs([])

    p = BaselinePolicy()
    C(p.trim == HOVER, "starts at the measured HOVER_THRUST", "%.4f" % p.trim)
    for _ in range(50):
        act(p, above)
    up = p.trim
    C(up > HOVER, "target persistently above -> trim winds UP", "%.4f" % up)
    for _ in range(400):
        act(p, above)
    C(abs(p.trim - p.trim_max) < 1e-9, "and clamps at trim_max", "%.4f" % p.trim)

    q = BaselinePolicy()
    for _ in range(400):
        act(q, below)
    C(abs(q.trim - q.trim_min) < 1e-9, "target persistently below -> clamps at trim_min",
      "%.4f" % q.trim)

    r = BaselinePolicy()
    for _ in range(50):
        act(r, blind)
    C(r.trim == HOVER, "no cue, no integration (nothing to wind up against)",
      "%.4f" % r.trim)
    r.reset()
    C(r.trim == HOVER, "reset restores the trim")

    z = BaselinePolicy(k_trim=0.0)
    for _ in range(50):
        act(z, above)
    C(z.trim == HOVER, "k_trim = 0 disables it entirely")


def check_baseline_gate_choice(C):
    C.head("baseline: which gate is the current one")
    p = BaselinePolicy()
    # Lookahead gate listed first, the active one second: race index decides, not order.
    o = obs([gate((12.0, -6.0, 0.0), index=6), gate((6.0, 2.0, 0.0), index=5)], active=5)
    a = act(p, o)
    C(a.roll_rate > 0, "steers the gate matching active_gate_index",
      "roll_rate %+.3f" % a.roll_rate)

    o = obs([gate((8.0, 3.0, 0.0), index=7)], active=5)
    act(p, o)
    C(p.last["cue"] == "gate", "no exact match -> the nearest gate ahead")

    o = obs([gate((8.0, 3.0, 0.0), index=2)], active=5, rb=ribbon(bearing=-0.3))
    a = act(p, o)
    C(p.last["cue"] == "ribbon" and a.roll_rate < 0,
      "only crossed gates visible -> prefer the ribbon")

    o = obs([gate((8.0, 3.0, 0.0), index=-1)], active=5)
    act(p, o)
    C(p.last["cue"] == "gate", "unknown index -> trust the producer's list order")


def check_supervisor(C):
    C.head("supervisor: trigger, level, hand back")
    inner = StubInner()
    s = RecoverySupervisor(inner)
    fixed = StubInner.FIXED

    a = act(s, obs([gate((10.0, 3.0, 0.0))], t_coll=1e3))
    C(not s.active and (a.roll_rate, a.pitch_rate, a.thrust) == fixed,
      "no collision -> the inner policy is passed straight through")

    a = act(s, obs([], roll=0.40, pitch=-0.30, t_coll=0.01, episodes=1))
    C(s.active, "t_since_collision_s reset -> supervisor takes over")
    C(a.roll_rate < -0.1, "banked right -> negative roll rate levels it",
      "roll %+.2f rad -> %+.3f rad/s" % (0.40, a.roll_rate))
    C(a.pitch_rate > 0.1, "nose down -> positive pitch rate levels it",
      "pitch %+.2f rad -> %+.3f rad/s" % (-0.30, a.pitch_rate))
    C(abs(a.thrust - HOVER) < 1e-9, "hover thrust while recovering", "%.4f" % a.thrust)
    C(a.yaw_mode == interface.YawMode.AUTO_ATTENTION and a.yaw_rate == 0.0,
      "yaw stays with attention")
    C(inner.calls == 1, "inner policy is not consulted while recovering",
      "%d call(s)" % inner.calls)

    act(s, obs([], roll=0.20, pitch=-0.10, t_coll=0.10))
    C(s.active, "still tumbling -> still active")

    act(s, obs([], roll=0.01, pitch=0.01, t_coll=0.20))
    C(s.active, "level but inside the re-acquire window -> still active")

    a = act(s, obs([], roll=0.01, pitch=0.01, t_coll=0.90))
    C(not s.active and (a.roll_rate, a.pitch_rate, a.thrust) == fixed,
      "level and re-acquire elapsed -> hands back")

    # Early handback when attention finds a gate before the timer runs out.
    inner2 = StubInner()
    s2 = RecoverySupervisor(inner2)
    act(s2, obs([], roll=0.5, t_coll=0.01))
    C(s2.active, "second supervisor triggered")
    a = act(s2, obs([gate((10.0, 0.0, 0.0))], roll=0.0, t_coll=0.10))
    C(not s2.active and (a.roll_rate, a.pitch_rate, a.thrust) == fixed,
      "level and a gate is valid again -> early hand back")
    C(inner2.resets >= 2, "inner policy reset on trigger (clears its frame stack)",
      "%d reset(s)" % inner2.resets)

    # The counter is an equivalent trigger when only it is published.
    inner3 = StubInner()
    s3 = RecoverySupervisor(inner3)
    act(s3, obs([], t_coll=4.0, episodes=0))
    act(s3, obs([], roll=0.5, t_coll=4.1, episodes=1))
    C(s3.active, "collision_episodes incrementing also triggers")

    # Failsafe: an attitude that never levels must not strand the aircraft.
    steps = 0
    while s3.active and steps < 200:
        act(s3, obs([], roll=1.2, t_coll=0.01, dt=0.05))
        steps += 1
    C(not s3.active and steps * 0.05 <= s3.max_hold_s + 0.1,
      "max_hold_s failsafe releases a never-levelling recovery",
      "%.2f s" % (steps * 0.05))

    # Extreme attitude, clamped like everything else.
    s4 = RecoverySupervisor(StubInner())
    a = act(s4, obs([], roll=3.0, t_coll=0.0))
    C(abs(a.roll_rate + envelope.RATE_CAP_RPS) < 1e-9,
      "levelling command clamps at the rate cap", "%+.3f rad/s" % a.roll_rate)


def check_envelope(C):
    C.head("envelope: every action produced by every check above")
    p = BaselinePolicy()
    hard = [
        obs([gate((0.5, 8.0, -6.0))], roll=-1.2, pitch=1.0, speed=40.0),
        obs([gate((1.0, -9.0, 9.0), normal=(-1.0, 3.0, -2.0))], roll=1.4, pitch=-1.3,
            speed=0.0, vel_bearing=-2.9),
        obs([gate((30.0, 0.0, 0.0), stale=0.4, conf=0.2, pose_valid=False, sigma=12.0)],
            att_conf=0.0, speed_conf=0.0),
        obs([], rb=ribbon(bearing=3.0, elev=-1.4)),
    ]
    for o in hard:
        act(p, o)

    nan = obs([gate((10.0, 3.0, 0.0))])
    nan.own.speed_est_mps = float("nan")
    a = act(p, nan)
    C(envelope.in_envelope(a), "a NaN in the observation still yields a legal action",
      "roll %+.3f thrust %.3f" % (a.roll_rate, a.thrust))

    bad = [a for a in SEEN if not envelope.in_envelope(a)]
    C(not bad, "all %d actions inside +-%.2f rad/s and [%.2f, 1.0] thrust"
      % (len(SEEN), envelope.RATE_CAP_RPS, envelope.THRUST_FLOOR),
      "%d violation(s)" % len(bad))
    yaw = [a for a in SEEN if a.yaw_rate != 0.0
           or a.yaw_mode != interface.YawMode.AUTO_ATTENTION]
    C(not yaw, "all %d actions are AUTO_ATTENTION with zero yaw rate" % len(SEEN),
      "%d violation(s)" % len(yaw))

    o = obs([gate((10.0, 3.0, -1.0), normal=(-1.0, -0.3, 0.1), stale=0.4)],
            roll=0.2, pitch=-0.3, vel_bearing=0.1)
    v1 = BaselinePolicy()(o).to_vector()
    v2 = BaselinePolicy()(o).to_vector()
    C(np.array_equal(v1, v2), "deterministic: identical observation, identical action")

    try:
        from pilot.control.surrogate import actions as sib
        agree = (abs(sib.RATE_CAP_RPS - envelope.RATE_CAP_RPS) < 1e-12
                 and abs(sib.THRUST_FLOOR - envelope.THRUST_FLOOR) < 1e-12)
        C(agree, "envelope constants match surrogate/actions.py",
          "%.3f / %.3f" % (sib.RATE_CAP_RPS, sib.THRUST_FLOOR))
    except Exception as exc:
        print("  ....  %-46s %s" % ("surrogate/actions.py not importable yet",
                                    type(exc).__name__))


def check_imports(C):
    C.head("imports")
    try:
        from pilot.control.policies import rl
        C(hasattr(rl, "RLPolicy"), "policies.rl imports without a checkpoint present")
    except Exception as exc:
        C(False, "policies.rl imports", "%s: %s" % (type(exc).__name__, exc))
    try:
        from pilot.control.evalsuite import run_policy, select, stress
        C(all(hasattr(m, "main") for m in (run_policy, select, stress)),
          "evalsuite imports without the surrogate present")
    except Exception as exc:
        C(False, "evalsuite imports", "%s: %s" % (type(exc).__name__, exc))


def main():
    print("pilot/control/policies -- selfcheck")
    print("canonical body NED: bearing>0 target RIGHT, elev>0 target ABOVE,")
    print("roll_rate>0 right wing down, pitch_rate>0 nose up. No mirror above the link layer.")
    C = Checks()
    check_baseline_steering(C)
    check_baseline_velocity_referenced(C)
    check_baseline_vertical(C)
    check_baseline_trim(C)
    check_baseline_frame_guard(C)
    check_baseline_speed(C)
    check_baseline_approach_point(C)
    check_baseline_degraded(C)
    check_baseline_gate_choice(C)
    check_supervisor(C)
    check_envelope(C)
    check_imports(C)

    print("\n%d checks, %d failed" % (C.n, len(C.bad)))
    for name in C.bad:
        print("  FAILED: %s" % name)
    print("\n%s" % ("PASS" if not C.bad else "FAIL"))
    return 0 if not C.bad else 1


if __name__ == "__main__":
    sys.exit(main())
