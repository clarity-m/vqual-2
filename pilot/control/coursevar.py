"""coursevar -- VQ2 course variations and their solved reference routes.

    from coursevar import deform, solve_route, generate

    var  = deform(seed=7)                  # one displaced course + what moved
    rt   = solve_route(var.course)         # the reference path through it
    bank = generate(400)                   # a few hundred, auto-filtered

PURPOSE. We intend the policy to LEARN THIS LAP. These variations are not samples
from the map's measurement posterior -- they are a deliberate tolerance envelope
around a course we are going to memorize, so the memory survives the map being
wrong by more than the map thinks it is. Displacements run 2-8 m against a median
per-gate sigma_xy_local of 0.52 m, i.e. 4-15 sigma. That is over-provision on
purpose and it is why this module does NOT re-solve the edge chain: solving would
drag the drawn displacement back toward what the braced loops allow, which is
exactly the constraint we are trying to fly outside of.

The 6-7 quadrant flip is deliberately NOT modelled here. See ROUTE_REWARD_SPEC.md
section 2.2 -- and note that its prescribed mechanism (rotating edge 6-7's bearing
alone) translates the back half by 22 m rather than rotating it, so it never did
what it claimed.

numpy only.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "course"))
import course_vq2 as cv                                          # noqa: E402

N_GATES = cv.N_GATES


# =========================================================================
# deformation
# =========================================================================

@dataclass
class Variation:
    """One displaced course, plus the record of what was moved."""

    course: object                     # cv.Course, with positions displaced
    seed: int
    moved: np.ndarray                  # gate indices that were displaced
    delta: np.ndarray                  # (n_moved, 3) displacement applied, metres
    nominal: np.ndarray                # (17, 3) the undeformed positions

    @property
    def shift_m(self):
        return np.linalg.norm(self.delta, axis=1)


def deform(seed=0, n_move=(3, 6), mag_m=(2.0, 8.0), elev_deg=20.0, base=None):
    """Displace `n_move` randomly chosen gates by `mag_m` each, independently.

    Gate 0 is never moved: it is the frame origin and the start line, so moving it
    would translate the whole course rather than deform it.

    Direction is a random azimuth with elevation limited to +-`elev_deg` -- the
    hangar is far wider than it is tall, and an 8 m vertical displacement would put
    gates through the floor or the ceiling rather than testing anything.
    """
    rng = np.random.default_rng(seed)
    c = cv.load() if base is None else base
    P = np.array(c.positions, float)
    nominal = P.copy()

    k = int(rng.integers(n_move[0], n_move[1] + 1))
    moved = rng.choice(np.arange(1, N_GATES), size=k, replace=False)
    moved.sort()

    az = rng.uniform(0.0, 2.0 * math.pi, size=k)
    el = np.radians(rng.uniform(-elev_deg, elev_deg, size=k))
    mag = rng.uniform(mag_m[0], mag_m[1], size=k)
    delta = np.stack([mag * np.cos(el) * np.cos(az),
                      mag * np.cos(el) * np.sin(az),
                      mag * np.sin(el)], axis=1)
    P[moved] += delta

    out = cv.Course(positions=P, yaw_deg=list(c.yaw_deg), tilt_deg=list(c.tilt_deg),
                    tilt_lean_deg=list(c.tilt_lean_deg), race_order=list(c.race_order),
                    inner_m=c.inner_m, outer_m=c.outer_m, raw=c.raw, seed=seed)
    return Variation(course=out, seed=seed, moved=moved, delta=delta, nominal=nominal)


# =========================================================================
# the reference route
# =========================================================================

@dataclass
class Route:
    poly: np.ndarray                   # (M, 3) arc-length polyline at `SPACING`
    s: np.ndarray                      # (M,) arc length
    miss: np.ndarray                   # (17,) crossing distance from each gate centre
    ok: bool
    fail_gate: int = -1
    fail_why: str = ""
    speed: np.ndarray = field(default=None, repr=False)

    @property
    def length_m(self):
        return float(self.s[-1]) if len(self.s) else 0.0


SPACING = 0.25


def solve_route(course, wn=1.5, zeta=1.0, L=3.0, a_max=12.0, v_max=8.0, dt=0.02,
                reach_tol=0.5, budget_s=25.0, yaw_fallback="bisector"):
    """Critically damped pursuit of gate approach/exit waypoints. ROUTE_REWARD_SPEC
    section 3, with the two corrections that make it terminate.

    FIX 1 -- the switch needs hysteresis. The spec switches from the approach point
    to the punch-through point when "the approach-side distance along n exceeds L"
    stops holding. A critically damped tracker converges on the approach point from
    below WITHOUT overshoot, so that distance tends to L from above and the switch
    never fires: as written the generator deadlocks at gate 0 forever. Here the
    approach phase ends when the drone is within `reach_tol` of the waypoint.

    FIX 2 -- the approach phase must be STICKY. Gate 7's measured plane sits 85 deg
    off the incoming 6->7 leg, so the route reaches that plane side-on and crosses it
    ~11 m from the centre while still flying toward the approach point. Under the
    spec's rule that crossing flips the target to the exit point, which is now behind
    the aircraft, and the route can never come back: it deadlocks at gate 7 on the
    nominal course and ~90% of sampled ones. Raising wn (the spec's retry) moves the
    miss 10.98 -> 10.52 m and lowering v_max makes it worse, because this is geometry,
    not tracking bandwidth. So the approach waypoint is held until it is REACHED,
    whatever the plane does meanwhile, and only then does the route punch through.
    """
    P = np.asarray(course.positions, float)
    nrm = np.array([cv.gate_normal(g, course, yaw_fallback) for g in range(N_GATES)])
    half = 0.5 * course.inner_m

    p = P[0] - (L + 3.0) * nrm[0]
    v = np.zeros(3)
    traj = [p.copy()]
    miss = np.full(N_GATES, np.nan)

    i = 0
    punching = False
    t_gate = 0.0
    while i < N_GATES:
        c, n = P[i], nrm[i]
        wp = c + L * n if punching else c - L * n
        if not punching and float(np.linalg.norm(p - wp)) < reach_tol:
            punching = True
            wp = c + L * n

        a = wn * wn * (wp - p) - 2.0 * zeta * wn * v
        na = float(np.linalg.norm(a))
        if na > a_max:
            a *= a_max / na
        v = v + a * dt
        sp = float(np.linalg.norm(v))
        if sp > v_max:
            v *= v_max / sp
        p_new = p + v * dt

        if punching:
            s0 = float((p - c) @ n)
            s1 = float((p_new - c) @ n)
            if s0 < 0.0 <= s1:
                f = -s0 / max(s1 - s0, 1e-12)
                x = p + f * (p_new - p)
                r = float(np.linalg.norm(x - c))
                if r > half:
                    return Route(np.array(traj), _arc(traj), miss, False, i,
                                 "crossed %.2f m from centre, outside the %.2f m "
                                 "aperture" % (r, half))
                miss[i] = r
                i += 1
                punching = False
                t_gate = 0.0

        p = p_new
        traj.append(p.copy())
        t_gate += dt
        if t_gate > budget_s:
            return Route(np.array(traj), _arc(traj), miss, False, i,
                         "no crossing within %.0f s" % budget_s)

    poly, s, spd = _resample(np.array(traj), dt)
    return Route(poly, s, miss, True, speed=spd)


def _arc(traj):
    t = np.asarray(traj)
    if len(t) < 2:
        return np.zeros(len(t))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))])


def _resample(traj, dt, spacing=SPACING):
    """Arc-length parameterized polyline at `spacing`, plus the speed along it."""
    s = _arc(traj)
    spd = np.concatenate([[0.0], np.linalg.norm(np.diff(traj, axis=0), axis=1) / dt])
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    s, traj, spd = s[keep], traj[keep], spd[keep]
    q = np.arange(0.0, s[-1], spacing)
    poly = np.stack([np.interp(q, s, traj[:, k]) for k in range(3)], axis=1)
    return poly, q, np.interp(q, s, spd)


# =========================================================================
# generation
# =========================================================================

MIN_EDGE_M = 5.0          # a leg shorter than this is not a leg
MIN_SEP_M = 3.0           # two gates closer than this overlap in a 2.7 m frame
MAX_TURN_DEG = 150.0      # a reversal, not a corner


def geometry_faults(course):
    """Automatic rejects -- the things no human should have to spot by eye.

    Deliberately NOT a shape test. Whether a variation is still recognisably the
    VQ2 course is the judgement being asked of the reviewer; this only removes the
    ones that are not a flyable course at all.
    """
    P = np.asarray(course.positions, float)
    f = []
    e = np.linalg.norm(np.diff(P, axis=0), axis=1)
    for g in np.nonzero(e < MIN_EDGE_M)[0]:
        f.append("edge %d-%d only %.1f m" % (g, g + 1, e[g]))
    for a in range(N_GATES):
        for b in range(a + 2, N_GATES):
            d = float(np.linalg.norm(P[a] - P[b]))
            if d < MIN_SEP_M:
                f.append("gates %d and %d %.1f m apart" % (a, b, d))
    for g in range(1, N_GATES - 1):
        u, v = P[g] - P[g - 1], P[g + 1] - P[g]
        cosa = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        turn = math.degrees(math.acos(max(-1.0, min(1.0, cosa))))
        if turn > MAX_TURN_DEG:
            f.append("gate %d turns %.0f deg" % (g, turn))
    return f


def generate(n=400, seed0=0, max_tries=None, progress=None, **kw):
    """`n` variations that pass the geometry filter AND solve a route.

    Returns (kept, stats). Each kept entry is (Variation, Route). Rejects are
    counted by cause so a low yield is diagnosable rather than mysterious.
    """
    kept, stats = [], dict(tried=0, geometry=0, route=0, kept=0, route_fail_gate={})
    max_tries = max_tries or 20 * n
    s = seed0
    while len(kept) < n and stats["tried"] < max_tries:
        stats["tried"] += 1
        var = deform(seed=s, **kw)
        s += 1
        if geometry_faults(var.course):
            stats["geometry"] += 1
            continue
        rt = solve_route(var.course)
        if not rt.ok:
            stats["route"] += 1
            g = int(rt.fail_gate)
            stats["route_fail_gate"][g] = stats["route_fail_gate"].get(g, 0) + 1
            continue
        kept.append((var, rt))
        stats["kept"] += 1
        if progress and len(kept) % progress == 0:
            print("  %d/%d kept (%d tried)" % (len(kept), n, stats["tried"]))
    return kept, stats


def max_lateral_accel(route):
    """max kappa * v^2 along the polyline -- the spec's dynamic-feasibility check."""
    if len(route.poly) < 5:
        return 0.0
    d1 = np.gradient(route.poly, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.linalg.norm(np.cross(d1, d2), axis=1)
    den = np.linalg.norm(d1, axis=1) ** 3 + 1e-12
    return float(np.nanmax(num / den * route.speed ** 2))
