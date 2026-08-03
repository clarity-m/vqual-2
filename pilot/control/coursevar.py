"""coursevar -- VQ2 course variations and their solved reference routes.

    from coursevar import deform, spline_route, generate

    var    = deform(seed=7)                # one displaced course + what moved
    rt, k  = spline_route(var.course)      # the smooth path the gates sit on
    bank   = generate(400)                 # a few hundred, auto-filtered

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


def _catmull_rom(P, per_seg=60, alpha=0.5):
    """Centripetal Catmull-Rom through every point in P.

    Interpolating, so the curve passes through each gate centre exactly. Centripetal
    parameterization (alpha = 0.5) rather than uniform: uniform Catmull-Rom forms
    cusps and self-intersecting loops exactly where the control points turn sharply,
    which on this course is gates 7, 9 and 13 -- the corners that matter most.

    Ends are extrapolated by reflecting the first and last segment, so the curve
    starts and ends with the heading the course implies rather than a phantom corner.
    """
    P = np.asarray(P, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(len(ext) - 3):
        p0, p1, p2, p3 = ext[i:i + 4]
        t = [0.0]
        for a, b in ((p0, p1), (p1, p2), (p2, p3)):
            t.append(t[-1] + max(float(np.linalg.norm(b - a)), 1e-9) ** alpha)
        t0, t1, t2, t3 = t
        tt = np.linspace(t1, t2, per_seg, endpoint=(i == len(ext) - 4))
        a1 = ((t1 - tt)[:, None] * p0 + (tt - t0)[:, None] * p1) / (t1 - t0)
        a2 = ((t2 - tt)[:, None] * p1 + (tt - t1)[:, None] * p2) / (t2 - t1)
        a3 = ((t3 - tt)[:, None] * p2 + (tt - t2)[:, None] * p3) / (t3 - t2)
        b1 = ((t2 - tt)[:, None] * a1 + (tt - t0)[:, None] * a2) / (t2 - t0)
        b2 = ((t3 - tt)[:, None] * a2 + (tt - t1)[:, None] * a3) / (t3 - t1)
        out.append(((t2 - tt)[:, None] * b1 + (tt - t1)[:, None] * b2) / (t2 - t1))
    return np.vstack(out)


def _speed_profile(poly, v_max, a_lat, a_long):
    """Curvature-limited speed, then forward/backward passes for longitudinal grip.

    v <= sqrt(a_lat / kappa) is the cornering limit; the two passes then enforce that
    the aircraft can actually brake into a corner and accelerate out of it. This is
    what makes the route a trajectory a drone could fly rather than a drawing.
    """
    d1 = np.gradient(poly, axis=0)
    d2 = np.gradient(d1, axis=0)
    kappa = (np.linalg.norm(np.cross(d1, d2), axis=1)
             / (np.linalg.norm(d1, axis=1) ** 3 + 1e-12))
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(kappa, 1e-9)))
    ds = np.concatenate([[0.0], np.linalg.norm(np.diff(poly, axis=0), axis=1)])
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_long * ds[i]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_long * ds[i + 1]))
    return v, kappa


def spline_route(course, v_max=8.0, a_lat=12.0, a_long=8.0, spacing=SPACING,
                 runout_m=6.0, yaw_fallback="bisector"):
    """The reference route as a smooth spline the gates sit on.

    Replaces the pursuit ODE of ROUTE_REWARD_SPEC section 3. That generator drove a
    waypoint pair (approach, exit) offset along each gate normal, which turns every
    gate into a VERTEX: the path arrives, stops turning, and leaves. On this course
    that produced hairpins at 6-7-8 sharp enough to read as doubling back, and no
    racing line does that. A spline turns ABOUT the gates instead of at them, so a
    corner is an arc with an entry, an apex and an exit.

    The gate centres are interpolated exactly, so the crossing distance is zero by
    construction and the binding physical question becomes the crossing ANGLE -- see
    `gate_crossings`.
    """
    P = np.asarray(course.positions, float)
    n0 = cv.gate_normal(0, course, yaw_fallback)
    n1 = cv.gate_normal(N_GATES - 1, course, yaw_fallback)
    pts = np.vstack([P[0] - runout_m * n0, P, P[-1] + runout_m * n1])

    dense = _catmull_rom(pts)
    s = _arc(dense)
    q = np.arange(0.0, s[-1], spacing)
    poly = np.stack([np.interp(q, s, dense[:, k]) for k in range(3)], axis=1)
    v, kappa = _speed_profile(poly, v_max, a_lat, a_long)
    return Route(poly, q, np.zeros(N_GATES), True, speed=v), kappa


def gate_crossings(course, route, yaw_fallback="bisector"):
    """Per gate: how obliquely the route crosses its plane, and what aperture is left.

    A gate is a square hole. Crossing its plane at angle `theta` off the normal
    narrows the usable opening to `inner_m * cos(theta)`, so this -- not distance
    from centre -- is what decides whether a smooth line physically fits through.
    """
    P = np.asarray(course.positions, float)
    ang, eff = np.zeros(N_GATES), np.zeros(N_GATES)
    tan = np.gradient(route.poly, axis=0)
    tan /= np.linalg.norm(tan, axis=1)[:, None] + 1e-12
    for g in range(N_GATES):
        j = int(np.argmin(np.linalg.norm(route.poly - P[g], axis=1)))
        n = cv.gate_normal(g, course, yaw_fallback)
        c = abs(float(tan[j] @ n))
        ang[g] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        eff[g] = course.inner_m * c
    return ang, eff


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
# Turn limits. The sharpest corner on the MEASURED course is 76.5 deg at gate 13, so
# a variation carrying a 120 deg hairpin is not a deformation of this course -- it is
# a different course, and one no racing line flies. Two bounds, both needed:
#   MAX_TURN_DEG  absolute: past 90 deg the route doubles back on itself.
#   TURN_MARGIN   per gate, against the nominal turn AT THAT GATE, so a corner that is
#                 straight on the real course cannot become a corner here.
# An earlier version of this filter used a single absolute cap of 150 deg, which is
# above anything the deformation can produce: it passed 63 of 120 courses containing a
# turn past 90 deg. Bounds have to sit inside the distribution to filter anything.
MAX_TURN_DEG = 90.0
TURN_MARGIN_DEG = 20.0


def turn_angles(P):
    """Per-gate heading change, degrees. 0 is straight through, 180 a full reversal."""
    P = np.asarray(P, float)
    t = np.zeros(N_GATES)
    for g in range(1, N_GATES - 1):
        u, v = P[g] - P[g - 1], P[g + 1] - P[g]
        c = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        t[g] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return t


_NOMINAL_TURNS = None


def nominal_turns():
    global _NOMINAL_TURNS
    if _NOMINAL_TURNS is None:
        _NOMINAL_TURNS = turn_angles(cv.load().positions)
    return _NOMINAL_TURNS


def geometry_faults(course):
    """Automatic rejects -- the things no human should have to spot by eye.

    Deliberately NOT a full shape test. Whether a variation is still recognisably the
    VQ2 course is the judgement being asked of the reviewer; this removes only the
    ones that are not a flyable course at all, or that manufacture a corner sharper
    than anything the real course has.
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
    t, tn = turn_angles(P), nominal_turns()
    for g in range(1, N_GATES - 1):
        if t[g] > MAX_TURN_DEG:
            f.append("gate %d turns %.0f deg" % (g, t[g]))
        elif t[g] - tn[g] > TURN_MARGIN_DEG:
            f.append("gate %d turns %.0f deg vs %.0f nominal" % (g, t[g], tn[g]))
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
        rt, _k = spline_route(var.course)
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


def min_turn_radius(route):
    """Tightest radius anywhere on the route, metres. The number that says whether a
    corner reads as a turn or as a hairpin."""
    if len(route.poly) < 5:
        return float("inf")
    d1 = np.gradient(route.poly, axis=0)
    d2 = np.gradient(d1, axis=0)
    k = (np.linalg.norm(np.cross(d1, d2), axis=1)
         / (np.linalg.norm(d1, axis=1) ** 3 + 1e-12))
    return float(1.0 / max(float(np.nanmax(k)), 1e-9))


def max_lateral_accel(route):
    """max kappa * v^2 along the polyline -- the spec's dynamic-feasibility check."""
    if len(route.poly) < 5:
        return 0.0
    d1 = np.gradient(route.poly, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.linalg.norm(np.cross(d1, d2), axis=1)
    den = np.linalg.norm(d1, axis=1) ** 3 + 1e-12
    return float(np.nanmax(num / den * route.speed ** 2))
