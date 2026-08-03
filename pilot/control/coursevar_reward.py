"""ROUTE_REWARD_SPEC section 4, implemented against a spline route and measured.

    python coursevar_reward.py

The reward under test, per step:

    s_t   = argmin over s in [s_{t-1}, s_{t-1} + WINDOW] of |route(s) - p_t|
    ds    = clip(s_t - s_{t-1}, 0, v_max*dt)
    dperp = |route(s_t) - p_t|
    r     = k_p * ds * exp(-(dperp / w)^2)

This module trains nothing. It flies ANALYTIC trajectories through the reward -- the
route itself at various speeds, corner-cutting lines, offset lines -- and reports what
each earns, because a shaping term is only as good as what it pays for.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "course"))
import coursevar as C                                            # noqa: E402
import course_vq2 as cv                                          # noqa: E402

DT = 1.0 / 55.0            # surrogate/env.py:337 -- the decision rate, NOT the spec's 0.02
WINDOW_M = 5.0
V_MAX = 8.0
W_CORRIDOR = 2.5           # max(2.5, 2*sigma_xy_local); sigma median 0.52 so 2.5 binds
K_P = 0.09 / (V_MAX * DT)  # scaled so nominal pace matches the current ~0.09/step


class RouteReward:
    """Windowed monotone projection onto an arc-length polyline."""

    def __init__(self, route, k_p=K_P, w=W_CORRIDOR, v_max=V_MAX, dt=DT,
                 window_m=WINDOW_M, clip_mult=1.0):
        self.rt, self.k_p, self.w = route, k_p, w
        self.max_ds = clip_mult * v_max * dt
        self.window = window_m
        self.s_i = 0

    def reset(self, p):
        self.s_i = int(np.argmin(np.linalg.norm(self.rt.poly - p, axis=1)))
        return self.s_i

    def step(self, p):
        hi = min(self.s_i + int(self.window / C.SPACING) + 1, len(self.rt.poly))
        seg = self.rt.poly[self.s_i:hi]
        j = self.s_i + int(np.argmin(np.linalg.norm(seg - p, axis=1)))
        ds = min(max((j - self.s_i) * C.SPACING, 0.0), self.max_ds)
        dperp = float(np.linalg.norm(self.rt.poly[j] - p))
        self.s_i = j
        return self.k_p * ds * math.exp(-(dperp / self.w) ** 2), ds, dperp


class RouteRewardContinuous(RouteReward):
    """Same reward, but s is projected onto the line SEGMENTS, not snapped to vertices.

    The spec's `arg min over s` is continuous in s; a literal implementation over a
    0.25 m polyline is not. At 8 m/s a step covers 0.145 m, so a vertex-snapped
    projection advances 0 or 1 index and Ds becomes a multiple of 0.25 m -- which the
    v_max*dt clip then truncates to 0.145. Every advance pays exactly the cap, so the
    lap total collapses to (number of route points) * cap * k_p: identical at 6, 8, 10
    and 12 m/s, and no longer a function of progress at all. Projecting onto segments
    restores a continuous s and with it the actual progress signal.
    """

    def reset(self, p):
        super().reset(p)
        self.s = self.s_i * C.SPACING
        return self.s

    def step(self, p):
        lo = max(int(self.s / C.SPACING), 0)
        hi = min(lo + int(self.window / C.SPACING) + 2, len(self.rt.poly))
        a = self.rt.poly[lo:hi - 1]
        b = self.rt.poly[lo + 1:hi]
        ab = b - a
        L2 = np.sum(ab * ab, axis=1) + 1e-12
        t = np.clip(np.sum((p - a) * ab, axis=1) / L2, 0.0, 1.0)
        foot = a + t[:, None] * ab
        d = np.linalg.norm(foot - p, axis=1)
        k = int(np.argmin(d))
        s_new = max((lo + k + t[k]) * C.SPACING, self.s)
        ds = min(s_new - self.s, self.max_ds)
        self.s = self.s + ds
        return (self.k_p * ds * math.exp(-(d[k] / self.w) ** 2), ds, float(d[k]))


def _offset_line(route, kappa, inward_m):
    """The route displaced toward its own centre of curvature by `inward_m`.

    This is what cutting a corner looks like: unchanged on the straights (no curvature,
    no displacement) and progressively inside the line as the corner tightens.
    """
    d1 = np.gradient(route.poly, axis=0)
    d2 = np.gradient(d1, axis=0)
    t = d1 / (np.linalg.norm(d1, axis=1)[:, None] + 1e-12)
    n = d2 - (np.sum(d2 * t, axis=1)[:, None] * t)
    nn = np.linalg.norm(n, axis=1)[:, None]
    n = np.where(nn > 1e-9, n / (nn + 1e-12), 0.0)
    frac = np.clip(kappa / max(float(np.nanmax(kappa)), 1e-9), 0.0, 1.0)[:, None]
    return route.poly + n * frac * inward_m


def fly(route, path, speed_mps, cls=RouteReward, **kw):
    """Walk `path` at constant ground speed, scoring every step."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    n = int(s[-1] / (speed_mps * DT))
    q = np.linspace(0.0, s[-1], max(n, 2))
    pts = np.stack([np.interp(q, s, path[:, k]) for k in range(3)], axis=1)

    rw = cls(route, **kw)
    rw.reset(pts[0])
    tot = clipped = 0.0
    per_step, dperps = [], []
    for p in pts[1:]:
        r, ds, dp = rw.step(p)
        tot += r
        per_step.append(r)
        dperps.append(dp)
        if ds >= rw.max_ds - 1e-12:
            clipped += 1
    return dict(total=tot, steps=len(per_step), mean=float(np.mean(per_step)),
                clip_frac=clipped / max(len(per_step), 1),
                dperp_max=float(np.max(dperps)), dist_m=float(s[-1]),
                time_s=float(s[-1] / speed_mps))


def main():
    course = cv.load()
    route, kappa = C.spline_route(course)
    print("route %.1f m | min turn radius %.2f m | k_p %.2f | corridor w %.1f m | "
          "dt 1/55 s" % (route.length_m, C.min_turn_radius(route), K_P, W_CORRIDOR))

    print("\n1. MAGNITUDE -- does it land on the ~0.09/step the current term produces?")
    for v in (6.0, 8.0, 10.0, 12.0, 14.0):
        r = fly(route, route.poly, v)
        print("   %4.1f m/s: %.4f/step   lap total %6.1f   Ds clipped %3.0f%% of steps"
              % (v, r["mean"], r["total"], 100 * r["clip_frac"]))

    print("\n2. SPEED INCENTIVE -- same, with the clip lifted to 3x v_max*dt")
    for v in (6.0, 8.0, 10.0, 12.0, 14.0):
        r = fly(route, route.poly, v, clip_mult=3.0)
        print("   %4.1f m/s: %.4f/step   lap total %6.1f" % (v, r["mean"], r["total"]))

    print("\n3. CORNER CUTTING -- route displaced toward its centre of curvature")
    base = fly(route, route.poly, 8.0)
    print("   on the line     : lap total %6.1f  (%.1f m flown)"
          % (base["total"], base["dist_m"]))
    for d in (0.5, 1.0, 1.5, 2.0):
        r = fly(route, _offset_line(route, kappa, d), 8.0)
        print("   cutting %.1f m in : lap total %6.1f  (%+5.1f%%)  %.1f m flown  "
              "max off-route %.2f m"
              % (d, r["total"], 100 * (r["total"] / base["total"] - 1),
                 r["dist_m"], r["dperp_max"]))

    print("\n4. CORRIDOR WIDTH -- what w stops cutting 1.5 m from paying?")
    cut = _offset_line(route, kappa, 1.5)
    for w in (2.5, 2.0, 1.5, 1.2, 1.0, 0.8):
        b = fly(route, route.poly, 8.0, w=w)
        r = fly(route, cut, 8.0, w=w)
        print("   w %.1f m: on-line %6.1f  cutting %6.1f  (%+5.1f%%)%s"
              % (w, b["total"], r["total"], 100 * (r["total"] / b["total"] - 1),
                 "   <- cutting no longer pays" if r["total"] < b["total"] else ""))


if __name__ == "__main__":
    main()
