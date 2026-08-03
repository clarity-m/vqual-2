"""Render course variations as one PNG each, for eyeball review.

    python coursevar_sheet.py --n 300 --out variations

One image per variation, named `v000.png` ... so the folder previews in order.
Each image carries the NOMINAL course as a ghost underlay, because the judgement
being asked is "is this still the VQ2 course" and that needs the map in frame.

The route is the loud element on purpose: every variation solves to the same
general path, so a bad deformation shows up as a route that kinks, doubles back,
or stops looking like the lap.

PLAN VIEW ORIENTATION. course_vq2's +x points BACK toward the start and the race
runs toward -x, so plotting raw x would draw the lap right-to-left. Screen x is
-world_x and screen y is +world_y: two mirrors, so handedness survives and a
right-hand turn on the course is a right-hand turn on the page.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "course"))
import coursevar as C                                            # noqa: E402
import course_vq2 as cv                                          # noqa: E402

GHOST = "#9aa3a0"
INK = "#1b2124"
ROUTE = "#1a7f6b"
MOVED = "#b5502f"
PAPER = "#f4f6f5"


def _plan(P):
    return -np.asarray(P)[:, 0], np.asarray(P)[:, 1]


def _side(P):
    return -np.asarray(P)[:, 0], np.asarray(P)[:, 2]


def frame(kept, nom_route, pad=6.0):
    """One bounding box for every image.

    Per-image autoscaling would silently rescale each plot to its own extent, so
    two variations of different size would draw the same on the page -- which is
    exactly the comparison the reviewer is being asked to make. Fixed limits mean
    a gate that moved 8 m LOOKS like it moved 8 m, in every frame.
    """
    pts = [nom_route.poly, np.asarray(cv.load().positions, float)]
    for var, rt in kept:
        pts.append(np.asarray(var.course.positions, float))
        pts.append(rt.poly)
    A = np.concatenate(pts, axis=0)
    x, y = -A[:, 0], A[:, 1]
    return (float(x.min() - pad), float(x.max() + pad),
            float(y.min() - pad), float(y.max() + pad),
            float(A[:, 2].min() - 3.0), float(A[:, 2].max() + 3.0))


def render(var, route, nom_route, path, box, index=None):
    ang, _eff = C.gate_crossings(var.course, route)
    P = np.asarray(var.course.positions, float)
    Q = np.asarray(var.nominal, float)
    x0, x1, y0, y1, z0, z1 = box

    # Size the canvas to the course, not the other way round: the lap is ~5x as
    # long as it is wide, so a square figure spends most of its pixels on empty
    # hangar and squashes the shape into a band too thin to judge.
    w_in = 13.0
    span = max(x1 - x0, 1e-6)
    h_plan = w_in * (y1 - y0) / span
    h_side = w_in * (z1 - z0) / span
    fig = plt.figure(figsize=(w_in, h_plan + h_side + 1.15), dpi=100, facecolor=PAPER)
    gs = fig.add_gridspec(2, 1, height_ratios=[h_plan, h_side], hspace=0.10,
                          left=0.035, right=0.99,
                          top=1.0 - 0.85 / (h_plan + h_side + 1.15), bottom=0.045)
    ax = fig.add_subplot(gs[0], facecolor=PAPER)
    az = fig.add_subplot(gs[1], facecolor=PAPER)

    # -- reference map: nominal chain + nominal route ----------------------
    for a, f in ((ax, _plan), (az, _side)):
        gx, gy = f(Q)
        a.plot(gx, gy, "-", color=GHOST, lw=1.0, alpha=0.55, zorder=1)
        a.plot(gx, gy, "o", color=GHOST, ms=3.4, alpha=0.7, zorder=1)
    rx, ry = _plan(nom_route.poly)
    ax.plot(rx, ry, "-", color=GHOST, lw=1.6, alpha=0.5, zorder=2)
    rx, ry = _side(nom_route.poly)
    az.plot(rx, ry, "-", color=GHOST, lw=1.3, alpha=0.5, zorder=2)

    # -- this variation ----------------------------------------------------
    rx, ry = _plan(route.poly)
    ax.plot(rx, ry, "-", color=ROUTE, lw=3.4, solid_capstyle="round", zorder=5)
    rx, ry = _side(route.poly)
    az.plot(rx, ry, "-", color=ROUTE, lw=2.2, solid_capstyle="round", zorder=5)

    for a, f in ((ax, _plan), (az, _side)):
        gx, gy = f(P)
        a.plot(gx, gy, "o", color=INK, ms=5.2, zorder=6)
        qx, qy = f(Q)
        for g in var.moved:                       # leader from map to displaced
            a.plot([qx[g], gx[g]], [qy[g], gy[g]], "-", color=MOVED, lw=1.3,
                   alpha=0.9, zorder=7)
            a.plot([gx[g]], [gy[g]], "o", color=MOVED, ms=8.0, zorder=8)

    gx, gy = _plan(P)
    for g in range(cv.N_GATES):
        ax.annotate(str(g), (gx[g], gy[g]), textcoords="offset points",
                    xytext=(7, 5), fontsize=8.5, color=INK, zorder=9,
                    fontfamily="monospace")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    az.set_xlim(x0, x1)
    az.set_ylim(z0, z1)
    ax.set_aspect("equal", adjustable="box")
    az.set_aspect("equal", adjustable="box")
    for a in (ax, az):
        a.grid(True, color=GHOST, alpha=0.18, lw=0.6)
        a.tick_params(labelsize=7, colors=GHOST, length=2)
        for sp in a.spines.values():
            sp.set_color(GHOST)
            sp.set_alpha(0.4)
    ax.set_xticklabels([])
    az.set_ylabel("z, m", fontsize=8, color=GHOST)

    tag = "" if index is None else "v%03d   " % index
    moved = " ".join("%d(%.1fm)" % (g, s) for g, s in zip(var.moved, var.shift_m))
    fig.text(0.035, 0.988, "%sseed %d" % (tag, var.seed), fontsize=13,
             color=INK, fontweight="bold", fontfamily="monospace", va="top")
    fig.text(0.035, 0.958, "moved  %s" % moved, fontsize=9.5, color=MOVED,
             fontfamily="monospace", va="top")
    fig.text(0.990, 0.988,
             "route %.0f m   min turn radius %.1f m   sharpest turn %.0f°   "
             "tightest gate %d at %.0f° off normal"
             % (route.length_m, C.min_turn_radius(route),
                float(np.max(C.turn_angles(var.course.positions))),
                int(np.argmax(ang)), float(np.max(ang))),
             fontsize=9, color=GHOST, fontfamily="monospace", va="top", ha="right")
    ax.legend(handles=[Line2D([], [], color=GHOST, lw=1.6, label="VQ2 as mapped"),
                       Line2D([], [], color=ROUTE, lw=3.0, label="this variation"),
                       Line2D([], [], color=MOVED, lw=1.6, marker="o",
                              label="displaced gate")],
              loc="upper left", fontsize=8.5, framealpha=0.9, facecolor=PAPER,
              edgecolor=GHOST)

    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(_HERE, "variations"))
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--sort", choices=["seed", "shift"], default="shift",
                    help="'shift' puts the most-displaced last, so the marginal "
                         "cases arrive together and the boundary is easy to find")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print("generating %d variations ..." % a.n)
    kept, stats = C.generate(a.n, seed0=a.seed0, progress=max(a.n // 8, 1))
    print("  tried %d | kept %d | geometry-reject %d | route-reject %d"
          % (stats["tried"], stats["kept"], stats["geometry"], stats["route"]))
    if stats["route_fail_gate"]:
        print("  route failures by gate:", dict(sorted(stats["route_fail_gate"].items())))

    nom_route, _k = C.spline_route(cv.load())
    if a.sort == "shift":
        kept.sort(key=lambda k: float(np.max(k[0].shift_m)))
    box = frame(kept, nom_route)

    print("rendering to %s ..." % a.out)
    for i, (var, rt) in enumerate(kept):
        render(var, rt, nom_route, os.path.join(a.out, "v%03d.png" % i), box, index=i)
        if (i + 1) % 25 == 0:
            print("  %d/%d" % (i + 1, len(kept)))

    shifts = np.array([float(np.max(v.shift_m)) for v, _ in kept])
    print("\ndone. %d images in %s" % (len(kept), a.out))
    print("sorted by max displacement, %.1f m (v000) -> %.1f m (v%03d)"
          % (shifts[0], shifts[-1], len(kept) - 1))


if __name__ == "__main__":
    main()
