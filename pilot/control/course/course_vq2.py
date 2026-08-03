"""course_vq2 -- the measured VQ2 course geometry, with domain randomization.

    from course_vq2 import load, sample, gate_corners, sanity_check

    course = load()                 # nominal, measured layout
    inst   = sample(seed=7)         # one randomized course, self-consistent
    quad   = gate_corners(4, inst)  # 4 corners of gate 4's 1.5 m aperture, metres

numpy only. No other dependency, no network, no sim.

FRAME (read this before using a coordinate)
    Right-handed, metres. Origin = gate 0 centre.
    +x  along-hangar, pointing BACK toward the start. The race runs 0 -> 16 toward -x.
    +y  across-hangar, toward the column row numbered 21-29; with z up and travel
        along -x, +y is on the RIGHT of the direction of travel.
    +z  up, gravity-referenced.
    The frame is RELATIVE. There is no absolute origin or heading w.r.t. the sim world,
    and the whole course may be rotated by a multiple of 90 deg relative to it. See
    README.md.

WHAT sample() DOES AND DOES NOT MODEL -- see docstring of sample().
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_JSON = os.path.join(_HERE, 'course_vq2.json')

INNER_M = 1.5
OUTER_M = 2.7
N_GATES = 17


@dataclass
class Course:
    """One course instance: nominal, or a domain-randomized sample."""

    positions: np.ndarray                  # (17, 3) metres, frame above
    yaw_deg: list                          # per gate: plane normal azimuth mod 180, or None
    tilt_deg: list                         # per gate: lean out of vertical, degrees
    race_order: list = field(default_factory=lambda: list(range(N_GATES)))
    inner_m: float = INNER_M
    outer_m: float = OUTER_M
    raw: dict = field(default_factory=dict, repr=False)
    seed: int | None = None
    # the edge measurements this instance was actually built from: the measured table
    # for load(), the drawn table for sample(). sanity_check() reconciles against these.
    edge_d_m: np.ndarray | None = None
    edge_bearing_deg: np.ndarray | None = None
    edge_dz_m: np.ndarray | None = None

    @property
    def n(self):
        return len(self.positions)

    def edges(self):
        return self.raw['edges']


# ---------------------------------------------------------------------------------------


def _doc():
    with open(_JSON) as f:
        return json.load(f)


def load(path=None):
    """The nominal measured course. Positions are exactly map_vq2.json's."""
    doc = json.load(open(path)) if path else _doc()
    P = np.array([g['position_m'] for g in doc['gates']], float)
    yaw = [g['yaw_deg'] for g in doc['gates']]
    tilt = [(0.0 if g['tilt_from_vertical_deg'] is None
             else g['tilt_from_vertical_deg']) for g in doc['gates']]
    return Course(positions=P, yaw_deg=yaw, tilt_deg=tilt,
                  race_order=list(doc['race_order']),
                  inner_m=doc['aperture']['inner_m'],
                  outer_m=doc['aperture']['outer_m'], raw=doc,
                  edge_d_m=np.array([e['d_horiz_m'] for e in doc['edges']]),
                  edge_bearing_deg=np.array([e['bearing_deg'] for e in doc['edges']]),
                  edge_dz_m=np.array([e['dz_up_m'] for e in doc['edges']]))


# ---- the chain solve -------------------------------------------------------------------


def _solve(edges, d, bear_deg, dz, sig_d, sig_b, sig_z):
    """Weighted least squares over p_b - p_a = d * u(bearing), gate 0 pinned.

    Overdetermined wherever the graph closes a loop, so the solution is a genuine
    course, not a set of independently jittered points.
    """
    m = len(edges)
    nun = N_GATES - 1
    A = np.zeros((2 * m, 2 * nun))
    b = np.zeros(2 * m)
    Az = np.zeros((m, nun))
    bz = np.zeros(m)
    for i, e in enumerate(edges):
        ga, gc = e['pair']
        th = math.radians(bear_deg[i])
        v = d[i] * np.array([math.cos(th), math.sin(th)])
        w = 1.0 / math.hypot(sig_d[i], d[i] * math.radians(sig_b[i]))
        for k in (0, 1):
            if ga != 0:
                A[2 * i + k, 2 * (ga - 1) + k] = -w
            if gc != 0:
                A[2 * i + k, 2 * (gc - 1) + k] = w
            b[2 * i + k] = w * v[k]
        wz = 1.0 / sig_z[i]
        if ga != 0:
            Az[i, ga - 1] = -wz
        if gc != 0:
            Az[i, gc - 1] = wz
        bz[i] = wz * dz[i]
    xy = np.linalg.lstsq(A, b, rcond=None)[0].reshape(nun, 2)
    z = np.linalg.lstsq(Az, bz, rcond=None)[0]
    P = np.zeros((N_GATES, 3))
    P[1:, :2] = xy
    P[1:, 2] = z
    return P


def sample(seed=0, include_alt_hypotheses=False, randomize_yaw=True, path=None):
    """One randomized course instance, consistent with the MEASURED uncertainty.

    DESIGN. The perturbation is applied in EDGE SPACE, not gate space: each measured
    pair's distance, grid bearing and signed height step is drawn from its own
    (sigma_d_m, sigma_bearing_deg, sigma_dz_m), and the whole 19-edge chain is then
    re-solved by the same weighted least squares that produced the nominal layout.
    Why: the map measures EDGES. Nothing ever measured a gate's coordinate. Jittering
    each gate independently would break the quantities we actually know -- adjacent
    spacings, the braced loops 7-8-9 / 12-13-14 / 13-14-15 which close to 1.2-1.6% --
    and would hand the policy a course that could not have produced our data. Solving
    the chain instead keeps every sample geometrically self-consistent: loops still
    close, and uncertainty accumulates along the chain the way real chain-map error does.

    The returned positions are nominal + (solve(perturbed) - solve(nominal)), so
    sample() with any seed reduces exactly to load() when the sigmas are zero and the
    nominal layout is reproduced bit-for-bit.

    LIMITATION -- read before trusting the envelope. Edge errors are drawn
    INDEPENDENTLY. They are not independent in reality:
      * edges measured in the same flight share that flight's compass bias, so a whole
        LEG can be rotated together in a way this sampler never draws;
      * the ceiling grid is only known mod 90 deg. The 2-3, 6-7 and 15-16 legs take
        their quadrant from the pilot's sketch, not from a shared measurement. If one of
        those branches is wrong, that leg rotates by 90 degrees -- a discrete failure
        this Gaussian sampler cannot express;
      * edge 1-2 has a rival measured value (13.0 m vs the accepted 8.32 m). It is
        exported as alt_hypothesis_d_m and drawn only when include_alt_hypotheses=True,
        in which case a coin flip picks the hypothesis before the Gaussian is applied.
    So treat the sampled envelope as a LOWER bound on layout uncertainty.

    Yaw and tilt: gate plane yaw is drawn about its measured value with its own MAD
    (and uniformly over 0-180 for the three gates the map refused, 8/12/13, when
    randomize_yaw is on). Tilt is 0 for every gate except the two that carry an
    unresolved tilt prior (8 and 9), which draw uniformly over 0-20 deg -- see README.
    """
    doc = json.load(open(path)) if path else _doc()
    edges = doc['edges']
    rng = np.random.default_rng(seed)

    d0 = np.array([e['d_horiz_m'] for e in edges])
    b0 = np.array([e['bearing_deg'] for e in edges])
    z0 = np.array([e['dz_up_m'] for e in edges])
    sd = np.array([e['sigma_d_m'] for e in edges])
    sb = np.array([e['sigma_bearing_deg'] for e in edges])
    sz = np.array([e['sigma_dz_m'] for e in edges])

    d = d0.copy()
    if include_alt_hypotheses:
        for i, e in enumerate(edges):
            if 'alt_hypothesis_d_m' in e and rng.random() < 0.5:
                d[i] = e['alt_hypothesis_d_m']

    d1 = d + rng.normal(0.0, sd)
    b1 = b0 + rng.normal(0.0, sb)
    z1 = z0 + rng.normal(0.0, sz)
    base = _solve(edges, d0, b0, z0, sd, sb, sz)
    pert = _solve(edges, d1, b1, z1, sd, sb, sz)
    nominal = np.array([g['position_m'] for g in doc['gates']], float)
    P = nominal + (pert - base)

    yaw, tilt = [], []
    for g in doc['gates']:
        y = g['yaw_deg']
        if not randomize_yaw:
            pass
        elif y is None:
            y = float(rng.uniform(0.0, 180.0))
        else:
            y = float((y + rng.normal(0.0, max(g['yaw_mad_deg'] or 5.0, 3.0))) % 180.0)
        yaw.append(y)
        pr = g.get('tilt_prior')
        if pr is not None:
            tilt.append(float(rng.uniform(pr['low_deg'], pr['high_deg'])))
        else:
            t = g['tilt_from_vertical_deg'] or 0.0
            s = g.get('tilt_sigma_deg') or 0.0
            tilt.append(float(abs(t + rng.normal(0.0, s))) if s else float(t))

    return Course(positions=P, yaw_deg=yaw, tilt_deg=tilt,
                  race_order=list(doc['race_order']),
                  inner_m=doc['aperture']['inner_m'],
                  outer_m=doc['aperture']['outer_m'], raw=doc, seed=seed,
                  edge_d_m=d1, edge_bearing_deg=b1, edge_dz_m=z1)


# ---- geometry helpers ------------------------------------------------------------------


def gate_basis(gate, course, yaw_fallback='bisector'):
    """(normal, right, up_in_plane) unit vectors for a gate, in course coordinates.

    `normal` is the gate's plane normal (an AXIS -- sign is not observable from a square,
    so it is returned pointing along +normal-azimuth and you should orient it yourself
    against the race direction). `right` is the horizontal in-plane axis, `up_in_plane`
    the other in-plane axis, tilted out of world-up by the gate's tilt.

    yaw_fallback: what to do at a gate whose yaw the map refused (8, 12, 13).
      'bisector' -- face the racing line (bisector of the in/out legs). An ASSUMPTION.
      'raise'    -- refuse to guess.
    """
    y = course.yaw_deg[gate]
    if y is None:
        if yaw_fallback == 'raise':
            raise ValueError('gate %d has no measured plane yaw' % gate)
        y = course.raw['gates'][gate]['yaw_race_bisector_deg']
    t = math.radians(course.tilt_deg[gate] or 0.0)
    a = math.radians(y)
    n = np.array([math.cos(t) * math.cos(a), math.cos(t) * math.sin(a), math.sin(t)])
    r = np.array([-math.sin(a), math.cos(a), 0.0])
    u = np.cross(n, r)
    return n, r, u / np.linalg.norm(u)


def gate_corners(gate, course, size_m=None, yaw_fallback='bisector'):
    """The 4 corners of a gate's aperture, (4, 3) metres, counter-clockwise in-plane.

    size_m defaults to the 1.5 m inner aperture (the flyable hole). Pass course.outer_m
    for the 2.7 m outer frame -- that is the thing you collide with.
    """
    s = 0.5 * (course.inner_m if size_m is None else size_m)
    _n, r, u = gate_basis(gate, course, yaw_fallback)
    c = np.asarray(course.positions[gate], float)
    return np.array([c - s * r - s * u, c + s * r - s * u,
                     c + s * r + s * u, c - s * r + s * u])


def gate_normal(gate, course, yaw_fallback='bisector'):
    """Plane normal oriented along the race direction (approach -> exit)."""
    n, _r, _u = gate_basis(gate, course, yaw_fallback)
    P = course.positions
    if gate == 0:
        travel = P[1] - P[0]
    elif gate == N_GATES - 1:
        travel = P[-1] - P[-2]
    else:
        travel = P[gate + 1] - P[gate - 1]
    return n if float(n @ travel) >= 0 else -n


# ---- verification ----------------------------------------------------------------------


def sanity_check(course, tol_m=2.0, tol_dz_m=1.0, max_sigma=5.0, verbose=False):
    """Assert the layout reconstructs the pair table it was built from.

    For load() that table IS map_vq2.json's measured pair medians, so this is a direct
    check that the export reproduces the map. For sample() it is the DRAWN table, so the
    same call verifies the sampler produced a geometrically self-consistent course
    rather than a bag of jittered points.

    Returns (max_abs_err_m, max_abs_dz_err_m). Raises AssertionError past tolerance.

    Tolerances are set from the measured behaviour, not from a wish. The NOMINAL layout's
    own worst edge is 14-15 at 0.449 m (dz 0.130 m): the least-braced corner of the
    12-13-14-15 quad, where three measured edges cannot all be satisfied at once, so any
    tolerance below that fails on the truth itself. Over 200 samples the worst-edge error
    runs median 0.75 / p90 1.14 / max 1.65 m (dz 0.29 / 0.49 / 0.81 m) -- the same
    over-determined corner, amplified by the draw. Defaults sit above that.

    Also asserts every drawn edge sits within max_sigma of its measured value, which
    catches a sampler that has quietly wandered outside the evidence.
    """
    P = course.positions
    worst = worst_dz = 0.0
    bad = []
    for i, e in enumerate(course.raw['edges']):
        a, c = e['pair']
        dm = float(course.edge_d_m[i])
        dzm = float(course.edge_dz_m[i])
        d = float(np.hypot(*(P[c][:2] - P[a][:2])))
        err = abs(d - dm)
        dzerr = abs(float(P[c][2] - P[a][2]) - dzm)
        if verbose:
            print('%-6s meas %7.2f  recon %7.2f  err %+6.3f | dz err %+6.3f'
                  % ('%d-%d' % (a, c), dm, d, d - dm,
                     float(P[c][2] - P[a][2]) - dzm))
        worst, worst_dz = max(worst, err), max(worst_dz, dzerr)
        if err > tol_m or dzerr > tol_dz_m:
            bad.append((a, c, round(err, 3), round(dzerr, 3)))
        nd = abs(dm - e['d_horiz_m']) / e['sigma_d_m']
        nb = abs(float(course.edge_bearing_deg[i]) - e['bearing_deg'])             / e['sigma_bearing_deg']
        if 'alt_hypothesis_d_m' not in e and max(nd, nb) > max_sigma:
            bad.append((a, c, 'drawn %.1f sigma from measured' % max(nd, nb), 0))
    assert not bad, 'pair reconstruction outside tolerance: %s' % bad
    assert P.shape == (N_GATES, 3)
    assert np.allclose(P[0], 0.0), 'gate 0 must be the origin'
    return worst, worst_dz


if __name__ == '__main__':
    c = load()
    print('nominal:', sanity_check(c, verbose=True))
    for s in range(5):
        print('seed %d:' % s, sanity_check(sample(s)))
