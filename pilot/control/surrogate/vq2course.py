"""VQ2 -- the measured course, reshaped into what `course.generate` already returns.

`course.py` invents a course per episode. This module supplies the *measured* VQ2 layout
instead, drawn from `pilot/control/course/` (17 gates, 268 m of racing line, built by the
perception side from vision + gravity + a skylight compass over five flights).

    pool = get_pool(cfg)
    c = mix(rng, m, cfg, pool)      # same dict `course.generate` returns

`mix` returns procedural courses for a `1 - cfg.vq2_frac` share of the rows, so the policy
keeps learning gate-SEEKING for the case where it is lost and the map cannot help it.
With `vq2_frac = 0` (the default) this module is inert and the surrogate behaves exactly
as it did before.

## Why the measured layout matters

The generator's invented statistics are wrong in one direction that matters. Its shortest
segment is 18 m at maximum difficulty; **10 of VQ2's 16 race edges are shorter than that**,
median 15.1 m, tightest 8.3 m. And the corners that actually decide this course pair a
sharp turn with a short exit -- gate 9 is a 66.6 deg turn with 10.1 m to the next gate,
gate 13 a 76.4 deg turn with 12.3 m -- roughly one second of flight to reacquire, align and
thread. The procedural generator never draws that combination.

## What is measured and what is assumed

MEASURED (see `course/README.md`): the 17 gate positions, the race order, the inter-gate
distances/bearings/height steps, and the per-episode uncertainty the sampler draws from --
0.31-0.89 m step-to-step, which is what a gate-relative policy actually feels.

ASSUMED HERE, because the map does not contain it:

* **Floor reference.** The map's `z` is relative to gate 0, not to the hangar floor, so
  gate 0's height above the floor is unknown. It is drawn per episode from
  `cfg.vq2_floor_clear_m`, and the ceiling is placed above the course's own high point by
  `cfg.vq2_headroom_m` -- so every episode is guaranteed to fit, and no single ceiling can
  be learned. Replace both with a measurement when one exists.
* **Gate plane yaw.** The map exports three disagreeing candidates (measured / grid-aligned
  / race-line bisector) and the perception side's guidance is to treat "enter along the
  normal" as a soft preference, steering at gate centres and the race line instead. So
  `vq2_yaw_mode='mixed'` draws uniformly among the candidates a gate has, plus jitter --
  turning an unresolved disagreement into domain randomization rather than picking a side.
  Gates 8, 12 and 13 have no trusted yaw at all and fall back to the bisector, which is
  both the map's own recommendation and what `course.generate` has always used.
* **Gate tilt.** Held at vertical for every gate. The shipped JSON randomizes gates 8 and 9
  over a uniform 0-20 deg prior, and the perception side has since overturned that: gate 9
  measures ~21-24 deg across four sessions and every other gate is vertical. 21-24 deg is
  *outside* the shipped prior, so the current file cannot sample the truth and training on
  it would bake in an artifact. `cfg.vq2_tilt_deg` takes `{gate: (lo, hi)}` once the
  corrected JSON lands.

Note that the tilt the package models is a LEAN -- the plane normal tips out of horizontal.
If gate 9 turns out to be rolled about its own normal instead, this surrogate cannot
represent it at all: `env._gate_geometry` reduces a crossing to a radial distance, which is
rotationally symmetric about the normal, so an in-plane roll of a square aperture is
invisible to it. That would need the square-aperture collision test, not a knob here.

## Frames

The map is right-handed with **z up**; the surrogate is NED with **z down**. The conversion
is a 180 deg rotation about x, `(x, y, z) -> (x, -y, -z)` -- a proper rotation, so the
course is not mirrored. The map has no absolute heading (it may be rotated by k*90 deg in
the sim world) and that is fine here: the surrogate's world frame is internal, never
encoded into an observation, and a rotation about the vertical is unobservable when gravity
is the only absolute reference.
"""

import importlib.util
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import course  # noqa: E402

# The package the perception side ships. `pilot/control/course/`, beside `surrogate/`.
PKG_DIR = os.path.join(os.path.dirname(_HERE), "course")

# Map (z up) -> NED (z down). 180 deg about x; det = +1, so no mirroring.
_TO_NED = np.array([1.0, -1.0, -1.0])

_pkg = None
_pool_cache = {}


def _load_pkg():
    """Import `course/course_vq2.py` BY PATH, not by module name.

    `pilot/control/` sits ahead of this directory on `sys.path`, so the sibling `course/`
    directory is a namespace-package portion named `course` -- colliding with
    `surrogate/course.py`. The procedural generator wins today only because `course/` has
    no `__init__.py`; adding one would silently shadow it. Loading by path is immune.
    """
    global _pkg
    if _pkg is None:
        path = os.path.join(PKG_DIR, "course_vq2.py")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"VQ2 course package not found at {path}. It ships in pilot/control/course/; "
                "set cfg.vq2_frac = 0 to train on procedural courses only.")
        spec = importlib.util.spec_from_file_location("_vq2_course_pkg", path)
        mod = importlib.util.module_from_spec(spec)
        # Register BEFORE exec: `course_vq2` defines a @dataclass, and dataclasses resolves
        # annotations through `sys.modules[cls.__module__]`, which is None until this is set.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _pkg = mod
    return _pkg


def _raw_json():
    with open(os.path.join(PKG_DIR, "course_vq2.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


# =========================================================================================
# geometry
# =========================================================================================
def _bisector_az(P):
    """Horizontal race-line bisector azimuth at each gate, radians. `P` is (S, 3), map frame.

    The bisector of the incoming and outgoing legs is the only gate facing that makes a
    corner flyable at speed -- the same choice `course.generate` makes, and the fallback
    `course/README.md` recommends where yaw was refused. Horizontal because the gates are
    measured to be vertical planes, so their normals carry no elevation.
    """
    d = np.diff(P, axis=0).astype(float)
    d[:, 2] = 0.0
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    din = np.vstack([d[0][None, :], d])          # gate 0 has no incoming leg
    dout = np.vstack([d, d[-1][None, :]])        # the last gate has no outgoing leg
    b = din + dout
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    b = np.where(nb < 1e-6, din, b / np.maximum(nb, 1e-9))   # a 180 deg reversal cancels
    return np.arctan2(b[:, 1], b[:, 0])


def _normals(P, az, tilt_rad):
    """Unit gate normals from plane azimuth and lean, oriented along the race direction.

    Yaw is an AXIS -- a square gives no sign -- so the sign is set by the travel direction
    rather than taken from the azimuth.
    """
    ct = np.cos(tilt_rad)
    n = np.stack([ct * np.cos(az), ct * np.sin(az), np.sin(tilt_rad)], axis=1)
    travel = np.empty_like(P)
    travel[1:-1] = P[2:] - P[:-2]
    travel[0] = P[1] - P[0]
    travel[-1] = P[-1] - P[-2]
    s = np.einsum("ij,ij->i", n, travel)
    return n * np.where(s >= 0.0, 1.0, -1.0)[:, None]


def _extend_runout(P, n_race, S):
    """Pad a 17-gate course out to `S` slots by flying straight on past the last gate.

    These slots are never raced (`n_gates` is 17, so the episode finishes first). They
    exist so the three detection slots and the corridor's `active + 1` lookup have
    somewhere to point. Level continuation at the course's mean spacing.
    """
    out = np.zeros((S, 3))
    out[:n_race] = P
    d = P[n_race - 1] - P[n_race - 2]
    d[2] = 0.0
    step = d / max(float(np.linalg.norm(d)), 1e-9) * float(
        np.mean(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    for k in range(n_race, S):
        out[k] = out[k - 1] + step
    return out


# =========================================================================================
# the pool
# =========================================================================================
class VQ2Pool:
    """`size` pre-sampled VQ2 courses, in NED, with gate 0 at z = 0.

    Pre-sampled because `course_vq2.sample()` costs ~1.8 ms -- it perturbs the edge table
    and re-solves the whole chain, which is what keeps a sample geometrically
    self-consistent rather than jittering gates independently. At the reset rate a
    vectorized env runs at, calling it live would dominate the step. Drawing an index from
    a few thousand pre-solved courses is free, and a few thousand is ample given the
    underlying uncertainty is only 0.3-0.9 m step-to-step.

    The altitude offset is NOT baked in: it is drawn per episode in `draw`.
    """

    def __init__(self, size, cfg, seed=0):
        pkg = _load_pkg()
        raw = _raw_json()
        gates = raw["gates"]
        self.n_race = int(pkg.N_GATES)
        self.S = int(cfg.n_gates_range[1]) + course.RUNOUT
        if self.S < self.n_race + course.RUNOUT:
            raise ValueError(
                f"n_gates_range[1] = {cfg.n_gates_range[1]} leaves no room for VQ2's "
                f"{self.n_race} gates plus {course.RUNOUT} run-out slots")

        K = int(size)
        rng = np.random.default_rng(seed)
        jitter = math.radians(float(cfg.vq2_yaw_jitter_deg))
        mixed = str(cfg.vq2_yaw_mode) == "mixed"
        tilt_spec = dict(cfg.vq2_tilt_deg or {})
        alt_p = float(cfg.vq2_alt_hypothesis_p)

        # Candidate plane azimuths per gate, in radians, from the JSON. `yaw_deg` is null
        # at gates 8/12/13 and `yaw_grid_bin_deg` encodes Claire's grid-alignment claim,
        # which the measured azimuths contradict -- see the module docstring.
        cand = []
        for g in gates:
            c = []
            if mixed:
                if g.get("yaw_deg") is not None:
                    c.append(math.radians(float(g["yaw_deg"])))
                if g.get("yaw_grid_bin_deg") is not None:
                    c.append(math.radians(float(g["yaw_grid_bin_deg"])))
            cand.append(c)

        self.pos = np.empty((K, self.S, 3))
        self.nrm = np.empty((K, self.S, 3))
        self.seg_len = np.empty(K)
        self.max_up = np.empty(K)
        self.min_up = np.empty(K)

        for i in range(K):
            # Positions only. `sample`'s own yaw draw is not used: it puts a UNIFORM
            # 0-180 deg plane on every gate whose yaw was refused, which would face gates
            # 8/12/13 in a random direction each episode and make them unlearnable.
            #
            # `include_alt_hypotheses` coin-flips edge 1-2 onto its rival 13.0 m reading.
            # That edge is CONTESTED -- the accepted 8.32 m rests on 5 rows from a single
            # flight, and a refused 34-row channel reads 13.0 m. Every VQ2 episode flies
            # 1->2, so a policy trained only on 8.32 m has memorized a spacing that may be
            # 4.7 m wrong. `course/README.md` recommends switching this on for exactly
            # this case. The package flips at 50%, so the rival lands in ~alt_p/2 of pool
            # courses.
            c = pkg.sample(seed=int(seed) * 1_000_003 + i, randomize_yaw=False,
                           include_alt_hypotheses=bool(rng.random() < alt_p))
            P = _extend_runout(np.asarray(c.positions, dtype=float), self.n_race, self.S)

            bis = _bisector_az(P)
            az = bis.copy()
            if mixed:
                for k in range(self.n_race):
                    opts = cand[k] + [bis[k]]
                    az[k] = opts[int(rng.integers(len(opts)))]
                az[:self.n_race] += rng.normal(0.0, jitter, self.n_race)

            tilt = np.zeros(self.S)
            for k, lohi in tilt_spec.items():
                if 0 <= int(k) < self.S:
                    tilt[int(k)] = math.radians(float(rng.uniform(*lohi)))

            N = _normals(P, az, tilt)
            self.pos[i] = P * _TO_NED
            self.nrm[i] = N * _TO_NED
            self.seg_len[i] = float(np.mean(
                np.linalg.norm(np.diff(P[:self.n_race], axis=0), axis=1)))
            self.max_up[i] = float(P[:self.n_race, 2].max())
            self.min_up[i] = float(P[:self.n_race, 2].min())

        self.K = K

    def draw(self, rng, m, cfg):
        """`m` courses, in the dict shape `course.generate` returns."""
        idx = rng.integers(0, self.K, size=m)
        pos = self.pos[idx].copy()
        nrm = self.nrm[idx].copy()

        # Place the floor. Parameterized by the LOWEST gate's height rather than gate 0's:
        # gate 0 is the map's origin but not necessarily its low point, and a sampled
        # course can put a gate below it (the per-edge dz draws accumulate). Anchoring on
        # the minimum is what actually guarantees the clearance. NED is down-positive, so
        # raising the course means subtracting.
        low = rng.uniform(*cfg.vq2_floor_clear_m, size=m)
        headroom = rng.uniform(*cfg.vq2_headroom_m, size=m)
        offset = low - self.min_up[idx]
        pos[:, :, 2] -= offset[:, None]
        z_ceil = -(self.max_up[idx] + offset + headroom)

        seg = self.seg_len[idx]
        start_p = pos[:, 0] - nrm[:, 0] * seg[:, None]
        d = float(np.clip(cfg.difficulty, 0.0, 1.0))
        corridor_r = np.full(m, course._lerp(cfg.corridor_easy_m, cfg.corridor_hard_m, d))

        return dict(pos=pos, nrm=nrm,
                    n_gates=np.full(m, self.n_race, dtype=np.int64),
                    start_p=start_p, start_dir=nrm[:, 0].copy(),
                    z_ceil=z_ceil, corridor_r=corridor_r, seg_len=seg)


def get_pool(cfg):
    """The pool for this config, built once per process.

    Keyed on the fields that shape it -- deliberately NOT on difficulty, speed_cap or the
    env seed. `train.py` rebuilds the whole env on every curriculum promotion, and the
    measured layout does not depend on the curriculum, so a fresh pool per rebuild would
    cost seconds and silently change the course set mid-run.
    """
    if cfg.vq2_frac <= 0.0:
        return None
    key = (int(cfg.vq2_pool_size), int(cfg.n_gates_range[1]), str(cfg.vq2_yaw_mode),
           float(cfg.vq2_yaw_jitter_deg), int(cfg.vq2_pool_seed),
           float(cfg.vq2_alt_hypothesis_p),
           tuple(sorted((int(k), tuple(v)) for k, v in dict(cfg.vq2_tilt_deg or {}).items())))
    if key not in _pool_cache:
        _pool_cache[key] = VQ2Pool(cfg.vq2_pool_size, cfg, seed=cfg.vq2_pool_seed)
    return _pool_cache[key]


def mix(rng, m, cfg, pool):
    """`course.generate`, with a `cfg.vq2_frac` share of the rows replaced by VQ2 courses."""
    if pool is not None and cfg.vq2_frac >= 1.0 and m > 0:
        # Pure VQ2: generating procedural courses only to overwrite every one of them is
        # pure waste, and at frac = 1 that is exactly what the path below would do.
        return pool.draw(rng, m, cfg)
    out = course.generate(rng, m, cfg)
    if pool is None or cfg.vq2_frac <= 0.0 or m == 0:
        return out
    take = rng.random(m) < float(cfg.vq2_frac)
    k = int(np.count_nonzero(take))
    if k == 0:
        return out
    v = pool.draw(rng, k, cfg)
    for key, val in v.items():
        out[key][take] = val
    return out
