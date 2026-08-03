"""Procedural courses, fresh every episode. The world frame lives here and stops here.

The VQ2 map is unobservable (spec 9.3 blocks gate geometry and both pose streams, and
`NOTES.md` says vision-based mapping is not started), so the policy must learn
gate-SEEKING, not a track. A course is therefore generated per episode and thrown away,
and nothing about it -- position, heading, gate geometry -- ever reaches the observation.
It is used for exactly two things: reward and collision tests.

What the generator is trying to reproduce, from `NOTES.md`:

* **~20 gates.** 18-22, Claire-observed and unconfirmed.
* **Segment length ~28 m.** The Station columns give 7 stations in 19.29 s, i.e. 2.76 s
  per station at constant cruise. At a racing 8-12 m/s that is 22-33 m per station, and
  VQ1's 6 gates over ~167 m is 28 m per gate independently. The two agree, so segment
  length is drawn around 28 m and tightened with difficulty.
* **The path winds far more than VQ1's near-straight line.** This is the stated real
  difficulty, so turn magnitude is the main difficulty axis: +-20 deg at difficulty 0,
  +-80 deg at difficulty 1, with a persistent turn direction so corners are corners
  rather than jitter.
* **Vertical profiles including ~20 deg descents.** The case that pushes a gate below the
  camera's -9.4 deg lower frame edge, which `CONVENTIONS.md` calls the binding
  constraint. The elevation draw reaches +-22 deg at difficulty 1 precisely to generate it.

Gate geometry is spec-exact: 1500 mm inner aperture, 2700 mm outer frame, 260 mm deep
(the depth is absorbed by the randomised collision sphere, per architecture T4).

A gate's stored normal is its TRAVEL direction -- the way you fly through it, the
bisector of the incoming and outgoing segments. `GateObs.normal_body` is the opposite:
`interface.py` requires the normal to point TOWARD the camera, which for a gate not yet
crossed is the approach side. The negation happens once, in `detect.py`.
"""

import numpy as np

GATE_INNER_M = 1.5          # spec-exact aperture, the calibrated rangefinder
GATE_OUTER_M = 2.7          # frame outer extent
GATE_DEPTH_M = 0.26         # modelled as a plane; the sphere margin absorbs it

# Extra gates generated past the last raced one. Detection slots reach active_gate+2 and
# the corridor needs a run-out segment, so the generator simply makes a longer course and
# `n_gates` decides where the race ends. No special-casing at the finish.
RUNOUT = 3


def generate(rng, m, cfg):
    """Generate `m` courses. Returns arrays; `pos`/`nrm` are `[m, n_gates_max+RUNOUT, 3]`."""
    d = float(np.clip(cfg.difficulty, 0.0, 1.0))
    lo, hi = cfg.n_gates_range
    n_slots = int(hi) + RUNOUT

    n_gates = rng.integers(int(lo), int(hi) + 1, size=m)

    seg_lo = _lerp(cfg.seg_len_easy[0], cfg.seg_len_hard[0], d)
    seg_hi = _lerp(cfg.seg_len_easy[1], cfg.seg_len_hard[1], d)
    turn_max = np.radians(_lerp(cfg.turn_deg_easy, cfg.turn_deg_hard, d))
    elev_max = np.radians(_lerp(cfg.elev_deg_easy, cfg.elev_deg_hard, d))
    corridor_r = _lerp(cfg.corridor_easy_m, cfg.corridor_hard_m, d)

    ceil_h = rng.uniform(cfg.ceiling_range_m[0], cfg.ceiling_range_m[1], size=m)
    alt_lo = cfg.floor_clear_m + 0.5 * GATE_INNER_M
    alt_hi = np.maximum(ceil_h - cfg.ceil_clear_m - 0.5 * GATE_INNER_M, alt_lo + 1.0)
    alt_mid = 0.5 * (alt_lo + alt_hi)
    alt_span = np.maximum(alt_hi - alt_lo, 1e-3)

    psi = rng.uniform(-np.pi, np.pi, size=m)
    alt = alt_mid + rng.uniform(-0.2, 0.2, size=m) * alt_span
    p = np.stack([np.zeros(m), np.zeros(m), -alt], axis=1)

    start_p = p.copy()
    pos = np.empty((m, n_slots, 3))
    turn_sign = rng.choice(np.array([-1.0, 1.0]), size=m)

    for k in range(n_slots):
        flip = rng.random(m) < cfg.turn_flip_p
        turn_sign = np.where(flip, -turn_sign, turn_sign)
        dpsi = turn_sign * turn_max * rng.uniform(cfg.turn_min_frac, 1.0, size=m)
        psi = psi + dpsi

        # Elevation is mean-reverting on altitude so the course stays inside the hangar
        # without the profile being clamped flat.
        pull = -cfg.alt_revert * (alt - alt_mid) / alt_span
        elev = np.clip(rng.normal(0.0, 0.55, size=m) * elev_max + pull * elev_max,
                       -elev_max, elev_max)

        seg = rng.uniform(seg_lo, seg_hi, size=m)
        step = np.stack([seg * np.cos(elev) * np.cos(psi),
                         seg * np.cos(elev) * np.sin(psi),
                         -seg * np.sin(elev)], axis=1)
        p = p + step
        alt = np.clip(-p[:, 2], alt_lo, alt_hi)
        p[:, 2] = -alt
        pos[:, k] = p

    # Travel normal: the bisector of the segment in and the segment out, which is the
    # only choice that makes a corner flyable at speed.
    prev = np.concatenate([start_p[:, None, :], pos[:, :-1]], axis=1)
    nxt = np.concatenate([pos[:, 1:], (2 * pos[:, -1] - pos[:, -2])[:, None, :]], axis=1)
    din = _unit(pos - prev)
    dout = _unit(nxt - pos)
    nrm = _unit(din + dout)

    return dict(pos=pos, nrm=nrm, n_gates=n_gates.astype(np.int64),
                start_p=start_p, start_dir=din[:, 0].copy(),
                z_ceil=-ceil_h, corridor_r=np.full(m, corridor_r),
                seg_len=np.full(m, 0.5 * (seg_lo + seg_hi)))


def _lerp(a, b, t):
    return float(a) + (float(b) - float(a)) * t


def _unit(v, eps=1e-12):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def point_segment_distance(p, a, b):
    """Distance from `[n,3]` points to the `[n,3]`-`[n,3]` segments. Used by the corridor."""
    ab = b - a
    denom = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9)
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / denom, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def arclength_point(pos, start, dists):
    """Points at arclengths `dists` `[n,k]` along the polyline `start -> pos[:,0] -> ...`.

    Used for the ribbon: the cyan corridor shows roughly the next 5 gates, so its
    direction samples are the course path sampled at increasing look-ahead.
    """
    n, segs = pos.shape[0], pos.shape[1]
    nodes = np.concatenate([start[:, None, :], pos], axis=1)          # [n, segs+1, 3]
    d = np.linalg.norm(np.diff(nodes, axis=1), axis=2)                # [n, segs]
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(d, axis=1)], axis=1)
    out = np.empty(dists.shape + (3,))
    for j in range(dists.shape[1]):
        s = np.clip(dists[:, j], 0.0, cum[:, -1] - 1e-6)
        i = np.clip(np.sum(cum[:, :-1] <= s[:, None], axis=1) - 1, 0, segs - 1)
        rows = np.arange(n)
        t = (s - cum[rows, i]) / np.maximum(d[rows, i], 1e-9)
        out[:, j] = nodes[rows, i] + t[:, None] * (nodes[rows, i + 1] - nodes[rows, i])
    return out
