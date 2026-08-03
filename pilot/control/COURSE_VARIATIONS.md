# Course variations — a tolerance envelope around a lap we intend to memorize

2026-08-02. Code: `coursevar.py` (deform + route solve), `coursevar_sheet.py` (render),
`coursevar_reward.py` (measures the spec's reward against a route).

## Picking this up

Everything lives on branch **`worktree-vq2-reward-eval`**, which already contains
`worktree-vq2-reward-merge` (merged in at `b192aa8`, clean). To take it:

```
git merge worktree-vq2-reward-eval        # from your own worktree, tree clean
```

That brings three new modules, this doc, and **two edits inside `surrogate/`** — the
race-start spawn (`env.py`) and a corrected docstring (`camera.py`), both at `6d54f3d`.
Those are the only files touched outside the new modules, and neither exists on any
other branch: a merge to master from a branch other than this one silently drops them.

DONE — course variations (300 reviewed and signed off), the spline route generator,
the observed race-start geometry, and a measurement of `ROUTE_REWARD_SPEC` section 4
that characterizes both of its defects with fixes demonstrated.

NOT DONE, and it is the substantive remainder: **the route reward is not wired into
`env.py`.** The reward there is still `k_progress * prog`, the old straight-line closure
term. `coursevar_reward.py` proves the route works as a reward reference and proves what
breaks; it is a harness, connected to nothing that trains. Wiring it needs routes
precomputed per pool course, a batched projection across `n_envs` rows per step (the
solver here is per-course Python), and the term itself in `env.py`. The deformation is
likewise a standalone bank — `vq2course.py` still builds the pool its own way.

## The decision this rests on

`ROUTE_REWARD_SPEC.md` is built on an anti-memorization thesis: regenerate the course
every episode so the policy is forced to locate gates through its observations rather
than reciting coordinates. **We are going the other way.** Either the drone can fly the
whole course or it can't, and a memorized lap structure is an asset toward that. These
variations are therefore not samples from the map's measurement posterior — they are a
deliberate **tolerance envelope** around a course we are going to learn, so the memory
survives the map being wrong by more than the map thinks it is.

That reframing changes what is and isn't a defect:

* The course README's rule *"no independent per-gate jitter, ever"* protects **posterior
  fidelity** — independent jitter manufactures courses that contradict the braced loops
  (7-8-9, 12-13-14-15, which close to 1.2–1.6%). Posterior fidelity is not what we are
  buying, so the rule does not bind here. `deform()` moves gates independently and does
  **not** re-solve the edge chain: re-solving would drag each drawn displacement back
  toward what the loops allow, which is precisely the constraint we want to fly outside.
* Displacements run 2–8 m against a median `sigma_xy_local` of **0.52 m** — that is
  **4–15σ**. Over-provision is the point.
* The 6-7 quadrant flip is not modelled. Dropped by decision, and it was broken anyway
  (below).

**Gate yaw is left at its measured value when a gate moves** — the map may be wrong about
where a gate is without being wrong about how it is mounted. The side effect is that a
displaced gate can end up badly oblique to its new racing line, which is the same
condition that breaks the spec's route generator at gate 7. The solver below handles it
(zero route failures over 300 draws), but a variation with a wildly oblique gate is a fair
thing to reject on sight.

A deformed course carries no `edge_d_m` / `edge_bearing_deg`, so `cv.sanity_check()` does
not apply to it by design — there is no drawn edge table to reconcile against.

Validity is judged by eye against the mapped course, not by an acceptance predicate.
`geometry_faults()` removes only what no reviewer should have to spot: sub-5 m legs, gates
displaced into one another, and corners sharper than the real course has (see the turn
filter below).

## Start conditions — every episode begins at gate 0

Observed in VQ2: the drone sits **on a platform, at rest, ~10 m out, level with gate 0,
pitched 20° nose-down** so the 20°-up camera looks ahead. And by decision, **every episode
starts there** — no mid-course spawns.

The route matches this: the lead-in runs 10 m back along gate 0's normal (horizontal,
z = 0.000, so the start is level with gate 0 without further work) and the speed profile
opens at **0 m/s**, reaching 8 m/s after 4 m — at pace 6 m before gate 0.

The 10 m matters for reward correctness, not tidiness. `s₀` is initialized by nearest
point, so at the old 6 m lead-in a drone spawning at 10 m sat 4 m *behind* the line, pinned
to the first vertex with `d⊥ = 4 m`, and every episode opened on a corridor multiplier of
**0.077**. It is now 1.000. Gate-0-only starts also make `s₀ = 0` always, which retires a
whole class of projection bug: no nearest-point ambiguity, no need to restrict the search
to a spawn gate's neighbourhood.

### What the surrogate does instead

Three deltas, none applied here — `env.py` is spawn config and outside this module.

| | observed | `env.py` today |
|---|---|---|
| spawn gate | always 0 | 25% mid-course (`start_probs[1]`) |
| speed | 0 (on a platform) | `U(0.35, 1.05) × cruise` = 3.5–10.5 m/s |
| pitch | −20° (nose-down) | `N(0, 0.06)` = level ±3.4° |
| platform | 3–4 m structure under the start | not modelled |

`a0 = where(mode == 1, random, 0)`, so only mode 1 leaves gate 0: dropping mid-course
spawns is `start_probs[1] = 0`, renormalizing to `(0.667, 0, 0.16, 0.107, 0.067)`. With
every spawn now at the start line there is no flying start left to preserve, so
`start_speed_frac` goes to zero outright.

The pitch gap is the one with teeth. `camera.py` mounts the camera **20° UP**, putting
body-forward at image row 296 of 360 with a vertical span of +49.4°/−9.4° about it. Gate 0
sits at the drone's own altitude, so:

* spawning level (today) renders gate 0 at **row 296** — 9.4° from leaving the frame, and
  any nose-down to accelerate pushes it out;
* spawning at −20° renders it at **row 180**, dead centre.

`camera.py`'s own docstring names that lower edge as the binding constraint on this course.

### The cost of gate-0-only starts, stated once

Mid-course spawns were buying **coverage**, not variety: with `collision_terminates=True`
a policy that dies at gate 3 never sees gates 4–16, and course deformation does not carry
it downcourse. Back-half exposure is now gated on front-half reliability.

The mechanism that recovers it without reintroducing mid-course spawns is
**`collision_terminates=False`** (`TRAINING_ARCHITECTURE.md` E5, already a config flag):
contact costs `k_collision` and hands back to a recovered state instead of ending the
episode, so a single gate-0 episode still reaches the back half. That also puts the
recovery-handback distribution back where it belongs — arising mid-race from an actual
crash, rather than being synthesized at the start line by spawn modes 2–4.

## The route is a spline, not a pursuit path

`spline_route()` is the generator. A centripetal Catmull-Rom curve is interpolated
through the 17 gate centres, arc-length resampled at 0.25 m, and given a speed profile
that respects the cornering limit (`v ≤ √(a_lat/κ)`) with forward and backward passes for
braking and acceleration — so it is a trajectory a drone could fly, not a drawing.

**Why it replaced the pursuit ODE.** The spec's generator drives a waypoint *pair* offset
along each gate normal, which makes every gate a **vertex**: the path arrives, stops
turning, and leaves. On this course that produced hairpins at 6-7-8 sharp enough to read
as doubling back, and no racing line does that. A spline turns *about* the gates rather
than *at* them, so a corner has an entry, an apex and an exit.

| | pursuit ODE | spline |
|---|---|---|
| min turn radius (nominal) | **0.32 m** | **3.18 m** |
| gate crossing distance | ≤ 0.126 m | 0 by construction |
| speed in tightest corner | — | 6.2 m/s (from 8.0) |

Because the centres are interpolated exactly, crossing *distance* stops being the
interesting number and crossing *angle* takes over: a gate is a square hole, so meeting
its plane at θ off the normal narrows the usable opening to `inner_m · cos θ`.
`gate_crossings()` reports both. On the nominal course the worst is **gate 7 at 46.2°,
leaving a 1.04 m opening** — passable, and far better than the 85.3° the straight chain
implies, because the spline is already turning as it arrives.

Centripetal parameterization (α = 0.5) matters here: uniform Catmull-Rom forms cusps and
self-intersecting loops exactly where control points turn sharply, which on this course is
gates 7, 9 and 13.

## The turn filter — and a bug in the first version of it

The sharpest corner on the **measured** course is 76.5° (gate 13). Independent per-gate
displacement readily invents corners far sharper, and a 120° hairpin is not a deformation
of this course — it is a different course.

The first version of `geometry_faults()` capped turns at an absolute 150°. That is above
anything the deformation can produce, so it filtered **nothing**: measured over 120
variations, **63 carried a turn past 90°** — an actual double-back — and the filter passed
every one. A bound has to sit inside the distribution to bound anything.

Now two limits, both needed:

* `MAX_TURN_DEG = 90` — absolute; past this the route doubles back on itself.
* `TURN_MARGIN_DEG = 20` — per gate, against the nominal turn **at that gate**, so a
  corner that is straight on the real course cannot become a corner here.

This costs yield: ~9% at 3–6 gates moving 2–8 m, against ~90% before. Rejection is cheap
(0.4 s per 40 kept) so the range is kept rather than traded away, but the accepted set is
**selection-biased and that is the intended behaviour**: a gate at a corner cannot move far
in an arbitrary direction and still leave the course recognisable, so corner gates are
displaced less, and along-track more than across-track. Realized median displacement is
3.9 m against the 2–8 m drawn. All 16 movable gates still appear.

## The pursuit solver — two corrections to spec section 3

`solve_route()` is kept for comparison and because the findings below are about the spec,
not about this module. Both failures were reproduced against the real `course_vq2` package.

**1. The phase switch has no hysteresis, so it deadlocks at gate 0.** The spec switches
from the approach waypoint to the punch-through waypoint when "the approach-side distance
along n̂ exceeds L" stops holding. A critically damped tracker converges on a stationary
target *from below, without overshoot* — so that distance tends to L from above and the
condition never flips. As written the generator sits at gate 0 forever. Fixed by ending
the approach phase when the waypoint is **reached** (within `reach_tol`, default 0.5 m).

**2. The approach phase must be sticky, or it deadlocks at gate 7.** Gate 7's measured
plane sits **85.3°** off the incoming 6→7 leg (yaw 87.89°, n=649 rows, MAD 6.69, against
a race bisector of 136.2°). The route therefore reaches that plane nearly edge-on and
crosses it ~11 m from the centre while still flying toward the approach point. Under the
spec's rule that crossing flips the target to the exit point — now behind the aircraft —
and the route can never recover. Measured: deadlock at gate 7 on the nominal course and
~90% of sampled ones.

The spec's §3.3 retry ladder does not address this, because it is geometry, not tracking
bandwidth:

| retry | result |
|---|---|
| ωn 1.5 → 2.93 (spec's 3 × +25%) | miss 10.98 → 10.52 m, still fails; saturation 6% → 44% |
| v_max 8 → 2 | miss gets *worse* (10.7 → 11.3 m) |
| L 3 → 10 | fails at every value; L = 12 breaks gate 3 instead |

Fixed by holding the approach waypoint until it is reached, whatever the plane does
meanwhile. With both corrections the nominal course solves **17/17 gates, worst miss
0.126 m** (spec acceptance is 0.55 m), 263 m long, peak lateral accel 8.6 m/s² < 12 —
correct, but with the 0.32 m min turn radius that sent us to the spline instead.

Note also that ζ = 1.0 does **not** give "no overshoot by construction" as §3.1 claims:
`sat()` is active 1–5% of steps at spec parameters and 44% at the ωn the retry ladder
climbs to, and a saturated second-order system can overshoot. It happens not to matter
at these parameters, but the guarantee isn't there.

## Findings on ROUTE_REWARD_SPEC still open

Recorded for whoever picks the reward work back up; none are addressed by this module.

1. **§2.4 rejects what §2.1 mandates.** §2.1 requires `include_alt_hypotheses=True` on
   every episode so the contested 1-2 leg is flown. That draws edge 1-2 at 13.0 m —
   **1.56× the nominal 8.32 m**, outside §2.4's `[0.7, 1.4]×` bound. Measured rejection:
   **41% with the flag on, 3% off**. The acceptance test silently deletes exactly the
   hypothesis the spec exists to train against.
2. **The Layer B 6-7 flip does not do what it says.** Adding ±90° to edge 6-7's bearing
   alone and re-solving gives a **rigid 22.0 m translation** of the back half with its
   shape unchanged (= |6-7| × √2 = 15.56 × 1.414) — not the quadrant rotation described.
   Rotating *every* edge bearing downstream of gate 6 gives 10 gates moving with median
   85.7 m / max 165 m, which matches the spec's own 103 m figure. In a chain solve with
   absolute per-edge bearings, changing one edge translates the downstream; it does not
   rotate it.
3. **§4's Δs clip pays the policy to fly slowly.** Clipping at `v_max·dt` makes total
   route return flat to 8 m/s and *decreasing* above it — at the env's own
   `cruise_speed_mps = 10` a policy books 20% less, at 12 m/s 33% less, and
   `enable_time_penalty` defaults to `False` so nothing offsets it.
4. **dt is 1/55 s, not 0.02** (`surrogate/env.py:337`). k_p ≈ 0.62, not 0.56.
5. **Route anchor.** §3.1 integrates "from the spawn point", §4 initializes s₀ by nearest
   point. It must be the latter — 25% of episodes spawn mid-course and 8% corridor-offset,
   so a spawn-anchored route cannot be precomputed per course.
6. **`STATE_PERCEPTION_FOR_CONTROL.md`**, cited for both §4 riders, is not in this repo.

## Running it

```
python coursevar_sheet.py --n 300 --out variations
```

Deterministic in `--seed0`, so the same 300 come back on demand and the PNGs are
reproducible output rather than precious data — they are gitignored for that reason.

Yield is ~9% under the turn filter; generation is fast enough that this is irrelevant.

Yield is ~9% under the turn filter; generation is fast enough that this is irrelevant.

Images are `v000.png` upward, **sorted by max displacement**, so the mild cases come
first and the marginal ones cluster at the end; the point where they stop looking like
VQ2 is the number worth knowing. Every image is drawn on **identical axes** — per-image
autoscaling would rescale each plot to its own extent and hide the very differences being
reviewed.

Each frame carries the mapped course as a grey ghost (chain + its solved route), this
variation's route in teal, and displaced gates in rust with a leader back to where the
map puts them. The header carries route length, min turn radius, sharpest gate turn, and
the worst gate-plane crossing angle. Plan view on top, elevation below. Screen x is `-world_x` so the lap reads
left-to-right, with y flipped to match so handedness survives.
