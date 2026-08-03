# Course variations — a tolerance envelope around a lap we intend to memorize

2026-08-02. Code: `coursevar.py` (deform + route solve), `coursevar_sheet.py` (render).

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
`geometry_faults()` only removes courses that are not flyable at all (sub-5 m legs,
gates inside one another, turn reversals) so review time isn't spent on obvious junk.

## The route solver — two corrections to spec section 3

The spec's pursuit ODE does not run as written. Both failures were reproduced against
the real `course_vq2` package before being fixed.

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
0.126 m** (spec acceptance is 0.55 m), 263 m long, peak lateral accel 8.6 m/s² < 12.

Yield over deformed courses: **~95%**, zero route failures — the rejects are the geometry
filter catching gates displaced into each other.

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

Images are `v000.png` upward, **sorted by max displacement**, so the mild cases come
first and the marginal ones cluster at the end; the point where they stop looking like
VQ2 is the number worth knowing. Every image is drawn on **identical axes** — per-image
autoscaling would rescale each plot to its own extent and hide the very differences being
reviewed.

Each frame carries the mapped course as a grey ghost (chain + its solved route), this
variation's route in teal, and displaced gates in rust with a leader back to where the
map puts them. Plan view on top, elevation below. Screen x is `-world_x` so the lap reads
left-to-right, with y flipped to match so handedness survives.
