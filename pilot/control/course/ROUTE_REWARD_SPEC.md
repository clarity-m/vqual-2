# Route-reward spec — dense progress reward on a solver-generated reference route

**For Alex.** 2026-08-02, ships with the corrected course (10-11 = 16.49 m — see the
README changebox; regenerate everything against this folder, not an older copy).

Proposal from Claire, written up by perception. It is a **training-time reward change
only**: the policy's inputs (73-D `interface.Observation`), outputs and the environment
dynamics are untouched. The reference route exists only inside the surrogate, where true
state is known, and is used only to compute reward. The policy never sees it — so there
is no observability violation of the VQ2 telemetry blocks, and nothing about the frozen
interface moves.

Why bother, in one paragraph: the current progress signal (~0.09/step) is sparse relative
to the decision rate, per-gate score has sat at ~0.32 on the measured course, and
`vq2_frac=1.0` means the policy is memorizing coordinates that carry known errors (the
6-7 quadrant branch is still sketch-sourced; 1-2 is contested; 10-11 was wrong by 17 m
until yesterday). A dense progress-along-route reward is the standard fix in the drone
racing RL literature (progress-along-centerline rather than gate-sparse), and regenerating
the route per episode over a **deformed** course forces the policy to locate the course
through its observations instead of reciting it.

---

## 1. Per-episode pipeline

```
seed -> deformed course (section 2) -> reference route (section 3) -> reward terms (section 4)
```

Everything derives from the episode's own sampled course. The route is never generated
from the nominal course while the episode flies a sampled one — reward and truth must
agree within an episode.

---

## 2. Deformation: what it is, exactly

Three layers. Layer A already exists in `course_vq2.sample()`; B and C are new and small.
"Same general shape" is not left to intuition — it is the acceptance test in 2.4.

### 2.1 Layer A — the measurement posterior (exists, use as-is)

`sample(seed, include_alt_hypotheses=True)`. Edge-space Gaussian draws at the measured
sigmas, whole chain re-solved by weighted least squares, so every sample is a
geometrically self-consistent course that could have produced our data. Tilt/yaw draws
included; gate 9 always leans, always the same way. Turn `include_alt_hypotheses` **on
for all episodes** — the 1-2 coin flip (8.32 vs 13.0 m) lives inside it and every episode
flies that leg.

### 2.2 Layer B — the systematic modes the Gaussian cannot express (new)

These are the sampler's own documented LIMITATION list, promoted from a warning into
draws. They are the error modes that actually worry us, and A never generates them.

* **Shared compass bias per chain segment.** Edges measured in one flight share that
  flight's compass error, so a whole leg can rotate together. Session IDs are not in this
  JSON (they are in `map_vq2.json` if you want exact grouping); the honest approximation:
  split the race chain into its contiguous bridge segments between braced loops —
  {0-1 … 6-7}, {9-10}, {15-16} — and add ONE shared draw `N(0, 1.5°)` to every edge
  bearing in a segment, on top of A's per-edge draws.
* **Discrete quadrant-branch flip on 6-7.** The one branch no measurement pins, and the
  worst one: wrong means the whole back half rotates ~90° about gate 6 (10 gates move,
  median 103 m). With probability **p = 0.10**: add +90° or −90° (coin flip) to edge
  6-7's bearing before the chain re-solve — the re-solve then rotates everything
  downstream, which reproduces the real failure exactly. Do **not** flip 15-16 (pinned by
  a full-bearing referee row, runner-up 88° away) or 2-3 (corroborated by two independent
  rows at 28.8° margin; if you want it anyway, p ≤ 0.03).
* Episodes carrying a flip are marked (`flip_6_7 = ±90°`) and exempt from the turn-angle
  acceptance check downstream of gate 6 — the flip is *deliberately* shape-breaking.
  That is its job: teach the policy that the memorized shape can be wrong and the gates
  it sees are the authority.

### 2.3 Layer C — shape-preserving augmentation beyond the uncertainty (anti-memorization)

Wider than what we measured, bounded by what keeps the course "the same course":

* **Global scale** `s ~ U[0.90, 1.10]` on all edge `d` and `dz`.
* **Low-frequency bearing warp**: `δψ_i = A · sin(2π f · i / 19 + φ)` over edge index
  `i`, with `A ~ U[0°, 6°]`, `f ∈ {1, 2}`, `φ ~ U[0, 2π)`. Smooth over the chain, so
  local turn geometry survives while the global shape flexes.
* **Height-profile scale** `U[0.85, 1.15]` on all `dz` (the 0→7 climb stays a climb,
  its steepness varies).

No independent per-gate jitter, ever — the README's argument stands: the map measured
edges, and independent gate jitter manufactures courses that contradict the braced loops.

### 2.4 "Same general shape", operationally — the acceptance test

After the chain re-solve, a sampled course is ACCEPTED iff:

1. race order 0→16 unchanged (true by construction);
2. every turn-per-gate within **±20°** of nominal (exempt: gates 7-16 in a
   flip-marked episode);
3. every edge length within **[0.7, 1.4]×** nominal and **≥ 5 m**;
4. the two big climb steps (5-6, 6-7 `dz`) keep their sign;
5. the route generator converges on it (section 3.3).

Reject → redraw with the next seed. Expected rejection rate is low (A alone passes
essentially always; C's bounds are set inside these limits); if you measure >10%
rejection, log it — that is the knobs disagreeing with the test, not noise.

### 2.5 Episode mixture (recommended, a knob)

| share | draw |
|---|---|
| 10% | nominal `load()` — the course as measured |
| 40% | A only |
| 30% | A + C |
| 20% | A + B + C |

Do not anneal toward nominal late in training — that re-teaches memorization in exactly
the phase where the policy consolidates.

---

## 3. The reference route — a critically damped pursuit ODE

Any generator meeting the acceptance test in 3.3 is fine (a spline fit would do); this
one is simple, deterministic and produces dynamically plausible geometry with two
parameters that mean something.

### 3.1 Definition

State `(p, v)`, integrated at `dt = 0.02 s` from the spawn point, tracking a moving
target `q` derived from the **sampled** course:

```
q(i, p) =  c_i − L·n̂_i   while the approach-side distance along n̂_i exceeds L
           c_i + L·n̂_i   after that (punch through the plane)
advance i -> i+1 when the plane of gate i is crossed inside the aperture

dp/dt = v
dv/dt = sat( ωn²·(q − p) − 2·ζ·ωn·v ,  a_max )      then cap |v| at v_max
```

with `n̂_i = gate_normal(i, course)` oriented along the race (use `yaw_fallback=
'bisector'` — the route needs only a coarse plane, and yaw noise at ±18° moves an
`L = 3 m` approach point by under a metre).

Parameters, recommended:

| param | value | meaning |
|---|---|---|
| ζ | 1.0 | critically damped — no overshoot by construction |
| ωn | 1.5 rad/s | stiffness of pursuit; the retry knob in 3.3 |
| L | 3.0 m | approach/exit standoff along the gate normal |
| a_max | 12 m/s² | keep the route inside what the plant can actually pull |
| v_max | 8 m/s to start | route aggressiveness; sweep upward as the policy improves |
| dt | 0.02 s | |

Resample the result to an arc-length parameterized polyline at **0.25 m** spacing.
That polyline — `route(s) -> (x, y, z)`, total length `S` — is the artifact the reward
consumes.

### 3.2 Determinism

Fully determined by (sampled course, parameter set). No RNG inside the generator —
resume/reproducibility safe, and two workers on the same seed get identical rewards.

### 3.3 Route acceptance (part of course acceptance, 2.4 item 5)

* crosses every gate plane within **0.55 m** of the sampled centre (inscribed-circle
   0.536 m margin for the 0.214 m bounding sphere, rounded);
* `κ·v² ≤ a_max` everywhere along the polyline (no impossible corners).

On failure: raise ωn by 25% and regenerate, up to 3 times; then lower v_max by 2 m/s
and try once more; then reject the course sample. The corners that will bind are the
known ones — turn ≥66° into an exit edge ≤12 m at gates 7, 9, 13.

---

## 4. Reward

Per step, replacing the current sparse progress term (gate-pass bonus, collision and
terminal handling all stay):

```
s_t   = arg min over s in [s_{t-1}, s_{t-1} + 5 m] of |route(s) − p_t|     (see below)
Δs    = clip(s_t − s_{t-1}, 0, v_max·dt)
d⊥    = |route(s_t) − p_t|
r_route = k_p · Δs · exp( −(d⊥ / w)² )
```

* **Windowed monotone projection** — the projection may only move forward, and only by
  5 m per step. This is what prevents shortcut credit (cutting a switchback and being
  re-projected far ahead) and backward reward farming. `s_0` initializes by global
  nearest point from spawn.
* **Corridor width `w` = max(2.5 m, 2·σ_xy_local of the upcoming gate).** LOOSE ON
  PURPOSE. The route is derived from sampled gate positions that are wrong by
  construction; a tight corridor trains the policy to trust the map through the back
  door, which is the disease this whole spec exists to cure. The corridor is a decaying
  multiplier, not a hard penalty — off-route flying earns less, it is not punished.
* **`k_p` scaled so the per-step magnitude at nominal pace matches the current ~0.09**
  progress signal — keeps PPO advantage scales, entropy/lr schedules valid. At
  `v_max = 8`: `Δs ≈ 0.16 m/step`, so `k_p ≈ 0.56`.

Two riders, from the perception transfer report (`STATE_PERCEPTION_FOR_CONTROL.md`):

* **Raise `k_nogate` 0.01 → 0.05 in the same retrain.** Frame-keeping is the biggest
  single lever on every observation-quality number we measured, and a route reward that
  pulls through corners geometrically will happily point the nose away from the next
  gate unless framing pushes back. These two terms must be tuned together — the route
  term must not be allowed to buy progress with blindness.
* **Do not add a gyro-rate detection penalty** — measured as a framing confound; the sim
  renders no motion blur (report §5).

---

## 5. What this does not fix

Surrogate→real transfer (the noise model is still the largest untested assumption) and
the live harness. Both are on perception's side of the fence and being worked
independently; ranked asks are in the transfer report §9. A policy trained under this
spec still exits through the same door — it is just carrying a map it no longer has to
trust.
