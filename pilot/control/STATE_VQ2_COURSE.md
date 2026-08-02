# VQ2 course in the surrogate — state, decisions, and what is still assumed

*2026-08-02. Covers the wiring of `course/` (the perception side's measured VQ2 map) into
surrogate training, plus a spec audit of the collision model done along the way.*

**Authority order:** code + `course/course_vq2.json` + checkpoint sidecars > this file >
`TRAINING_ARCHITECTURE.md` / handoff markdown. Where this file and the architecture doc
disagree about course generation, this one is newer. For anything about the **reward,
curriculum or evaluation**, `STATE_RL_TRAINING.md` is the authority — see §0.

---

## 0. Merged with `reward-tuning` — read this before §6 or §9

*Added 2026-08-02, after the merge.*

This document was written on a branch that forked from `a76c2a4` **in parallel with**
`reward-tuning`, and therefore described a surrogate with none of the reward work in it:
no clearance potential, `γ=0.99`, curriculum promoting on **completion** at 0.70,
`margin_range_m=(0.10, 0.40)`, the wall-clock and per-gate timeouts ORed into one flag that
`ppo.py` bootstrapped wholesale. That is the configuration `STATE_RL_TRAINING.md` records
as producing 0% completion for 77 updates with **55.5% of episodes ending on the floor**.

The branches are now merged. `env.py` merged without conflict — the two sides touched
disjoint regions — and the three `evalsuite/` conflicts were union merges of orthogonal
keyword arguments. What that changes for this document:

* **§6's run command was the pre-fix one.** Replaced; see §6.
* **§9's training rows are not evidence of learning.** `ev` is value-function fit and rises
  whenever the critic learns to predict return, *including* a return the policy is farming
  — `STATE_RL_TRAINING.md` §"The reward-hacking episode" is the worked example. Those rows
  are retained below as plumbing checks and relabelled.
* **"all episodes 17-gate" was misleading.** `start_probs=(0.50, 0.25, ...)` spawns 25% of
  training episodes at a uniformly random gate index. `n_gates` is 17; the number of gates
  *flown* is not. Evaluation no longer inherits this — `run_policy.EVAL_START_PROBS` pins
  eval to the start line, with the randomized starts opt-in behind `--random-starts`.
* **Completion is still the wrong promotion signal here, and is now unused.** 16 crossings
  means promoting at 0.70 completion needs a per-gate rate of **0.978**, against a
  best-ever measured 0.705. The merged curriculum promotes on pooled per-gate accuracy
  instead, and `select.py` ranks on it.
* **A `reward-tuning` checkpoint could not previously be loaded here at all.**
  `restore_env_config` iterates `dataclasses.fields(cls)` and *silently skips* unknown
  keys, so a `runB1` sidecar would have come back carrying the old reward config with no
  error. Post-merge the field sets agree.

Two consequences for VQ2-only that were not visible before the merge, both open:

1. **Nothing trains for time.** §5 justifies VQ2-only on the grounds that ranking is on
   time (spec §9.4), but `k_time` sits behind `enable_time_penalty`, which
   `curriculum.py` flips on at 80% completion — unreachable. A course-memorizing policy is
   never optimized for the objective that motivates memorizing it.
2. **Two independent conservatisms on the pass test now compound.** §4 finds the
   circle-vs-square test credits roughly a third to a half of the real aperture; the merge
   moves `margin_range_m` to `(0.0, 0.40)` so difficulty 0 is at least the *physical*
   aperture rather than 30% inside it. That removes one of the two, not both. Under
   VQ2-only the remainder applies at the same gates every episode.

---

## 1. What changed

`course.py` invented every course the surrogate ever trained on. That was correct while
the VQ2 map was unobservable — `course.py`'s own docstring cites spec §9.3 blocking gate
geometry and both pose streams. The perception side has since measured the course, so the
premise no longer holds.

| file | what |
|---|---|
| `surrogate/vq2course.py` | **new.** Measured courses in the dict shape `course.generate` already returns |
| `surrogate/env.py` | `EnvConfig` gains the `vq2_*` fields; `_reset_envs` calls `vq2course.mix` |
| `evalsuite/run_policy.py` | `build_config(..., vq2_frac=)` + `--vq2-frac`; forwarded by `select.py` and `stress.py` |
| `train/colab_vq2.ipynb` | **new.** Colab runner |
| `course/` | the perception side's package, committed (it was untracked, so it never reached a clone) |

`vq2_frac` defaults to **0**. The surrogate is byte-identical to before unless asked —
verified by stepping two `vq2_frac=0` envs on the same seed and comparing observations.

---

## 2. Why it matters — the generator was wrong in one direction

Measured from `course_vq2.json` (16 race edges, gates 0→16). **Numbers below are the
2026-08-02 corrected package** — see §2b for what the correction moved.

| | `course.generate` | measured VQ2 |
|---|---|---|
| gates | 18–22 | **17** |
| segment | easy 26–34 m, hard **18–26 m** | 8.32–22.46 m, mean 15.70, **median 15.26** |
| turn per gate | ≤20° easy, ≤80° hard | mean 32.3°, max 76.5°, 5 of 15 over 40° |
| elevation | ≤22°, `alt_revert=0.3` mean-reverting | sustained **+10.95 m climb over gates 0→7** |
| path length | — | **251.1 m** |

**11 of the 16 race edges are shorter than the shortest segment the generator can
produce.** Its floor is 18 m even at maximum difficulty — and after the correction the
generator's *longest* segment (34 m) now exceeds the real course's longest edge (22.46 m),
so the two distributions barely overlap at all.

Worse, the corners that decide this course pair a sharp turn with a *short exit* — about
one second of flight at cruise to reacquire, align and thread:

| gate | turn | exit edge |
|---|---|---|
| 7 | 75.8° | 11.42 m |
| 9 | 66.5° | 10.10 m |
| 13 | 76.5° | 12.31 m |

## 2b. The 2026-08-02 correction — edge 10-11 was wrong by 17 m

Perception re-measured edge 10-11: it read **33.84 m and is 16.49 m**. The old rows were a
real measurement of the *wrong pair* — they saw gate 12 at the far end from a single
vantage, so gate 11 and the whole tail 12→16 sat ~15.6 m too far along-course.

What moved: course length **268.22 → 251.13 m**; gates 11–16 all shifted (gate 11 by
15.6 m, 12–16 by 14.6–15.6 m); gate 11 dropped 1.97 m in z. A **new edge 10-12** (34.97 m)
makes 10-11-12 a closed triangle rather than two bridges, which *lowered* per-gate sigma
for 11–16 (2.11–2.38 → 1.98–2.18 m). The edge table went 19 → 20 pairs.

Any checkpoint trained before this flew a course whose back half is 15 m out of place.
Gate 0 remains the lowest gate, so the floor work in §3 is unaffected.

The generator draws 80° turns and it draws 18 m edges, but never the two together.

Two smaller consequences: the real elevation profile is a sustained **climb**, which
`alt_revert = 0.3` actively cancels — and the camera is tilted 20° **up** (spec §3.8), so a
climb keeps gates in frame where the generator's modelled ~20° descent hides them. And
`RIBBON_LOOKAHEAD` was tuned against an assumed "~28 m spacing" (`env.py`); at 16.8 m real
mean those distances reach much further down the course than intended. Not yet revisited.

---

## 3. Measured vs assumed

**Measured** (`course/README.md` is the authority): 17 gate positions, race order,
inter-gate distances / bearings / signed height steps, and the per-episode uncertainty the
sampler draws from. The number that matters for a gate-relative policy is
`sigma_xy_local_m` = **0.31–0.89 m** step-to-step, not the 2.4 m accumulated figure — that
one is a gauge artifact of anchoring the chain at gate 0.

**Assumed here, and labelled as such in the code:**

| thing | why | how handled |
|---|---|---|
| Floor position | the map's `z` is relative to gate 0, not the floor | `vq2_floor_clear_m`, randomized per episode |
| Ceiling | never measured | placed above the course's own high point by `vq2_headroom_m` |
| Gate plane yaw | three exported candidates disagree | `vq2_yaw_mode='mixed'` draws among them per episode |
| Gate tilt | magnitude confirmed 2026-08-02, axis still open | gate 9 **21–24°**, all others vertical, via `vq2_tilt_deg` |

Three notes on those:

**Floor. MEASURED 2026-08-02 — the bottom of gate 0's frame touches the ground.** With the
spec-exact 2700 mm outer frame that puts gate 0's centre at **1.35 m**. Gate 0 is also the
map's lowest gate: every other gate is at positive `z` relative to it (gate 4 is nearest at
+0.40 m, gate 7 highest at +10.95 m).

The old `vq2_floor_clear_m = (2.6, 4.5)` was never a measurement. Its derivation is in the
comment at `env.py:173` — *"floor_clear_m plus half the aperture or gates start
underground"* — so 2.6 is just the smallest value the **spawn clamp** tolerated. Measured,
the whole course had been floating **2.25 m** off the ground.

The fix is coupled and both halves are required:

```
--env-kwarg floor_clear_m=0.5 --env-kwarg "vq2_floor_clear_m=(1.30,1.45)"
```

`floor_clear_m` (default 1.8) sets the spawn clamp to `+0.6` → 2.40 m. Correcting only
`vq2_floor_clear_m` leaves every episode spawning at 2.40 m, **above the 2.18 m top of gate
0's aperture**. Measured across 256 courses: current config puts gate 0's frame bottom
2.25 m above ground; the paired fix puts it at +0.10 m with spawn altitudes from 1.10 m.

Why it matters: at the true floor height, gates **0, 3, 4, 5** sit at 1.35 / 2.54 / 1.75 /
1.99 m — all inside `clear_ref_m = 2.5`, where the clearance term is active and a floor
strike is live. Under the old setting they sat at 3.6–5.5 m, outside it entirely. The
opening third of the course was being flown with margin that does not exist.

Anchoring stays on the *lowest sampled* gate rather than gate 0, because accumulated `dz`
draws can put gate 4 below gate 0 in a given sample. Consequence: gate 0's centre averages
~1.43 m rather than exactly 1.35. An 8 cm conservatism, left alone.

**Yaw.** The map exports measured / grid-aligned / race-bisector azimuths that disagree,
and perception's guidance is to treat "enter along the normal" as a *soft preference* and
steer at gate centres plus the race line. So the disagreement is turned into domain
randomization rather than resolved by picking a side. Gates 8/12/13 have no trusted yaw and
fall back to the bisector — the map's own recommendation, and what `course.generate` has
always computed. The package's own `sample(randomize_yaw=True)` is **not** used: it assigns
a uniform 0–180° plane to every refused-yaw gate, which would face gates 8/12/13 in a random
direction every episode and make them unlearnable. Only positions come from `sample()`.

**Tilt. Now measured, including direction — no knob required.** The 2026-08-02 package
exports gate 9 at `tilt_from_vertical_deg 21.0`, `tilt_sigma_deg 5.0`, `tilt_clip_deg
[12, 30]`, and the part that actually matters: `tilt_lean_azimuth_deg 129.7` — the azimuth
the gate's **top** leans toward. Every other gate exports tilt 0 with its own *measured
residual* (1.5–4.0°) as sigma, which is noise rather than a prior over unknown geometry.

`vq2course` reads all of this from `pkg.sample()` directly. `cfg.vq2_tilt_deg` survives
only as a manual override for experiments, and using it **loses the lean azimuth** for that
gate, so prefer `None`.

**The lean-vs-roll question is closed: it is a lean.** What was measured is the azimuth the
top leans toward, which is exactly what `_normals` models. The square-aperture collision
test is still worth doing (§4) but is no longer blocked on this.

**The direction was the load-bearing half, and we had it backwards.** A tilt magnitude with
no direction is a coin flip, and a policy that guesses wrong aims at the frame instead of
the aperture. `_normals` built the elevation with a fixed `+sin(t)` and then flipped the
whole normal to face along the race — which made the lean direction a function of travel
direction, not of the measurement. Measured over 256 pool courses: **pre-fix, gate 9 leaned
the wrong way in 100% of them** (mean deviation 159.5° from the measured azimuth — not
random, systematically inverted, because the travel orientation is deterministic here).
Post-fix, 100% land on the correct side at a mean recovered azimuth of 128.7° against the
measured 129.7°.

The fix sets the elevation sign from the lean azimuth *before* the travel flip. That is
safe because the top-lean vector `−n_z·n_horiz` is invariant under `n → −n` — both factors
flip together — so orienting the normal along the race cannot undo the lean.

Recovered lean azimuth spreads ±22° across courses. That is correct, not slop: a lean must
be perpendicular to the gate plane (the map confirms this to 2.7°), so it follows our
sampled plane azimuth, which is itself drawn across the measured / grid / bisector
candidates plus jitter.

---

## 4. Spec audit of the collision model

Done while answering "are the collisions from the spec?" Recorded because the answer is
*partly*, and the gaps are not documented anywhere else.

**From the spec, exactly** (VADR-TS-003 issue 00.03): chassis 280×280×160 mm → 0.214 m
bounding-sphere half-diagonal (**§3.6**, not §3.7 as `env.py`'s comment says); gate outer
2700 mm, inner 1500 mm, depth 260 mm (**§3.7**).

**Not in the spec at all.** §3.2 says only *"a rigid-body drone flight model including
thrust generation, aerodynamic drag, gravity, and collision physics"*. There is no spec
definition of what constitutes a hit, whether one ends a run, floor/ceiling contact, safety
margins, or a corridor. `TRAINING_ARCHITECTURE.md` is honest about this — *"Collisions —
geometry, not physics"* — and still lists as an open live-run question whether a collision
invalidates a run or just costs time.

### Known deviations

**The spec's gate is square; the code tests a circle.** `_gate_geometry` reduces a crossing
to a radial distance `lat` and compares it against half-*widths* 0.75 and 1.35 — the
inscribed circle of a square. Consequences run in opposite directions:

- *Pass test is far stricter than reality.* `lat + r ≤ 0.75` credits a pass only inside a
  disc, but the real 1500 mm square admits the drone anywhere within a square of half-width
  `0.75 − 0.214 = 0.536`, whose corners sit at radius 0.758. With a mid-range margin the
  surrogate credits roughly a third to a half of the genuinely passable aperture and scores
  the rest as crashes. Fail-safe, but it trains much tighter centring than the gate demands.
- *Outer test is slightly lenient.* `lat − r < 1.35` calls anything beyond a clean miss, but
  real frame material reaches `1.35 × √2 ≈ 1.91 m` at the diagonals. A diagonal strike in
  roughly the 1.66–1.91 m band is a real collision scored as a miss. Thin sliver, but it is
  the fail-open direction.

**Other gaps:** gate depth (260 mm) is dropped — the gate is a zero-thickness plane, with
the sphere margin absorbing it. Floor and ceiling are invented (nothing in the spec defines
a ceiling). **Obstacles are absent entirely** — spec §3.1 lists "vertical and horizontal
obstacles", "boundary elements" and "terrain and environmental structures"; the surrogate
models none, and the map explicitly excludes them too. Backwards crossings are undetected
(`cross = d0 < 0 & d1 ≥ 0` is directional). And termination-on-contact is a training
decision, not spec behaviour — "collision physics" implies a rigid-body response.

### Why the square/circle question is now load-bearing

If gate 9's ~21–24° turns out to be an **in-plane roll** of the frame about its own normal
rather than a **lean** of the plane, the surrogate cannot represent it at all — a radial
test is rotationally symmetric about the normal, so rolling a square aperture is invisible
to it. The `course_vq2.gate_basis` model treats tilt as a lean (the normal tips), which is
representable. **Open question to perception.**

---

## 5. VQ2-only training

`vq2_frac=1.0` is a legitimate strategy: the course is fixed and deterministic (§3.5),
attempts are unlimited and ranking is on time (§9.4), so overfitting to it is the point.
Three things change beyond the flag.

**Eval must match, or checkpoint selection is meaningless.** `build_config` had no VQ2 knob,
so `select.py` / `run_policy.py` / `stress.py` scored checkpoints on procedural courses.
Ranking a VQ2-only policy against courses it never trained on picks the wrong checkpoint
with no sign that it did. Now `--vq2-frac`, and it must match the training value.

**The contested 1-2 edge must be sampled.** Accepted 8.32 m rests on five rows from one
flight; a refused 34-row channel reads 13.0 m. `course/README.md` says to enable it "if the
policy will ever fly 1→2" — under VQ2-only every episode does. `vq2_alt_hypothesis_p`
(default 0.5) puts the rival in ~25% of pool courses; measured 258 of 1024.

**`difficulty` loses half its meaning.** Course geometry is fixed now, so difficulty drives
only perception noise, collision margin and corridor radius. `speed_cap` is unaffected.
Annealing *map uncertainty* instead was considered and rejected: the value of sampling is
robustness to a map that may be wrong, and starting on the nominal course teaches the
opposite.

### The one risk sampling cannot cover

`course/README.md`: edges from one flight share that flight's compass bias, and the 2-3,
6-7 and 15-16 legs take their mod-90 quadrant from Claire's sketch rather than a shared
measurement — so a wrong branch rotates a whole leg by 90°, *"a discrete failure a Gaussian
cannot express"*. It calls the sampled envelope **"a lower bound"**.

A pure-VQ2 policy has memorized a track that may be wrong in a way no amount of sampling
reaches. `vq2_frac=0.9` keeps gate-seeking alive for 10% of the run as insurance. Left at
the operator's discretion; default is 1.0 in the notebook because that is what was asked
for.

---

## 6. How to run

```bash
# post-merge. --gates-per-episode and --ent-coef are NOT optional: see STATE_RL_TRAINING.md
python pilot/control/train/train.py --env surrogate --name vq2_run1 \
    --total-steps 200000000 --n-envs 2048 --n-steps 128 \
    --curriculum-window 1000 --gates-per-episode 17 --ent-coef 0.001 \
    --env-kwarg vq2_frac=1.0

# rank checkpoints — --vq2-frac MUST match training
python pilot/control/evalsuite/select.py --baseline \
    --ckpt vq2_run1_s5046272 --vq2-frac 1.0 --seeds 0-49
```

`--gamma` now defaults to 0.997 and is wired through to the env, so shaping and PPO cannot
disagree. `--gates-per-episode 17` matches the VQ2 course length; the curriculum grows K
from `gates_start` regardless, so a smaller value is a legitimate warm start, not a cap.

**Colab:** `train/colab_vq2.ipynb`, or open it directly at
`colab.research.google.com/github/clarity-m/vqual-2/blob/worktree-vq2-course-training/pilot/control/train/colab_vq2.ipynb`.
The repo is public, so the token prompt takes a blank. Use a **CPU runtime** — the env is
vectorized NumPy and dominates the step, so a GPU only touches the PPO update. The notebook
symlinks `checkpoints/` into Drive so a disconnect does not lose the run; `--resume` then
continues it. Budget longer than local: Colab's free VMs are 2-core.

Free-tier Colab generally permits one active session, so the "parallel hypotheses" plan
needs Pro or separate accounts.

### Knobs

All via `--env-kwarg`. `vq2_frac`, `vq2_pool_size` (2048), `vq2_pool_seed`, `vq2_yaw_mode`
(`'mixed'` | `'bisector'`), `vq2_yaw_jitter_deg` (4.0), `vq2_alt_hypothesis_p` (0.5),
`vq2_tilt_deg` (`None`), `vq2_floor_clear_m` (2.6, 4.5), `vq2_headroom_m` (1.75, 4.0).

---

## 7. Implementation notes worth knowing

**The pool is pre-sampled and cached per process.** `course_vq2.sample()` costs ~1.8 ms
because it perturbs the edge table and re-solves the whole chain — which is what keeps a
sample geometrically self-consistent instead of jittering gates independently. At a
vectorized env's reset rate, calling it live would dominate the step. The cache is keyed on
the fields that shape the pool and deliberately **not** on difficulty, speed_cap or the env
seed: `train.py` rebuilds the entire env on every curriculum promotion, so a per-instance
pool would cost seconds each time and silently change the course set mid-run.

**Frames.** Map is right-handed z-up; surrogate is NED z-down. Conversion is
`(x, y, z) → (x, −y, −z)`, a 180° rotation about x — det +1, so no mirroring. The map's
missing absolute heading and k·90° ambiguity are **irrelevant here**: the surrogate's world
frame is internal, never encoded into an observation, and a rotation about the vertical is
unobservable when gravity is the only absolute reference.

**`sys.path` tripwire.** `pilot/control/course/` now shadows `surrogate/course.py` — both
are importable as `course`, and `pilot/control/` sits *ahead* of `surrogate/` on the path.
The procedural generator wins today only because `course/` has no `__init__.py`, making it a
namespace-package portion, which loses to a real module. **Adding an `__init__.py` there
would silently swap the two.** `vq2course.py` loads the package by file path so it does not
depend on this, but anything else importing `course` does.

---

## 8. Open items

1. ~~**Gate 9 tilt — lean or in-plane roll?**~~ **CLOSED 2026-08-02: it is a lean**, and
   the lean azimuth (129.7°) is exported and now consumed. See §3. The one outstanding
   ask is a *lateral* pass at gate 9 — every view we have is head-on (θ ≤ 19.2°), which is
   why the magnitude is ±5°; above θ ≈ 45° the PnP-free edge channel measures the lean
   directly. Perception notes the detector's aspect-ratio filter discards the most oblique
   views, so it must be loosened for that pass.
2. ~~**Corrected `course_vq2.json`** pending from perception.~~ **LANDED 2026-08-02.** The
   package in `course/` is the corrected one: edge 10-11 re-measured (§2b) and gate 9's
   tilt plus lean azimuth exported. Regenerate anything derived from the old copy.
3. ~~**Floor reference unmeasured.**~~ **Answered 2026-08-02**: gate 0's frame bottom
   touches the ground → centre at 1.35 m. See §3. Requires the paired `floor_clear_m`
   change; any VQ2 run started before this is training on a course floating 2.25 m off the
   ground, with the opening four gates outside the clearance band entirely.
4. **Obstacles.** Spec §3.1 lists them, the surrogate has none, the map excludes them.
   `NOTES.md` mentions support columns in the hangar. Largest remaining fidelity gap.
5. **Square-aperture collision test.** Worth doing regardless of item 1 — see §4.
6. **`RIBBON_LOOKAHEAD` retune** against 16.8 m real mean spacing, not the assumed 28 m.
   `[4, 8, 14, 22, 32, 45]` against a 15.3 m median edge puts the last two slots 2–3 gates
   downcourse. On a *fixed* course that is either free information or wasted capacity, and
   which one is untested.
6b. **Spawn back-off uses a course mean on a 4:1 edge spread.** `vq2course` sets `seg_len`
   to the per-course mean edge and `_reset_envs` draws `back = U(0.45, 0.95) × seg_len`
   → 7.5–15.9 m, against real edges of 8.3–33.9 m. Measured: **22% of mid-course spawns
   (~5% of all training episodes) back off farther than the preceding gate is away**,
   concentrated on `a0` = 2, 8, 10 (P = 0.90, 0.54, 0.70) — and 8 and 10 are the exits from
   two of the three sharp corners §2 identifies as deciding the course. At a 73° corner the
   back-off vector does not point at the previous gate, so this is not necessarily a spawn
   *through* a frame and may read as useful recovery randomization; it is nonetheless
   unintended and lands hardest on the gates that matter. Fix is to use the per-gate
   incoming edge rather than the course mean. Does not affect evaluation, which now pins
   to the start line.
7. **`pilot/PRODUCER.md`** referenced by perception but not in the repo.
8. **Observation-stream comparison** — perception offered to diagnose input chatter. Note
   that a *surrogate* dump cannot show association flips, PnP normal flips or range-source
   switching: `surrogate/detect.py` is a synthetic noise model, not a detector. The useful
   version is dumping surrogate and `Producer` streams over similar geometry and comparing,
   which would validate the noise model against the real thing. Not yet done.

---

## 9. Verification record

Run on this branch, 2026-08-02:

| check | result |
|---|---|
| `surrogate/selfcheck.py` | **17/17 PASS** |
| `train/selftest.py` | **ALL CHECKS PASSED** (incl. resume round-trip) |
| `vq2_frac=0` regression | observations byte-identical to pre-change over 50 steps |
| checkpoint config round-trip | survives `env_config_dict` → `restore_env_config`, incl. the tilt dict |
| floor/ceiling clearance | all gates clear over 512 courses (lowest 2.603 m vs 2.55 m required) |
| contested edge sampling | 258 of 1024 pool courses on the 13.0 m rival (~25%, as designed) |
| training, `vq2_frac=0.8` | *plumbing only.* 131k steps, ~9.7k FPS. `ev` is critic fit, not performance |
| training, `vq2_frac=1.0` | *plumbing only.* 65k steps. `n_gates == 17`; gates *flown* not measured |

Re-run on the merged branch, 2026-08-02:

| check | result |
|---|---|
| `surrogate/selfcheck.py` | **17/17 PASS** |
| `train/selftest.py` | **ALL CHECKS PASSED** (incl. both resume paths) |
| `vq2_frac=0` regression | `n_gates` 18–22, obs finite — procedural path untouched |
| `vq2_frac=1.0` on merged env | `n_gates == 17`, clearance potential finite, per-gate counters live |
| training, `vq2_frac=1.0` | 393k steps, ~11.5k FPS, **`coll` 0.969 → 0.52**, `grate` 0 → 0.007, `ent` 2.75 stable, `kl` ~0.003 |

That last row is a **plumbing check, not a result** — 393k steps is ~0.2% of a real run and
`grate` 0.007 means nothing yet. What it does show is the collision rate falling under the
merged reward where the pre-merge reward had no altitude term at all, and `gate_rate`
driving the curriculum in place of completion.

**Known pre-existing gap (not introduced by the merge):** `course/verify_course.py` part (b)
raises `FileNotFoundError` on `pilot/control/perception/map_vq2.json`, which is not in the
repo on any branch. Part (a) passes.
| throughput | no regression vs procedural (~24.9k raw env-steps/s at n=256) |
| eval CLI | `--vq2-frac` parses and forwards through all three entry points |

**Not verified:** the Colab notebook has never been executed on Colab. Its JSON structure
and every code cell's syntax are checked, and every command in it was run locally, but the
Drive mount and `#@param` form fields are Colab-side behaviour taken on trust.
