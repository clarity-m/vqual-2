# VQ2 course in the surrogate — state, decisions, and what is still assumed

*2026-08-02. Covers the wiring of `course/` (the perception side's measured VQ2 map) into
surrogate training, plus a spec audit of the collision model done along the way.*

**Authority order:** code + `course/course_vq2.json` + checkpoint sidecars > this file >
`TRAINING_ARCHITECTURE.md` / handoff markdown. Where this file and the architecture doc
disagree about course generation, this one is newer.

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

Measured from `course_vq2.json` (16 race edges, gates 0→16):

| | `course.generate` | measured VQ2 |
|---|---|---|
| gates | 18–22 | **17** |
| segment | easy 26–34 m, hard **18–26 m** | 8.3–33.9 m, mean 16.8, **median 15.1** |
| turn per gate | ≤20° easy, ≤80° hard | mean 30.5°, max 76.4°, 5 of 15 over 40° |
| elevation | ≤22°, `alt_revert=0.3` mean-reverting | ≤18.8°, sustained **+10.95 m climb over gates 0→7** |
| path length | — | 268.2 m (217.2 m straight extent) |

**10 of the 16 race edges are shorter than the shortest segment the generator can
produce.** Its floor is 18 m even at maximum difficulty.

Worse, the corners that decide this course pair a sharp turn with a *short exit* — about
one second of flight at cruise to reacquire, align and thread:

| gate | turn | exit edge |
|---|---|---|
| 7 | −73.5° | 11.2 m |
| 9 | +66.6° | 10.1 m |
| 13 | −76.4° | 12.3 m |

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
| Gate tilt | shipped prior overturned, correction pending | **held vertical**; `vq2_tilt_deg` takes it later |

Three notes on those:

**Floor.** Anchored on the course's **lowest** gate, not gate 0. Gate 0 is the map's origin
but not its low point, and a sampled course can put a gate below it because the per-edge
`dz` draws accumulate. Anchoring on gate 0 put gates 0.8 m underground; caught by an
assertion, not by reading.

**Yaw.** The map exports measured / grid-aligned / race-bisector azimuths that disagree,
and perception's guidance is to treat "enter along the normal" as a *soft preference* and
steer at gate centres plus the race line. So the disagreement is turned into domain
randomization rather than resolved by picking a side. Gates 8/12/13 have no trusted yaw and
fall back to the bisector — the map's own recommendation, and what `course.generate` has
always computed. The package's own `sample(randomize_yaw=True)` is **not** used: it assigns
a uniform 0–180° plane to every refused-yaw gate, which would face gates 8/12/13 in a random
direction every episode and make them unlearnable. Only positions come from `sample()`.

**Tilt.** The shipped JSON gives gates 8 and 9 a uniform 0–20° prior. Perception has since
overturned it — gate 9 measures **~21–24° across four sessions**, everything else vertical.
21–24° is *outside* the shipped prior, so the current file cannot sample the truth at all;
training on it would bake in an artifact. Held at vertical until the corrected JSON lands.

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
# local, ~35–40 min for 20M steps at ~9.4k FPS on a laptop CPU
python pilot/control/train/train.py --env surrogate --total-steps 20000000 \
    --n-envs 256 --n-steps 128 --frame-stack 6 --name vq2_run1 \
    --env-kwarg vq2_frac=1.0

# rank checkpoints — --vq2-frac MUST match training
python pilot/control/evalsuite/select.py --baseline \
    --ckpt vq2_run1_s5046272 --vq2-frac 1.0 --seeds 0-49
```

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

1. **Gate 9 tilt — lean or in-plane roll?** Decides whether a knob suffices or the
   square-aperture collision test is required. Asked; unanswered. Tilt held vertical
   meanwhile.
2. **Corrected `course_vq2.json`** pending from perception (gate 9 ~21–24°, all others
   vertical). Lands as `--env-kwarg "vq2_tilt_deg={9:(21.0,24.0)}"`, no code change.
3. **Floor reference unmeasured.** One measurement of any gate's height above the hangar
   floor collapses `vq2_floor_clear_m` from a randomized assumption to a fact.
4. **Obstacles.** Spec §3.1 lists them, the surrogate has none, the map excludes them.
   `NOTES.md` mentions support columns in the hangar. Largest remaining fidelity gap.
5. **Square-aperture collision test.** Worth doing regardless of item 1 — see §4.
6. **`RIBBON_LOOKAHEAD` retune** against 16.8 m real mean spacing, not the assumed 28 m.
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
| training, `vq2_frac=0.8` | 131k steps, ~9.7k FPS, ev 0.07 → 0.82, coll 0.98 → 0.90 |
| training, `vq2_frac=1.0` | 65k steps, ev 0.19 → 0.58, all episodes 17-gate |
| throughput | no regression vs procedural (~24.9k raw env-steps/s at n=256) |
| eval CLI | `--vq2-frac` parses and forwards through all three entry points |

**Not verified:** the Colab notebook has never been executed on Colab. Its JSON structure
and every code cell's syntax are checked, and every command in it was run locally, but the
Drive mount and `#@param` form fields are Colab-side behaviour taken on trust.
