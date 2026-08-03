# Retrospective — VQ2 control, 2026-08-03

Written at the deadline, from measurements taken the same day. Every number here is
reproducible from artifacts in the repo; where something is inferred rather than measured
it says so.

The short version: **no single modelling mistake cost this project the deadline. A
diagnosis gap did.** The failure mode was never measured directly, so the curriculum
optimised three axes that were not binding while the one that was — control authority
against the course's actual corners — stayed frozen at half strength for 200M steps.

---

## 1. The plumbing failure, first, because it nearly cost more than the deadline

`STATE_PERCEPTION_FOR_CONTROL.md` — perception's measured answers over **13,008
ground-truth gate instances**, the single most valuable empirical artifact in the project
— existed only as an **untracked file inside a worktree directory** (`perception-error.md`
in `vq2-reward-merge`). It was rescued at 23:05 on 2026-08-02 by commit `4a2ea73`, whose
entire purpose was catching files that "would have gone with the folders, silently, with
nothing in git to notice."

The same commit rescued `ablate_vertical_rate.py` (the observability probe the whole
vertical-rate architecture case rests on) and `runB1_s205258752` — a **trained 205M-step
checkpoint that was in no branch at all**.

Three consequences:

* **The raw data is gone.** The commit records that the analysis scripts and the 13,008-row
  table lived in a job tmp dir. Only the prose summary survived, so the measurements can no
  longer be re-analysed — only re-quoted.
* **`surrogate/noise.py` was never updated.** Its last two commits (`864869f`, `7413239`)
  are both about the *mechanism* for scaling noise; nobody touched the numbers, because the
  person who could have did not know the measurements existed. Training therefore ran for
  the entire project against a hand-specified fallback that the measurements contradict in
  three places: `p_detect` 0.62–0.92 modelled vs **0.53–0.60** measured, ρ(0.1 s) 0.954 vs
  **0.43**, and a size-fallback range bias documented as long that is **3.5 m short**.
* **The gap was known and not closed.** `ROUTE_REWARD_SPEC` §4 cites that document for both
  of its riders, and the spec review explicitly flagged it as missing from the repo. The
  absence was logged as a finding and still took until the final night to act on.

CLAUDE.md's "Do not open new git worktrees. Background isolation has been turned off" is
this lesson already written down — after the cost. **The rule that would have prevented it
is narrower: work products from an isolated agent are not done until they are committed,
and a measurement is not delivered until it has changed the model it invalidates.**

## 2. The diagnosis gap

`viz3d.py` existed the whole time. Its own docstring makes the case:

> `_probe_baseline.py` prints one line per seed: a cause and a miss vector. That says a gate
> frame was struck 0.9 m off-centre. It cannot say what approach put the aircraft there,
> whether the gate was ever in camera frame, or how near the path ran to the corridor wall
> — and those are exactly the live hypotheses.

It was first run against a trained checkpoint on **the last day**. Twelve seeds, ninety
seconds, and the result was unambiguous: **12/12 episodes ended on a gate frame**, median
miss **0.590 m** against a usable margin of **0.536 m**, with **76% of the squared miss on
the vertical axis**. That is the entire story of the final two weeks, and it was one
command away throughout.

Why it did not happen is worth recording, because it is not carelessness. In perception the
ground truth *is* images, so the work forces looking. In control the surrogate emits
numbers, and `grate=0.458` reads like a measurement — three decimals, a 1000-episode window
— while being a summary in which the actual failure is invisible. **Nothing in the loop ever
forced anyone to look at the aircraft.**

The transferable rule: *any metric that summarises over episodes needs a paired artifact
that does not.* Not more logging — logging is what we had. A rendering, at every archive
checkpoint, that a human glances at for ten seconds.

## 3. The curriculum froze, and the expensive part was not the part we noticed

`gate_rate_promote` shipped at **0.60**, justified in `curriculum.py` as "roughly twice the
baseline" from a measured 0.291. That figure was stale. Measured 2026-08-03 with
`_probe_noise_ceiling.py` (120 seeds each): the reactive baseline is **0.412** under
pad-only spawns and **0.350** under the old mix. The trained ceiling is **0.458**.

So the gate sat *above* what the architecture could reach, and nothing ever promoted.
`vq2_ladder1` at 95M steps: `n_promotions: 0`, `n_demotions: 0`, `n_stalls: 8`,
`_since_change: 340` — the curriculum changed nothing across its entire life.

The obvious cost was `gates_per_episode` pinned at 3, so the policy only ever saw gates 0–2.
**The larger cost was `speed_cap`.** Over `vq2_arch1`'s complete 200M-step log, 762 updates:

```
speed_cap  unique [0.5]
difficulty unique [0.0]
```

`speed_cap` is misnamed. It does not cap velocity — it scales the two **rate** channels
(`ppo._to_physical`: `a[:, :2] *= speed_cap`), so it is an authority limit on roll and
pitch. At 0.5 the maximum commanded body rate is **79 °/s**, against a course that needs:

| gate | turn | entry edge | bank at 8 m/s |
|---|---|---|---|
| 2 | 55.8° | **8.3 m** (shortest on the course) | 27° |
| 7 | 75.8° | 16.4 m | **42°** |
| 9 | 66.5° | 14.4 m | **40°** |
| 13 | 76.5° | 13.3 m | **40°** |

Gate 2 is the first real corner and has the shortest entry edge on the course — about 1.0 s
of flight at 8 m/s. Establishing 27° of bank at 79 °/s consumes 0.35 s of that before any
lateral correction, and the rate budget is **shared** between roll and pitch, so a policy
correcting the dominant vertical error is rate-starved on both axes simultaneously.

A policy that cannot physically fly the corners will not reach the back half however many
gates the episode allows. **`speed_cap=0.50` printed on every log line for 200M steps and
read as normal.**

## 4. What is actually binding: geometry, not perception

Measured 2026-08-03, same policies, everything else held:

| intervention | effect on `gate_rate` |
|---|---|
| perfect detections (`noise_scale` 1.0 → 0.0) on `vq2_pass4` | 0.302 → 0.231 (flat, within 1.5 se) |
| aperture 1.0× → 2.0× on `vq2_arch1` | **0.440 → 0.741** |

Gates per episode on the aperture sweep went **0.82 → 2.58**. Perfect information buys
nothing; geometry buys everything.

This **corrects an inference made earlier the same day.** The fitted miss distribution is
Rayleigh with σ = 0.702 ± 0.010 m (nine points, ±1.4%), and that σ was decomposed into
~0.46 m perception and ~0.53 m control from `pos_noise_frac` at a ~10 m commitment range.
If that were right, removing sensor noise should have cut σ to ~0.53 and lifted `grate` from
0.25 to ~0.40. It did not move. **The miss is dominated by control and trajectory dynamics,
not by where the gate is reported to be.** The policy is not flying accurately to a noisy
estimate; it is flying inaccurately to a good one.

Caveat: `vq2_pass4` is a weak checkpoint (`k_collision_gate = 5.0` made clipping cheaper
than a floor strike, removing the incentive to thread). The same sweep on `vq2_promote1`
would be needed before treating this as settled for the architecture generally.

## 5. Things that worked, so they are not discarded with the rest

* **The vertical-rate and thrust-residual architecture did what it was built for.** Floor
  strikes went **0.73 → 0.03** per episode; ceiling to 0.00. The diagnosis behind it (the
  73-D vector reports no vertical rate; `thrust ∝ 1/(cos roll·cos pitch)` is smaller than
  the policy's own exploration noise) was correct and the fix held.
* **`test_train_deploy_agreement.py`.** Six combinations of {consecutive, dilated} ×
  {affine, residual} × {plain, vertical-rate}, all at 0.000e+00. The failure it guards — a
  checkpoint that scores well and flies differently — never happened.
* **The sign-convention discipline.** The mirrored body-rate convention, the
  `ATTITUDE`/`ODOMETRY` per-field correction, the rule that rate flips live in the link
  layer only. None of these bit during this work, which is the point.
* **Checkpoints self-describe.** `frame_offsets`, `thrust_residual`, `vertical_rate`,
  `speed_cap` and now `gate_inner_scale` are written into the checkpoint, so `RLPolicy`
  reconstructs the right architecture with no flags. This repeatedly prevented silent
  mismatches during evaluation.

## 6. Failure modes migrate, and nobody re-diagnosed

The sequence, visible only once per-cause collision logging existed (added 2026-08-03,
commit `75ec012` — before that the 55.5/34.5/8.6 split had to be *reconstructed* post-hoc
from terminal altitude):

```
ceiling 0.24  ->  floor 0.73  ->  gate frame 0.95
```

Each fix worked and exposed the next constraint, and the runs kept going. `arch1` spent
200M steps at 0/12 completion after its binding constraint had changed. **Re-diagnose after
every fix; the number that mattered last week is not the number that matters now.**

## 7. Exploration was flagged as suspect and never tested

`log_std` is state-independent, so the entropy bonus has zero gradient into either trunk —
exploration is a fixed isotropic blob the policy cannot modulate by situation. Over
`arch1`'s run, entropy fell monotonically **2.36 → −1.78**, and `grate` flattened exactly as
it crossed zero near 90M steps (`+0.004` over the final 50M). `HANDOFF_ARCHITECTURE.md`
listed this as open item #4. No `--ent-coef` arm was ever run.

## 8. What to do differently

1. **Render before you train.** Twelve seeds through `viz3d.py` at every archive
   checkpoint. If the endings are all one cause, that cause is the project.
2. **Every curriculum threshold is a measured number with a date.** `gate_rate_promote`
   0.60 against a 0.458 ceiling froze four axes at once. Measure the baseline and the
   current ceiling, then set the gate below it.
3. **Log per-cause, not per-event.** One boolean `collision` flag hid an entire migration
   of failure modes for the length of the project.
4. **Name things after what they do.** `speed_cap` limits control authority, not speed. A
   correctly named `rate_authority_frac` would have been questioned the first time someone
   read 0.50 on a log line.
5. **A measurement is not delivered until the model it invalidates has changed.**
   `perception-error.md` existed, was cited, was flagged as missing, and still never reached
   `noise.py`.
6. **Diagnostics before duration.** `arch1` spent 200M steps (~2 VM-hours) establishing a
   plateau that one 90-second render explained. The project knew the bottleneck was
   hypotheses/hour and still ran hypothesis-last.

## 9. State at the deadline

* **Best artifact: `vq2_promote1`**, `gate_rate` ~0.485 at `difficulty` 0.25 /
  `speed_cap` 0.70 — above the measured reactive baseline of 0.412, which is the bar
  CLAUDE.md sets for the deliverable ("beats the reactive baseline"), though not a completed
  lap.
* **No policy completes a lap, and none is close.** At `gate_rate` 0.46 a 22-gate lap is
  ~3×10⁻⁸. Ten percent completion needs ~0.90 per gate; fifty percent needs ~0.97.
* **`vq2_aperture1` was mid-anneal at the deadline.** Any checkpoint with
  `gate_inner_scale > 1.0` is **not shippable** — it was optimised against a hole larger
  than the real one. `train.py` warns at exit and the value is stamped in `env_config`;
  check it before submitting anything.
* **`vq2_pass4` is not a candidate** — `gate_inner_scale` 1.454, and on real geometry it
  averages 0.17 gates.

## 10. The highest-value next experiments, ranked by evidence

1. **`speed_cap = 1.0` from the start.** Never tested. The course demands 40°+ banks at
   gates 7, 9 and 13 and the policy has had 79 °/s for its entire existence.
2. **Finish the aperture anneal.** 0.44 → 0.74 measured, and it triples gates/episode,
   which is the first time any policy would see the back half of the course.
3. **`--ent-coef` sweep.** The one open item with a clear mechanism and zero runs.
4. **Fold the measured perception numbers into `noise.py`.** The model is optimistic on
   continuity and wrong in the sign of the range bias; every transfer estimate rests on it.
5. **The crossing-error decomposition.** At plane crossing, distance from the true centre
   versus from where perception said the centre was. Total minus tracking error is the
   sensor's share, with no distribution shift — the honest version of the
   `noise_scale = 0` test, which is confounded for any trained policy and now says so.
