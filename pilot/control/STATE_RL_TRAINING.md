# RL training — measured state

Written 2026-08-02. Everything here is a MEASUREMENT on the surrogate unless labelled
otherwise. **Nothing in this document has been flown live.** Branch: `reward-tuning`.

`TRAINING_ARCHITECTURE.md` is design intent; this is what happened when it ran.

## Headline

The RL policy beats the reactive baseline on per-gate accuracy by **2.3–2.5×**, at the
physical gate aperture, at both operating points tested:

| policy | difficulty 0.00 / speed_cap 0.50 | difficulty 0.30 / speed_cap 0.80 |
|---|---|---|
| baseline (reactive PID) | 0.307 | 0.266 |
| **runB1** | **0.705** | **0.678** |

Translated to what matters live — expected gates before a frame strike is `p/(1−p)`:
**~2.1–2.4 gates per life for runB1 against 0.36–0.44 for the baseline**, roughly 5–6×.

**Neither policy completes a course.** Completion over 20 gates needs a per-gate rate of
~0.989; the best measured is 0.705. The `compl` figures in the training logs (0.003–0.012)
are the mid-course-spawn artifact, not full courses — 25% of episodes spawn at a random
gate index, so a spawn at gate 19 of 22 needs only three gates.

## What was actually wrong

`run1` (5.05 M steps) sat at 0% completion for 77 straight updates. Three causes, all
measured, none of them the optimiser:

1. **The completion metric was unreachable by construction.** Courses are 18–22 gates and
   completion is per-gate-rate^20. Measured per-gate rate of the *hand-written baseline*
   was 0.291, so completion was ~10⁻¹¹. The curriculum promotes at 0.70 completion and so
   never promoted; `select.py` ranks on the same number and so ranked nothing.
2. **The reward had no altitude term at all.** The floor was a terminal discovered on
   contact. Instrumenting run1's checkpoint: **55.5% of episodes ended on the FLOOR**
   (34.5% gate frame, 8.6% ceiling). The policy flew at 14.5° bank / 14.1° nose-down
   commanding 0.278 throttle where holding altitude needed 0.315 — a ~2.1 m/s² sink — and
   its thrust bias had moved *down* from hover. It was optimising the reward correctly.
3. **The discount horizon was shorter than the time to hit the floor.** At γ=0.99 and
   45–65 Hz the horizon is ~1.8 s; the floor arrived ~2.4 s after the sink began, by which
   point −8 was discounted to 0.27 of face value. "Keep flying" was worth ~10 against a
   crash costing ~18.

## What changed

* **K-gate episodes** (`EnvConfig.gates_per_episode`) ending as a **truncation**, not a
  finish. Courses are still generated full-length, so `active/n_gates`, the lookahead
  slots and the corridor keep their full-course distributions. Nothing beyond ~4 gates
  ahead is observable anyway — `interface.N_GATES` is 3 plus a ribbon reaching one further.
* **Clearance potential** on floor/ceiling, `γΦ(s′) − Φ(s)`, saturating at `clear_ref_m`.
* **γ 0.99 → 0.997**, wired through to the env so shaping and PPO cannot disagree.
* **`margin_range_m` (0.10, 0.40) → (0.0, 0.40)** so difficulty 0 is the physical aperture
  rather than 30% tighter than it.
* **Curriculum promotes on pooled per-gate accuracy**, not completion, and grows K as an
  axis. Thresholds set against what is *achievable* (baseline 0.291), not against what a
  finished policy needs.
* **Per-gate timeout split from the wall clock.** They were ORed into one flag that
  `ppo.py` bootstrapped wholesale, crediting a stuck policy with the rest of the course.
* **`gates_this_episode`** reported instead of the absolute index, which had credited a
  freshly initialised network with 2.05 gates — the mean spawn offset.

## The reward-hacking episode, because it will recur

`runA` (30 M steps) had the progress term as textbook potential-based shaping,
`γΦ(s′) − Φ(s)` with `Φ = −rem`. Episode return climbed +10.7 → +12.4 across updates
72–88 while per-gate rate stayed flat at 0.07 and collisions at 0.94.

Cause: that form pays a **stationary** aircraft `rem·(1−γ)` every step. At a typical 25 m
and γ=0.997 that is +0.075/step against a real closure signal of ~0.09/step — a survival
bonus that *grows with distance from the gate*. Decomposing runA's +12.4: crossings +0.4,
collisions −7.5, clearance drift −1.8, **progress drift +15**. The drift was the entire
return.

Ng/Harada/Russell requires a **bounded** potential to be safe in practice, and `rem` is
not bounded. Progress now telescopes at γ=1. The clearance potential keeps its γ because
it saturates, so its drift is a harmless −0.009/step.

**The diagnostic signature**: return rising while task performance is flat. Return alone
is ambiguous — `grate` and `coll` together are what distinguish learning from hacking.

## Runs

| run | steps | difficulty | speed_cap | per-gate rate | note |
|---|---|---|---|---|---|
| run1 | 5.05 M | 0.0 | 0.5 | 0.029 | pre-fix baseline |
| runA | 30 M | 0.0 | 0.5 | 0.08 | reward-hacked the drift term |
| runB2 | 85 M | — | — | — | `clear_ref_m=4.5`; ~20% *behind* B1, single seed |
| **runB1** | **207 M** | **0.65** | **0.85** | **0.557** | the champion |

runB1 at 90 M was 0.59 at difficulty 0.30; the curriculum then spent accuracy on
difficulty, reaching 0.65 while holding ~0.56. That is the schedule working as designed —
it buys robustness, not completion.

`--ent-coef 0.005 → 0.001` dropped policy σ from 0.40 to 0.299. Exploration noise had been
contributing roughly twice the lateral error budget at the gate, so this is a real lever on
accuracy and is not exhausted.

## Open, in order of how much they matter

1. **Nothing has been flown live.** Every number here is surrogate-side. `README.md`'s
   warning stands: "great on surrogate, bad in sim" means suspect surrogate signs first.
2. **Does a collision invalidate a run, or just cost time?** Worth one live run and it
   decides *what gets submitted*. `EnvConfig.collision_terminates=False` now models the
   second regime so the question can be studied before spending the run — but the
   recovered state (zero velocity, 0.45 rad tumble) is an assumption, not a measurement.
3. **The recovery supervisor has never executed.** It cannot fire on the surrogate while
   collisions are terminal, which is exactly why (2) exists.
4. **P3 is still the hand-specified fallback.** Every transfer number here is against
   synthetic detections whose parameters are guesses with reasons.
5. **Completion needs per-gate ≈ 0.89+** and the best measured is 0.705. Whether that is
   reachable or is the architecture ceiling (no vertical-rate channel in the 73-D vector,
   k=6 frame stack ≈ 0.11 s) is untested. The thrust residual — commanding
   `hover/(cos φ cos θ) + Δ` so holding altitude is the zero action — is the untried
   intervention with the clearest mechanism.

## Reproducing

```
python pilot/control/train/train.py --env surrogate --name <run> \
    --total-steps 200000000 --n-envs 2048 --n-steps 128 \
    --curriculum-window 1000 --gates-per-episode 22 --ent-coef 0.001
```

Add `--no-curriculum --difficulty-start X --speed-cap-start Y` to freeze the schedule and
let accuracy climb instead of being spent on difficulty.

Checkpoints carry their own `speed_cap`; `policies/rl.py` reads and applies it, and the
env *clips* at `RATE_CAP × speed_cap` — so any evaluation must pass a `--speed-cap`
matching the checkpoint or it silently clips the policy's commands.
