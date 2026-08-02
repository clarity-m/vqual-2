# Controls pipeline — current state (for humans)

**As of 2026-08-01**, built from what’s actually on disk under `pilot/control/` — source, `plant.json`, training logs, and checkpoint sidecars — not from older handoff writeups (those lag the code in several places).

---

## What we’re trying to ship

One controller that flies the race sim better than a simple reactive baseline. Perception fills a fixed 73-number observation; control returns roll rate, pitch rate, and thrust. Yaw is left to an automatic “look at the gate” servo. Nothing in the policy is allowed to use absolute world position or heading — those are blocked on race day.

Live sim time is scarce (~15–25 runs per session, one UDP port). Almost all tuning and learning therefore happens in a NumPy **surrogate**: a fitted drone model + fake courses + fake camera detections that still produce the same 73-D observation.

---

## Big picture

```
Recorded flights  →  Fit drone model  →  Surrogate races  →  Train / tune policy
   (P0)                 (P1 ✓)              (P2 ✓)            (T4 / baseline)
                              ↑                    ↓
                     Replay validation      Eval on held-out seeds
                              ↑                    ↓
                     Detection noise         Deploy + recovery supervisor
                     (P3: still guessed)         (D6: coded, not proven live)
```

| Piece | Status in one line |
|---|---|
| Drone physics model | **Done** — refit after card-2 flights; validated |
| Surrogate simulator | **Built** — self-checks exist and are meant to pass |
| Detection noise model | **Guessed** — deliberately pessimistic placeholders |
| Reactive baseline | **Written & evolving** — not yet a reliable lap-finisher on the surrogate |
| PPO training | **Running / ran** (~5M steps) — **never completed a course** |
| Live deployment path | **Code exists** — not evidenced end-to-end on disk |

---

## What’s solid: the plant

The drone model in `plant.json` is the foundation everything else sits on.

- Horizontal drag fits extremely well (held-out R² ≈ 0.99).
- Vertical thrust is a **measured lookup table** (21 points from nearly-off throttle to full), not a hand-clamped formula. Hover lands at **0.270** throttle, matching flight notes.
- Vertical drag (`kz ≈ 0.0436`) comes from special “apex” maneuvers that park the throttle and coast through zero vertical speed — so thrust and drag don’t trade off against each other the way they do in racing flight.
- Rate loops are nearly ideal (gains ~0.90–0.97) with ~10 ms delay.
- Open-loop replay of recorded commands is the mandatory safety check before trusting the model for tuning: a sign error can look like a good fit and then poison every gain.

**Still imperfect, not blocking:** vertical drag seems to change a bit with forward speed; a “body lift” term would help held-out vertical fit but isn’t in the shipped model yet. The middle of the thrust table is thinner than the ends. No clean finished lap exists for end-to-end plant validation — hold-out is nine minutes of ordinary flying.

**Do not re-fly “maneuver D”** style sessions that were meant to chase bad referee scores: those failures are a recording glitch (velocity stream stuck repeating), not something more piloting fixes.

---

## What’s built: the surrogate

A vectorized environment can run thousands of parallel races with no pixel rendering. Each episode:

1. Builds a fresh winding course (18–22 gates).
2. Steps the fitted plant at high internal rate; the policy decides at a randomized ~45–65 Hz.
3. Projects gates through a camera model (~30 fps) and corrupts them with a noise model.
4. Runs a stub “attention” module that picks what to look at and drives yaw.
5. Scores the policy with dense progress toward the true gate approach point, plus bonuses/penalties for crossings, collisions, corridor exits, etc.

**Action limits used in training** are tighter than the sim’s absolute max: ±2.75 rad/s and thrust no lower than 0.10. That’s the envelope the plant was actually identified on (and a conservative floor even though low throttle is now measured).

**Known structural gaps:**

- **Noise is not measured.** Visibility ranges in particular look too pessimistic relative to probes on real frames (gates still look orange much farther out than the fallback assumes). Training against vanishing gates teaches search behavior the real course may not need.
- **Attention is a stub**, not Claire’s real module. Mismatch here transfers badly.
- **Courses mean-revert in altitude**, so they struggle to reproduce the sustained one-way descent seen on the real VQ1 map — exactly the case that pushes gates under the camera’s lower edge.
- Every training episode throws the course away: good for generalization, bad for comparing a surrogate lap time to a live one. A fixed measured 6-gate map would close that loop.

---

## The reactive baseline (submission floor)

If learning never works, this is what ships. It’s intentionally boring: proportional guidance toward the gate approach point, with a few hard-won fixes:

- Steer relative to **where you’re going** (drag-based velocity bearing), not where the nose points — under auto-yaw the nose is already glued to the target.
- Climb by nulling the angle between **velocity** and the target (not just elevation error), using a leaky estimate of climb rate from the accelerometer.
- Slowly adapt hover trim so plant randomization doesn’t leave a permanent aim-low.
- Pitch guard keeps the gate inside the camera’s awkward lower frame edge (−9.4°).

Unit checks for this policy live in `policies/selfcheck.py` and have been updated to match the current vertical law. Separate probe scripts under `evalsuite/` exist to classify crashes (floor / ceiling / gate frame) and to sweep gains.

**Honest status:** the baseline is the floor and it still needs to finish easy surrogate courses reliably. Probe comments from earlier today describe gate-plane misses and vertical wander that motivated the climb-rate rewrite; treat live numbers from a fresh `run_policy.py` / `_probe_sweep.py` run as ground truth, not those comments.

---

## Learning (PPO) — what’s happening

The training harness is complete: stacked observations, running normalization, curriculum on difficulty and speed, checkpoints with frozen stats for deployment, curriculum stall logic that weakens near-gate progress reward (never the collision penalty).

A real run named **`run1`** is on disk:

- About **5.05 million** environment steps (256 envs, frame stack 6).
- **Completion rate stayed 0% for the entire log.**
- Difficulty never left “easiest”; speed cap stayed at half authority.
- Mean gates passed per episode hovered around **two**.
- Collision rate dipped mid-run then climbed back toward ~90%+.
- Episode return went deeply negative, then recovered into small positives — some learning of “don’t die instantly,” not of finishing.

Curriculum did notice a stall and eased near-gate progress shaping once (`progress_gate_scale` 1.0 → 0.85 in the checkpoint sidecar). That alone did not unlock completions.

There is also an earlier **smoke** run (~400k steps) with the same qualitative failure.

**Bottom line:** the pipeline trains; it has not produced a flyable learned policy yet. Architecture docs mention a discount-factor concern but don’t actually spell out the analysis — treat that as an open hypothesis, not a settled diagnosis.

---

## Evaluation and deployment

- **`evalsuite/run_policy.py`** flies baseline or a checkpoint over seeded surrogate worlds and reports completion / collisions / time.
- **`select.py`** ranks candidates under harder shared draws.
- **`stress.py`** repeats eval at ~30 Hz decision rate (loaded-machine regime).
- **`policies/rl.py`** wraps a checkpoint for the live stack (same normalize → stack → network → action map as training).
- **`RecoverySupervisor`** levels and hovers after a graze, then hands control back — scripted on purpose so RL doesn’t have to learn tumble recovery.

None of that replaces the missing ingredient: a policy (reactive or learned) that finishes courses on the surrogate first.

---

## Sign conventions (why people lose weeks)

The simulator mirrors body rates and some attitude streams relative to textbook NED. Commands and gyros agree with each other and still be wrong together. The camera and gravity (when parked) are the referees that actually settle signs. The plant fit is the one place a silent mirror produces a model that “works” in replay statistics and then fails when a policy trained on it meets the real sim. Prefer “suspect signs, then noise” when surrogate and live disagree.

---

## What to do next (practical order)

1. **Get the baseline through gates** on easy surrogate seeds. Sweep gains if needed. This is the floor for the deadline.
2. **Explain or fix `run1`’s zero completion** before burning another 20M steps blind — reward timescale, noise pessimism, course geometry, collision margins, and the unused terminal observation in PPO are all candidates.
3. **Measure detection visibility** from frames already on disk (orange mask + projected gates). Replace the guessed max-range band.
4. **Improve the course generator** (sustained descents) and optionally lock a fixed eval course from the measured 6-gate map.
5. Only then: full eval selection, supervisor stress, and scarce live sim runs.

---

## How to poke it yourself

From the repo root:

```powershell
python pilot/control/surrogate/selfcheck.py
python pilot/control/train/selftest.py
python pilot/control/policies/selfcheck.py
python pilot/control/evalsuite/run_policy.py --policy baseline --seeds 0-9 --difficulty 0.0
```

Training logs: `pilot/control/train/checkpoints/run1_log.csv`  
Latest archived checkpoint sidecar: `…/run1_s5046272.json` (weights in the matching `.pt`)

For a machine-oriented contract dump (constants, APIs, gotchas), see `CONTROL_STATE_AGENTS.md` beside this file. For the deep read on the plant fit and the surrogate specifically — every number verified by *running* the checks, not by reading source — see `STATE_PLANT_SURROGATE.md`.
