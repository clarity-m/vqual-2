# RL training pipeline — architecture

Owner: Alex (kongalex@umich.edu). Deadline 2026-08-03 06:00 PST.

**Status, 2026-08-02. → `STATE_RL_TRAINING.md` holds the measured state; read it
first.** P1 done and validated (`README.md`), refitted on card 2. P2 built and
self-checking 17/17. P3 is still the hand-specified fallback, not measured. T4
**now trains a policy that beats the reactive baseline 2.3–2.5× on per-gate
accuracy** — 0.705 against 0.307 at the physical aperture, `runB1` at 207M steps.
But **neither policy completes a course**: completion over 20 gates needs a
per-gate rate of ~0.989 and the best measured is 0.705. E5 is wired and its
batched path is now checked against the reference; D6 is coded and **still
untested end to end**.

The discount question under T4 is settled rather than open: γ=0.99 at 45–65 Hz is a
~1.8 s horizon against a ~2.4 s time-to-floor, discounting the collision terminal to
0.27 of face value. γ is now 0.997. But it was not the main cause — `run1`'s failure
was never the optimiser. The completion metric was unreachable by construction
(per-gate-rate^20 over 18–22 gates), and the reward contained no altitude term at
all while 55.5% of episodes ended on the floor. Measurements in
`STATE_RL_TRAINING.md`.

**Nothing below has been flown live.**

*(An earlier version of this header said the course generator, synthetic
detections, P3 and T4 "do not" exist. That was true when it was written and is
no longer; the code on disk is ahead of it. Where this document and the code
disagree, the code and `plant.json` win — the stale sentences that are known to
remain are flagged in place.)*

The deliverable is an `interface.Policy` that beats the baseline. This document
describes the *learned* route to one. The reactive baseline (PID off the gate
approach point) shares every component here except stage T4 and is built first —
it is the floor, RL is the upside.

Read first: `../interface.py` (frozen contract), `README.md` (sign traps, dead
ends), `../NOTES.md` (measured facts), `../SYSID.md` (data collection),
`../CONVENTIONS.md` (every sign, frame and axis fact — the single source).

## One-paragraph summary

The policy is trained entirely offline, inside a fitted NumPy surrogate — never
against the live sim, which is a serialized evaluation resource (~15–25
runs/session). The surrogate steps the fitted plant over procedurally generated
courses and emits the exact 73-D observation vector the live pilot sees,
corrupted by a detection-noise model measured from real frames. PPO trains a
small recurrent policy on dense progress reward, over roll/pitch/thrust only
(yaw stays with the `AUTO_ATTENTION` servo). Checkpoints are selected on
held-out randomized courses under worst-case noise, then evaluated live in
Training mode.

## Pipeline stages

    P0 data           P1 plant fit        P2 surrogate         P3 noise model
    recordings   -->  fit + VALIDATE  --> env over random  --> measured detection
    (CSVs in git;     (replay gate,       courses              stats (real frames)
     frames: laptop)   DONE 07-31)            |                     |
                                              v                     v
                                        T4 PPO training  <--  randomization
                                              |
                                              v
                                        E5 evaluation --> D6 deployment
                                        (surrogate held-out, then live sim)

Stages P0–P2 are the critical path and are exactly the work the reactive
baseline needs too. T4 is the only RL-specific stage. P1 has passed its
validation gate, so the open critical path is P2's course generator and
synthetic detections, then P3.

### P0 — Data

* Recordings: `pilot/sessions/<timestamp>/` — the telemetry CSVs are now
  tracked in git, so the plant fit no longer depends on any one machine. The
  frame JPEGs are still unbacked-up on one laptop; they gate P3 only.
* Ground truth comes from the VQ1 build (identical physics, pose not blocked).
  Existing sessions: `20260731-143025` (doublets), `20260731-144815` (taps,
  frames), `20260731-150712` (thrust/cruise/skids). See `../SYSID.md`.
* Still missing: one clean completed lap for held-out validation — no session has
  ever finished one (`race_finish_time_ns` is −1 in all eight, and `active_gate_index`
  never got past 1). It is maneuver C of card 2 in `../SYSID.md`, along with the two
  measurements that would retire the low-throttle thrust clamp and the jointly-fitted
  `kz`. Three minutes of flying, and the only remaining lever on plant accuracy.

### P1 — Plant fit — DONE 2026-07-31

Fitted, validated, recorded in `plant.json` / `plant.py`; the pipeline, the
numbers and the dead ends are in `README.md`. Held-out R² 0.92–0.999,
open-loop replay drifts 2.6 m over a 5 s / 35 m window, and every sign
corruption of the model is caught by 17x or more. Two stale worries from the
design draft, retired: the yaw truth sign is settled (`sysid_frames.py`,
sixteen candidate conventions scored against the kinematic identity, 6x
margin), and the lags are bracketed interior (+15 ms thrust cmd → motor,
+10 ms rate cmd → gyro) — the old −15-sample edge peak was a symptom of the
missing vertical-drag term, not of the correlation window.

The validation gate stands for any REFIT: replay recorded commands open-loop
through the fitted model, overlay predicted vs recorded pose AND IMU
(`sysid_replay.py`), validate on the designated hold-out (`20260731-131305`,
nine minutes of ordinary flying — a completed lap would be better and does not
exist yet, see P0). NOTHING trains on the surrogate until the replay passes.

Still open on the plant — exactly what card 2 in `../SYSID.md` flies:

* the thrust curve is unmeasured below throttle 0.10 and hand-clamped at zero,
  the one unmeasured hack in the model (see T4 for the training-side guard)
* `kz` is fitted jointly with thrust — the collinearity that broke the first
  attempt is modelled, not broken; the apex arcs read it independently

### P2 — Surrogate environment

A vectorized NumPy (or JAX) gym-style environment, thousands of instances in
parallel. Renders no pixels; emits `Observation.to_vector()` (73-D) exactly.

* **Dynamics:** fitted plant at 120 Hz internal step; policy decisions at a
  per-episode randomized 45–65 Hz (matches the measured load-dependent rate).
  The recordings also show a ~32 Hz IMU tier on loaded sessions; checkpoints
  get stress-tested at ~30 Hz even though training stays at 45–65.
* **Courses: procedurally generated, fresh every episode.** The real map is
  unobservable under VQ2, so the policy must learn gate-seeking, not a track.
  Sample ~18–22 gates; winding turns (VQ2 winds far more than VQ1); segment
  lengths consistent with the observed ~2.76 s/station cruise; vertical
  profiles including ~20° descents — the case that pushes gates below the
  camera's −9.4° lower frame edge, which is the binding constraint.
* **Synthetic detections tick at the camera's measured ~30 fps, not per
  policy step** — at 45–65 Hz decisions, roughly every other step sees an
  unchanged detection with `staleness_s` grown by ~33 ms, and that rhythm is
  part of what the policy must learn. Project true gates through the camera
  model (20° up-tilt, +49.4°/−9.4° vertical span, fx=fy=320), then corrupt:
  visibility only in-frustum and in-range; dropout; latency; staleness
  accumulation while tracks coast; position/normal noise growing with range
  and obliquity; `normal_valid` failing near head-on (the PnP tilt
  degeneracy); `pose_valid=False` fallback with long-biased range on oblique
  gates; occasional false positives. Parameter values come from P3, drawn
  per-episode from their uncertainty — never fixed at point estimates.
* **Own-state channel:** IMU noise from parked recordings; gravity roll/pitch
  with confidence degrading under acceleration; drag-bearing `vel_bearing`
  invalid near hover; `speed_est` uncertainty tied to the drag-fit residual.
* **Attention + yaw servo: run the REAL attention code, not a
  re-implementation.** The observation carries `Attention.kind` and
  `target_dir_body`, produced by perception's attention policy — what to look
  at, when to hand off, what SEARCH does. A behavioural mismatch there is a
  transfer risk on par with the noise model, so the surrogate imports Claire's
  actual attention module and runs it over the synthetic detections, and its
  version is frozen with every checkpoint. The azimuth-only servo the live
  stack runs under `YawMode.AUTO_ATTENTION` is simulated with it, including
  the `R_COMMIT` handoff (value TBD — currently an open measurement).
* **Initial states cover the recovery handback.** D6's supervisor hands
  control back at hover: low speed, level, possibly no gate in frustum,
  arbitrary offset from the corridor. If every episode starts in a sane
  course-following state, the handback lands out of distribution exactly when
  it matters. Episodes therefore randomize their starts: mid-course spawns,
  hover starts, corridor offsets, no-gate-visible starts.
* **World frame exists only in here.** It computes reward and collision tests
  and is never encoded into the observation.

### P3 — Noise model (the transfer risk)

Dropout, latency, false-positive and range-error statistics measured by
running the real perception pipeline (Claire's) over recorded frames and
scoring against VQ1 ground truth. Randomized over their measurement
uncertainty at training time. This is the component the README warns about:
a policy exploits any regularity in synthetic detections, so clean detections
are a bug, not a simplification. Scoring is blocked on perception existing
well enough to score — coordinate with Claire early.

Training does NOT wait on P3. Fallback: a hand-specified, deliberately
pessimistic noise model with wide randomization — worse than anything
measured, far better than clean — swapped for measured parameters the moment
they exist. A policy trained on the fallback is riskier and is flagged as
such at model selection.

**Half of P3 is not blocked on Claire and never was (2026-08-01).** Scoring a
*detector* needs a detector, but the visibility half needs only geometry: the
gate centres of the 6-gate map are measured to 0.5–1.2 m from `gate_advance`
plus truth pose, so every recorded frame has a known true range, obliquity and
projected position. Running the `NOTES.md` orange HSV mask over
`20260731-204841-vq1-lap-slow` finds a blob at the predicted place on 95% of
in-frustum (gate, frame) pairs — including 95% at 30–45 m, against a fallback
`max_range_m` of 14–30. That measures detectability against range, the burst
structure of the misses, and the size-to-range calibration, and it needs no
perception pipeline. What still waits on Claire is the error model of a real
fit: PnP failure rate, normal validity, and the hangar false positives, none of
which an HSV mask on VQ1 can speak to.

### T4 — Training

* **Actions (3-DoF):** roll rate, pitch rate, thrust. Yaw stays with
  `AUTO_ATTENTION` — it has almost no reward signal (free gimbal), so
  learning it is wasted samples. Outputs squashed to the envelope the plant
  fit actually saw, ~±2.5–3 rad/s — at `MAX_RATE_RPS = 6.0` the rate-loop
  model is extrapolation. Thrust head initialized to bias near hover
  (`interface.HOVER_THRUST` = 0.27, measured; 0.25 was the early low
  estimate — reference the constant, this number has drifted once already).
  Thrust output floored at ~0.10 until card 2's apex data replaces the hand
  clamp: below that the model has no data at all, randomization cannot cover
  a region with nothing measured in it, and a policy living there is
  exploiting an artefact.
* **Network:** small MLP (~2×256) with short memory — stack the last 4–8
  observations or a small GRU. Detections drop out and coast; a memoryless
  policy flies blind through every dropout.
* **Algorithm:** PPO, standard settings. Steps are 73 floats; millions of
  steps are cheap. Running observation normalization (statistics frozen into
  the deployed checkpoint) and reward scaling are mandatory, not tuning: the
  73-D vector mixes metres in the tens, m/s² up to ~50, radians and one-hot
  flags, and PPO is fragile to that.
* **Reward** (computed from privileged surrogate state — legitimate, reward
  exists only at training time):
  * dense per-step progress toward the TRUE gate's approach point, computed
    from surrogate world state — never from the observation's `pos_body` /
    `normal_valid`, which are noisy and droppable; a reward that follows the
    detections silently changes target on every dropout. Approach distance
    `d` is TBD. Estimating the target from noisy observations is the
    policy's problem, not the reward's. The dominant term; ~20
    crossings/episode is far too sparse alone
  * crossing bonus on `active_gate_index` advancing
  * terminal penalty: collision, or leaving a generous course corridor
  * small regularizers: action-rate (jerk) penalty; mild penalty when no gate
    is in frustum (substitutes for learning yaw/visibility management)
  * time penalty only after completion is reliable — the leaderboard counts
    completed runs first, fast runs second
* **Collisions — geometry, not physics.** The surrogate never simulates a
  bounce; it hit-tests and terminates:
  * gate plane crossed outside the 1500 mm inner aperture (spec-exact; the
    dominant hazard, negotiated ~20×/run) — tested as a plane although the
    frame is 260 mm deep; the randomized sphere margin below is what absorbs
    that simplification
  * floor / randomized ceiling bound
  * optional randomized clutter just outside the corridor
  * drone = bounding sphere inflated by a per-episode randomized margin
    (10–40 cm) — the main transfer trick; the policy learns clearance that
    absorbs plant error and detection noise
  * KNOWN GAP: the 73-D observation has no obstacle channel; columns and
    walls are invisible until `t_since_collision_s` resets. Defense is
    corridor discipline (the gate-to-gate corridor is flyable by
    construction) + the ribbon as a pre-turn prior. If real flights show
    in-corridor collisions, that is an `interface.py` change proposal, not a
    control fix.
* **Curriculum:** (1) difficulty — gentle wide courses and low noise first,
  anneal to tight turns and full measured noise; (2) speed — cap commanded
  aggressiveness early so exploration survives the collision terminations,
  relax as completion rate rises. If completion stalls, weaken the progress
  term near gates; do not weaken the collision penalty.
* **Domain randomization, per episode:** plant parameters at ±10–20% (thrust
  gain, drag, loop time constants, latency) — deliberately much wider than
  the fit's measured residuals now that those exist; a transfer choice, not
  a leftover placeholder. Also randomized: control period, detection-noise
  parameters, collision margin. The real sim should be indistinguishable
  from one more draw.

### E5 — Evaluation

* **Model selection:** held-out course seeds. Rank by completion rate under
  worst-case noise draws (a high percentile, not the literal max), tie-broken
  by MEAN-case time — selecting on worst-case time as well just breeds
  over-conservative policies.
* **Live sim = evaluation only.** Training mode first. Budget runs; they are
  shared and serialized. The signature failure "great on surrogate, bad in
  sim" means: suspect surrogate signs FIRST, noise model second — in that
  order (see `README.md`).
* Open question to spend one live run on: does a collision episode invalidate
  a run, or just cost time? Brush a gate deliberately in Training mode; watch
  whether `active_gate_index` keeps advancing and the run registers. This
  sets the margin/recovery trade-off.

### D6 — Deployment

* Frozen network wrapped in `interface.Policy`. Consumes the permitted 73-D
  vector only; audit-clean by construction (§9.2/§9.3).
* **Scripted recovery supervisor** around the policy: on a collision episode
  (`t_since_collision_s` reset), override to level (gravity roll/pitch),
  hover thrust, let attention re-acquire, hand back. ~50 deterministic lines.
  8-minute budget vs a ~1–2 minute lap makes recovery nearly free; do not ask
  RL to learn tumble recovery. The handback state is covered in training by
  P2's randomized episode starts, and the supervisor gets its own live
  Training-mode test (deliberate tumble, verify recovery) piggybacked on E5's
  collision-semantics run.
* `Action.yaw_mode = AUTO_ATTENTION`; signs applied in the link layer only.

## Failure modes this design guards against

| failure | guard |
|---|---|
| mirrored surrogate (silent sign error) | P1 replay validation gate before any training |
| policy exploits clean synthetic detections | P3 measured noise, randomized over uncertainty |
| overfit to a guessed course map | fresh procedural course every episode |
| overfit to one plant point-estimate | per-episode plant randomization, deliberately wider than fit residuals |
| corner-shaving through gate frames | inflated randomized collision margin + hard termination |
| timid hovering under crash penalties | speed curriculum, not penalty reduction |
| blind flight through detection dropout | observation stacking / GRU memory |
| unseen obstacles off the racing line | corridor termination in training; ribbon prior |
| dead policy after a graze at race time | scripted recovery supervisor outside the network |
| surrogate attention diverges from the real attention policy | run Claire's actual attention code in the surrogate; freeze its version with the checkpoint |
| policy exploits the unmeasured low-throttle clamp | thrust output floored at 0.10 until card 2 data lands |
| recovery handback lands out of distribution | randomized episode starts: hover, mid-course, no-gate-visible |

## Schedule risk, stated plainly

P1 is done and validated; its first failure (thrust R² ≈ 0.05) is diagnosed
and written down in `README.md`, and the remaining plant risk is narrow — the
low-throttle clamp and the jointly-fitted `kz`, both retired by card 2 if it
gets flown. The open critical path is P2's course generator and synthetic
detections, with P3 on the fallback noise model until Claire's pipeline can
be scored. The reactive baseline uses the identical P0–P2 infrastructure and
tunes in minutes; it is built first and is the submission floor.

Go/no-go, pre-committed rather than judged at 3 a.m.: if PPO is not training
against the surrogate by 2026-08-02 06:00 PST (T−24 h), T4 is abandoned and
all remaining time goes to baseline tuning and live evaluation.
