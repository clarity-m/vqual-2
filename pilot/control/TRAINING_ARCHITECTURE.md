# RL training pipeline — architecture

Owner: Alex (kongalex@umich.edu). Status: DESIGN — nothing below exists yet.
Deadline 2026-08-03 06:00 PST.

The deliverable is an `interface.Policy` that beats the baseline. This document
describes the *learned* route to one. The reactive baseline (PID off the gate
approach point) shares every component here except stage T4 and is built first —
it is the floor, RL is the upside.

Read first: `../interface.py` (frozen contract), `README.md` (sign traps, dead
ends), `../NOTES.md` (measured facts), `../SYSID.md` (data collection).

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
    (Claire's         (replay gate)       courses              stats (real frames)
     laptop)                                  |                     |
                                              v                     v
                                        T4 PPO training  <--  randomization
                                              |
                                              v
                                        E5 evaluation --> D6 deployment
                                        (surrogate held-out, then live sim)

Stages P0–P2 are the critical path and are exactly the work the reactive
baseline needs too. T4 is the only RL-specific stage.

### P0 — Data

* Recordings: `pilot/sessions/<timestamp>/` — exist on ONE laptop, not backed
  up, not in git. Copying them is task zero.
* Ground truth comes from the VQ1 build (identical physics, pose not blocked).
  Existing sessions: `20260731-143025` (doublets), `20260731-144815` (taps,
  frames), `20260731-150712` (thrust/cruise/skids). See `../SYSID.md`.
* Still missing: one clean completed lap for held-out validation — no session has
  ever finished one (`race_finish_time_ns` is −1 in all eight, and `active_gate_index`
  never got past 1). It is maneuver C of card 2 in `../SYSID.md`, along with the two
  measurements that would retire the low-throttle thrust clamp and the jointly-fitted
  `kz`. Three minutes of flying, and the only remaining lever on plant accuracy.

### P1 — Plant fit

Model, per the spec's rigid-body claim (§3.2) and what the recordings can
identify:

* rigid body: mass-normalized thrust curve `T(cmd)` with actuator lag,
  gravity, drag `D(v_body)` on **body-frame** velocity
* inner loop: first-order rate response per axis, `rate_meas -> cmd`, latency
  bracketed properly (the −15-sample edge peak in `plantfit.py` means the
  window was too narrow, widen it)
* `ACTUATOR_OUTPUT_STATUS` (~95 Hz, permitted, currently unused) pins the
  inner-loop timing

Non-negotiable order of operations (see `README.md` for why each exists):

1. Correct truth signs FIRST: `truth_roll = ATTITUDE.roll`,
   `truth_pitch = ODOMETRY.pitch`, `truth_yaw = -ATTITUDE.yaw` (yaw pending
   confirmation away from the 180° degenerate heading).
2. Weight by actual timestamp deltas — IMU rate is load-dependent
   (47.7–62.9 Hz measured), there is no nominal period.
3. Self-check the attitude rotation: `R·(specific force) + g ≈ dv/dt`.
4. **Validation gate:** replay recorded command sequences open-loop through
   the fitted model; overlay predicted vs recorded pose AND IMU. A wrong sign
   diverges immediately. Fit on maneuvers, validate on a lap the fit never
   saw. NOTHING trains on the surrogate until this passes.

### P2 — Surrogate environment

A vectorized NumPy (or JAX) gym-style environment, thousands of instances in
parallel. Renders no pixels; emits `Observation.to_vector()` (73-D) exactly.

* **Dynamics:** fitted plant at 120 Hz internal step; policy decisions at a
  per-episode randomized 45–65 Hz (matches the measured load-dependent rate).
* **Courses: procedurally generated, fresh every episode.** The real map is
  unobservable under VQ2, so the policy must learn gate-seeking, not a track.
  Sample ~18–22 gates; winding turns (VQ2 winds far more than VQ1); segment
  lengths consistent with the observed ~2.76 s/station cruise; vertical
  profiles including ~20° descents — the case that pushes gates below the
  camera's −9.4° lower frame edge, which is the binding constraint.
* **Synthetic detections:** project true gates through the camera model
  (20° up-tilt, +49.4°/−9.4° vertical span, fx=fy=320), then corrupt:
  visibility only in-frustum and in-range; dropout; latency; staleness
  accumulation while tracks coast; position/normal noise growing with range
  and obliquity; `normal_valid` failing near head-on (the PnP tilt
  degeneracy); `pose_valid=False` fallback with long-biased range on oblique
  gates; occasional false positives. Parameter values come from P3, drawn
  per-episode from their uncertainty — never fixed at point estimates.
* **Own-state channel:** IMU noise from parked recordings; gravity roll/pitch
  with confidence degrading under acceleration; drag-bearing `vel_bearing`
  invalid near hover; `speed_est` uncertainty tied to the drag-fit residual.
* **Attention + yaw servo:** simulate the same azimuth-only servo the live
  stack runs under `YawMode.AUTO_ATTENTION`, including the `R_COMMIT` handoff
  (value TBD — currently an open measurement).
* **World frame exists only in here.** It computes reward and collision tests
  and is never encoded into the observation.

### P3 — Noise model (the transfer risk)

Dropout, latency, false-positive and range-error statistics measured by
running the real perception pipeline (Claire's) over recorded frames and
scoring against VQ1 ground truth. Randomized over their measurement
uncertainty at training time. This is the component the README warns about:
a policy exploits any regularity in synthetic detections, so clean detections
are a bug, not a simplification. Blocked on perception existing well enough
to score — coordinate with Claire early.

### T4 — Training

* **Actions (3-DoF):** roll rate, pitch rate, thrust. Yaw stays with
  `AUTO_ATTENTION` — it has almost no reward signal (free gimbal), so
  learning it is wasted samples. Outputs squashed well inside
  `MAX_RATE_RPS = 6.0`; thrust head initialized to bias near hover (0.25).
* **Network:** small MLP (~2×256) with short memory — stack the last 4–8
  observations or a small GRU. Detections drop out and coast; a memoryless
  policy flies blind through every dropout.
* **Algorithm:** PPO, standard settings. Steps are 73 floats; millions of
  steps are cheap.
* **Reward** (computed from privileged surrogate state — legitimate, reward
  exists only at training time):
  * dense per-step progress toward the current gate's approach point
    (`pos_body + d·normal_body` when `normal_valid`, gate centre otherwise)
    — the dominant term; ~20 crossings/episode is far too sparse alone
  * crossing bonus on `active_gate_index` advancing
  * terminal penalty: collision, or leaving a generous course corridor
  * small regularizers: action-rate (jerk) penalty; mild penalty when no gate
    is in frustum (substitutes for learning yaw/visibility management)
  * time penalty only after completion is reliable — the leaderboard counts
    completed runs first, fast runs second
* **Collisions — geometry, not physics.** The surrogate never simulates a
  bounce; it hit-tests and terminates:
  * gate plane crossed outside the 1500 mm inner aperture (spec-exact; the
    dominant hazard, negotiated ~20×/run)
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
* **Domain randomization, per episode:** plant parameters over the fit's
  residual uncertainty (±10–20% on thrust gain, drag, loop time constants,
  latency), control period, detection-noise parameters, collision margin.
  The real sim should be indistinguishable from one more draw.

### E5 — Evaluation

* **Model selection:** held-out course seeds, worst-case (not mean) noise
  draws. Rank by completion rate, then time.
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
  RL to learn tumble recovery.
* `Action.yaw_mode = AUTO_ATTENTION`; signs applied in the link layer only.

## Failure modes this design guards against

| failure | guard |
|---|---|
| mirrored surrogate (silent sign error) | P1 replay validation gate before any training |
| policy exploits clean synthetic detections | P3 measured noise, randomized over uncertainty |
| overfit to a guessed course map | fresh procedural course every episode |
| overfit to one plant point-estimate | per-episode plant randomization over fit residuals |
| corner-shaving through gate frames | inflated randomized collision margin + hard termination |
| timid hovering under crash penalties | speed curriculum, not penalty reduction |
| blind flight through detection dropout | observation stacking / GRU memory |
| unseen obstacles off the racing line | corridor termination in training; ribbon prior |
| dead policy after a graze at race time | scripted recovery supervisor outside the network |

## Schedule risk, stated plainly

P1 already failed once (thrust R² ≈ 0.05, lag unbracketed, drag on the wrong
variable) and is not a solved subproblem. P0→P2 is serialized and gates
everything. The reactive baseline uses the identical P0–P2 infrastructure and
tunes in minutes; it is built first and is the submission floor. T4 proceeds
only if the P1 validation gate passes with enough runway left to train and
evaluate.
