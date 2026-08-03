# RL training pipeline — architecture

Owner: Alex (kongalex@umich.edu). Deadline 2026-08-03 06:00 PST.

**Status, 2026-08-01.** P1 done and validated (`README.md`), refitted on card 2.
P2 built and self-checking 17/17 — course generator, synthetic detections and
the vector env all exist. P3 is still the hand-specified fallback, not measured
(planned vs current: see summary + P3). T4's harness is built, self-tests
pass (including truncation bootstrap and `--resume`), and a real run (`run1`,
~5M steps) has **not** produced a flyable policy — zero completions throughout.
A discount / reward-horizon concern is an open hypothesis under T4, not a
finished analysis. E5 is wired; D6 is coded, untested end to end.

This document is the *design intent* for the learned route. Where intent and
disk disagree, the code and `plant.json` win; state dumps live in
`CONTROL_STATE_*.md` / `STATE_PLANT_SURROGATE.md`. Planned vs currently
implemented is called out in place when they diverge.

The deliverable is an `interface.Policy` that beats the baseline. The reactive
baseline (PID off the gate approach point) shares every component here except
stage T4 and is built first — it is the floor, RL is the upside.

Read first: `../interface.py` (frozen contract), `README.md` (sign traps, dead
ends), `../NOTES.md` (measured facts), `../SYSID.md` (data collection),
`../CONVENTIONS.md` (every sign, frame and axis fact — the single source).

## One-paragraph summary

The policy is trained entirely offline, inside a fitted NumPy surrogate — never
against the live sim, which is a serialized evaluation resource (~15–25
runs/session). The surrogate steps the fitted plant over procedurally generated
courses and emits the exact 73-D observation vector the live pilot sees,
corrupted by a detection-noise model.

**Noise — planned vs current.** *Planned:* corruptions drawn from statistics
measured on real frames (Claire's perception scored against VQ1 truth),
randomized over measurement uncertainty. *Current:* `surrogate/noise.py` is
still the hand-specified, deliberately pessimistic fallback — not measured.
Visibility from orange-mask probes already suggests the fallback
`max_range_m=(14,30)` is too short; the PnP / false-positive half still waits
on Claire. Training proceeds on the fallback and any such checkpoint is
flagged riskier at model selection.

PPO trains a small MLP with frame-stack memory on dense progress reward, over
roll/pitch/thrust only (yaw stays with the `AUTO_ATTENTION` servo). Checkpoints
are selected on held-out randomized courses under worst-case noise, then
evaluated live in Training mode.

## Pipeline stages

    P0 data           P1 plant fit        P2 surrogate         P3 noise model
    recordings   -->  fit + VALIDATE  --> env over random  --> planned: measured
    (CSVs in git;     (replay gate,       courses              from real frames
     frames: laptop)   DONE + card 2)         |                current: fallback
                                              v                     v
                                        T4 PPO training  <--  randomization
                                              |
                                              v
                                        E5 evaluation --> D6 deployment
                                        (surrogate held-out, then live sim)

Stages P0–P2 are exactly the work the reactive baseline needs too. T4 is the
only RL-specific stage. P1 and P2 are built; the open critical path is now
**getting any controller (baseline or RL) to finish surrogate courses**, then
measured P3 / attention swap / live E5–D6.

### P0 — Data

* Recordings: `pilot/sessions/<timestamp>/` — the telemetry CSVs are now
  tracked in git, so the plant fit no longer depends on any one machine. The
  frame JPEGs are still unbacked-up on one laptop; they gate measured P3.
* Ground truth comes from the VQ1 build (identical physics, pose not blocked).
  Fit / hold-out session lists live in `plant.json` meta; see `../SYSID.md`
  and `STATE_PLANT_SURROGATE.md`.
* Still missing: one clean completed lap for end-to-end plant validation — no
  session has ever finished one (`race_finish_time_ns` is −1 everywhere).
  Furthest recorded is `active_gate_index` **5**, not a full course. Card 2's
  apex / terminal / low-throttle flights that retire the old thrust clamp and
  joint `kz` **have been flown** and are in the current `plant.json`; the
  remaining data gap is the finished lap, not those two measurements.

### P1 — Plant fit — DONE (card-2 refit on disk)

Fitted, validated, recorded in `plant.json` / `plant.py`; the pipeline, the
numbers and the dead ends are in `README.md` / `STATE_PLANT_SURROGATE.md`.
Held-out R² ≈ 0.99 / 0.99 / 0.90, open-loop replay is the mandatory gate, and
every sign corruption of the model is caught by a large margin. Yaw truth
sign and rate/thrust lags are settled.

The validation gate stands for any REFIT: replay recorded commands open-loop
through the fitted model, overlay predicted vs recorded pose AND IMU
(`sysid_replay.py`), validate on the designated hold-out (`20260731-131305`).
NOTHING trains on the surrogate until the replay passes.

**Retired by card 2** (do not re-open as if unflown):

* thrust is a **measured** 21-knot table (`np.interp`, no hand clamp); idle ≈
  0.35 m/s², hover throttle 0.270, full ≈ 51.67 m/s²
* `kz` from apex arcs (≈ 0.04359), not a joint racing-flight fit with thrust

**Still open on the plant:** forward-speed dependence of `kz` / optional
body-lift `c·u²` (helps hold-out z, not in `plant.json`); thin mid-table
samples 0.50–0.82; no completed-lap validation.

### P2 — Surrogate environment — BUILT

A vectorized NumPy gym-style environment (`surrogate/`), thousands of
instances in parallel. Renders no pixels; emits `Observation.to_vector()`
(73-D) exactly. Self-check: `python pilot/control/surrogate/selfcheck.py`
→ 17/17.

* **Dynamics:** fitted plant at high internal rate; policy decisions at a
  per-episode randomized 45–65 Hz (matches the measured load-dependent rate).
  The recordings also show a ~32 Hz IMU tier on loaded sessions; checkpoints
  get stress-tested at ~30 Hz (`evalsuite/stress.py`) even though training
  stays at 45–65.
* **Courses: procedurally generated, fresh every episode.** The real map is
  unobservable under VQ2, so the policy must learn gate-seeking, not a track.
  Sample ~18–22 gates; winding turns; segment lengths and turn/elev ranges
  anneal with difficulty.
  *Planned:* vertical profiles including sustained ~20° descents — the case
  that pushes gates below the camera's −9.4° lower frame edge.
  *Current:* `EnvConfig.alt_revert = 0.9` mean-reverts altitude inside a
  ceiling band, so the generator cannot reproduce the measured monotone
  ~24 m descent over ~140 m of path on the VQ1 6-gate map. Fix the prior;
  optionally also lock that map as a fixed eval course.
* **Synthetic detections tick at the camera's measured ~30 fps, not per
  policy step** — at 45–65 Hz decisions, roughly every other step sees an
  unchanged detection with `staleness_s` grown by ~33 ms. Project true gates
  through the camera model (20° up-tilt, +49.4°/−9.4° vertical span,
  fx=fy=320), then corrupt with the P3 parameter set (see below).
* **Own-state channel:** IMU noise from parked recordings; gravity roll/pitch
  with confidence degrading under acceleration; drag-bearing `vel_bearing`
  invalid near hover; `speed_est` uncertainty tied to the drag-fit residual.
* **Attention + yaw servo — planned vs current.**
  *Planned:* import Claire's real attention module over synthetic detections,
  freeze its version with every checkpoint; behavioural mismatch is a
  transfer risk on par with the noise model.
  *Current:* parameterized stub in `surrogate/attention.py` (`R_COMMIT_M =
  6.0` placeholder, `EnvConfig.attention_factory = None`). Azimuth servo
  under `AUTO_ATTENTION` runs against that stub. Swap when the real module
  exists.
* **Initial states cover the recovery handback.** Mid-course, hover,
  corridor-offset, and no-gate-visible starts (default probs
  0.50 / 0.25 / 0.12 / 0.08 / 0.05 with normal).
* **World frame exists only in here.** It computes reward and collision tests
  and is never encoded into the observation. `info["terminal_obs"]` preserves
  the pre-reset ending observation for PPO truncation bootstrap.

### P3 — Noise model (the transfer risk)

**Planned method.** Dropout, latency, false-positive and range-error
statistics measured by running the real perception pipeline (Claire's) over
recorded frames and scoring against VQ1 ground truth. Randomized over their
measurement uncertainty at training time. Clean detections are a bug: a
policy will exploit any regularity in synthetic tracks.

**Current method.** Training does *not* wait on that measurement.
`surrogate/noise.py` is a hand-specified, deliberately pessimistic fallback
(`max_range_m=(14,30)`, `p_detect=(0.62,0.92)`, bursty dropout, etc.), drawn
per episode and scaled toward the worse end with difficulty. A policy trained
on the fallback is riskier and must be flagged as such at model selection.
Swap for measured parameters when they exist — same `NoiseParams` surface.

**Visibility half is not blocked on Claire (2026-08-01).** Scoring a
*detector* needs a detector, but detectability vs range needs only geometry:
gate centres from `gate_advance` + truth pose, scored with the `NOTES.md`
orange HSV mask. On `20260731-204841-vq1-lap-slow`, a blob sits where geometry
predicts on ~95% of in-frustum pairs — including ~95% at 30–45 m, against the
fallback's 14–30 m band. What still waits on Claire is the error model of a
real fit: PnP failure rate, normal validity, hangar false positives.

### T4 — Training — HARNESS BUILT; no flyable policy yet

* **Actions (3-DoF):** roll rate, pitch rate, thrust. Yaw stays with
  `AUTO_ATTENTION`. Squash home is `surrogate/actions.policy_to_action`:
  `RATE_CAP_RPS = 2.75` (not `MAX_RATE_RPS = 6.0`), thrust in
  `[THRUST_FLOOR, 1.0]` with `THRUST_FLOOR = 0.10`. Thrust head bias-init at
  `interface.HOVER_THRUST` (0.27); `u=0` maps to thrust **0.55**, not hover.
  *Floor — planned vs current:* the original reason for 0.10 was an
  unmeasured / hand-clamped low-throttle region. Card 2 retired that; the
  floor remains as a deliberate training guard (lower only in one commit +
  retrain).
* **Network:** MLP 2×256, frame stack k=6 (default). Stacking, not GRU, is
  what shipped — no hidden state across deploy reset / recovery handback.
* **Algorithm:** PPO (`gamma=0.99`, `gae_lambda=0.95`, …). Obs norm + reward
  scaling mandatory. Time-limit truncations bootstrap from `V(s_T)` via
  `info["terminal_obs"]`; genuine terminals (collision / corridor / finish)
  bootstrap at 0. Opt-in `--resume` restores a `<stem>_resume.pt` sidecar
  when present.
* **Discount / reward-horizon — planned note vs current.**
  *Planned (the header used to point here):* a written analysis of whether
  `gamma=0.99` and the dense-progress timescale are wrong for ~20-gate,
  collision-dominated episodes — i.e. whether value targets are too
  short-sighted (or shaping too local) to ever credit finishing.
  *Current:* that write-up was never added; `gamma=0.99` is simply the
  CLI/PPO default. Empirically `run1` (~5.05M steps) and the baseline both
  sit at **0% completion** on easy surrogate seeds. Treat discount /
  horizon as an **open hypothesis** alongside gate-pass tolerance, course
  geometry, and the fallback noise model — not as a settled diagnosis.
  Truncation bootstrap is a separate, already-landed correctness fix and
  does not substitute for that analysis.
* **Reward** (privileged world state only):
  * dense progress toward the TRUE gate approach point (`approach_d_m=2.0`
    in `EnvConfig`; architecture once marked `d` TBD — the code picked 2.0)
  * crossing bonus; collision / corridor terminals; finish bonus
  * jerk + no-gate regularizers; time penalty only after completion is
    reliable (~80% rolling); stall path may lower `progress_gate_scale`
    (never `k_collision`)
* **Collisions — geometry, not physics.** Gate plane vs 1500 mm inner
  aperture; floor / ceiling; sphere margin 10–40 cm per episode. Pass
  requires `lat + sphere_r ≤ 0.75 m` — tight against observed miss
  distributions. KNOWN GAP: no obstacle channel in the 73-D obs.
* **Curriculum:** difficulty + `speed_cap` from rolling completion
  (promote ≥0.70, demote ≤0.25); stall weakens near-gate progress scale
  toward 0.4. Training starts at difficulty 0.0 / speed_cap 0.5.
* **Domain randomization, per episode:** plant ±~10–15% (drag/thrust),
  rate_gain ±10%, delay ×(0.7–1.4), control period, noise draws, collision
  margin.

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
| policy exploits clean synthetic detections | *Planned:* P3 measured noise. *Current:* pessimistic fallback in `noise.py`, flagged at selection |
| overfit to a guessed course map | fresh procedural course every episode |
| overfit to one plant point-estimate | per-episode plant randomization, wider than fit residuals |
| corner-shaving through gate frames | inflated randomized collision margin + hard termination |
| timid hovering under crash penalties | speed curriculum, not penalty reduction |
| blind flight through detection dropout | observation stacking (k=6 shipped; GRU was the alternative) |
| unseen obstacles off the racing line | corridor termination in training; ribbon prior |
| dead policy after a graze at race time | scripted recovery supervisor outside the network |
| surrogate attention diverges from real attention | *Planned:* Claire's module, frozen with ckpt. *Current:* stub — transfer risk until swap |
| policy exploits unmeasured low-throttle region | floor at 0.10; card 2 measured the region, floor kept as training guard |
| recovery handback lands out of distribution | randomized episode starts: hover, mid-course, no-gate-visible |
| time-limit truncations bias value targets | bootstrap `V(s_T)` from `info["terminal_obs"]` (shipped) |

## Schedule risk, stated plainly

P1 is done (card-2 refit on disk); the old plant risks (low-throttle clamp,
joint `kz`) are retired. P2 is built and self-checking. P3 is still the
fallback. T4 trains but has **not** produced completions (`run1` at ~5M
steps is 0% throughout; baseline likewise fails easy surrogate seeds). The
open critical path is finishing courses on the surrogate (baseline floor
first), then measured P3 / real attention / live E5–D6.

The pre-committed go/no-go (abandon T4 if PPO is not training by
2026-08-02 06:00 PST) is **met on the harness side** — PPO trains against
the surrogate. The remaining judgment call is whether further RL budget is
worth it while completion is still zero, versus spending the deadline on
baseline tuning and live evaluation.
