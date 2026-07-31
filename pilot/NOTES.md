# vqual-2 — notes

Source of truth. Spec `260624_Technical_Spec_0003.pdf` (VADR-TS-003 iss 00.03),
simulator AI-GP v1.0.3391, example client `PyAIPilotExample-v4`.

**Deadline: 2026-08-03 06:00 PST.**

Measured facts and binding decisions only. Textbook dynamics are not restated here.

## What VQ2 blocks (spec 9.3)

`ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY`, and gate geometry (track packets still
arrive, geometry nulled). Remaining: `HEARTBEAT`, `HIGHRES_IMU`, `TIMESYNC`, `COLLISION`,
`ACTUATOR_OUTPUT_STATUS`, race-status packet (id 1), camera on UDP 5600.

**There is no pose to referee a perception pipeline against.** `active_gate_index` in the
race-status packet is the only externally-sourced spatial fact — sim-computed, advances on
a genuine crossing.

Spec §4.3 still lists `ATTITUDE` as Simulator→Client, contradicting §9.3. The spec is
internally inconsistent; measurement wins.

Two sim modes (9.2): **Training** (free) and **Qualification** (timed, counts). Human
interaction during a submitted timed run is immediate DQ (§7), max 8 min. Teleop is
Training-only. Completing the course auto-submits.

## Geometry (spec 3.7, 3.8)

* Gate outer 2700 mm, **inner 1500 mm**, depth 260 mm. Drone 280 × 280 × 160 mm.
* Camera at body origin, tilted **20° up**. Body NED: x fwd, y right, z down.
  Body → IMU identity.
* Pinhole, no distortion: 640 × 360, cx/cy 320/180, **fx = fy = 320**.
* The spec's "VFoV = 90°" is the **horizontal** FoV. True VFoV is **58.7°**, so the frame
  spans **+49.4° to −9.4°** about body-forward. That narrow lower edge is the binding
  constraint on this course: a gate at own altitude renders low, and pitching down to
  accelerate pushes it lower.
* Known inner square ⇒ monocular rangefinder, spec-exact: **`range_m = 480 / gate_px`** —
  but this assumes a **fronto-parallel** gate. An oblique gate projects narrower, so the
  size-only estimate is biased **long**, and it is used exactly when the pose fit that
  would correct it has failed. Treat size-only range as uncertain, not metric.
* Four corners ⇒ `solvePnP` ⇒ gate pose. The square is symmetric, so PnP gives an
  **undirected** normal plus a two-fold tilt ambiguity near head-on. Direction is resolved
  by signing it toward the camera (for a gate not yet crossed we are on its approach side
  by construction); tilt by temporal consistency, or declared invalid.

## Measured against the live sim

| fact | value |
|---|---|
| `HIGHRES_IMU` rate | **61.4 Hz on the CAEN VM** (18 Hz on the laptop was a LOW-FRAMERATE symptom, not a design limit) |
| camera | every packet delivered **twice**; dedup by frame_id ⇒ 30.05 fps. ~60 fps means you are counting duplicates |
| sim realtime factor | 1.005 on the VM |
| `COLLISION` | a **contact sample**, not a crash — parked drone emits ~250/s. Count episodes (gap > 0.5 s) |
| `COLLISION.horizontal_minimum_delta` | impulse magnitude in kg·m/s, despite the name |
| baro | `abs_pressure`, `pressure_alt`, `temperature` all `nan`. No barometric altitude |
| race status | keeps counting from the previous race until reset; `race_time_s` can read hundreds of seconds on a fresh connect |
| hover thrust | **~0.25** (not the 0.55 first guessed) |
| `SIM_RESET` settle | **3–4 s**. Wait for `race_start_boot_time_ms` to change, then check pose, then a guard margin — a fixed sleep samples mid-transition and returns the *previous* pose |

Physics is unchanged from VQ1 (all three spec revisions diffed) — only §4.5 Telemetry
changed. **All vqual-1 recordings are valid system-ID data for this plant.**

### The VQ1 truth streams are each wrong on a different axis (2026-07-31)

Refereed against gravity in `HIGHRES_IMU` on a parked drone (session 20260731-135735,
`|a|` = 9.8100 so genuinely at rest). Magnitudes agree to five decimals; only signs differ:

| | gravity (referee) | `ATTITUDE` | `ODOMETRY` (from quaternion) |
|---|---|---|---|
| roll | **+0.00025** | +0.00025 ✓ | −0.000247 ✗ |
| pitch | **−0.31068** | +0.31068 ✗ | −0.31066 ✓ |
| yaw | — | −3.140550 | −3.140550 |

    truth_roll  =  ATTITUDE.roll      # == -ODOMETRY.roll
    truth_pitch =  ODOMETRY.pitch     # == -ATTITUDE.pitch

Independently reproduces vqual-1's "ATTITUDE pitch and ODOMETRY roll are sign-inverted",
re-derived rather than inherited. **Anything using these as ground truth must apply the
per-field correction first** — a fit refereed against raw `ODOMETRY` roll or raw
`ATTITUDE` pitch is mirrored, and mirrored silently.

**Yaw is NOT settled and cannot be settled from a parked recording.** Both streams read
−179.9°, which is the degenerate point where a sign flip is invisible — the same trap that
hid vqual-1's yaw error for three sessions (fine at the 180° start heading, growing with
every degree of turn). Needs data at a heading well away from 180°.

Also measured, and real rather than an offset: **the launch pad is inclined 17.8°
nose-down** (roll 0.01°, so a clean pitch incline).

### Two sim builds ship side by side (measured with `msgscan.py`, 2026-07-31)

`AIGP_3391` is VQ2. `AIGP_VQ1_3391` is VQ1 **and still streams pose**:

| message | VQ2 build | VQ1 build |
|---|---|---|
| `ATTITUDE` | — | **113.8 Hz** |
| `LOCAL_POSITION_NED` | — | **92.8 Hz** |
| `ODOMETRY` | — | **72.2 Hz** |
| `ACTUATOR_OUTPUT_STATUS` | 96.5 Hz | 92.8 Hz |
| `HIGHRES_IMU` | 62.9 Hz | 47.7 Hz |
| `HEARTBEAT` | 10.2 Hz | 10.0 Hz |
| `ENCAPSULATED_DATA` | 4.1 Hz | 4.0 Hz |

Three consequences:

* **`ATTITUDE` at ~114 Hz is the physics rate**, so `HIGHRES_IMU` is a deliberate
  downsample, not a bandwidth limit.
* **IMU rate is load-dependent** — 62.9 vs 47.7 Hz between builds. There is no nominal
  period to design around; anything fitted from recordings must weight by actual
  timestamp deltas. VQ1 recordings carry a *sparser* IMU than the VQ2 pilot will see,
  which errs safe.
* **`ACTUATOR_OUTPUT_STATUS` (~95 Hz) is permitted under VQ2 and currently unused.** It
  is the highest-rate signal we are allowed and reflects what the FC did with our rate
  command — the best available observable for fitting the inner loop.

## Sign conventions — PARTIALLY settled

| axis | sim vs canonical body NED | evidence |
|---|---|---|
| roll rate | **INVERTED** (hence `ACRO_ROLL = -2.5`) | flight |
| yaw rate | **INVERTED** | flight, Claire 2026-07-31: `E` commands +2.0 rad/s and the drone yaws LEFT. Same on both builds |
| pitch rate | **UNVERIFIED** | rests only on the code comment `w = nose down = forward`. Nobody has watched it |

**Correction, 2026-07-31.** An earlier version of this section declared all three settled,
with pitch and yaw "to spec". That was wrong. It came from reading `KEYS_AXIS` — which
says only which key is *positive*, never which physical direction that is — and filling
the gap with an assumed convention (`E` = yaw right). The assumption then got recorded as
if it were a trace, and was used to retire a standing "pitch and yaw want confirming"
caveat that had been correct all along.

The lesson is the one this project keeps relearning from the other side: **a convention
must be refereed against an independent observation, never against an assumption about
what a key or a comment means.** A code comment is a write-up, and "re-derive rather than
trust the write-up" applies to our own files too.

**Pitch is still open.** Watch the nose while pressing `W` and settle it.

For system ID this does not corrupt data: `cmd.csv` records what was commanded and the
VQ1 truth streams record what happened, so a fit recovers the true sign on its own. It
matters for hand-labelling a maneuver direction, and for teleop feeling right.

**Where a sign error is actually dangerous: the surrogate fit** — the only place a world
frame appears, and the only place the error is silent. See `interface.py`.

`YAW_GYRO_SIGN` remains unverified but feeds only the HUD readout.

## Control architecture — decided

Flight stack is **body-referenced end to end**; nothing converts to a world frame.

**Rate-only** (`SET_ATTITUDE_TARGET`, `type_mask` bit 16 = DCL extension making body rates
real rad/s). Roll owns the trajectory; yaw is a free gimbal for the camera and costs
nothing in path. Guidance target is a virtual approach point at `pos_body + d·normal_body`,
which is what makes "enter at the gate normal" expressible.

Interface frozen in `interface.py` (73-D observation, 4-D action). Perception above it,
control below it, split across two people.

### Why velocity setpoints are unusable

**The sim ignores `coordinate_frame`.** `MAV_FRAME_BODY_NED` behaves identically to
`MAV_FRAME_LOCAL_NED` — `vx`/`vy` are always world axes. Verified in flight: press forward,
yaw 60°, press forward again, drone travels the same world direction both times.

So the control interface demands a frame whose relation to the body is unobservable.
Roll/pitch are recoverable from gravity; **yaw is not recoverable from any permitted
stream** (gravity symmetric about it, no magnetometer). Body rates sidestep this entirely.

### Velocity direction from the accelerometer

Thrust acts along body −z by definition, so it contributes zero to horizontal body
accelerometer components — in flight those measure only drag, antiparallel to airspeed:

    bearing = atan2(−ay, −ax)

Instantaneous, no integration, nothing to diverge. Gated on `|(ax,ay)| > ALIGN_MIN_ACCEL`
since drag → 0 near hover. Measures **airspeed** direction (this sim appears windless) and
says nothing about the vertical. At 20° nose-down, `g·tan20° = 3.6 m/s²` — a clean signal.

Observed: yaw produces a genuine skid (nose one way, momentum the other), i.e. the sim
models yaw realistically. Good news for transfer to the physical qualifier.

### Absolute yaw is out of bounds — DECISION 2026-07-30

Angle mode works: clearing `ATTITUDE_IGNORE` makes the sim honour an attitude quaternion,
achieved bank tracking commanded at **slope 0.890, rms 1.90°** (`probe_angle.py`).

Rejected anyway. A quaternion encodes absolute yaw, and commanding absolute yaw is reading
it — a compass obtained through the actuator instead of the blocked telemetry stream. The
standard: *does it give information no permitted stream provides?* Roll/pitch self-levelling
— no (gravity is observable; quality gain, not capability gain). Absolute yaw — yes, and
it is exactly what §9.3 removes. §9.2 reserves a code audit for "manipulation of the
simulator constraints"; not a bet worth placing.

**Yaw-freedom probe: NEGATIVE** (`probe_yawfree.py`). Quaternion roll/pitch with yaw left
to a commanded body *rate* does not work — yaw is welded to the quaternion. The positive
control (acro ±1.5 rad/s) moved the yaw mixer channel +0.51/+0.66/+0.52 while both test
conditions produced noise. Corroborating: every attitude-active trial shows `zgyro` ≈
−0.12..−0.20 rad/s regardless of commanded sign — the FC slewing toward the quaternion's
absolute yaw.

Consequence: `ATTITUDE_IGNORE` stays set. Do not reopen for a smoother inner loop.

## The course

Indoor hangar — dark, lit signage, ceiling light strips, support columns. Not visually
hostile; detection looks no harder than VQ1.

* **~20 gates** (Claire-observed, unconfirmed), bright orange/red and glowing against dark.
* **Cyan guidance corridor** showing roughly the next 5 gates. Present in submission mode,
  not just training. **Absent from long stretches of real flight** — hence not load-bearing.
* **The path winds** far more than VQ1's near-straight line. This is the real difficulty.
* **White ceiling lights** are the main false-positive risk; the **first gate blooms**.
* Columns are labelled "Station N", left row counting down and right row up, sum invariant
  at 40–41 — two position rulers that error-check each other. 7 stations in 19.29 s
  (2.76 s/station, constant cruise after ~10 s of acceleration).

Measured on the parked start frame (`evidence/2026-07-30-startview.jpg`, OpenCV HSV, hue 0..179):

| class | mask | share | mean S | mean V |
|---|---|---|---|---|
| orange gate | `(H<=12 or H>=170) & S>110 & V>110` | 2.0% | 202 | 241 |
| cyan path | `85<=H<=100 & S>110 & V>110` | 1.5% | 197 | 169 |
| white lights | `S<50 & V>200` | 0.9% | 3 | 238 |

66.5% of the frame is V<40; the three classes separate on saturation alone (202/197/3), and
the lights are additionally separated by position (centroid y=66). Bloom cost only 1.7% of
the gate bounding box, so the orange mask is near-solid — but fill contours rather than
trusting a filled mask.

## teleop.py

    python3 pilot/teleop.py              # fly + record + live view
    python3 pilot/teleop.py --listen     # record only, sends nothing (safe mid-flight attach)
    python3 pilot/teleop.py --no-view    # no cv2 window, lower loop jitter

Flies like an acro quad — no self-levelling, releasing keys is not a hover. Throttle is a
*held value* that integrates while a key is down; `F10` panics back to hover.

`W/S` pitch, `A/D` roll, `Q/E` yaw, `UP/DOWN` throttle, `LCTRL` boost, `LALT` precision,
`C` align-to-velocity. Commands on F-keys: `F5` arm, `F6` disarm, `F7` zero heading, `F8`
quit, `F9` reset, `F12` marker.

**Commands live on F-keys because the sim eats keystrokes.** The global hook is not
exclusive — every keystroke reaches teleop *and* the sim, which binds ordinary game keys
(SPACE restarts the run; found the hard way). Continuous axes sit on letters/arrows where an
echo is cosmetic. `suppress=True` was considered and not shipped: unverifiable here, and a
control scheme that fails silently is worse than one that collides visibly.

HUD `hdg` is **diagnostic only** — the unbounded integral of `zgyro`; nothing in the control
path reads it.

Writes `pilot/sessions/<timestamp>/`: `imu.csv`, `race.csv`, `cmd.csv`, `frames.csv`,
`actuators.csv`, `collisions.csv`, `events.jsonl`, `frames/*.jpg` (raw). `cmd.csv` pairs
input with response for system ID.

## Gotchas

* **Only one client can hold UDP 14550** — teleop and the pilot cannot run together.
* The example client's `main.py` calls `get_thread_for_join()` on a `TimeSync` built via
  its constructor rather than `create_timesync()`, so `.thread` is `None` and the join
  raises on exit. Don't copy that shutdown path.
* The example sends at 250 Hz; spec 4.4 caps below 100 Hz. teleop uses 50 Hz.
* CAEN VM bring-up, the file-drop bridge, and data-movement rules: `../user-vm-cmds.md`.

## Open

* **`R_COMMIT`** (attention handoff range) unset — needs a real approach measurement.
* **Gate count** (~20) wants confirming.
* **Vertical profile** of the course unknown. VQ1 descended ~20°, putting gates below the
  frame's −9.4° edge; check whether VQ2 does the same.
* **Acro ↔ Stabilized flight-mode toggle** unexplored (`SetFcFlightMode` in the pak, HUD
  shows `FLIGHT MODE: ACRO`). If the FC self-levels while we send ordinary body rates, no
  quaternion is transmitted and absolute yaw never enters the protocol — clean by
  construction. All probes above were run in ACRO, so their results are conditional on it.
* Angle-mode slope 0.890 and the shallow negative side are probably ground contact
  resisting roll; would need re-measuring in flight if it ever mattered.
* **Recordings are backed up nowhere** and are excluded from git. The plant fit depends on
  them.
