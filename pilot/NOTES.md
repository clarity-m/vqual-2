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
| hover thrust | **~0.27** (not the 0.55 first guessed; refined from 0.25 in flight 2026-07-31). `THRUST_HOVER` is only the default — override per flight with `teleop.py --hover` |
| `SIM_RESET` settle | **3–4 s**. Wait for `race_start_boot_time_ms` to change, then check pose, then a guard margin — a fixed sleep samples mid-transition and returns the *previous* pose |

Physics is unchanged from VQ1 (all three spec revisions diffed) — only §4.5 Telemetry
changed, so the VQ1 *build* is a valid rig for this plant.

**But vqual-1's own recordings are not system-ID data** (checked 2026-07-31; an earlier
note here claimed they were). They are `meta.jsonl` only — `t, x, y, z, roll, pitch, yaw,
gate, det` at **~9 Hz**, with no gyro, no thrust, no command channel and no motor outputs.
Differentiating 9 Hz attitude for a racing quad aliases rather than merely adding noise,
and the pose stream carries outliers (one frame pair implies 1881 m/s). Their real value is
elsewhere: pose paired with gate truth makes them **labelled perception data**.

The usable plant data is the 2026-07-31 VQ1-build sessions recorded through `teleop.py`,
now tracked in `sessions/` — see `sessions/README.md`.

### The VQ1 truth streams are each wrong on a different axis (2026-07-31)

Moved to **`CONVENTIONS.md`** — the per-field corrections, the gravity referee that
established them, and the yaw settlement all live there now. In one line:

    truth_roll = ATTITUDE.roll     truth_pitch = ODOMETRY.pitch     truth_yaw = -ATTITUDE.yaw

Also measured here, and real rather than an offset: **the launch pad is inclined 17.8°
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

## Sign conventions

**`CONVENTIONS.md` is the single source for every sign, frame and axis fact.** It carries
the per-stream mirror table, the keyboard-to-airframe mapping, the camera geometry, the
provenance of each fact, and the list of what is still unverified.

The two things worth knowing before you open it:

* **The mirror covers rates and attitude. It does NOT cover position.**
  `LOCAL_POSITION_NED` is plain canonical NED. "The sim is mirrored" as a blanket
  statement is how a half-corrected surrogate gets built.
* **A convention is settled only by a referee outside the simulator.** Commanded rate
  against measured gyro correlates at +0.96 and proves nothing. On 2026-07-31 three
  telemetry-versus-telemetry arguments agreed on a yaw sign that was wrong; the pilot's
  keybinds settled it in one sentence.

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

Indoor hangar. **17 gates** (Claire-observed 2026-07-31, superseding an earlier "~20,
unconfirmed"). **The path winds** far more than VQ1's near-straight line — this is the real
difficulty. Support columns are numbered "Station N" and serve as a position ruler: 7
stations in 19.29 s (2.76 s/station, constant cruise after ~10 s of acceleration).

**How the course LOOKS — appearance, colour masks, signage, the lit active gate, and the
station numbering — lives in `perception/NOTES.md`.** It is not restated here; two copies
of a measurement drift, and the vision file is the one that gets updated.

## Gotchas

* **ACRO is the only flight mode reachable** — not a choice we made. The sim is built on the
  DCL commercial game (hence the `DCL` bit-16 `type_mask`), whose pak carries ACRO / ANGLE
  ("manual throttle, stabilized") / ARCADE / GPS and their rate presets — but this build's
  menu exposes graphics and sound only, `Input.ini` binds nothing mode-related, and the
  `.sav` is an opaque UE4 blob. Settled 2026-07-31; do not re-investigate. **A string in the
  pak is evidence the code exists, not that we can reach it.** The `F11` levelling assist
  exists because of this.
* **Only one client can hold UDP 14550** — teleop and the pilot cannot run together.
* **The `F11` levelling assist changes what a session is good for.** `cmd.csv` still holds
  exactly what went out over MAVLink (`teleop.py:1018` records post-assist rates), so the
  data is *valid* — but the commands are generated from the state by the outer loop, and
  `LEVEL_MAX_RATE` clamps it at 3.0 rad/s, so the excitation is smoother and smaller than
  it looks. Assist ON: `20260731-195307`, `20260731-204841`. Assist OFF: `143025`,
  `144815`, `150712` — those three stay the primary rate-loop identification data. Yaw is
  never touched by the assist on any of them.
* **Frame/telemetry alignment: use `frames.csv sim_time_ns`, not `t_recv_wall_ns`.** The
  receive stamp trails capture by ~38 ms of JPEG encode plus UDP, which at racing rates is
  tens of pixels. Both are on the same wall epoch as the telemetry timestamps.
* **Every VQ1 gate shares one normal, along world x** (Claire, 2026-07-31), so its aperture
  lies in the world y-z plane. That is what makes labelling possible from gate centres
  alone — orientation needs no estimation. Do not assume it carries to VQ2, whose path winds.
* The example client's `main.py` calls `get_thread_for_join()` on a `TimeSync` built via
  its constructor rather than `create_timesync()`, so `.thread` is `None` and the join
  raises on exit. Don't copy that shutdown path.
* The example sends at 250 Hz; spec 4.4 caps below 100 Hz. teleop uses 50 Hz.
* CAEN VM bring-up, the file-drop bridge, and data-movement rules: `../user-vm-cmds.md`.

## Open

* **`R_COMMIT`** (attention handoff range) unset — needs a real approach measurement.
  Measurable from VQ1 flying, since gate dimensions are identical.
* **The VQ2 course cannot be mapped FROM TELEMETRY.** §9.3 blocks gate geometry *and* both
  pose streams, and world-frame mapping needs position and attitude. So gate count and the
  vertical profile have no telemetry measurement available. VQ1's map is a different course
  (6 gates, ~167 m).
  **It can be mapped from vision, and that is in bounds** — the camera is permitted, so
  anything derived from it is fair game, unlike absolute yaw which came from a blocked
  stream via the actuator. Training mode is free and unlimited and the course is fixed, so:
  fly slow laps, run SfM offline over gates plus background (the Station columns are
  uniquely numbered, which solves data association), and relocalise against the result at
  race time. Scale is free from the known 1500 mm aperture. Build it **relative** — only
  ever query "where is gate k+1 relative to gate k" — and global drift stops mattering.
  **Started 2026-07-31** (`perception/mapbuild.py`, `perception/stations.py`); it builds but
  is not yet usable geometry — status, measurements and the current failure mode are in
  `perception/NOTES.md`. Until it is trustworthy, downstream must randomise over the course
  rather than know it, and even once it exists the map is a prior for pre-turning and
  attention, never terminal guidance: live vision must always be able to override it.
* Frames are still backed up nowhere — gigabytes of JPEG, excluded from git, one laptop.
  The telemetry CSVs are now tracked, so the plant fit no longer depends on that laptop.
