# vqual-2 — notes

Source of truth for this project. Spec: `260624_Technical_Spec_0003.pdf` (VADR-TS-003,
issue 00.03, 2026-06-24). Simulator: AI-GP v1.0.3391, example client `PyAIPilotExample-v4`.

## What VQ2 changes

Spec 9.3 blocks these outright:

| Blocked | Consequence |
|---|---|
| `ATTITUDE` | no orientation from the sim, at all |
| `LOCAL_POSITION_NED` | no position |
| `ODOMETRY` | no pose or velocity |
| gate info | track packets still arrive, gate geometry is nulled |

What remains: `HEARTBEAT`, `HIGHRES_IMU`, `TIMESYNC`, `COLLISION`,
`ACTUATOR_OUTPUT_STATUS`, the encapsulated race-status packet (id 1), and the camera
stream on UDP 5600.

Consequence worth stating plainly: **there is no pose to referee a perception pipeline
against**. The only externally-sourced spatial fact is `active_gate_index` in the race
status packet — it advances when a gate is genuinely crossed, computed by the sim rather
than by us. Anything else about where the drone is has to be inferred from camera + IMU.

Two flight modes exist in the sim UI (spec 9.2): **Training** (free, not scored) and
**Qualification (VQ2)** (timed, counts). Spec 7: human interaction during a submitted
timed run is immediate disqualification. Teleop is a Training-only tool.

## Geometry / camera (spec 3.7, 3.8)

* Gate outer 2700 × 2700 mm, inner square 1500 × 1500 mm, depth 260 mm.
* Drone 280 × 280 × 160 mm.
* Camera shares the body origin, **tilted 20° up**. Body frame NED: x fwd, y right, z down.
* Pinhole, no distortion: 640 × 360, cx/cy = 320/180, fx/fy = 320/320, VFoV 90°.
* Body → IMU is identity.

## teleop.py

Hand-fly with the keyboard and record every surviving stream. Run it, don't rebuild it:

    python3 pilot/teleop.py              # fly + record + live camera view
    python3 pilot/teleop.py --listen     # record only, sends nothing at all
    python3 pilot/teleop.py --no-view    # no cv2 window, lower control-loop jitter

**Body rates only** (`SET_ATTITUDE_TARGET`), because that is what the autonomous pilot
will command — see the frame problem below. Velocity mode was removed rather than kept
as a convenience, so hand-flown recordings and autonomous runs share one control path.
It flies like an acro quad: **no self-levelling**, and releasing the keys is not a hover.

Keys: `W/S` pitch, `A/D` roll, `Q/E` yaw, `UP`/`DOWN` throttle, `LCTRL` boost, `LALT`
precision; commands on function keys — `F5` arm, `F6` disarm, `F7` zero heading readout,
`F8` quit (cuts throttle), `F9` sim reset, `F10` throttle back to hover, `F12` marker.

Throttle is a *held value*, not a stick deflection: it integrates while a key is down
and stays where you leave it. `F10` is the panic button back to `THRUST_HOVER` (0.55,
unverified). Quitting cuts throttle to zero and disarms.

The heading readout in the HUD is **diagnostic only** — nothing in the control path
reads it. It is the integral of `zgyro`, drifts without bound, and exists so the
recording carries an orientation trace and so the gyro sign can be eyeballed in flight.

**Why commands live on F-keys.** The global hook is not exclusive: every keystroke
reaches teleop *and* the simulator window. The sim is a game and binds ordinary game
keys — SPACE restarts the run (found the hard way; `SpaceBar` is bound in the pak), and
`ESC`/`R`/number keys are the usual menu-and-restart suspects. So continuous axes sit on
letters and arrows, where an echoed keystroke is at worst a cosmetic camera twitch, and
one-shot commands sit on F5–F12, which games essentially never bind. All bindings are in
the `KEYS_*` dicts at the top of teleop.py; nothing else reads raw key names.

Suppressing keys so the sim never sees them (`keyboard`'s `suppress=True`) was
considered and **not** shipped: it could not be verified here, because synthetic
keypresses aren't observable in this environment, and a control scheme that fails
silently is worse than one that collides visibly. If you ever want it, test it
interactively first.

Writes `pilot/sessions/<timestamp>/`: `imu.csv`, `race.csv`, `cmd.csv`, `frames.csv`,
`actuators.csv`, `collisions.csv`, `events.jsonl`, and `frames/*.jpg` (raw, not re-encoded).
`cmd.csv` logs what was commanded, so a later system-ID pass can pair input with response.

`--listen` is the safe way to attach to a flight already in progress: no setpoints, no
arm/disarm, not even a heartbeat.

## Measured against the live sim, 2026-07-30

Not from the spec — observed, and in two places the spec or the example is misleading.

* **Every camera packet is delivered twice.** Naively reassembling gives 60 fps and
  double-writes every frame; deduplicating by frame_id gives 29.7 fps, matching the
  spec's 30 Hz. If a future tool reports ~60 fps, it is counting duplicates.
* **`COLLISION` is a contact sample, not a crash.** A drone resting on the ground emits
  ~250 messages/second. 8 s of a parked airframe = 2032 messages = one contact episode.
  Count episodes (gap > 0.5 s), never messages.
* **`HIGHRES_IMU` arrives at ~18 Hz**, not the 120 Hz physics rate (spec 3.2). Far too
  slow to dead-reckon attitude through a turn; treat it as a coarse accelerometer, not
  an INS.
* `abs_pressure`, `pressure_alt` and `temperature` are all `nan`. No barometric altitude.
* `COLLISION.horizontal_minimum_delta` is impulse magnitude in kg·m/s despite the name.
* Race status keeps counting from a previous race until the sim is reset, so
  `race_time_s` can read several hundred seconds on a fresh connect.

## Gotchas

* **The sim eats your teleop keystrokes.** See the F-key rule above. SPACE restarting
  the run mid-flight is the concrete instance that surfaced this.
* **Only one client can hold UDP 14550.** Teleop and an autonomous pilot cannot run
  simultaneously; the second one to start fails to bind.
* The example client's `main.py` calls `ts_loop.get_thread_for_join()` on a `TimeSync`
  built via its constructor rather than `create_timesync()`, so `.thread` is `None` and
  the join raises on exit. Don't copy that shutdown path.
* The example sends commands at 250 Hz; spec 4.4 caps the command rate below 100 Hz.
  teleop sends at 50 Hz.
* `SET_ATTITUDE_TARGET.type_mask` bit 16 (non-standard, DCL extension) opts into body
  rates being real rad/s. Set it.

## The course (Claire's observation + start-frame measurement, 2026-07-30)

The "high-fidelity 3D-scanned environment" of the README does **not** mean a visually
hostile one. It is an indoor hangar: dark, with lit signage, ceiling light strips and
support columns. Detection looks no harder than VQ1, possibly easier.

* **Gates: bright orange, glowing, against a dark background.** 20 of them (VQ1 had 6).
* **A cyan guidance corridor** is drawn along the ground showing roughly the next 5
  gates, as in VQ1.
* **The path winds** far more than VQ1's near-straight line. This, not the visuals, is
  the real difficulty increase.
* **White ceiling lights** are the main false-positive risk, and the **first gate has a
  bloom flare**; the rest are evenly lit.

Measured on the parked start frame (`pilot/evidence/2026-07-30-startview.jpg`, 640x360,
OpenCV HSV where hue is 0..179):

| class | mask | share | mean S | mean V | where |
|---|---|---|---|---|---|
| orange gate | `(H<=12 or H>=170) & S>110 & V>110` | 2.0% | 202 | 241 | centroid (335,174), near frame centre |
| cyan path | `85<=H<=100 & S>110 & V>110` | 1.5% | 197 | 169 | x[100..539] y[138..359], runs to the bottom edge |
| white lights | `S<50 & V>200` | 0.9% | 3 | 238 | centroid y=66, up in the ceiling |

66.5% of the frame is V<40. The three classes separate cleanly on saturation alone
(202 / 197 / 3), and the white lights are additionally separated by position. Bloom cost
only 1.7% of the gate bounding box to a bright desaturated core, so the orange mask is
near-solid rather than hollow — but expect worse on the flared first gate, and fill
contours rather than trusting a filled mask.

**The cyan corridor is the sanctioned answer to the yaw problem.** It is a continuous
cue that shows where the course goes *around a corner*, before the next gate is visible
— which on a winding 20-gate course is most of the time. More importantly its direction
in the image gives heading error **relative to the path**, and that is a body-referenced
quantity obtained from the camera, exactly the way the spec intends (§5.3: Perception
before Control). It is the legitimate substitute for the heading we declined to extract
from the sim.

This closes the architecture: **image-space error -> body rates.** Horizontal offset of
the path/gate drives roll and yaw, vertical offset drives pitch and throttle. No world
frame, no attitude estimate, no heading — the same loop a human acro pilot closes by
eye. Every piece we were missing was missing only because we were trying to fly in a
frame we could not observe.

### Velocity direction is readable from the accelerometer, with no fusion

Thrust acts along body −z **by definition**, so it contributes exactly zero to the
body-frame horizontal accelerometer components. Whatever attitude the drone is at,
`ax`/`ay` can only be measuring non-thrust forces — in flight, drag — and drag is
antiparallel to velocity through the air:

    (ax, ay) ∝ −(vx, vy)_body        bearing = atan2(−ay, −ax)

So the **direction of travel relative to the nose** is an instantaneous algebraic read
of one permitted sensor. No integration, no filter, no drift, nothing to diverge. This
is the single most useful thing the IMU gives us, and it is what makes coordinated
turning possible without any attitude or heading estimate.

Scale check: at a steady 20° nose-down, drag balances `g·tan20° = 3.6 m/s²` — a large,
clean signal. It degrades only near hover, where drag → 0 and the bearing becomes noise,
so it is gated on `|(ax,ay)| > ALIGN_MIN_ACCEL`.

Two honest limits: it measures **airspeed** direction (a wind field would bias it; this
sim appears to have none), and it says nothing about the vertical component.

teleop exposes it as the `C` key: hold to yaw the nose onto the direction of travel,
with the bearing shown live in the HUD as `vel`. The same reading is what a coordinated
turn controller should drive to zero.

### Do not steer with yaw (the obvious controller is wrong)

Yaw in this sim behaves realistically: it rotates the airframe, and momentum carries the
drone on its original path — a skid, nose one way and travel the other (observed in
teleop 2026-07-30). Confirm with the `hdg` readout, which integrates `zgyro`; the IMU is
strapped to the body (§3.8, body→IMU identity) so it cannot see a camera-only rotation.
If `hdg` tracks the yaw input, the body is genuinely rotating.

The naive vision controller — "gate is left of centre, yaw left until centred" —
therefore produces a flat spin while flying straight past the gate. Centring a target in
frame and travelling to it are different jobs:

* **Roll** curves the flight path (`lateral accel = g·tanθ`). This is what turns you.
* **Yaw** points the body-fixed camera so the corridor stays in frame through a corner.

Horizontal pixel error must drive **both**. This is the coordination a human racer does
by reflex, and it is also why the sim not auto-coordinating is good news: what is tuned
here has a chance of transferring to the physical qualifier, where banking is the only
way to turn.

## The frame problem — read this before designing the autonomous pilot

**The sim ignores `coordinate_frame` on velocity setpoints.** Commanding
`MAV_FRAME_BODY_NED` behaves identically to `MAV_FRAME_LOCAL_NED`: `vx`/`vy` are always
world axes. Verified in flight 2026-07-30 (Claire) — press forward, yaw 60°, press
forward again, and the drone travels the same world direction both times.

Put that next to §9.3 and the shape of the problem appears:

> The control interface requires you to command in a frame whose relationship to your
> own body is not observable through any message the competition leaves you.

You must express intent in world axes, while every message that would tell you how your
body is oriented in those axes — `ATTITUDE`, `ODOMETRY`, `LOCAL_POSITION_NED` — is
blocked. Perception is inherently body-referenced ("the gate is 15° left of my nose"),
so *every* vision-driven velocity command needs a body→world rotation, and the rotation
angle is exactly the quantity that was taken away.

What is and isn't recoverable from what's left:

* **Roll and pitch: recoverable.** The accelerometer sees gravity, which fixes the
  down vector whenever the drone isn't accelerating hard. Noisy under racing loads.
* **Yaw: not recoverable from the IMU at all.** Gravity is rotationally symmetric about
  it. Integrating `zgyro` is dead reckoning with no correcting observation — unbounded
  drift, and at ~18 Hz through hard turns it will be bad. There is no magnetometer
  (mag fields are dead alongside the `nan` baro).
* **Yaw from vision: recoverable**, and this is the intended path — spec 5.3 puts
  Perception ahead of Control for a reason. Feature tracking or optical flow between
  frames gives yaw change; a recognisable gate gives an absolute bearing.

**Conclusion, and the decision taken 2026-07-30: build on body rates only.**
`SET_ATTITUDE_TARGET` body rates are body-referenced *by construction* — a roll rate is
a roll rate regardless of where north is — so rate control needs no heading estimate and
sidesteps the whole problem. It is also what a real racing quad does: rates + FPV, which
matches the physical hardware this qualifier feeds into.

teleop is therefore rate-only. Velocity mode was deleted rather than left in as a
hand-flying convenience: keeping it would mean hand-flown recordings came from a control
path the autonomous pilot can never use, and the world-frame trap would stay one
keystroke away.

## Angle mode EXISTS — the sim honours an attitude quaternion (2026-07-30)

`SET_ATTITUDE_TARGET` with `ATTITUDE_IGNORE` **cleared** (mask 16, keeping only the DCL
rad/s bit) is accepted: the drone assumes the commanded bank angle and holds it. So the
sim's FC will close the attitude loop with *its own* attitude estimate — the very thing
§9.3 withholds from us.

Evidence (`pilot/probe_angle.py`, logs in `pilot/evidence/`), measured on the ground with
thrust 0.20 so the drone never lifts. Roll is recovered from the accelerometer's gravity
vector, an observable fully independent of the motor-output channel used to detect it:

| commanded | achieved |
|---|---|
| +5° | +4.08° |
| −5° | −3.61° |
| +10° | +9.52° |
| −10° | −5.17° |
| +20° | +19.65° (reps: 19.75, 19.66, 20.03) |
| −20° | −17.71° (reps: −17.16, −17.75, −17.62) |
| +30° | +24.42° |
| −30° | −29.09° |

Fit: `achieved = 0.890 × commanded`, rms residual 1.90°. It tracks the setpoint, so this
is a real attitude controller and not the airframe tipping onto a mechanical stop.

**Method notes — two earlier versions of this probe gave confident wrong answers:**

1. *Magnitude vs a quiescent floor.* Compared motor asymmetry under an attitude command
   against an undisturbed baseline and concluded "honoured". Wrong: a commanded roll
   *rate* tips the drone, and afterwards the rate loop fights the rocking, producing
   |asym| 0.45 with **zero** commanded. The attitude phase produced *less* than that.
   Fix: test the **sign**, which a disturbance cannot fake.
2. *Detecting readiness from pose.* Waiting for "pad pose + still + idle motors" passed
   instantly, because the drone never leaves the pad during these probes — so the check
   matched the OLD state before the reset landed. Claire, watching live, caught it as a
   ~1350 ms early start. Fix: wait for a discrete event (`race_start_boot_time_ms`
   changing in the race-status packet), *then* look at pose, then a guard margin.

Reset timing generally: a `SIM_RESET` takes **~3–4 s** to fully settle. A fixed sleep
under that samples mid-transition and returns the previous pose — this bit twice.

**Why this matters, and the one question that decides it.** Angle mode would let us
command absolute bank angles without ever estimating our own attitude, cutting the state
estimation burden to navigation alone. But the quaternion also encodes **yaw**, which is
absolute and world-referenced, and absolute yaw is exactly what we cannot know. The
ground test commanded yaw = 0 throughout and the drone could not yaw against friction,
so it says nothing about yaw handling.

### DECISION 2026-07-30 (Claire): do not command absolute yaw

Commanding absolute yaw **is** reading absolute yaw. If the FC servos to a commanded
yaw, our heading in the world frame is then known — a compass, obtained without
receiving a single blocked message. The information arrives through the actuator rather
than the telemetry stream, but it is the same information §9.3 removed along with
`ATTITUDE`. The leak is structural, not incidental: a quaternion cannot be sent without
specifying yaw, so there is no "quaternion for roll/pitch only".

The standard we hold to: **does it give us information no permitted stream provides?**

* Roll/pitch self-levelling — **no**. Gravity is directly observable in `HIGHRES_IMU`,
  which the spec provides. The FC's estimate is cleaner than ours, but that is a quality
  gain, not a capability gain. Defensible.
* Absolute yaw — **yes**. Unrecoverable from any permitted stream (gravity is symmetric
  about it, no magnetometer). Pure capability gain, and exactly the capability §9.3
  removes. Out of bounds.

The contrary reading — that §4.3 lists `SET_ATTITUDE_TARGET` as supported and §9.3
restricts only telemetry — is not obviously wrong, but it does not survive §9.3's stated
purpose ("to ensure competitive integrity"), and §9.2 reserves a code audit where the
Race Director suspects "manipulation of the simulator constraints". Not a bet worth
placing on the qualification.

Engineering argument points the same way: a human acro pilot flies with no attitude
instrument at all, purely from the FPV image. Rate-only plus vision is human-parity,
unambiguously within spirit, and does not rest on a gap that may be patched before
September.

**Consequence:** absolute-yaw commanding is off the table and teleop stays rate-only.

### Yaw-freedom probe: NEGATIVE (2026-07-30, `pilot/probe_yawfree.py`)

Tested whether roll/pitch can come from the quaternion while yaw is left to a commanded
body *rate* — the variant that would leak nothing. It cannot. **Yaw is welded to the
quaternion.**

| condition | paired diffs, yaw mixer channel | reading |
|---|---|---|
| control — acro, ±1.5 rad/s | +0.51, +0.66, +0.52 | strong, consistent |
| test_A — mask 16 (attitude + all rates) | +0.003, −0.009, −0.001 | noise |
| test_B — mask 19 (attitude + yaw rate only) | −0.0009, **−0.199**, −0.0004 | two noise, one outlier |

The positive control is what makes the negative trustworthy: in acro the drone actually
rotated at ±1.0 rad/s on `zgyro` with motor asymmetry ±0.26, splitting cleanly by sign.
The metric sees yaw easily; the test conditions produce nothing.

Corroborating detail: *every* attitude-active trial shows `zgyro` ≈ −0.12..−0.20 rad/s
regardless of commanded sign — the FC slewing toward the quaternion's absolute yaw.
Direct confirmation that attitude mode commands absolute heading.

**Tooling bug worth not repeating:** the verdict rule reported test_B as "signs agree",
which was spurious — two of its three paired differences are ~5e-4, noise that happened
to share a sign, and the third came from the one rep whose reset failed (`not pad-ready`
on both control trials). *A sign test needs a magnitude floor.* The rule only reached the
right answer because it also required agreement in direction with the control.

**Still open, and more promising:** the sim's own **Acro ↔ Stabilized** flight-mode
toggle (the HUD shows `FLIGHT MODE: ACRO`; the pak has `SetFcFlightMode` and separate
Acro/Stabilized setting defaults). That is a different mechanism: if the FC self-levels
while we keep sending ordinary body-rate messages, no quaternion is ever transmitted and
absolute yaw never enters the protocol — clean by construction rather than by masking.
Note also that the probe above was run with the FC in ACRO, so its result is conditional
on that mode.

Also unresolved: slope 0.890 rather than 1.0, and the negative side reading consistently
shallow, are both probably ground contact resisting the roll. Re-measure in flight.

## Sim health — check before trusting any measurement

The sim displayed **`LOW FRAMERATE — ADJUST GRAPHIC SETTINGS`** during the 2026-07-30
session, which means every measurement in this file was taken on a sim that was not
keeping up. Most likely casualty: the **~18 Hz `HIGHRES_IMU` rate**, which is a
suspicious gap against the stated 120 Hz physics rate and may simply be a symptom rather
than a design limit. Clear the warning, then re-measure the IMU rate — if it rises to
60–120 Hz, gyro integration over short windows becomes usable and the control loop gains
real bandwidth, which relaxes several pessimistic conclusions here.

The HUD also shows, for humans only (none of it is in telemetry): speed in km/h, `cam
20°` confirming the spec's camera tilt, the race timer, and the current FC flight mode.

**Deadline: 3 days remaining as of 2026-07-30**, i.e. ~2026-08-02. Recon and a working
detector matter more than a fast lap.

## Open

* **Is yaw physical?** On a real quad, yaw generates NO lateral force — it comes from
  prop reaction torque and only rotates the airframe. Turning the flight path requires
  banking (lateral accel = `g·tanθ`); a real racer who yaws at speed skids, nose one way
  and momentum the other. If this sim swings the trajectory around on yaw alone, it is
  coupling yaw to the velocity direction, and a controller tuned on that will not
  transfer to the physical qualifier in September.
  **Test, needs no pose telemetry:** at speed on a straight, marker (`F12`), hold pure
  yaw ~1 s with no roll, marker again. Then read `yacc` in `imu.csv` between the markers:
  realistic skid gives ~0 (drag only); a coordinated flat turn requires centripetal
  `v·ω`, which at 10 m/s and 2 rad/s is ~20 m/s² ≈ 2g of lateral specific force that a
  level airframe has no way to produce. The contradiction is unmissable.
  Either way yaw is still needed: the camera is body-fixed, so the nose must be pointed
  at the corridor through a corner or the cue we steer by leaves the frame.
* Hover thrust is ~**0.25**, not the 0.55 originally guessed (found in flight).
* Sign conventions — **SETTLED 2026-07-31**, retiring the "pitch and yaw want confirming"
  caveat that stood here. Traced through `KEYS_AXIS` → `ACRO_*` and flight-confirmed:
  roll rate is **inverted** vs NED (hence `ACRO_ROLL = -2.5`); **pitch and yaw are to
  spec** (`W` = +1 = nose down → −2.5, correct NED; `E` = +1 = yaw right → +2.0). Two of
  three axes never deviated, and roll announced itself on the first flight.
  The residual risk is not here — it is the **surrogate fit**, the only place a world
  frame appears, where a wrong sign is silent rather than self-announcing. See the sign
  block in `interface.py`.
* `YAW_GYRO_SIGN` is **unverified**, and now only affects the diagnostic readout. Yaw
  right; if the heading falls, flip it.
* Whether `zgyro` is even populated in flight is unconfirmed — it reads 0.0 while
  parked, which is correct but uninformative. The HUD prints `(no gyro!)` if `zgyro`
  never goes non-zero.
* Vertical profile of the course unknown. In VQ1 the course descended ~20° while the
  camera is fixed 20° UP, so the next gate fell below the frame — check whether VQ2
  does the same.
