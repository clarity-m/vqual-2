# Axes, signs and frames — the single source

Every sign, frame and axis fact for vqual-2 lives here. `NOTES.md`, `interface.py` and
`sessions/README.md` point at this file rather than restating it, because this project has
already been bitten twice by two copies of a convention drifting apart.

**If you change a sign, change it here first.**

---

## The standing rule

> A convention is settled only by a referee **outside** the thing being tested.

Everything in this simulator is internally consistent, including the parts that are
mirrored. Commanded rate against measured gyro correlates at **+0.96** and proves nothing,
because both ends carry the same mirror. On 2026-07-31 three separate arguments — world
velocity vs body velocity, integrated gyro vs attitude, quaternion integration vs odometry
— all agreed on a yaw sign that was **wrong**, because all three compared sim telemetry
against sim telemetry and were structurally blind to a shared mirror.

Referees that actually work, cheapest first:

| referee | settles | blind to |
|---|---|---|
| **the pilot** (what a key does to the nose, watched live) | any axis | nothing — this is the strongest and the cheapest |
| **gravity** in `HIGHRES_IMU`, near stationary | roll, pitch | yaw; swamped by linear acceleration in fast flight |
| **the camera** (image motion vs commanded rotation) | any axis | needs care with the image-shift sign convention itself |
| **orange pixel mask** vs projected gates | projection/frame maths | nothing, but needs gates in view |
| **a closed loop that flies** (the levelling assist) | roll, pitch | yaw — the assist never touches it |
| **the kinematic identity** `R_wb·a_body + g == dv_world/dt` (`control/sysid_frames.py`) | all three axes at once, *and* the sign of world gravity | needs flight covering a wide range of headings; degenerate if the drone never turns |

The kinematic identity is the only referee in this table that works **in fast flight** —
gravity is swamped there, and it is precisely where a plant fit lives. It spans three
independent streams (`HIGHRES_IMU`, `ATTITUDE`, `LOCAL_POSITION_NED`), so no shared mirror
can satisfy it, which is the property the three failed arguments above all lacked.

Reading the source is **not** a referee. `KEYS_AXIS` says which key is positive; it can
never say which physical direction that is.

---

## Canonical frames

Textbook NED throughout, for both body and world:

    x forward / north      y right / east      z DOWN

    roll_rate  > 0  ->  right wing down
    pitch_rate > 0  ->  nose up
    yaw_rate   > 0  ->  nose right

Camera: shares the body origin, tilted **20° up**, pinhole, no distortion.
640 × 360, `cx`/`cy` = 320/180, `fx` = `fy` = 320.

---

## The mirror — WHICH STREAMS, NOT "the sim"

**The mirror covers rates and attitude. It does NOT cover position.** Saying "the sim is
mirrored" as a blanket fact is how a half-corrected model gets built.

| stream | mirrored? | correction |
|---|---|---|
| commanded body rates (`SET_ATTITUDE_TARGET`) | **yes** | `× -1` |
| `HIGHRES_IMU` gyro | **yes** | `× -1` |
| `ATTITUDE` roll | **yes** (per-field, below) | `truth_roll = ATTITUDE.roll` |
| `ATTITUDE` pitch | **yes** | `truth_pitch = ODOMETRY.pitch` |
| `ATTITUDE` yaw | **yes** | `truth_yaw = -ATTITUDE.yaw` |
| `LOCAL_POSITION_NED` x,y,z,vx,vy,vz | **no — canonical NED** | none |
| `ODOMETRY` velocity | **unknown — avoid** | use `LOCAL_POSITION_NED` velocities |
| `HIGHRES_IMU` accelerometer | **no — canonical** | none (settled 2026-08-01, below) |

### VQ1 truth streams are each wrong on a different axis

    truth_roll  =  ATTITUDE.roll      # == -ODOMETRY.roll
    truth_pitch =  ODOMETRY.pitch     # == -ATTITUDE.pitch
    truth_yaw   = -ATTITUDE.yaw

Refereed against gravity on a parked drone, session `20260731-135735` (`|a|` = 9.8100, so
genuinely at rest); magnitudes agree to five decimals and only signs differ.

**Where this bites:** mixing a corrected attitude with an uncorrected position produces a
model that is mirrored in exactly one term. It fits the recordings, yields sensible
trajectories, and silently mistunes every gain trained against it — presenting as
"great on the surrogate, bad in the sim", which reads like a sim-to-real gap and sends you
tuning the noise model. `perception/label.py` needs its yaw negation for precisely this
reason: its position input is clean and its yaw input is not, so nothing cancels.

---

## Where signs are applied in code

**Rate signs are applied in the link layer ONLY** — when encoding a MAVLink message, and
correspondingly when reading `HIGHRES_IMU`'s gyro. No file above the link layer may contain
a sign flip. `interface.py` states this as `SIGN_ROLL = SIGN_PITCH = SIGN_YAW = -1`.

`perception/label.py` is the one deliberate exception, and it is not a rate: it negates
`ATTITUDE.yaw` because it consumes attitude and position together and they disagree.

---

## Keyboard → physical, as actually flown

`ACRO_ROLL = -2.5`, `ACRO_PITCH = +2.5` (negated at the stick), `ACRO_YAW = +2.0`.
`cmd.csv` records the value sent over MAVLink, post-assist, with no further sign applied.

| key | `cmd.csv` rate | the airframe does |
|---|---|---|
| `d` | −2.5 | roll **right** |
| `a` | +2.5 | roll **left** |
| `w` | −2.5 | nose **up** |
| `s` | +2.5 | nose **down** |
| `q` | +2.0 | yaw nose **left** |
| `e` | −2.0 | yaw nose **right** |

Every row is consistent with a −1 mirror on all three axes: to make the airframe do the
canonical-positive thing, you send a negative rate. **The mirror is a property of the
simulator and has not changed.** The `cmd.csv` column above is the physical fact; which
key produces it is a binding, and one of them was rebound:

> **`q`/`e` were REBOUND 2026-08-01** (Claire, after the first successful slow session), so
> they now read the way the letters suggest. Before that date `q` sent −2.0 and `e` sent
> +2.0 — i.e. **every session recorded up to and including 2026-08-01 was flown with `q` =
> nose-right**. Sessions are unaffected either way: `cmd.csv` stores rates, not keys, and
> −2.0 means nose-right in all of them. `teleop.py` records the live bindings in
> `events.jsonl` under `session_start.keys_axis` from the rebind onward.

**That original inversion is the fact that settled the yaw sign**, after three
telemetry-versus-telemetry arguments got it wrong (Claire, 2026-07-31). Rebinding the keys
does not weaken that evidence — the pilot's observation was about what the airframe did,
not about which letter she pressed. `w` = nose up still follows the same pattern and has
**not** been rebound, because it is the axis every recorded session was flown with.

---

## Camera geometry

True vertical FoV is **58.7°**, not the 90° the spec calls "VFoV" — that figure is the
*horizontal* FoV. The frame spans **+49.4° to −9.4°** about body-forward.

Body-forward therefore renders **below** image centre, at

    v = 180 + 320·tan(20°) = 296   of 360

which is the binding constraint on this course: a gate at own altitude renders low, and
pitching down to accelerate pushes it lower.

Body(FRD) → camera(x right, y down, z forward) is `RELABEL @ Ry(-20°)`. **The tilt is
negative**: forward `(1,0,0)` maps to `v = CY - FY·tan(θ)`, and a camera pointing *up* must
put forward *below* centre, which needs θ < 0. Derived, then confirmed by sweep.

Range from apparent size, spec-exact on the 1500 mm inner aperture:

    range_m = 320 × 1.5 / gate_px = 480 / gate_px

Valid only for a fronto-parallel gate. An oblique gate projects narrower, so size-only
range is biased **long** — and it is used exactly when the PnP fit that would correct it
has failed.

`SOLVEPNP_IPPE_SQUARE` **fixes its object-point winding**: TL, TR, BR, BL with +y up in
object space. Any other order returns a silently garbage pose.

---

## Degeneracies that produce confident wrong answers

Both of these have already cost this project multiple sessions. They are not hypotheticals.

1. **Yaw sign near the 180° start heading.** `+180` and `−180` are the same number, so
   body-right resolves identically under either hypothesis and every check passes. Hid
   vqual-1's yaw error for three sessions — fine at the start heading, growing with every
   degree of turn. Never settle a yaw sign here.
2. **Projection checks with the target near image centre.** Both yaw hypotheses predict the
   same pixel there, so a visual overlay check passes regardless. This is how `label.py`
   shipped with the wrong yaw sign and *passed inspection*. Score only on frames where the
   competing hypotheses differ by a wide margin — the separation filter is the entire test.

---

## Provenance

| fact | referee | where |
|---|---|---|
| roll, pitch rates mirrored | camera taps + gravity + assist flies stable | `camreferee.py`, `20260731-144815` |
| yaw rate mirrored | pilot's keybinds (as bound on 2026-07-31); camera `E` tap; orange mask | `20260731-203428` |
| `truth_yaw = -ATTITUDE.yaw` | orange mask, 18.0 px vs 169.6 px, 267 frames, 2 sessions | `perception/label.py` regression test |
| same, independently | kinematic identity, 0.050 m/s² median vs 0.110 for `+yaw`, 6x on the tail, 26 321 samples | `control/sysid_frames.py`, re-runs per session |
| `g_z = +9.81`, i.e. world z really is down | same sweep — flipping gravity is the worst of the sixteen candidates at 19.6 m/s² | `control/sysid_frames.py` |
| `LOCAL_POSITION_NED` canonical | HUD readout vs motion a human watched | forward/right/up all negative; VQ1 descends |
| camera tilt negative | derivation + convention sweep | `perception/label.py` |
| PnP winding | PnP range −18.6 m vs size range −0.64 m on the same quad | `perception/detect.py` `score()` |
| accelerometer canonical | complementary filter on `HIGHRES_IMU` scored against VQ1 truth pose over 5 sessions; flipping the accel sign is **75×** worse | `test_imu_level.py` |

### The accelerometer is NOT mirrored — measured 2026-08-01

The gyro carries the −1 mirror; the accelerometer does not. Sweeping both signs and
scoring a gyro+accel complementary filter against VQ1 truth (`truth_roll = ATTITUDE.roll`,
`truth_pitch = ODOMETRY.pitch`) over 55 000 IMU samples separates them:

| gyro | accel | roll median | pitch median |
|---|---|---|---|
| −1 | **+1** | **1.12°** | **1.54°** |
| +1 | +1 | 8.11° | 3.01° |
| −1 | −1 | 178.9° | 21.8° |
| +1 | −1 | 171.9° | 21.9° |

So at rest, level, the accelerometer reads `(0, 0, −g)` and

    roll  = atan2(−ay, −az)        pitch = atan2(ax, hypot(ay, az))

This is an **outside** referee in the sense the standing rule demands: VQ1's truth pose is
not derived from `HIGHRES_IMU`. Two honest limits — the two signs are not settled equally
(flipping the accel is 75× worse, flipping the gyro only 4.2×, because the accel trim drags
a wrong-signed integration back toward truth), and it was measured on the VQ1 build, like
everything else here.

**Consequence:** roll and pitch are observable under VQ2 from a permitted stream, which is
what `teleop.py --imu-level` uses. Yaw is not, and is not estimated.

---

## Still unverified

* **Camera roll extrinsic.** The convention sweep scored `+1` and `−1` identically — the
  test cannot resolve it. Label overlays constrain it indirectly; not directly tested.
* **Camera tilt magnitude.** Sign measured, 20° taken from spec.
* **Camera–IMU time offset.** Unmeasured. `frames.csv sim_time_ns` is capture-side and
  trails `t_recv_wall_ns` by ~38 ms of encode plus UDP, so the residual should be small —
  but "should be" is not a measurement. Needs high angular rate to observe at all
  (observability scales with `ω·t_d`), obtainable hovering with sharp taps.
* **`ODOMETRY` velocity frame.** Appears to be built on the mirrored yaw. Avoid.
* **`HIGHRES_IMU` accelerometer axis signs.** Never refereed *on their own*. The kinematic
  identity constrains them jointly with attitude — it could not close to 0.050 m/s² if the
  accelerometer axes disagreed with the attitude convention — but the sweep varied attitude
  and gravity signs, not accelerometer signs, so a compensating flip in both would survive
  it. Treat as constrained, not verified.
* **Everything was measured on the VQ1 build.** Commands traverse the same MAVLink path and
  the spec diff says only §4.5 Telemetry changed, so it should carry to VQ2 — an inference,
  not a measurement.
