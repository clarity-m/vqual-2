## teleop.py

    python3 pilot/teleop.py --slow-lap      # <- the mapping-lap command (see below)
    python3 pilot/teleop.py                 # fly + record + live view
    python3 pilot/teleop.py --listen        # record only, sends nothing (safe mid-flight attach)
    python3 pilot/teleop.py --no-view       # no cv2 window, lower loop jitter
    python3 pilot/teleop.py --no-thrust-snap  # old throttle, holds where you leave it

### `--slow-lap` — the one flag for a mapping lap

Added 2026-08-01. Equivalent to `--shaping slow --imu-level`: softer, capped rate
commands, and the `F11` levelling assist made available under VQ2. **It arms nothing by
itself** — `F11` still has to engage levelling — so the aircraft cannot fly differently
without a deliberate keypress. Both settings are written to `events.jsonl`.

Neither half has been flight-tested. Offline checks: `python3 pilot/test_shaping.py`
(31 checks) and `python3 pilot/test_imu_level.py` (19 checks, replays real recordings
against VQ1 truth pose).

### Input shaping (`--shaping`, `--slow-lap`)

A keyboard axis is 0 or 1: without shaping every input is full deflection, and the only
way to spell a small correction is to tap. Three knobs, all **feedforward** — nothing here
reads a measurement, so nothing here can oscillate.

| preset | expo | max rate | stick ramp |
|---|---|---|---|
| `full` (default) | 0 | uncapped | 12/s |
| `slow` | 0.55 | 1.4 rad/s (80°/s) | 5/s |

* **expo** softens the centre: `out = (1−e)x + e·x³`. On a keyboard the axis is only
  between 0 and 1 during the slew ramp, so this is what makes a *tap* gentle while a held
  key still reaches full rate. Half stick → 59% of linear.
* **max rate** is a hard clamp in rad/s applied **last**, on the way to the wire — so
  `LCTRL` boost, the align loop and the levelling assist are all inside it. A clamp rather
  than a scale on `ACRO_*`, so boost cannot defeat it and the recorded limit is one number
  rather than a product of four.
* **stick ramp** replaces `SLEW_PER_S` — on the three rate axes only. **The throttle axis
  keeps 12/s under every preset**, because the snap-to-hover release edge is timed off it
  (a 5/s ramp would stretch the pre-snap integration from ~80 ms/0.04 to ~200 ms/0.1), and
  the snap is measured but not yet flight-tested. Shaping has no business changing it as a
  side effect.

`full` is *arithmetically* identical to the pre-shaping code — expo 0 is an early return,
max rate 0 disables the clamp, and the ramp is unchanged. `test_shaping.py` asserts this
tick-by-tick against the old algebra written out separately, at all three boost/precision
scales, and separately asserts the throttle channel is byte-identical under both presets.

**The preset goes in `events.jsonl`**, twice: inside `session_start` and again as its own
`input_shaping` event. `cmd.csv` feeds a plant fit, and identical stick work under
different shaping produces a different command trace — a run flown under `slow` is a
different experiment. `--expo` / `--max-rate` / `--stick-slew` override individual values
and rename the preset to `<name>+custom`, so nothing is grouped with a preset it was not
flown under.

The `slow` numbers are starting points, not tuned in flight: 80°/s is far more than a lap
needs (the VQ1 slow lap was flown at a true median 2.9 m/s), and the cap is the part that
matters.

### `--imu-level` — the levelling assist, under VQ2

Added 2026-08-01, at Claire's request ("strafe/forward/back inputs slewing to correct, and
autostabilizing against gravity for hover if no input"). Both are the same feature: with
`F11` engaged the sticks command a bank *angle*, so `A`/`D` and `W`/`S` become strafe and
forward/back that recover to level on release.

`F11` needed VQ1's `ATTITUDE` stream, which VQ2 blocks. **Roll and pitch are observable
under VQ2 anyway**: gravity is a vector the accelerometer sees. `NOTES.md` already ruled
this in bounds when it rejected absolute yaw — *roll/pitch self-levelling gives no
information a permitted stream lacks; it is a quality gain, not a capability gain*. Yaw is
neither estimated nor levelled, here or anywhere.

The accelerometer alone is not enough, and `--imu-tilt-comp` is the evidence: it reads
gravity only when not accelerating, which rejects 40–78% of frames and rejects them
preferentially *when banked*. So this is a **complementary filter** — the gyro carries
attitude through the manoeuvre, the accelerometer trims the drift out when it is
trustworthy. `LEVEL_IMU_TAU = 2.0 s`, accel gate `||a| − g| ≤ 1.0`.

**Measured against VQ1 truth**, replaying recorded `imu.csv` through the shipped estimator
(`test_imu_level.py`; median / p90 of |error|, degrees):

| session | roll | pitch | what it was |
|---|---|---|---|
| `20260731-204841-vq1-lap-slow` | 1.34 / 3.67 | 1.37 / 4.10 | clean 6/6 slow lap — **the target profile** |
| `20260731-195307` | 0.68 / 3.18 | 1.15 / 4.98 | lap with resets |
| `20260731-131305` | 0.03 / 7.42 | 0.22 / 6.35 | long mixed flying |
| `20260731-150712` | 0.03 / 15.9 | 1.08 / 9.61 | thrust excursions, skids |
| `20260731-143025` | 3.53 / 29.5 | 3.89 / 11.8 | rate doublets — worst case by design |

**Accuracy is a function of how hard you fly, and that is the whole story.** On lap-like
flying it holds 1–2° median and better than 5° at p90; on sustained high rate it degrades
to p90 30°, because that is exactly when the accelerometer has nothing to say. It is
opt-in for that reason, and the flag's banner says so at startup.

The sign sweep that fell out of this settled `CONVENTIONS.md`'s last open accelerometer
question — **the accelerometer is canonical, not mirrored** — with VQ1 truth pose as an
outside referee. Table and caveats are in `CONVENTIONS.md`.

Why it is safe to close a loop on this when the thrust snap deliberately does not: there
*is* a valid measurement here, which is the exact thing altitude lacks. And the structure
matters — the gyro path is instantaneous and the accelerometer trim runs at τ = 2 s, far
slower than any airframe mode, which is what keeps a complementary filter from oscillating.
It is a P loop on that estimate, clamped at `LEVEL_MAX_RATE` and clamped again by the
shaping cap.

Bonus: under `--imu-level` the assist's **tilt compensation finally works on VQ2**, and it
is strictly better than `--imu-tilt-comp` — that one reads the raw accelerometer and goes
stale exactly when banked, while the estimator carries attitude through the bank on the gyro.

Nothing extra is recorded. `imu.csv` already holds every input, so any estimate this made
is reproducible offline, and `cmd.csv` keeps its existing columns.

### Speed and steadiness readouts

Two advisory numbers on the HUD and console line; nothing acts on either.

`v~` is an **airspeed estimate from drag** — VQ2 has no velocity telemetry at all, so this
is the only speed number that exists. Same algebra the align key uses for direction, read
for magnitude: thrust is along body −z, so the horizontal accelerometer components carry
only drag. Calibrated against `LOCAL_POSITION_NED` on VQ1, 35 203 in-flight samples:
`|(ax,ay)| = 0.041·v²` holds to ~10% from 2 to 8 m/s, and reads *low* above 10 m/s — i.e.
it under-reports exactly when you are already too fast, which is the safe direction. It is
blanked while disarmed because **the launch pad is inclined 17.8°**, so a parked drone
shows a rock-steady 3.00 m/s² that inverts to a fictional 8.6 m/s.

`steady` is seconds since the accelerometer last read gravity and nothing else
(`||a| − g| ≤ 0.35`) — i.e. since the last instant that could yield a gravity-referenced
gate height. `perception/NOTES.md` reports 4 of 5 labelled frames unusable for height for
exactly this reason, so it is worth seeing while flying rather than afterwards.

**Honest gap:** the amber threshold (6 m/s) comes from the course's cruise pace
(2.76 s/station × 15.97 m/station), *not* from a fit against station-read coverage. No
recording pairs a measured speed with a coverage figure — the 38%-coverage session's
`cmd.csv` never shows armed, and the 1% session is dominated by pad samples.

### Yaw on key release — checked, nothing to fix

Measured across 20 recorded sessions: after the yaw command returns to zero, `|zgyro|`
falls to ~0.04 rad/s within **0.1 s** and to 0.00 by 0.25 s. There is no residual yaw rate
and no drift — the flight controller closes the rate loop hard. **So no yaw damping was
added**, and adding any would have been a loop fighting a loop that already works. What
made yaw feel heavy is bang-bang input, which the shaping above addresses.

### Two keys read backwards — labels fixed, signs untouched

`Q`/`E` and `W`/`S` do the opposite of what the letters suggest, and always have
(`CONVENTIONS.md`, settled against the pilot). The printed `KEYMAP` and the source comment
used to say the wrong thing — `# w = nose down = forward` was simply false. Both now say
what the keys actually do. **The signs are deliberately not flipped**: that changes the
feel of every session ever recorded and wants doing on purpose, not hours before a lap.

Flies like an acro quad — releasing the keys is not a hover, and by default there is no
self-levelling (`F11` adds it, below). Throttle integrates while a key is down and
**snaps to the measured hover point when you let go** (below); `F10` panics back to hover,
`--hover` overrides the hover point (default `0.266`).

### Snap to hover (`THRUST_HOVER`, default on)

Added 2026-08-01. ACRO is the only reachable flight mode (`NOTES.md` → Gotchas), so the
thrust channel is fully manual — and it has exactly one correct setting, which makes
holding altitude down a 17-gate course pure hand-work. Releasing `UP`/`DOWN` now sets
thrust to that setting. Holding a throttle key behaves exactly as before; the offline
check compares the held-key ramp against the pre-snap algebra tick by tick.

**This is not an altitude hold, and deliberately not a loop.** Nothing in VQ2 observes
height or vertical speed — position and velocity are blocked, and double-integrating the
accelerometer drifts — so there is no measurement for a feedback controller to close on,
and a loop with no valid measurement oscillates. A constant cannot. Vertical drift is
still the pilot's to trim.

`THRUST_HOVER = 0.266`, measured, provenance in the constant's comment. An IMU sample is
a hover sample when `||a| − g| ≤ 0.35` (not accelerating) and gravity lies along body −z
(not banked); the thrust commanded alongside it is the hover point. Claire's session
`20260731-222724` gave 0.266 from 221 samples. `hovercheck.py` re-runs the measurement
over every session and adds two filters the first pass lacked — **armed**, and
`thrust ≥ 0.05`, because *a drone sitting on the pad also reads steady and level* at
whatever thrust the stick is at. Pooled n=17329, median **0.265**; the four longest
sessions give 0.265 / 0.265 / 0.267 / 0.272. So 0.266 is confirmed, and the honest spread
is 0.265–0.272 rather than the original p10 0.20 — that low tail was pad samples.

Timing detail: the throttle *axis* is slew-limited at `SLEW_PER_S = 12/s`, so a physical
key release takes ~80 ms to fall through `THRUST_STICK_EPS`. Thrust integrates for that
80 ms (≤ 0.04) and then snaps. Inherited from the stick path, not added by the snap.

`--no-thrust-snap` restores the old hold-where-you-leave-it throttle — use it if you are
reproducing the conditions the acro system-ID recordings were flown under. Each release
edge writes a `thrust_snap` event to `events.jsonl`, so a later reader can line the snaps
up against `cmd.csv`.

#### `--imu-tilt-comp` — experimental, **off**

Hover thrust ought to rise as `hover / cos(tilt)`, and under VQ2 the accelerometer is the
only thing that sees tilt (`cos(tilt) = −a_z/|a|`, valid only when not accelerating).
Implemented, low-passed at τ = 0.5 s, and gated to samples with `||a| − g| ≤ 0.5`; a
rejected or stale (> 0.5 s) sample decays the factor back to 1.0, i.e. to the plain flat
snap, so every failure mode lands on the behaviour we would have had without it.

It defaults off because the measurement is missing exactly when it is needed. Measured on
the three longest sessions: the gate accepts **45% / 60% / 22%** of frames, with gaps of
up to **5.0 s** between accepted samples — and among accepted *in-flight* samples the
implied tilt has median **2.0°**, p90 **4.5–5.8°**. The gate passes near-level frames
almost exclusively, so the compensation computes ≈1.00 nearly always and is stale
whenever a real bank is on. Shipped only because it is cheap to A/B in flight; those
numbers are what it has to beat. The `F11` assist's tilt compensation is unaffected — it
reads *truth* attitude, which is a real measurement, and is VQ1-only for that reason.

Offline checks: `python3 pilot/test_thrust_snap.py` (21 checks, no sim or keyboard
required). It cannot test aircraft response — that needs a flight.

`W/S` pitch, `A/D` roll, `Q/E` yaw, `UP/DOWN` throttle, `LCTRL` boost, `LALT` precision,
`C` align-to-velocity. Commands on F-keys: `F5` arm, `F6` disarm, `F7` zero heading, `F8`
quit, `F9` reset, `F11` levelling assist, `F12` marker.

**Two keys do the opposite of what they read like**, because the sim's rate convention is
mirrored (`NOTES.md` → Sign conventions; `camreferee.py` re-runs the measurement):

| key | sends | actually does | intuitive? |
|---|---|---|---|
| `D` | −2.5 | rolls RIGHT | yes |
| `W` | −2.5 | pitches **UP** — flies BACKWARD | **no** |
| `E` | +2.0 | yaws **LEFT** | **no** |

The comment `w = nose down = forward` in `teleop.py` said the opposite; corrected
2026-08-01, along with the `KEYMAP` block the script prints at startup, which had the same
error. Under the interface rule that signs live in the link layer only, `ACRO_PITCH` and
`ACRO_YAW` want flipping so `W` is nose-down and `E` is yaw-right; `ACRO_ROLL` is already
correct. **Still not done** — it changes the feel of every recorded session, so it wants
doing deliberately rather than mid-batch, and hours before a mapping lap is not that.

### Levelling assist (`F11`) — our own angle mode

Added 2026-07-31 because the sim's ANGLE mode is unreachable (`NOTES.md` → Gotchas) and
hand-flying a clean acro lap is hard. The stick commands a bank *angle*;
an outer P loop turns the angle error into the body rate that goes out the ordinary acro
path. Release the sticks and it returns to level instead of holding attitude.

    LEVEL_MAX_ANGLE 35 deg    LEVEL_GAIN 4.0 (rad/s per rad)    LEVEL_MAX_RATE 3.0 rad/s

**Roll/pitch levelling confirmed working in flight (Claire, 2026-07-31).**

Two throttle changes rode in with the assist. Tilt compensation is still assist-only;
return-to-hover has since been generalised to plain acro:

* **Tilt compensation.** Thrust acts along body −z, so at tilt θ you need
  `hover / cos(θ)` to hold altitude. Without it hover sags every time you turn, which is
  most of what makes altitude hard to hold through a lap. Uses the *achieved* attitude, so
  it compensates the bank you are at rather than the one you asked for. Clamped at 1.6×
  (~51°) because 1/cos runs away near 90°. As multipliers on the hover point:
  ×1.04 at 15°, ×1.15 at 30°, ×1.22 at 35°, ×1.30 at 35° roll + 20° pitch.
* **Return to hover.** Throttle keys slew away from the compensated hover point and it
  snaps back the moment you release, so throttle is an offset rather than an absolute.
  (As of 2026-08-01 this is no longer assist-only: plain acro snaps back too, to the
  *uncompensated* hover point — see "Snap to hover" above. The assist still owns the
  tilt compensation, because only it has a real attitude measurement.)
  An earlier version eased back over τ = 0.7 s; flight test says the step is not felt,
  because thrust reaches velocity through mass and drag, which is already a first-order
  lag — the airframe supplies the smoothing and a filter here would just add a second lag
  in series. It is also the better choice for the data: a step excites the plant, whereas
  a pre-smoothed command shares its shape with the response, which is precisely what makes
  a command→thrust lag hard to bracket (`plantfit.py` failed on exactly that).

Neither observes altitude. **This is not an altitude hold** — letting go returns you to
hover *thrust*, not to a hover, and vertical drift is still yours to trim. Vertical speed
is not measured anywhere: VQ2 blocks position and velocity, and double-integrating the
accelerometer drifts.

Four properties that made this preferable to getting angle mode from the sim, even if the
sim would have given it:

* **`cmd.csv` still records real body-rate commands.** Nothing about the recorded data
  changes, so cmd → response system ID needs no reconstruction. An FC-side angle mode
  would have hidden the rate setpoint inside the flight controller where we cannot see it.
* **No quaternion is transmitted; `ATTITUDE_IGNORE` stays set.** Absolute yaw never enters
  the protocol. Yaw is untouched by the assist and stays pure acro — levelling yaw would
  mean holding a heading, and holding a heading means knowing one.
* **It self-gates to VQ1** *(as originally shipped)*. The outer loop needs truth attitude,
  which VQ2 does not send, so under VQ2 `F11` prints why it did nothing.
  **Superseded 2026-08-01 by `--imu-level`**, which supplies an attitude VQ2 *does* permit
  — see the `--imu-level` section above. The gate is now the flag, and it is off by
  default, so a run without it behaves exactly as this bullet describes. What has not
  changed: this is teleop, which is not the pilot, and `interface.Policy` cannot see it.
* **Data collection only.** It closes a loop around ground truth, which the raced pilot may
  never do. It lives in teleop, which is not the pilot; `interface.Policy` cannot see it.

The truth attitude it consumes carries the per-axis sign correction (`roll` from ATTITUDE,
`pitch` = −ATTITUDE.pitch), and the physical→command conversion reuses `ACRO_ROLL`'s sign
and the explicit minus on the pitch line rather than writing the mirror out again.

Offline checks behind it: signs agree with the stick path, and the closed loop settles on
target from +30°→0, 0→+20°, +10°→−25° under the measured mirror. That checks the algebra
*given* the convention; it cannot re-check the convention, which `camreferee.py` settled.
Gains have not been tuned against anything — levelling flies, but nobody has looked for a
better `LEVEL_GAIN` than the first guess.

**If you fit a model to assisted flight:** commands are now generated from the state by
this controller, so command and state are correlated through it, which biases an open-loop
fit in a way that looks clean. Stick input still offsets the target and so does excite the
loop, but prefer the acro doublet sessions for identifying the rate loop itself.

**Commands live on F-keys because the sim eats keystrokes.** The global hook is not
exclusive — every keystroke reaches teleop *and* the sim, which binds ordinary game keys
(SPACE restarts the run; found the hard way). Continuous axes sit on letters/arrows where an
echo is cosmetic. `suppress=True` was considered and not shipped: unverifiable here, and a
control scheme that fails silently is worse than one that collides visibly.

HUD `hdg` is **diagnostic only** — the unbounded integral of `zgyro`; nothing in the control
path reads it.

Writes `pilot/sessions/<timestamp>/`: `imu.csv`, `race.csv`, `cmd.csv`, `frames.csv`,
`actuators.csv`, `collisions.csv`, `events.jsonl`, `frames/*.jpg` (raw), plus
`attitude.csv` / `position.csv` / `odometry.csv` whenever those arrive — VQ1 only, and the
HUD shows `[TRUTH]` when they do. `cmd.csv` pairs input with response for system ID.