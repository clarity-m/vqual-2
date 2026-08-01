## teleop.py

    python3 pilot/teleop.py                 # fly + record + live view
    python3 pilot/teleop.py --listen        # record only, sends nothing (safe mid-flight attach)
    python3 pilot/teleop.py --no-view       # no cv2 window, lower loop jitter
    python3 pilot/teleop.py --no-thrust-snap  # old throttle, holds where you leave it

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

So the comment `w = nose down = forward` in `teleop.py` is wrong. Under the interface rule
that signs live in the link layer only, `ACRO_PITCH` and `ACRO_YAW` want flipping so `W` is
nose-down and `E` is yaw-right; `ACRO_ROLL` is already correct. Not yet done — it changes
the feel of every recorded session, so it wants doing deliberately rather than mid-batch.

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
* **It self-gates to VQ1.** The outer loop needs truth attitude, which VQ2 does not send,
  so under VQ2 `F11` prints why it did nothing. It cannot leak into a race.
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