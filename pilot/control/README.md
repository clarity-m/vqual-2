# control/ — Alex (kongalex@umich.edu)

Everything **below** the measurement vector: plant fit, surrogate, control policy.

Read `../interface.py` first. It is the frozen contract and is owned by neither side —
propose changes, don't edit unilaterally.

## What this half owns

**The deliverable is a policy** — `interface.Policy`, beating the baseline. PID, MPC or
learned; nothing above the interface cares which.

The other two are **means, and optional**:

1. **Plant fit** — mass, drag, thrust curve, rate-loop response, from `cmd.csv` → `imu.csv`.
   vqual-1's recordings are valid data: the spec diff shows VQ1 and VQ2 physics are
   identical, only telemetry changed.
2. **Surrogate** — a steppable NumPy sim: fitted plant + course map + *synthetic
   detections*. It renders no pixels and does not need to; the policy consumes the 73-D
   observation, not images.

They exist for one reason: live sim runs are ~15–25 per session, shared and serialized
behind a single exclusive UDP port, so tuning against the real thing is rate-limited. A
surrogate buys iterations. If a policy flies without one — reactive control off the
guidance ribbon, hand-tuned gains, whatever works — that is a win, not a shortcut. Note
also that the first-cut fit did not converge (see the dead ends below), so the model route
is not a solved subproblem with a known cost.

## READ THIS FIRST: the ground truth is sign-corrupted

The VQ1 truth streams are each wrong on a **different** axis. Refereed against gravity on
a parked drone (`|a|` = 9.8100, genuinely at rest), magnitudes agreeing to five decimals:

    truth_roll  =  ATTITUDE.roll      # == -ODOMETRY.roll
    truth_pitch =  ODOMETRY.pitch     # == -ATTITUDE.pitch
    truth_yaw   = -ATTITUDE.yaw

**Fit against raw `ODOMETRY.roll` or raw `ATTITUDE.pitch` and your model is mirrored** —
and mirrored silently, which is the exact failure this README warns about below, arriving
through the data rather than through the code.

Separately: **the simulator's entire body-rate convention is mirrored versus MAVLink NED,
commands and gyro alike** (`SIGN_ROLL = SIGN_PITCH = SIGN_YAW = -1`, settled against the
camera — `pilot/camreferee.py`). A negative commanded rate produces a positive NED
rotation, and the gyro reports the same mirrored sign as the command.

That last detail is the trap: commanded-rate versus measured-gyro correlates at **+0.96**
and looks like confirmation. It is a closed loop inside one convention and cannot see a
mirror applied to both ends. Only a referee outside the convention settles it.

## Two things that will cost you a week if skipped

**If you fit a plant, validate it before tuning anything on it.** Replay recorded commands through the
fitted model, compare predicted vs. recorded IMU. This is the *only* place in the project
where a world frame appears, so it is the only place a sign error is silent rather than
self-announcing — and a wrong sign yields a model that fits, looks sensible, and quietly
mistunes every gain trained against it. It then presents as "great on the surrogate, bad in
the sim", which reads like a sim-to-real gap and sends you off fixing the noise model.

**The noise model is a transfer risk too.** A policy will exploit any regularity in
synthetic detections. Dropout, latency and false-positive statistics must be *measured off
real frames* and randomised over their uncertainty. This is the point where a good ML
instinct and a good robotics instinct disagree; robotics wins.

*(An earlier draft said the noise model was the risk and the plant model was not. That
understated the plant — see below. Budget for both.)*

## What has already been tried, and failed

`plantfit.py` in this directory is a **map of dead ends, not a starting point**. Claude
overstepped the interface boundary and had a go; the negative results are the useful part:

* **Steady-state thrust identification is malformed, not merely noisy.** At constant
  velocity thrust must balance gravity, so `thrust/m == g` whatever was commanded. Steady
  data reveals the hover command and nothing else about the curve.
* `thrust/m` from `-zacc` directly, no steady-state assumption: **R² 0.051** vs command.
* Against `ACTUATOR_OUTPUT_STATUS` motor sum — the physically correct regressor, 95 Hz,
  permitted under VQ2: **R² 0.048**. No better than the command.
* The command → thrust lag cross-correlation **peaked at the edge of the search window**
  (−15 samples). The delay is not bracketed. Widen it before modelling the actuator.
* Drag binned by speed is cleanly monotonic (1.31, 3.35, 4.46, 4.06, 4.98 m/s²) but the
  power-law fit gives **R² 0.11**, and forced `v²` is worse than the mean. The diagnosis is
  clear: drag depends on velocity in the **body** frame, and the recordings contain large
  deliberate sideslip. Regressing on `|v|` is the wrong variable — you need the rotation,
  which needs the corrected attitude above. That rotation has a built-in self-check:
  `R·(specific force) + g` must equal the measured `dv/dt`.

## Data status

Three sessions with ground truth (2026-07-31): `20260731-143025` (rate doublets),
`20260731-144815` (isolated single-axis taps, with frames), `20260731-150712` (thrust
excursions, cruise ladder, skids, ~5 min, 9099 frames).

Coverage: thrust command 0.00–0.86, horizontal speed to 34.3 m/s, 2586 genuinely steady
samples, real sideslip. `pilot/SYSID.md` describes what each segment was for.

**Missing: a clean completed lap for held-out validation.** Fit on the maneuvers, validate
on a lap the fit never saw. No lap has survived to the finish yet.

## Do not start from vqual-1's `fly.py`

1535 lines of world-frame velocity setpoints steered by `ATTITUDE` yaw. VQ2 blocks both.
It completed the R1 course and is genuinely dead here — the actuation basis differs all the
way down. `link.py` (MAVLink plumbing) does transfer.
