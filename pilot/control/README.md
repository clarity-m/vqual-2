# control/ — Alex (kongalex@umich.edu)

Everything **below** the measurement vector: plant fit, surrogate, control policy.

Read `../interface.py` first. It is the frozen contract and is owned by neither side —
propose changes, don't edit unilaterally.

## What this half owns

**The deliverable is a policy** — `interface.Policy`, beating the baseline. PID, MPC or
learned; nothing above the interface cares which.

The other two are **means, and optional**:

1. **Plant fit** — drag, thrust curve, rate-loop response, from `cmd.csv` → `imu.csv`.
   **Done, 2026-07-31** — see below. (Mass is not in the list: an accelerometer measures
   specific force, so mass is not identifiable from this data and nothing needs it.)
2. **Surrogate** — a steppable NumPy sim: fitted plant + course map + *synthetic
   detections*. It renders no pixels and does not need to; the policy consumes the 73-D
   observation, not images. The plant half exists (`plant.Sim`); the course and the
   detection noise model do not.

They exist for one reason: live sim runs are ~15–25 per session, shared and serialized
behind a single exclusive UDP port, so tuning against the real thing is rate-limited. A
surrogate buys iterations. If a policy flies without one — reactive control off the
guidance ribbon, hand-tuned gains, whatever works — that is a win, not a shortcut.

## READ THIS FIRST: the ground truth is sign-corrupted

The VQ1 truth streams are each wrong on a **different** axis. Refereed against gravity on
a parked drone (`|a|` = 9.8100, genuinely at rest), magnitudes agreeing to five decimals:

    truth_roll  =  ATTITUDE.roll      # == -ODOMETRY.roll
    truth_pitch =  ODOMETRY.pitch     # == -ATTITUDE.pitch
    truth_yaw   = -ATTITUDE.yaw

**Fit against raw `ODOMETRY.roll` or raw `ATTITUDE.pitch` and your model is mirrored** —
and mirrored silently, which is the exact failure this README warns about below, arriving
through the data rather than through the code.

All three lines, **including the yaw one, are now re-derived rather than assumed**
(`sysid_frames.py`). The yaw sign could not be settled from a parked recording — both
streams sit at the degenerate −179.9° heading — and it is settled here instead by scoring
all sixteen candidate conventions against `R_wb·a_body + g == dv_world/dt` over the full
−180..+180° range of headings: `−ATTITUDE.yaw` wins by 6x on the residual tail.

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

*(`sysid_replay.py` is that check, and it earns its keep: a per-axis force fit at R² 0.99
barely notices a mirrored rotation, while the trajectory it implies bends 80° the other
way. Note also what the replay found — a **yaw** sign error moves the drone not at all and
only shows in attitude, so a position-only gate would have passed it.)*

**The noise model is a transfer risk too.** A policy will exploit any regularity in
synthetic detections. Dropout, latency and false-positive statistics must be *measured off
real frames* and randomised over their uncertainty. This is the point where a good ML
instinct and a good robotics instinct disagree; robotics wins.

*(An earlier draft said the noise model was the risk and the plant model was not. That
understated the plant — see below. Budget for both.)*

## The plant is fitted — 2026-07-31

`sysid_*.py` in this directory, in order. Each stage prints PASS or FAIL and refuses to
be useful without the one before it:

    python3 pilot/control/sysid_frames.py  pilot/sessions/2026*/   # settle the frame
    python3 pilot/control/sysid_latency.py pilot/sessions/2026*/   # bracket the lags
    python3 pilot/control/sysid_fit.py                             # -> plant.json
    python3 pilot/control/sysid_replay.py                          # the gate
    python3 pilot/control/plant.py                                 # model self-check

`plant.json` is the result and `plant.py` is the model plus a steppable NumPy `Sim` —
that is the plant half of the surrogate; the course map and synthetic detections are not
built yet.

| | fitted | in sample | held out |
|---|---|---|---|
| body-x drag | `k = 0.0487` | R² 0.9985 | R² 0.9929 |
| body-y drag | `k = 0.0496` | R² 0.9989 | R² 0.9942 |
| body-z thrust + drag | `T/m = -6.08 + 59.59·throttle`, `kz = 0.0364` | R² 0.9205 | R² 0.9158 |
| rate loop | gain 0.970 / 0.963 / 0.904, delay 10 ms, τ < 10 ms | R² 0.98–1.00 | |

Held out is `20260731-131305`, nine minutes of ordinary flying the fit never saw. Open-loop
replay of recorded commands drifts 2.6 m over a 5 s window in which the drone travelled
35 m, and every sign corruption of the model is caught by 17x or more.

Two cross-checks that fell out of it, both worth more than the coefficients:

* **Hover throttle from the fitted curve is 0.267**, against ~0.27 measured in flight and
  recorded in `NOTES.md`. Independent route, same answer.
* **The rate mirror is exactly −1.000 on all three axes**, re-derived from truth attitude
  rather than inherited from `camreferee.py`, which had settled the sign but could not
  settle the magnitude.

### Why the first cut got R² 0.05

One missing term, and the regressor was never the problem. `-a_z` is not thrust; it is
thrust **plus vertical drag**, and in these recordings the two nearly cancel — throttle
goes up exactly when the drone is moving fast along body z. Mean `|w|` in the top throttle
bin is 29 m/s against 1 m/s at hover. So `-a_z` against throttle measures the difference
of two large numbers that track each other, which is why the command and the motor sum
returned the same R² 0.05: **R² 0.0435 without the drag term, 0.9205 with it**, same data,
same regressor.

### The dead ends, and what each one turned out to be

`plantfit.py` is kept as the record. Every negative result in it reproduces. Of the
diagnoses attached to them, one was a red herring (the lag), one was right and was half the
answer (drag belongs in body axes), and the one nobody wrote down is the other half: the
same vertical drag also sits in the *thrust* axis, where it had no business being ignored.

* **Steady-state thrust identification is malformed.** Confirmed — at constant velocity
  `thrust/m == g` whatever was commanded, so steady data can only reveal hover trim. The
  fix is dynamic data, not more steady data.
* `thrust/m` from `-zacc` vs command, **R² 0.051**; vs `ACTUATOR_OUTPUT_STATUS` motor sum,
  **R² 0.048**. Both reproduced. The motor sum is indeed the physically correct regressor
  and it correlates with the command at 0.9998, so it was never going to help: the missing
  variable was on the *response* side.
* The command → thrust lag **peaked at the window edge**. Widened to ±0.6 s: `thrust cmd →
  motor` peaks at **+15 ms** and `rate cmd → gyro` at **+10 ms**, both interior, both
  positive. But `motor → -a_z` has **no peak at any lag** — that path is not a latency
  problem at all, it is the missing drag term, and no realignment could have found it.
* Drag on `|v_world|` gives **R² 0.11**. Confirmed, and the diagnosis was exactly right:
  in body axes the same data gives **R² 0.999** per axis. It is component-wise quadratic
  (`-k·v_i|v_i|`); the isotropic `-k·|v|·v_i` form gives 0.80, so this is a fact about
  this simulator rather than aerodynamics.

### Three things about the recordings that were not known before

Each of these silently corrupts a fit, and each is handled in `sysid_data.py`:

* **The simulator's boot clock restarts at every `SIM_RESET`, mid-session.** Six times in
  `20260731-150712`, four in `20260731-131305`. Wall time keeps running, so a session read
  on the wall clock looks continuous while splicing together separate runs across a
  teleport back to the pad — and anything differentiated across that seam invents a
  several-hundred-m/s² sample. A session is now loaded as a list of continuous epochs.
* **`cmd.csv`'s `armed` column is not usable on the VQ1 build.** It carries the HEARTBEAT
  `SAFETY_ARMED` flag, and epoch 0 of `20260731-150712` reaches 34.3 m/s and throttle 0.86
  with `armed` never once true. Filtering on it discards the widest-envelope data there is.
* **`20260731-130744` does not reconcile.** Its median kinematic-referee residual is
  1.45 m/s² against 0.007–0.22 for every other epoch. Not understood; excluded from the
  fit rather than diagnosed under deadline.

### What is still open on the plant

* **The thrust curve is affine and goes negative below throttle 0.10.** It is the best fit
  over the flown range (0.08–0.86) and it beats a zero-intercept quadratic on held-out
  data, so it stays — but `plant.thrust()` clamps at zero, because a surrogate that lets a
  policy push *down* harder than gravity would teach a policy to exploit a fit artefact.
  Low-throttle data would settle it.
* **No clean completed lap** for end-to-end validation. The held-out session is ordinary
  flying, which is the next best thing.
* **The rate-loop time constant is only bounded** (τ < 10 ms). At 61 Hz the inner loop
  looks like a pure delay; identifying τ properly needs a faster response measurement than
  the IMU can give, and probably is not worth it — the loop tracks at gain 0.90–0.97 with
  R² 0.98–1.00 in every epoch that commanded rotation, so there is very little left to
  model. Yaw is the low one at 0.904; roll and pitch are 0.97.
* **The vertical drag coefficient is 0.036 against 0.049 horizontal.** Fitted jointly with
  thrust, so it is the one drag number that had a thrust parameter to trade against.

## Data status

Three sessions with ground truth (2026-07-31): `20260731-143025` (rate doublets),
`20260731-144815` (isolated single-axis taps, with frames), `20260731-150712` (thrust
excursions, cruise ladder, skids, ~5 min, 9099 frames).

Coverage: thrust command 0.00–0.86, horizontal speed to 34.3 m/s, 2586 genuinely steady
samples, real sideslip. `pilot/SYSID.md` describes what each segment was for.

`20260731-131305` is now the **designated hold-out** — the fit has never seen it, and
`sysid_fit.py` records that split in `plant.json` so the replay cannot quietly validate on
its own training data.

**Still missing: a clean completed lap.** No lap has survived to the finish yet, so
generalisation is tested against nine minutes of ordinary flying instead of against a lap.

## Do not start from vqual-1's `fly.py`

1535 lines of world-frame velocity setpoints steered by `ATTITUDE` yaw. VQ2 blocks both.
It completed the R1 course and is genuinely dead here — the actuation basis differs all the
way down. `link.py` (MAVLink plumbing) does transfer.
