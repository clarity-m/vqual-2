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
   observation, not images. All three parts exist (`surrogate/`, self-check passing), but
   the course generator and the detection noise model are *specified* rather than
   *measured* — see "What is still open on the surrogate".

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
    python3 pilot/control/sysid_apex.py                            # card 2 direct reads
    python3 pilot/control/sysid_fit.py                             # -> plant.json
    python3 pilot/control/sysid_replay.py                          # the gate
    python3 pilot/control/plant.py                                 # model self-check

`plant.json` is the result and `plant.py` is the model plus a steppable NumPy `Sim`.
`surrogate/` wraps it with the course generator and the synthetic detection producer;
`surrogate/selfcheck.py` passes 15/15 against this fit, reading hover 0.270 and a 30.99
m/s terminal climb straight out of the new thrust table.

**Refitted 2026-08-01 on card 2** (`pilot/SYSID.md`). Card 2's maneuvers A and B were
flown, and they retired the two things this section used to list as open — the hand
clamp on the low-throttle thrust curve and the jointly-fitted `kz`.

| | fitted | in sample | held out |
|---|---|---|---|
| body-x drag | `k = 0.0487` | R² 0.9975 | R² 0.9936 |
| body-y drag | `k = 0.0496` | R² 0.9974 | R² 0.9945 |
| body-z drag | `kz = 0.0436`, from the apex arcs alone | | |
| body-z thrust | 21-knot measured table, hover 0.270, full 51.7 m/s² | R² 0.9311 | R² 0.9023 |
| rate loop | gain 0.970 / 0.963 / 0.904, delay 10 ms, τ < 10 ms | R² 0.98–1.00 | |

Held out is `20260731-131305`, nine minutes of ordinary flying the fit never saw. Open-loop
replay of recorded commands drifts 2.1 m over a 5 s window in which the drone travelled
35 m — down from 2.6 m before the refit — and every sign corruption is caught by 14x or
more, with the drag controls at 2–12x.

### What card 2 changed, and why it is not just a tweak

**`kz` was 20% low and the old number was not identified.** The previous `plant.json`
shipped 0.03639 against the 0.04359 measured here. In racing flight throttle and `w|w|`
correlate at −0.90, so a fit that estimates `kz` and the thrust curve together returns
whatever that collinearity produces — bin the same data by `|w|` and the joint `kz` reads
0.091 where there is no lever arm and 0.035 where there is. The apex arcs park the
throttle and coast through `w = 0`, which makes `kz` a slope with thrust as a free
intercept. Four arcs agree to **0.0436 ± 0.0002**, and the two terminal descents — a
different maneuver, with thrust identically zero — say 0.0435 and 0.0436. Six reads, four
figures. *(`sysid_fit.py` prints −25% rather than −20%, because it re-runs the superseded
joint fit on the enlarged card-2 fit set and gets 0.03259; the 20% is against what
actually shipped.)*

Two caveats on those six reads. Only **four of the eight apex arcs** are used: the
throttle 0.15 and 0.20 reps coasted through less than 9 m/s of `w`, and with that little
lever arm they return `kz` of 0.06–0.10. `sysid_apex.py` excludes them on the span, not
on the answer, which is the only ordering that is not circular. And neither terminal
descent actually plateaued — both were still accelerating at +1.7 m/s over the last
second — so those two are fitted slopes, not steady reads.

**The thrust curve is a table because the flight is not a polynomial.** Wherever `|w|` is
small the drag term is under 0.2 m/s², so `-a_z` *is* thrust; that turns 12 158 of the
23 261 fit samples into direct reads spanning throttle 0.00 to 1.00. What they show:
a deadband below 0.05 (0.35 m/s² of idle thrust at zero), a rise steepening to ~76 m/s²
per unit throttle near 0.6, then flattening to 51.7 at full. An affine curve puts hover at
0.249 and misses a direct read by 2.7 m/s²; a quadratic manages 0.252 and 4.0. The table
puts hover at **0.270**, against the ~0.27 measured in flight, and honours every direct
read to 0.05 m/s².

**What the old clamp cost.** The affine curve went negative below throttle 0.10 and was
clipped to zero, so the model predicted a free fall at throttle 0.10 where the drone
actually holds 2.3 m/s² — a 23%-of-gravity error in exactly the regime a policy enters
every time it chops throttle into a descent.

**One thing the refit opened rather than closed.** With the thrust curve pinned
independently, `kz` can be refitted on the recordings as a genuine second opinion, and it
comes back **0.0443 at `|u| < 2` falling to 0.0412 at `|u|` 10–15 m/s**. So the vertical
drag coefficient really does depend on forward speed by about 7%, which the component-wise
model has no term for. The arcs measure the `|u| ≈ 0` end and that is what `plant.json`
carries; a compromise at 0.0420 would buy +0.017 on held-out body z at the cost of
contradicting the direct measurement. `sysid_fit.py` prints both every run.

Two cross-checks that fell out of it, both worth more than the coefficients:

* **Hover throttle from the fitted curve is 0.270**, against ~0.27 measured in flight and
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
  1.45 m/s² against 0.007–0.22 for every other epoch. Excluded from the fit; the cause is
  a stalled velocity stream rather than anything about the flight (see *Data status*).

### What is still open on the plant

* **`kz` depends on forward speed and the model has no term for it** — 0.0443 near
  vertical, 0.0412 at `|u|` 10–15 m/s. Of the candidate extra terms, a body-lift term
  `c·u²` is the one the data now supports: with the measured thrust curve in place it
  buys **+0.026 on held-out body z** over refitting `kz` alone (0.9197 → 0.9456), and
  **+0.043 over what `plant.json` ships** (0.9023). The alternatives do not come close —
  `c·u|u|` is −0.015 and a `c·|u|w` cross term is +0.014. It is not in `plant.json`: it
  was measured after the refit was closed and has not been through the replay gate.
  Note the tension it exposes — the shipped `kz` of 0.04359 is the arcs' `|u| ≈ 0`
  measurement and is the *highest* of the three estimates, so on ordinary flying the
  model over-predicts vertical drag; the pooled recordings want 0.04189 and the held-out
  session alone wants 0.03945. A lift term explains that spread; retuning `kz` only hides
  it.
* **No clean completed lap** for end-to-end validation. Card 2's maneuver C was attempted
  and did not finish; `race_finish_time_ns` is still −1 in every session ever recorded and
  the furthest any run has reached is `active_gate_index` 5. The held-out session is
  ordinary flying, which is the next best thing.
* **The rate-loop time constant is only bounded** (τ < 10 ms). At 61 Hz the inner loop
  looks like a pure delay; identifying τ properly needs a faster response measurement than
  the IMU can give, and probably is not worth it — the loop tracks at gain 0.90–0.97 with
  R² 0.98–1.00 in every epoch that commanded rotation, so there is very little left to
  model. Yaw is the low one at 0.904; roll and pitch are 0.97.
* **The table's 0.50–0.82 stretch rests on four bins totalling 240 samples.** Both ends
  are anchored hard — hover by 6928 samples, full throttle by 250 steady ones — but the
  middle-upper range is thin, and it is where a policy accelerates. Another terminal run
  held at 0.6 and 0.8 would fix it in thirty seconds of flying.

### What is still open on the surrogate

The plant half is now measured. The other two halves are not, and both are measurable
from data already on disk — no further piloting is required for either.

* **Detection statistics are guessed, and the one number that has been checked was
  wrong in the dangerous direction.** `noise.py` is the P3 fallback: every range in it is
  a reasoned guess. A feasibility probe over `20260731-204841-vq1-lap-slow` — project the
  measured gate centres into all 2937 recorded frames using truth pose and the
  `camera.py` model, then run the `NOTES.md` orange HSV mask — puts a blob where geometry
  says one should be on **95%** of in-frustum (gate, frame) pairs, and **95% at 30–45 m**,
  where `max_range_m = 14–30` says the gate should already be gone. Training against a
  world where gates vanish at 20 m teaches search behaviour the real course does not
  require. The same probe found the blob runs ~2.1x wider than `480/range` predicts,
  because the orange mask fits the **2700 mm outer frame** and the spec rangefinder
  assumes the **1500 mm inner aperture** — a size-only range read off the mask without
  fitting the hole is biased short by nearly half.
* **The course generator's shape priors come from NOTES.md arithmetic, not from a
  course.** Two sessions flew the whole 6-gate map and their crossing points agree to
  0.5–1.2 m, so its geometry is measured: segments 23.5/24.4/29.2/39.4/23.9 m (mean 28.1,
  against the generator's 28 drawn over 22–34, so the long segment is off the top of the
  range), turns +5.7/−12.1/+15.6/−19.4° — **alternating, not persistent** — and a
  **monotone 24 m descent over 140 m of path**. That last one is a structural miss: the
  generator's elevation is mean-reverting on altitude inside a ceiling band, so it cannot
  produce a sustained one-way descent, which is exactly the case `CONVENTIONS.md` calls
  the binding camera constraint. Caveat: this is VQ1's map, and VQ2's winds more, so it
  calibrates the *vertical* prior and the segment range but must not be used to pull the
  turn magnitudes down.
* **There is no fixed course to compare surrogate and live runs on.** Every episode is
  generated and thrown away, which is right for training and useless for validation: a
  lap time on the surrogate cannot be compared with a lap time on the sim. The 6-gate map
  above is the only course that can actually be flown live, so landing it as a fixed
  evaluation course is what closes that loop.

## Data status

Card 1, 2026-07-31: `20260731-143025` (rate doublets), `20260731-144815` (isolated
single-axis taps, with frames), `20260731-150712` (thrust excursions, cruise ladder,
skids, ~5 min, 9099 frames).

Card 2, 2026-08-01: `20260801-004843` (**A**, eight apex arcs at throttle 0.05/0.10/
0.15/0.20, two reps each — the cleanest epoch in the dataset at a referee median of
0.011 m/s²) and `20260801-005059` / `20260801-005518` (**B**, vertical terminal runs;
the same open-loop script twice, agreeing to 0.1 m/s, so the second is a repeatability
check and only the first is in the fit).

Coverage is now thrust command 0.00–1.00 and body speed to 31 m/s vertical / 34.3 m/s
horizontal, with 2586 genuinely steady samples and real sideslip. `pilot/SYSID.md`
describes what each segment was for.

`20260731-131305` is the **designated hold-out** — unchanged across both fits, so they
are comparable — and `sysid_fit.py` records that split in `plant.json` so the replay
cannot quietly validate on its own training data.

**Run the frame referee on everything, never on card 2 alone.** Its margin depends on
attitude variety, and card 2 is deliberately vertical: all sessions pooled scores 2.6x
(PASS), card 1 alone 3.2x, and the card 2 sessions on their own 1.0x — a FAIL that says
"this data cannot tell the conventions apart", not "the convention is wrong". The winning
signs are identical in all three. A narrow subset will fail the same way for the same
harmless reason, so pass it the whole corpus.

**Still missing: a clean completed lap.** Card 2's maneuver C was flown and did not
finish, so generalisation is still tested against nine minutes of ordinary flying.

**Two sessions are excluded and the reason is now known.** `20260731-130744` (referee
median 1.45 m/s²) and `20260731-233219` (2.01) share one property: `LOCAL_POSITION_NED`
velocity repeats bit-identically in 53% and 43% of its rows while the drone maneuvers
hard, so the referee is differentiating a held signal. The earlier theory — that the
morning sessions' half-rate IMU was to blame — does not survive `233219`, which ran at
15.9 Hz and scores *worse* than the 31.8 Hz session it was supposed to explain. It is a
recording artefact, not a fact about the simulator, and re-flying reproduces it rather
than resolving it.

## Do not start from vqual-1's `fly.py`

1535 lines of world-frame velocity setpoints steered by `ATTITUDE` yaw. VQ2 blocks both.
It completed the R1 course and is genuinely dead here — the actuation basis differs all the
way down. `link.py` (MAVLink plumbing) does transfer.
