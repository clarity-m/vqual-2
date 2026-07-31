# control/ — Alex (kongalex@umich.edu)

Everything **below** the measurement vector: plant fit, surrogate, control policy.

Read `../interface.py` first. It is the frozen contract and is owned by neither side —
propose changes, don't edit unilaterally.

## What this half owns

1. **Plant fit** — mass, drag, thrust curve, rate-loop response, from `cmd.csv` → `imu.csv`.
   vqual-1's recordings are valid data: the spec diff shows VQ1 and VQ2 physics are
   identical, only telemetry changed.
2. **Surrogate** — a steppable NumPy sim: fitted plant + course map + *synthetic
   detections*. It renders no pixels and does not need to; the policy consumes the 73-D
   observation, not images.
3. **Policy** — `interface.Policy`. PID, MPC or learned; the surrogate scores them
   identically, so the choice is an implementation detail.

## Two things that will cost you a week if skipped

**Validate the fit before tuning anything on it.** Replay recorded commands through the
fitted model, compare predicted vs. recorded IMU. This is the *only* place in the project
where a world frame appears, so it is the only place a sign error is silent rather than
self-announcing — and a wrong sign yields a model that fits, looks sensible, and quietly
mistunes every gain trained against it. It then presents as "great on the surrogate, bad in
the sim", which reads like a sim-to-real gap and sends you off fixing the noise model.

**The transfer risk is the noise model, not the plant model.** A policy will exploit any
regularity in synthetic detections. Dropout, latency and false-positive statistics must be
*measured off real frames* and randomised over their uncertainty. This is the point where a
good ML instinct and a good robotics instinct disagree; robotics wins.

## Do not start from vqual-1's `fly.py`

1535 lines of world-frame velocity setpoints steered by `ATTITUDE` yaw. VQ2 blocks both.
It completed the R1 course and is genuinely dead here — the actuation basis differs all the
way down. `link.py` (MAVLink plumbing) does transfer.
