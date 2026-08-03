# vqual-2 — handoff

For Alex (kongalex@umich.edu). Written 2026-07-31. **Deadline 2026-08-03 06:00 PST.**

You own everything **below the measurement vector**: plant fit, surrogate, control policy.
Perception and the attention policy stay with Claire and Claude.

## Start here, in this order

1. `pilot/interface.py` — the frozen contract. Read the header block before anything else.
2. `pilot/control/README.md` — your half, and the two mistakes that cost a week.
3. `pilot/NOTES.md` — measured facts about the simulator. Everything in it was observed,
   not assumed; where it contradicts the spec, it says so and the measurement wins.
4. `user-vm-cmds.md` — only if you need the CAEN VM. You probably don't.

## The problem in one paragraph

AI Grand Prix virtual qualifier round 2. A drone races a ~20-gate course in a simulated
hangar. The spec (§9.3) blocks `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY` and all gate
geometry, so there is **no pose, no position, and no heading** — only a camera at 30 fps,
an IMU at ~61 Hz, and a race packet that tells you which gate is next. Absolute yaw is not
recoverable from any of it (gravity is symmetric about yaw, there is no magnetometer).

The whole architecture follows from that: we fly **body rates only**, and every quantity in
the interface is body-referenced. Nothing ever converts to a world frame.

## The interface

```python
class Policy:
    def reset(self) -> None: ...
    def __call__(self, obs: Observation) -> Action: ...
```

* `Observation` — 73-D via `.to_vector()`. Up to 3 gates (body-frame position, normal,
  confidence, staleness), the guidance ribbon, own IMU state, race state, attention target.
* `Action` — `roll_rate`, `pitch_rate`, `yaw_rate` (rad/s), `thrust` (0..1).

PID, MPC, or learned — the surrogate scores them identically, so it's your call. If you go
learned, note `YawMode.AUTO_ATTENTION` hands yaw to a fixed servo and leaves you a 3-DoF
problem; yaw has almost no direct reward signal, so learning it is mostly wasted effort.

**Propose changes to `interface.py`, don't make them.** It is owned by neither side, and it
is the only thing keeping two people's code compatible.

## Your deliverables

**Policy.** `interface.Policy`, beating the baseline. This is the only hard requirement.

**Plant fit and surrogate are means, not ends.** The reasoning behind them: live sim runs
are ~15–25 per session, shared, and serialized behind one exclusive UDP port, so a
steppable NumPy sim (fitted plant + course map + *synthetic detections*, no pixels — the
policy consumes 73 numbers, not images) is what buys tuning iterations. If you get a
policy that flies without one — reactive control off the guidance ribbon, hand-tuned gains,
anything — that counts. Nothing in the interface assumes a model exists.

**The plant fit is done as of 2026-07-31** — `pilot/control/plant.json`, with
`pilot/control/plant.py` as the model and a steppable NumPy `Sim`. Drag fits to R² 0.99 on
a held-out session, thrust and rate loop to 0.92 and 0.99, and hover throttle comes out at
0.267 against the ~0.27 measured in flight. `pilot/control/README.md` has the numbers, the
five-stage pipeline that produced them, and what is still open.

The earlier note here said the first cut did **not** converge (thrust R² ≈ 0.05 against
both command and motor sum) and that the model route was not a solved subproblem. The
first half was true and is reproduced by the new tooling; the cause was one missing term.
`-a_z` is not thrust, it is thrust *plus vertical drag*, and in racing flight the two
nearly cancel — throttle goes up exactly when body-z speed is high. Add `kz·w|w|` to the
same regression on the same data and R² goes 0.0435 → 0.9205.

What remains of the surrogate is the part that was never started: the course map and the
**synthetic detection noise model**, which is the transfer risk the section below is about.

## Two things that will cost you a week if skipped

**If you fit a plant, validate it before tuning on it.** Replay recorded commands through the fitted model
and compare predicted vs. recorded IMU. The surrogate is the *only* place in this project
where a world frame appears, so it's the only place a sign error is silent instead of
self-announcing — and a wrong sign yields a model that fits the data, looks sensible, and
quietly mistunes every gain trained against it. It surfaces as "great on the surrogate, bad
in the sim", which reads like a sim-to-real gap and sends you off fixing the wrong thing.

`pilot/control/sysid_replay.py` is that check and it stays useful for any change to the
model: it replays open loop against held-out truth and scores deliberately mirrored copies
of the fit alongside the fit, so a model that has lost a sign fails visibly rather than
quietly. Worth knowing what it measured: a per-axis force fit at R² 0.99 barely notices a
mirrored rotation, and a **yaw** sign error does not move the drone at all — it shows up
only in attitude, so a position-only check would pass it.

**The transfer risk is the noise model, not the plant model.** A policy exploits any
regularity in synthetic detections. Dropout, latency and false-positive statistics must be
*measured off real frames* and randomised over their uncertainty. This is where a good ML
instinct and a good robotics instinct disagree; robotics wins.

## Data

Recordings live in `pilot/sessions/<timestamp>/` — `imu.csv`, `cmd.csv`, `race.csv`,
`frames.csv`, `actuators.csv`, `collisions.csv`, `events.jsonl`, `frames/*.jpg`.

**They are not in git** (hundreds of thousands of JPEGs) and currently exist on exactly one
laptop. Ask Claire for a copy; don't assume they're backed up.

`cmd.csv` pairs command with response, which is what makes system ID possible.

**Two sign facts you must have before touching the data** — both settled by measurement on
2026-07-31, both able to mirror a model silently:

1. The VQ1 truth streams are each wrong on a *different* axis.
   `truth_roll = ATTITUDE.roll`, `truth_pitch = ODOMETRY.pitch`, `truth_yaw = -ATTITUDE.yaw`.
   Roll and pitch were refereed against gravity on a parked drone; **yaw could not be**
   (parked, both streams sit at the degenerate −179.9° heading) and was settled separately
   on 2026-07-31 by a kinematic referee over flying data — `pilot/control/sysid_frames.py`,
   which brute-forces all sixteen candidate conventions and re-derives the other two lines
   as a side effect.
2. The simulator's whole body-rate convention is mirrored vs MAVLink NED — commands *and*
   gyro. Comparing them to each other correlates at +0.96 and proves nothing.

Details and the failed approaches: `pilot/control/README.md`.

### The VQ1 sim is a ground-truth rig — use it

VQ1 and VQ2 have **identical physics and identical gate dimensions** (all three spec
revisions diffed; only §4.5 Telemetry changed). But VQ1 does **not** block pose telemetry.

So flying the VQ1 sim with the same rate-only control path gives you `cmd` → **true pose**,
instead of `cmd` → IMU-and-hope. That directly de-risks your largest task. `teleop.py`
records `attitude.csv`, `position.csv` and `odometry.csv` automatically whenever those
messages arrive, and shows `[TRUTH]` in the HUD when they do.

**vqual-1's own recordings are not usable for this**, despite an earlier draft here saying
they were. They hold pose at ~9 Hz and nothing else — no gyro, no thrust, no command
channel — so there is no input to regress against, and differentiating 9 Hz attitude for a
racing quad aliases. They are labelled *perception* data, not plant data.

**The rule:** ground truth is used to *check* estimators offline. It never feeds the pilot.
The VQ2 pilot consumes permitted streams only.

## Rules that constrain design

* **No human interaction during a submitted timed run** — immediate DQ (§7). Max 8 minutes.
* **Absolute yaw is out of bounds.** The sim *will* honour an attitude quaternion, and we
  measured it working (bank tracks command, slope 0.890, rms 1.90°). We rejected it: a
  quaternion encodes absolute yaw, and commanding absolute yaw is reading it — a compass
  obtained through the actuator instead of the blocked telemetry stream. §9.2 reserves a
  code audit for "manipulation of the simulator constraints". `ATTITUDE_IGNORE` stays set.
  Please don't reopen this for a smoother inner loop; it was decided deliberately.

## Do not start from vqual-1's `fly.py`

It completed the R1 course, so it looks like a head start. It isn't. 1535 lines of
world-frame velocity setpoints steered by `ATTITUDE` yaw — VQ2 blocks both, and the
actuation basis differs all the way down. `link.py` (MAVLink plumbing) does transfer.

## Logistics

* **Repo**: `C:\Users\USER\Projects\vqual-2`, git, **no remote yet**. Needs one before you
  can pull — ask Claire.
* **Partition**: `pilot/perception/` (ours) and `pilot/control/` (yours). `interface.py`
  sits above both. Staying inside your directory means we never conflict.
* **Sim access**: you likely don't need it. Your half runs entirely on the surrogate and
  the recordings. The live sim is one exclusive UDP port on one machine, so it is a
  serialized integration resource — coordinate with Claire rather than assuming access.
  If you do get a slot, **card 2 at the bottom of `pilot/SYSID.md` is what to fly**: three
  minutes that retire the plant's last unmeasured hack (the low-throttle thrust clamp), the
  one coefficient still fitted jointly with another, and the missing completed lap. Adding
  terms to the model is exhausted — every plausible one now makes held-out fit worse or
  barely better — so that flight is the only remaining lever on plant accuracy.
* **Environment**: Python 3, NumPy, OpenCV, pymavlink. Pin `numpy==1.26.4` and
  `opencv-python==4.10.0.84`; unpinned installs pull opencv 5 → numpy 2 and break things.

## Open questions nobody has answered yet

* `R_COMMIT` — the range at which attention hands off from the current gate to the next.
  Needs a real approach measurement.
* Gate count (~20) is Claire-observed, unconfirmed.
* Vertical profile of the course is unknown. In VQ1 it descended ~20°, which put gates below
  the camera's −9.4° lower frame edge.
*(Closed: the flight-mode question. Earlier drafts of this file asked whether the sim's
Acro↔"Stabilized" toggle would self-level for us. It resolved to **no, ACRO is the only
mode reachable** — the pak carries the parent game's ACRO/ANGLE/ARCADE/GPS strings, but
this build's menu exposes graphics and sound only and nothing else names a mode. Details
and the levelling assist that replaced it: `pilot/NOTES.md`.)*
