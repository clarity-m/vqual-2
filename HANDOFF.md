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

* `Observation` — 70-D via `.to_vector()`. Up to 3 gates (body-frame position, normal,
  confidence, staleness), the guidance ribbon, own IMU state, race state, attention target.
* `Action` — `roll_rate`, `pitch_rate`, `yaw_rate` (rad/s), `thrust` (0..1).

PID, MPC, or learned — the surrogate scores them identically, so it's your call. If you go
learned, note `YawMode.AUTO_ATTENTION` hands yaw to a fixed servo and leaves you a 3-DoF
problem; yaw has almost no direct reward signal, so learning it is mostly wasted effort.

**Propose changes to `interface.py`, don't make them.** It is owned by neither side, and it
is the only thing keeping two people's code compatible.

## Your three deliverables

**1. Plant fit.** Mass, drag, thrust curve, rate-loop response, from `cmd.csv` → `imu.csv`.

**2. Surrogate.** Steppable NumPy sim: fitted plant + course map + *synthetic detections*.
It renders no pixels and doesn't need to — the policy consumes 70 numbers, not images.
This is what buys tuning iterations; live sim runs are ~15–25 per session, shared, and
serialized behind one exclusive UDP port.

**3. Policy.** Beat the baseline.

## Two things that will cost you a week if skipped

**Validate the fit before tuning on it.** Replay recorded commands through the fitted model
and compare predicted vs. recorded IMU. The surrogate is the *only* place in this project
where a world frame appears, so it's the only place a sign error is silent instead of
self-announcing — and a wrong sign yields a model that fits the data, looks sensible, and
quietly mistunes every gain trained against it. It surfaces as "great on the surrogate, bad
in the sim", which reads like a sim-to-real gap and sends you off fixing the wrong thing.

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

### The VQ1 sim is a ground-truth rig — use it

VQ1 and VQ2 have **identical physics and identical gate dimensions** (all three spec
revisions diffed; only §4.5 Telemetry changed). But VQ1 does **not** block pose telemetry.

So flying the VQ1 sim with the same rate-only control path gives you `cmd` → **true pose**,
instead of `cmd` → IMU-and-hope. That directly de-risks your largest task. `teleop.py`
records `attitude.csv`, `position.csv` and `odometry.csv` automatically whenever those
messages arrive, and shows `[TRUTH]` in the HUD when they do.

All vqual-1 recordings are also valid system-ID data for the same reason.

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
* **Environment**: Python 3, NumPy, OpenCV, pymavlink. Pin `numpy==1.26.4` and
  `opencv-python==4.10.0.84`; unpinned installs pull opencv 5 → numpy 2 and break things.

## Open questions nobody has answered yet

* `R_COMMIT` — the range at which attention hands off from the current gate to the next.
  Needs a real approach measurement.
* Gate count (~20) is Claire-observed, unconfirmed.
* Vertical profile of the course is unknown. In VQ1 it descended ~20°, which put gates below
  the camera's −9.4° lower frame edge.
* Whether the sim's Acro ↔ Stabilized flight-mode toggle self-levels while accepting
  ordinary body rates. If it does, we get roll/pitch stabilisation with no quaternion ever
  transmitted — clean by construction. All probes so far ran in ACRO.
