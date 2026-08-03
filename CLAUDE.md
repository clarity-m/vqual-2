# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI Grand Prix **virtual qualifier round 2 (VQ2)**: a drone races a ~20-gate course in a
simulated hangar. Spec §9.3 blocks `ATTITUDE`, `LOCAL_POSITION_NED`, `ODOMETRY` and all gate
geometry, so there is **no pose, no position, no absolute heading** — only a 30 fps camera, a
~61 Hz IMU, and a race packet naming the next gate. Absolute yaw is not recoverable from any of
it (gravity is symmetric about yaw; no magnetometer).

The entire architecture follows from that: the drone flies **body rates only**, and every
quantity crossing a module boundary is body-referenced. Nothing converts to a world frame except
the plant fit and surrogate — which is precisely why sign errors are dangerous there and nowhere
else.

Deadline is 2026-08-03 06:00 PST; the repo is under active, time-boxed work.

## Read before editing anything

1. `pilot/interface.py` — the frozen contract. Its header block is required reading.
2. `pilot/CONVENTIONS.md` — **the single source** for every sign, frame and axis fact. If you
   change a sign, change it there first.
3. `pilot/control/CONTROL_STATE_AGENTS.md` — disk-verified state of the control pipeline
   (dated 2026-08-01), including a table of where older docs have drifted from the code.
   Two docs are newer and supersede it in their areas: `STATE_RL_TRAINING.md` (what training
   actually produced) and `STATE_VQ2_COURSE.md` (the measured VQ2 map, now wired into the
   surrogate via `surrogate/vq2course.py`, opt-in through `vq2_frac`).
4. `pilot/NOTES.md` — measured facts about the simulator. Where it contradicts the spec, the
   measurement wins.

**Authority order when docs disagree:** code + `plant.json` + checkpoint sidecars/logs >
`CONTROL_STATE_*.md` / `STATE_PLANT_SURROGATE.md` > `TRAINING_ARCHITECTURE.md` and the
`HANDOFF*.md` files (design intent, several places stale).

## Ownership and the contract

`pilot/perception/` belongs to Claire; `pilot/control/` belongs to Alex. `pilot/interface.py`
sits above both and is owned by neither — **propose changes to it, do not edit it unilaterally**.
It is the only thing keeping the two halves compatible.

```python
class Policy:
    def reset(self) -> None: ...
    def __call__(self, obs: Observation) -> Action: ...
```

`Observation.to_vector()` is 73-D (`OBS_DIM`); `Action` is `roll_rate`, `pitch_rate`, `yaw_rate`
(rad/s) and `thrust` (0..1). Under `YawMode.AUTO_ATTENTION` yaw goes to a fixed azimuth servo,
leaving the policy a 3-DoF problem (roll, pitch, thrust) — that is the default everywhere in
`control/`. The surrogate and the live sim present the identical interface, so a policy runs
against either unchanged.

The deliverable is one `interface.Policy` that beats the reactive baseline on the live sim. The
plant fit and surrogate are means, not ends.

## Commands

There is no test framework — **each subsystem ships a `selfcheck`/`selftest` script that prints
PASS/FAIL**, and those are the tests. Run everything from the repo root.

```powershell
python pilot/control/plant.py                      # plant model self-check
python pilot/control/sysid_replay.py               # the replay gate (see traps)
python pilot/control/surrogate/selfcheck.py        # 9 stages, 8 assert + 1 report-only
python pilot/control/train/selftest.py             # PPO harness, ~1 min CPU, no surrogate needed
python pilot/control/policies/selfcheck.py         # baseline / supervisor / envelope drift
```

System ID pipeline, in order — each stage refuses to be useful without the one before it:

```powershell
python pilot/control/sysid_frames.py  pilot/sessions/2026*/   # settle the frame convention
python pilot/control/sysid_latency.py pilot/sessions/2026*/   # bracket the lags
python pilot/control/sysid_apex.py                            # kz from the apex arcs
python pilot/control/sysid_fit.py                             # -> plant.json
python pilot/control/sysid_replay.py                          # mandatory gate
```

Training (`--total-steps` is an **absolute** target, not steps-this-session):

```powershell
python pilot/control/train/train.py --env surrogate --total-steps 20000000 --n-envs 256 --n-steps 128 --frame-stack 6 --name run2 --seed 1
python pilot/control/train/train.py ... --name run1 --resume run1_s5046272   # opt-in only
python pilot/control/train/train.py --env testenv --total-steps 400000 --n-envs 64 --no-curriculum
```

`--env-kwarg KEY=VALUE` sets any extra `EnvConfig` field (python literal); unknown fields warn
rather than crash.

Evaluation:

```powershell
python pilot/control/evalsuite/run_policy.py --policy baseline --seeds 0-9 --difficulty 0.0
python pilot/control/evalsuite/run_policy.py --policy rl --ckpt run1_s5046272 --seeds 0-9
python pilot/control/evalsuite/select.py      # rank checkpoints
python pilot/control/evalsuite/stress.py      # same eval at decision_hz_range=(28,32)
```

Live sim (Training mode only, one exclusive UDP port, ~15–25 runs per session — a serialized
shared resource, coordinate before assuming access): `python pilot/teleop.py --hover 0.27`.

**Environment:** Python 3 + NumPy + OpenCV + pymavlink; `torch` (CPU is enough) for `train/`.
Pin `numpy==1.26.4` and `opencv-python==4.10.0.84` — unpinned installs pull opencv 5 → numpy 2
and break things.

## Pipeline architecture

```
P0 recordings -> P1 plant fit -> P2 surrogate -> T4 PPO -> E5 eval -> D6 deploy
   sessions/       plant.json     surrogate/     train/    evalsuite/  policies/
                        ^              ^
                 sysid_replay gate   P3 noise model (surrogate/noise.py)
```

* **P1 `plant.py` / `plant.json`** — body-axis quadratic drag (`kx/ky/kz`), a 21-knot measured
  thrust table (`np.interp`, hover 0.270), first-order rate loops with ~10 ms delay. `plant.py`
  also exposes a steppable NumPy `Sim`. Done and validated; refitted on card-2 data 2026-08-01.
* **P2 `surrogate/`** — fitted plant + procedural course + *synthetic detections*. It renders no
  pixels: the policy consumes 73 numbers, not images. `VecSurrogate` is the vectorized env
  (`obs [n,73]`, `step(physical [n,3])`), `SingleSurrogate` the 1-env adapter. Exists to buy
  tuning iterations the rate-limited live sim cannot.
* **P3 `surrogate/noise.py`** — detection dropout / latency / false positives. **Still the
  hand-specified pessimistic fallback, not measured.** This is the largest transfer risk.
* **T4 `train/`** — PPO on a 2x256 tanh MLP with frame stacking. See `train/README.md` for the
  exact 7-key checkpoint format and the load-side recipe.
* **E5 `evalsuite/`** — `run_policy.py` is primary; `_probe_*.py` are diagnostic scripts whose
  comments record prior findings.
* **D6 `policies/`** — `baseline.py` (reactive PID floor), `rl.py` (checkpoint -> Policy),
  `supervisor.py` (recovery: level + hover, hand back), `envelope.py` (`clamp_action`).

Current reality worth knowing (measured, `STATE_RL_TRAINING.md`): the RL policy beats the
reactive baseline **2.3–2.5× on per-gate accuracy** — 0.705 against 0.307 at the physical
aperture, `runB1` at 207M steps. **Neither policy completes a course**, and that gap is
arithmetic rather than tuning: completion over 20 gates needs a per-gate rate around 0.989
against the 0.705 measured. Nothing has been flown live.

## Traps that have already cost this project weeks

**Signs are settled only by a referee outside the thing being tested.** The simulator's body-rate
convention is mirrored versus MAVLink NED on *both* commands and gyro, so commanded-rate vs
measured-gyro correlates at +0.96 and proves nothing. Three separate telemetry-vs-telemetry
arguments confidently concluded the wrong yaw sign. Reading the source is not a referee.

**The mirror covers rates and attitude, not position.** `LOCAL_POSITION_NED` is canonical NED;
`ATTITUDE` needs per-field correction (`truth_roll = ATTITUDE.roll`, `truth_pitch =
ODOMETRY.pitch`, `truth_yaw = -ATTITUDE.yaw`). Applying the mirror to one and not the other
leaves a model mirrored in exactly one term — it fits the data, looks sensible, and silently
mistunes every gain trained on it. It then presents as a sim-to-real gap.

**Rate sign flips live in the link layer only** (`SIGN_ROLL = SIGN_PITCH = SIGN_YAW = -1`). No
file above it may contain one. `perception/label.py` is the single deliberate exception and it is
not a rate.

**Never tune on a fitted plant before `sysid_replay.py` passes.** A per-axis force fit at R² 0.99
barely notices a mirrored rotation, and a yaw sign error does not move the drone at all — it
shows only in attitude, so a position-only check passes it.

**Run the frame referee on all sessions, never a subset.** Its margin depends on attitude
variety; card-2-only scores 1.0x and FAILs for lack of variety, not because anything is wrong.

**Absolute yaw is out of bounds by decision, not by capability.** The sim will honour an attitude
quaternion and we measured it working. It was rejected: a quaternion encodes absolute yaw, and
commanding it is reading it through the actuator. `ATTITUDE_IGNORE` stays set; do not reopen this
for a smoother inner loop.

**Do not start from the vqual-1 `fly.py`.** 1535 lines of world-frame velocity setpoints steered
by `ATTITUDE` yaw — VQ2 blocks both. Only `link.py` (MAVLink plumbing) transfers.

## Invariants enforced by code

* `surrogate/actions.policy_to_action` is the **sole squash home** (`RATE_CAP_RPS = 2.75`, not
  `interface.MAX_RATE_RPS`; `THRUST_FLOOR = 0.10`). `policies/envelope.py` mirrors the constants
  and `selfcheck` asserts they have not drifted. Never reimplement the mapping.
* Observation pipeline order is **normalize -> stack**, never stack -> normalize — that is what
  keeps `obs_mean`/`obs_std` 73-D and independent of k. Deployment must match.
* The surrogate feeds **canonical NED** into the integrator; do not call `plant.rates_ned()` on
  already-NED commands (that path exists for recording-convention inputs only).
* PPO bootstraps **time-limit truncations from `V(s_T)`** via `info['terminal_obs']` +
  `info['timeout']`; collisions, corridor exits and finishes still bootstrap at zero.
* **Never weaken `k_collision`.** The only reward levers the curriculum may pull are
  `progress_gate_scale`, `speed_cap` and `enable_time_penalty`. Demotion never drops below start
  values.
* `train.py --resume` without a `<stem>_resume.pt` sidecar is a **degraded** continuation
  (weights only, no Adam/curriculum/RNG state) and prints a banner saying so.

## Data

Recordings are `pilot/sessions/<timestamp>/` — `imu.csv`, `cmd.csv` (command paired with
response, which is what makes system ID possible), `race.csv`, `frames.csv`, `actuators.csv`,
`collisions.csv`, `events.jsonl`, `frames/*.jpg`.

The telemetry CSVs are tracked (force-added past `.gitignore`); **the JPEGs are not, and are only
partly on any given machine** — count files before scoping anything that runs over frames.

* Fit sessions and the designated hold-out (`20260731-131305`) are recorded in `plant.json` meta
  so the replay gate cannot validate on its own training data.
* `20260731-130744` and `20260731-233219` are **excluded**: their `LOCAL_POSITION_NED` velocity
  repeats bit-identically in ~half the rows, so the referee differentiates a held signal. It is a
  recording artifact — re-flying reproduces it rather than fixing it.
* The sim boot clock **restarts at every `SIM_RESET`, mid-session**. Load a session as a list
  of continuous epochs (`sysid_data.py` does this); differentiating across a seam invents
  several-hundred-m/s² samples.
* The `armed` column in `cmd.csv` is unusable on the VQ1 build — filtering on it discards the
  widest-envelope data there is.

VQ1 and VQ2 have identical physics and gate dimensions (all three spec revisions diffed; only
§4.5 Telemetry changed), and VQ1 does *not* block pose. So VQ1 flights are the ground-truth rig.
**The rule: ground truth checks estimators offline, and never feeds the pilot.**

## Conventions for changes here

Design work lands as committed markdown beside the other pipeline docs in `pilot/control/`, not
as scratch notes. When a doc's claims are superseded, update the drift table in
`CONTROL_STATE_AGENTS.md` rather than leaving two versions of a fact in the tree — this project
has already been bitten twice by two copies of a convention drifting apart.
