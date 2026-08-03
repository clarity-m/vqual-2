# Network architecture — review and the two changes that came out of it

2026-08-02. Code: `train/network.py`, `train/framestack.py`, `train/actionmap.py`.
Companion to `TRAINING_ARCHITECTURE.md` (design intent) and `STATE_RL_TRAINING.md`
(measured results). Everything below is a measurement on the surrogate unless labelled
otherwise. **Nothing here has been flown live.**

## Verdict

`network.py` is not the problem. It is a conventional, correctly-implemented actor-critic
and the review found no defects in it. **The binding constraints are in the observation
and in what the network is asked to infer from it.**

## The network, as it stands

2×256 tanh MLP, separate policy and value trunks over a flattened `k × 73` input;
tanh-squashed Gaussian with a state-independent log-std; orthogonal init with `mu` gain
0.01 and the thrust head biased to hover. The checkpoint contract is clean and genuinely
shared with deployment. The claim in the module docstring that the tanh Jacobian cancels
in the PPO ratio is correct — it depends only on the stored pre-tanh `z`.

One property worth knowing rather than fixing: because `log_std` is state-independent,
**the entropy bonus contributes exactly zero gradient to either trunk**. It is a pure
regularizer on one global 3-vector, so "exploration" is a fixed isotropic blob the policy
cannot modulate by situation. Consistent with the measured result that `--ent-coef`
0.005 → 0.001 bought real accuracy (σ 0.40 → 0.299).

## Finding 1 — the policy cannot estimate its own vertical rate

`STATE_RL_TRAINING.md` open item 5 lists this as untested. It is testable, and it is
worse than "no vertical-rate channel" suggests: the *entire* velocity estimate is
horizontal.

```
vel_bearing = atan2(-accel_y, -accel_x)      # horizontal drag
speed       = sqrt(a_horiz / K_DRAG)          # horizontal drag
```

`accel_z` is thrust plus gravity, not ż. So the only route to vertical rate is
differencing gate `pos_body.z` across the frame stack. Measured against the real
detector (difficulty 0.3, `noise_scale` 1.0, 78k samples, slot 0 verified to be the
active gate):

| range | median \|err\| | p99 | fraction > 0.75 m |
|---|---|---|---|
| 2–5 m | 0.24 m | 5.6 m | 19.7% |
| 5–8 m | 0.27 m | 7.9 m | 16.7% |
| 8–12 m | 0.28 m | 4.1 m | 14.3% |
| 12–18 m | 0.33 m | 5.8 m | 17.6% |

Two things fall out. The typical detection is decent, but **roughly 1 in 6 valid
detections places the gate's z outside the aperture half-width** — pointing the policy at
a spot outside the hole it is flying through. And differencing that over the k=6 window
(0.111 s) leaves ~4–5 m/s of noise even on the robust error scale, against a signal of
**0.23 m/s** (run1's measured 2.1 m/s² sink, over that window).

This is why 55.5% of run1's episodes ended on the floor, and why the clearance potential
had to be added. That shaping term is compensating for an unobservable.

### Where the signal actually is

Linear probe against true vertical velocity — a lower bound on what the MLP can extract —
over a 0.70 s dilated stack:

| channel group | R² | RMSE |
|---|---|---|
| accelerometer (51–53) | **0.332** | 1.83 m/s |
| own-state (48–61) | 0.472 | 1.62 m/s |
| **perception-derived (0–47)** | **0.125** | 2.09 m/s |
| all 73 | 0.562 | 1.48 m/s |

*(signal σ(vz) = 2.19 m/s)*

**The channels originating from the camera carry almost none of it.** The policy receives
no pixels — "perception-derived" means gate `pos_body`/`normal_body`/confidence/staleness
and the ribbon bearings, i.e. perception's outputs. Vertical rate lives in the IMU, and it
lives there through *integration*. No amount of stacking perception channels recovers it.

## Finding 2 — the frame stack is shorter than it looks

k=6 at 55 Hz spans 0.111 s, but the camera runs at 30.05 Hz: **only ~3.3 distinct
detections** in the window, fewer at `p_detect` 0.62–0.92. Most of the 438-wide input
re-reads the same detection. Meanwhile `max_coast_s` is 0.35–1.10 s, so dropouts are
actually covered by the producer's coast, not by the stack — the stated purpose
("a memoryless policy flies blind through every dropout") is being served by something
else.

## Finding 3 — the thrust term is smaller than the policy's own exploration noise

A multirotor must raise thrust as `1/(cos roll · cos pitch)` merely to hold altitude.
Under the plain affine map the network learns that trigonometry implicitly, from reward,
through a tanh. The correction it must find:

| attitude | thrust correction | as a multiple of exploration σ |
|---|---|---|
| 10° | 0.0084 | 0.10 |
| 15° | 0.0194 | 0.24 |
| 20° | 0.0358 | 0.43 |
| **run1 measured deficit** | **0.0370** | **0.45** |

run1 flew at 14.5° bank / 14.1° nose-down commanding 0.278 where 0.315 was needed. That
deficit is exactly the size of the compensation term, and it is **less than half the noise
the policy explores with** — a systematic sub-σ bias the optimiser cannot cleanly separate
from its own sampling.

## What changed

Both opt-in; defaults reproduce previous behaviour exactly.

### `FrameStack(offsets=...)`

The k frames need not be consecutive. `--frame-offsets 32,16,8,4,2,0` spans 0.58 s at
**identical input width and parameter count**. Probe against true vertical velocity:

| offsets | span | R² | RMSE |
|---|---|---|---|
| `[0]` | 0.000 s | 0.417 | 1.68 m/s |
| `[5,4,3,2,1,0]` (default) | 0.110 s | 0.460 | 1.62 m/s |
| `[10,8,6,4,2,0]` | 0.220 s | 0.492 | 1.58 m/s |
| `[32,16,8,4,2,0]` | 0.704 s | 0.562 | 1.48 m/s |
| `[64,32,16,8,4,0]` | 1.409 s | 0.588 | 1.44 m/s |

**Real but modest.** An earlier back-of-envelope estimate suggested a longer baseline
would flip vertical rate from unobservable to observable; the probe says otherwise, because
it ignored that the network reads all 73 channels rather than gate z alone. Take this
because it costs nothing, not because it fixes anything.

Implemented as a ring buffer — at `n_envs=2048` and depth 65 a per-step shift would copy
~39 MB to move data that has not changed. Verified bit-identical to the previous
implementation on the default path over 200 steps including mid-episode resets.

### `ResidualThrustMap`

`u[2]` becomes a residual about attitude-compensated hover, so **`u[2] = 0` means "hold
altitude" at any attitude** and the correct thrust head bias drops to exactly 0. The
`1/(cos·cos)` term moves out of the learned function and into the mapping; the network
only learns the part that depends on where the gate is. Clamped at 60° of combined tilt,
where the reciprocal would demand thrust the aircraft does not have.

Attitude comes from the **measured** roll/pitch in the observation (indices 54/55,
verified against `interface.Observation` rather than counted by hand), never from truth,
so training and deployment compute the identical command from the identical input.

## Guarding the thing that would fail silently

A checkpoint that scores well in training and flies a *different* policy live — because
deployment stacked frames differently or applied a different thrust map — is the failure
mode this whole area invites. Both options are therefore recorded in the checkpoint
(`frame_offsets`, `thrust_residual`), absent for legacy checkpoints, and `tests` cover:

* every combination of {consecutive, dilated} × {affine, residual} produces **identical
  physical commands** through the training path and through `RLPolicy` (max difference
  0.000e+00);
* a checkpoint written without the options carries neither key and loads with both
  defaults;
* the default `FrameStack` path is bit-identical to the previous implementation.

End-to-end training runs with both options on.

## Still open

1. **Perception noise is a hand-specified guess.** `p_pose_fail` is 0.04–0.28 — a 7×
   range that straddles whether the ~1-in-6 gross-error rate above is real. If it is 4%,
   per-gate accuracy has headroom above 0.705; if 28%, 0.705 may be near the ceiling.
2. **Is the gate-position error correlated in time?** Finding 1 assumes it is white. If
   it is a slowly-drifting bias, differencing cancels the shared part and vertical rate
   becomes far more observable — differenced noise goes as `σ√(2(1−ρ))`, so ρ=0.9 is a
   3.2× improvement. This is the one measurement that could overturn the finding, and it
   is obtainable from recordings with known ground truth.
3. **An explicit vertical-rate estimator** — a complementary filter on accel plus
   attitude, fed in as a derived channel. Justified by the ablation: the information is in
   the IMU and needs integration, which a fixed window cannot do however wide. This is the
   real fix for Finding 1 and is not implemented.
4. **Neither change has been trained to convergence.** They are wired, tested for
   agreement, and smoke-tested — not yet shown to help.
5. **Spawn attitude** is a separate finding, handed to the environment work: the surrogate
   spawns at ~0° pitch where the real vehicle starts ~20° nose-down on a platform, which
   puts gate 0 at image row 294/360 instead of 178 and leaves 11% of forward starts with
   the gate out of frame. `camera.py`'s docstring describes the effect of pitching down
   backwards; the code is correct.
