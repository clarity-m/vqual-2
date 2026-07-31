# System-ID flight card — VQ1 build, ground truth recording

Fly this to produce the data the plant fit and surrogate are built from. ~6 minutes
of flying. **Use the VQ1 build** (`AIGP_VQ1_3391`) — the VQ2 build sends no pose.

Why not just fly a lap: a lap has correlated inputs and a narrow envelope, so a model
fitted to it only knows how to fly that lap. These maneuvers excite one thing at a time.

## Before flying

    python3 pilot/msgscan.py --secs 5        # expect ATTITUDE / LOCAL_POSITION_NED / ODOMETRY
    python3 pilot/teleop.py --listen --sessions <local-disk>   # ~15 s

Then stop and check `attitude.csv`, `position.csv`, `odometry.csv` are non-empty and the
HUD showed `[TRUTH]`. If msgscan shows the messages but the CSVs are empty, the recorder
is at fault, not the sim — say so and stop.

**Then confirm VQ1 honours body rates at all.** Nobody has tested this: vqual-1 flew
velocity setpoints exclusively, so the DCL `type_mask` bit-16 path has never run against
this build. Arm, small roll input, see if it rolls. If it does not, this card is void and
the ground-truth rig needs rethinking.

## Recording

    python3 pilot/teleop.py --sessions <local-disk>

**Press `F12` between every maneuver.** It writes a marker into `events.jsonl`, which is
what makes segments findable afterwards instead of hunting a continuous trace. Cheap to
press, expensive to omit.

Leave a second or two of quiet between maneuvers so each starts from a settled state.

## The maneuvers

One axis at a time. Coupled inputs make the fit ill-conditioned.

| # | maneuver | identifies |
|---|---|---|
| 1 | Hover steady ~5 s | trim thrust, noise floor, accelerometer bias |
| 2 | **Roll doublet** — right ~0.5 s, left ~0.5 s, back to centre. Three reps: gentle, medium, hard | roll rate-loop gain + time constant |
| 3 | **Pitch doublet** — same pattern, three reps | pitch rate loop |
| 4 | **Yaw doublet** — same pattern, three reps | yaw rate loop, prop reaction torque |
| 5 | **Throttle steps** — hover, then step up ~2 s, back, step down ~2 s, back. Two magnitudes | thrust curve, mass |
| 6 | **Cruise ladder** — accelerate to a steady speed and HOLD until it stops accelerating. Repeat at 3–4 distinct speeds, slow to fast | drag coefficient. Terminal speed is where drag balances, so this is a direct read |
| 7 | **Deliberate skids** — at cruise, yaw 30–60° off the direction of travel and hold ~2 s. Both directions | validates `(ax,ay) ∝ −v`. The only condition where true velocity is *not* straight ahead, so the only way to check the drag-bearing estimator |
| 8 | **One ordinary lap** | held-out validation — see below |

Doublets rather than steps: they return you near the starting state, so you drift less and
the excitation spectrum is richer.

Maneuver 7 is the one to not skip. Skidding is a control *hazard* but the exact excitation
the velocity-direction estimator needs, and that estimator is load-bearing for coordinated
turning in VQ2.

## After

Fit on maneuvers 1–7. **Validate on maneuver 8, held out.** If the model predicts a lap it
never saw, it is real; if it only predicts the doublets, it memorised them.

Then the mandatory check before any gain is tuned: replay recorded commands through the
fitted model and compare predicted vs. recorded IMU *and* true pose. The surrogate is the
only place in this project a world frame appears, so it is the only place a sign error is
silent instead of self-announcing.

## Note on rates

`HIGHRES_IMU` runs slower on the VQ1 build (47.7 Hz) than on VQ2 (62.9 Hz) because of the
extra pose streams, and neither is fixed. Weight everything by actual `time_usec` deltas;
never assume a nominal period. Truth arrives at 72–114 Hz, well above the IMU, so the fit
should be driven by truth and the IMU treated as the noisy sensor it will be in VQ2.
