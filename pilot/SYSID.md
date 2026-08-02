# System-ID flight card — VQ1 build, ground truth recording

Fly this to produce the data the plant fit and surrogate are built from. ~6 minutes
of flying. **Use the VQ1 build** (`AIGP_VQ1_3391`) — the VQ2 build sends no pose.

This card was flown on 2026-07-31 and the plant fit came out of it. **Card 2 at the bottom
of this file is the follow-up** — three minutes aimed at the specific holes the fit exposed.

**Card 2 was flown on 2026-08-01 and the plant was refitted on it.** Both cards' data is
banked; see the status block on card 2 for what is left, which is not much and does not
justify a sim slot on its own.

Why not just fly a lap: a lap has correlated inputs and a narrow envelope, so a model
fitted to it only knows how to fly that lap. These maneuvers excite one thing at a time.

## Before flying

    python3 pilot/msgscan.py --secs 5        # expect ATTITUDE / LOCAL_POSITION_NED / ODOMETRY
    python3 pilot/teleop.py --listen --sessions <local-disk>   # ~15 s

Then stop and check `attitude.csv`, `position.csv`, `odometry.csv` are non-empty and the
HUD showed `[TRUTH]`. If msgscan shows the messages but the CSVs are empty, the recorder
is at fault, not the sim — say so and stop.

**VQ1 honours body rates — CONFIRMED 2026-07-31** (Claire flew the VQ2 teleop against the
VQ1 build in acro; handled normally). The DCL `type_mask` bit-16 path works on both
builds, so the same rate-only control path spans the ground-truth rig and the race sim.

## Recording

    python3 pilot/teleop.py --no-frames --no-view --sessions <local-disk>

`--no-frames` because system ID needs no camera, and it drops the session from hundreds of
megabytes to about a megabyte of CSV — trivial to move. `--no-view` because the cv2 window
adds control-loop jitter, and command *timing* accuracy is exactly what identifying a rate
loop depends on.

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

## Correct the truth streams before using them

`ATTITUDE` and `ODOMETRY` are each sign-inverted on a **different** axis (refereed against
gravity — see NOTES.md). Apply this first, or every downstream fit is mirrored:

    truth_roll  =  ATTITUDE.roll      # == -ODOMETRY.roll
    truth_pitch =  ODOMETRY.pitch     # == -ATTITUDE.pitch
    truth_yaw   = -ATTITUDE.yaw

Yaw was unresolved when this card was written — parked, both streams sit at −179.9°, the
degenerate heading where a sign flip is invisible. It is now **settled**: the flying data
covers the full −180..+180° range, and `pilot/control/sysid_frames.py` scores all sixteen
candidate conventions against `R_wb·a_body + g == dv_world/dt`. `-ATTITUDE.yaw` wins with a
6x margin on the residual tail. Re-run that script rather than re-deriving it by hand.

## After

Fit on maneuvers 1–7. **Validate on maneuver 8, held out.** If the model predicts a lap it
never saw, it is real; if it only predicts the doublets, it memorised them.

Then the mandatory check before any gain is tuned: replay recorded commands through the
fitted model and compare predicted vs. recorded IMU *and* true pose. The surrogate is the
only place in this project a world frame appears, so it is the only place a sign error is
silent instead of self-announcing.

## Note on rates

`HIGHRES_IMU` runs slower on the VQ1 build (47.7 Hz when this was written) than on VQ2
(62.9 Hz) because of the extra pose streams, and neither is fixed. Weight everything by
actual `time_usec` deltas; never assume a nominal period. Truth arrives at 72–114 Hz, well
above the IMU, so the fit should be driven by truth and the IMU treated as the noisy sensor
it will be in VQ2.

**How un-fixed it is, measured 2026-07-31:** across the eight recorded sessions the IMU rate
is 31.8–34.6 Hz for the four flown in the morning and 60.6–61.6 Hz for the four flown that
afternoon — a factor of two, with `ATTITUDE` and `ODOMETRY` steady at ~112 and ~71 Hz
throughout. 47.7 Hz is neither tier. Nothing in `teleop.py` requests a message interval, so
whatever changed is not recorded. Details and why it matters for the referee:
`sessions/README.md`.

---

# Card 2 — the three gaps card 1 left, 2026-07-31

## STATUS: FLOWN 2026-08-01 — read this before flying any of it again

A and B were flown; C crashed at gate 0 and D was not flown. The plant was refitted on
what came back (`pilot/control/README.md`, `pilot/control/HANDOFF_SURROGATE.md`).

| maneuver | session | outcome |
|---|---|---|
| A — apex reads | `20260801-004843` | **Retired gap 1 and gap 2.** `kz = 0.04359 ± 0.00016` from four wide arcs, thrust as a 21-knot measured table. The hand clamp is gone. |
| B — terminal up/down | `20260801-005059` (+ `005518` rerun) | Terminal descents agree with the arcs at 0.0435 / 0.0436. In the fit. |
| C — a complete lap | `20260801-005715` | **Crashed ~14 s in at gate 0.** No usable lap. But the *geometry* gap it was aimed at is closed anyway: `20260731-195307` and `20260731-204841-vq1-lap-slow` both flew the whole 6-gate VQ1 map, crossing points agreeing to 0.5–1.2 m, and the second carries 2937 frames. |
| D | not flown | **Do not fly it.** Not a piloting problem — see below. |

**Do not fly D.** The two sessions it would extend (`20260731-130744`, `20260731-233219`)
fail the kinematic referee because `LOCAL_POSITION_NED` velocity repeats bit-identically
in 53% / 43% of rows while the drone manoeuvres hard, and the referee differentiates a
held signal. That is a recording artefact and re-flying reproduces it.

**What is actually left**, and none of it blocks anything: throttle 0.50–0.82 is thin
(~240 samples), so terminal holds at 0.6 and 0.8 for ~30 s each would thicken the middle
of the thrust table; and `kz` varies ~7% with forward speed (0.0443 at `|u| < 2`, 0.0412
at 10–15), which a body-lift `c·u²` term captures — measured, +0.043 held-out R², not yet
in `plant.json` because it has not been through the replay gate.

The rest of this card is kept as the procedure, not as a work item.

**~3 minutes of flying, ~10 on the clock.** Written *after* the plant fit closed
(`pilot/control/plant.json`), so unlike card 1 every maneuver here is aimed at a weakness
that has been measured rather than anticipated. Same build, same recorder flags — except for
maneuver C, which wants the frames kept.

Card 1 worked: drag fits to R² 0.99 and the rate loops to 0.98–1.00 on held-out data. But
adding more terms to the model no longer helps — every physically plausible extra term
tested buys ≤0.015 held out and two of them make it *worse*. The model is at the ceiling of
**this data**, and the data has three specific holes. More flying is the only thing left
that moves it.

## The gaps, with the numbers that motivate them

1. **The low-throttle thrust curve rests on 74 samples.** 65% of the training data sits in
   throttle 0.25–0.30 and 0.4% sits below 0.10. The fitted curve `T/m = −6.08 + 59.59·thr`
   crosses zero at throttle 0.102 and goes *negative* below it, so `plant.py` clamps it at
   zero by hand. That clamp is the only unmeasured hack in the model, and it is exactly
   what a policy hits every time it chops throttle into a descent.
2. **`kz = 0.0364` is the least independent number in `plant.json`.** It is the one
   coefficient fitted jointly with another, because in racing flight throttle rises exactly
   when body-z speed does. That collinearity is what held the first attempt at this fit to
   R² 0.04; it is now *modelled* rather than *broken*, which is not the same thing.
3. **No lap has ever been completed.** `race_finish_time_ns` is −1 in all eight sessions
   and `active_gate_index` never got past 1 — one gate, twice, ever. Card 1's maneuver 8
   was never flown to a finish, so the held-out set is "ordinary flying", not "a lap", and
   there is no data at all on what passing a gate looks like.

## A. Apex reads — the one to fly

**~60 s. Do not skip this even if you skip the rest.**

At the top of a vertical arc the body-z velocity `w` passes through zero, and there the
vertical drag term `kz·w|w|` vanishes *identically*. So at that instant `−a_z = T/m` — a
direct read of the thrust curve with no regression, no drag model, and nothing for the
throttle to trade against. In the entire training set, 17 samples have throttle below 0.10
and `|w|` under 1 m/s. This maneuver manufactures them on purpose.

The whole maneuver is: **climb, then park the throttle low and coast over the top.**

    hover -> hold UP ~0.5 s (to 0.50) -> hold DOWN to the test throttle
          -> hands off 1.5 s -> F10 back to hover

| test throttle | UP key | DOWN key | `w = 0` at | still throttle first | envelope | recover at |
|---|---|---|---|---|---|---|
| 0.05 | 0.5 s | 0.9 s | 1.7 s | 0.3 s | 6.4 m | 10.0 m/s |
| 0.10 | 0.5 s | 0.8 s | 1.7 s | 0.4 s | 5.5 m | 9.4 m/s |
| 0.15 | 0.5 s | 0.7 s | 1.9 s | 0.7 s | 5.9 m | 5.3 m/s |
| 0.20 | 0.5 s | 0.6 s | 2.4 s | 1.3 s | 7.2 m | 0.8 m/s |
| 0.00 | 0.7 s (to 0.60) | 1.2 s | 2.4 s | 0.6 s | 14.1 m | 8.3 m/s |

Two reps each. Times are from the throttle *stopping*, and the envelope is the full
climb-plus-fall excursion, so start each arc that much clear of both floor and ceiling.

**1.5 s of hold, not more.** The measurement is finished the instant `w` crosses zero;
holding 3 s instead turns a 6 m excursion into 26 m and recovers at 15 m/s for no extra data.

Throttle 0.00 needs a taller pop to leave any margin, so fly it only if you have the height.
The model currently predicts *identical* zero thrust for 0.00, 0.05 and 0.10, so disproving
that at 0.05 is just as good.

**Fly this with the levelling assist OFF.** Under the assist the throttle keys slew *away*
from hover and snap back the moment you release, so parking at 0.05 is impossible. In plain
acro the throttle is a held absolute that stays exactly where you leave it, which is the one
thing this maneuver needs. Use `F11` to level the drone first if you like, then toggle it off
before the arc.

**You do not need to hit the throttle numbers.** `cmd.csv` records what you actually sent,
and the analysis reads whatever value you were parked at. What matters is that the throttle
is **stopped and untouched** before `w` crosses zero — the last column above is how much
still time you get, and it is 0.3 s at worst. A throttle still moving through the crossing
re-introduces the exact correlation the maneuver exists to break, and the sample becomes
worth no more than the 16 939 already recorded.

Throttle slews at 0.5/s (`THRUST_SLEW`), so "snap" is not available and the table accounts
for that — the entry is deliberately a ramp. Nothing here is timing-critical: get low, let
go, wait.

Two gaps for the price of one: each arc sweeps `w` from about +6 m/s to −10 m/s at a *fixed*
throttle, so fitting `−a_z` against `w|w|` **within a single arc** reads `kz` with thrust as
a plain constant offset and no throttle correlation at all (gap 2). Covering both signs of
`w` also isolates the climb/descent drag asymmetry, which was the only extra term that
improved held-out fit.

Those timings come from the model being tested, so treat them as a starting point, not a
spec. The measurement is valid wherever the apex actually lands.

## B. Vertical terminal runs — only if the hangar is tall enough

**~30 s.** Level, straight up at full throttle until vertical speed stops rising, then zero
throttle straight down until the descent stops accelerating. Predicted 34.7 m/s up and 16.4
m/s down, which takes **27 m of clear run to reach 90% of the climb** (36 m for 95%) and 23 m
for 90% of the descent — ramp included, so those are honest numbers for the real throttle.

If the ceiling will not allow it, **skip it** — maneuver A already gives `kz`. But if you
fly it, report the height you had and whether the speed visibly plateaued. A run that never
reached terminal, recorded as though it did, is worse than no run: it biases `kz` low and
nothing downstream can tell.

## C. One completed lap

**~60–90 s.** Fly it conservatively. The artifact is a *finished* lap, not a fast one — if
it ends in a collision or a reset, press `F12` and go again.

Record this one **with the JPEGs kept** (keep `--no-view`, drop `--no-frames`). A finished lap
with pose telemetry attached is the only labelled perception data this project would have,
and it costs little: `--no-frames` only skips the *disk write*. Reassembly of the frame
stream happens either way — every one of the eight existing sessions reassembled ~30 fps
regardless of the flag — so dropping it adds a write, not a new load on the control loop.

## D. Worth ten minutes: reproduce the session that does not close

Re-fly 20 s of what `20260731-130744` was doing — early free flight, no card, up to 30 m/s —
and run the referee on it. That session is the only one where the kinematic identity fails
(median residual 1.45 m/s²) and the cause is still unknown, but writing this card narrowed
it. **The IMU ran at half rate for the four morning sessions** (31.8–34.6 Hz) and at
60.6–61.6 Hz for the four afternoon ones, with `ATTITUDE` and `ODOMETRY` unchanged at ~112
and ~71 Hz throughout — and referee residuals split on exactly that line: 0.110–0.216
morning versus 0.029–0.091 afternoon. Frame rate was ~30 fps in all of them, so frames are
not the discriminator, and nothing in `teleop.py` requests a message interval.

So `130744` is the extreme of a systematic tier difference, not a lone mystery. That is not
a complete explanation — at 34.6 Hz `131305` still reaches 30 m/s and closes fine, so rate
alone does not produce a 1.45 — but it does mean the re-fly is a real test rather than a
shot in the dark: fly the same thing at today's 61 Hz. If it closes, the morning rate was
implicated and the exclusion can be retired. If it still fails, the cause is in the flying
or the sim, and that is worth knowing before VQ2 depends on it.

## Before flying

Card 1's pre-flight check was "the CSVs are non-empty". That passes on data the referee
later rejects, which is how `20260731-130744` got recorded. Record 20 s, stop, and run:

    python3 pilot/control/sysid_frames.py pilot/sessions/<new-session>/

Require **PASS** before flying the rest. Verified both ways: it passes on `143025` in under a
second and it *fails* on `130744`, so this is the check that would have caught the bad session
at the pad instead of a day later. **Roll, pitch and turn during those 20 s** — the referee
discriminates between candidate conventions using attitude variety, so a straight-and-level
sample can pass on too thin a margin to mean anything. Read the margin it prints, not just
the verdict.

`F12` between every maneuver *and* between reps — the arcs are about 4 s each and markers are
the only thing that makes twenty of them findable afterwards. Mid-session `SIM_RESET`s are
fine now: `sysid_data.py` detects the boot-clock restart and splits on it, so do not reflow
the card around them. Ignore the `armed` flag; it reads 0 through whole flights on this build.

## After

Refit. The apex samples should let `sysid_fit.py` replace the hand clamp with a measured
low-throttle curve and refit `kz` from the arcs instead of jointly with thrust, then
`sysid_replay.py` gets a real lap to be validated against. If the apex reads disagree with
the current curve near hover — where the existing data is thick and the fit is trustworthy —
suspect the new data, not the fit, and check the throttle was actually still.
