# VQ1 telemetry — system-ID dataset

Recorded 2026-07-31 against the **VQ1 build** (`AIGP_VQ1_3391`) with `pilot/teleop.py`,
flying the rate-only acro path that VQ2 uses. VQ1 and VQ2 have identical physics and gate
dimensions (all three spec revisions diffed; only §4.5 Telemetry changed), so this is
valid plant data for VQ2 — with the advantage that VQ1 does not block pose telemetry.

**Telemetry only. No frames.** The JPEG streams are gigabytes and are perception's
problem, not the plant's; they stay out of git. `frames.csv` is included and lists frame
IDs and timestamps, so it still tells you when frames exist and how they line up in time.

## What is here

| session | dur | cmd rows | thrust range | max speed | what it was |
|---|---|---|---|---|---|
| `20260731-130744` |   71 s |  3 031 | 0.00–0.52 | 30.0 m/s | early free flight |
| `20260731-131222` |   30 s |  1 435 | 0.00–0.38 | 14.8 m/s | short hop |
| `20260731-131305` |  479 s | 20 442 | 0.00–0.57 | 29.9 m/s | long mixed flying |
| `20260731-132659` | 1196 s | 56 685 | 0.00–0.28 |  8.7 m/s | long, slow, low thrust |
| `20260731-135735` |   19 s |      0 | —         | —        | `--listen` check, no commands |
| `20260731-143025` |   35 s |  1 733 | 0.00–0.29 | 20.6 m/s | **rate doublets** |
| `20260731-144815` |   48 s |  2 394 | 0.00–0.34 | 21.2 m/s | **isolated single-axis taps** |
| `20260731-150712` |  303 s | 15 134 | 0.00–0.86 | 34.3 m/s | **thrust excursions, cruise ladder, skids** |
| `20260731-195307` |  199 s |  9 909 | 0.00–0.77 | 15.9 m/s | lap w/ resets, 5 gates. **assist ON** |
| `20260731-203428` |   18 s |    859 | 0.00–0.28 |  4.1 m/s | yaw taps (`e`/`q`). **assist ON** |
| `20260731-204841-vq1-lap-slow` | 98 s | 4 868 | 0.00–0.81 | 17.8 m/s | **clean 6/6 lap, 1 contact.** assist ON |

The last three also carry frames (5 962 / 531 / 2 937), which the earlier sessions do not.

### Card 2 and later (recorded 2026-07-31 late / 2026-08-01, `pilot/sysid_card2.py`)

| session | dur | cmd rows | thrust range | max speed | what it was |
|---|---|---|---|---|---|
| `20260731-233219` |  152 s |  5 910 | 0.00–1.00 | 41.1 m/s | maneuver D attempt. **Excluded** |
| `20260801-004337` |   21 s |  1 014 | 0.00–0.40 |  0.0 m/s | preflight, never left the pad. **Excluded** |
| `20260801-004403` |   28 s |  1 367 | 0.00–0.40 |  8.7 m/s | preflight, referee PASS |
| `20260801-004843` |   97 s |  4 827 | 0.00–0.50 | 11.4 m/s | **card 2 A — apex arcs** |
| `20260801-005059` |   21 s |  1 039 | 0.00–1.00 | 31.1 m/s | **card 2 B — terminal runs** |
| `20260801-005518` |   21 s |  1 038 | 0.00–1.00 | 31.1 m/s | identical rerun of `005059`. **Excluded** |
| `20260801-005715` |   21 s |    810 | 0.00–0.69 | 20.5 m/s | card 2 C — crashed at gate 0 |

`20260731-233219` and `20260801-005715` carry frames (2 397 / 508 on this machine).

**Three of these are excluded from the fit** and `pilot/control/plant.json` records why:
`233219` because the kinematic referee does not close (median 2.01 m/s², the
`LOCAL_POSITION_NED` velocity repeating bit-identically in 43% of rows — the same
recording artefact as `130744`); `005518` because it is the same open-loop script as
`005059` run twice and agreeing to 0.1 m/s, so counting it twice would double-weight one
measurement; `004337` because it has no usable samples at all and its wall→sim clock fit
is degenerate (±1382 s of spread over 20.5 s of recording).

**`20260731-204841-vq1-lap-slow` is the held-out completed lap** — all six gates, a single
collision episode, no resets. Do not fit on it.

## The levelling assist, on the three newest

`cmd.csv` records what actually went out over MAVLink (`teleop.py:1018` writes post-assist
rates), so these are valid input-output data. But under the assist the commands are
generated *from the state* by an outer P loop clamped at `LEVEL_MAX_RATE` = 3.0 rad/s, so
the excitation is smoother and smaller than the numbers suggest — session 195307 tops out
at p99 ≈ 2.3 rad/s on every axis. **Identify the rate loop from the three assist-off cards
(`143025`, `144815`, `150712`); use these three as coverage and validation.** Yaw is never
touched by the assist, on any session.

The three in bold are the deliberate system-ID cards; `pilot/SYSID.md` says what each
segment was for. The rest is ordinary flying, useful as extra coverage and as held-out
data.

Per-session files: `imu.csv`, `cmd.csv` (command paired with the response that
followed), `actuators.csv` (motor outputs, 95 Hz), `race.csv`, `collisions.csv`,
`frames.csv`, `events.jsonl`, and the VQ1-only truth streams `attitude.csv`,
`position.csv`, `odometry.csv`.

**`imu.csv` is not one rate.** The four morning sessions (`130744` … `132659`) record
HIGHRES_IMU at **31.8–34.6 Hz**; the four from `135735` on record it at **60.6–61.6 Hz**.
`ATTITUDE` (~112 Hz) and `ODOMETRY` (~71 Hz) are unchanged across all eight, and frames run
~30 fps in all eight, so this is not frame load. Nothing in `teleop.py` requests a message
interval, so what changed around 13:30 is not recorded. It matters because kinematic-referee
residuals split on the same line — 0.110–0.216 median for the morning sessions against
0.029–0.091 for the afternoon ones — so **prefer the 61 Hz sessions for anything
timing-sensitive**, and weight by real `time_usec` deltas rather than any nominal period.
(The 47.7 Hz figure quoted in `../SYSID.md`'s closing note matches neither tier; treat all
three numbers as load-dependent rather than fixed.)

## Read this before fitting anything

**The truth streams are each wrong on a different axis.** Refereed against gravity on a
parked drone:

    truth_roll  =  ATTITUDE.roll      # == -ODOMETRY.roll
    truth_pitch =  ODOMETRY.pitch     # == -ATTITUDE.pitch
    truth_yaw   = -ATTITUDE.yaw       # CONFIRMED 2026-07-31, three external referees

**`pilot/CONVENTIONS.md` is the single source** — per-stream mirror table, provenance, and
what is still unverified. Two things before you fit anything: `truth_yaw` was asserted here
before it had been demonstrated and has since been confirmed; and **`LOCAL_POSITION_NED` is
NOT mirrored**, so the failure mode is a half-correction, not a missing one. Prefer its
velocities over `ODOMETRY`'s, which appear to be built on the mirrored yaw.

Fit against raw `ODOMETRY.roll` or raw `ATTITUDE.pitch` and the model is mirrored, silently.

**The simulator's whole body-rate convention is mirrored versus MAVLink NED** — commands
*and* gyro. Commanded rate against measured gyro correlates at +0.96 and proves nothing,
because both ends carry the same mirror. Only a referee outside the convention settles it
(`pilot/camreferee.py` did).

Full detail: `pilot/NOTES.md`, `pilot/control/README.md`.

## The split, as used

**`20260731-131305` is held out** (2026-07-31): nine minutes, wide thrust range, near-30
m/s, and ordinary flying rather than a maneuver card, so it tests whether a model
generalises past the excitation it was fitted from. It has been held out across every
fit so far, which is what makes their numbers comparable — keep it that way, or the
replay ends up validating on its own training data.

**The fit set is discovered, not listed.** `sysid_fit.py` takes every session directory
on disk that is neither held out nor explicitly excluded, so a new recording is picked up
by dropping it in this folder. As of 2026-08-01 that is 13 sessions, 22 epochs, 40 604
samples — up from the 5 sessions and 23 261 samples the first two fits used. What is
*removed* stays an explicit, justified list in `sysid_fit.EXCLUDED`; what is added is
just data.

**The levelling assist is detected, not remembered.** The rate loop can only be
identified from open-loop sticks, so assist-on sessions are dropped from that stage while
still contributing force samples. Sessions recorded since 2026-08-01 write a
`level_assist` event into `events.jsonl` and `sysid_fit.assist_on()` reads it; the three
older assist-on sessions have nothing to detect, so they are named in
`ASSIST_ON_LEGACY` — that list exists only for them and should not need to grow.

**`20260731-130744` is excluded from both.** It is the one session where the kinematic
referee does not close: median residual 1.45 m/s² against 0.007–0.22 everywhere else. Cause
still unknown, but partly narrowed — it is a morning session, and the whole morning tier
runs the IMU at half rate and carries 2–8x worse referee residuals (see above). That does
not fully explain it: `131305` shares the low rate, reaches 30 m/s, and closes fine, so
sample rate alone does not produce a 1.45. Card 2 maneuver D in `../SYSID.md` is the test.

## Two things about these files that are not obvious

* **The sim boot clock restarts at every `SIM_RESET`, mid-session** — six times in
  `20260731-150712`, four in `20260731-131305`. `t_wall_ns` keeps running, so on the wall
  clock a session looks like one continuous flight while actually splicing separate runs
  across a teleport back to the pad. Split on the device stamp before differentiating
  anything (`pilot/control/sysid_data.py` does).
* **`cmd.csv`'s `armed` column is the HEARTBEAT flag and reads 0 through whole flights** on
  this build. Epoch 0 of `20260731-150712` reaches 34.3 m/s at throttle 0.86 with `armed`
  never true. Do not use it to find the flying stretches; use thrust and speed.
