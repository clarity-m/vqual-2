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

The three in bold are the deliberate system-ID cards; `pilot/SYSID.md` says what each
segment was for. The rest is ordinary flying, useful as extra coverage and as held-out
data.

Per-session files: `imu.csv` (~61 Hz), `cmd.csv` (command paired with the response that
followed), `actuators.csv` (motor outputs, 95 Hz), `race.csv`, `collisions.csv`,
`frames.csv`, `events.jsonl`, and the VQ1-only truth streams `attitude.csv`,
`position.csv`, `odometry.csv`.

## Read this before fitting anything

**The truth streams are each wrong on a different axis.** Refereed against gravity on a
parked drone:

    truth_roll  =  ATTITUDE.roll      # == -ODOMETRY.roll
    truth_pitch =  ODOMETRY.pitch     # == -ATTITUDE.pitch
    truth_yaw   = -ATTITUDE.yaw

Fit against raw `ODOMETRY.roll` or raw `ATTITUDE.pitch` and the model is mirrored, silently.

**The simulator's whole body-rate convention is mirrored versus MAVLink NED** — commands
*and* gyro. Commanded rate against measured gyro correlates at +0.96 and proves nothing,
because both ends carry the same mirror. Only a referee outside the convention settles it
(`pilot/camreferee.py` did).

Full detail: `pilot/NOTES.md`, `pilot/control/README.md`.

## Suggested split

Nothing here has been designated held-out yet. Pick one session, do not fit on it, and
validate against it — `20260731-131305` is a reasonable choice: nine minutes, wide thrust
range, near-30 m/s, and it is ordinary flying rather than a maneuver card, so it tests
whether the model generalises past the excitation it was fitted from.
