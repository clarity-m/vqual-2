# producer.py — the perception side of the frozen interface

`pilot/producer.py` fills `pilot/interface.py`'s `Observation` (OBS_DIM=73) at frame rate
from the three permitted inputs: camera frames, `HIGHRES_IMU`, and the race-status packet.
Control (the `Policy` in `interface.py`) consumes the Observation and returns an `Action`;
neither half imports anything else from the other.

## The API control sees

```python
from producer import Producer
prod = Producer()                      # loads gatenet colab-v3-rot + conf head colab-conf-rot

obs = prod.step(frame_bgr, imu_rows, race, t_s)   # -> interface.Observation
vec = obs.to_vector()                             # 73-dim float32, masking done by interface
yaw = prod.auto_yaw_rate(obs)                     # AUTO_ATTENTION servo (canonical signs,
                                                  # rad/s; link layer applies the -1 mirror)
```

* `frame_bgr` — 640×360 BGR array, or `None` on a frame-less tick (IMU still integrates,
  tracks coast).
* `imu_rows` — every RAW `HIGHRES_IMU` row received since the last step, as dicts with
  `t_wall_ns, xacc..zacc, xgyro..zgyro` (exactly `imu.csv`'s columns) or tuples
  `(t_ns, ax, ay, az, gx, gy, gz)`. The gyro −1 mirror is applied INSIDE the producer's
  `ImuFilter.push` — that call is the read side of the link layer per CONVENTIONS.md. Do
  not pre-flip.
* `race` — dict: `active_gate_index`, `race_time_s`, `armed`, `n_gates_total` (17),
  `t_since_collision_s`, `collision_episodes`. Missing keys default sanely.
* `t_s` — seconds on any consistent clock (used for staleness only).

Constructor flags: `use_net=False` (contour-only fallback, no torch import),
`use_compass=False` (no skylight; SEARCH degrades to a plain sweep).

## Wiring into teleop's link layer (live)

`teleop.py`'s telemetry thread already owns all three streams:

* frames: the vision-receive thread hands the decoded JPEG (`cv2.imdecode`) to
  `prod.step` at frame arrival; run the producer on the vision thread, control on its own.
* IMU: buffer each `HIGHRES_IMU` message as the raw tuple; hand the buffer to the next
  `step` and clear it (the recorder's `imu.csv` row is exactly the right shape).
* race: `Telemetry.snapshot()` fields map 1:1 (`active_gate`, `race_time_s`, `armed`,
  collision episodes are already episode-ised there).
* Action out: when `action.yaw_mode == YawMode.AUTO_ATTENTION`, replace `action.yaw_rate`
  with `prod.auto_yaw_rate(obs)` BEFORE the link layer encodes (the link layer then applies
  the sign mirror exactly as for any commanded rate).

## Replay / validation

```
python3 pilot/producer.py --selftest                     # frame-convention + PnP + vector checks
python3 pilot/producer.py --replay <session-dir-or-name> [--video] [--limit N] [--stride N]
python3 pilot/producer.py --replay <session> --paircheck # static-pair separation check
python3 pilot/producer.py --replay <session> --no-net    # contour-only
```

Outputs land in `pilot/perception/producer_runs/<session>/`:
`obs.npy` (N×73), `diag.csv` (per-frame current-gate diagnostics + attitude_conf +
|gyro|), `strip.png` (12 annotated frames sampled across regimes: far / mid / near /
fallback / coasting / ribbon-only / nothing), `replay.mp4` (with `--video`), `report.txt`.
The console report prints: valid/pose_valid rates, staleness, per-crossing range
monotonicity, coast→reacquire miss distances (the rotation-coasting check), attitude_conf
under high body rates, the per-stage latency table, and the map frame anchor fit.

## What is real vs stubbed

| field | status |
|---|---|
| gates[0..2] pos/normal/validities | REAL — detector + gatenet(+conf, p=0.082) + IPPE pose, 3 rejection layers, degradation ladder per the contract |
| gate coasting | REAL — rotation-coasted through the gyro each step (`p ← p − ω×p·dt`); NOT translated (drag velocity has no magnitude without the speed fit) |
| ribbon | REAL but deliberately simple — cyan-mask row-binned direction samples; not load-bearing by contract |
| own.gyro/accel/roll/pitch/attitude_conf | REAL — streaming complementary filter (skylight's validated alphas); conf degrades with ‖a‖−g mismatch |
| own.vel_bearing | REAL — drag-antiparallel algebraic read; invalid below 0.35 m/s² horizontal specific force |
| own.speed_est / speed_conf | REAL since 2026-08-02 — drag inversion `|v_xy| = sqrt(|a_xy|/k)`, k=0.0425/m, published at producer.py:2050. IN-PLANE (body xy) speed only; the body-z component is not observable |
| attention + yaw servo | REAL — R_COMMIT=4 m handoff, informed SEARCH via map+compass when the frame anchor is resolved |
| compass / map lookahead | INTERNAL ONLY — never enters the Observation; mirror `s` and offset `c` of the map frame are fit online from co-measured gate pairs and stay unresolved until ≥2 distinct pairs have been seen |

Thresholds and their provenance are constants at the top of `producer.py`.
