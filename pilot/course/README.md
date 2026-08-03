# VQ2 course geometry — drop-in package for the training environment

**For Alex.** Copy this whole folder. One line to use it:

```python
import sys; sys.path.insert(0, 'pilot/course')
from course_vq2 import load, sample, gate_corners, gate_normal, sanity_check
```

`load()` gives the nominal measured course; `sample(seed)` gives one domain-randomized
course drawn at the size of our real measurement error. `numpy` is the only dependency.

Files: `course_vq2.json` (the data), `course_vq2.py` (loader + sampler + geometry),
`course_samples.png` (the randomization envelope), `build_course.py` / `build_tilt.py`
(regenerate the JSON from the perception map), `verify_course.py` (the checks below),
`ROUTE_REWARD_SPEC.md` (proposed dense progress reward on a solver-generated reference
route over deformed courses — a training-time reward change, Claire's proposal 2026-08-02).

> ### ⚠ CHANGED 2026-08-02 — regenerate against this file, do not reuse an older copy
> **Edge 10-11 was wrong and has been re-measured.** It read 33.84 m; it is 16.49 m.
> The old rows were a real measurement of the *wrong pair* — they saw gate 12 at the far
> end, from a single vantage, so gate 11 and the whole tail 12-16 sat about 15.6 m too
> far along-course. Consequences for anyone training on this geometry:
>
> * **Course length 250.93 m, was 268.22 m — 17.29 m shorter.**
> * Gates 11-16 all moved: gate 11 by 15.6 m, gates 12-16 by 14.6-15.6 m.
> * Gate 11 also dropped 1.97 m in z (its height now comes from a measured signed
>   dz of -3.66 m, not from the 10-12 / 11-12 pair alone).
> * There is a **new edge 10-12** (34.97 m) and 10-11-12 is now a **closed triangle**
>   (0.46 m closure on a 71 m perimeter). 10-11 and 11-12 stopped being bridges.
> * Per-gate sigma for 11-16 *fell* (2.11-2.38 -> 1.98-2.18 m) because that corner is
>   now braced rather than a bare chain.
>
> How it was caught and settled: `pilot/perception/NOTES.md` (2026-08-02, edge 10-11) and
> `map_vq2.json` -> `status.CORRECTION_2026_08_02_edge_10_11`.

---

---

## READ THIS FIRST — the honest limits

1. **This is a RELATIVE map. There is no absolute origin and no absolute heading.**
   It was measured from camera frames, gravity and a ceiling-light compass — never from
   the sim's pose stream, which VQ2 blocks. In practice the aircraft's start pose defines
   the frame you will actually fly in. **A policy may not consume world coordinates.**
2. **The whole course may be rotated by a multiple of 90°** relative to the sim world.
   The ceiling grid we take direction from is four-fold symmetric and we never resolved
   which quadrant is which. Shape is known; absolute orientation is not.
3. **Positions accumulate error along the chain.** Gate 0 is the gauge, so σ grows from
   0 at gate 0 to 2.4 m at gate 16. That is a *gauge* statement, not extra ignorance:
   the step from one gate to the next is known to 0.3–0.9 m everywhere
   (`sigma_xy_local_m`). Use local geometry, not absolute coordinates.
4. **Nine of the twenty edges are BRIDGES** (0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 9-10, 15-16) — they sit on a chain with no second
   path, so no loop in our data can check them. Their σ is inflated by hand, not
   verified. The braced parts (7-8-9, **10-11-12** since 2026-08-02, 12-13-14-15) are the parts we
   can actually prove: their leave-one-out distance error is 0.12–0.71 m.
5. **Edge 1-2 is contested.** The accepted 8.32 m rests on 5 rows from one flight; a
   refused 34-row channel reads 13.0 m. Exported as `alt_hypothesis_d_m` and drawn only
   when you pass `include_alt_hypotheses=True`.
6. **Gate plane YAW is refused at gates 8, 12 and 13**, and is approximate everywhere
   else (MAD 2.4–17.9°). **Do not build anything that needs an exact gate plane.**
7. **Heights are the most reliable part of the file.** Gravity-referenced, height-solve
   residual ~0.00 m, per-pair dz MAD 0.03–0.44 m.

---

## The frame — stated exactly, because a wrong axis is silent

Right-handed, metres, `z` up.

```
                       +y   (toward the column row numbered 21…29)
                        ^
                        |
   gate 16  <-----------+-----------> gate 0 = ORIGIN (0,0,0)
   (x = -217 m)         |             (x = 0)
        RACE DIRECTION  |
                        v
                       -y   (toward the column row numbered 12…20)

   +z is UP, gravity-referenced (accel/gyro filter). Gate 0's height is the zero.
```

* **Origin** — gate 0's centre.
* **+x** — along the hangar, pointing **back toward the start/pad**. The race runs
  0 → 16 in the **−x** direction; gate 16 sits at x = −217.1 m. Landmark tie:
  +x is the direction of **increasing station number on the 12–20 column row**
  (corr(x, sketch station) = +0.996 over all 17 gates).
* **+y** — across the hangar, toward the column row numbered **21–29** — the
  "Station 26 / 27 / 28" side (corr(y, sketch across) = +0.929). Equivalently, with z up
  and travel along −x, **+y is on the right-hand side of the direction of travel**.
* **+z** — up. Gravity comes from the IMU filter, so heights are independent of
  everything else in the file.

**"y = east" cannot be answered.** We have no compass to the sim world, only to the
ceiling grid, and only mod 90°. The axes above are the map's *sketch-aligned* axes:
the levelled ceiling-grid frame rotated by `grid_to_sketch_rotation_deg` = 97.39° and
y-mirrored. Use the column-row landmarks, not a cardinal direction.

Gate bearings in the JSON follow the same convention:
`bearing_deg = degrees(atan2(dy, dx))`, so
`p[b] = p[a] + d_horiz_m * [cos(bearing), sin(bearing)]` reproduces the layout exactly.

---

## What is MEASURED vs what is ASSUMED

| Quantity | Status |
|---|---|
| Inter-gate distance, 19 pairs | **measured** — accepted medians over 5–556 rows per pair |
| Inter-gate horizontal bearing | **measured** — ceiling-light compass, ~0.4°/frame, cross-session agreement median 1.05° |
| Signed height step, all 19 pairs | **measured** — gravity-referenced |
| Gate order 0→16 | **measured** — from `active_gate_index` crossings; certain |
| Aperture 1.5 m inner / 2.7 m outer | **spec** (VADR-TS-003), exact |
| Gate plane yaw, 14 gates | **measured but noisy** (MAD 2.4–17.9°) |
| Gate plane yaw, gates 8 / 12 / 13 | **NOT MEASURED** — refused; `yaw_deg: null` |
| 16 gates are vertical planes | **measured** — residual 1.5–4.0°, exported as `tilt_sigma_deg` |
| Gate 9 tilted 21°, top leaning toward azimuth 130° | **measured** — direction well determined; magnitude ±5° (obliquity-limited) |
| Absolute position / heading in the sim world | **ABSENT** |
| Gate thickness, mounting posts, obstacles | **ABSENT** — nothing but gate centres and planes is in this file |

### Sixteen gates are vertical. **Gate 9 is tilted ~21°, and we know which way.**

> **CORRECTED 2026-08-02.** An earlier version of this file said no gate was measurably
> tilted and shipped gates **8 and 9** with `tilt_from_vertical_deg: null` plus a uniform
> 0–20° prior. That was wrong, and it was a *reduction* error, not a data error — the
> rows never changed. **Gate 8 is vertical (3.6°). Gate 9 is tilted.** If you built
> anything against the old file, the change that matters is that gate 9's plane is not
> perpendicular to the ground and its lean direction is now exported.

**What is exported.** Gate 9: `tilt_from_vertical_deg: 21.0`, `tilt_sigma_deg: 5.0`,
`tilt_clip_deg: [12, 30]`, and — the part a policy actually needs —
`tilt_lean_azimuth_deg: 129.7` (`tilt_lean_unit_xy: [-0.64, +0.77]`): the direction the
gate's **top** leans, i.e. mostly toward **+y** (the Station 21–29 column row) with a
component along the race direction. Every other gate: `tilt 0.0` with
`tilt_sigma_deg` 1.5–4.0°, which is that gate's *measured residual*, not a prior over
unknown geometry.

**Magnitude is the weak half, direction is the strong half.** Three estimators:

| estimate | channel | regime it is valid in |
|---|---|---|
| 21–25° | PnP normal elevation / twin-resolved 3-D axis (23.1°, residual 2.7°) | biased **up** head-on — and all our rows are head-on |
| 19.7° | the `gate8.png` screenshot, gate seen **edge-on**, PnP-free; two hangar columns in the same frame read 1.0–1.2° as in-frame controls | the *only* observation in the edge channel's valid regime |
| 13.2° | PnP-free slope fit `sin ψ = a·sin θ + b` over gate 9's flight rows | weak **lower bound** only: the same fit returns 19.0° on gate 3, which is vertical |

All three land in 13–25°; **none supports a vertical gate.** The exported 21 ± 5° is a
reconciliation, and the sigma is wide on purpose. **The limit is obliquity coverage, not
row count**: every gate-9 detection we have is a head-on approach (θ ≤ 19.2°), and no
strafe session passes gate 9. *One lateral pass at gate 9 closes this* — above θ ≈ 45°
the PnP-free edge channel measures the lean directly, with no model and no twin. (The
detector's aspect-ratio filter discards the most oblique views, so loosen it for that
pass.)

**The direction is solid**, and it was checked rather than assumed. The plane normal is
an axis whose sign is unobservable, but the top-lean vector `−n_z · n_horiz` is invariant
under `n → −n`; what is *not* invariant is the IPPE twin, which mirrors the normal about
the line of sight and so flips the lean by 180°. Resolved by RANSAC over both candidates
of all 130 near rows: the selected branch forms **one tight cluster (2.7° residual, p90
7.5°, 91.5% inliers, and 95% of rows kept IPPE's own preferred solution)**, while the
mirror branch **does not form an axis at all (24.1° residual)**. A mirror artefact cannot
do that — it moves with the vantage. Corroboration: three flights whose vantages differ
by 15° agree on the azimuth to **0.3°** (129.5 / 129.7 / 129.8), where an artefact would
have moved ~30°; and a lean must be perpendicular to the gate plane, which the map's
independently measured plane azimuth (132.4° mod 180, n = 382) confirms to 2.7°.

The lean azimuth inherits the frame's k·90° ambiguity relative to the *sim* world
(limit 2 above). Relative to the course's own axes and to the race line — which is what a
policy flies — it is exact.

`gate13.png` independently confirms gate 13 is vertical (its edge-on bar is parallel to a
hangar column).

### Methodological note — why the first pass got this wrong, twice

Worth reading before you re-derive **any** geometric quantity from
`gatetilt_rows.json`, because both mistakes were invisible in the number that was
reported and obvious once the rows were cut differently.

1. **Pooling.** The first pass took the median over all ~8 500 detections per gate. Most
   rows are far and small, where a 20° lean moves a 30 px quad by a couple of pixels: the
   tilt is geometrically invisible there and the estimator returns noise centred a few
   degrees above zero. Pooling therefore drags a genuinely tilted gate **down** to the
   noise floor while pushing every vertical gate **up** to it, and the two become
   indistinguishable. That is exactly the "every gate reads 1.3–4.9°, so nothing is
   tilted" non-result. Stratify by measurement quality and read the **trend**, not the
   level — a real property survives as views improve, noise shrinks:

   | gate | all rows | ≥60 px | ≥120 px | ≤15 m |
   |---|---|---|---|---|
   | **9** | 14.6 | **24.0** | **21.4** | **23.9** | ← *rises* with proximity |
   | 8 | 11.3 | 3.0 | 2.6 | 4.2 |
   | others | 2.6–10.6 | 0.9–6.1 | 0.9–7.3 | 1.9–11.9 | ← all *fall* toward vertical |

2. **Then stratifying on the wrong variable.** Fixing (1) finds the tilted gate but
   cannot *size* it. On this course "near and large" means **head-on**, and head-on is
   simultaneously where the PnP estimator is biased **up** and where the PnP-free edge
   estimator is **blind** (a gate leaning toward or away from the camera still projects
   its side edges as vertical lines). Selecting big close rows selects rows where both
   channels are at their worst, in opposite directions — which is how one reduction of
   this row set read 24° and another read 3°. The variable both blind spots depend on is
   **obliquity** θ, the angle between the viewing ray and the gate normal. Cut that way
   (`pilot/perception/gatetilt_obliquity.py`), gate 9's PnP does *not* decay with θ
   (24.5° → 22.0°, so the 24° is not a conditioning artefact) while its edge reading
   *rises* (3.1° → 6.4°, as `sin ψ = sin φ · sin θ` demands of a real tilt) — and at
   matched θ the control gates read 1.8–9.5° against gate 9's 24.5°.

**General form: stratify by the variable the *estimator's error* depends on, not by the
one that looks like data quality — and push control gates through the identical cut.**
The controls are what caught it. A fourth channel built during this pass (top/bottom edge
foreshortening, `gatetilt_foreshorten.py`) was **discarded for failing exactly that
test**: it fits 15–35° of tilt to *every* gate including known-vertical ones, so the
detector's quads carry a systematic top/bottom asymmetry. Its gate-9 numbers are used for
nothing. The file is kept because the control failure is the result.

### Gate faces and the column grid — an open conflict, not smoothed over

Claire reports that gate faces are parallel to the hangar columns, i.e. gate yaw is
quantized to the ceiling/floor/column grid. **Our measured yaws do not reproduce that.**
Taken mod 90°, the 14 accepted plane azimuths spread over 11–88° with no cluster at the
grid bin (deviations −37° to +44°). They agree far better with the **race-line bisector**
(median |error| 10.9°; under 6° at gates 1, 3, 6, 10, 14, 16). Every gate therefore
exports all three: `yaw_deg` (measured), `yaw_grid_bin_deg` + `yaw_grid_dev_deg` (nearest
grid-quantized value and the deviation), and `yaw_race_bisector_deg`. Given how noisy
IPPE azimuth proved on this course, the visual claim may be right and our azimuths wrong.
Either way: **do not depend on an exact gate yaw.** `gate_corners()` falls back to the
race-line bisector at 8/12/13 and says so.

---

## Per-gate table

`sigma_xy_m` is from gate 0 (accumulates); `sigma_xy_local_m` is the step from the
previous gate — the number a policy actually feels.

| gate | x | y | z | σ_xy | σ_xy local | σ_z | yaw° | tilt |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0 | -0.0 | -0.0 | 0.00 | 0.00 | 0.00 | 173.6 | vertical |
| 1 | -16.3 | 11.6 | 3.2 | 0.52 | 0.52 | 0.23 | 150.6 | vertical |
| 2 | -23.9 | 15.0 | 2.6 | 1.03 | 0.89 | 0.49 | 165.6 | vertical |
| 3 | -35.2 | 8.0 | 1.2 | 1.16 | 0.52 | 0.54 | 36.4 | vertical |
| 4 | -51.6 | -6.0 | 0.4 | 1.33 | 0.65 | 0.58 | 39.9 | vertical |
| 5 | -73.3 | -12.3 | 0.6 | 1.44 | 0.54 | 0.62 | 173.7 | vertical |
| 6 | -88.2 | -10.4 | 5.6 | 1.51 | 0.48 | 0.66 | 169.0 | vertical |
| 7 | -103.6 | -8.4 | 10.9 | 1.64 | 0.63 | 0.72 | 87.9 | vertical |
| 8 | -105.5 | 2.6 | 10.1 | 1.67 | 0.32 | 0.74 | **null** | vertical (3.6°) |
| 9 | -112.1 | 15.3 | 9.7 | 1.73 | 0.53 | 0.73 | 132.4 | **TILTED 21±5°, top leans az 130°** |
| 10 | -122.2 | 14.6 | 8.8 | 1.93 | 0.85 | 0.85 | 11.4 | vertical |
| 11 | -138.3 | 11.3 | 5.1 | 1.98 | 0.43 | 0.87 | 51.5 | vertical |
| 12 | -153.8 | -0.3 | 7.1 | 2.00 | 0.46 | 0.87 | **null** | vertical |
| 13 | -163.8 | -9.0 | 7.9 | 2.05 | 0.44 | 0.89 | **null** | vertical |
| 14 | -173.9 | -1.9 | 8.0 | 2.04 | 0.33 | 0.88 | 134.6 | vertical |
| 15 | -181.9 | 9.2 | 6.1 | 2.07 | 0.43 | 0.90 | 171.0 | vertical |
| 16 | -202.6 | 13.1 | 5.5 | 2.18 | 0.68 | 0.94 | 174.7 | vertical |

The course climbs 10.9 m from gate 0 to gate 7 (the 5-6 and 6-7 steps are +5.00 and
+5.31 m) and then stays high. Heights are real, not inferred from the sketch.

---

## Per-edge table (randomize here if you prefer edge space)

`sd`, `sb`, `sdz` are the sigmas the sampler draws from.

| edge | d (m) | MAD | n | sessions | bearing° | dz (m) | σ_d | σ_bear | σ_dz | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-1 | 20.00 | 0.74 | 86 | 4 | 144.7 | +3.25 | 0.60 | 1.20 | 0.23 | bridge |
| 1-2 | 8.32 | 0.66 | 5 | 1 | 155.7 | −0.62 | 1.20 | 2.67 | 0.44 | thin, single-session, bridge, **CONTESTED (rival 13.0 m)** |
| 2-3 | 13.25 | 0.59 | 79 | 2 | 211.7 | −1.44 | 0.60 | 1.83 | 0.23 | bridge, single-channel |
| 3-4 | 21.62 | 0.80 | 57 | 2 | 220.5 | −0.79 | 0.60 | 1.82 | 0.23 | bridge |
| 4-5 | 22.54 | 0.83 | 130 | 2 | 196.3 | +0.24 | 0.60 | 1.20 | 0.23 | bridge |
| 5-6 | 15.06 | 0.73 | 201 | 2 | 172.5 | +5.00 | 0.60 | 1.20 | 0.23 | bridge |
| 6-7 | 15.56 | 0.27 | 556 | 1 | 172.9 | +5.31 | 0.78 | 1.56 | 0.29 | single-session, bridge, single-channel |
| 7-8 | 11.40 | 0.59 | 99 | 2 | 100.5 | −0.87 | 0.40 | 1.20 | 0.15 | braced |
| 7-9 | 24.97 | 1.37 | 21 | 2 | 109.1 | −1.12 | 0.44 | 2.57 | 0.15 | braced |
| 8-9 | 14.48 | 1.20 | 100 | 3 | 118.0 | −0.49 | 0.40 | 3.28 | 0.15 | braced |
| 9-10 | 10.06 | 0.13 | 12 | 1 | 184.2 | −0.91 | 1.17 | 1.56 | 0.44 | thin, single-session, bridge, single-channel |
| 10-11 | 16.49 | 0.28 | 96 | 1 | 191.6 | −3.66 | 0.52 | 1.56 | 0.20 | single-session, braced, **re-measured 2026-08-02, four staged vantages** |
| 10-12 | 34.97 | 0.50 | 156 | 1 | 205.2 | −1.75 | 0.52 | 1.56 | 0.20 | single-session, braced, **re-measured 2026-08-02, four staged vantages** |
| 11-12 | 19.67 | 0.72 | 28 | 1 | 217.6 | +2.32 | 0.52 | 1.56 | 0.20 | single-session, braced |
| 12-13 | 13.45 | 0.20 | 14 | 1 | 220.6 | +0.81 | 0.78 | 1.56 | 0.29 | thin, single-session, single-channel, braced |
| 12-14 | 19.92 | 0.66 | 15 | 1 | 184.4 | +0.85 | 0.52 | 1.56 | 0.20 | single-session, single-channel, braced |
| 13-14 | 12.39 | 0.37 | 129 | 2 | 145.7 | +0.12 | 0.40 | 1.99 | 0.15 | single-channel, braced |
| 13-15 | 25.81 | 0.80 | 24 | 2 | 134.9 | −1.83 | 0.40 | 1.20 | 0.15 | single-channel, braced |
| 14-15 | 13.22 | 0.32 | 9 | 1 | 125.4 | −1.72 | 0.78 | 1.56 | 0.29 | thin, single-session, single-channel, braced |
| 15-16 | 21.01 | 0.50 | 446 | 1 | 169.4 | −0.59 | 0.78 | 1.56 | 0.29 | single-session, bridge |

There is deliberately **no 4-6 edge**: the map refused it after the direction solve showed
it re-measures 4-5 under a wrong name.

---

## How the sigmas were derived (they are not invented)

The map runs its own leave-one-out check: drop an edge, predict it from the rest of the
graph, compare. On the 8 edges that sit inside a braced loop (the only ones where this is
possible) the errors are **|Δd| rms 0.601 m** and **|Δbearing| rms 1.677°**. A LOO error
contains both the measurement error and the prediction error, so a single edge's own σ is
about `err/√2` = **0.42 m and 1.19°**.

Meanwhile the *formal* standard error of each pair median, `1.4826·MAD/√n`, runs
0.02–0.44 m — **smaller than the observed spread for almost every edge**. Row count is
therefore not what limits us; a systematic floor is. So the model is:

```
sigma_d   = max(0.40 m, 1.4826*MAD/sqrt(n))  ×  penalties
sigma_bear= max(1.20°, row_MAD, cross_session_spread) × penalties
sigma_dz  = max(0.15 m, 1.4826*MAD_dz/sqrt(n_dz)) × penalties

penalties: ×1.5 if n < 15 (thin) · ×1.3 if measured in one session only
           ×1.5 if the edge is a bridge (no loop can check it)
edge 1-2:  sigma_d forced to 1.20 m, plus an exported rival hypothesis
```

Per-gate σ is then **derived**, not assumed: 20 000 Monte-Carlo draws of the edge table,
each re-solved through the full chain, and the marginal spread of each gate reported as
`sigma_xy_m` / `sigma_z_m` (plus the local, previous-gate version).

**Not modelled** (state this in any writeup): edges from the same flight share that
flight's compass bias, so a whole *leg* can rotate together in a way the sampler never
draws; and the 2-3, 6-7 and 15-16 legs take their mod-90 quadrant from Claire's sketch
rather than from a shared measurement, so a wrong branch rotates that leg by 90° — a
discrete failure a Gaussian cannot express. **Treat the sampled envelope as a lower
bound.**

---

## How to randomize

```python
from course_vq2 import load, sample, gate_corners, gate_normal

nominal = load()
for episode in range(N):
    course = sample(seed=episode)              # a whole self-consistent course
    ...                                        # course.positions is (17, 3) metres
```

`sample()` perturbs **edges, then re-solves the chain** — it does not jitter gates
independently. That matters: the map measured edges, never coordinates. Independent
gate jitter would break adjacent spacings and the braced loops (which close to 1.2–1.6%
today) and would hand the policy a course that could not have produced our data. Solving
the chain keeps every sample geometrically self-consistent, and lets uncertainty
accumulate along the chain the way real chain-map error does.

Options:

* `sample(seed, include_alt_hypotheses=True)` — coin-flips the 1-2 rival value (13.0 m).
  Worth switching on for a fraction of episodes if the policy will ever fly 1→2.
* `sample(seed, randomize_yaw=False)` — keep nominal gate planes.
* Prefer edge space yourself? `course.raw['edges']` is the full table.

**Tilt in a sample.** Sixteen gates draw a small Gaussian about 0 with their own measured
residual (`tilt_sigma_deg`, 1.5–4.0°). Gate 9 draws N(21°, 5°) clipped to
`tilt_clip_deg` = [12, 30], and its lean azimuth draws N(129.7°, 5°) — so a policy trained
on `sample()` sees a gate 9 that is *always* leaning, always the same way, with the
magnitude uncertain. That is the correct shape of what we know. (The old file randomized
gates 8 **and** 9 uniformly over 0–20°, which taught a policy that gate 9 might be
vertical and that gate 8 might not be. Both are wrong.)

`course.tilt_lean_deg[g]` carries the lean azimuth (`None` where the gate is vertical and
the direction is meaningless). `gate_basis()` / `gate_corners()` / `gate_normal()` use it
to set the sign of the normal's elevation, so the corners you get for gate 9 lean the
measured way rather than an arbitrary way.

Geometry helpers:

```python
gate_corners(4, course)                      # (4,3) corners of the 1.5 m aperture
gate_corners(4, course, size_m=course.outer_m)   # the 2.7 m frame — the collider
gate_normal(4, course)                       # plane normal oriented along the race
```

Both take `yaw_fallback='bisector'` (default — face the racing line, an **assumption**)
or `'raise'` at gates 8/12/13.

---

## What a policy may NOT rely on

* Absolute world coordinates or absolute heading — they are not in this file and cannot
  be recovered from it. Fly gate-relative.
* Any particular global orientation: the course may be rotated by k·90° in the sim.
* An exact gate plane at **gates 8, 12, 13** (`yaw_deg` is null) — and, honestly, at any
  gate, since the measured yaws are 2.4–17.9° MAD and conflict with the visual
  grid-alignment claim.
* Gate **9** being vertical — it is not; it leans ~21°, top toward azimuth 130°. Nor on
  that magnitude to better than ±5°: every view we have of gate 9 is head-on. The lean
  *direction* is the reliable part.
* The 1-2 spacing being 8.32 m.
* Anything about obstacles, gate posts or the hangar interior — not measured here.

Safe to rely on: the race order, the aperture sizes, the height profile, local
gate-to-gate spacing to ~0.3–0.9 m, and the overall shape of the course.

---

## Verification (rerun with `python3 pilot/course/verify_course.py`)

```
nominal: worst edge 0.449 m, worst dz 0.270 m
100 samples, worst-edge reconstruction error:  med 0.758  p90 1.181  max 1.442 m
100 samples, worst-edge dz error:              med 0.344  p90 0.505  max 0.680 m

max |exported horizontal distance - map measured_pairs horiz_m| = 0.450 m  (edge 14-15)
max |exported dz - map pair_dz_up_m|                            = 0.270 m  (edge 11-12)
max |exported gate position - map positions_m|                  = 0.0000 m

gate 9 nominal: tilt 21.0 deg, top displaced 0.97 m horizontally toward azimuth 132.4 deg
other 16 gates, nominal tilt: max 0.0 deg
200 samples: gate 9 tilt 21.0 +- 4.9 deg (range 12-30), lean azimuth 130.3 +- 4.7 deg
200 samples: every other gate tilt max 12.0 deg (all clipped at 3 sigma)
```

Re-run after the 2026-08-02 **edge 10-11** correction. The dz discrepancy moved from
14-15 to 11-12 (0.27 m): 10-11 now carries a directly measured signed dz of -3.66 m that
must be reconciled with the 10-12 / 11-12 pair, which is what an over-determined corner
looks like. The earlier tilt correction still touches gate planes only, never coordinates.

The 0.450 m is not export loss — it is the map's own residual at 14-15, the least-braced
corner of the 12-13-14-15 quad where three measured edges cannot all be satisfied at
once. Exported positions are bit-identical to `map_vq2.json`.

`course_samples.png` shows the envelope: the top panel is the raw sample fan (uncertainty
growing away from the gate-0 gauge), the bottom panel the same samples rigidly aligned to
the nominal, which removes the gauge and exposes *local* shape uncertainty — visibly
wobblier through 7-8-9 and 12-13-14-15 (thin edges, drawn red) than along 4-5-6.
Course topology is preserved in every sample.

---

## Provenance

Built by `build_course.py` + `build_tilt.py` from
`pilot/perception/map_vq2.json` (`layout_directions_2026_08_02` and `measured_pairs`) and
`pilot/perception/gatetilt_rows.json`. The map itself comes from vision, gravity and a
ceiling-light compass over five flights — no pose stream, no sim ground truth. Full
derivation and every refusal: `pilot/perception/NOTES.md`.
