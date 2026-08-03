# perception — measured facts

Vision-side record. `README.md` next to this file says what this half **owns** and how it
is meant to be built; this file is what has been **measured**, and what it cost to learn.
Everything here is about what the camera sees. Non-visual course and protocol facts stay
in `../NOTES.md`, which remains source of truth for the project as a whole.

Same rule as the parent file: measured facts and binding decisions only, and each trap is
kept with the evidence that settled it.

## The course, as it looks

Indoor hangar — dark, lit signage, ceiling light strips, numbered support columns. Not
visually hostile; detection looks no harder than VQ1.

**VQ1 and VQ2 look nothing alike.** VQ1 is a dark city/blocks environment; VQ2 is the
hangar. The *gates* are visually identical — same orange `AI-GP` frames, same 1500 mm
aperture — so VQ1 recordings teach gate appearance, which transfers. They teach nothing
about hangar clutter or the white-ceiling-light false positive, which do not. Any detector
trained only on VQ1 must be pseudo-labelled onto VQ2 frames before it is trusted there;
that step is load-bearing, not polish.

* **17 gates** — count and provenance in `../NOTES.md`, cited here because the map is
  tested against it. Visually: bright orange/red, glowing against dark. **Open:** whether 17
  counts distinct physical gates or gates in race order — if the course revisits a gate those
  differ, and it changes what a correct map looks like.
* **The active gate is lit differently** (Claire-observed 2026-07-31): the gate you are
  flying to renders with a bright interior fill and a yellow ground glow, while every other
  gate is a plain orange frame. Paired with `active_gate_index` from the race packet this is
  **absolute gate identity, one gate per frame, sim-supplied** — not tracking, not inference.
  It does not matter that this may be a training-mode cue like the cyan corridor: the map is
  built offline in training and used as a prior. Not yet exploited by any code here.
* **Cyan guidance corridor** showing roughly the next 5 gates. Present in submission mode,
  not just training. **Absent from long stretches of real flight** — hence not load-bearing.
* **White ceiling lights** are the main false-positive risk; the **first gate blooms**.

Measured on the parked start frame (`../evidence/2026-07-30-startview.jpg`, OpenCV HSV,
hue 0..179):

| class | mask | share | mean S | mean V |
|---|---|---|---|---|
| orange gate | `(H<=12 or H>=170) & S>110 & V>110` | 2.0% | 202 | 241 |
| cyan path | `85<=H<=100 & S>110 & V>110` | 1.5% | 197 | 169 |
| white lights | `S<50 & V>200` | 0.9% | 3 | 238 |
| **station signage** | `S<70 & 95<V<195` | — | ~0 | **100–155** |

66.5% of the frame is V<40; the first three classes separate on saturation alone
(202/197/3), and the lights are additionally separated by position (centroid y=66). Bloom
cost only 1.7% of the gate bounding box, so the orange mask is near-solid — but fill
contours rather than trusting a filled mask.

**The signage row is the correction that cost a build.** `stations.py` first reused the
white-lights mask, on the assumption that signage and ceiling lights are the same colour
and can only be told apart by shape. They are not: the signage is **grey**, V ≈ 100–155,
while the lights sit above V=200. The lights are *brighter than the text*, so an upper
bound on V separates the known false positive on colour, and shape only has to clean up
what is left. Reusing the light threshold returned **zero** candidates on every frame.

## The Station columns

Every support column carries a unique number, in two forms, both legible at the recorded
640×360:

* a **large two-digit face number** on the upper column face — upright, roughly
  fronto-parallel, 15–30 px tall on a near column;
* **vertical `Station N` text** down the column side — larger glyphs (21–25 px) but rotated
  along the column axis.

**The two rows are numbered in opposite directions and a facing pair sums to 41.**
Confirmed on frames 500 and 900 of `20260731-222724` (`16|25`, `12|29`) and independently on
Claire's screenshots (`18|23`, `19|22`). Read both sides and a misread digit is *caught*
rather than silently placing a gate in the wrong bay. Stations run to at least 30, so the
hangar is roughly 40 stations long. Cruise pace: 7 stations in 19.29 s, 2.76 s/station,
constant after ~10 s of acceleration.

**Unique numbering solves data association**, which is the problem the map is currently
failing at (below). A gate seen beside Station 24 is *the* gate at Station 24, whether or
not any track survived the occlusion.

## Status

**Detector** (`detect.py`, `label.py`, `refine.py`) — recall 0.74 after falling back to the
2700 mm outer boundary, held-out centre error 1.8 px after the pose-stream refit, PnP range
+0.75 m.

**Map** (`mapbuild.py`) — builds, and is not yet usable geometry. On `20260731-222724`:
59 tracks, 283 edges, 35 nodes placed, edge spread median 1.88 m, MDS stress median 1.73 m.
Against the true **17** gates, 35 placed nodes is roughly **2 nodes per real gate**. Three
readings from the 2026-07-31 plots, kept because each says something different:

* **Fragmentation is the dominant error, and `MAX_GAP = 4` is the prime suspect.** A track
  closes after 4 frames — **0.13 s** — of no match. A near gate's frame sweeping across a
  far one, a gate clipping the image edge, or five detector misses in a row all end the
  track, and reacquisition mints a new node. 11 node pairs sit under 3 m of each other, all
  but one of them low-observation tracks.
* **Part of the map is not measured at all.** Nine nodes — including tracks 0 and 1, the two
  longest — lie *exactly* collinear across 25 m. Those are positioned by the Floyd–Warshall
  completion in `solve()`, which places unconstrained points along the geodesic, not by any
  distance that was observed. Their coordinates are an artifact.
* **The map is 35 m thick** (extents 66.7 / 42.4 / 35.3 m). If the course is anywhere near
  planar, the third axis is a direct measure of how wrong the reconstruction is.
* The measured-vs-reconstructed scatter is one-sided below ~35 m: distances are *shrunk*,
  not scattered. Symmetric noise would scatter. One-sided pull means the distance set is not
  embeddable in 3-D — which is what duplicate nodes for one physical gate would do.

**`stations.py`** — **works.** Glyph extraction → two-digit pairing → template classification,
no OCR dependency. Templates in `station_templates.npz`, built from `station_labels.json`
(94 pairs harvested from `20260731-222724`, the unambiguous ones hand-labelled by eye; all
ten digits covered, 4–52 examples each, closest template pair `5` vs `8` at cos 0.868).

Measured coverage: **38% of frames on `20260731-222724` yield at least one station number**,
16 distinct stations, reads concentrated on 24–29 where the flight went. On the larger and
faster `20260730-215221` coverage falls to **8%**, and on `20260730-174527` to **1%** — the
columns blur at speed, so coverage is a function of how the session was flown.

Two known limits: it reads *face numbers*, so it returns nothing when the only near column
presents its vertical text (frame 900 of `20260731-222724`: `12` breaks into four fragments
of 4–8 px); and reads with counts of 1–2 in a session (`7`, `15`, `30`) should be treated as
probable misreads until a second channel confirms them.

## THE MAP — `map_approx.json`

**There is a functional gate map, in station units.** Claire built a top-down sketch by hand
from aerial captures (`map-aerial/`, 2026-08-01): two rows of numbered columns and one red
bar per gate giving position *and* plane orientation. `readmap.py` extracts it — 17 bars
found, matching the gate count, with the station rows read from the sketch and cross-checked
by the sum-to-41 invariant and a rows-are-level test.

This is the piece vision could not supply, and the two halves are exactly complementary:
`mapbuild.py` measures inter-gate distances well but cannot say which gate is which;
the sketch knows which gate is which but has no scale.

**Coordinates are in station units, deliberately** — `stations.py` reads station numbers off
the camera, so a gate at `along=17.04` is directly comparable to a live reading with no
conversion. `along` is in **LEFT-ROW numbering**: a right-row column numbered N is level with
`along = 41 - N`, so Station 29 sits at along 12. Mixing the two directions is the easy
mistake; the first render of the map made it.

**Two things the sketch does not carry:**

* **Metric scale: 15.97 m/station** (p10 13.38, p90 18.49, n=15 labelled pairs over 6 gate
  pairs). Per-pair medians 0-1 15.62, 1-2 14.88, 2-3 16.33, 3-4 18.24, 2-4 18.65, 8-9 13.44 �
  no systematic structure left, so the residual spread is measurement noise.
  **RETRACTED: "the sketch is locally inconsistent by 1.65x".** I reported that pair 0-1
  implied 10.46 m/station against 17.31 for pairs among 1-4 and concluded Claire's drawing was
  locally wrong. It was my measurement. Claire spotted that in those frames **gates 1 and 2
  overlap in the image**, and confirmed independently � from flying VQ2 repeatedly and knowing
  the reset position � that the near gate really is gate 0. Two merged gates form one orange
  blob LARGER than a single gate's outer boundary, so the outer-boundary fallback reads it too
  big and places the far gate too close: 20 m where geometry demands ~26 m. That compresses
  0->1 from ~17 m to ~10.5 m. Dropping the 6 affected frames moves pair 0-1 to 15.62 and the
  overall spread ratio from 1.89 to 1.38.
  The tell was visible in the data and I missed it: gate 0's range varied 4.7 -> 9.6 m across
  the frames while the far gate stayed at 20-21 m in all of them. Two static gates cannot do
  that. **A pair of static objects whose measured separation changes with viewpoint is a
  measurement fault, and that check costs nothing.**
  **HEIGHTS** (gravity-referenced, no pose stream): 0-1 +1.19 m (n=8, spread 2.72), 2-3
  -1.12 (n=4, spread 1.06), 3-4 +0.66 (n=2), 1-2 -1.37, 2-4 +1.07, 8-9 -2.24 (n=1 each).
  Gate-to-gate height changes are of order a metre or two. These have NOT been recomputed
  with the overlap frames excluded.
  **DETECTOR FAILURE MODE � the guidance ribbon occludes the aperture edge** (Claire,
  2026-08-01, `20260730-184043/00023248`). The cyan ribbon is drawn OVER the gate's inner
  edge, so the aperture contour follows the ribbon's silhouette instead of the orange frame.
  Measured on that detection: 17.3% of the quad interior is ribbon-cyan and the fitted quad
  has edges 68.3 / 69.4 / **47.0** / 65.5 px � the contaminated side pulled ~30% short.
  **Resolution is NOT the fix.** The gate spans 69 px at 8.4 m, so it is not sampling-limited;
  more pixels image the same occlusion more sharply. The spec also fixes the camera at
  640�360, fx = fy = 320, so a detector tuned higher would not transfer to race time � it
  would be legitimate for offline mapping only, and would still not help here.
  What would: treat dilated cyan as UNKNOWN rather than background and drop those contour
  segments from the quad fit (the cyan class is already characterised and separates cleanly
  from orange); or fit lines to the clean sides and intersect, which is the partial-quad
  work already queued for clipped gates. This matters more than one frame � the ribbon marks
  the racing line, so it passes through the aperture of exactly the gate being flown at, and
  preferentially corrupts the nearest and most important gate.
    **NEW DETECTOR FAILURE MODE � merged gates.** Two gates aligned on the view axis merge into
  a single orange blob; the fallback then measures the pair as one gate. Distinct from the
  decoration and clipping failures. Not yet detected automatically; `quality()` does not catch
  it, because nothing about the blob is individually anomalous.
* **Superseded: metric scale from 5 pairs** (`labelgates.py`,
  `labels_222724.json`, 2026-08-01). Median **14.94 m/station**, p10 13.43, p90 19.30 — the
  spread across pairs IS the error bar, since these are independent measurements of one
  constant. Two things it exposes: gates 1–2 were measured twice and disagreed by 18%
  (9.1 m vs 7.6 m), which bounds PnP distance repeatability independently of any sketch
  error; and the 21.74 m outlier is the pair 3–4, where gate 3 is the entry already flagged
  as least certain for sitting 40 px off the drawn line. **First vertical number too:**
  gate 1 is **+4.28 m** above gate 0, from gravity alone, no pose stream.
  Height is the data-limited half — 4 of 5 labelled frames read |a| between 4.96 and 12.66
  m/s², too dynamic for the accelerometer to be reading gravity, so `score_frames()` now
  weights steadiness. `20260731-222724` is a straight-fly test and rarely steady; a calmer
  session is the place to get heights.
* **The superseded assignment-free fit, kept because the lesson generalises.** `fit_scale()` matches vision's
  measured metre distances against the sketch's station-unit distances over a swept scale —
  assignment-free, so it never needs to know which gate is which. It is **degenerate**: every
  scale from **4.7 to 18.2 m/station** sits within 2× of the best cost, 23% of the swept
  range. 17 gates give 136 pair distances, a near-continuum, so any measured distance matches
  *some* pair at almost any scale. The apparent optimum of 10.9 m is noise, and
  `metres_per_station` stays `null`. The drawn bar lengths would imply 12.6 m/station (if a
  bar is the 1500 mm inner width) or 22.7 m (if the 2700 mm outer) — both inside the
  degenerate plateau, so they corroborate nothing. Scale only matters for fusing the map with
  PnP ranges; guidance in station units does not need it.
* **Vertical is ABSENT.** The sketch is top-down only, so the map says nothing about gate
  height. Claire's own caveat, kept in the JSON under `accuracy`: order and topology are
  certain, positions and plane angles are approximate, vertical does not exist.

**RACE ORDER IS RESOLVED** (2026-08-01). Claire drew the flown course line onto the sketch
(`map-aerial/approx_map_path.png`), cross-referenced against video of a completed lap.
`race_order()` floods the line geodesically from the entry end and orders gates by arc
length. Not a y-sort — the course doubles back around the Station 22 and 26 loops, which a
y-sort would interleave. All four of Claire's landmarks come out in sequence: gates 1–3 loop
right around **22**, gates 5–6 run far left past **17** then **16**, gate 9 is the far-right
excursion at **26**, gate 13 the far-left one at **13**, then 15–16 exit past 12.

Gates must be read from `approx_map.png`, not from the path image: the drawn line crosses
each red bar and splits it into two components, giving 30 "gates" instead of 17. Both images
share one canvas, so coordinates align.

Least-certain entry is race index **3** (along 18.50), which sits 40 px off the line where
every other gate is within 17 px.

## THE DATA WALL — why the map is not finished

**No VQ2 recording covers the course.** Checked every session with a `race.csv`
(2026-07-31): `active_gate_index` never exceeds **1** in any VQ2 session. The only sessions
that reach gates 2–5 are `20260731-195307` and `20260731-204841-vq1-lap-slow`, and both are
VQ1's 6-gate course. **At most one gate has ever been crossed in VQ2.**

The station reads say the same thing geometrically: across the three largest VQ2 sessions
every confident read falls in **stations 22–28**, a handful of bays in a hangar that runs to
at least station 30 and probably ~40.

**Correction, 2026-08-01.** An earlier version of this section said the recordings "cover only
the opening" of the course. That was too strong, and measurement contradicts it: one frame of
`20260731-222724` carries **15 gate detections**, and detection range runs to **75 m** with a
median of 35 m. Much of the course is *visible* from where the aircraft flew. **The wall is
ORDER, not coverage** — a functional map answers "where is gate k+1 relative to gate k", and
only `active_gate_index` says which detection is gate k. It advances only on a crossing.

Consequences, and they are the reason to stop rather than tune:

* **The 35-node reconstruction cannot be repaired into a course.** Merging duplicates and
  reading stations both make it *cleaner*, but nothing in it assigns race order. The sketch
  now supplies the geometry that reconstruction was trying to recover, which makes this a
  cross-check rather than the critical path.
* **The strongest identity channel is unusable on existing data.** Lit-gate + `active_gate_index`
  gives sim-supplied absolute identity, but the index never advances, so there is nothing to
  key on. The channel is sound; the recordings do not exercise it.
* **Station coverage is a function of flying**, 38% slow versus 1% fast — so the same fix
  serves both problems.

**What is needed: one VQ2 training session that flies the whole course slowly, recording.**
It resolves both open items at once — crossings give race order, and identified gates give
the metric scale the assignment-free fit cannot. Run `checksession.py` on it before building
anything: it fails a recording whose `active_gate_index` never advances, which is the exact
defect that went unnoticed until a map had already been built on one.
Training is free and unlimited, the course is fixed, and `teleop.py` already records
everything. Slow matters more than clean — coverage is 38× better slow than fast, a wobbly
lap that visits all 17 gates beats a tidy one that visits two, and gate crossings are what
make `active_gate_index` advance and hand identity over for free.

## THE MEASURED MAP — `map_vq2.json` (updated 2026-08-01, pausing lap)

Claire flew `20260801-144858-vqual2-lap-pausing` exactly as requested: gates 0–14 slowly,
pausing at co-visibility points, **25 markers** in `events.jsonl` (`kind=marker`) at the
moments the visible gates were fully in frame. `checksession.py`: usable (with warnings).
What it bought, and what it exposed:

* **12 measured gate pairs** (was 8), and repeatability jumped an order of magnitude:
  pair 0-1 reads **20.12 / 20.02 / 20.06 m on three separate flights** (old cross-lap
  spread was 2.2 m median — that spread was identity contamination, not measurement noise).
* **5-6 RESOLVED: 16.3 m.** The 0-11 lap (115735) was the wrong one, and its own rows
  contain both modes (29 rows at ~15 m, 35 at ~38 m) — a fragment of its gate-6 identity
  had slid onto a farther gate. Primary lap unimodal at 15.6; bridge-graph LOO predicts 14.1.
* **All 17 gates now sit in five internally-measured components** — [0-2] [3-6] [7-9]
  [10-13] [14-16] — but NOTHING measured links the components: 2-3, 6-7, 9-10, 13-14 have
  zero clean co-visible rows (12-13: n=2 at ~10.4 m). The reason is a detector limit, not a
  flying limit: **at every marker moment the linking gate WAS in frame but clipped**, and
  metric rows refuse clipped quads. Partial-quad fitting converts those exact refusals into
  the missing edges — it is now doubly the highest-value detector change.
* **Braced-rigid did not grow** (still effectively the 4-5-6 triangle plus hinges; bridge
  edges have MDS stress 10-22 m, LOO |err| median ~7 m). Positions in `map_vq2.json` are
  topology, not geometry; the pair table is the product.
* **The pausing lap broke the vote-based identity in a new way**: 20-30 s hover windows let
  ELEVEN tracks win "gate 13" at ranges 2-44 m (one with range *rising* through the
  approach). Three refusals fixed it (all in mapvq2.py): a closing-speed funnel binding the
  observation nearest each crossing; per-gate pruning of any track that contradicts a
  closer-approach track in-frame by >3 m; and per-frame gate ambiguity refusal checked on
  ALL observations, because impostor rows survive exactly where the true gate fails
  `clean()`. Plus one bug of the classic shape: `measure()` held same-gate duplicates in a
  plain dict and the LAST insertion silently won the frame (0-1 came out bimodal 20/34 m).
  Every one of these was caught by **rendering identities on frames**, none by aggregates.
* **Bimodal pairs are now split, with Claire's markers as the referee**: two static gates
  have one separation, so a two-mode distribution is a data-association fault by
  definition; `aggregate()` keeps the cluster containing marker rows (else the larger) and
  reports the dropped count. 7-8 is the live case: fast laps say 26.6-28.5 m, the pausing
  lap says **11.04 m** (MAD 0.32, 25 marker rows, crossing-anchored gate 8) — sided with
  11.04, flagged in the JSON's `known_open_disagreement`.
* **Scale: the sketch is not locally to scale.** Per-pair implied scale spans 13.6-65.5
  m/station (median 22.4). Gates 0-9 cluster at 14-24; 10-11, 14-15, 15-16 imply 36-66 —
  the sketch compresses the far end of the course 2-4x. A single metres_per_station
  constant is not extractable from gate pairs; treat 15.97 as the near-course value only.

## Open

* Build the digit templates and classify; tag gate detections with station number; re-run
  the map. **Prediction to test against:** if fragmentation is the whole story, node count
  collapses from 35 toward 17 with no merging heuristic at all. Kill criterion: if it stays
  above ~30, identity is not the dominant error and the distances themselves are suspect.
* Vertical-text reading (cluster glyphs into a collinear group in any orientation, de-rotate
  by the group axis). Costs hours rather than minutes; only worth it if measured face-number
  coverage is actually short. It would also survive image roll in acro flight, which the
  face-number path may not.
* Exploit the lit-active-gate cue as a second identity channel, to referee the first.
* `R_COMMIT` (attention handoff range) unset — needs a real approach measurement, available
  from VQ1 flying since gate dimensions are identical.
* Frames are backed up nowhere — gigabytes of JPEG, excluded from git, one laptop.

## Auto-label error is rate-dependent, not per-gate (2026-08-01, triage finding)

Claire noticed one gate's projected truth quad sitting consistently off the real gate.
Measurement (gatebias_measure.py, 2246 label-vs-inner-detection matches) falsified the
obvious reading (bad surveyed position): all five leave-one-out refits land at the
estimator's noise floor, and same-frame gates shift by EQUAL PIXELS regardless of
apparent size -- a rotation error's signature, impossible for a position error.

The real cause: attitude interpolation/aliasing during fast rotation. Median label
offset vs body rate: ~1.0 px below 0.2 rad/s -> 3.4 px above 1 rad/s -> 4.0 px median,
11 px p90 above 2 rad/s. Not a clock offset (best global dt is -1 ms, no improvement).
It reads as "one bad gate" in triage because within one turn segment the rotation
direction is constant, so every crop of the gate being approached leans the same way.

Consequences: (1) instances recorded above ~1 rad/s carry ~3x label error -- a
down-weight candidate for training, and an eval caveat (some "net error" there is
label error); (2) do NOT move any gate's 3D position to fix label offsets -- it fixes
nothing (gatebias_delta3d.py: best constant 3D delta explains 2.67 -> 2.53 px only).
Per-instance offsets cached in gatebias_offsets.csv (--reuse).

## Auto-label v2: three error mechanisms, measured and handled (2026-08-01, labelfix_*)

Claire's hand triage of 223 slides (70 N / 153 Y, 31% bad) decomposed into three
mechanisms, and `autolabels_vq1_v2.json` + `labelfix_negatives_v2.json` now carry the
fixes. v1 files untouched; `(key, gate)` is the join key across versions.

* **A -- attitude aliasing above ~1 rad/s** (~3.4 px median label error, all gates
  equally). Not fixable after the fact; every v2 instance now carries `body_rate`
  (rad/s, triagecheck_rates.py's matching path) so training down-weights without
  re-scanning. 21.6% of kept instances sit above 1 rad/s.
* **B -- labels on the BACK of a nearer gate.** Invisible to both old checks: aperture-quad
  overlap scores 0.0 when the label lands on the nearer gate's SOLID BOARD, and the
  orange-band filter passes because the nearer gate supplies the orange. Fix
  (`labelfix_occlusion.py`): project every nearer gate's solid region (2.7 m board minus
  aperture opening) and flag when it covers >= 0.5 of the target's band+quad extent --
  seeing THROUGH an open aperture is legitimate and does not count. 521 instances
  (4.2%) moved to `labelfix_negatives_v2.json` with reason `behind-gate`; both triage
  example slides flag at 0.96/1.00; contact sheet `labelfix_occl_sheet.png` read back
  clean (flagged = boards/wordmarks, through-aperture untouched).
* **C -- the "gate 1 orientation" hypothesis, half right.** Full-pose fit
  (`labelfix_gate1fit.py`: all four corners of detector inner quads, rate <= 0.5 only,
  behind-gate frames excluded, rvec regularized because the gate-0 control exposed a
  flat rotation valley on head-on views) says gate 1's surveyed CENTRE is off by
  **0.57 m, almost all along world -x**, orientation only 1.7 deg. Identical per session
  to 1 cm. Held-out corner error 2.59/27.6 -> 1.67/2.7 px (median/p90); the two shear
  slides render dead-on (`labelfix_gate1_valid.png`). Gate-0 control: 1.75 deg, 0.09 m.
  WHY EVERY EARLIER INSTRUMENT MISSED IT: an along-x error is near the line of sight
  head-on, so it barely moves projected centres (gatebias 1.51 px "healthy") and hides
  inside the PnP-vote x-MAD of 0.54 m (refit said 0.10 m); it only turns into pixels on
  close oblique views -- exactly the CLIPPED instances gatebias_offsets.csv excluded.
  A measurement conditioned on detector match + unclipped could not see it. 3114 gate-1
  instances regenerated from the refit pose (`gate1_refit: true`; orange/occl numbers
  on those are stale by construction).

**Acceptance vs the 223 verdicts** (`labelfix_score.py`): of 70 N -- 12 removed
(behind-gate), 19 corrected (>2 px), 39 unchanged of which 29 have rate > 1 (mechanism
A, down-weighted). Of 153 Y -- 5 disturbed: 3 gate-1 corrections of 2-6 px that render
as improvements, 2 removals whose labels sit on a nearer gate's board (defensible;
likely triage slips). **Residual: 10 low-rate N** -- and they are mechanism C again,
smaller, on OTHER gates: diagnostic fits (not applied) find gate 2 delta -0.25 m
(held-out 1.68->1.29 px), gate 4 delta -0.17 m + 2.2 deg (p90 7.0->3.7 px), gate 3
unchanged. v2's verdict (clean enough for one retrain at ~4.5% unhandled) was
superseded the same day by v3 below, which applies that follow-up.

**v3 (`autolabels_vq1_v3.json` + `labelfix_negatives_v3.json`, labelfix_build_v3.py):
the same refit applied to gates 2 and 4 -- TRANSLATION-ONLY.** Full-pose fits improved
held-out error on both (gate 2 1.68/2.75 -> 1.29/1.99 px, gate 4 1.63/7.04 ->
1.49/3.74) but their rotation components were NOT session-consistent (gate 2: 7.4 vs
1.8 deg between sessions) while the centre deltas agreed to a few cm -- and pinning
rotation to zero kept essentially the whole gain (1.32/2.13 and 1.52/3.56). So v3
applies centre-only deltas: gate 2 **[-0.26 -0.01 +0.05] m**, gate 4 **[-0.18 0.00
-0.01] m** -- the same -x signature as gate 1, smaller. Gate 3, run as the extra
control, came back NULL (|delta| 0.01 m, 0.5 deg, n=74) and is untouched; its held-out
p90 (~58 px) is detector mismatches in a thin test set, not pose. 2562 + 2833 instances
regenerated (`gate2_refit`/`gate4_refit`), 6 newly behind-gate, v2 negatives carried
over (527 total).

**v3 vs the 223 verdicts** (labelfix_score_v3.json): of 70 N -- 12 removed, 23
corrected, 35 unchanged (28 at rate > 1). **Residual falls 10 -> 7** (3 on gate 3 --
small far views, moves 0.0 px, i.e. NOT pose; 4 on gate 4 near views). Of 153 Y -- 2
removed (same defensible board cases as v2) and 24 "corrected", but that word
overstates it: median move is 7.2% of gate size (p90 11.5%), half are rate > 1 slides
whose labels carried mechanism-A error anyway, and the rendered worst low-rate cases
(`labelfix_v3_check.png`) land on or slightly better than the old quads -- refinements
from a held-out-validated pose, not damage. FINAL VERDICT: **v3 + rate-down-weighting
is the label set for the one retrain.** Unexplained residue is 7/223 = 3.1%, none of it
pose-fixable by these mechanisms (gate-3 residuals don't move under refit; gate-4 near
views are the detector-coverage hole partial-quad fitting addresses).

## THE SKYLIGHT COMPASS — per-frame yaw from the ceiling light grid (2026-08-02, skylight_*)

Yaw was the one axis gravity cannot give and VQ2's telemetry block removed. Claire's
observation — the ceiling lights are square panels in a flat grid parallel to the floor —
closes it. `skylight.py`: every detected light-edge segment's interpretation plane must
contain the segment's horizontal 3-D direction in the gravity-levelled frame, so each
segment votes for a heading **mod 90**; weighted circular median over ~60 segments/frame.
No vanishing points. Gravity comes from a gyro+accel complementary filter (canonical
signs per `CONVENTIONS.md`: gyro ×-1, accel unchanged); a per-session pass unwraps the
mod-90 branch with the gravity-projected gyro integral, re-anchoring on every confident
frame so gyro drift never survives one sighting.

**Compass CSVs exist for five sessions** (`skylight_<session>.csv`: per-frame
`theta_mod90`, `psi_mod90`, confidence, `psi_unwrapped`): pausing lap 98% confident
frames, 155701/202110/202923 99-100%, strafe 75%. Vote spread median 1.2-2.1 deg,
n_seg median ~60.

Validated three independent ways (`skylight_validate.py`, log `skylight_validate.log`):

* **Marker-hover constancy** (52 windows over 4 sessions): raw p-p median 1.73 deg —
  but the two worst windows (13.7, 28.5 deg) were REAL yaw, Claire's gyro shows 10.9 and
  25.3 deg of it during the "hover". Gyro-compensated (`skylight_marker_gyro.py`):
  residual p-p median 1.73 -> per-frame std median **0.36 deg** (p90 ~1). The <1 deg
  target is met as 1-sigma noise, not as 45-frame peak-to-peak.
* **Gyro cross-check** (pausing lap, 9617 confident pairs): slope of d(psi_grid) on
  d(psi_gyro) = **+0.949, sign +1 confirmed** (canonical nose-right increases psi); RMS
  frame-to-frame mismatch 0.79 deg (0.93 deg on the |rate|>20 deg/s subset); best
  camera-IMU offset -10 ms (flat). `vercheck/skylight_gyro_check.png`: tracks through
  90-deg turns, residual a few deg over 90 s.
* **Rotation/map check**: gate 5's grid-frame azimuth (psi + levelled detection bearing)
  during a 50-deg nose sweep at 27 m: std 5.26 deg vs **24.5 deg for the wrong sign** —
  geometric sign confirmation. Co-visible-pair angles vs `map_vq2.json` distances:
  |error| median 3.66 deg over 100 pair-frames (tests PnP+levelling; heading cancels).

**Reflection guard is real and appearance-free**: every accepted component must sit
entirely above the levelled horizon (`HORIZON_MARGIN_DEG`), so glossy-floor mirror images
are rejected by gravity alone; gate decoration falls to the orange-surround test.
Evidence: `vercheck/skylight_reflections.png` (frames 00148278/00151803, session 155701)
— green accepted lights above the drawn horizon, red `below-horizon` boxes on the bright
floor reflections beneath it.

**Metric scale of the grid: ratios landed, the metre did NOT.**

* Grid spacing in units of ceiling height H above camera: **0.3403 and 0.5666 per axis**
  (n>420 each; ratio 1.6650 = 5:3 within 0.1% — the cells are 3:5 rectangles, one
  design module).
* H itself is declared NOT MEASURED. The V1 light-motion-vs-gate-PnP estimate (median
  5.91 m, MAD 4.52, p10 negative) had two setup faults — nearest-light matching ALIASES
  on a periodic grid at 1 s baselines, and PnP jitter ~ baseline — fixed in V2 by
  chaining frame-to-frame light motion (aliasing-free) against endpoint-smoothed PnP.
  V2: MAD 4.52 -> 0.52 at ~5 m baselines. But the median moves with the setup: 6.07 ->
  4.71 m as chains lengthen, ~4.7 vs ~5.0 with/without a border-clip filter, and single
  long-baseline windows disagree 4.25 vs 8.7 m. Two static planes whose measured
  separation depends on the measurement setup = measurement fault (the map lesson,
  again). Bound: **H in [4.5, 6.5] m**; the gate-PnP displacement side is the weak leg.
* Flagged hypothesis, not result: **H = 6.17 m -> pitches 2.10 and 3.50 m** is the
  unique value in the band where the 15-16 station spacing (21.01 m, `map_vq2.json`) is
  an integer cell count on BOTH axes (10 x 2.10 = 6 x 3.50 = 21.0) and the pitches are
  simple numbers. Three coincidences, zero confirmations. Test that would settle it: one
  slow pass <10 m from a gate with lights confidently tracked (PnP error shrinks with
  range), or partial-quad fitting tightening PnP itself.

**Floor-grid alignment** (`skylight_pitch.py`): yellow floor lines vs light grid, median
+0.32 deg (MAD 2.72, n=272) — the floor/bay grid is parallel to the light grid, so
station-column geometry and the compass share one frame.

What the compass enables now: a continuous, absolute-mod-90 heading channel at ~0.4 deg
1-sigma on 98%+ of frames in normal flight — the missing yaw reference for the
world-frame velocity problem, up to the 4-fold grid ambiguity. The mod-90 fold is the
main open limit: quadrant disambiguation needs one absolute cue (station-number sighting,
`active_gate_index` + map bearing, or the race-start attitude). Strafe-session confidence
drops to 75% (camera pointed at gates, lights clipped); fast rolls survive via the gyro
unwrap. Caveat: gravity filter alpha tuned on these sessions; sustained aggressive
acro may need the tilt-error term revisited.

## MAP COMPLETE — one component, all 16 consecutive links (2026-08-02, mapedges_final4.log)

**Headline: `map_vq2.json` is now a single connected component over all 17 gates — 20
measured pairs: every consecutive link 0-1 … 15-16 plus skips 4-6, 7-9, 12-14, 13-15.**
The two missing links came from Claire's 2026-08-02 staged strafes, entering through a
new intent channel (`mapedges_strafe.py`, same discipline as `mapedges_hover.py`:
session-name intent + per-window render audit + known-median referee + identity-free
measurement; sheets in `vercheck/strafe0802_windows_*.png`, accepted pairs in
`vercheck/strafe0802_pairs_montage.png`).

* **2-3 = 13.34 m, MAD 0.59, n=79** (33 net-involved), |dz| 1.46 — three vantages in
  two sessions (2-3-c gold clusters t~22-26 s and t~30-33 s near gate 1; 2-3-b marker 1
  from beyond gate 3). Gate 2 anchored by its own crossing at t=136.1 s of 010914.
  Cross-vantage medians 12.8/13.3/13.9 — static-pair consistent.
* **6-7 = 16.42 m, MAD 0.27, n=556, all contour** — five hover viewpoints in 005431.
  The 202110 trap (partner behind gate 6 was gate 5) is excluded by SIGN, not just by
  distance: the partner sits **+5.3 m ABOVE gate 6** on all four checked windows
  (signed v·g_down = −5.2/−5.3/−5.7/−4.8), while gate 5 sits 4.8 m BELOW gate 6
  (5-6 dz +4.83, up-positive). d 16.4 also ≠ 15.44 (5-6). Gate 6 anchored by the
  Station 16 column (sketch along 16.3); scale 16.9 m/station — in family.
* **The scene at 1/2/3, settled by renders**: gate 1 is the big banked ELEVATED gate
  (0-1 dz +3.3); gate 2 the floor gate ~8 m below/behind it (crossed at the end of
  010914); gate 3 the next ribbon-threaded floor gate. Every earlier "2+3" confusion
  was fragments of gate 2 under gate 1.

**Checkpoint swap**: the gatenet mint channel now runs `gatenet_runs/colab-v3-nw/best.pt`
(replacing `block/best.pt`; all guards unchanged — PnP residual 0.3 px, jitter
stability, range continuity, cross-channel corroboration). No previously-accepted pair
moved materially: max shifts 13-15 −0.44, 12-14 −0.34 (n 8→15), 14-15 +0.26,
11-12 −0.23; everything else ≤ 0.2 m. n changes: 10-11 45→76 (MAD 0.65→0.42),
13-14 212→129 (d unchanged, 12.37→12.31), 7-9 17→21.

**Refusals that still matter**
* **1-2 is the least-trusted edge**: accepted value 8.32 m rests on n=5 contour rows,
  while a REFUSED gatenet channel reads 13.0 m with n=34 and the sketch scale mildly
  favours the larger value (8.32 → 13.6 m/station, lowest in the table). Unresolved;
  next flight should keep gates 1 and 2 co-visible and unclipped from 15-20 m out.
* strafe-12-13-14 (004153) refused WHOLE: no ribbon renders that far down-course and a
  12↔14 swap (13.45 vs 12.37) is inside the 3 m referee tolerance — undetectable on
  these renders, and those edges already have n=14/129/15.
* 2-3-c t~98-104 window refused: montage showed the g3 NET quad straddling two
  OVERLAPPING gates (the merged-gate failure mode, again caught by looking at the
  frame); 2-3 keeps n=79 from vantages that agree without it.
* 005208 (2-3-a) refused: no window with both staged gates measurable.
* Net-row corroboration drops (unchanged policy): 0-1@23.9 n=11, 1-2@13.0 n=34,
  12-13@19.2 n=9, plus singletons — listed in mapedges_final4.log.

**Solve quality**: MDS stress med 12.3 m / p90 19.8 m over 257 nodes, 374 edges.
LOO |err| median 9.5 m, p90 21.2 — worse than the per-component numbers of 08-01 BY
CONSTRUCTION: the single component is chain-like between the braced clusters
(12-13-14-15 quad, 4-5-6, 7-8-9), so holding out a bridge edge has no second path.
Heights: strafe/hover/intent rows carry |dz| only → 2-3, 6-7 and most of 12-16 have no
signed dz inside the solve (6-7's +5.3 m is measured but lives here and in the status
JSON, not in the layout). m/station median 19.2, p10-p90 15.2-26.7; the 10-11 outlier
(57.6) is the sketch's across-row compression, unchanged.

Viz regenerated: `map_current_viz.png` (now scripted — `map_viz.py`, same design:
sketch positions, measured edge labels, component colours — one colour now).

## THE DIRECTION-ACCURATE LAYOUT -- pair vectors from the skylight compass (2026-08-02, mapdir.py)

**Headline: `map_vq2.json` now carries `layout_directions_2026_08_02` -- the first
direction-accurate 17-gate layout.** Every accepted pair measurement was re-derived as a
VECTOR: distance (the accepted medians, unchanged) plus absolute grid-frame bearing,
using the validated formula `bearing = psi_unwrapped + atan2(v_lev.y, v_lev.x)` per row
(2008 usable rows, compass-confident frames only, |d - accepted| <= 3 m). Compass CSVs
were extended to the two fast laps (121520: 94% confident, 115735: 74%). The solve is
linear (p_b - p_a = d*u(bearing)), no MDS, **no reflection ambiguity** -- the levelled
grid frame is handed. Viz regenerated: `map_current_viz.png` now draws MEASURED node
positions, measured gate-plane bars (PnP normals + compass, mod 180; gates 8/12/13
refused at MAD > 20 deg -- IPPE head-on valley), signed dz labels, and keeps the sketch
as a small topology-check inset. **The measured layout reproduces the sketch's up/down
excursion pattern gate for gate** (1-2 up, 5-6 down, 9-10 up, 13 down, ...), with the
grid->sketch fit at rotation 97.4 deg + mirror, |res| median 6.3 deg = the sketch's own
angular accuracy; 97.4 mod 90 = 7.4 deg, i.e. station rows parallel to the light grid
within sketch accuracy, as skylight_pitch.py predicted.

**Numbers** (mapdir_solve_final2.log):
* Cross-session bearing agreement: **median 1.05 deg, max 3.28 deg** over the 10 pairs
  measured in 2+ sessions -- the compass channel repeats across flights at the level
  its 0.4 deg/frame validation promised.
* Vector-solve residuals median 0.00 / p90 0.31 m; **loop closures 0.6-0.8 m on 46-52 m
  cycles (1.2-1.6%)** for 7-8-9, 12-13-14, 13-14-15.
* **LOO with directions: non-bridge |d err| median 0.64 m, p90 0.69 m** (was 9.5 m
  median with 4 unplaceable hinges, distance-only). Bridges (0-1...6-7, 9-10, 10-11,
  11-12, 15-16) stay honestly unpredictable -- directions do not fix disconnection.
* **Full signed height ladder for all 17 gates** (gauge gate 0): the audited channel
  track labels order each pair, so hover/strafe/intent dz now carries SIGN (it was
  |dz|-only in measured_pairs). Solve residual ~0; z runs 0 to +10.9 m (gate 7 highest;
  6-7 step +5.31 m and 5-6 +5.00 m reproduce the staged-strafe +5.3/+4.8 findings).

**Quadrant provenance** (all in the JSON): pausing lap = reference; 121520, 115735,
155701, 202110, 202923 branch-resolved by shared pairs, residuals 0.3-2.8 deg -- these
are measurement. The 2-3 strafes (005725, 010914), the 6-7 strafe (005431) and the
15-16 strafing session rest on the SKETCH-PRIOR branch (margins 14-39 deg); the two 2-3
sessions agree with each other to 1.8 deg, and the 15-16 axis (n=433, mod-180) agrees
with the one thin full-bearing lap row to 1.8 deg. So the 2-3 / 6-7 / 15-16 leg
DIRECTIONS are measured mod 90 and sketch-resolved beyond that -- stated, not hidden.

**REFUSED: the 4-6 edge (22.74 m) re-measures 4-5.** Directions caught what distances
could not: the 4-6 vector (22.74 @ 67.7, dz +0.19) and the 4-5 vector (22.46 @ 66.4,
dz -0.31) differ by 0.32 m -- one physical object under two names -- and the partner's
height matches gate 5, not gate 6 (+4.8 m per the signed ladder). The 22.46/22.74/15.65
"braced triangle" closes numerically, which is why the distance solve never flinched;
the direction solve's 25% loop closure exposed it, and the render confirmed
(vercheck/mapdir_anomalies.png). Sketch corroborates: 4-6 = 1.61 u is the near-collinear
LONG side, predicting ~36 m; the direction solve predicts 37.3 m. The 4-5-6 "brace" was
fake; real bracing there needs a genuine 4-6 (or 3-5) measurement.

**1-2 ADJUDICATED: 8.32 m stands; 13.0 m refuted at the source.** The "13.0 net
channel" (n=34 in final4's drop list) is not a channel: its rows live entirely in the
tiebreak opening dash and span **4.4-24.4 m across vantages** -- two static gates cannot
do that; it is the identity-slide population REFUSE_ROWS already documented there,
re-entering through net seeds. The 8.32 pausing channel is viewpoint-consistent in
distance (7.5-9.7) AND bearing (MAD ~2 deg). Not closable by loop (1-2 is a bridge), so
it remains the thinnest edge: n=5, one session. The co-visible 15-20 m pass is still the
right next flight, now with a sharper ask: it would also convert the 0-1/1-2 bridges
into a checkable chain.

Tools: `mapdir.py --collect / --angles / --solve --write` (rows in `mapdir_rows.json`,
plane angles in `mapdir_angles.json`, logs `mapdir_collect.log`,
`mapdir_solve_final2.log`); anomaly renders `vercheck/mapdir_anomalies.png`;
`map_vq2_predir.json.bak` is the pre-directions map JSON. One channel-fidelity note:
re-running the intent channel hit AUDIT-MISMATCH on targeting marker 14 (4 mini-tracks
vs the audited 3 -- window contributes nothing now); 13-14 keeps n=129 from the hover
channels regardless.

## THE PRODUCER -- flight-time Observation filling, validated on replay (2026-08-02, producer.py)

**`pilot/producer.py` now fills the frozen interface end to end**: frames + HIGHRES_IMU +
race packet -> `interface.Observation` (73-D), with the AUTO_ATTENTION yaw servo
producer-side per the YawMode contract. Runbook + live wiring in `pilot/PRODUCER.md`;
replay artifacts in `perception/producer_runs/`. Nothing committed.

Architecture per frame: detect.py contours + interior-colour test -> slot association
(identity = active_gate_index + map-DISTANCE lookahead bootstrap) -> gatenet
(colab-v3-rot) on seeded crops where the detector is weak (clipped/outer/large/
ribbon-suspect), gated by the paired conf head (colab-conf-rot, p=0.082) -> IPPE_SQUARE
via solvePnPGeneric (both tilt solutions; normal_valid False near head-on; residual
computed MODULO the square's dihedral symmetry -- raw residual rejects exact head-on
quads at 33.9 px, measured on a synthetic) -> GateObs degradation ladder. Tracks are
ROTATION-coasted through the canonical gyro (p <- p - w x p dt); ribbon = cyan row-bins;
roll/pitch from the skylight-validated complementary filter; vel_bearing from drag;
speed_est stubbed at conf 0 (contract-optional).

**Validation, primary lap `20260801-121520-vq2-lap-0-15` (6303 frames, all 17 gates):**
* current gate valid 83.9% of frames, pose_valid 59.2%, mean staleness 0.15 s; lookahead
  slots valid 45%/35%; attention CUR 77% / NEXT 22%; active gate follows race order
  through all 17 crossings, approach range decreasing (frac-decreasing median 0.78,
  min 0.50 across crossings).
* Camera->body sign check PASSES analytically and in code: image-centre target at range
  R -> bearing 0, elevation +20 deg (camera boresight 20 deg above body-forward).
* Rotation-coast direction verified two ways: synthetic 1 rad/s nose-right for 1 s moves
  a dead-ahead gate to bearing -57 deg with range preserved; on replay, coasted
  predictions land median 15-37 px from re-acquisitions (a wrong sign would be hundreds).
  Metric miss median ~5 m is RANGE along the line of sight -- translation is not coasted
  (no speed estimate), so range_sigma_m inflates at 10 m/s-equivalent while coasting.
* Fly-through degradation (read back on rendered strips): full-frame-orange frames show
  the slot coasting with staleness, never a garbage pose. strip.png sheets for lap +
  strafes read back clean; one caught flaw: g1/g2 labels swapped once on the banked 1-2
  pair (the least-trusted map edge).
* Static-pair check, strafe 6-7 (`20260802-005431`): two-nearest-clean-PnP separation
  **16.50 m, MAD 0.18, n=118** vs the map's accepted 16.42 m -- constant across
  viewpoint. 15-16 strafing session gave only one 0.6 s co-clean window (12.0 m, n=16,
  identity ambiguous) -- consistent separation within the window.
* to_vector round-trip: 73 dims, invalidated gate block zeroes (selftest asserts).
* attitude_conf: median 0.88 overall, 0.51 during |gyro|>1 rad/s, p10 0.00 -- degrades
  with BOTH |a|-g mismatch and sustained body rate (a coordinated bank keeps |a| near g
  while apparent gravity tilts; rate is the observable proxy).

**Latency (CPU, this laptop)**: unthrottled runs measured detect 5.5-7.6 / net 6.8-8.4
per batch (<=3 crops, ~60% of frames) / compass 13-15 per call (every 10th frame) /
PnP ~1 -> ~12-17 ms median total. Under thermal throttle the same code measures 28-32 ms
median (every stage inflates equally, including untouched ones -- same signature as the
gatenet training epochs). p90 spikes to ~60 ms; live use should drop late frames and let
tracks coast (IMU keeps integrating).

**2026-08-02 addendum -- the wrong-gate attention bug (found by Claire in the HUD)**

BUG, highest-severity class: with attn=GATE_CURRENT the current slot locked whichever
clean SMALL quad the detector liked while the real current gate filled the frame
huge/clipped -- the attention arrow then pointed one gate ahead. Openings: "gate 0" held
at 38 m (0-1 edge: 20.4 m) while gate 0 sat dead ahead at ~11 m; act=7 held a "g7" at
56-59 m WITH normal_valid, minted from an ~8 px quad. Measured: **30.45% of lap frames**
(20260801-121520) carried a current-slot range beyond the map-consistent bound.

Root cause in code (producer.py): (1) the current-slot bootstrap required
`source == 'inner'`, so a huge clipped current gate -- outer-only or undetected -- could
NEVER win its own slot, while the next gate's small clean inner quad could; (2) no
map-range prior existed, so nothing rejected a 38 m "gate 0"; (3) association stickiness
then reinforced the wrong track indefinitely (a far gate measures consistently);
(4) IPPE tilt ranking on sub-20 px quads asserted normal_valid on noise; (5) ribbon
attention took the farthest sample of an UNDIRECTED line (can point backward on a
switchback); (6) the interior-colour test on net quads used the lax current-gate variant
for lookahead slots too.

Fix (association/attention only; PnP + degradation ladder untouched): map-range prior
`cur_bound0 = 1.3 x edge(k-1,k)` set at each crossing (plain edge(0,1) at start/reset)
plus a measurement-ratcheted admission bound (`min` over `max(1.3r, r+5)`, n_obs>=2);
candidates beyond the bound never enter the current slot, and a current track beyond
cur_bound0 for >0.5 s is evicted. The bootstrap now admits OUTER quads largest-first
(the current gate is by construction the nearest upcoming gate). Crossing continuity was
already structural (tracks keyed by absolute gate index inherit k+1's lookahead track).
normal_valid is size-gated at 20 px. Ribbon attention is disambiguated by `prog_dir`
(gyro-coasted last-known current-gate direction, compass-free; map prediction as
fallback) and REFUSES to SEARCH when every sample opposes the prior. Lookahead net quads
get the strict decoration test.

Before -> after, full lap replay (6303 frames) + strafe:
* over-bound current frames (same static rule both runs): **30.45% -> 2.24%**; the
  residual concentrates where the bound inherits a thin map edge (act 2 via 1-2 = 8.32 m
  n=5; act 15 via 14-15 = 13.35 m n=9).
* current gate valid **84.1% -> 90.8%**, pose_valid 59.2 -> 63.2%, staleness
  0.153 -> 0.138 s; race order through all 17 crossings both runs; approach
  monotonicity median 0.78 -> 0.95, min 0.50 -> 0.72.
* per-leg: 11 (23 -> 100%), 4 (45 -> 100%), 6 (64 -> 100%), 9 (66 -> 100%); legs 2/8
  give up 13-23% valid to the tighter admission on thin-edge legs -- invalid over
  impostor, by design; better 1-2/14-15 edges recover it.
* normal_valid on sub-20 px current gates: 1029 -> 31 frames.
* strafe 6-7 canary unchanged: 16.50 m MAD 0.18 n=118 (vs map 16.42).
* opening strip read back: gate 0 holds the current slot, arrow on it, range 11.0 ->
  8.6 m decreasing; act=7 current at 8.7-10.8 m (was 56-59 m); handoffs into 5/12 clean.
* ribbon anti-parallel picks (>120 deg from where the gate was next measured): 0 both
  runs on this lap -- the switchback case Claire saw live did not occur here; the
  refusal is prophylactic on this recording.

Artifacts: `producer_runs/lap-121520-{BEFORE,AFTER}/`, strips
`producer_runs/bugfix_*.png`, checks in `bugfix_metrics.py`. Known gap flagged, not
touched: hud.py's net path applies NO interior-colour test (its owner's call).

**Known limitations, stated**
* The map-frame MIRROR fit (s, c in producer.MapModel) is unstable run-to-run (s flipped
  with the anchor window); treat compass-rotated map vectors as a soft SEARCH prior
  only. The mirror-free map-DISTANCE bootstrap is what actually fills lookahead slots.
  Fix queued: keep all anchors, demand a bigger inter-hypothesis margin, seed s from a
  one-time offline calibration against this lap.
* Bootstrap can latch a wrong gate for ~0.3 s at acquisition (largest-clean-first);
  self-corrects via largest-first ordering + range funnels; residual identity risk is
  the 1-2 banked pair. (2026-08-02: the map-range bound + 0.5 s eviction now referee
  this -- see the addendum above; the residual risk is a wrong-TIGHT bound on the thin
  1-2 and 14-15 edges.)
* speed_est unfitted (speed_conf 0.0); coasting is rotation-only.

---

## 2026-08-02 addendum -- three HUD-found bugs (residual gate, interior test, IPPE twin)

Claire flew the HUD live and reported three failures. All three are fixed in `producer.py`
(+ the small matching diff in `hud.py`, which imports `interior_colour` / `is_decoration` /
`n_corners_in_frame` from producer so the two cannot drift). Scripts: `residgate.py`,
`residgate2.py`, `cornergate.py`, `interiorgate.py`, `interiorgate2.py`, `gatetilt.py`,
`tiltstrat.py`, `bugfix2_metrics.py`, `bugfix3_metrics.py`, `bugfix3_strips.py`.

### Bug 1 -- the residual gate starved the normal at close range

The reported symptom (a 0.87 px residual on a ~500 px gate refused) is a **HUD** bug, not a
producer bug: `hud.PNP_RESID_GATE` was 0.3 px ABSOLUTE while the producer's gate was
`max(6, 0.05*size)` -- 20x looser. This matters, because the first fix attempt tightened
the PRODUCER to the derived size-only rule and **regressed every headline number**:

| run | valid | pose_valid | over-bound | final-15m normal_valid |
|---|---|---|---|---|
| AFTER (association fix only) | 90.8% | 63.2% | 2.24% | 79.7% |
| BUGFIX2 (size-only max(0.3,0.015s)) -- REJECTED | 84.5% | 39.2% | 1.65% | **50.5%** |
| BUGFIX3 (shipped) | **91.0%** | 61.1% | **1.69%** | 79.4% |

The size-only rule made Bug 1's own target metric worse by 29 points. Root cause: at close
range the quad is CLIPPED, so its residual is measured against **extrapolated** corners.

Claire's insight fixed it: for a close gate, size is the wrong trust variable -- what
decides whether the pose is constrained is **how much of the gate is observed**. Measured
in `cornergate.py` on the pose-level referee (PnP on predicted quad vs PnP on GT quad),
VQ1 val, catastrophe = range err >30% or direction >10 deg:

| n corners in frame | n | resid med | good % | CATASTROPHE % |
|---:|---:|---:|---:|---:|
| 0 | 3 | 18.54 | 0.0 | 100.0 |
| 1 | 36 | 1.01 | 30.6 | 44.4 |
| 2 | 268 | 0.27 | 82.5 | **3.4** |
| 3 | 34 | 0.32 | 97.1 | 0.0 |
| 4 | 1918 | 0.05 | 95.2 | 1.2 |

**The cliff is between ONE and TWO in-frame corners, not at three** -- two corners of a
known 1.5 m square already pin the pose. Rule sweep (pooled VQ1 val + VQ2 hand):

| rule | catch | keep good | keep good >=120 px |
|---|---|---|---|
| max(6, 0.05s) (pre-fix producer) | 0.0% | 100.0% | 100.0% |
| max(0.3, 0.015s) (BUGFIX2) | 29.4% | 98.5% | 78.1% |
| corners>=2 alone | 37.3% | 99.5% | 95.8% |
| **corners>=2 AND max(0.3, 0.05*size)** (shipped) | **49.0%** | **99.5%** | **95.8%** |
| corners>=3 AND max(0.3, 0.05s) | 62.7% | 89.2% | 35.4% |

The shipped row dominates both size-only rules on all three columns. `corners>=3` would
buy 14 points of catch for 60 points of big-gate coverage -- the wrong trade, since
big-gate coverage IS the bug. Constants: `MIN_CORNERS_IN_FRAME=2`, `PNP_RESID_REL=0.05`.
The HUD now shows `c<n>` (in-frame corner count) per quad so a refused big gate reads as
"clipped past the limit" rather than as an unexplained refusal.

### Bug 2 -- interior test rejected real gates against the LIT CEILING

Decoration interiors are a UNIFORM bright fill; a real aperture against a lit ceiling still
contains dark gaps. Escape hatch: `bright AND dark_frac < IN_DARK_FRAC`, `dark` = V < 90,
frac 0.05. On Claire's hand labels (confident labels only, `interiorgate2.py`):

| rule | deco rejected | true-gate loss |
|---|---|---|
| bright only (old) | 96.8% | 9.71% (27/278) |
| **bright & dark90<0.05** (shipped) | **96.6%** | **1.44% (4/278)** |
| bright & dark120<0.02 | 96.6% | 0.72% (2/278) |

Target was >=95% deco rejection at <=2% true loss: met. (`dark120<0.02` is marginally
better on this set and was NOT adopted -- it triggers on a smaller pixel fraction, so it is
noisier on small quads, and the margin does not justify the churn.) Verified visually in
`bugfix2_upward_strip.png`: 9/9 tiles are genuine gate apertures against bright background
that the old rule threw away.

**Read the two Bug-2 numbers as answering different questions.** On the LIVE lap's
upward-looking frames the hatch flips only 11/1877 rejections (0.6%) -- most bright-interior
rejections there are genuine decoration. On the population that matters (actual gates seen
against the lit ceiling) it cuts loss 9.7% -> 1.4%.

### Bug 3 -- IPPE twin ambiguity

Shipped: prefer the near-vertical twin, gravity-referenced, gated on `attitude_conf >= 0.3`
and a >=8 deg tilt gap between twins, as a FALLBACK after temporal consistency. Where both
twins are near-vertical, `normal_valid` stays False per the contract.

**`VERTICALITY_EXEMPT_GATES` corrected {8} -> {9}.** Gate 8 measures vertical in both
channels; the earlier note naming it came from a misfiled screenshot. Exempting the wrong
gate is doubly wrong -- it withholds the prior where it works and applies it where the plane
really is tilted.

**The two tilt channels agree on WHICH gate but not on HOW MUCH, and this is unresolved.**
The PnP-normal elevation channel reads gate 9 at ~21-24 deg on close/large views. The
PnP-free channel (`gatetilt.edge_tilt`: image edge line vs gravity, no PnP, no compass)
reads gate 9 at 3.3 deg in the best-view bucket (size >=60 px, IQR 2.10, per-session 2.7 /
6.6 / 4.2). Gate 9 is the standout in BOTH -- every other gate falls to 0.7-2.3 deg as views
improve -- so the exemption is safe either way, but the magnitude is not settled. Likely
cause of the disagreement: the edge-line channel is **blind to a gate that leans about a
horizontal axis perpendicular to the line of sight when viewed head-on**, which is exactly
the geometry of an approach gate; and `|elevation|` from PnP is positively biased precisely
where conditioning is worst (head-on), so its rise with proximity is also what a twin
artefact looks like. Neither channel settles it alone. Do not quote a gate-9 tilt magnitude
without saying which channel produced it.

**The grid-bin azimuth snap was DROPPED, not deferred.** The 14 map-accepted gate azimuths
spread 11-88 deg mod 90 with deviations up to +-44 deg from any grid bin and show no
clustering. The claim may still be true visually with our azimuths too noisy to see it, but
a quantiser with no support in the data converts a wrong pose into a confidently wrong one
-- the exact failure Bug 3 is about.

**Sign flips did NOT improve** (`bugfix3_metrics.py`, twin swaps counted as sign reversals
of the normal's component perpendicular to the line of sight, consecutive `normal_valid`
frames <0.2 s apart): 3.71% (AFTER) -> 4.02% (BUGFIX3), n~3700 pairs each, ~1 SE apart, so
unchanged rather than worsened. The verticality prior fires only as a fallback and rarely
clears its margin. Honest state: Bug 3's wrong-side symptom is addressed by the corrected
exemption plus the existing temporal-consistency path, but the flip rate is not yet
evidence of improvement. The brief asked for flips "vs map-predicted planes"; the map's
per-gate azimuths carry MAD 7-18 deg and cannot referee a sign at this precision, so the
intra-track count is the reported number.

### Regression set (all green)

* lap 20260801-121520: valid **91.0%** (was 90.8), over-bound **1.69%** (was 2.24),
  **17/17** crossings, approach monotonicity median **0.95** min 0.72 (unchanged).
* final-15 m normal_valid: **79.4%** vs 79.7% baseline (BUGFIX2's 50.5% repaired).
* strafe 6-7 static-pair canary: **16.51 m MAD 0.18, n=115** vs 16.50 / 0.18 / 118 --
  frame geometry unmoved.
* pose_valid 61.1% vs 63.2%: the intended cost of catching 49% of pose catastrophes
  (pre-fix rule caught 0%).
* Strips: `bugfix3_regimes.png` -- big-near gates at 2.0-6.0 m carry a normal with
  staleness 0.00 s (Bug 1's target, re-measured not coasted); act=12 gives sensible
  approach-side normals. CAVEAT: that sheet's "upward" row is a BAD SELECTOR -- the
  ceiling-brightness proxy saturated on frame-filling orange gates, so those three tiles
  are mid-crossing, not upward views. Bug 2's visual evidence is
  `bugfix2_upward_strip.png`, not that row.

---

## 2026-08-02 addendum -- the IPPE twin, resolved PER TRACK instead of per frame (`normalfuse.py`)

Claire's insight, and the last unfixed perception bug. Attention pointed at the WRONG SIDE
of a gate (caught live at act=12), and the verticality prior did NOT fix it: intra-track
normal sign-flip rate 3.71% -> 4.02%, unchanged within noise.

### Why the per-frame resolver could not work

The old resolver picked a tilt branch by comparing the new candidates against **the
producer's own previous pick**. That is a feedback loop with no external referee: a wrong
first pick is self-confirming, and no amount of work on the tiebreak changes that. The
verticality fallback could not rescue it either, and there is an analytic reason it
"rarely cleared its margin": IPPE's two solutions are related by a reflection of the plane
about the line of sight,

        n_false(t) = 2 (n_true . l(t)) l(t) - n_true                            (1)

so for a HORIZONTAL LOS and a VERTICAL gate face, the false twin is horizontal too --
verticality is structurally blind in exactly the geometry of a level approach.

**Equation (1) is verified on our own data, not assumed**: over 8015 logged two-solution
frames of the lap, the angle between `reflect(c0, los)` and `c1` is **0.12 deg median /
0.63 p90 / 3.9 p99**.

### The fix

A gate is a static world object, so `n_true` is a FIXED direction. Rotate both candidates
into a **track-common frame** (the body frame at track birth, carried by the gyro-integrated
body rotation the producer already maintains for coasting) and (1) says the true branch is
the same vector every frame while the false branch SWINGS, because `l` swings. Two
hypotheses are carried per track, each a slow running mean plus a predictive
log-likelihood; the winner is the one whose assigned candidate keeps landing where its own
mean already was. Nothing consults the previous PICK, which is what removes the feedback.

Deliberate design points:
* the hypothesis MEANS are slow averages, not trackers. If a mean chased its assigned
  candidate, the false hypothesis would follow the swinging twin and the self-confirming
  loop would simply reappear one level up.
* consistency evidence is accrued **per degree of LOS motion**, not per frame: dwelling at
  30 Hz supplies correlated noise, not information.
* priors (verticality; `course_vq2.json` `yaw_race_bisector_deg` via the map anchor and the
  compass) go into the SAME likelihood so they trade off instead of overriding, and their
  net contribution to the ratio is hard-clamped (`PRIOR_LLR_CAP = 6` nats).
* the map prior uses the BISECTOR, never `yaw_deg` -- `yaw_deg` was measured from PnP
  normals plus the compass, i.e. from the very channel the prior is meant to referee, and
  using it would close the loop again. The bisector comes from gate positions.
* no locking: reset on re-association past the coast horizon, halving on a leader
  overtake, death with the track on a crossing or race reset.

Emission: accumulator decided (LLR >= 4 nats AND baseline >= 8 deg AND >= 4 samples) ->
accumulated normal; else a frame whose 2nd IPPE solution is >1.6x worse is still allowed
(that is the REPROJECTION deciding, not a guess); else `normal_valid` False per contract.

### Measured, lap `20260801-121520-vq2-lap-0-15` (6303 frames)

| run | lateral sign flips | normal_valid overall | final-15 m |
|---|---|---|---|
| BUGFIX3 (per-frame resolver) | **4.02%** (146/3632) | 67.8% | 79.4% |
| **NORMFUSE (shipped)** | **1.08%** (44/4084) | **75.7%** | **83.8%** |
| NOMAP (map prior off) | 1.16% (44/3796) | 71.4% | 81.5% |
| NOPRIOR (map prior as evidence, no prior-ONLY decisions) | 1.23% (47/3807) | 71.3% | 82.8% |

**Flips 4.02% -> 1.08% on MORE normal_valid pairs (3632 -> 4084), and availability went UP,
not down** -- the consistency was not bought by refusing.

Source mix on valid current-slot frames: accum-consistency 54.5%, accum-coast 10.9%,
unambiguous 12.7%, accum-prior 6.1%, accum-cons+prior 4.7%, refused 11.2%.

### Separation vs angular baseline (the regime question)

Baseline = angular spread of the LOS in the track-common frame. **Body rotation contributes
NOTHING** -- it is removed by the common frame, and (1) depends only on `l` in the world.
The LOS moves only through TRANSLATION relative to the gate, and a dead-straight approach
translates ALONG `l`.

| baseline (deg) | n | median LLR_cons | p90 | decided% | decided% (no map prior) |
|---|---|---|---|---|---|
| 0-2 | 913 | 0.00 | 0.29 | 5.0 | 0.0 |
| 2-5 | 899 | 0.21 | 1.27 | 27.0 | 0.0 |
| 5-10 | 855 | 1.52 | 6.48 | 50.8 | 32.4 |
| 10-20 | 1367 | 4.87 | 15.78 | 64.2 | 66.6 |
| 20-35 | 1925 | 4.43 | 35.36 | 80.3 | 67.7 |
| 35-60 | 1502 | 33.07 | 60.36 | 95.5 | 88.4 |
| 60+ | 554 | 5.81 | 39.42 | 80.5 | 84.5 |

Consistency evidence crosses the 4-nat threshold at **~10-20 deg of LOS baseline** and is
decisive by 35 deg. Below 5 deg it supplies essentially nothing (median LLR 0.00-0.21) and
every decision there comes from the map prior. The synthetic sweep in
`normalfuse._selftest` agrees: LLR 0.47 at 4 deg, 1.65 at 10 deg, 22.1 at 25 deg.

Self-agreement (does the verdict at baseline b match the same episode's final,
best-supported verdict?) is **100% in every populated bin** over 1700 reference episodes.
Read that as "no within-episode reversals were observed", not as accuracy -- the same
accumulator produces both sides of the comparison.

### WHEN THIS FIX DOES NOT HELP -- stated plainly

* **A dead-straight, high-speed final approach generates no baseline and the mechanism has
  no signal at all.** That is structural, not a tuning limit.
* On THIS lap the final approaches have plenty (baseline p10/median/p90 = 3.8/34.5/79.2 deg
  in the last 15 m, 80.0% above the 8 deg gate; last 8 m: 5.7/34.5/81.9, 88.6%). **That
  number is optimistic and must not be read as a race-pace result** -- this is a slow
  exploratory lap with dwelling, hovering and lateral drift, exactly the recording bias the
  brief warned about. The 0-5 deg bins (25% of logged frames even here) are the straight-in
  regime, and there only the map prior acts.
* The fallback in that regime is the map prior alone, which is what `ALLOW_PRIOR_ONLY`
  buys: +4.4 pts overall availability and +1.0 pt final-15 m for +0.15 pt of flips
  (NORMFUSE vs NOPRIOR). Turning it off (`--no-prior-only`) falls back to `normal_valid`
  False, which is the contract-safe choice if the compass is ever suspect.
* The prior is WEAK and its agreement is worth stating honestly: the accumulated normal's
  levelled azimuth vs `yaw_race_bisector_deg` is **median 26.9 deg, MAD 17.3, p90 75.1**
  over 4993 decided frames. Per gate the disagreement is mostly a STABLE OFFSET (MAD
  0.8-5 deg at gates 0,1,4,5,6,7,9,11,14,15) -- i.e. the accumulated normal is
  self-consistent and the offset is the bisector's own error (median 10.9 deg vs the map's
  measured azimuths, 48 deg tail at gate 7) plus compass/anchor error. Outliers: gate 3
  (64.6 deg), gate 14 (43.7), gate 10 (37.3, MAD 22).
* Because the flip rate is a WITHIN-TRACK consistency measure, it cannot detect a
  systematically wrong but STABLE pick. The strips below are the only check on that, and
  there is no VQ2 ground truth for gate normals.
* If the compass or the map MIRROR (`MapModel.s`, known unstable run-to-run) is wrong, the
  prior points at a reflected azimuth and can push a no-baseline track from "refuse" to
  "wrong". `PRIOR_LLR_CAP` bounds that: it can never override a consistency verdict that
  has baseline behind it.

### Regression set (all green, nothing moved)

* lap: valid **91.0%** (unchanged), pose_valid 61.2% (was 61.1), over-bound **1.67%** (was
  1.69), **17/17** crossings, approach monotonicity median **0.95** min **0.72**.
* strafe 6-7 canary: **16.51 m MAD 0.18, n=115** -- identical to BUGFIX3.
* latency unchanged (the accumulator is microseconds per track per frame): total median
  29.2 ms on the lap.
* `producer.py --selftest` passes; `normalfuse.py` carries its own synthetic selftest.

### Visual verification (read back, not assumed)

`producer_runs/normalfuse_act{12,3,6,15}.png` -- winner solid green, rejected twin thin
magenta, with `L<llr> b<baseline> <source>` per track.
* **act=12** (Claire's report): at 12.1 m the gate is strongly foreshortened with its LEFT
  edge nearer; the winner points LEFT-and-toward-camera, the rejected twin down-right.
  That is the approach side, and it is the same side across the whole approach.
* **act=6**: near head-on, 15.7 -> 10.8 m, all six tiles `accum-consistency` with the same
  winner orientation -- no flicker, which is the point.
* **act=3**: oblique pass, 8.1 -> 3.9 m, all `unambiguous` (a very oblique gate separates
  the branches by reprojection alone) and stable.
* **act=15**: long dwell at 12-14 m, stable, with honest `refused` frames mixed in.

### HUD

`hud.py` (additive): the rejected twin is drawn thin magenta alongside the accepted normal
whenever a normal is asserted, with the LLR margin, the LOS baseline and the decision
source. Without the loser drawn, a wrong-side normal looks exactly like a confident one --
which is how the act=12 symptom went unnoticed. Also, at Claire's request, every drawn gate
now leads with its ABSOLUTE RACE INDEX; the CURRENT slot is drawn white/bold with a `CUR`
tag and a highlight box, lookahead slots as `+1`/`+2`, and quads from the HUD's own
independent recompute are labelled `unassigned` so they cannot read as identified gates.

Files: `perception/normalfuse.py` (accumulator + selftest),
`perception/normalfuse_metrics.py` (all numbers above),
`perception/normalfuse_strips.py` (the sheets). Runs:
`producer_runs/lap-121520-{NORMFUSE,NOMAP,NOPRIOR}/`, `producer_runs/strafe67-NORMFUSE/`.
`producer.py --normal-debug` dumps `normal_log.npy`; `--legacy-tilt`,
`--no-map-normal-prior`, `--no-prior-only` reproduce the ablations.

---

## 2026-08-02 addendum (2) — the drag speed model, translation-coasting, and two slot swaps

Race-pace data arrived: `20260802-153626` (53 s, one continuous run, mid-lap collisions and
backtracking) and `20260802-161838` (179 s containing EIGHT failed attempts; the real lap is
the last 63.3 s). Both are aggressive in ROTATION — 25% and 37% of frames above 1 rad/s —
which turns out to be the axis that matters here, more than speed.

### Segmentation first, and it is reproducible

`producer.py --from-last-reset` trims a replay to the data after the final sim reset. A reset
is `active_gate_index` dropping; the rule is **the last reset that leaves at least 5 s behind
it**, because `161838` ENDS on a reset and the naive rule selects an empty tail — every metric
then returns n=0, which reads exactly like a clean run. The cut is printed with its frame:

    161838: cut at t=115.58 s (gate 3 -> 0), frame 00136420.jpg, 3468 of 5368 frames dropped,
            63.3 s and 15 crossings remain
    153626: NO reset in the file. Its collisions and backtracking are mid-lap.

### The drag law: quadratic in the IN-PLANE speed, not the total speed

`interface.SelfObs` documents the physics — thrust acts along body -z by definition, so the
horizontal body accelerometer components measure only drag. Fitting that against VQ1 truth
velocity (`dragfit.py`, `dragval.py`, `dragfit_report.txt`) gives

    a_body[0:2] = -k * |v_xy| * v_xy        k = 0.04249 /m        R2 0.956

**The in-plane form is the whole result.** Binning the implied coefficient by TOTAL speed
makes it saturate above ~14 m/s and the law look broken (R2 0.826, per-session k spread
2.35x); binning by |v_xy| makes it flat from 2 to 25 m/s (R2 0.956, spread 1.31x). The
airframe has its own coefficient in the rotor plane. It matters beyond goodness of fit,
because it makes the inversion algebraic with no assumption inside it:

    |v_xy| = sqrt(|a_xy| / k)

which is the magnitude belonging to the bearing `_selfobs` already published. No integration,
nothing to diverge — the same standing `vel_bearing_rad` has.

Leave-one-SESSION-out (a whole session, so no split boundary can leak the way the gatenet
label edit did on 08-01): **median 0.53 m/s, p90 0.88 m/s, ~7% relative, flat across every
band from 2 to 34 m/s.**

### What is refused, and why that is the honest answer

The TOTAL speed needs the body-z velocity component. It is not observable: thrust sits on
that axis, and `pressure_alt` is nan in every recording. Closing the system with a
world-horizontal-velocity assumption (`v_body . g_hat = 0`) was fitted and **rejected**: LOSO
p90 14.96 m/s with a -14.3 m/s bias in the 22-40 m/s band, because the recorded flight-path
angle has a 26-34 deg median and only a third to two thirds of frames sit under 20 deg. The
code for it survives in `dragmodel.estimate()` for the record; the producer calls
`inplane()`. So `speed_est_mps` carries the IN-PLANE speed and says so, `(vel_bearing_rad,
speed_est_mps)` are the body-horizontal velocity vector, and nothing claims to know the climb
rate.

### Two VQ2-native referees (the fit is VQ1-derived, so it needs one)

**Leg odometry — the strong one.** Integrate the drag speed between consecutive crossings and
compare with the map's MEASURED edge. Path must exceed chord, and is understated because the
vertical component is missing. Over the two race laps, **24 legs, median path/chord = 1.02
(p10 0.88, p90 1.17)**. It is sharp on scale: k x2 gives 0.72, k /2 gives 1.44. Nothing in it
touches VQ1, pose telemetry or the camera.

One outlier worth someone's attention: **leg 10-11 reads 0.53 in both laps** (chord 33.88 m,
path ~18 m). Path below chord is geometrically impossible, so either the map's longest edge is
overestimated or that leg involves a large altitude change the estimator cannot see. Flagged,
not resolved.

**Gate-PnP range closure — underpowered, and the first version was a manufactured null.**
Consecutive-frame differencing gives corr -0.09, purely because PnP range noise (~3% of range,
0.45 m at 15 m) is four times the per-frame closure at 5 m/s. Windowed, it is consistent
(median error 0.37 m/s in the 2-5 m/s band) but yields only ~21-31 independent windows over a
2-5 m/s span — no dynamic range, so its correlation means nothing. Superseded by leg odometry.

### The purpose-flown straight-fly session: the segmentation was INVERTED, and k is unmoved

**Corrected 2026-08-02 (second pass). The paragraph that stood here said the opposite of the
truth and it is worth keeping the correction rather than the conclusion.** Claire's note on
`20260802-163548` says "marker 1 indicates when straight fly/drift begins". The first pass
overruled her from the IMU alone: it read the pre-reset segment as 42 s of straight flight and
the post-marker segment as a parked aircraft, and it validated on the frozen one.

The camera says the reverse, and not marginally — mean absolute difference between consecutive
frames, 160x120 greyscale, which involves no IMU at all:

| segment | mean \|dI\| | verdict |
|---|---|---|
| t 3.5-47.3 s (pre-reset) | **0.073** | the image does not change for 44 s |
| t 55-82 s (post marker 1) | **5.63** | real motion, to the end of the session |

Looking at the frames settles it in one glance: at t=20 s and t=45 s the view is a single flat
orange surface filling the frame, pixel-identical between them. The aircraft took off, hit
something at t=2.1 s (one collision episode), came to rest jammed against it with the lens
blocked, and stayed there until Claire reset the sim at t=46.76 s. Two facts independent of
both camera and accelerometer agree: `active_gate_index` advances **0 -> 1 at t=53.97 s** — a
gate is crossed after the reset, which nothing parked on the pad can do — and `events.jsonl`
shows disarm at 47.3 s and re-arm at 49.3 s, i.e. a second flight, the one the markers annotate.

**WHY THE IMU COULD NOT SETTLE IT, which is the transferable part.** Steady flight at CONSTANT
VELOCITY has zero net acceleration, so the accelerometer reads pure gravity in the body frame:
a constant vector at the trim tilt, |a| = g, and a near-zero gyro. A parked aircraft on the
tilted VQ2 pad reads a constant vector at the pad tilt, |a| = g, near-zero gyro. Here they
match to three digits:

    parked on the pad   t 48-52 s   a = (-2.999, +0.001, -9.340)   tilt 17.8 deg
    the real drift      t 62-82 s   a = (-3.000,  0.000, -9.338)   tilt 17.8 deg

No statistic of the IMU separates these. In particular "how many IMU rows repeat exactly"
cannot, because *repeated rows are what a steady state produces* — and the figure quoted for it
(79.6% of post-marker rows being one parked triple) was wrong anyway: it is 8.2% post-marker
and 7.3% over the session, because it counted rows matching a presumed value rather than
genuine stalls. **Constant velocity and a tilted park are accelerometer-indistinguishable. An
independent referee is required, and the camera is the one that was already in the recording.**

That referee is now a reusable function, `staticdet.moving_mask(session, t)` — see below.

**What the corrected reading changes about the coefficient: nothing, to five decimals.**
`163548` never entered the fit and could not: VQ2 streams no velocity, and the drift holds one
speed. The fit is VQ1-only and comes out at `k = 0.04249` before and after the static cut is
applied to every fit session (it removes 328 frames, 5.4 s, from `20260731-150712` and no
frames anywhere else — `dragval.load`'s existing airborne cut on truth altitude was already
doing this job on VQ1). Speed range covered 2.3-34.3 m/s total, 1.1-25.5 m/s in-plane; LOSO
`|v_xy|` median 0.53 m/s, p90 0.88, unchanged.

The **uncertainty** on k, which was not stated before: the OLS standard error is 4e-5 and is
meaningless (61 Hz residuals are not independent). Resampling whole SESSIONS gives
**k = 0.0422 +- 0.0016, 95% CI [0.0371, 0.0441]**, and the per-session fits span 0.0366-0.0478
(1.31x). So k is known to roughly +-4%, not to four digits.

**The session's value as a referee is much smaller than hoped, in both directions.** The frozen
segment has no motion to measure. The real drift is flown down an empty hangar aisle: replayed
through the producer it yields **0.1% pose-valid frames**, so range-closure cannot run there
either. What survives is a single fly-by of gate 1 — 12 fresh detections over 0.43 s, with
|gyro| exactly 0.0000 so body bearing is inertial bearing. Fitting straight-line geometry
(`driftcheck.flyby`) gives **v = 6.5 m/s against the drag model's 8.25 m/s, ratio 0.79**. It is
recorded, and it is NOT decisive: v scales linearly with the closest-approach distance, which
comes from the size-derived range of a gate at the frame edge, and the sight line drops 6 deg
of elevation across the 0.43 s, so the flight is not in the plane the fit assumes. Those
systematics are larger than the resampling spread. **Leg odometry arbitrates** — 24 legs,
8-25 m/s, using only crossing times and the map, nothing from `163548` or VQ1 —
and it reproduces exactly: **median path/chord = 1.02** (p10 0.88, p90 1.17), with `k x2 ->
0.72` and `k /2 -> 1.44`. Path must be >= chord and is understated, so 1.02 is where a correct
coefficient lands and the referee still carries the verdict.

**The ask for a better speed model is unchanged and now better specified:** a VQ1-build flight
with speed VARIATION. A VQ2 straight-fly can only ever check one speed, and to check even that
it needs a gate held in view for seconds, not 0.43 s.

### The static/parked detector (`staticdet.py`), and what it caught downstream

`moving_mask(session, t_s)` -> bool per query time, True where the aircraft is moving through
the scene. Mean |dI| between consecutive 160x120 greyscale frames, median-filtered over 0.5 s;
static requires the value to stay below `M_STATIC = 0.25` for at least `STATIC_MIN_S = 2.0 s`,
because parked is a sustained state and a single glance at an untextured wall is not.

Calibrated on the two known windows above and scored against VQ1 truth velocity:

| rule | VQ1 truth-moving (>2 m/s) called parked | 163548 static caught | 163548 moving lost |
|---|---|---|---|
| 0.25, run >= 2 s | **1.16%** | 100% | 0% |
| 0.50, run >= 2 s | 3.12% | 100% | 0% |
| 0.50, no run rule | 6.74% | 100% | 0% |

Any threshold in 0.15-0.30 gives the same verdicts — the gap between frozen (p99 0.125) and
moving (p10 2.5) is ~20x. The residual 1.16% is one session, `20260731-150712` (the skid card),
where the camera spends moments against a featureless surface at speed; the run-length rule
does not remove it because those stretches genuinely last seconds. It abstains (returns
"moving") on sessions with no frames, so it can only ever remove frames it has evidence
against. Its claim is deliberately negative: *the image is not changing, therefore the aircraft
is not moving through the scene.*

**What it caught.** Over the 42 s in which `163548`'s aircraft did not move at all, the
producer reported `speed_est_mps` **8.06 / 8.08 / 8.10 m/s (p10/p50/p90) at speed_conf 1.00**.
`sqrt(|a_xy|/k)` has no zero: a stationary airframe on any tilted surface is reported as
several m/s at full confidence, and `COAST_TRANSLATE` will translate every coasted track at
that speed. This is not a new bug introduced by anything here — it is the standing behaviour of
the estimator, made visible. Nothing in the IMU can catch it; `staticdet.moving_mask` can, and
that is the intended use if a live parked-detection is ever wanted.

### Downstream re-check after the correction

Because `k` did not move, nothing downstream moved. Stated explicitly so it is not re-derived:
`speed_est_mps` behaviour unchanged; the translation-coasting tables above for `153626` and
`161838` stand as measured; the strafe 6-7 canary stands at **separation median 16.51 m, MAD
0.18**. The referee was carrying the verdict all along, which is the good outcome here — the
inverted segmentation never reached the coefficient because `163548` was never a fit input.

### Translation-coasting

`COAST_TRANSLATE`: coasted tracks are now translated by the drag-derived own velocity as well
as rotated by the gyro (`pos_body -= v_body*dt`, in-plane only). `range_sigma_m` is charged
`COAST_SIGMA_K` x the accumulated coast distance rather than a flat 10 m/s. A contact guard
(`A_MAX = 52 m/s^2`, the drag at the fastest frame ever recorded) REFUSES rather than filters:
a collision reads 879 m/s^2 in `153626`, which would otherwise inject 143 m/s of translation.

Measured on `153626` (both fixes off -> both on; the fixes are separable by
`--no-coast-translate` / `--no-joint-slots`):

| metric | before | after |
|---|---|---|
| current gate valid | 89.3% | **95.3%** |
| pose_valid | 43.3% | **59.3%** |
| mean staleness | 0.337 s | **0.203 s** |
| out-of-bound | 3.59% | **1.32%** |
| reacquire miss, image | 21 px (p90 219) | **9 px** (p90 139) |
| reacquire miss, metric | 3.25 m (p90 9.72) | 3.37 m (p90 8.24) |
| slot ordering inversions | 10.54% | **7.43%** |
| approach monotonicity | 0.93 / min 0.85 | 0.90 / min 0.78 |

and on `161838` (the 63.3 s post-reset lap, 15 crossings, 37% of frames above 1 rad/s):

| metric | before | after |
|---|---|---|
| current gate valid | 96.6% | **98.2%** |
| pose_valid | 62.4% | **65.5%** |
| mean staleness | 0.143 s | **0.117 s** |
| out-of-bound | 4.14% | **3.70%** |
| reacquire miss, image | 13 px (p90 177) | **7 px** (p90 145) |
| reacquire miss, metric | 4.27 m (p90 15.14) | 5.80 m (p90 17.62) |
| slot ordering inversions | 19.65% | **10.89%** |
| approach monotonicity | 0.97 / min 0.80 | 0.89 / min 0.80 |

**The metric-space miss did not move, and the honest reading is that it is the wrong metric.**
It is computed against `480/size_px` — the size-only range of the re-acquiring detection —
whose own error is metres, so it measures the reference more than the prediction. The
image-space miss halving is the real signal, and the strips confirm it directly: at v = 5.2 m/s
the coasted and measured markers separate by 34 px before and 9 px after, while at v = 0 the
tiles are pixel-identical. The fix acts exactly where speed is nonzero, which is the signature
it should have.

**Monotonicity regressed consistently** (0.93 -> 0.90 on `153626`, 0.97 -> 0.89 on `161838`,
unchanged at 0.95 on the gentle lap). Partly population -- 6 points more of `153626` is now
valid, including harder frames -- but not entirely. It is the one number that moved the wrong
way on both race laps, and it is the metric this change is most likely to be quietly hurting.

**RESIDUAL REGRESSION, not explained away.** The gentle standing lap `20260801-121520` reads
**91.0% -> 89.6%** current-gate valid. Out-of-bound improved there (1.67% -> 1.42%), staleness
improved (0.162 -> 0.145 s), monotonicity is unchanged, and only 2 joint repairs and 0
evictions fired -- so the usual suspects are ruled out. A plausible reading is that some of the
lost "valid" is wrong tracks now being refused rather than coasted (fresh slot-geometry
violations on that lap fell from 31.7% to 24.1%), but that is a HYPOTHESIS, not a measurement,
and it should not be treated as settled. The trade being accepted is -1.4 points on a gentle
lap against +6.0 and +1.6 on the two race laps.

Two thresholds were moved on evidence during this work, and both are worth knowing about
because each was the fix hurting in a regime it was not meant for:
`COAST_SPEED_CONF_MIN` 0.25 -> 0.55 (translating on a noise-floor speed estimate perturbed
predictions enough to lose associations on near-hover laps), and the joint repair's
feasibility verdict restricted to INNER pose-bearing detections (judging on `range_size_m`,
known biased long, convicted frames that were never swapped -- 61 spurious repairs on the
gentle lap, down to 2).

**The diagnosis itself was only weakly confirmed.** `miss vs gap*speed` is +0.28 on `153626`
(n=42) and +0.35 on `161838` (n=69) — the right sign and the largest of the three
correlations, but not decisive at that n. On the strafe canary it is +0.80 (n=10). The case
for the change rests on the downstream metrics, not on the correlation.

### attitude_conf: the p10 = 0.00 is earned, and the usual explanation is wrong

`conf = conf_mag * conf_turn`. `conf_turn` is a positive rational function and **cannot reach
zero** (0.32 at 3 rad/s), so every zero comes from `conf_mag`. Measured on `153626`: of the 924
zero-confidence samples, **100% have conf_mag == 0**, and `|a|` on them has median 5.78 m/s^2
against g = 9.81. Gravity genuinely is not measurable there. Not a bug.

**But it is not centripetal acceleration doing it.** `|gyro|` on the zero samples has median
0.55 rad/s, and p10 is 0.00 in EVERY gyro band including 0.0-0.5. The collapse is driven by
LINEAR acceleration (thrust modulation), which is what `conf_mag` was written for; the apparent
correlation with `|gyro|` in the replay report is incidental, both rising together in
aggressive flight.

**Collisions do not poison it.** Around the one episode: `|a|` max 890 m/s^2 during, yet conf
median 0.29 during vs 0.42 before, 0.38 at +0.5-2.5 s and 0.42 at +2.5-6.5 s. The gravity
filter's trim rate is gated on the same `|a|-g` mismatch (alpha 0.008 -> 0.001 -> 0.0001), so a
contact spike makes it stop trusting the accelerometer rather than be dragged by it. Recovery
is immediate.

**Consumers.** Properly gated: the PnP verticality prior (`VERTICALITY_CONF_MIN`) and
normalfuse's (`att_conf >= 0.3`). **Ungated:** `self.imu.level_rotation()` at the map-frame
anchoring and `_map_predict` call sites, and `hud.py:561` — these use the gravity vector with
no confidence check at all. Mitigating: `g` is gyro-propagated and only slowly trimmed, and the
sim gyro is near-noiseless, so it does not jump when conf collapses. Reported, not changed —
it is adjacent to the concurrent normal-fusion work.

**The coast rotation does NOT depend on attitude_conf.** It uses `ImuFilter._Rstep`, built from
the gyro alone. So the confidence collapse is *not* a second contributor to the reacquire miss.

### Two slot swaps (Claire's live catches) — and why the obvious fixes are wrong

Frame A: `active=1`, "g2 16.5m / g3 8.3m" with map edges 1-2 = 8.32 and 2-3 = 13.34 — lookahead
labels swapped. Frame B: `active=11`, "cur 28.0m / g12 8.0m" with 11-12 = 19.25 — the CURRENT
slot swapped with k+1. Both at 17+ deg/s yaw.

Three things had to be got right, and two of them are negative results:

1. **The map-range bound cannot do it.** The current bound is 1.3 x edge(k-1,k) = 44 m in frame
   B, so 8 m and 28 m both pass. It is structurally loose whenever the incoming edge is long.
2. **The triangle inequality cannot do it either.** `|r_a - r_b| <= d(a,b) <= r_a + r_b` is
   exact and needs no compass, pose or frame anchor — but **both sides are symmetric in r_a and
   r_b**, so it is structurally blind to a pure two-gate swap. |28-8| = 20 either way.
3. **Ordering can, but not as a hard rule.** Checking the map: d(k,k+2) > d(k,k+1) at all 15
   gates — except **gate 4, where the margin is 0.28 m** (22.74 vs 22.46). A tight ordering rule
   is a coin flip there, and a rule that fires wrongly at one gate is worse than the bug, since
   it converts a recoverable mislabel into a confident one. So ordering is applied with
   `SLOT_ORDER_TOL = 4.0 m`: no opinion where the course is degenerate, decisive on Claire's
   frames, which are inverted by 8.2 m and 20.0 m.

The fix is a **joint assignment over all three slots that runs only as a REPAIR**: if the greedy
assignment is geometrically feasible it is returned untouched. That design came from a measured
failure — the first version scored every permutation and took the cheapest, which improved slot
geometry (46.6% -> 41.7%) but pushed out-of-bound from 1.32% to 3.85%, because it re-decided
slot 0 on 117 frames where nothing was wrong. The current slot already has three referees built
for it this morning and they took out-of-bound 30.45% -> 2.24%. A pass that cannot fire on a
good frame cannot churn one. Second line of defence: a current track that stays farther than a
fresh lookahead detection for `CUR_ORDER_EVICT_S` is evicted — approach monotonicity used as a
live guard rather than a report.

Regression channel (`_slot_audit`), split by staleness because the first version counted
coasted ranges and reported an implausible 44-56%:

| session | metric | before | after |
|---|---|---|---|
| `153626` | ordering inversions | 10.54% | **7.43%** |
| `153626` | map geometry, fresh+pose slots | 44.03% | **17.76%** |
| `161838` | ordering inversions | 19.65% | **10.89%** |
| `161838` | map geometry, fresh+pose slots | 36.93% | **32.18%** |
| `121520` (gentle) | ordering inversions | n/a (pre-metric) | 11.18% |

(The "fresh" definition was tightened mid-work to require `pose_valid`, so the before/after
pairs above are on the final definition only where both were re-run; `161838`'s before uses
the looser one and its improvement is therefore understated rather than overstated.)

These only appear under aggressive motion: the same metric is near zero on the gentle mapping
laps, which is why two months of replay never surfaced it and Claire found it in one flight.

### Standing regressions

Strafe 6-7 canary unchanged: **separation median 16.51 m, MAD 0.18**, p10 16.14, p90 16.90 —
identical to the standing value.

### Files

`perception/dragmodel.py` (the model + the refused total-speed path), `dragfit.py` (VQ1 loader),
`dragval.py` (fit + LOSO, now with the image-motion cut), `dragfit_report.py` -> `dragfit_report.txt`,
`dragleg.py` (leg odometry referee -- the one with power), `dragreferee.py` (PnP closure,
underpowered), `staticdet.py` (the static/parked detector; `python3 staticdet.py` prints its
calibration and its validation against VQ1 truth), `driftcheck.py` (the 163548 post-mortem,
rewritten -- its first version had the segmentation inverted), `attcheck.py` (attitude_conf decomposition), `coaststrips.py` (verification
strips -> `producer_runs/strips/`).

---

## 2026-08-02 addendum — the transfer pass for control (`STATE_PERCEPTION_FOR_CONTROL.md`)

Answering `STATE_SURROGATE_FOR_PERCEPTION.md` section 9 items 4, 5 and 6. Full document at
the repo root; scripts `xfer_*.py`, plots `vercheck/xfer_*.png`, replays
`producer_runs/XFER-{gentle121520,fast153626,fast161838}/` (one producer snapshot, three
regimes: gentle exploratory lap, and two race-pace laps segmented to the real lap).

**`10-11 = 33.88 m` IS WRONG — the map's longest edge.** Found by the drag/coast agent's
independent channel and confirmed here. Integrating the drag speed model between crossings
gives a flown PATH length; path/chord must be >= 1 and is understated (in-plane speed only).
Median over 39 legs is 1.08; leg 10-11 reads **0.53 / 0.60 / 0.60** on the three laps that
fly it. Impossible, and altitude cannot rescue it (dz −1.75 m, and a climb pushes the ratio
UP). Mechanism, from re-auditing the rows: all 76 rows are ONE session, ONE contiguous
171-frame window, bearing spread **1.5 deg** — a single vantage, so the static-pair
viewpoint referee was never exercised; the far gate is 11–15 px against
`mapvq2.MIN_SIZE_PX = 14`; the reported dz drifts 1.3 m within the window. Looking at the
frames (`vercheck/xfer_leg1011.png`) settles the mechanism: three or four gates are strung
down-course at ~8 / ~22–25 / ~32 / ~41 m and the accepted rows pair the 8 m gate with the
41 m one, skipping two. Leading reconstruction — the far detection was **gate 12**: taking
33.88 @ 205.1 as the 10->12 vector and keeping 11-12 (19.67 @ 217.6) gives **10-11 = 15.30 m
@ 188.9 deg**, only gate 11 moves (19.6 m), path/chord becomes 1.18, and the m/station
outlier collapses from **57.6 to 26.0** (course median 19.2, p10-p90 15.2-26.7) — an
independent channel agreeing. The gate-13 alternative reconstructs 8.0 m and path/chord
2.24; refused. Practical consequence already observed: the producer's current-gate bound is
`1.3 x edge(k-1,k)`, so the inflated edge set a ~44 m bound at gate 11, too loose to reject
Claire's 28 m impostor. **A wrong map edge caused a tracking bug.** Not yet written into
`map_vq2.json` or `course_vq2.json` — the replacement is a reconstruction inheriting the
thin 11-12 edge, and one 10/11 co-visible hover from 12-18 m settles it. `xfer_pathcheck.py`,
`xfer_leg1011.py`.

**MOD-90 QUADRANTS: 15-16 PINNED, 2-3 corroborated, 6-7 not pinnable.** The branch is per
SESSION, so any branch-resolved session carrying a contested leg pins it. 15-16: one
full-bearing contour row in the branch-resolved `20260801-121520` (t=196.3 s) reads 167.52
sketch-frame at d 21.36 (accepted 21.01) against the chosen 169.36 — **1.84 deg with the
runner-up 88 deg away**, and it also breaks the mod-180 sign tie the sketch prior could not
(its provenance line is literally "err 16.5 vs next-best 16.5"). Branch continuity over the
200 s lap verified: 30 re-anchors, snap residual median 1.76 / p90 5.69 / **max 9.76 deg**,
worst case a 4.9 s blind stretch still 35 deg inside the 45 deg half-fold. 2-3: two gatenet
rows in the branch-resolved `20260801-202923`; both select the chosen branch (good row by
28.8 deg, runner-up 61 deg) — corroboration, not comfort. 6-7: **no branch-resolved session
holds a 6-7 row and it is a bridge**, so nothing existing can pin it. Cost of a wrong
branch: 2-3 rotates 14 gates (median 132 m), 6-7 rotates 10 (103 m), 15-16 one (30 m).
`xfer_quadrant.py`, `xfer_quadrant2.py`, `xfer_unwrapgap2.py`, `xfer_branchcost.py`.

**MOTION BLUR IS NOT A THING IN THIS SIM, and the raw gyro curve is a framing confound.**
Control asked for a detection-vs-gyro curve to parameterise a `p_detect` blur term. Pooled
over three regimes, P(fresh) falls 0.81 -> 0.62 from 0 to 1.75 rad/s — but restricted to
gates well inside the frame (|bearing| < 20 deg, 8 deg of vertical margin) the curve is
**FLAT, 0.91-0.95 from 0 to 1.4 rad/s**. Range-matching removes most of it too. Confirmed by
looking: at 0.00 / 0.77 / 1.85 / **11.88 rad/s** the frames are equally sharp (Laplacian
variance 4518/3868/4445/3945, corr(rate, sharpness) −0.06 over 212 frames,
`vercheck/xfer_blur_look.png`). The real mechanism is the gate leaving the frustum, which
their geometry already models. Same measurement lesson as the three on 2026-08-01: the
setup decided the answer, and the control is what caught it.

**GATE 9 HAS NO IN-PLANE ROLL.** Normal-based measurement is blind to it, so a separate
channel: quad orientation as the 4x-angle circular mean of the four edge directions,
referenced to the image projection of gravity at the gate's own 3-D location, minus the same
quantity for a synthetic square at the resolved pose with zero roll. Quotients the 4-fold
symmetry exactly; never runs PnP on the observed quad; blind to lean by construction
(verified at 38 deg foreshortening); injection positive control recovers 5/10/20 deg 1:1.
Gate 9 reads **−0.95 deg [−1.32, −0.53]**, n=130 at size >= 60 px, flat across obliquity —
against controls gate 3 +0.38, gate 8 +0.28 and **gate 13 −3.77** (known vertical, four
times larger). All 16 gates span −3.8 to +2.3; gate 9 is unremarkable. Upper bound ~5 deg.
So the surrogate's radial-distance pass test stays adequate and no square-aperture collision
test is needed. `xfer_gate9roll*.py`, `vercheck/xfer_gate9roll_summary.png`.

**COLUMNS — a first-order obstacle model, honestly labelled.** Station numbering WRAPS
(facing pairs sum to 41: 16|25, 12|29, 18|23, 19|22, 15|26), so it runs 1-20 up one row and
21-40 back the other; row A (drawn 12-20) is −y, row B (21-29) is +y. Station pitch **25.4 m
+- 2.0** (affine fit n=17, R2 0.989, but 7.0 m rms residual — the sketch, not the map, is the
limit), row separation **18.6 +- 3.0 m**, 20 columns per row (MEASURED), radius **1.0 m
GUESSED** — no pixel-width-at-known-range measurement exists anywhere in the repo. This
supersedes `map_approx.json`'s 15.97 m/station, which used an across-per-station ratio of
1.13 where the metric fit says 0.73 (the sketch is drawn ~1.5x too wide). Claire's "at gate
5, gates 6 and 8 visible but 7 blocked by a pillar" is REPRODUCED: perpendicular LOS
distance 1.20 m for 5->7 (2.5 deg off-axis) vs 2.67 / 1.94 m at 35-53 deg off-axis for 5->6
and 5->8; MC P(column within 1 m of LOS) 0.45 vs 0.14 / 0.17. Only **6-7** is genuinely
threatened (clearance 1.2 m, P(<1.5 m) = 0.51); gates 8-16 have 5-10 m and can ignore
columns. `xfer_columns*.py`, `vercheck/xfer_columns.png`.

**NOISE-MODEL COMPARISON, the headline numbers** (their table / gentle / race / race):
current-gate valid 83.9-90.5 / 89.6 / 95.3 / 98.2%; `normal_valid` 48.4-49.7 / **66.6 / 75.0
/ 76.3%** (they are training the policy needlessly blind); `pose_valid` 59.7-61.9 / 61.0 /
59.4 / 65.5% (in family); staleness mean 0.090-0.118 / 0.145 / 0.201 / **0.117 s** (in
family); unchanged-since-last-step 62.7-66.2 / **22.0 / 28.5 / 27.6%** (their camera clock is
~2.4x stickier than replay); ribbon valid 53.9-60.3 / **95.8 / 100 / 100%**, fresh cyan
82-84% (their duty cycle is ~25 points pessimistic). Shape differences that matter more than
the means: dropout is **bimodal** (median 0.10-0.24 s, 41-54% of bursts over 0.2 s, tail to
2.6-4.8 s at race pace), validity autocorrelation is +0.83-0.90 at lag 1 and still +0.14-0.42
at lag 30 — so a 6-frame stack sees an essentially constant validity state, and near-i.i.d.
dropout in their `noise.py` trains on an easier interleaving than reality. Position error is
NOT measurable without truth; four proxies given, of which the bounded one is the
crossing-range residual (**+1.1 to +2.2 m long bias at contact**, where truth is <= 0.75 m).
`xfer_noise_compare.py`, `vercheck/xfer_noise_shape.png`, `xfer_crossrange.py`.

---

## 2026-08-02 — edge 10-11 was wrong; the station columns are an absolute ruler

**THE STATION COLUMNS ARE AN ABSOLUTE ALONG-COURSE RULER, AND CLAIRE'S SKETCH CARRIES
THEIR NUMBERS.** The hangar's numbered columns are evenly spaced physical structure along
the course, and `map_approx.json` records a station number per gate. Regressing the map's
along-course x on that number is therefore an EXTERNAL check: it costs one script, it uses
nothing the map was built from, and it is the cheapest independent test we have of any map
claim. **Run it routinely, not as a last resort.** Fitted on gates 0-10 it gives
22.59 m/station — independently consistent with the column pitch derived from frames,
25.4 +- 2 m — with residuals mean 0.00 m, sd 3.13 m. It is what localised this fault: not
scatter, but a *rigid step*, which says exactly one edge is wrong and the tail is
translated.

### What was wrong

`10-11 = 33.84 m`. Not noise, not a bad fit: a good measurement **of the wrong pair**. The
76 rows saw **gate 12** at the far end, so the number belonged to 10-12, and gate 11 plus
the tail 12-16 sat ~15.6 m too far along-course.

Three channels, none able to see the identity, all said it was impossible:

1. **Drag path.** Integrating the validated drag speed model between the gate-10 and
   gate-11 crossings gives path/chord **0.53** (fast lap `20260802-161838`) and **0.67**
   (pausing lap) against a 33.84 m chord, while every other leg of those laps sits at
   1.02-1.08. A flown path shorter than the straight-line chord is impossible, and
   dz = -1.7 m is far too small to rescue it (a climb pushes the ratio *up*).
2. **Station ruler.** Gates 0-10 residual mean 0.00 m (sd 3.13); gates 11-16 **all** at
   -17.8 to -25.5 m, mean **-19.79**. A clean rigid step at gate 11.
3. **Frame re-detection** (`xfer_leg1011.py`, `vercheck/xfer_leg1011.png`). The window
   shows gates strung at ~8 / ~22-25 / ~32 / ~41 m of range and the accepted rows pairing
   the near gate with one two along.

### Why the standing referee never fired — the generalisable part

**Single vantage.** All 76 rows come from ONE session, ONE contiguous 171-frame window
(`20260801-144858-vqual2-lap-pausing`, fid 32235-32406), bearing spread **1.5 deg**, far
gate 11-15 px against `MIN_SIZE_PX` 14. The rule that has caught every other
misidentification in this map — *two static gates cannot change separation with viewpoint*
— **needs two viewpoints to fire**, and there was one. Nothing in the rows looked wrong:
MAD 0.42 m, tight and repeatable across 76 frames, because a single vantage measuring the
wrong object is tight and repeatable. **Tightness is not evidence of identity.** The map
even recorded "10-11 = 34.0 repeats across two laps" — both "repeats" were the same
vantage; that claim is retracted in `mapvq2.STATUS`.

**Standing rule, added:** an edge measured from ONE vantage has had no identity check at
all, regardless of n or MAD. `n_vantages` now sits beside `n` in `measured_pairs`.

### How it was settled

Claire flew `20260802-180755-vm-strafe-10-11` for this edge alone: markers 1-2 / 3-4 / 5-6
/ 7-8 delimit **four staged vantages** on the 10/11 pair. Measured by
`mapedges_strafe1011.py` with that session's own skylight compass (97% confident frames):

| pair | vantage 1 | 2 | 3 | 4 | pooled horiz | MAD | n | dz_up |
|---|---|---|---|---|---|---|---|---|
| 10-11 | not co-visible | 16.24 | 16.66 | 16.65 | **16.49 m** | 0.28 | 96 | **-3.66 m** |
| 10-12 | 35.29 | 35.15 | 34.95 | 34.80 | **34.97 m** | 0.50 | 156 | -1.72 m |
| 11-12 | — | 21.58 (n=7) | 18.94 | 19.14 | 19.20 m | 0.64 | 52 | +2.06 m |

**Gate 11 is NAMED by an independent number**, not by a vote: the map measured
11-12 = 19.67 m / dz +2.32 m in a *different* session (n=28), and this session reproduces
it at 19.20 m / +2.06 m, and its bearing to **0.7 deg**. The rival candidate (the 34.9 m
object) gives 21.96 m to the third gate and fails. The cyan ribbon confirms it by eye:
`vercheck/strafe1011_ident.png`.

### A real detector gap, found on the way

**`detect.detections()` returns NOTHING for a gate at 4-7 m.** Its aperture stops forming
a hole contour (ribbon glow and bloom fill it) exactly when its checkerboard squares start
to, so the frame yields half a dozen *decoration* squares carrying fictitious 20-47 m
ranges — and the outer-boundary fallback then refuses the gate body itself, because that
body is now a "used parent" of those decoration holes. A first pass here took the nearest
detection as gate 10 and measured decoration: separations scattered 4-54 m with implied
height differences of +30 m. `mapedges_strafe1011.big_outer()` recovers the gate by fitting
that outer boundary anyway, with `detect.py`'s own 2700 mm model. Same family as the four
recorded failure modes, and worth fixing in `detect.py` proper.

Second trap in the same session: the near gate ALSO appears via the outer fallback at
13.8 m (85% `outer` source) alongside its inner-aperture reading at 16.4 m (92% `inner`),
with equal apparent size — **one gate, two channels, 2.7 m apart**. The outer fit is biased
outward by bloom and signage, which shortens the range. The 19.67 m referee picks the inner
reading; a naive cluster count would have picked the wrong one.

### Result and acceptance test

* **10-11 = 16.49 m horizontal (+-0.5 m), bearing 191.6 deg, dz_up -3.66 m.** The
  uncertainty is the cross-vantage spread (16.24 / 16.66 / 16.65), not the MAD.
* **10-12 = 34.97 m** entered as its own edge. The old single-vantage rows read this pair
  **1.1 m short** — the same direction the outer-fit bias produces at range.
* **10-11-12 is now a closed triangle**: 0.46 m closure on a 71 m perimeter (0.6%). 10-11
  and 11-12 stopped being bridges.
* Vector-solve residuals: median **0.03 m**, p90 0.39 m.
* Direction-LOO on non-bridge edges: median 0.59 m, p90 0.68 m. Held-out 10-11 is predicted
  at 16.33 m against 16.49 measured, bearing error +1.5 deg.
* Height-solve residual median 0.02 m over 20 pairs.
* Every other measured pair still reproduces: 12-13-14 closes 1.4%, 13-14-15 1.2%, 7-8-9
  1.6%; 15-16 (n=446), 13-15 and 12-14 untouched.
* **Station ruler, the acceptance test: gates 11-16 residual mean -19.79 m -> -5.25 m**
  (sd 2.90, range -10.9 to -3.3), against gates 0-10 at mean 0.00, sd 3.13, range -4.3 to
  +6.8. The band now overlaps the fitted band and the rigid step is gone. Excluding gate 16
  (whose 15-16 edge takes its quadrant from the sketch, at the end of the sketch where the
  hand drawing is weakest) the tail sits at -4.1 m, inside the 0-10 spread. **A ~4-5 m
  offset remains and is not claimed to be zero** — about 0.2 station, comparable to the
  sketch's own precision plus chain error over six gates, and no longer diagnostic of a
  single bad edge.
* `metres_per_station`: the **57.57 outlier is gone** (it *was* this edge). 10-11 now reads
  28.06 and 10-12 23.66; median 19.57, p10 15.33, p90 25.61. Largest is now 15-16 at 36.29.

### Published

`map_vq2.json` (`measured_pairs` 10-11 / 10-12, `layout_directions_2026_08_02` re-solved,
`status.CORRECTION_2026_08_02_edge_10_11`), `map_current_viz.png`, and the control-side
package `pilot/course/course_vq2.json` + `course_vq2.py` + `README.md` — regenerated via
`build_course.py`, re-verified with `verify_course.py`. **Course length 250.93 m, was
268.22 m: 17.29 m shorter.** Per-gate sigma for gates 11-16 *fell* (2.11-2.38 -> 1.98-2.18
m) because that corner is braced now. Anyone training against a pre-correction copy learned
a course 17 m too long with gates 11-16 about 15 m out of place.

Reproducibility is mechanised, not documented: `mapvq2.RELABEL_ROWS` (window-scoped, so it
cannot eat a future genuine 10-11 row) renames the old rows, and `mapdir.solve()` applies
the same relabel plus the staged-vantage edge, so `mapdir.py --solve --write` reproduces
the corrected map instead of silently reverting it. The pre-correction file is kept at
`map_vq2_pre1011fix.json.bak`.

## 2026-08-02 addendum (3) — speed_est had no zero, and the transient story was wrong

Two claims about `speed_est_mps` were on the table. One was a measured bug and is fixed. The
other was a plausible physical argument that turns out to be false, and the falsification is
the more useful half of this entry.

### The bug: `sqrt(|a_xy|/k)` has no zero, so a parked aircraft flies

Reproduced with the veto disabled (`producer.py --replay 20260802-163548 --no-speed-veto`),
over the 42 s in which the aircraft was jammed against an obstacle with the lens blocked:

| t 4-46 s, truth STATIC | speed_est p10/p50/p90 | speed_conf p10/p50/p90 |
|---|---|---|
| before | **8.06 / 8.08 / 8.10 m/s** | **1.00 / 1.00 / 1.00** |
| after | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

The cause is arithmetic, not a coding slip: `|a_xy| = 3.0 m/s^2` is what the VQ2 pad's
17.8 deg tilt pushes through the airframe at rest, and the inversion maps it to 8.4 m/s at
full confidence. `COAST_TRANSLATE` then flew every coasted track at that speed —
BEFORE the fix the session re-acquired only 3 coasted tracks with a median miss of 13.5 m
and 439 px; AFTER, 12 re-acquisitions at 6.1 m and 11 px, because the tracks stayed where
the aircraft actually was.

`vel_bearing_rad` had the identical hole and is fixed with it. `HOVER_ACC_MIN` is 0.35 m/s^2
and the parked pad reads 3.0, eight times over, so the producer was also publishing a
confident heading of travel for an aircraft that was not travelling.

### The fix: the camera, online, at 0.5 ms a frame

`ImageMotion` in `producer.py` is a live port of `staticdet.py` — mean |dI| between
consecutive 160x120 greyscale frames, trailing 0.5 s median, static declared after
`IMG_STATIC_MIN_S = 2.0 s` continuously below `IMG_M_STATIC = 0.25`. **The thresholds are
staticdet's, unchanged and not re-derived**; they were calibrated against VQ1 truth velocity
(1.16% of truth-moving frames called parked) and against the two known windows of `163548`
(100% of the static caught, 0% of the moving lost), and any value in 0.15-0.30 gives the
same verdicts. Two differences forced by causality, both stated in the code: the median is
trailing rather than centred, and a park is declared 2 s after it begins rather than from
its first frame. On `163548` the declared park is a single segment, **5.67-47.47 s**, against
a truth window of 4-47 s — the 1.67 s of lead-in is exactly that price.

Cost: **0.5 ms median per frame** (p90 0.6-0.7) against a 17.8-29.5 ms total. It needs no
model, no weights and no state beyond one 160x120 array.

**Margin on the sessions that must not trip it.** The smoothed value never comes near the
threshold on real flight: minimum over the whole session is **3.17** on `153626`, **0.284**
on `161838`, **0.382** on the strafe canary, against 0.051-0.057 while jammed. No exactly
duplicated frame pairs anywhere (a stalled video stream is the one plausible false-park
mechanism; the 2 s run rule and this measurement both say it does not occur here). Zero
frames called parked in all three.

### The transient claim, measured and FALSIFIED

The claim: the inversion assumes steady state, so while accelerating the horizontal specific
force is drag PLUS the linear acceleration and `speed_est` is biased during exactly the
transients a race consists of.

**It is false, and the reason is worth keeping.** An accelerometer measures SPECIFIC FORCE —
non-gravitational forces over mass — not coordinate acceleration. Thrust is body -z by
definition, so the horizontal body pair is drag ALONE whether or not the aircraft is
accelerating; the acceleration is the CONSEQUENCE of thrust + drag + gravity, not a further
additive term in what the sensor reads. Measured rather than argued (`dragtransient.py`
section 2, VQ1 truth, n=17463): regress the drag residual `r = a_meas_xy + k|v_xy|v_xy` on
the body-horizontal coordinate acceleration, where a full leak would give slope 1.000 —

    body-x  slope -0.0073   body-y  slope -0.0013   (all frames)
    body-x  slope +0.0414   body-y  slope +0.0171   (|gyro| < 0.2 rad/s)

**What is real is smaller and differently caused: noise amplification.** The residual does
grow with the acceleration, 0.38 -> 1.19 m/s^2 median from the 0-2 to the 20-40 m/s^2 band,
but it is only 4-6% of the acceleration and is not aligned with it (corr +0.05 to +0.18). In
speed terms:

| regime | n | bias | med abs err | p90 abs err | relative |
|---|---|---|---|---|---|
| steady, \|dv/dt\| < 10 m/s2 | 16457 | +0.15 | 0.51 | 0.82 | 7.2% |
| accel, >= 10 | 1006 | +0.30 | 0.54 | 1.30 | 6.0% |
| hard, >= 25 | 114 | +1.01 | 1.00 | 2.12 | 5.2% |

Note the last column: the RELATIVE error is flat or better under transients, because the
aircraft is faster there. The tail roughly doubles; the typical error does not move.

It is also not an artefact of the truth stream. A timing-offset sweep between the
interpolated position stream and the 61 Hz IMU has a flat minimum within +-16 ms and does
not remove the high-`|dv/dt|` residual (`dragtransient.py` section 3).

### What was built for it, deliberately bounded

`transient_factor(jerk, |gyro|)` multiplies `speed_conf` by a ramp in
`[TRANSIENT_FLOOR = 0.6, 1]`, using `|d|a_xy|/dt|` from 10 to 50 m/s^3 and `|gyro|` from 1.0
to 3.0 rad/s — both read off the measured bands, neither fitted. It fires on 11.6% of VQ1
frames and sorts the tail weakly but genuinely (corr(1 - factor, |e|) = +0.21; p90 error
1.33 in the 0.55-0.70 confidence bucket against 0.81 in the 0.85-1.00 bucket).

**It cannot switch coast translation off, and that is the design decision.** The published
`speed_conf` carries the transient factor; the `COAST_TRANSLATE` gate reads a separate
`last_speed_conf_noise`, the `|a_xy|` noise-floor term alone. Killing translation during
aggressive rotation would lose the coast exactly when detections are dropping and coasting
is doing the work, in exchange for a 2x change in a per-frame displacement of a few
centimetres. The STATIC veto does stop translation, because that is a validity claim rather
than a precision one, and it stops it three ways over: `speed_conf = 0` is below
`COAST_SPEED_CONF_MIN`, `v_body` is returned as `None`, and the branch tests
`imgmot.static` explicitly.

### Regression: bit-identical except the number that was supposed to change

`--no-speed-veto` restores the old estimator exactly, so the A/B is one flag on one binary
rather than a diff against an older run whose other inputs may have moved. Comparing the
full `diag.csv` column by column, BEFORE vs AFTER:

| session | columns that differ |
|---|---|
| `153626` (race pace, collisions) | **only `speed_conf`**, 391 of 1592 frames |
| `161838` (race pace, post-last-reset) | **only `speed_conf`**, 697 of 1900 frames |
| `005431` strafe 6-7 (canary) | **only `speed_conf`**, 77 of 2426 frames |

So current-gate valid (95.3 / 98.2 / 72.9%), pose_valid (59.3 / 65.5 / 51.4%), staleness
(0.203 / 0.117 / 0.248 s), out-of-bound (1.32 / 3.70 / 0.00%), reacquire miss
(3.37 m 9 px / 5.80 m 7 px / 4.68 m 10 px) and ordering inversions (7.43 / 10.89 / 2.64%)
are unchanged to the last digit, by construction rather than by luck. **The standing strafe
6-7 canary is unmoved: separation median 16.51 m, MAD 0.18, n=115.** On `163548` the
tracking columns DO move, which is the point — the phantom translation is gone.

### Declined, with reasons

* **Gate-PnP range closure as a speed cross-check.** Already measured on 08-02 and recorded
  above: PnP range noise is ~3% of range, four times the per-frame closure at 5 m/s, and
  windowing it leaves 21-31 independent windows over a 2-5 m/s span — no dynamic range, so
  its agreement or disagreement means nothing. It would also put a per-track range history
  into the per-frame path. Leg odometry already arbitrates the coefficient at 24 legs and
  8-25 m/s and is not on the hot path at all.
* **Refusing speed during transients.** The measurement does not support it: the typical
  error is unchanged and the relative error improves. A refusal would cost the coast in the
  regime it was built for.
* **Re-deriving the static thresholds.** They are calibrated, the accept band is 2x wide,
  and re-fitting them on the same sessions would have manufactured agreement.

`dragtransient.py` prints every table above (`python3 dragtransient.py`, ~40 s).
Runs: `perception/producer_runs/SPD-{BEFORE,AFTER}-*`.

### The gentle laps had the same phantom, and removing it helps slightly

Not on the required list, checked anyway because they are the sessions with real pauses in
them — the one place a static detector could plausibly misfire on purpose-flown data.

`20260801-121520` (gentle lap): the camera calls parked on **0.8%** of frames (52), all of
them genuine pauses, and BEFORE the fix those frames published **up to 8.92 m/s at
speed_conf 1.00**. AFTER: 0.00 / 0.00. The tracking columns move a little and they move the
right way — current-gate valid **87.1 -> 87.5%**, pose_valid 58.4 -> 58.6%, staleness 0.162
-> 0.161 s, reacquire miss unchanged at 3.94 m / 5 px, ordering inversions unchanged at
15.02%, out-of-bound 2.26 -> 2.27%. `20260801-144858` (lap with pausing): 0.5% parked (50
frames), speed 0.00 at conf 0.00 there, current-gate valid 97.0%, pose_valid 77.7%.

## 2026-08-02 (late) -- the slot INVARIANT: two-sided map bands, in-order filling, exclusivity

Claire found two more association bugs on a producer-path HUD replay of
`20260801-121520-vq2-lap-0-15`, and they are the same bug wearing two hats. Every previous
fix in this area constrained the case that had just been seen; this pass states the
invariant instead:

> **The three slots are a MONOTONE, INJECTIVE assignment onto gates k, k+1, k+2 at
> map-consistent ranges.** In order, no gaps skipped, no object in two slots.

### What was actually wrong

**Bug 1 -- a nearer gate promoted to an outer slot.** Claire's frame at `active=5`: current
gate at 7.5 m, the `+1` slot showing nothing, the `+2` slot labelled `g7 23.0 m`. From the
corrected map, gate 7 cannot be nearer than ~30 m from there, and gate 6 is at ~23 m. The
object in `+2` was gate 6.

Why every existing guard missed it, in order:

* the per-slot range prior (`cur_rmax`) has **only an UPPER bound**, so a candidate far too
  NEAR for its slot is admitted without complaint;
* `_slot_order_ok` needs all three slots populated -- with `+1` empty,
  `7.5 <= (nothing) <= 23.0` is trivially satisfied;
* `_slot_pair_ok`, the triangle inequality, is **exact but vacuous on a near-collinear
  course**. d(5,7) = 32.31 m and r0 = 7.5 m admit gate 7 anywhere in [24.8, 39.8] m before
  tolerance -- which is precisely where gate 6 is. Distances alone cannot separate gates on
  a straight leg; the TURN ANGLE has to enter.

**Bug 2 -- one detection claimed by two slots.** Confirmed **producer-side**, not a HUD
artefact. At `active=6`, frames `00104161`-`00104175`: CUR = g6 at 14.7-15.7 m and `+1` =
g7 frozen at 15.45 m, positions coincident, while the map puts gates 6 and 7 **16.42 m
apart**. Detection-level exclusivity (`used`) was already enforced and never saw it,
because the two slots never shared a detection: they converge through the **net seeds** (a
coasted track's seed box and a detection's seed box crop the SAME gate, and gatenet
obligingly regresses it out of both) and through plain coasting. Rate before the fix: **380
of 1412 frames (26.9%)** in the replayed window, 164 of 293 frames at `active=6` alone.

### The fix

`SLOT_BANDS` (flag `--no-slot-bands` restores the old behaviour in one binary):

1. **Two-sided map bands per lookahead slot** (`Producer._slot_band`). The current gate's
   MEASURED range r0 places the aircraft r0 short of gate k along the k-1 -> k map edge --
   exact at both ends of a leg, interpolating in between -- and the map then gives an
   EXPECTED range for every later gate, not merely a bound. Tolerance
   `5.0 m + 0.55 * r0`, because the residual error is lateral deviation from that edge and
   scales with r0. At r0 = 7.5 m on the 5/6/7 leg: gate 6 -> [13.6, 31.8] m, gate 7 ->
   [29.9, 48.1] m. The 23.0 m object falls in gate 6 and outside gate 7 with 7 m of margin,
   where the triangle inequality had no opinion at all. Compass-free and anchor-free: map
   distances plus one map edge direction used only relative to itself, so neither
   `MapModel.s` nor `psi_int` enters.
2. **Bands bind in three places**: candidate ADMISSION in the association loop, the
   map-distance lookahead BOOTSTRAP, and `_assign_ok` -- so the joint pass now scores
   band-consistency as part of feasibility.
3. **In-order filling** (`Producer._reslot`). A lookahead occupant outside its own band is
   moved to the nearest EMPTY lookahead slot whose band does contain it, and dropped if
   there is none. An empty intermediate slot must never let a nearer gate skip outward, and
   a range that matches the empty slot's band is positive evidence it IS that slot. A moved
   DETECTION is re-measured from scratch under its new index and the misplaced track is
   deleted rather than re-keyed, so no accumulated per-gate normal evidence follows a label
   across gates.
4. **Exclusivity** (`Producer._dedupe_slots` + `_seed_clash`). Net seed boxes that would
   crop the same object are no longer both issued; and after the update, two slots holding
   coincident positions are resolved by the map -- the object cannot be a LATER gate than
   the nearest slot claiming it, so the farther slot loses its track.

### Measured, on the `active=4..7` window of the gentle lap (1412 frames, net on)

| | BEFORE (`--no-slot-bands`) | AFTER |
|---|---|---|
| duplicate-claim frames | 380 (26.9%) | **0 (0.0%)** |
| lookahead-slot band violations | 937/1699 (**55.2%**) | 26/1312 (**2.0%**) |
| ... at `active=6` | 200/200 (100%) | 0/30 (0%) |
| CUR slot valid | 98.2% | 96.8% |
| CUR staleness (valid frames) | 0.104 s | 0.090 s |
| `+1` slot valid | 78.8% | 58.0% |
| `+2` slot valid | 42.9% | 36.3% |

Claire's two frames, before -> after:

* `active=5`, `00104028/33/36`: `+2` slot g7 read 23.7-26.7 m (gate 6's range) -> **31.4 m**,
  with gate 6 holding `+1` at 23.1-24.0 m. Monotone and map-consistent.
* `active=6`, `00104161/65/70/75`: `+1` slot g7 read 14.9-15.5 m **on the current gate's own
  detection** -> **31.55 m**, i.e. the real gate 7 (d(6,7) = 16.42 m at r0 ~ 15 m predicts
  31.4 m). No coincident slots anywhere in the window.

### Read this before trusting the numbers above

The lookahead slots lose 21 and 7 points of "valid". Most of that was **false
availability** -- 55% of pre-fix lookahead slots violated their own map band, i.e. carried
the wrong gate's label -- but not all of it, and a refused slot is still a refused slot.
CUR valid also drops 1.4 points on this window (staleness improves), most plausibly because
band-infeasible greedy assignments now route through the joint pass more often.

**NOT VERIFIED** (deliberate, for turnaround): the standing three-lap metric sweep
(`121520` gentle, `153626` and `161838` race pace), the strafe 6-7 canary (16.51 m,
MAD 0.18), reacquire miss, out-of-bound %, and any regime outside `active=4..7` of one lap.
`--selftest` passes. The tolerances `SLOT_BAND_TOL_M/K` are reasoned, not fitted.
