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
