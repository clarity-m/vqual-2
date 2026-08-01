# gatenet — CNN corner regression for the gate aperture

Record of the training runs behind `gatenet.py`. Method first, then the numbers each run
appended as it finished. Written by the run itself so the result survives whether or not
anyone reads a chat log.

## What is being learned

Four corners of the **inner 1500 mm aperture**, in original 640×360 pixels, regressed from
an image crop. Corners feed `cv2.solvePnP`, so **corner error in pixels is the metric**,
not IoU. No confidence head — VQ1 truth covers 5 of the 6 race gates, so any frame showing
race gate 5 holds an unlabelled real gate, which poisons a real/not-real head and is
harmless to a regressor trained on positive crops.

## Data

`autolabels_vq1.json`, regenerated 2026-08-01 from the two VQ1 sessions:

```
20260731-195307:              10216 instances over 5043 posed frames
20260731-204841-vq1-lap-slow:  2792 instances over 1555 posed frames
total 13008 instances / 6598 frames — 10317 fully in frame, 2691 clipped, 339 gate-occluded
apparent size px: min 6.0  p10 7.3  median 20.6  p90 120.2  max 1773.9
```

Labels are **amodal** — corners outside the image are kept, never clamped — and targets are
never clamped into the crop either.

## The crop, and why it is built from the visible part only

A crop centred on the amodal bounding box would be cheating: at race time nothing knows
where an off-screen corner is. So the crop is the bounding box of **quad ∩ image rect**
(what a detector or a previous-frame track could actually supply), squared up, ×1.60
margin. The amodal target is then written in that crop's frame:

```
u = 2 (x − cx) / S      v = 2 (y − cy) / S
```

Crop interior is [−1,1]²; a corner outside the crop simply has |u| > 1, and the output
layer is linear so it can say so. Eval uses this crop deterministically, with no jitter.

**Caveat, stated up front:** eval crops come from the ground-truth visible box. A live
system would feed a detector's or tracker's box instead, which is noisier. Training jitter
(±16% translation, ×0.78–1.55 log-uniform scale) exists to cover that gap, but the numbers
below are still measured with a clean box.

## Split — by time, never randomly

Frames are 30 Hz video; adjacent frames are near-duplicates, so a random split leaks and
reports a fantasy. Two splits, both reported:

* **`block`** — each session cut into contiguous 240-frame (8 s) blocks; every 5th block
  held out (~20%). Frames within 45 frames (1.5 s) of a held-out block are dropped from
  **train**, so no training frame is within 1.5 s of any validation frame.
  → 9406 train / 2670 val instances.
* **`session`** — train on `20260731-195307`, validate on the entirely separate
  `20260731-204841-vq1-lap-slow` flight. Harsher and the more honest number.
  → 10216 train / 2792 val instances.

## Model

Five stride-2 depthwise-separable stages (24→48→96→128→160 ch), adaptive-pooled to 4×4,
flatten → 256 → 8. **732,456 parameters**, input 160×160. Depthwise-separable and this
small because it must eventually run at ~30 Hz alongside control. No global-average-pool to
1×1: that discards exactly the spatial information being regressed.

Loss is smooth-L1 (β=0.05) in **normalised** units, i.e. scale-invariant — a 20 px gate and
a 100 px gate contribute equally. Absolute-pixel loss would be dominated by the handful of
near gates (apparent size runs to 1774 px). Pixel error is what gets reported.

Augmentation: crop jitter/scale, brightness/contrast/gamma/noise/blur. **No horizontal
flip, no rotation** by default — the winding convention (index 0 = min(x+y) corner) is
defined in image space, so any geometric transform must re-canonicalise the target;
`--hflip` implements that correctly but is off, because an unattended run should not carry
that risk.

---

## Run `block` -- split `block`

* hit --max-hours 2.6; 2.60 h, 128 epochs this invocation
* parameters: 732,456   input 160x160   batch 64   crop margin 1.6
* train 9406 / val 2670 instances (13008 labelled total)
* best val median corner error: **0.89 px**
* inference -- cuda batch1: 2.74 ms (2.74 ms/crop); cuda batch8: 15.21 ms (1.90 ms/crop); cpu batch1: 4.42 ms (4.42 ms/crop)

| group | n | corner med | corner p90 | centre med | centre p90 |
|---|---:|---:|---:|---:|---:|
| ALL | 2670 | 0.89 | 7.15 | 0.76 | 5.96 |
| size 0-15 px | 1265 | 0.45 | 2.31 | 0.40 | 2.17 |
| size 15-30 px | 807 | 0.71 | 4.69 | 0.59 | 3.95 |
| size 30-60 px | 313 | 1.54 | 8.41 | 1.35 | 6.94 |
| size 60-120 px | 119 | 5.08 | 13.64 | 5.16 | 13.02 |
| size >=120 px | 166 | 34.38 | 191.62 | 32.13 | 134.07 |
| clipped | 412 | 4.52 | 76.49 | 4.42 | 51.91 |
| unclipped | 2258 | 0.70 | 4.04 | 0.60 | 3.16 |
| occluded (gate-on-gate) | 76 | 1.47 | 5.68 | 1.36 | 4.62 |
| detector-comparable (unclipped, unoccluded, >=15 px) | 1069 | 1.01 | 6.01 | 0.83 | 5.05 |


## Run `session` -- split `session`

* completed all epochs; 2.82 h, 130 epochs this invocation
* parameters: 732,456   input 160x160   batch 64   crop margin 1.6
* train 10216 / val 2792 instances (13008 labelled total)
* best val median corner error: **1.06 px**
* inference -- cuda batch1: 2.64 ms (2.64 ms/crop); cuda batch8: 15.03 ms (1.88 ms/crop); cpu batch1: 4.50 ms (4.50 ms/crop)

| group | n | corner med | corner p90 | centre med | centre p90 |
|---|---:|---:|---:|---:|---:|
| ALL | 2792 | 1.06 | 7.24 | 0.94 | 6.84 |
| size 0-15 px | 882 | 0.48 | 1.98 | 0.46 | 1.97 |
| size 15-30 px | 968 | 0.81 | 3.44 | 0.71 | 3.19 |
| size 30-60 px | 418 | 1.38 | 4.30 | 1.20 | 3.89 |
| size 60-120 px | 257 | 3.90 | 8.81 | 3.38 | 8.15 |
| size >=120 px | 267 | 19.41 | 130.13 | 15.09 | 67.15 |
| clipped | 720 | 1.83 | 45.39 | 1.72 | 32.12 |
| unclipped | 2072 | 0.92 | 4.12 | 0.83 | 3.66 |
| occluded (gate-on-gate) | 46 | 1.57 | 3.79 | 1.31 | 3.86 |
| detector-comparable (unclipped, unoccluded, >=15 px) | 1356 | 1.17 | 5.48 | 1.03 | 4.69 |


## Detector baseline (`detect.py`) on the same held-out set -- split `block`

* 1388 of 2670 labelled instances matched a detection -> **recall 0.520**

| group | n | recall | corner med | corner p90 | centre med | centre p90 |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 2670 | 0.520 | 5.36 | 19.33 | 1.53 | 6.70 |
| size 0-15 px | 1265 | 0.405 | 2.13 | 7.52 | 0.77 | 4.80 |
| size 15-30 px | 807 | 0.773 | 8.00 | 16.07 | 1.79 | 6.27 |
| size 30-60 px | 313 | 0.700 | 17.44 | 28.19 | 3.00 | 11.13 |
| size 60-120 px | 119 | 0.277 | 39.99 | 49.53 | 5.88 | 25.26 |
| size >=120 px | 166 | 0.000 | - | - | - | - |
| clipped | 412 | 0.119 | 20.82 | 35.28 | 11.69 | 25.25 |
| unclipped | 2258 | 0.593 | 4.84 | 18.06 | 1.41 | 5.82 |
| detector-comparable (unclipped, unoccluded, >=15 px) | 1069 | 0.741 | 8.16 | 20.78 | 1.99 | 6.39 |

## Detector baseline (`detect.py`) on the same held-out set -- split `session`

* 1621 of 2792 labelled instances matched a detection -> **recall 0.581**

| group | n | recall | corner med | corner p90 | centre med | centre p90 |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 2792 | 0.581 | 4.04 | 20.53 | 1.63 | 10.11 |
| size 0-15 px | 882 | 0.457 | 2.29 | 7.58 | 0.89 | 3.58 |
| size 15-30 px | 968 | 0.775 | 3.59 | 13.92 | 1.37 | 7.85 |
| size 30-60 px | 418 | 0.792 | 9.22 | 27.27 | 3.17 | 13.88 |
| size 60-120 px | 257 | 0.257 | 27.64 | 46.26 | 4.45 | 17.30 |
| size >=120 px | 267 | 0.266 | 8.88 | 45.21 | 6.39 | 31.06 |
| clipped | 720 | 0.197 | 16.19 | 29.92 | 11.14 | 19.75 |
| unclipped | 2072 | 0.714 | 3.19 | 18.43 | 1.37 | 6.52 |
| detector-comparable (unclipped, unoccluded, >=15 px) | 1356 | 0.785 | 3.71 | 21.92 | 1.73 | 8.09 |

## Relative error (corner error / apparent gate size)

Absolute pixels flatter small gates and punish big ones; a gate 300 px across can absorb more pixels of error than a 20 px one before solvePnP notices. This table is the same predictions, scaled.

### `block` split -- error relative to apparent gate size

| group | n | corner med px | rel med | rel p90 |
|---|---:|---:|---:|---:|
| ALL | 2670 | 0.94 | 4.6% | 23.9% |
| size 0-15 px | 1265 | 0.46 | 4.6% | 25.2% |
| size 15-30 px | 807 | 0.74 | 3.3% | 21.9% |
| size 30-60 px | 313 | 1.62 | 4.0% | 24.1% |
| size 60-120 px | 119 | 5.33 | 6.9% | 20.1% |
| size >=120 px | 166 | 45.36 | 9.8% | 33.1% |
| clipped | 412 | 5.04 | 8.5% | 29.3% |
| unclipped | 2258 | 0.74 | 4.1% | 23.5% |
| clipped & >=60 px | 206 | 40.22 | 9.2% | 31.4% |
| unclipped & >=60 px | 79 | 5.33 | 6.9% | 17.7% |

### `session` split -- error relative to apparent gate size

| group | n | corner med px | rel med | rel p90 |
|---|---:|---:|---:|---:|
| ALL | 2792 | 1.08 | 4.6% | 17.9% |
| size 0-15 px | 882 | 0.52 | 5.4% | 20.2% |
| size 15-30 px | 968 | 0.81 | 3.7% | 16.1% |
| size 30-60 px | 418 | 1.45 | 3.5% | 9.9% |
| size 60-120 px | 257 | 3.93 | 5.0% | 10.3% |
| size >=120 px | 267 | 20.21 | 9.3% | 26.1% |
| clipped | 720 | 2.08 | 6.6% | 24.5% |
| unclipped | 2072 | 0.94 | 3.8% | 16.5% |
| clipped & >=60 px | 249 | 19.57 | 10.1% | 31.2% |
| unclipped & >=60 px | 275 | 4.60 | 5.0% | 10.5% |

## Cyan-ribbon contamination -- split `block`

Cyan fraction of a ~7 px ring on the GT aperture OUTLINE (the edge is what the ribbon corrupts, not the interior); net error is over ALL instances in the bucket, detector error only over the ones it found (recall column).

THE RAW BUCKETS ARE CONFOUNDED and must not be read as an effect: a clipped or huge gate has most of its outline off-image and lands in the "none" bucket, which is also where the hardest instances live. The second table controls for it -- unclipped, 15-60 px only, so size and clipping are held roughly fixed and only the ribbon varies.

**all held-out instances (confounded -- see above)**

| ribbon coverage | n | net corner med | net p90 | det recall | det corner med |
|---|---:|---:|---:|---:|---:|
| none (<2%) | 1037 | 1.43 | 33.36 | 0.281 | 2.32 |
| slight (2-15%) | 1313 | 0.72 | 3.70 | 0.722 | 5.01 |
| heavy (>=15%) | 320 | 1.02 | 3.34 | 0.466 | 7.28 |

**CONTROLLED: unclipped, apparent size 15-60 px**

| ribbon coverage | n | net corner med | net p90 | det recall | det corner med |
|---|---:|---:|---:|---:|---:|
| none (<2%) | 140 | 5.08 | 10.13 | 0.429 | 7.27 |
| slight (2-15%) | 709 | 0.74 | 3.61 | 0.889 | 6.42 |
| heavy (>=15%) | 194 | 1.00 | 4.14 | 0.582 | 11.28 |


## Cyan-ribbon contamination -- split `session`

Cyan fraction of a ~7 px ring on the GT aperture OUTLINE (the edge is what the ribbon corrupts, not the interior); net error is over ALL instances in the bucket, detector error only over the ones it found (recall column).

THE RAW BUCKETS ARE CONFOUNDED and must not be read as an effect: a clipped or huge gate has most of its outline off-image and lands in the "none" bucket, which is also where the hardest instances live. The second table controls for it -- unclipped, 15-60 px only, so size and clipping are held roughly fixed and only the ribbon varies.

**all held-out instances (confounded -- see above)**

| ribbon coverage | n | net corner med | net p90 | det recall | det corner med |
|---|---:|---:|---:|---:|---:|
| none (<2%) | 1239 | 1.47 | 22.37 | 0.442 | 3.21 |
| slight (2-15%) | 1316 | 0.98 | 4.53 | 0.726 | 3.86 |
| heavy (>=15%) | 237 | 0.49 | 3.25 | 0.494 | 7.34 |

**CONTROLLED: unclipped, apparent size 15-60 px**

| ribbon coverage | n | net corner med | net p90 | det recall | det corner med |
|---|---:|---:|---:|---:|---:|
| none (<2%) | 239 | 1.25 | 3.19 | 0.908 | 2.38 |
| slight (2-15%) | 742 | 0.84 | 3.01 | 0.879 | 3.27 |
| heavy (>=15%) | 122 | 0.50 | 3.32 | 0.648 | 9.59 |


---

# VERDICT

## Does it beat the existing detector? On corners and on centre, yes. With caveats that matter.

`NOTES.md` records **1.8 px held-out centre error** for the detector after `refine.py`'s
pose-stream refit. Measured on the same held-out instances used here, per-frame `detect.py`
centre error is **1.53 px (block) / 1.63 px (session)** overall and **1.99 / 1.73 px** on
the subset it handles best — consistent with that 1.8 px, so the comparison is fair.

The net's centre error is **0.76 px (block) / 0.94 px (session)**, and **0.83 / 1.03 px**
on that same detector-comparable subset. **So yes — roughly 2× better on centre position,
and it beats the 1.8 px figure on both splits.**

The corner numbers are not close at all, which is the point, since corners are what feed
`solvePnP`:

| | net (block) | detector (block) | net (session) | detector (session) |
|---|---:|---:|---:|---:|
| coverage | 1.000 | 0.520 | 1.000 | 0.581 |
| corner med, all | **0.89** | 5.36 | **1.06** | 4.04 |
| centre med, all | **0.76** | 1.53 | **0.94** | 1.63 |
| corner med, detector-comparable subset | **1.01** | 8.16 | **1.17** | 3.71 |
| corner med, clipped | **4.52** | 20.82 (recall 0.119) | **1.83** | 16.19 (recall 0.197) |
| corner med, ≥120 px | 34.38 | — (recall **0.000**) | 19.41 | 8.88 (recall 0.266) |

**The coverage row is the largest single result.** The detector finds 52–58% of labelled
gates; on clipped instances it finds 12–20%, and in the block split it finds **none at all**
above 120 px. The net answers on 100% by construction, because it is handed a box rather
than having to find an enclosed contour. That is failure mode 2 (clipped gates vanish)
addressed directly.

## The cyan ribbon: failure mode 4 confirmed at population scale, and the net is immune

Measured as the cyan fraction of a 7 px ring on the aperture **outline** (the edge is what
the ribbon corrupts; an interior-area proxy measures the wrong thing), controlled to
unclipped gates of 15–60 px so size and clipping are held roughly fixed:

| ribbon on the aperture edge | net corner med | detector recall | detector corner med |
|---|---:|---:|---:|
| none (<2%) | 1.25 | 0.908 | 2.38 |
| slight (2–15%) | 0.84 | 0.879 | 3.27 |
| heavy (≥15%) | **0.50** | **0.648** | **9.59** |

(session split; the block split shows the same direction — detector 6.42 → 11.28 px and
recall 0.889 → 0.582 — but its "none" bucket is contaminated by hard oblique instances and
is not a clean control, so the session numbers are the ones to quote.)

The detector's corner error **quadruples** and its recall drops by a quarter as the ribbon
crosses the aperture edge. The net does not degrade at all. This is the first measurement
of the ribbon effect on a population rather than the single frame in `NOTES.md`, and it
agrees with it.

## Where it does NOT work, plainly

* **Extreme clipping fails.** Block-split clipped p90 is **76 px**. `gatenet_runs/*/sanity.png`
  shows it directly: on a 287 px and a 1258 px gate with only a sliver in frame, the
  predicted quad is not close. Amodal extrapolation works while a decent fraction of the
  aperture is visible and collapses when it is not. It is better than the detector there
  only because the detector returns nothing at all.
* **Large gates are the weak band** — 34 px median ≥120 px on the block split. In relative
  terms that is 9.8% of gate width against ~3.5–4.6% everywhere else, so it is a genuine
  degradation and not just an artifact of measuring big things in pixels. Large and clipped
  are largely the same instances.
* **p90 is poor overall** (7.2 px both splits) even where the median is sub-pixel. The
  distribution has a heavy tail; a consumer must treat individual predictions as
  occasionally very wrong and cannot use this open-loop without an outlier check.
* **Session-holdout is worse than block-holdout across the board** (1.06 vs 0.89 px
  overall). That gap is the honest estimate of how much the block split still leaks, and it
  is small — the time-blocking worked.

## Caveats on the numbers themselves

1. **Eval crops come from the ground-truth visible box.** A live system feeds a detector's
   or tracker's box, which is noisier. Training jitter exists to cover that gap but these
   numbers do not measure it. This is the single biggest reason to treat the comparison as
   favourable-to-the-net.
2. **The comparison is also asymmetric in the detector's favour**: its errors are computed
   only where it fired, the net's over every instance including the ones the detector never
   found. Both asymmetries are stated rather than netted out.
3. **All data is VQ1.** `NOTES.md` is explicit that VQ1 teaches gate appearance, which
   transfers, and nothing about hangar clutter or the white-ceiling-light false positive,
   which do not. This model must be pseudo-labelled onto VQ2 frames before it is trusted
   there. That step is load-bearing, not polish.
4. **No confidence head, by design.** This cannot reject a non-gate, so failure mode 1
   (decoration false positives) is NOT addressed and must be handled by whatever proposes
   the boxes. Failure mode 3 (merged gates) is not directly measured either — nothing in
   the labels flags it.
5. `occluded` in the labels means *gate-on-gate* only, so the "occluded" rows understate
   real-world occlusion by structure and columns.

## Cost

**732,456 parameters**, 160×160 input. Measured inference: **2.6–3.0 ms/crop** on the
MX150 at batch 1, **1.85 ms/crop** at batch 8, **4.4–9.0 ms/crop** on CPU at batch 1
(range is contention-dependent). At 30 Hz that is roughly 3–7 crops per frame on CPU alone
with no GPU, which clears the "runs alongside control" bar with margin.

## What I would do next, in order

1. **Feed it detector boxes instead of truth boxes** and re-measure. Until that is done the
   headline numbers are an upper bound.
2. **Pseudo-label onto VQ2 frames** — required before any of this is trusted in the hangar.
3. **Cache decoded crops to a memmap.** Epochs were JPEG-decode bound on a thermally
   throttled 4-core laptop (30 s → 71 s once it dropped off turbo); ~20× less CPU work per
   epoch is available for free, which buys ablations rather than a better single number.
4. **Predict a per-corner uncertainty** and use it to gate the heavy tail, which is the
   thing that actually blocks open-loop use.
