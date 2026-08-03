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

---

# Runbook: the single audited v3 retrain (prepared 2026-08-01, not yet run)

Everything below is prepared and smoke-tested; when the VQ2 hand labels arrive it is
merge + upload + run-all, no code left to write. Baseline hyperparameters are held
(batch 256, lr 3e-4*(batch/64), 200 epochs, seed 0, block split); the ONE training
variable that changes is `--weight-mode rate` (down-weight fast-rotation auto-labels,
1.0 -> 0.30 raised-cosine over 0.5-2 rad/s; hand labels always 1.0 -- rationale in
`gatenet.py::instance_weight`). Negatives stay unused: no confidence head this run.

## Steps (Claire)

1. **Merge** the hand labels (from `labelui.html`, usually `labels_gates.json` in
   Downloads) into the audited v3 auto-labels:

   ```
   python3 pilot/perception/mergelabels.py \
       --hand <path to labels_gates.json> \
       --auto pilot/perception/autolabels_vq1_v3.json \
       --drop-unsure \
       --out pilot/perception/labels_merged_v3.json
   ```

   It refuses to run if any hand key cannot be mapped -- that is it protecting your
   labels, read its message. Then build the VQ2 frame zip:

   ```
   python3 pilot/perception/_mkframezip_vq2.py
   ```

2. **Upload to Drive**, folder `MyDrive/vqual2/` (same one as before):
   * into `perception/`: current `gatenet.py`, `packcrops.py`, and
     `labels_merged_v3.json` (replace the old copies of the .py files -- they gained
     `--weight-mode` / `--labels`); `autolabel.py` + `label.py` are unchanged and
     already there.
   * into the folder root: `vq2_frames.zip` (new). `vq1_frames.zip` and
     `gatenet_runs/block/best.pt` are already there from the baseline run; the
     notebook checks for all of them before training and stops early if one is missing.

3. **Open `pilot/perception/gatenet_colab_v3.ipynb` in Colab** (GPU runtime),
   **Run all**. ~3 h cap. If it disconnects: reconnect, set `RESUME = True` in the
   train cell, Run all again -- it continues from the last mirrored checkpoint.

4. **Read the decision off the final cross-eval cell.** It scores the new checkpoint
   AND the production model `gatenet_runs/block/best.pt` on the IDENTICAL new val
   split and prints a delta table plus an ACCEPT/REJECT line.

## Accept/reject rule (decided now, before the numbers exist)

**ACCEPT** (new model becomes production) only if BOTH:
* the new model beats `block/best.pt` on the cross-eval table's **ALL** row
  (corner med), and
* it does **not regress the `clipped` row** (clipped is where the net earns its keep
  over the detector).

Otherwise **production stays `gatenet_runs/block/best.pt`** and the run is a diagnosis,
not a loss: the body_rate / src breakdown rows separate label error from net error for
the first time. Never compare either model's number against the old 0.89 px headline --
the label file changed, so the val set changed; only the cross-eval table is a
comparison (this exact confusion is what the filtered-v1 "regression" turned out to be).

Afterwards: copy `gatenet_runs/colab-v3/` back into `pilot/perception/gatenet_runs/`
and paste both cross-eval tables + the decision table here under a heading naming the
GPU. Smoke evidence for the prep itself: `smoke_tmp/` (disposable; fake 'hand' tags,
2 epochs, res 64 -- its numbers mean nothing beyond "the plumbing works").

## 2026-08-01 late: leakage audit overturns all three REJECTs -- production is colab-v3-nw

Every cross-eval today graded checkpoints on the val split of the CURRENT label file.
Deleting labels (v1 -> v3) shifted every block boundary, so 35% of that val sat inside
the OLD model's training blocks -- 68% of the >=120 px bucket, 45% of clipped, exactly
the rows that decided REJECT each time. Claire flagged the pattern ("something feels
off"); the audit (val_clean_keys.json, crosseval_clean.py) scored all three checkpoints
on the 1073 instances held out under BOTH splits:

  clean subset          OLD block     v3+rate      v3 no-wt
  ALL                   1.26/5.20     1.08/4.81    1.06/4.17   (med/p90 px)
  clipped               2.73/35.65    3.21/37.50   2.61/36.85
  clipped & >=60 px    15.37/72.87    8.65/61.86   9.29/53.85
  size >=120           34.07/97.05   34.56/93.05  35.61/88.16

VERDICT: colab-v3-nw (v3 labels + Claire's 364 VQ2 hand labels, NO rate weighting)
beats the old model everywhere on clean data, including clipped. ACCEPT under the
frozen rule. The old model's 10 px >=120 headline was memorization; the honest
close-range number for EVERY model is ~34 px median -- the PnP residual gate and
temporal consistency remain load-bearing there. Rate weighting: mildly harmful, drop it.

Rule for every future cross-eval: two checkpoints are comparable ONLY on instances
held out of BOTH their training splits. Any label edit moves the block boundaries,
so recompute the clean intersection (mkcolab stages val_clean_keys.json) or the
incumbent wins by home-field advantage.

---

# Runbook: the confidence head (`colab-conf`, prepared 2026-08-02, not yet run)

`gatenet_conf.py` adds the per-crop trust/reject signal -- the last known perception
gap (decoration false positives, coherent hallucinations, the PnP residual's blind
spot). Design: a 328k-param head on the FROZEN production trunk
(`gatenet_runs/colab-v3-nw/best.pt`), two outputs -- gate-vs-not logit, and predicted
log10 regressor error (the "clipped/degraded" signal as a continuous quantity; a third
class was considered and rejected in the module docstring). The regressor path is
bit-identical with the head attached (asserted, `torch.equal`). Marginal flight-time
cost: **0.21-0.24 ms** on this laptop's CPU (head on the shared feature map; budget <1 ms MET).

Data (after per-run dedupe, cap 6/run): class 1 = 12306 merged-v3 instances (85
unsure excluded); class 0 = **743**: 433 mined VQ2 decoration FPs, 167 behind-gate,
95 no-orange phantoms (17 degenerate boxes excluded per provenance), 48
verified-empty randoms. Split: 240-frame blocks on the raw frame counter, 45-frame
guard -- val holds 147 negatives (87 deco). Mining safety argument and the
LABEL_POLICY.md invariants are in `gatenet_conf.py` (`mine()`, `check_invariants()`);
the smoke run re-asserts them, and the notebook re-asserts them on the uploaded files.

## Accept/reject rule (decided now, before the numbers exist)

**DEPLOY the head only if it rejects >= 80% of decoration false positives at <= 2%
true-gate loss on the clean eval set** (clean = conf-val positives on
`val_clean_keys.json` frames + conf-val hand instances; deco = conf-val mined crops).
Catastrophe catch vs the PnP-residual baseline (84% of >10 px at 12.6% flags) is
reported as a complement, not a gate. If the frozen head fails, `FINETUNE=True` is the
escape hatch and then the notebook's cross-eval cell must clear the finetuned
regressor on the clean subset (ALL + clipped rows) before the PAIR ships.

## Steps (Claire, tonight)

1. `python3 pilot/perception/mkcolab.py` -- already run; re-run if anything changed.
2. Upload `colab_upload/perception/` -> Drive `MyDrive/vqual2/perception/` (replace on
   collision) and `colab_upload/drive_root/` -> `MyDrive/vqual2/`. Only the files
   marked new/STALE actually need to move; `conf_frames_vq1.zip` (1.6 MB) and
   `gatenet_conf_colab.ipynb` are the two that must land in the folder root.
   Already on Drive, do not re-upload: `vq1_frames.zip`, `vq2_frames.zip`,
   `gatenet_runs/colab-v3-nw/best.pt`.
3. Open **`gatenet_conf_colab.ipynb`** in Colab (GPU runtime), **Run all**. ~1 h.
   If it disconnects: reconnect, set `RESUME = True` in the train cell, Run all again.
4. Read the decision off the eval cell: the threshold table prints the ACCEPT/REJECT
   line against the frozen rule above; `eval_report.md` is mirrored to
   `gatenet_runs/colab-conf/` on Drive.

Afterwards: copy `gatenet_runs/colab-conf/` back into
`pilot/perception/gatenet_runs/` and paste the eval tables + decision here. Smoke
evidence for the prep: `smoke_tmp/conf_runs/` + `smoke_tmp/conf_neg_sheet.png`
(2 epochs, CPU, subset -- plumbing proof only; SMOKE PASSED 2026-08-02, latency
0.21-0.24 ms marginal, invariants hold, regressor bit-identical).

## 2026-08-02: confidence head VERDICT CORRECTION -- ACCEPT at p=0.020

The colab-conf eval printed REJECT, but its accept-line implementation checked only
the threshold TARGETED at 80% deco rejection (79.3% = 69/87, a discretization miss)
instead of asking whether ANY operating point satisfies the frozen rule. One row
down: p=0.020 rejects 89.7% of decoration FPs at 0.00% true-gate loss (0/233 clean
positives; 95% CI upper ~1.6%, inside the <=2% rule). The frozen rule as WRITTEN
("deploy only if >=80% deco rejected at <=2% true-gate loss") is satisfied with
margin. ACCEPT. Operating point p=0.020, one step below the edge on purpose.

Bonus: the predicted-corner-error output catches 100% of >10 px catastrophes at a
5% flag rate on head-unseen val (n=5 small; 85.7%->100% by 15% flags on the 1352-
instance diagnostic set) vs the PnP residual's 84% at 12.6%. Flight stack now has
three independent rejectors: interior colour, PnP residual, learned confidence.

Meta-lesson, same family as the leakage audit: a frozen rule protects nothing if
the CODE checking it paraphrases it. The check must quantify over operating points
exactly as the rule text does.

## 2026-08-02: rotation augmentation ACCEPT -- production is colab-v3-rot

Same-split cross-eval (labels_merged_v3, no leakage possible) AND clean subset agree:
clipped 3.15 -> 2.87 (clean 2.61 -> 2.26, best posted), >=120 13.09 -> 12.66,
60-120 clean p90 8.67 -> 7.59; every other row within +-0.1 px noise. Camera-roll
augmentation (+-23.1 deg exact homography about the tile centre, gatenet_rotaug.py)
was the single variable. Checkpoint gatenet_runs/colab-v3-rot/best.pt (copied local).

PAIRING NOTE: the colab-conf confidence head was trained FROZEN on the colab-v3-nw
trunk. Its calibration is invalid on rot features. Until the head is retrained with
CKPT=colab-v3-rot (one config change, ~1 h), confidence scoring requires the nw trunk;
do not attach the existing head to rot.

## 2026-08-02: colab-conf-rot ACCEPT at p=0.082 (same verdict-line bug, same correction)

Head retrained frozen on the colab-v3-rot trunk (production pairing). Notebook again
printed REJECT by checking only the 80%-target row (79.3%); the 90% row satisfies the
frozen rule with margin: 89.7% deco rejected at 0.00% clean-gate loss. ACCEPT,
operating point p=0.082. AUC 0.9989/0.9992.

Catastrophe channel: 5/6 caught on head-unseen val at any flag rate (one stubborn
miss); diagnostic set 92.1% @ 5% flags but plateaus 93.7% (nw head reached 100% by
15%). Operational read: decoration rejection excellent, catastrophe channel ~PnP-
grade; it deploys LAYERED with the PnP residual + interior-colour test, so the miss
has independent backstops. Deployed pair: colab-v3-rot trunk + colab-conf-rot head.
