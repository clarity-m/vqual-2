# VQ2 hand labelling — what to do

Two separate jobs. **Do the triage first if you only have time for one** — it unblocks a
decision about gatenet that nothing else can settle.

---

## Job 1 — label triage (~10 min, 223 slides, one keystroke each)

Answers: are the clipped / large auto-labels *wrong*, or is the model failing on correct
labels? If the labels are bad the fix is in `autolabel.py`; if they are good the model
needs loss weighting. Nothing cheaper settles it.

1. Open `pilot/perception/triage/triage.html` (double-click; any browser).
2. Each slide shows one auto-label: **whole frame** on the left, **zoom** on the right.
   The green quad is the label. Magenta dot is corner 0.
3. Press **Y** if the green quad sits on the gate's inner aperture, **N** if it does not,
   **S** if you cannot tell. It advances automatically.
   *Corners outside the image are meant to be there* — the labels are amodal, so judge
   whether the quad is where the aperture **would** be.
4. Press **E** when you stop (or finish) and keep `triage_verdicts.json`.

Autosaves every keypress, resumes where you left off. The slides are deliberately mixed
and unlabelled: some are ordinary easy gates acting as a control, and knowing which is
which would spoil the comparison.

---

## Job 2 — gate corner labels (~30 min, 159 frames)

1. Open `pilot/perception/labelui.html`.
2. Click **Open folder**, choose `pilot/perception/vq2_label/frames`.
   Chrome will ask to upload 159 files — confirm; nothing leaves the machine.
3. Label the **inner aperture** of every gate: 4 clicks per gate, back to front.
   * `O` occluded, `C` clipped, `U` unsure — on the selected quad.
   * `Enter` = "reviewed, no gate here" (a **verified negative**, different from skipping).
   * `R` = put the frame back to **unreviewed**, as if you had never opened it — the
     inverse of `Enter`. It removes the frame from the export entirely (deleting the last
     quad does *not*: that leaves a verified negative). Confirms first if the frame has
     labels, because they are discarded.
   * `N` / `P` — or `PgDn` / `PgUp` — jump to the next / previous **unreviewed** frame.
   * `Backspace` undoes; `?` shows all keys.
4. Press **S** to export `labels_gates.json`. Do this every few minutes, not just at the
   end — localStorage on a `file://` page is not worth trusting with 30 minutes of work.

### Zoom, and near gates whose corners are off the image (added 2026-08-01)

`+` / `-` (or the wheel) now go **below 1x** as well as above it:

| level | what it is for |
|---|---|
| `12x … 1.5x` | fine corner placement, 0.5 steps — unchanged |
| **`1x`** | one image pixel per screen pixel |
| **`0.75x` `0.5x` `0.35x` `0.25x`** | **new** — zoom OUT to reach amodal corners far outside the frame |

A gate that fills the frame has its corners hundreds of pixels off the image, sometimes
above the top *and* below the bottom at once. Press `-` a few times: at **0.25x** the
entire clickable area — **±1400 image px around the frame**, 860×790 on screen — is
visible at once, so every corner is reachable without scrolling. The real 640×360 image
stays marked by the bright dashed white rectangle, and corner dots, lines and the
residual readout keep their size on screen however far you zoom out, so nothing shrinks
into invisibility. Click accuracy is exact at every level (measured: 0.00 px at 0.25x,
0.35x, 0.5x and 0.75x).

Zooming keeps whatever you were looking at in the middle of the view, so `-` then `+`
comes back to the same place. Corners are stored **unclamped** as always; the margin only
decides what is clickable.

**Order is deliberate.** Files are numbered so that whatever fraction you get through is
a balanced sample across all six categories, not the first N seconds of a lap. Just work
top to bottom and stop whenever.

### PnP assist (optional, added 2026-08-01)

The gate aperture is a known 1.5 m square and the camera intrinsics are known, so the
tool can fit a physically exact square projection to your clicks. All of it is additive —
nothing changes if you never press the keys.

* **Q — snap.** Replaces the selected quad's 4 corners with the nearest exact square
  projection and shows the pre-snap residual (how far your clicks were from any real
  square: green < 0.3 px, amber 0.3–1, red > 1). **Z** reverts the last snap (one level).
* **Q with exactly 3 corners placed — solve the 4th.** The solver drops in the 4th
  corner, drawn as a **hollow dot** until you drag it (or leave it — it's a real corner
  either way). Click the 3 corners *consecutively around the quad*, not across a
  diagonal. Geometry caveat: 3 corners of a square admit **2–4 valid squares**, so if
  the placed corner is not on the gate, press **Q again** to cycle the alternatives —
  you can see the gate, the solver can't.
* **Live residual.** Every finished quad shows its residual under the z-tag, same
  colours. A red number on a gate you labelled carefully usually means one corner is
  mis-clicked — or the quad genuinely isn't a square projection (see below).
* Amodal corners are fine everywhere: snapping and solving never clamp, corners far
  outside the image are just numbers.

**When NOT to snap:** the solver models one thing only — a flat 1.5 m square through the
ideal pinhole. If what you're labelling isn't that in this view (bent/partial structure,
a merged blob you're splitting by judgement, anything where you're clicking where the
edge *would* be from context the solver doesn't have), trust your clicks over the
solver and skip Q. A high residual on an amodal guess is expected, not an error.

### What is in here, and why (159 frames, measured — see `index.json`)

| category | n | why it is here |
|---|---:|---|
| `deco` | 37 | wordmark / checkerboard false positives — the negatives a confidence head needs, and VQ1 has none |
| `ribbon` | 37 | cyan ribbon crossing the aperture **outline** (heaviest first) |
| `merged` | 27 | gates crowding or nested inside one another — failure mode 3, currently undetected |
| `clipped` | 27 | gate running off the image edge — failure mode 2 |
| `missed` | 24 | the detector returned **nothing** although a large gate fills the frame |
| `empty` | 7 | genuinely gate-free — press `Enter` and move on |

`empty` is small because it genuinely is: of 4779 frames scanned, only 27 have no
detection *and* negligible orange. A lap almost always has a gate in view.

`missed` is the highest-value group per frame — those are exactly gatenet's worst band
(≥120 px and clipped), and there is currently no VQ2 label for any of them.

Frames come from the three VQ2 sessions, with the two regions `session-notes.txt` flags
in the primary lap (the index reversal and your 32 s backtrack) excluded.

---

## Afterwards — merging

```
python3 pilot/perception/mergelabels.py --hand <path to labels_gates.json>
```

Writes `pilot/perception/labels_merged.json`: the VQ1 auto-labels plus your VQ2 frames,
every instance tagged `src: "hand" | "auto"` so they can be weighted apart. Add
`--drop-unsure` for corner-regression training. It refuses to run rather than silently
drop a label it cannot place.
