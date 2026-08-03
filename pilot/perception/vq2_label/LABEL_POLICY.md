# VQ2 hand-label policy (Claire, 2026-08-01)

What the labeller was told to do, recorded because the data cannot express it.

## Rules in force

* **At most 3 gates per frame**, chosen as the largest / most prominent in view.
* Label a gate only if all four corners can be placed without guessing (roughly
  >=15-20 px apparent size). Marginal ones get `unsure` rather than a guessed quad.
* Amodal corners are placed outside the image, never clamped.
* `clipped` = runs off the image edge. `occluded` = **gate-on-gate only**.
  Background structure covering a gate gets no flag: label it if the corners are
  confident, `unsure` if not, skip it if it is mostly hidden.
* The frames in `frames/` prefixed `NNN_` are ordered so that ANY prefix Claire
  completes is a balanced sample across the six failure groups.

## The consequence that matters

**These frames are NOT exhaustively labelled.** A frame may contain real gates that
carry no label — small background gates, and anything past the 3-gate cap.

Safe: the current pipeline. `gatenet` is a crop regressor; each label becomes one
training crop and unlabelled image regions are never used as evidence of absence.

NOT safe without re-labelling: anything that treats "unlabelled region = background".
That includes a detector trained on full frames and a confidence head whose negatives
are sampled from unlabelled areas. Confirmed negatives live in explicit negatives
files (`labelfix_negatives_v3.json`, reason `behind-gate` / `no-orange`) built from
measured evidence -- use those, never the complement of the hand labels.

The 7 frames in the `empty` group ARE verified-empty and may be used as negatives;
they were checked individually, not assumed.
