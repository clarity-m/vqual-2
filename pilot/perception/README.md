# perception/ — Claire + Claude

Everything **above** the measurement vector: fill an `interface.Observation` each frame.

This file is the **design**: what this half owns and how it is meant to work. `NOTES.md`
next to it is the **record**: what has actually been measured, what currently works, and
the traps that cost a build. Vision facts go there, not into `../NOTES.md`.

## What this half owns

1. **Gate detection + PnP.** Red/orange segmentation → contour → quadrilateral fit →
   `solvePnP` against the known 1500 mm inner square → gate pose *including its normal*.
   The normal is what makes "enter at the gate normal" expressible at all: the guidance
   target is `pos_body + d * normal_body`, a virtual approach point behind the gate.
   Degrade in the defined order — PnP → centroid+size → coast on staleness → invalid.
2. **Ribbon extraction.** Cyan corridor → bearing samples at increasing look-ahead.
   Explicitly not load-bearing: it is absent from long stretches of real flight.
3. **The attention policy** — what to look at, when to hand off, what to do when nothing
   is visible. The hard half of "attention"; the yaw servo that serves it is trivial.

## Notes

* Frame spans **+49.4° to −9.4°** about body-forward (the spec's "VFoV 90°" is the
  *horizontal* FoV; true VFoV is 58.7°). The narrow lower edge is the binding constraint:
  a gate at own altitude renders low, and pitching down to accelerate pushes it lower.
* **Un-rotate by roll before detecting.** Roll comes from the gravity vector in
  `HIGHRES_IMU` — directly observable, no estimator.
* Range from apparent size: `range_m = 480 / gate_px`, spec-exact.
* `active_gate_index` closes the loop *behind* the aircraft. Vision looks forward; the race
  packet confirms crossings, so no vision budget is spent on a gate already committed to.
