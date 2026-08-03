# RESUME — vqual-2, paused 2026-08-03

The AI Grand Prix R2 submission window (2026-08-03 06:00 PST) closed without a
submission. The control policy — plant fit, surrogate, learned policy, everything below
the measurement vector — was not finished in time.

**Nothing measured here is invalidated by that.** The course is fixed, the sim build is
fixed, and the spec is fixed, so every perception result, the map, and the sign
conventions all carry forward unchanged to any future round or re-run. What expired was
a date, not a finding.

Read in this order when picking this up again:

1. This file — for what state the work was left in.
2. `pilot/NOTES.md` — non-visual course facts, the frame problem, spec constraints.
3. `pilot/perception/NOTES.md` — the vision source of truth (2026-08-02 sections are newest).
4. `pilot/CONVENTIONS.md` — **the single source for every sign, frame and axis fact.**
   Do not re-derive these and do not trust the other two over it.

## Where the halves stood

| half | owner | state |
|---|---|---|
| perception → `Observation` | Claire | WORKING. `producer.py` fills all 73 dims for real; see PRODUCER.md's real-vs-stubbed table. Nothing in it is a stub. |
| map | Claire | COMPLETE. `map_vq2.json`: one rigid component, 17/17 gates, 20 pairs, gate identity automatic from crossings. |
| live harness | Claire | WRITTEN, `autopilot.py` — producer → Policy → link layer, closes the loop on the sim. Run against `ServoPolicy`. |
| plant fit + surrogate + policy | Alex Kong, then Claire overnight 08-02/03 | NOT FINISHED. This is what the deadline was lost to. **THIS WORK IS NOT IN THIS REPO** — see below. |

## The control half lives on Alex's machine, not here

Claire took control over on the night of 2026-08-02/03 so Alex could sleep, and worked on
HIS computer. Nothing from that effort — noise model, generated environment built from the
VQ2 map, gate-pass-rate plateau investigation — exists in this repository. Confirmed
2026-08-03: no merge commits at all, one stale worktree branch from 07-31, nothing
committed since 08-01. Do not go looking for it here.

**A merge on that machine silently removed the noise model, some of the controls, and
adjusted difficulty parameters.** Claire discovered this only after the fact. Two
consequences, both of which matter more than the missed deadline:

1. **Recovery is very likely possible, on that machine.** Background agents worked in
   worktrees, so their branches should still exist there, and `git reflog` holds
   unreferenced commits for 90 days by default. `git branch -a`, `git worktree list`,
   `git reflog`, `git fsck --dangling` — in that order. No rush, but not infinite either.

2. **THE PLATEAU FINDING IS SUSPECT AND MUST BE RE-MEASURED.** If the merge dropped the
   noise model and moved difficulty parameters, then a gate-pass-rate plateau measured
   after it was measured on a silently different problem. Treat "control plateaus at
   rate X" as unproven until it is reproduced with the noise model restored and the
   difficulty parameters confirmed. This is the same failure that has now bitten this
   project four times — the measurement SETUP quietly decided the answer — except this
   time the setup was corrupted by tooling rather than by design, which makes it harder
   to smell. Check the harness before believing the plateau.

The interface between them (`pilot/interface.py`, OBS_DIM=73, 4-D acro `Action`) is
frozen and was never the bottleneck.

## The three gaps that stand between "policy exists" and "policy flies"

Written 2026-08-03 while the reasoning was fresh. These are about DEPLOYMENT, and only
the third is about the policy itself.

1. **torch is not installed on the CAEN VM.** The roaming `pip --user` set is
   `numpy==1.26.4`, `opencv-python==4.10`, `pymavlink`, `keyboard`. Both gatenet and any
   learned policy need torch there. Per `user-vm-cmds.md`, an unpinned install pulls
   numpy 2, shadows conda's 1.26.4, and breaks matplotlib/scipy/numba — so this is a
   pinned CPU-wheel install, verified once, never a race-day step. `--no-net` runs
   torch-free but discards gatenet, which is where the coverage win lives.

2. **Timing on VM hardware is unmeasured.** Producer total is 29.2 ms median against a
   ~33 ms frame budget — from REPLAY, not from the VM with the sim rendering and owning
   the GPU. No record of `autopilot.py` having been run on the VM appears in the notes or
   sessions; treat it as unrun until shown otherwise.

   **→ This is the cheapest first move on resuming.** Run `autopilot.py` on the VM with
   the existing `ServoPolicy`. It tests the harness, the install and the timing budget
   at once, and it does not depend on the policy half existing.

3. **The interface guarantees shape, not statistics.** A policy trained against the
   surrogate sees surrogate-generated Observations; live it sees producer output with
   real dropout, staleness, coasting and residual false positives. Same 73 floats,
   different distribution. This presents as "great on the surrogate, bad in the sim",
   which reads like a sim-to-real plant gap and sends you off tuning the noise model —
   `interface.py` warns about exactly this failure under SIGN CONVENTIONS. Guard:
   replay recorded sessions through the producer and train on THOSE obs vectors, so the
   policy meets real perception noise before it meets the sim.

## Known-open, carried forward

* Partial-quad fitting (fit the clean sides, intersect) — one fix for two of the four
  recorded detector failure modes, clipping and ribbon-occludes-aperture. Was the
  highest-value next code change and still is.
* `R_COMMIT` (attention handoff range) still unset; measurable from VQ1 flying.
* Pair 1-2 = 8.32 m (n=5) vs the refused net 13.0 m (n=34) — unresolved tension in the map.
* Frames are backed up nowhere: gigabytes of JPEG, git-excluded, one laptop. The
  telemetry CSVs are tracked, so the plant fit does not depend on that laptop, but the
  frames do.
