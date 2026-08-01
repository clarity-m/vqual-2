# Handoff — plant refit from Card 2 (2026-08-01)

Session context: [card 2 plant / surrogate](41e5c45f-73bd-415a-9d03-cfb8960de0d5).
Working tree is **uncommitted**. Base commit was `89ccb8e` (VQ1 plant + sysid pipeline).

## What was asked

Card 2 maneuvers A/B/C were flown; D was not, and the user does not want to manually
pilot D. Question: how to improve the surrogate from here. Chosen path: **refit the
plant** (apex/terminal analysis → thrust curve + `kz` → replay gate), then scope what
the remaining data unlocks without more piloting.

## What landed

### Pipeline

```
python3 pilot/control/sysid_frames.py  pilot/sessions/*/   # full corpus; see caveat
python3 pilot/control/sysid_latency.py pilot/sessions/*/
python3 pilot/control/sysid_apex.py                        # NEW — card 2 direct reads
python3 pilot/control/sysid_fit.py                         # -> plant.json
python3 pilot/control/sysid_replay.py                      # gate
python3 pilot/control/plant.py                             # self-check
python3 pilot/control/surrogate/selfcheck.py               # 15/15 vs this fit
```

### New / modified files (this work)

| path | status | role |
|---|---|---|
| `sysid_apex.py` | **new, untracked** | `kz` + low-throttle / full-throttle `T/m` from apex arcs and terminal runs |
| `sysid_fit.py` | modified | order: apex → `kz` → bins → knots → plant; joint vertical fit kept as comparison only |
| `sysid_data.py` | modified | `flying()` / `usable()` accept `min_thrust=0` so zero-throttle data enters |
| `plant.py` | modified | thrust = `np.interp` over `thrust_knots`; clamp removed |
| `plant.json` | modified | refit result |
| `README.md` | modified | numbers, card 2 findings, open plant + surrogate items |
| `TRAINING_ARCHITECTURE.md` | modified | half of P3 is not blocked on Claire |

**Not produced by this session** (already untracked on disk): `surrogate/`, `train/`,
`policies/`, `evalsuite/`, `sysid_card2.py`, `.cursor/`.

### Fitted plant (current `plant.json`)

| | before (`89ccb8e`) | after |
|---|---|---|
| `kx` / `ky` | 0.0487 / 0.0496 | unchanged |
| `kz` | 0.0364 (joint with thrust) | **0.0436** from apex arcs |
| thrust | affine `c0+c1·thr` + `max(0,…)` clamp | **21-knot table** |
| hover throttle | ~0.266 | **0.270** (matches flight ~0.27) |
| full `T/m` | ~53.5 | **51.7** |
| open-loop replay / 5 s | 2.6 m | **2.1 m** |
| held-out body z R² | 0.916 | **0.902** (see open items) |

Fit sessions: `150712`, `143025`, `144815`, `20260801-004843` (A), `005059` (B).
Hold-out: `131305` (unchanged across both fits).
Excluded: `130744`, `233219` (stale velocity stream), `005518` (identical rerun of `005059`).

### Why the refit was not a tweak

- Racing flight has `corr(throttle, w|w|) ≈ −0.90`, so joint `kz`+thrust was
  unidentified. Apex arcs park throttle and coast through `w=0` → `kz` is a slope.
  Four wide arcs: **0.04359 ± 0.00016**. Terminal descents (thrust ≡ 0): **0.0435 / 0.0436**.
- Affine thrust went negative below thr 0.10 and was clamped → model predicted free fall
  where the drone holds ~2.3 m/s². Table has deadband, steep mid rise, flatten at top;
  worst direct-read miss 0.05 m/s² vs 2.7–4.0 for polynomials.

### Validation status

| gate | result |
|---|---|
| `sysid_apex.py` | PASS |
| `sysid_fit.py` | writes `plant.json` (exit 0) |
| `plant.py` | PASS |
| `sysid_replay.py` | PASS (corruptions clearly worse) |
| `surrogate/selfcheck.py` | 15/15 PASS (hover 0.270, terminal climb 30.99 m/s) |
| `sysid_frames.py` **all sessions** | PASS, margin **2.6×** |
| `sysid_frames.py` card 2 alone | FAIL at 1.0× — attitude variety thin; **same winning signs**; do not treat as convention failure |

## Findings that change the next plan

### Do not fly D

`130744` and `233219` fail the kinematic referee because `LOCAL_POSITION_NED` velocity
**repeats bit-identically** in 53% / 43% of rows while the drone maneuvers hard. The
referee differentiates a held signal. Re-flying reproduces the recording artefact; it is
not a fact about the simulator that more piloting will fix.

### Card 2 C did not complete — but the lap geometry already exists

`20260801-005715` crashed ~14 s in at gate 0. Two earlier sessions flew the whole
**6-gate** map with crossing points agreeing to **0.5–1.2 m**:

- `20260731-195307`
- `20260731-204841-vq1-lap-slow` (also has **2937 JPEGs**)

Measured course (NED mean of those two; start at origin):

| gate | x | y | z | seg (m) | turn (°) | elev (°) |
|---|---|---|---|---|---|---|
| 0 | −23.5 | 0.05 | −1.0 | 23.5 | — | +2.4 |
| 1 | −47.3 | −2.3 | 3.8 | 24.4 | +5.7 | −11.2 |
| 2 | −75.1 | 0.9 | 12.1 | 29.2 | −12.1 | −16.6 |
| 3 | −112.2 | −5.1 | 23.8 | 39.4 | +15.6 | −17.3 |
| 4 | −135.8 | −0.8 | 24.0 | 23.9 | −19.4 | −0.4 |

vs `course.py` assumptions: mean seg 28.1 m OK, but max seg **39.4 > 34**; turns
**alternate**, not persistent; **monotone ~24 m descent** over 140 m — generator is
mean-reverting on altitude and **cannot** produce that profile. Caveat: this is VQ1's
map; VQ2 winds more — use for vertical/segment priors and fixed eval, not to shrink
turn magnitudes.

### Detection visibility is measurable and the fallback is wrong

Feasibility probe (deleted after run): project measured gate centres into frames of
`204841-vq1-lap-slow` with truth pose + `camera.py`, score `NOTES.md` orange HSV mask.

- In-frustum (gate, frame) pairs: blob at predicted place on **~95%**, including **~95%
  at 30–45 m**
- Fallback `noise.py` `max_range_m = 14–30` → trains search behaviour the real course
  does not need
- Blob width ~**2.1×** `480/range` — mask fits **2700 mm outer** frame; spec rangefinder
  assumes **1500 mm inner** aperture

Visibility half of P3 does **not** need Claire's detector. Error model of a real PnP
fit (normal validity, hangar FPs) still does.

## Open issues (honest)

### Plant

1. **`kz` vs `|u|`** — 0.0443 at `|u|<2`, 0.0412 at 10–15. Body-lift `c·u²` buys
   **+0.026** held-out R² over refitting `kz` alone (0.920 → 0.946), **+0.043** over
   shipped plant (0.902). Alternatives worse (`c·u|u|` −0.015, `|u|w` +0.014). Not in
   `plant.json`, not replay-gated. Explains why held-out body-z R² *fell* slightly after
   the refit: old joint `kz` absorbed lift.
2. **Throttle 0.50–0.82 thin** (~240 samples). Ends anchored hard.
3. Only **4/8** apex arcs used for `kz` (0.15/0.20 reps: `w` span &lt; 9 m/s → excluded
   on span, not answer). Terminal descents **never plateaued** (+1.7 m/s over last
   second) — fitted slopes that happen to agree with arcs.

### Surrogate (larger than remaining plant gap)

4. **Detection noise still guessed** — visibility ranges should be replaced by measured
   curves from existing frames + measured gates.
5. **Course generator structural miss** — no sustained one-way vertical trend.
6. **No fixed eval course** — every episode is generated and thrown away; cannot compare
   surrogate vs live lap times. Measured 6-gate map is the only course you can fly live.

### Hygiene

7. **Nothing committed** — including the whole `surrogate/` / `train/` / `policies/`
   stack that was never in git.
8. **`SYSID.md` still describes card 2 as unflown.**
9. Checkpoint `train/checkpoints/smoke256_*` post-dates the refit; older
   `selftest` / `wiringcheck` names from earlier status are gone from disk.

## Recommended next steps (zero piloting)

1. **Measured detection visibility** — sysid-style stage over recorded frames + projected
   truth; replace `noise.py` visibility ranges. Highest leverage silent wrongness.
2. **Course** — widen segment range, add sustained descent prior, land 6-gate map as
   fixed eval course.
3. **Body-lift `c·u²`** — add to plant, re-run replay + surrogate self-check.

Optional flying (not blocking): terminal holds at thr 0.6 and 0.8 (~30 s) to thicken
the middle of the thrust table. **Do not fly D.**

## How to re-verify quickly

```powershell
cd pilot\control
python sysid_apex.py
python sysid_fit.py
python plant.py
python sysid_replay.py
python surrogate\selfcheck.py
# frames: pass ALL session dirs, not card-2 alone
$s = (Get-ChildItem ..\sessions -Directory).FullName
python sysid_frames.py @s
```

## Sessions reference (card 2)

| session | role |
|---|---|
| `20260801-004843` | A — eight apex arcs (thr 0.05/0.10/0.15/0.20 × 2) |
| `20260801-005059` | B — terminal up/down (in fit) |
| `20260801-005518` | B rerun (excluded from fit; repeatability) |
| `20260801-005715` | C attempted — crash at gate 0, no usable lap |

## Doc sync status

- `control/README.md` — updated for this refit
- `TRAINING_ARCHITECTURE.md` — P3 visibility note added
- `SYSID.md` — **not** updated (still reads as pre-flight)
- This file — handoff for the next agent / human
