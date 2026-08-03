# VISUALIZER — 3D rollout viewer for the surrogate

**Status:** built and verified. **Written:** 2026-08-02.
**Files:** `evalsuite/viz3d.py` + `evalsuite/viz3d_template.html`. Purely additive — no
changes to `env.py`, `single.py`, `course.py`, `camera.py`, or `run_policy.py`.

```powershell
python pilot/control/evalsuite/viz3d.py --policy none --seeds 0-3
python pilot/control/evalsuite/viz3d.py --policy baseline --seeds 0-9 --open
python pilot/control/evalsuite/viz3d.py --policy rl --ckpt run1_s5046272 --seeds 0-5 --difficulty 0.0
```

---

## 1. Why

`run1` logged ~5.05M steps at **0% completion** (`CONTROL_STATE_AGENTS.md` §5) and the
reactive baseline stalls around gate 2 of 18–22. Both state docs put "explain the zero
completions" at the top of the work queue.

The only inspection tool for that was `_probe_baseline.py`, which prints one line per
seed: a cause and a miss vector. That says a gate frame was struck 0.9 m off-centre. It
cannot say what approach put the aircraft there, whether the gate was ever in camera
frame, or how near the path ran to the corridor wall — and those are exactly the live
hypotheses. This writes the rollout out as geometry you can turn over.

## 2. What it found on first use

Worth recording, because it is a concrete answer to the open question.

**`run1_s5046272` at difficulty 0.0, normal starts, seeds 0–5: five of six episodes end in
`FLOOR`** within 3.3–7.0 s; the sixth clips a gate frame 1.17 m low. The path is a
ballistic arc — climb, stall, dive — terminating at ~13.6 m/s downward. It never turns
toward gate 0. The camera panel reads `slot0 g-1 conf 0.00 | fov 0`: no detection at all,
and the active gate never inside the frame.

So the failure is not "flies past gates" or "orbits". It is vertical, it is immediate, and
it happens with the gate out of view the whole time. Detection visibility (P3 fallback) and
the vertical control law are implicated ahead of reward horizon or gate tolerance.

For contrast, the baseline at difficulty 0.2 over seeds 0–9: 4 gate-frame strikes with
misses 0.44–0.81 m against a 0.75 m aperture radius, one ceiling, one gate-stall timeout.
Those are aim errors, a different failure entirely.

## 3. Settled decisions

| Decision | Choice |
|---|---|
| Renderer | Self-contained interactive HTML, **zero new dependencies**. Hand-rolled canvas 3D: drag-orbit, scroll-zoom, episode picker. No matplotlib/plotly/three.js/CDN — opens offline from `file://`. |
| Environment | **Local only.** No Colab/notebook inline path. |
| Path source | `--policy baseline\|rl\|none`; `none` renders bare courses. |
| Per-episode content | Termination marker + cause; corridor + floor/ceiling; gates as correctly-sized annuli with runout gates distinct; onboard camera view. |

The zero-dependency constraint is not incidental: the repo has **no** plotting library and
no `requirements.txt` / `pyproject.toml` anywhere, and `user-vm-cmds.md:87-90` warns that
unpinned pip installs shadow conda's numpy 1.26.4 and break exactly this class of library.

## 4. Three corrections the code forced

1. **Runout gates are non-physical.** `env._gate_geometry` gates collisions on
   `exists = idx < n_gates` (`env.py:584,591`) and `detect.tick` gates visibility on the
   same expression. The 3 runout gates cannot be hit and cannot be seen — they only shape
   the corridor polyline and the bisector normals. Drawn dashed, and the legend says
   *non-physical*, because otherwise the picture lies about where a crash could occur.
2. **The two existing tools disagree on termination priority.**
   `run_policy.REASON_FLAGS` (`run_policy.py:42-45`) is
   `finished → collision → corridor → timeout`; `_probe_baseline.py:62-64` is
   `finished → corridor → timeout → collision`. Same episode, two labels. This follows the
   probe, and ships all four raw booleans in the page so the label is never load-bearing.
3. **There is no gate-axis bug.** `_probe_baseline.py:57-59` uses `side = cross(up, n)` and
   `env.py:370` uses the exact negation, but env multiplies it by a zero-mean Gaussian, so
   the sign is unobservable there. This file picks the probe's and says so.

Two related traps, both now handled:

- `info["collision"]` merges gate-frame, floor and ceiling into one flag
  (`env.py:443-457`), so the subtype is re-derived from position against the env's **own**
  thresholds: FLOOR is `alt <= sphere_r` (`env.py:455`), CEILING is
  `alt >= ceil_h - sphere_r` (`env.py:456`). `_probe_baseline.py:68` uses a fixed 0.05
  tolerance instead and drops near-ceiling strikes into `collision?` — see §8.
- The timeout subtype must compare against **post-step** clocks. `env.py:484` tests
  `t_gate > gate_timeout_s` after the step advances it, and everything here is captured
  before; comparing the pre-step value reports a gate stall that fires at 11.99 s as a
  wall-clock timeout.

And the one that bites everybody: **`VecSurrogate` auto-resets inside `step()`**
(`env.py:530-532`), so the true state after a terminating step belongs to the next
episode. Everything is captured before the step; the terminal point is solved along the
last segment.

## 5. Python side (`viz3d.py`)

Reused rather than reimplemented, all from `run_policy.py`: `_Env`, `build_config`,
`make_policy`, `add_common_args`, `add_policy_args`, `parse_seeds`, `parse_pair`,
`override`, `config_dict`, `write_json`.

**`capture_episode`** flies one seed, recording privileged truth *before* every step at
full rate. **`classify_terminal`** extends `_probe_baseline.py:43-71`: same cause logic,
but the marker is placed at the solved crossing rather than at `p + v·dt`, which at 10 m/s
and 55 Hz is up to ~0.2 m past the event — enough to put it on the wrong side of a 0.6 m
frame ring. **`check_episode`** re-derives the corridor from `course.point_segment_distance`
and warns if the tool's model disagrees with `env._corridor_exit`; that is the bug which
would otherwise ship as a confidently drawn tube in the wrong place. Warnings print rather
than assert, so a real finding does not kill the run (`--no-check` disables).

`GateObs.index` (`interface.py:154`) supplies the absolute gate index per detection slot,
so the only privileged reads outside `single.true_state()` are `env.vec.start_p`
(required — anchors corridor segment 0), `env.vec.t_gate`, and `env.vec.det.is_fp`. All
wrapped in `getattr` with graceful degrade; `start_p` falls back to a back-projection and
flags itself in the page.

**JSON schema** — struct-of-arrays, rounded on emit (2 dp metres, 4 dp unit vectors and
quats, 3 dp times). Path is full rate; `detail` is strided (default 3, ~20 Hz) with the
last 40 steps and every gate-crossing step forced in. `meta.cam` ships `R_CB` and
`fwd_v = 296.5` explicitly so the page's ported projection cannot drift. `detail.fov` is
`camera.in_frustum` computed in Python — it exists to pin the JS port (§8.4).

**Units, stated once in the module docstring:** metres, seconds, radians. World is NED with
z DOWN, altitude is `-z`, `z_ceil` is negative. Body frame is FRD. Detections are body frame.

**Output** defaults to `pilot/evidence/<date>-rollout3d-<policy>[-<ckpt>].html`, matching
the dated-artifact convention in `.gitignore:6-9`. That directory is git-tracked, so each
run dirties `git status` — intended, but worth knowing.

## 6. Page (`viz3d_template.html`)

Two files rather than one because the repo uses `%`-formatting and a single `%` in embedded
CSS (`width: 100%`) detonates a `"..." % (...)` expression. The template carries one marker
line, `<script id="payload" …>`; Python substitutes `json.dumps(...)` with `</` escaped to
`<\/` (legal JSON, cannot terminate the tag early). The template ships with an empty payload
and is itself openable.

**Frame conversion happens exactly once**, at `toDisp(x,y,z) → [y, x, -z*KZ]` (NED → east,
north, up). Gate circles are generated in NED and converted per-vertex, so exaggeration
yields the correct ellipse rather than a scaled radius. `KZ` never touches the camera panel,
which is a real optical projection.

**Orbit camera** — target/yaw/pitch/distance, three dot products per vertex, no 4×4
matrices, pitch clamped to ±89° so the horizontal-right basis never degenerates. Near-plane
clipping at `zn = 0.05` via `clipSeg` for polylines and Sutherland–Hodgman `clipPoly` for
fills; without it the hand-rolled projection tears whenever geometry passes behind the eye.
Depth by painter's algorithm on view-space centroid, far-first, with the floor grid pulled
out and drawn first (a floor strike ends the episode at `sphere_r`, so nothing can be behind
it) and the ceiling drawn last at alpha 0.10.

**Primitives.** Gates are two coplanar circles (r 0.75 and 1.35) wound oppositely and
filled `evenodd`, which gives the aperture as a real hole; raced gates filled, runout
dashed grey, active accented, passed dimmed, each with a 2 m travel quill. The corridor is
*not* a global tube: the bright live pair — `prev→g[active]` and `g[active]→g[active+1]`,
with `prev = start_p` when `active <= 0`, exactly `env._corridor_exit` — always draws, and
the full hull is a toggle. Path is coloured by altitude margin `min(alt-sphere_r, ceil-alt)`
by default, with a ground shadow and drop lines carrying the depth cue. The termination
marker draws last with a cause label, and for GATE FRAME the in-plane miss vector annotated
`side / up / |lat|`.

**Camera panel** — 640×360 logical in a 320×180 box. Principal point at (320,180) and
**body-forward marked at v = 296.5**, labelled, because the 20° up-tilt is the single most
misread number in `camera.py`. Ground-truth gates as outlines, detections as filled discs
(alpha = confidence, dashed when stale, **red when `is_fp`**). A gate in front but outside
the frame gets an edge marker with its elevation, which is how the −9.4° lower limit shows
itself. Truth and detections are overlaid deliberately: `noise.py` is an unmeasured P3
fallback, so "locked onto a phantom" and "range-biased detection" are hypotheses only an
overlay can kill.

**Two unrolled strips** below the scene — altitude vs arclength (with floor and ceiling
bands and gate ticks) and lateral offset vs arclength (with ±`corridor_r`). These are
aspect-free and are where a FLOOR, CEILING or CORRIDOR death reads at a glance.

### Two defaults that changed after seeing it render

- **Vertical exaggeration defaults to 1× (true scale)**, not the 6× originally planned. The
  auto-fit now frames the *flown* region rather than all ~700 m of course, so the anisotropy
  it was meant to fix is mostly gone — and at 6× a gate circle becomes a 8 m tall ellipse,
  which defeats judging a 0.81 m miss against a 0.75 m aperture. Slider goes to 20× with a
  watermark whenever it is not 1.
- **The corridor hull defaults off.** At `corridor_r` ≈ 15 m the rings are larger than
  everything they contain. The live pair — the only segments that can end the episode —
  always draws.

Auto-fit frames path ∪ gates in `[start_gate-1, last_active+3]` ∪ terminal point;
<kbd>F</kbd> refits that, <kbd>G</kbd> fits the whole course.

## 7. CLI

Inherited via `add_common_args`: `--seeds --difficulty --speed-cap --decision-hz --n-gates
--time-penalty --max-steps --json --quiet`. Via `add_policy_args`: `--ckpt --gains
--supervisor --device`. Its `--policy` pins `choices=("baseline","rl")`, so the parser is
built with `conflict_handler="resolve"` and re-declares `--policy` with `none` added —
public API, no argparse internals touched.

| flag | default | why |
|---|---|---|
| `--out PATH` | `pilot/evidence/<date>-rollout3d-<policy>.html` | the artifact |
| `--open` | off | `webbrowser.open()` — stdlib |
| `--detail-stride N` | 3 | ~20 Hz detail; path stays full rate |
| `--tail-steps N` | 40 | full-detail window before termination |
| `--random-starts` | off | default forces `start_probs=(1,0,0,0,0)`. The training mix spawns 25% mid-course, 12% hovering and 8% offset by `σ = 0.45·corridor_r` — which can spawn *outside* the corridor and die on step 1. Required to reproduce `_probe_baseline`. |
| `--max-episodes N` | 24 | guard against `--seeds 0-999`; renders the first N and names what was skipped |
| `--no-detections` | off | drops `detail.det` for a smaller file |
| `--only-cause C` | none | filter to one ending, e.g. `FLOOR` |
| `--no-timestamp` | off | makes two runs byte-identical |
| `--full-config` | off | embeds `config_dict(cfg)` |
| `--no-check` | off | skip the geometry invariants |

## 8. Verification (all run 2026-08-02)

1. **Smoke, no torch / checkpoint / policy** — `--policy none --seeds 0-2`. 38 kB, <2 s.
   Exercises course generation, gate axes, corridor, floor/ceiling and the whole page except
   the trajectory. This is the path that works on any machine.
2. **Classifier vs the existing probe** — `--policy baseline --seeds 0-9 --difficulty 0.2
   --random-starts` against `_probe_baseline.py "" 10`. **All 10 seeds agree** on cause,
   gates, need, `t_s`, `alt`, the miss vector and the cue histogram.
   *One deliberate difference:* seed 4 (alt 11.85, ceiling 12.28, sphere 0.31). The probe
   reports `collision?`; this reports `CEILING`, because it uses the env's own
   `alt >= ceil_h - sphere_r` rather than the probe's `ceil - 0.05`. The probe is wrong here.
3. **Geometry invariants** — gate axes unit and mutually orthogonal to 1e-9, raced-gate
   altitudes inside the generator's band, and the corridor recomputed per path sample
   against `course.point_segment_distance`. No warnings on any seed exercised.
4. **Camera-panel cross-check** — `detail.fov` from Python's `camera.in_frustum` vs the
   page's ported `quatToRot`/`R_CB` projection; a mismatch counter shows in the rail. This
   is the only Python↔JS duplication in the design, and that flag is what pins it.
5. **Offline** — `grep -cE 'https?://|src=|@import|integrity='` → **0**; exactly one payload
   line; payload parses as JSON. Rendered headless in Chrome from a `file://` URL.
6. **Determinism** — two runs with `--no-timestamp` produce byte-identical JSON *and* HTML
   (sha256 `f71d2562…` / `aad42793…`).

## 9. Known limits

- **Priority disagreement (§4.2)** — `run_policy --json` and this tool can label the same
  episode differently. The raw flags ship in the page; the label is presentational.
- **Terminal extrapolation is first-order** in the segment it solves on. Collisions are
  tested per *substep* (`env.py:439-445`), so a plane crossed and re-crossed inside one
  decision step would alias away. It cannot happen at 10 m/s and 1/165 s substeps; the
  classifier returns `FRAME?` rather than asserting when nothing explains a collision.
- **Painter's algorithm has no correct answer** for a translucent gate ring intersecting the
  path. Accepted — the shadow, drop lines and 2D strips carry the real information. Text is
  not depth-sorted either, so only the active gate and the terminal gate are labelled.
- **`--policy rl` needs torch**, and `make_policy` raises `SystemExit` without `--ckpt`. The
  `none` path needs neither, which is why the smoke test is the portable one.
- *Optional follow-up, deliberately not done here:* adding `start_p` to
  `single.true_state()` is a one-line change that would remove the only **required**
  privileged reach into `env.vec`. It touches the surrogate contract, so it belongs in its
  own commit.
