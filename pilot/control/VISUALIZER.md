# VISUALIZER — 3D rollout viewer for the surrogate

**Status:** design, not yet built. **Written:** 2026-08-02.
**Scope:** two new files under `pilot/control/evalsuite/`. Purely additive — no changes to
`env.py`, `single.py`, `course.py`, `camera.py`, or `run_policy.py`.

---

## 1. Why

PPO run `run1` logged ~5.05M steps at **0% course completion** (`CONTROL_STATE_AGENTS.md` §5), and the
reactive baseline — the submission floor — stalls around 2 gates of 18–22. Both state docs put "explain
the zero completions" at the top of the work queue, with reward horizon, course geometry,
collision/gate-pass tolerance, and the unmeasured detection noise all still live suspects.

Today the only way to inspect an episode is fixed-width text: `_probe_baseline.py` prints a cause and a
miss vector per seed. That tells you *that* the drone hit a gate frame 0.9 m off-centre; it cannot show
the approach that put it there, whether the gate was ever in camera frame, or how close the path ran to
the corridor wall.

This is the repo's first renderer: fly a few seeds, emit **one self-contained HTML file** with an
orbitable 3D view of the generated course and the flown path, plus the onboard camera view. Goal: make
"where and why does the episode end" answerable in one look.

## 2. Settled decisions

| Decision | Choice |
|---|---|
| Renderer | Self-contained interactive HTML, **zero new dependencies**. Hand-rolled canvas 3D: drag-orbit, scroll-zoom, episode picker. No matplotlib/plotly/three.js/CDN — must open offline. |
| Environment | **Local only.** No Colab/notebook inline path. |
| Path source | User-selectable `--policy baseline\|rl\|none`; `none` renders bare courses. |
| Per-episode content | Termination marker + cause; corridor + floor/ceiling; gates as correctly-sized annuli with runout gates distinct; onboard camera view. |

The zero-dependency constraint is not incidental: the repo has **no** plotting library and no
`requirements.txt` / `pyproject.toml` anywhere, and `user-vm-cmds.md:87-90` warns that unpinned pip
installs shadow conda's numpy 1.26.4 and break exactly this class of library.

## 3. Three corrections to earlier assumptions

1. **Runout gates are non-physical.** `env._gate_geometry` gates collisions on `exists = idx < n_gates`
   (`env.py:584,591`) and `detect.tick` gates visibility on the same expression. The 3 runout gates
   cannot be hit and cannot be seen — they only shape the corridor polyline and the bisector normals.
   The legend must say *non-physical*, or the picture lies about where a crash could occur.
2. **The two existing tools disagree on termination priority.** `run_policy.REASON_FLAGS`
   (`run_policy.py:42-45`) is `finished → collision → corridor → timeout`; `_probe_baseline.py:62-64` is
   `finished → corridor → timeout → collision`. Same episode, two labels. Follow the probe (so the
   cross-check in §8 works), but **ship all four raw booleans and display them** so the label is never
   load-bearing.
3. **There is no gate-axis bug.** `_probe_baseline.py:57-59` uses `side = cross(up, n)` and `env.py:370`
   uses the exact negation, but env multiplies it by a zero-mean Gaussian, so the sign is unobservable.
   Pick one, comment it, move on.

Two related traps:

- `info["collision"]` merges gate-frame, floor and ceiling strikes into one flag (`env.py:443-457`), so
  floor/ceiling must be re-derived from the captured position.
- `_probe_baseline.py:68` tests the ceiling at `alt >= ceil - 0.05`, but the env terminates at
  `alt > ceil - sphere_r` (`env.py:456`). Use the env's own thresholds or near-ceiling strikes mislabel
  as `collision?`.

## 4. File layout

Two new files, flat siblings in `evalsuite/` (plain names — this is meant to stay, not a `_probe_`):

```
pilot/control/evalsuite/viz3d.py              (~470 lines Python)
pilot/control/evalsuite/viz3d_template.html   (~900 lines HTML+CSS+JS)
```

**Why two files.** The repo uses `%`-formatting throughout; a single `%` in embedded CSS (`width: 100%`)
detonates a `"..." % (...)` expression, forcing `%%`-doubling across 900 lines of foreign syntax. A
standalone template also keeps the JS lintable and highlightable, and lets you iterate with F5 without
re-running a rollout. The **output** is still one self-contained file, which is the actual requirement.

**Emit mechanism.** The template contains exactly one marker line:

```html
<script id="payload" type="application/json">{"episodes":[]}</script>
```

Python locates it by `id="payload"` and substitutes `json.dumps(blob, separators=(",",":"))` with
`.replace("</", "<\\/")` applied — `\/` is legal JSON, so the payload can never terminate the `<script>`
early. Fail loudly if the marker or template is missing. The template ships with an empty payload and
renders "no episodes — run viz3d.py", so it is itself valid and openable.

**Default output:** `pilot/evidence/%Y-%m-%d-rollout3d-<policy>[-<ckpt>].html`, matching the
dated-artifact convention in `.gitignore:6-9` and the existing `2026-07-30-angle-sweep.json`. Note in the
docstring that `pilot/evidence/` is git-tracked, so each run dirties `git status` — intended, but say it.

## 5. Python side (`viz3d.py`)

Standard 4-level `sys.path` bootstrap (`run_policy.py:31-34`), absolute `pilot.control.` imports each
tagged `# noqa: E402`. Import `camera` and `course` directly (numpy-only); reach `env`/`single` only
through `run_policy`'s helpers, since `env.py` does bare `import interface`.

**Reuse wholesale, do not reimplement:** `_Env` (`:62`), `build_config` (`:123`), `make_policy` (`:287`),
`add_common_args` (`:307`), `add_policy_args` (`:320`), `parse_seeds` (`:265`), `parse_pair` (`:280`),
`config_dict`, `write_json` (`:329`).

### `capture_episode(policy, config, seed, max_steps, detail_stride, tail_steps) -> dict`

The one function that matters. **Capture before stepping, always** — the env auto-resets inside `step()`
(`env.py:530-532`), so the post-step `true_state()` is the next episode's spawn:

```python
for i in range(1, max_steps + 1):
    prev   = env.true_state()                     # p, v, q, active_gate, t_s, sphere_r, ...
    fields = getattr(env.vec, "_fields", None)    # matches THIS obs; holds gate_index
    a      = policy(obs)
    _push_path(prev, a)                           # every step
    if i % detail_stride == 0:
        _push_detail(prev, obs, fields, env.vec, a, policy)
    dt = float(obs.dt_s)                          # true decision period, env.py:738
    obs, info = env.step(a)
    if info["done"]:
        term = classify_terminal(info, prev, dt, world, config)
        break
```

Post-loop, force-include the last `tail_steps` detail frames and every step where `active` incremented.

`_capture_world(env)` reads `env.true_state()` once (`single.py:78-85`) plus two privileged extras it does
not expose — `env.vec.start_p[0]` (**required** for corridor segment 0) and `env.vec.n_slots_course`. Wrap
every `env.vec.*` reach in `getattr` with graceful degrade, so a sibling refactor loses a panel rather
than crashing. Per-gate axes are computed **in Python once** so the JS never picks a sign:

```python
UP0  = np.array([0.0, 0.0, -1.0])       # NED: -z is up
side = _unit(np.cross(UP0, n))          # matches _probe_baseline.py:57-59
up   = np.cross(n, side)                # right-handed with n, already unit
```

**`--policy none` skips the rollout entirely** after one `true_state()` capture. Do not hover — a hovering
drone hits the floor in ~2 s and stamps a misleading FLOOR marker on every course.

### JSON schema

Struct-of-arrays (≈40% smaller as text than array-of-objects). Round on emit: 2 dp metres, 4 dp unit
vectors/quats, 3 dp times. Units stated once in the module docstring: **metres, seconds, radians; world is
NED with z DOWN, altitude is `-z`, `z_ceil` is negative; body frame is FRD.**

```jsonc
{ "meta": { "generated", "policy", "ckpt", "difficulty", "speed_cap", "normal_starts",
            "detail_stride", "gate_inner_m": 1.5, "gate_outer_m": 2.7,
            "cam": { "fx":320,"fy":320,"cx":320,"cy":180,"w":640,"h":360,
                     "fwd_v":296.0,        // camera.py:9 — the 20-deg-tilt trap, shipped explicitly
                     "R_CB": [[..]] } },   // so the JS port can never drift
  "episodes": [{
    "seed", "cause", "flags": {finished,collision,corridor,timeout},   // raw, per correction 2
    "gates", "start_gate", "n_gates", "steps", "t_end", "ret",
    "world": { "start":[3], "z_ceil", "corridor_r", "sphere_r",
               "raced": 20,                    // g[0:raced] physical; [raced:25] runout decoration
               "g": { cx,cy,cz, nx,ny,nz, sx,sy,sz, ux,uy,uz } },   // 25 each
    "path":   { t,x,y,z,spd,a,u0,u1,u2 },      // FULL RATE — the shape is the point
    "cross":  [ {k,i,t,lat,side,up} ],         // gate-plane crossings
    "detail": { i, qw,qx,qy,qz, tg, cue, fov,  // STRIDED (~20 Hz)
                det: [ {x,y,z,c,s,px,gi,f}, {...}, {...} ] },   // 3 slots, body frame
    "term":   { kind, p[3], v[3], alt, floor, ceil, gate, lat, miss_side, miss_up, t }
  }]
}
```

`detail.fov` is `camera.in_frustum` computed **in Python** — it pins the JS port (§8.4). `det[j].gi`
(which real gate, `-1` if none) and bit 8 of `det[j].f` (`env.vec.det.is_fp`) are privileged and are what
let the camera panel colour a phantom red. Detections themselves come from the `interface.Observation` the
policy was handed — that's the contract, and exactly the policy's input.

Sizes: ~12 KB path + ~25 KB detail per 400-step episode; 6 episodes ≈ 250 KB. Guarded by
`--max-episodes` and `--no-detections`.

### `classify_terminal(info, prev, dt, world, cfg) -> dict`

Extends `_probe_baseline.py:43-71` into a named function:

```
end = prev["p"] + prev["v"]*dt                      # _probe_baseline.py:33-34
alt = -end[2]; floor = prev["sphere_r"]; ceil = -prev["z_ceil"]

1. finished       -> FINISHED
2. corridor_exit  -> CORRIDOR
3. timeout        -> TIMEOUT  (sub: t_gate > cfg.gate_timeout_s ? "gate stall" : "wall clock")
4. else collision:
     alt <= floor  -> FLOOR                          # env.py:455 thresholds, not the probe's 0.05
     alt >= ceil   -> CEILING                        # env.py:456
     else scan k in [active-1, active+3) & [0, n_gates):
         d0=(prev.p-g[k])·n[k]; d1=(end-g[k])·n[k]
         if d0 < 0 <= d1:
             x = lerp(prev.p,end, -d0/(d1-d0)) - g[k];  x -= (x·n[k])*n[k];  lat = |x|
             lat + r >  0.75 and lat - r < 1.35  -> GATE FRAME   # env.py:603
             lat + r <= 0.75                     -> clean pass, keep scanning
             else                                -> flew wide of the structure
     nothing explains it -> FRAME?                   # never assert
5. loop exhausted -> MAX_STEPS
```

Two extensions over the probe, because `p + v·dt` is first-order and off by ~0.2 m at 10 m/s — enough to
place the marker on the wrong side of a 0.6 m ring:

- **GATE FRAME** marker sits at the interpolated plane-crossing point, not at `end`.
- **FLOOR/CEILING** marker solves linearly for the `t` where `-z` crosses the threshold.

### Console output

Keep the house fixed-width table so the tool degrades gracefully headless — same columns as
`_probe_baseline.py:88-97` (`seed why gate need t_s alt ceil miss cue`).

## 6. HTML/JS side (`viz3d_template.html`)

**Layout:** 3D scene canvas (main) with a 320×180 camera panel pinned bottom-right; two 2D strip canvases
below; episode picker + stats + toggles in a right rail; timeline slider along the bottom. Handle
`devicePixelRatio` once via `canvas.width = cssW*dpr; ctx.setTransform(dpr,0,0,dpr,0,0)`.

**Coordinate conversion — the only place NED is touched.** At episode load:

```js
function toDisp(x, y, z) { return [y, x, -z * KZ]; }   // NED -> [east, north, up*exag]
```

Downstream code never sees NED. Generate gate circle points **in NED first**, then convert per-vertex, so
exaggeration produces the correct ellipse rather than a scaled radius. `KZ` must **never** touch the
camera panel — that is a real optical projection.

**Orbit camera:** target `T`, yaw `θ`, pitch `φ∈[-89°,89°]` (clamping is what keeps
`cross(fwd, up_world)` non-degenerate), distance `d`. Three dot products per vertex, no 4×4 matrices:

```js
fwd = [cosφcosθ, cosφsinθ, sinφ];  eye = T - d*fwd;
right = norm(cross(fwd,[0,0,1]));  up = cross(right,fwd);
c  = [dot(P-eye,right), dot(P-eye,up), dot(P-eye,fwd)];
f  = 0.5*H/tan(0.5*fovY);  sx = W/2 + f*c[0]/c[2];  sy = H/2 - f*c[1]/c[2];
```

**Near-plane clipping is mandatory** (`zn = 0.05`) — the most common hand-rolled-3D failure. Needs
`clipSeg(a,b,zn)` for polylines and `clipPoly(verts,zn)` (Sutherland–Hodgman against one plane, ~15 lines)
for filled polygons.

**Depth order:** painter's algorithm over a flat drawable list keyed on view-space centroid `z`, sorted
far-first. Floor grid is pulled out and drawn first (the drone can never be below it — a floor strike
terminates at `sphere_r`); ceiling grid drawn last at alpha 0.10. Path batched ~8 segments per item.
Screen-space text drawn last, unsorted.

### Primitives

- **Gates as annuli** — two coplanar circles `r=0.75` and `r=1.35`, 48 samples,
  `P(θ) = c + r(cosθ·s + sinθ·u)` using the Python-supplied axes. Render both loops in **one path**, wound
  oppositely, then `ctx.fill('evenodd')` — that gives the hole for free. Raced: filled + solid outline.
  Runout (`k >= raced`): dashed grey outline only, legend **"non-physical: cannot be hit or seen"**.
  Active gate accented; passed gates dimmed. A 2 m quill along `n` makes travel direction unambiguous.
- **Corridor** — *not* a global tube. A static wireframe hull (one 24-segment ring of radius `corridor_r`
  per polyline node, normal to that node's bisector, plus 4 longitudinal rails) at alpha 0.18, **plus the
  live pair** — segments `prev(active)→g[active]` and `g[active]→g[active+1]`, with `prev = start_p` iff
  `active <= 0`, exactly `env._corridor_exit` (`env.py:607-615`) — drawn bright and following the
  timeline. Legend: only the bright pair can terminate the episode. Filled capsules are rejected; without
  a z-buffer they hide the gates.
- **Path** — coloured by a dropdown scalar, **default altitude margin** `min(alt-sphere_r, ceil-alt)`,
  since FLOOR/CEILING is the leading hypothesis. Plus a ground shadow at `z_up=0` (alpha 0.25) and drop
  lines every ~25 samples; those carry more depth information than any shading model here.
- **Termination marker** — ±1 m 3-axis cross, filled disc, leader line to a screen-space cause label,
  drawn last, colour-coded. For GATE FRAME, emphasize that gate and draw the in-plane miss vector from
  centre to crossing point annotated `side +0.91 / up -0.45 / |1.02|`. That is the money shot.
- **Drone** — body-axes triad from `q` plus an 8 m **camera frustum wireframe**, so "was the gate even in
  view" is answerable from the 3D view alone.

### Camera panel

320×180 CSS canvas drawn at 640×360 logical (`ctx.scale(0.5,0.5)`). Image border, principal point at
`(320,180)`, and the **body-forward marker at `(320,296)`** explicitly labelled — the 20° up-tilt is the
single most misread number in `camera.py`. Elevation ticks at 0°, ±10°, ±20°.

> **Show truth and detections overlaid — truth as outline, detections filled.** `noise.py` is an
> unmeasured P3 fallback, so "locked onto a phantom" and "detection is range-biased" are live hypotheses
> that only an overlay can kill. Truth: `pos_b = Rᵀ(c - p)` via a `quatToRot` ported from `vmath.py:15-29`,
> then `project` via `meta.cam.R_CB`; sample each raced gate's inner circle at 24 points, clip to
> `z_cam > 0.05`. If a gate is in front but off-frame, draw an edge arrow with the off-frame angle — that
> reads directly as "gate is 15° below the lower edge", the `-9.4°` binding constraint. Detections: filled
> disc of radius `px/2`, alpha = confidence, dashed when stale, **red when the `is_fp` bit is set**.

### Interaction

Pointer capture + left-drag orbit; shift/right-drag pan; `wheel` with `{passive:false}` +
`preventDefault` to zoom (clamp `d ∈ [2,4000]`); keys `1`/`2`/`3` for top-down / side-on /
terminal-parked views, `F` refit, `←/→` episode, `,`/`.` timeline step, `space` play. Redraw on
interaction; `requestAnimationFrame` only while playing.

## 7. Anisotropy — 500 m of course in a 12 m band

Three layers, all on by default:

1. **Auto-fit on select** — AABB over gates ∪ path expanded by `corridor_r`.
2. **Altitude exaggeration `KZ`**, slider 1×–20×, **default 6×**, applied at the single `toDisp` so every
   element stretches consistently. Persistent `×6 vertical` watermark so nobody misreads a shallow descent
   as a plunge; `KZ=1` is one keystroke away.
3. **Two unrolled 2D strips**, and honestly this is where the diagnosis happens:
   - **altitude vs arclength** with floor band, ceiling band and gate altitude ticks — a FLOOR/CEILING
     death is a single glance;
   - **lateral offset vs arclength** with `±corridor_r` bounds — a CORRIDOR death is a single glance.

   Needs an `arclengthOf(p)` in JS (~20 lines: project onto each segment, nearest wins, add prefix). Note
   `course.arclength_point` (`course.py:124`) is the *forward* map and is dead code; the inverse is what's
   wanted, so leave it dead. Strips are aspect-free and **overlay cleanly across seeds** — that's where
   "everything dies at 60 m" becomes visible. No 3D overlay of multiple courses; different seeds are
   different worlds, so it would be meaningless.

## 8. CLI

```
python pilot/control/evalsuite/viz3d.py --policy baseline --seeds 0-5
python pilot/control/evalsuite/viz3d.py --policy rl --ckpt run1_s5046272 --seeds 0-5 --open
python pilot/control/evalsuite/viz3d.py --policy none --seeds 0-3
```

Reused via `add_common_args`: `--seeds --difficulty --speed-cap --decision-hz --n-gates --time-penalty
--max-steps --json --quiet` (all apply). Reused via `add_policy_args`: `--ckpt --gains --supervisor
--device`. Its `--policy` hard-codes `choices=("baseline","rl")` (`run_policy.py:321`), so widen it with
the public escape hatch rather than poking argparse internals:

```python
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], conflict_handler="resolve")
add_policy_args(ap)
ap.add_argument("--policy", default="baseline", choices=("baseline", "rl", "none"))
add_common_args(ap)
```

New flags:

| flag | default | why |
|---|---|---|
| `--out PATH` | `pilot/evidence/<date>-rollout3d-<policy>.html` | the artifact |
| `--open` | off | `webbrowser.open()` — stdlib, zero deps |
| `--detail-stride N` | 3 | ~20 Hz detail; path stays full rate |
| `--tail-steps N` | 40 | full-detail window before termination |
| `--normal-starts` / `--random-starts` | **normal ON** | `cfg.replace(start_probs=(1,0,0,0,0))`. The default `(0.50,0.25,0.12,0.08,0.05)` means 25% mid-course, 12% hover, 8% corridor-offset starts — training devices that make the pictures incomprehensible. `--random-starts` restores them and is required for the cross-check in §9.2. |
| `--max-episodes N` | 24 | guard against `--seeds 0-999`; renders the first N and prints a loud note naming what was skipped (default keeps `add_common_args`' `0-19` working untouched) |
| `--no-detections` | off | drops `detail.det` for a small file |
| `--only-cause C` | none | filter to one ending kind; pairs with a wide `--seeds` |
| `--no-timestamp` | off | makes two runs byte-identical for diffing |
| `--full-config` | off | embeds `config_dict(cfg)` |

Standard shape: `def main(argv=None): ... return 0` / `if __name__ == "__main__": sys.exit(main())`.
Module docstring in house style: one-line summary, indented invocations, then paragraphs of *why*.

## 9. Verification

1. **Smoke — no checkpoint, no torch, no policy:** `--policy none --seeds 0-2`. Exercises course
   generation, axes, corridor, floor/ceiling, JSON emit, and every HTML path except the trajectory. <2 s.
   This must work on any machine.
2. **Classifier regression against the existing probe:**
   `--policy baseline --seeds 0-5 --difficulty 0.2 --random-starts` vs
   `python pilot/control/evalsuite/_probe_baseline.py "" 6`. Both use `build_config(0.2, 1.0)` and
   `SingleSurrogate(cfg, seed)`, so the `why` column must match **exactly, seed by seed** — a free
   regression test on `classify_terminal`. (Needs `--random-starts`; the probe uses default
   `start_probs`.)
3. **Geometry asserts inside the tool** (always on, cheap): per gate `|n|=|s|=|u|=1` and mutual
   orthogonality to 1e-9; raced-gate altitude inside `[floor_clear_m+0.75, ceil-ceil_clear_m-0.75]`
   (`course.py:60-61`); and **corridor consistency** — recompute `course.point_segment_distance` for every
   path sample against the captured `start_p`/`g_pos`/`active` and assert `d <= corridor_r` for all but
   possibly the last. If that fires on a non-CORRIDOR episode, the *visualizer's* corridor model is wrong,
   which is exactly the bug that would otherwise ship silently.
4. **Camera-panel cross-check:** `detail.fov` comes from Python's `camera.in_frustum`; the JS panel must
   draw the active gate inside the frame **iff** `fov==1`. This is the only Python↔JS duplication in the
   design and this flag pins it. Add a dev-only mismatch counter.
5. **Offline proof:** open via `file://` with networking off;
   `grep -c 'https\?://\|src=\|@import\|integrity=' out.html` → 0. Check in two browsers.
6. **Determinism:** two runs with `--no-timestamp` must produce byte-identical `--json` blobs.
7. **The actual diagnostic run** — the point of all this: `--policy rl --ckpt run1_s5046272 --seeds 0-11`
   beside `--policy baseline --seeds 0-11`. Do RL episodes end FLOOR (thrust collapse), GATE FRAME (aim),
   or TIMEOUT (no forward drive)? Is the gate *ever* in frame? Does `cue` sit on `blind`?

## 10. Risks

- **Priority disagreement (§3.2)** — `run_policy --json` and this tool can label the same episode
  differently. Mitigated by shipping raw flags; the label is presentational.
- **First-order terminal extrapolation** — mitigated by solving the exact plane/floor/ceiling crossing.
  Residual: collisions are tested per *substep* (`env.py:439-445`), so a plane crossed and re-crossed
  inside one decision step aliases away. Cannot happen at 10 m/s and 1/165 s substeps; the classifier
  returns `FRAME?` rather than asserting.
- **Privileged reads not on `true_state()`** — `start_p`, `t_gate`, `det.is_fp`, `_fields["gate_index"]`
  all reach into `env.vec`. Every one wrapped in `getattr` with graceful degrade. *Optional follow-up, out
  of scope here:* adding `start_p` to `single.true_state()` is a one-line change that removes the only
  **required** internal reach — propose separately rather than editing the surrogate contract in this PR.
- **Mid-course/hover starts** — `--normal-starts` default-on. Note mode 3 offsets the spawn laterally by
  `σ = 0.45·corridor_r` (`env.py:372`), so an episode can legitimately spawn *outside* its corridor and
  die on step 1. Surface that rather than hide it.
- **`--policy rl` needs torch**; `make_policy` raises `SystemExit` without `--ckpt`. The `none` path needs
  neither, which is why smoke test 1 is the portable one.
- **Painter's algorithm has no correct answer** for a translucent ring intersecting the path. Accepted —
  the shadow, drop lines and 2D strips carry the real information. Text is not depth-sorted either, so
  label only `active ± 2` and the terminal gate to avoid smearing.
