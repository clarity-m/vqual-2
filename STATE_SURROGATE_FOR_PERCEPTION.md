# The control training environment, for the perception side

*2026-08-02. Written for someone who knows the camera and the detector but has not been in
the RL code. Everything here is measured on the current build unless labelled otherwise.*

**Nothing in this document has been flown live.** Every number is surrogate-side.

Companion docs: `STATE_VQ2_COURSE.md` (how the map is wired in and what is assumed about
it), `STATE_RL_TRAINING.md` (what the policy actually learned), `course/README.md` (the
map itself — the authority for anything about measurement).

---

## 1. There are no pixels

Nothing in training renders an image. `cv2` appears in `pilot/perception/`, `teleop.py`
and `camreferee.py`; the surrogate imports none of them.

What stands in for vision is a three-stage geometric pipeline:

1. **`surrogate/camera.py`** — the real intrinsics: 640×360, `fx = fy = 320`,
   `cx/cy = 320/180`, 20° up-tilt, vertical span **+49.4° to −9.4°** about body-forward.
2. **`surrogate/detect.py`** — transforms the *true* gate positions into the body frame and
   calls `camera.in_frustum()`. A matrix multiply and a bounds check. Geometrically
   visible ⇒ detected.
3. **`surrogate/noise.py`** — corrupts whatever survived. All of the perception realism
   lives here: bursty dropout, 30–130 ms latency, range and bearing error, PnP tilt
   degeneracy, `pose_valid` failure with a long-biased fallback range, false positives,
   ribbon duty cycling, IMU noise and bias, and a camera frame clock.

So "the policy is robust to detection noise" means robust to `noise.py`'s *parameterization*
of it, not to your detector. **Every transfer number we have is against synthetic tracks
whose parameters are guesses with reasons behind them.** That is the largest untested
assumption in the stack.

Clean detections are treated as a bug, not a feature. Zero the noise and the policy learns
to trust `pos_body` exactly, then transfers badly. We keep it dirty deliberately.

---

## 2. What the synthetic detector produces

Measured by `surrogate/selfcheck.py` on the current build. **If your real numbers differ
substantially from these, that difference is the finding.**

| | difficulty 0.0 | difficulty 1.0 |
|---|---|---|
| current gate valid | 83.9% | 90.5% |
| `normal_valid` | 48.4% | 49.7% |
| `pose_valid` | 59.7% | 61.9% |
| staleness, mean | 0.090 s | 0.118 s |
| unchanged since last step | 62.7% | 66.2% |
| false tracks | 2.59% of valid slots | 4.52% |
| ribbon valid | 60.3% | 53.9% |
| \|reported − true\| `pos_body` | median 0.97 m, p90 4.02 m | median 1.28 m, p90 5.57 m |
| at median range | 12.4 m | 8.4 m |

`normal_valid` sits under 50% at both ends — we assume the gate normal is unavailable about
half the time. If your PnP does materially better than that, we are training the policy to
be needlessly blind.

One partial cross-check exists: on `20260731-204841-vq1-lap-slow` an orange HSV blob sits
where geometry predicts on ~95% of in-frustum pairs, **including at 30–45 m**, against the
fallback's `max_range_m` of 14–30 m. So the fallback is probably pessimistic on range,
which biases transfer in the safe direction.

---

## 3. What the policy consumes

A 73-D vector from `interface.Observation.to_vector()`, stacked 6 frames — roughly 0.11 s
of history at the 45–65 Hz decision rate. Three gate detection slots plus a ribbon reaching
one gate further.

That is the entire input. One consequence worth knowing: **vertical velocity is not
directly observable** from any bounded window of that vector — sink rate has to be inferred
from position differences across the stack. We think this is a ceiling on altitude control,
and a longer stack does not fix it.

---

## 4. The course is now yours, exclusively

Until recently every course was procedurally generated, because the VQ2 map was unmeasured.
`course/course_vq2.json` is now wired in and training runs at `vq2_frac=1.0` — **every
episode is the measured course**, zero synthetic ones. The course is fixed, attempts are
unlimited and ranking is on time, so overfitting to it is the intent.

Your measurement was a correction, not a refinement:

| | generator | measured |
|---|---|---|
| gates | 18–22 | **17** |
| segment | 18–34 m | 8.32–33.88 m, median **15.3**, mean 16.8 |
| turn per gate | ≤80° | mean 30.5°, max 76.4°, 5 of 15 over 40° |
| elevation | mean-reverting | sustained **+10.95 m climb over gates 0→7** |
| path length | — | 268.2 m |

**Ten of the sixteen race edges are shorter than the shortest segment the generator could
produce** — its floor was 18 m even at maximum difficulty.

The corners that decide the course pair a sharp turn with a short exit:

| gate | turn | exit edge |
|---|---|---|
| 7 | −73.5° | 11.2 m |
| 9 | +66.6° | 10.1 m |
| 13 | −76.4° | 12.3 m |

About one second at cruise to reacquire, align and thread. The generator drew 80° turns and
it drew 18 m edges — never the two together.

This shows up directly in training: our best policy scores per-gate **0.557 on procedural
courses and ~0.32 on yours** at the same difficulty and speed cap. The real course is ~1.7×
harder than anything we had been training against. About 80% of episodes still end on a
gate frame.

The climb is good news for the camera — with the sensor tilted 20° *up*, climbing keeps
gates in frame where the generator's modelled descent hid them.

---

## 5. What we assume about the map, and how

Labelled as assumptions in the code, not quietly chosen:

| thing | why | handling |
|---|---|---|
| floor height | **measured 2026-08-02** — see below | gate 0 centre at 1.35 m |
| ceiling | never measured | placed above the course's own high point |
| gate plane yaw | three exported candidates disagree | drawn among them per episode |
| gate tilt | see §6 | gate 9 leans 21–24°, all others vertical |

### Floor — measured, and it moves the whole course down 2.25 m

Perception confirmed 2026-08-02 that **the bottom of gate 0's frame touches the ground**.
With the spec-exact 2700 mm outer frame that puts gate 0's centre at **1.35 m**, and gate 0
is the lowest gate in the map — every other gate is at positive `z` relative to it.

The surrogate had been randomizing this over **2.6–4.5 m**, and that range was never a
measurement: `env.py:173` records its derivation as *"floor_clear_m plus half the aperture
or gates start underground"* — i.e. it was the smallest value the **spawn clamp** would
tolerate. Measured, the whole course had been floating **2.25 m** off the ground.

The correction is coupled, because the spawn clamp is what forced the old value:

```
--env-kwarg floor_clear_m=0.5 --env-kwarg "vq2_floor_clear_m=(1.30,1.45)"
```

`floor_clear_m` (default 1.8) sets the spawn altitude clamp at `+0.6`, so 2.40 m. Fixing
only `vq2_floor_clear_m` leaves the drone spawning at 2.40 m — **above the top of gate 0's
aperture at 2.18 m** — every single episode. Both have to move.

Why this matters for transfer: with the floor at its true height, gates **0, 3, 4 and 5**
sit at 1.35 / 2.54 / 1.75 / 1.99 m, all inside the 2.5 m band where the clearance term is
active and a floor strike is a live risk. Under the old setting they sat at 3.6–5.5 m,
comfortably outside it. **The policy has never flown near the floor on this course**, and
floor strikes were the original failure mode — 55.5% of episodes in the first run.

One residual: the anchor is the *lowest sampled* gate, and gate 4 (nominal +0.40 m)
occasionally dips below gate 0 once the per-edge `dz` draws accumulate, so gate 0's centre
averages ~1.43 m rather than exactly 1.35. An 8 cm conservatism, left alone.

**Yaw.** The measured / grid-aligned / race-bisector azimuths disagree, so we turned the
disagreement into domain randomization rather than picking a side. Gates 8, 12 and 13 have
no trusted yaw and fall back to the bisector — the map's own recommendation.

We deliberately do **not** use the package's `sample(randomize_yaw=True)`: it assigns a
uniform 0–180° plane to every refused-yaw gate, which would face 8/12/13 in a random
direction every episode and make them unlearnable. We take positions from `sample()` and
handle yaw ourselves.

**The contested 1–2 edge** is sampled both ways — accepted 8.32 m and the refused channel's
13.0 m, the latter in ~25% of pool courses. Under VQ2-only every episode flies 1→2, so
`course/README.md`'s "enable it if the policy will ever fly that leg" applies
unconditionally.

**The risk sampling cannot cover** is the one `course/README.md` flags itself: edges from
one flight share that flight's compass bias, and the 2-3, 6-7 and 15-16 legs take their
mod-90 quadrant from a sketch rather than a shared measurement. A wrong branch rotates a
whole leg by 90° — a discrete failure a Gaussian cannot express. A policy that has memorized
this course has memorized something that may be wrong in a way no sampling reaches. Known
and accepted; the sampled envelope is a lower bound.

---

## 6. Gate 9 tilt — confirmed, and what is still open

**Confirmed by perception, 2026-08-02:** gate 9 measures **21–24°**, every other gate is
vertical. This overturns the shipped JSON's uniform 0–20° prior over gates 8 *and* 9, which
could not sample 21–24° at all.

The knob exists and is verified working — no code change needed:

```
--env-kwarg "vq2_tilt_deg={9:(21.0,24.0)}"
```

Measured over 256 sampled courses: gate 9 normal leans 21.02–23.99° out of horizontal,
gate 8 exactly 0.00°, everything else vertical.

### Still open: is it a lean or an in-plane roll?

The confirmation settles the **magnitude**, not the **axis**, and that distinction decides
whether a knob is enough:

* **A lean** — the plane's normal tips out of horizontal. This is what `vq2course._normals`
  models (`n = [cos t·cos az, cos t·sin az, sin t]`) and what the knob above applies. Fully
  representable today.
* **An in-plane roll** — the square frame rotated about its own normal. **The surrogate
  cannot represent this at all.** `env._gate_geometry` reduces a crossing to a radial
  distance and compares it against half-widths, which is rotationally symmetric about the
  normal, so rolling a square aperture is invisible to it. Fixing that means building a
  square-aperture collision test, not setting a config value.

If the answer is "lean", we are done. If it is "roll", the knob is the wrong tool and the
21–24° is currently unmodelled regardless of what we set.

---

## 7. Where the collision model departs from the geometry

**From the spec exactly** (VADR-TS-003 issue 00.03): chassis 280×280×160 mm → 0.214 m
bounding-sphere half-diagonal (**§3.6**); gate outer 2700 mm, inner 1500 mm, depth 260 mm
(**§3.7**).

**Note the spec defines no collision semantics at all.** §3.2 says only "a rigid-body drone
flight model including thrust generation, aerodynamic drag, gravity, and collision physics."
What counts as a hit, whether one ends a run, floor and ceiling contact, safety margins and
the corridor are all ours.

Two deviations, running in opposite directions:

**The pass test is a circle; the gate is a square.** We reduce a crossing to a radial
distance and compare against half-widths 0.75 / 1.35 m — the inscribed circle. The real
1500 mm aperture admits the drone anywhere within a square of half-width
`0.75 − 0.214 = 0.536 m`, whose corners sit at radius 0.758 m. **We credit roughly a third
to a half of the genuinely passable aperture** and score the rest as crashes. Conservative,
but it trains much tighter centring than the gate demands.

**The outer test is slightly lenient.** Beyond 1.35 m radius is called a clean miss, but
real frame material reaches `1.35 × √2 ≈ 1.91 m` at the diagonals, so a diagonal strike in
roughly the 1.66–1.91 m band is a real collision scored as a miss. Thin sliver, but it is
the fail-open direction.

**Other gaps:** gate depth (260 mm) is dropped — the gate is a zero-thickness plane with
the sphere margin absorbing it. Backwards crossings are undetected. Floor and ceiling are
invented. And **obstacles are absent entirely** — spec §3.1 lists vertical and horizontal
obstacles, boundary elements and terrain structures; the surrogate models none and the map
excludes them. Largest remaining fidelity gap.

---

## 8. Two things on our side that likely make your job harder

**The frame-keeping incentive is effectively zero.** The penalty for having no gate in the
frustum is `k_nogate = 0.01` per step, against a progress signal of about 0.09. The network
has had almost no reason to keep gates framed. Given the frame spans +49.4° to −9.4° — so
the narrow *lower* edge binds, and pitching up to brake pushes gates out of the bottom —
that is a real omission. `policies/baseline.py` has an explicit frame guard; the RL policy
was left to infer one from a signal too weak to teach it.

**Motion blur is not modelled.** Detection probability depends on range and obliquity but
not on angular rate. The policy has never been penalised for the manoeuvre most likely to
blind a real detector. Adding a gyro-magnitude term to `p_detect` in `detect.py` would
cover it and is a small change.

Both are fixable, both need a retrain, neither has happened.

---

## 9. What would help most, ranked

1. **Gate 9 — lean or in-plane roll?** See §6. Magnitude is settled; the axis is not, and
   it decides between a config knob and a new collision test.
2. **The corrected `course_vq2.json`.** Lands as the `--env-kwarg` in §6 with no code
   change, so it is cheap to adopt the moment it arrives.
3. ~~One measurement of any gate's height above the hangar floor.~~ **Answered
   2026-08-02** — gate 0's frame bottom touches the ground, so its centre is at 1.35 m.
   See §5. This was the highest-value single number on the list and it changed the course
   geometry more than anything else we have been given.
4. **The mod-90 quadrant branches** on legs 2-3, 6-7 and 15-16. If any can be pinned by a
   shared measurement rather than the sketch, that is worth more than tightening any sigma —
   it is the one error mode sampling cannot express.
5. **Stream comparison, with a caveat.** A *surrogate* observation dump cannot show
   association flips, PnP normal flips or range-source switching, because `detect.py` is a
   noise model and not a detector. The useful version is dumping surrogate and real
   `Producer` streams over comparable geometry and comparing distributions — that validates
   the noise model, which is the assumption everything else rests on.
6. **Obstacles.** Spec §3.1 lists them, the map excludes them, the surrogate has none, and
   `NOTES.md` mentions support columns in the hangar.
