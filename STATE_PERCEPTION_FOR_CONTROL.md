# The perception side, for the control side

*2026-08-02. Companion to `STATE_SURROGATE_FOR_PERCEPTION.md`, and an answer to its
section 9 items 4, 5 and 6 (plus a loose end on item 1, and one thing you did not ask for
that you need before the next training run).*

Everything below is **measured on real VQ2 recordings** through `pilot/producer.py`, the
same code that will fill `interface.Observation` in flight. Where a number is a proxy for
something we cannot measure, it says so before it says the number.

Scripts are `pilot/perception/xfer_*.py`; plots are `pilot/perception/vercheck/xfer_*.png`
and were read back before being cited; the three producer replays are
`pilot/perception/producer_runs/XFER-{gentle121520,fast153626,fast161838}/`.

---

## 0. READ THIS FIRST — one map edge is wrong, and you are training on it now

**`10-11 = 33.88 m` — the longest edge in `course_vq2.json` — is not survivable. The
evidence supports roughly 15 m.**

The finding is not from vision. Integrating the newly-fitted drag speed model
(`|v_xy| = sqrt(|a_xy| / k)`, k = 0.0425 /m) between race-packet crossings gives a flown
**path length** per leg. Path divided by the map's chord must be **≥ 1** by definition, and
it is *understated* here because the drag speed is the body-horizontal component only and
the aircraft is pitched 20–40°. Over 39 legs across three flights the median is **1.08**.
Leg 10-11 reads **0.53** and **0.60** on the two laps that fly it, and **0.60** on the
gentle lap. A flown path shorter than the straight line is geometrically impossible.
(`xfer_pathcheck.py`.)

| leg | 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-7 | 7-8 | 8-9 | 9-10 | **10-11** | 11-12 | 12-13 | 13-14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| path/chord, race lap A | 0.86 | 1.27 | 1.10 | 1.18 | 1.08 | 1.00 | 0.98 | 1.16 | 0.97 | 1.12 | — | — | — | — |
| path/chord, race lap B | 0.90 | 1.51 | 0.97 | 1.12 | 1.04 | 1.03 | 0.99 | 1.18 | 0.91 | 1.16 | **0.53** | 0.99 | 0.93 | 1.38 |
| path/chord, gentle lap | 0.86 | 1.44 | 1.05 | 1.14 | 1.06 | 1.03 | 1.11 | 1.14 | 1.15 | 4.86* | **0.60** | 1.20 | 1.31 | 1.07 |

\* a 32 s deliberate hover; the gentle lap is not a clean path-length channel.

Altitude cannot explain it — the 10-11 height step is −1.75 m over 33.88 m, and in any case
a climb makes the true 3-D path *longer*, which pushes the ratio further above 1, not below.

**How a 15 m edge came to be recorded as 34 m.** The map's own audit trail already flagged
this edge as weak, and it turns out to be weaker than the flags say (`xfer_leg1011.py`):

* All **76 rows come from one session** (`20260801-144858-vqual2-lap-pausing`) and, within
  it, from **one contiguous 171-frame window** (frames 7358–7529, ~5.7 s).
* Across that window the measurement **bearing spans 1.5°** (57.0–58.5°). This is a single
  vantage. The project's standing referee — *a static pair's separation must not change
  with viewpoint* — was never exercised on this edge. It could not have caught anything.
* The far gate sits 34 m from the near one, i.e. ~38–44 m from the camera, which is a
  **11–15 px** aperture against `mapvq2.MIN_SIZE_PX = 14.0`. The edge rests on quads at the
  acceptance floor.
* The reported height step **drifts 1.3 m within the window** (−1.8 m early, −0.5 m late)
  on what must be a static pair.
* Re-detecting those frames and looking at them (`vercheck/xfer_leg1011.png`): the drone is
  a few metres short of gate 10 with **three or four gates strung out down-course** — at
  ~8 m, ~22–25 m, ~32 m and ~41 m. The accepted rows pair the 8 m gate with the **41 m**
  one, skipping the two in between.

**Leading reconstruction: the far detection was gate 12, not gate 11.** This is the
project's known failure mode — one gate wearing another's name — and it has already been
caught three times (5-6, 7-8, 14-15). Taking the measured 33.88 m @ 205.1° as the **10→12**
vector and keeping the separately measured 11-12 (19.67 m @ 217.6°) gives

> **10-11 = 15.30 m at bearing 188.9°**, and **every other gate position is unchanged** —
> only gate 11 moves, by **19.6 m**, from (−152.8, 0.2) to about (−137.3, 12.2).

Three independent things then fall into place:

1. **Path/chord for 10-11 becomes 1.18** — squarely inside the 0.86–1.51 band every other
   leg occupies.
2. **The metres-per-station outlier disappears.** 10-11 was the standout at 57.6 m/station
   against a course median of 19.2 (p10–p90 15.2–26.7). At 15.30 m it reads **26.0** — in
   family, from a completely independent channel (Claire's sketch).
3. The alternative "the far gate was gate **13**" reconstructs 10-11 as 8.0 m and a
   path/chord of 2.24, which nothing else supports. Gate 12 is the better fit.

**What this costs you right now.** Course path length drops **268.2 → ~249.6 m**. Gate 11
is the only gate that moves, but it moves 19.6 m — larger than the map's *entire* quoted
σ budget at that gate (σ_xy 2.11 m). Every episode you have run at `vq2_frac=1.0` has flown
a 10→11 leg that is more than twice its real length, into a gate placed 20 m from where it
is. It also had a practical consequence on our side, which is how the bug surfaced: the
producer derives its current-gate range bound from the incoming edge, so an inflated 10-11
set the bound at gate 11 to ~44 m — too loose to reject a 28 m impostor in the second of
Claire's two live gate-swap frames. **A wrong map edge directly caused a tracking bug.**

**Status, honestly.** The impossibility is *proved*: 0.53 cannot be right, and altitude
cannot rescue it. The replacement value is a *reconstruction* — it inherits the 11-12
edge, itself n=28, single-session and a bridge. Until it is re-flown we would ship

```
10-11:  d = 15.3 m,  bearing 188.9 deg,  sigma_d = 3.0 m,  sigma_bearing = 8 deg
```

i.e. keep the shape, widen the uncertainty far past the map's normal 0.78 m, and **do not
use 33.88 in any episode**. The flight that settles it is one hover between gates 10 and 11
with both co-visible and unclipped from 12–18 m out — the same shot the map already knows
how to consume, and it can be combined with the 6-7 flight in §5.

---

## 1. Limits, before the table, because two of them change how you read it

1. **There is no ground truth for position.** VQ2 blocks `LOCAL_POSITION_NED`, `ODOMETRY`
   and `ATTITUDE`, so we cannot compute your `|reported − true| pos_body` row at all. §4
   gives four proxies and what each is blind to. Do not read any of them as that row.
2. **"valid" does not mean the same thing on both sides.** Our `GateObs.valid` stays True
   while a track coasts, per the interface contract — it is "we believe a gate is there",
   not "we measured it this frame". The row comparable to your *detection* rate is **"no
   fresh measurement this step"**, and that same row is what your *"unchanged since last
   step"* should be compared against — not our bitwise-unchanged row, since we
   rotation-coast and the number changes every frame even when nothing was measured.
3. **Two regimes, deliberately.** Your difficulty knob is a guess; our regimes bracket it.
   `gentle` is a slow exploratory lap with dwelling and hovering. `race` is two hand-flown
   fast laps. Read the spread between them as the uncertainty on any single number.
4. **Regime is not one axis.** The fast laps are faster *and* the pilot was chasing gates
   harder, so framing improves while rate rises. Several rows move in the direction you
   would not predict, for that reason alone.
5. All three replays are one snapshot of `producer.py` (2026-08-02, post-`normalfuse`,
   uncommitted, concurrently being edited by the drag/coast work). Absolute values drift a
   point or two against `perception/NOTES.md`; the three-way comparison is internally
   consistent because it is one snapshot.

---

## 2. Your section-2 table against ours

| | **you** 0.0 | **you** 1.0 | **us** gentle 121520 | **us** race 153626 | **us** race 161838 |
|---|---|---|---|---|---|
| frames / duration | — | — | 6303 / 210 s | 1592 / 53 s | 1900 / 63 s |
| current gate valid | 83.9% | 90.5% | **89.6%** | **95.3%** | **98.2%** |
| `normal_valid` | 48.4% | 49.7% | **66.6%** | **75.0%** | **76.3%** |
| `normal_valid` given valid | — | — | 74.3% | 78.7% | 77.7% |
| `pose_valid` | 59.7% | 61.9% | **61.0%** | **59.4%** | **65.5%** |
| `pose_valid` given valid | — | — | 68.1% | 62.3% | 66.8% |
| staleness, mean | 0.090 s | 0.118 s | **0.145 s** | **0.201 s** | **0.117 s** |
| staleness, median / p90 | — | — | 0.000 / 0.468 s | 0.000 / 0.764 s | 0.000 / 0.333 s |
| unchanged since last step | 62.7% | 66.2% | **22.0%** | **28.5%** | **27.6%** |
| false tracks (proxy, 2.4) | 2.59% | 4.52% | **1.4%** | **1.3%** | **3.7%** |
| ribbon valid | 60.3% | 53.9% | **95.8%** (fresh 82.4%) | **100%** (82.3%) | **100%** (83.8%) |
| reported-vs-true `pos_body` | 0.97 / 4.02 m | 1.28 / 5.57 m | *not measurable, see 4* | | |
| at median range | 12.4 m | 8.4 m | 7.3 m | 6.9 m | 7.6 m |

Context rows that are not in your table but change how to read it:

| | gentle 121520 | race 153626 | race 161838 |
|---|---|---|---|
| lookahead slot 1 / 2 valid | 65.5% / 49.0% | 94.7% / 75.4% | 92.5% / 81.7% |
| gyro magnitude median / p90 (rad/s) | 0.16 / 1.05 | 0.52 / 1.34 | 0.77 / 1.38 |
| frames above 1 rad/s | 10.6% | 25.2% | 36.6% |
| `attitude_conf` median | 0.88 | 0.56 | 0.46 |
| attention SEARCH / CUR / NEXT | 2.4 / 72.9 / 24.7% | 0.0 / 70.2 / 29.8% | 0.0 / 77.4 / 22.6% |
| producer latency median / p90 | 57 / 98 ms | 57 / 104 ms | 66 / 113 ms |

### 2.1 `normal_valid` — you are training the policy needlessly blind

You assume it under 50% at both ends and wrote that a materially better PnP would be a
finding. We measure **66.6–76.3%** overall, **74–79%** conditioned on a valid slot, and it
goes **up** at race pace, not down. This is not the old number: the `normalfuse.py`
per-track IPPE-twin accumulator (2026-08-02) took it from 67.8% to 75.7% on the gentle lap
*while* cutting the intra-track sign-flip rate from 4.02% to 1.08% — availability and
consistency improved together, which is why we believe it.

**The caveat matters more than the level.** The normal is resolved by the **angular
baseline** the approach generates in the track-common frame, and body rotation contributes
nothing to it — only translation relative to the gate does. A dead-straight, high-speed
final approach generates no baseline and the mechanism has no signal at all. Structural,
not a tuning limit. Measured decision rate against baseline (gentle lap, 8015 two-solution
frames):

| LOS baseline | 0–2° | 2–5° | 5–10° | 10–20° | 20–35° | 35–60° | 60°+ |
|---|---|---|---|---|---|---|---|
| n | 913 | 899 | 855 | 1367 | 1925 | 1502 | 554 |
| decided | 5.0% | 27.0% | 50.8% | 64.2% | 80.3% | 95.5% | 80.5% |
| decided, no map prior | 0.0% | 0.0% | 32.4% | 66.6% | 67.7% | 88.4% | 84.5% |

Below ~5° the only thing acting is a weak map prior. So **model `normal_valid` as a
function of approach straightness, not as a constant** — the single most useful change we
can suggest to `noise.py`. Raise the level to ~0.70 and make it degrade as the approach
gets straighter and faster. If your policy flies straighter than our pilot did, it will
fall toward your 50% in exactly the final metres where you want it.

### 2.2 `ribbon valid` — you are about 25 points pessimistic

Fresh cyan (over 0.15% of the frame) is present on **82–84%** of frames in all three
regimes; with the contract's coasting the flag reads **96–100%**. Median cyan share when
present is 0.9–1.2% of the frame. The interface docstring's "measurably absent from long
stretches of real flight" is the claim that has aged worst; on these recordings the ribbon
is close to always available.

We are **not** asking you to make it load-bearing. The contract's refusal is still right,
because the ribbon direction is an undirected line and can point backwards on a switchback
— the producer refuses those samples explicitly. But the duty cycle in `noise.py` is too
harsh, and a policy taught to ignore a cue that is nearly always present is leaving a real
pre-turn signal on the table.

### 2.3 Staleness is in family; the difficulty slope is right; live will be worse

0.117 s on the faster lap is your difficulty-1.0 number to three decimals; 0.201 s and
0.145 s bracket it. Nothing to change in the model.

**But these are replay numbers — every recorded frame is processed.** Live, the producer
costs 57–66 ms median and 98–113 ms p90 on this laptop under thermal throttle, so at 30 fps
we will drop roughly every other frame and let tracks coast. **Live staleness will exceed
this table by about one dropped-frame period.** Budget for 0.20–0.30 s and a **15–20 Hz
vision rate**, with IMU at full rate.

Related, and the one place our numbers make your model look optimistic: you model
62.7–66.2% of steps as carrying a repeated value; we measure a fresh measurement on 72–78%
of steps, i.e. only **22.0 / 28.5 / 27.6% unchanged**. Your model is about 2.4x stickier
than replay reality. Folding in the live latency above will raise ours, probably to 40–55%,
still short of yours. Worth re-deriving your camera-clock model rather than tightening the
number.

### 2.4 False tracks: what we can and cannot count

No truth, so no direct count. Two internal referees:

* **Map-range violation of the current slot** — reported range exceeds `1.3 x edge(k-1,k)`
  from the map, ratcheted by observation. This is the metric that caught the wrong-gate
  attention bug (30.45% to 2.24%). Now **1.42 / 1.32 / 3.70%**, in family with your 2.59 /
  4.52%. Note the circularity it shares with section 0: the bound is only as good as the
  map edge it is derived from.
* **Slot map-geometry violations** — the triangle inequality between two occupied slots
  against the map distance between those gates. Over all valid slots: 35.9 / 48.3 / 49.2%.
  Restricted to slots **measured this frame with `pose_valid`**: **24.1 / 17.9 / 32.5%**.
  Much worse than your false-track rate, and the honest number to worry about — but read it
  correctly: it fires on a range error as readily as on a wrong label, and it is a
  per-frame indicator over the pair, not a per-slot rate. Range-ordering inversions, the
  only test that can see a pure two-gate swap, run **11.2 / 7.4 / 10.9%**.

Together: the **current** slot is clean at the 1–4% level; the **lookahead** slots carry a
several-times-larger error, mostly in range. Your model applies one error law to all three
slots. Reality does not. If the policy uses lookahead positions metrically, put the extra
noise there.

---

## 3. Where our distribution differs in SHAPE, not just in mean

`vercheck/xfer_noise_shape.png`.

**Dropout is bursty and bimodal, not Bernoulli.** Runs of "no fresh measurement":

| | n runs | median | p90 | max | share over 0.2 s |
|---|---|---|---|---|---|
| gentle 121520 | 150 | 0.10 s | 0.95 s | 11.96 s | 41% |
| race 153626 | 37 | 0.24 s | 0.92 s | 4.83 s | 54% |
| race 161838 | 68 | 0.14 s | 0.49 s | 2.59 s | 46% |

Half the dropouts are a frame or two; 41–54% last longer than 0.2 s; the tail reaches
2.6–4.8 s at race pace, and 12 s on the gentle lap during a deliberate hover facing away.
You already model bursty dropout — the parameter to match is that a burst is **either very
short or nearly a second**, bimodal rather than geometric.

**Validity is strongly autocorrelated.** Autocorrelation of the fresh indicator at lags
1 / 5 / 10 / 30 frames: gentle +0.89 / +0.78 / +0.67 / +0.42; race +0.90 / +0.70 / +0.49 /
+0.29 and +0.83 / +0.62 / +0.42 / +0.14. With a 6-frame stack (about 0.11 s) the policy
sees an essentially **constant** validity state — a stack rarely contains both a seen and
an unseen sample. If `noise.py` draws dropout near-i.i.d. per step, the policy is training
on a much easier interleaving than it will meet, and the stack is doing work at training
time that it will not do in flight.

**Staleness is a spike at zero plus a flat tail.** 72–78% of valid frames read exactly
0.000 s; the remainder is nearly uniform out to about 0.5 s, then thin. A Gaussian or
exponential latency does not reproduce that; a two-state fresh/coasting model does.

---

## 4. The position-error row: what we honestly offer instead

Four proxies, each with its blind spot.

**(a) Crossing-range residual — the only one with a bounded truth.** When the race packet
advances from gate k, the aircraft is in gate k's plane, so the true range to that gate's
centre is at most the aperture half-width, about 0.75 m. Extrapolating the reported
current-gate range to that instant (`xfer_crossrange.py`):

| | usable crossings | median residual | MAD | p90 abs | rms |
|---|---|---|---|---|---|
| gentle 121520 | 16 | **+2.24 m** | 1.74 | 8.65 | 5.06 |
| race 153626 | 10 | **+1.08 m** | 2.09 | 7.68 | 4.14 |
| race 161838 | 15 | **+1.99 m** | 1.34 | 4.15 | 2.89 |

Second cut, no extrapolation: the minimum reported range in the 0.5 s before each crossing
is 2.24 / 0.72 / 2.05 m median, i.e. **+1.5 / −0.0 / +1.3 m** over the 0.75 m bound. Both
cuts agree on a **long bias of order 1–1.5 m at contact**, consistent in sign across all
three regimes. Blind spots: it is range along the line of sight only; it conflates our
error with genuine off-centre passage and race-packet timing; and it samples only the near
field, which is where our detector is best. Treat it as a **lower bound** on your row, at a
range 6–10 m shorter than the range your row is quoted at.

**(b) Coast-to-reacquire miss.** When a track coasts through a dropout and is re-measured,
the prediction-to-measurement gap is a self-consistency error. Median gap 0.32–0.36 s:
**3.97 / 3.41 / 5.85 m** median, p90 10.6 / 8.2 / 17.6 m; in the image, 5 / 10 / 7 px
median. The metric miss is dominated by range along the line of sight because the coast is
rotation-only. Correlation of the miss against gap x speed is +0.36 / +0.01 / +0.37 — the
missing translation term, as expected. This doubles the error (both ends are ours), so it
is an upper bound rather than an estimate.

**(c) Static-pair separation.** Two static gates must keep a constant separation from every
viewpoint. On the 6-7 strafe canary: **16.51 m, MAD 0.18, n=115** against the map's 16.42 m,
constant across viewpoint — about 0.2 m consistency at 16 m, from two clean head-on PnPs.
On the three lap replays the same measurement over whatever pair is nearest gives MAD
1.97 / 1.96 / 2.20 m. Read the strafe number as what the pipeline can do and the lap numbers
as what it does unassisted. Section 0 is what happens when this referee is not exercised.

**(d) Approach monotonicity.** Fraction of decreasing range steps in the 2 s before each
crossing: median **0.95 / 0.90 / 0.91**, min 0.72 / 0.78 / 0.78, over 17 / 11 / 15 crossings.

**If you want the real row, there is a route that needs no new flying.** The VQ1 build
streams pose and has identical physics, and we hold `20260731-204841-vq1-lap-slow` plus
13008 pose-derived auto-labels. Running the producer over a VQ1 recording against VQ1 gate
truth would give a genuine reported-versus-true distribution. It is roughly half a day (the
producer's map priors are VQ2-specific and would have to be disabled) and we have not done
it. Say the word if that row is worth it.

---

## 5. Motion blur (your section 8): the fix you propose models a mechanism that is not there

You wrote that detection probability does not depend on angular rate and that adding a gyro
term to `p_detect` would cover it. We measured it. **Do not add that term.**

Pooled over all three regimes, frames where the current gate is tracked and was measured
within the last second (`vercheck/xfer_gyro_detect.png`):

| gyro magnitude (rad/s) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.12 | 1.38 | 1.75 |
|---|---|---|---|---|---|---|---|---|
| n | 3370 | 1345 | 935 | 712 | 530 | 644 | 841 | 141 |
| P(fresh), all | .807 | .826 | .796 | .770 | .738 | .722 | .703 | .624 |
| P(fresh), **well-framed only** | **.931** | **.946** | **.915** | **.928** | **.930** | **.912** | **.919** | **.855** |
| P(fresh), range 5–15 m | .845 | .890 | .851 | .849 | .839 | .855 | .816 | .802 |

The raw curve falls about 11% per rad/s — that is the number you would fit. **It is a
framing confound.** Restrict to frames where the gate sits within 20° of boresight and at
least 8° inside the vertical span and the curve is **flat from 0 to 1.4 rad/s** (0.91–0.95,
CIs overlapping), with only the top bin dipping to 0.855 on n=69. Range-matching alone
removes most of it too.

Confirmed by looking at the frames (`vercheck/xfer_blur_look.png`): at 0.00, 0.77, 1.85 and
**11.88 rad/s** the images are equally sharp — Laplacian variance 4518 / 3868 / 4445 /
3945, and corr(rate, sharpness) = −0.06 over 212 sampled frames. **This simulator does not
render motion blur.**

So the mechanism behind the raw decline is that at high body rate the gate leaves the
frustum — which `camera.in_frustum()` already models exactly. Your section 8 item *one*
(the `k_nogate = 0.01` frame-keeping incentive) is the real omission; item *two* is not
one. If you are going to spend a retrain, spend it on the frame guard.

One caveat the other way: the residual dip at 1.5–2.0 rad/s survives the framing control
weakly (0.855, n=69). If you want a term anyway, cap it at about a 10% multiplicative loss
beyond 1.5 rad/s and nothing below.

---

## 6. The mod-90 quadrant branches (your #4): two of three pinned, and the third is the one that matters

**Mechanism.** The skylight compass is absolute only mod 90°, but `psi_unwrapped` is
continuous *within* a session, so every pair measured in one recording shares **one** branch
offset. A session whose branch was fixed against the pausing-lap reference by a shared pair
is measurement; the rest fall back to Claire's sketch. So any single session containing both
a branch-fixed pair and a contested leg pins that leg. (`xfer_quadrant.py`.)

| leg | referee | verdict |
|---|---|---|
| **15-16** | one full-bearing contour row in `20260801-121520` (branch-resolved via 0-1, 4-5, 5-6; residual 0.34°) | **PINNED.** The row reads 167.52° sketch-frame at d = 21.36 m against the accepted 21.01 m; the chosen branch gives 169.36°, so **1.84° error with the runner-up 88° away**. It also settles the mod-180 sign, which the sketch prior could not: that session's provenance line reads "err 16.5° vs next-best 16.5°", a tie, because the strafing rows are an unordered axis. |
| **2-3** | two gatenet rows in `20260801-202923` (branch-resolved via 3-4, 7-9, 13-14, 13-15; residual −1.28°) | **CORROBORATED, not comfortable.** The rows read 240.5° (d = 14.53 m, 1.2 m from accepted) and 217.4° (d = 41.53 m, a misidentified partner — discard it). Both select the chosen branch, the good one by 28.8° with the runner-up at 61°. Real margin, noisy referee. |
| **6-7** | none exists | **NOT PINNABLE from any data we hold.** No branch-resolved session contains a 6-7 row, and 6-7 is a bridge in the resolved-only graph, so no chain predicts it either. `20260801-202110` staged 6-7 but its "gate 7" partner was gate 5, excluded by sign (+5.3 m above gate 6 versus gate 5's 4.8 m below). |

**Does a branch survive a 200 s lap?** Checked, because it is load-bearing for 15-16
(`xfer_unwrapgap*.py`). The anchoring rows sit at t = 11.5–66.2 s and the 15-16 referee row
at t = 196.3 s. 94.4% of frames are compass-confident; there are 30 re-anchors after an
unconfident gap and the snap residual is **median 1.76°, p90 5.69°, max 9.76°** — the worst
case, a 4.9 s blind stretch at t = 119 s, still re-anchored 35° inside the 45° half-fold.
The branch carries.

**What a wrong branch costs**, since all three legs are bridges (`xfer_branchcost.py`):

| leg | gates that rotate | median displacement | worst |
|---|---|---|---|
| 2-3 | 14 | 132 m | 274 m (gate 16) |
| **6-7** | **10** | **103 m** | **183 m (gate 16)** |
| 15-16 | 1 | 30 m | 30 m |

The residual risk is now concentrated entirely on **6-7**, and it rotates the whole back
half of the course about gate 6.

### The flight that settles 6-7, and the 10-11 edge, in one go

**Requirement for 6-7:** ONE continuous recording (never stop the recorder) containing both
(a) at least ~20 frames with gates 6 and 7 co-visible and cleanly detected, and (b) at least
~20 frames of a pair already measured in a branch-resolved session — **5-6** (n=232 over two
laps) or **7-8** (n=100 over three) are the ones in reach. They need not be simultaneous.

Easiest execution, because 5-6-7 is nearly straight (bearings 172.5° and 172.9°, so all
three line up from the 4-to-5 approach):

1. Start recording. Fly the course normally to gate 5 and **cross it** — the
   `active_gate_index` advance is what fixes identity and stops the 202110 trap recurring.
   The approach to gate 5 supplies the 5-6 rows by itself.
2. Decelerate between gates 5 and 6 and hold a hover with **gates 6 and 7 both in frame and
   unclipped** for about 10 s. This is the same shot as `20260802-005431`, which produced
   556 rows, so it is known to work.
3. Do not stop the recorder in between. Let the ceiling into the frame occasionally: the
   compass needs a confident frame every few seconds, and the measurement above says a 5 s
   blind stretch is still 35° inside a slip.
4. Disambiguation is free — gates 5, 6 and 7 sit about 5 m apart in height (+5.00 and
   +5.31 m) and heights are gravity-referenced, so a mislabelled partner is caught by sign.

**Add, in the same recording, the 10-11 shot from section 0:** carry on to gate 10, cross
it, and hold a hover with **gates 10 and 11 co-visible and unclipped from 12–18 m out** for
about 10 s, with nothing else orange between them. That is the single most valuable
observation on this list, because it replaces a reconstruction with a measurement on the
edge you are currently training against.

**Also worth it if she is passing:** gates 1 and 2 co-visible and unclipped from 15–20 m.
That is the contested 8.32 m edge you sample both ways; it rests on n=5 from one flight and
is the thinnest number in the map.

---

## 7. Obstacles (your #6): a crude but honest column model

The hangar has two rows of numbered support columns. `stations.py` reads the signage, the
map frame is aligned to those rows, and the course README's landmark ties come from Claire's
digitised sketch (`map_approx.json`, per-gate `along_station` and `across`). That regression
is the only route to metric column geometry — **`stations.py` gives identity, never range**
— so everything here is INFERRED unless marked otherwise.

**The numbering wraps.** Facing pairs sum to 41 (16|25, 12|29, 18|23, 19|22, and 15|26 in
`gate8.png`). Numbering therefore runs 1 to 20 up one row and 21 to 40 back the other: row A
(drawn 12–20) is on **−y**, row B (drawn 21–29) on **+y**, and row-B station N sits at
along-position 41−N. That invariant is MEASURED.

Model, in the map frame of `course/README.md` (metres, z up, origin gate 0):

```python
a = (25.4, 0.64)          # metres per station, along the rows
b = (0.0, 18.6)           # row A -> row B
c = (-507.1, -20.35)      # origin offset
col_A = [c + N*a          for N in range(1, 21)]     # -y row
col_B = [c + (41-N)*a + b for N in range(21, 41)]    # +y row
# vertical cylinders of radius R spanning the whole flight envelope in z
```

| symbol | value | sigma | status |
|---|---|---|---|
| station pitch | 25.4 m | 2.0 | INFERRED — 2-D affine fit, n=17, R2 0.989, **but 7.0 m rms residual**; three corroborating channels give 26.0 / 22.8 / 26.7 |
| row bearing | +x, +1.4° | 1.0° | INFERRED (consistent with exactly +x) |
| row separation | 18.6 m | **3.0** | INFERRED, weakest number: the sketch across-axis carries about ±0.3 unit, roughly ±6 m |
| columns per row | 20 | — | **MEASURED** (the sum-to-41 invariant) |
| column radius | **1.0 m** | 0.5–2.0 | **GUESSED** — no pixel-width-at-known-range measurement exists anywhere in the repo |
| z extent | floor to ceiling | — | **GUESSED** — the ceiling is not measured; skylight bounds the light plane at 4.5–6.5 m above camera |

Two things before you use it. The fit **contradicts** the 15.97 m/station in
`map_approx.json`, and we know why: that figure used an across-units-per-station ratio of
1.13 where the metric fit says 0.73 — the sketch is drawn about 1.5x too wide relative to
its length, which biased every mixed-axis pair low. And the model is supported only **over
the span the gates cover** (x from 0 to about −220, roughly 9 stations); columns drawn
beyond that are extrapolation, not measurement.

**Corroboration.** Claire reports that from gate 5, gates 6 and 8 are visible but **7 is
blocked by a pillar**. The model reproduces exactly that asymmetry: perpendicular distance
from the sightline to the nearest modelled column is **1.20 m for 5-to-7** (2.5° off-axis,
3.5 m in front of gate 7, enough to mask about half a 2.7 m frame) against 2.67 m and 1.94 m
for 5-to-6 and 5-to-8, both of which are 35–53° off-axis, i.e. beside the camera. Monte
Carlo over the fit and the map sigmas, n=2000: P(a column within 1 m of the LOS) = **0.45
for 5-to-7** versus 0.14 and 0.17. Consistent at 2.5–3x; not proof, since the
discriminating margin is smaller than the fit residual — which is why the probabilities and
not the nominal distances are the result. `vercheck/xfer_columns.png`.

**Do obstacles matter? Almost entirely on one edge.** Nominal clearance from each straight
race edge to the nearest modelled column, and P(clearance < 1.5 m):

| edge | 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | **6-7** | 7-8 | 8-9 | 9-10 | 10-11 | 11-12 | 12-13 | 13-14 | 14-15 | 15-16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clearance (m) | 4.8 | 3.2 | 5.7 | 3.5 | 3.4 | 2.7 | **1.2** | 3.7 | 9.9 | 7.0 | 4.1 | 8.2 | 5.3 | 7.9 | 8.3 | 7.4 |
| P(under 1.5 m) | .09 | .19 | .00 | .06 | .12 | .21 | **.51** | .04 | .00 | .00 | .10 | .02 | .04 | .01 | .01 | .03 |

Whole-lap worst clearance: median 0.74 m, P(some edge under 1.5 m) = 0.80, essentially all
of it from **6-7**, with 5-6 and 1-2 secondary. Gates 8 through 16 have 5–10 m and can
ignore columns entirely. Caveat: the 10-11 row above is computed with the **old** 33.88 m
geometry and moves once section 0 is applied.

**Note the coincidence, because it is not one.** The 6-7 edge is simultaneously the one
quadrant branch we cannot pin and the only edge a column threatens. Both follow from the
same fact — gates 6 and 7 sit tight against the row-A column line and we have measured that
leg from exactly one staged strafe. The flight in section 6 fixes the branch; a couple of
frames of a numbered column co-visible with a gate in that same flight would also give the
first *measured* column position.

---

## 8. Gate 9 (your #1): it is a LEAN, and there is no additional in-plane roll

Two separate answers.

**The axis is a lean.** What we measured is the *normal's elevation* out of horizontal, and
an in-plane roll leaves the normal invariant, so the 21–24° cannot be a roll. The edge-on
screenshot shows a leaning bar against two in-frame hangar columns reading 1.0–1.2°.
`course/README.md` exports `tilt_lean_azimuth_deg = 129.7`, with the direction resolved by
RANSAC over both IPPE branches: the selected branch forms one tight axis (2.7° residual,
91.5% inliers) and the mirror branch does not form an axis at all (24.1°).

**And there is no roll on top of it — measured, not assumed** (`xfer_gate9roll*.py`,
`vercheck/xfer_gate9roll_summary.png`). The channel takes the observed quad's orientation as
the 4x-angle circular mean of its four edge directions, referenced to the image projection
of gravity at the gate's own 3-D location, and subtracts the same quantity computed for a
synthetic 1.5 m square at the resolved pose with **zero** roll. It quotients the square's
4-fold symmetry exactly rather than trying to break it, and it never runs PnP on the
observed quad, so it does not inherit PnP's degeneracy about the normal. It is blind to lean
by construction — a lean rotates the two edge pairs in opposite senses, which the 4x mean
cancels; verified at 38° foreshortening. Positive control: injecting 5 / 10 / 20° of roll
into synthetic quads at each row's own pose recovers it 1:1 on gate 9 and on every control
gate, so a null is informative.

Gate 9 reads **−0.95°, bootstrap CI [−1.32, −0.53]** (n=130 at size >= 60 px), flat across
obliquity strata. Control gates through the identical cut: gate 3 **+0.38°**, gate 8
**+0.28°**, gate 13 **−3.77°** — the known-vertical control reads four times larger than
gate 9. Over all 16 gates with n >= 10 the per-gate medians span −3.8° to +2.3° and gate 9
is unremarkable inside that population. The gates that drift at high obliquity are the
*controls*, not gate 9 — which is the signature of an estimator artefact, and it is not
where the hypothesis is.

**Verdict: NO in-plane roll. Any roll above about 5° would have moved gate 9 outside the
control envelope; above 8° it would be unmistakable.** Your rotationally-symmetric
radial-distance pass test stays adequate — **you do not need to build a square-aperture
collision test.** `--env-kwarg "vq2_tilt_deg={9:(21.0,24.0)}"` is the right and sufficient
tool.

Caveat, stated: every quad comes from one detector, whose top/bottom asymmetry already
killed `gatetilt_foreshorten.py`. That fault is a scale/shear asymmetry, which a 4x circular
mean is first-order insensitive to — but this is one channel, not two.

---

## 9. What we would most like from you, ranked

1. **Re-run with 10-11 corrected, or at least widened, before anything else** (section 0).
   Everything else on this list is a refinement; that one is an error in the geometry every
   episode is currently flying. Shipping `d = 15.3 m, bearing 188.9°, sigma_d = 3.0 m,
   sigma_bearing = 8°` is strictly better than 33.88 m at sigma 0.78, and gate 11 is the
   only gate whose position moves.
2. **Dump the surrogate's `Observation` stream and send it.** Everything in section 2 is a
   table against a table. Given `obs.npy`-shaped arrays from `noise.py` over comparable
   geometry we can compare *distributions* — gap lengths, autocorrelation, staleness shape,
   the joint of validity with range and rate — which is what actually validates the noise
   model. Your own item 5, and still the highest-value methodological item. Our side is
   already in the right format (`interface.Observation.to_vector`, 73 float32).
3. **Tell us which fields the policy leans on**, ideally as a gradient or ablation ranking
   over the 73 dims. Perception effort is fungible and we are currently guessing. Two
   specific questions: (a) is `normal_body` load-bearing, or does the policy steer at the
   gate centre? If it is load-bearing, the straight-approach baseline problem in 2.1 is a
   transfer risk, not a curiosity. (b) Do you use the lookahead slots' positions
   *metrically*, or only their bearings? Their range error is several times the current
   slot's (2.4).
4. **`speed_est_mps` is stubbed at 0.0 with `speed_conf` 0.0.** It is contract-optional and
   the drag fit now exists (k = 0.0425 /m; leave-one-session-out median 0.53 m/s, about 7%
   relative, flat from 2 to 34 m/s) — it is the same fit that found the 10-11 error. If the
   policy depends on it, say so and we ship it; if not we leave it stubbed and spend the
   time on normals. Either way remember it is the **body-horizontal in-plane** speed, not
   the total: at 20–40° of pitch the body-z component is a third to a half of the total and
   is not observable at all.
5. **Confirm the loop shape you expect.** Live, vision costs 57–66 ms median and 98–113 ms
   p90 on this laptop, so we will deliver fresh vision at about 15–20 Hz with IMU-driven
   coasting in between, not 45–65 Hz of fresh detections. If the 6-frame stack assumes fresh
   samples, that mismatch is worth knowing before the retrain rather than after.
6. **Pick the difficulty you will actually ship at, and tell us**, so the next measurement
   pass can be aimed at that regime instead of bracketing it.
7. **A frame-keeping term.** You already identified `k_nogate = 0.01` as too weak. From our
   side it is the single biggest lever on every number in section 2: `valid` swings 89.6 to
   98.2% and lookahead slot 2 swings 49 to 82% between our regimes, and the difference is
   almost entirely where the nose was pointing. The producer's AUTO_ATTENTION yaw servo
   helps for free if you leave `yaw_mode` alone — but yaw cannot fix elevation. The frame
   spans +49.4° to −9.4°, so pitching up to brake pushes gates out of the **bottom**, and
   only the policy can prevent that.

---

## Appendix — artifacts

| what | where |
|---|---|
| producer replays (obs.npy, diag.csv, strip.png, report) | `pilot/perception/producer_runs/XFER-{gentle121520,fast153626,fast161838}/` plus matching `.log` |
| **10-11 edge audit** | `xfer_pathcheck.py`, `xfer_leg1011.py`, `vercheck/xfer_leg1011.png` |
| noise-model comparison | `xfer_noise_compare.py`, `xfer_noise_compare.json`, `vercheck/xfer_noise_shape.png` |
| detection vs body rate, and the blur look | `xfer_gyrocurve.py`, `xfer_blurlook.py`, `vercheck/xfer_gyro_detect.png`, `vercheck/xfer_blur_look.png` |
| crossing-range proxy | `xfer_crossrange.py`, `xfer_crossrange.json` |
| quadrant branches | `xfer_quadrant.py`, `xfer_quadrant2.py`, `xfer_unwrapgap.py`, `xfer_unwrapgap2.py`, `xfer_branchcost.py` |
| column / obstacle model | `xfer_columns.py`, `xfer_columns2.py`, `xfer_columns3.py`, `vercheck/xfer_columns.png` |
| gate-9 in-plane roll | `xfer_gate9roll.py`, `xfer_gate9roll_ctl.py`, `xfer_gate9roll_sum.py`, `xfer_gate9roll_rows.json`, `vercheck/xfer_gate9roll_summary.png` |

All paths relative to `C:\Users\USER\Projects\vqual-2\pilot\perception\` except the first
row's `.log` files and the `producer_runs` directories, which are in the same place.
