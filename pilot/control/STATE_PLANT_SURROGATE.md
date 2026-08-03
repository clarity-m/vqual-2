# Plant fit and surrogate — verified state, 2026-08-01

Scope: **P1 (plant) and P2 (surrogate) only.** For the whole pipeline see
`CONTROL_STATE_HUMANS.md` (prose) and `CONTROL_STATE_AGENTS.md` (contracts).
T4's `run1` and the reactive baseline are deliberately out of scope here.

**What makes this file different from the others:** every number below is either read
straight out of `plant.json` / a log file on disk, or is output produced by *running*
the self-check that claims it. The other state documents were assembled by reading
source. Where they and this file disagree, re-run the two commands in §4 and believe
the output.

Verified this session (re-run 2026-08-01 evening):

| command | result |
|---|---|
| `python pilot/control/plant.py` | **PASS** (terminal climb 30.99 == 30.99; hover drift +0.182 m/s) |
| `python pilot/control/surrogate/selfcheck.py` | **17/17 PASS** (9 stages: 8 asserting, #9 report-only) |

---

## 1. Plant — done, refit on card 2, verified

`plant.json` is the authority. Current contents:

| | value |
|---|---|
| body drag `kx` / `ky` / `kz` | 0.04867 / 0.04963 / **0.04359** |
| `kz_source` | `sysid_apex arcs`, spread 1.6e-4; terminal-descent cross-check 0.04347 |
| thrust | **21 measured knots**: idle 0.35 m/s² at throttle ~0, hover **0.270**, full **51.67** m/s² |
| rate-loop gain | 0.970 / 0.963 / 0.904 |
| delays | rate 10 ms, thrust 15 ms; `rate_tau_max` 0.01 s |
| samples | 23 261 |
| fit R² (x/y/z) | 0.9975 / 0.9974 / 0.9311 |
| **hold-out R²** | 0.9936 / 0.9945 / **0.9023** |

Fit sessions `20260731-150712`, `143025`, `144815`, `20260801-004843` (apex arcs),
`005059` (terminal runs). Hold-out `20260731-131305`, unchanged across both fits so
they are comparable. Excluded: `130744` and `233219` (`LOCAL_POSITION_NED` velocity
repeats bit-identically in 53% / 43% of rows — a recording artefact; re-flying
reproduces it), `005518` (identical rerun of `005059`, kept as a repeatability check).

`python pilot/control/plant.py` prints:

```
terminal climb   integrated 30.99 m/s   closed form 30.99 m/s
hover drift      +0.182 m/s after 5 s   (expect 0.147, the 15 ms thrust delay at zero thrust)
PASS
```

The integrator and the closed-form algebra agree, which is the point of that check —
one of them being wrong is otherwise silent.

**What card 2 retired.** The affine thrust curve went negative below throttle 0.10 and
was clipped to zero by hand; the drone actually holds ~2.3 m/s² there. That clamp — the
model's one unmeasured hack — is gone, replaced by the measured table. And `kz` is no
longer fitted jointly with thrust: in racing flight `corr(throttle, w|w|) ≈ −0.90`, so
the joint estimate returned whatever that collinearity produced (0.03639 shipped
previously, against 0.04359 measured here).

### Open on the plant

* **`kz` depends on forward speed and the model has no term for it.** 0.0443 at
  `|u| < 2`, 0.0412 at `|u|` 10–15 m/s. A body-lift `c·u²` term is the one the data
  supports: **+0.043 held-out body-z R²** over what `plant.json` ships (0.9023 →
  0.9456). Alternatives are worse (`c·u|u|` −0.015, `|u|w` +0.014). **It is not in
  `plant.json` and has never been through the replay gate** — it was measured after the
  refit closed. This is also why held-out body-z *fell* 0.916 → 0.902 across the refit:
  the old joint `kz` was absorbing lift.
* **Throttle 0.50–0.82 rests on ~240 samples.** Both ends are anchored hard (hover by
  6928 samples, full throttle by 250 steady ones); the middle-upper stretch, where a
  policy accelerates, is thin. ~30 s of terminal holds at 0.6 and 0.8 would fix it.
* **Only 4 of 8 apex arcs fed `kz`.** The throttle 0.15 and 0.20 reps coasted through
  less than 9 m/s of `w`; `sysid_apex.py` excludes them on the span, not on the answer,
  which is the only non-circular ordering. Neither terminal descent actually plateaued
  (+1.7 m/s over the last second), so those two reads are fitted slopes.
* **No completed lap exists in any session ever recorded.** `race_finish_time_ns` is −1
  everywhere; the furthest any run reached is `active_gate_index` 5. End-to-end
  validation is therefore against nine minutes of ordinary flying, not a lap.

---

## 2. Surrogate — built and internally consistent; one of three halves measured

The surrogate is three things bolted together: the fitted plant, a course generator, and
a synthetic detection producer. Only the first is measured.

### Measured and verified

* **Plant half.** Self-check stage 2 asserts `surrogate ≡ plant.Sim(−cmd)` to
  **1.78e-15** over 400 steps while the control `Sim(+cmd)` diverges **10.6 m** — the
  rate mirror is exercised, not assumed. Hover at 0.270 holds altitude to −0.467 m over
  3 s; terminal climb matches the closed form at 30.99 m/s.
* **Camera.** Spec-exact and passing: +49.4° / −9.4° vertical span about body-forward, a
  gate dead ahead at own altitude renders at v = 296 of 360 (below centre, the binding
  lower-edge constraint), `480/gate_px` recovers 20.000 m at 20 m fronto-parallel.
* **Observation encoding.** The fast encoder matches `interface.Observation.to_vector()`
  to 0.0 over 240 observations.

### Guessed

* **Detection statistics (P3 fallback).** `surrogate/noise.py` carries
  `max_range_m = (14, 30)` and `p_detect = (0.62, 0.92)`; every range in it is a reasoned
  guess. The one number that has been checked was wrong in the dangerous direction: a
  probe over `20260731-204841-vq1-lap-slow` — measured gate centres projected into
  recorded frames with truth pose, scored with the `NOTES.md` orange HSV mask — put a
  blob where geometry says one should be on **~95%** of in-frustum (gate, frame) pairs,
  **including ~95% at 30–45 m**, where the fallback says the gate is already gone.

  **How much that costs, measured 2026-08-01** — fly a privileged guidance law and record
  true range to the current gate against track validity. At difficulty 0.0 the
  current-gate track is valid on **63.0%** of steps and `normal_valid` on only **20.6%**;
  since the guidance target is `pos_body + d·normal_body`, the approach point is
  unavailable four steps in five. But only **11.4%** of steps are beyond the drawn
  `max_range_m`, and within range the track is still valid just 70.7%. **So the range
  band is not the dominant term — `p_detect`, its range taper and the burst dropout
  are.** `README.md` frames `max_range_m` as the main error; on this measurement it is
  the smaller half. Validity by true range: 74% at 10–15 m, 57% at 20–25 m, 25% at
  25–30 m.

  Self-check stage 9 (report-only) shows the error the policy is handed: median
  |reported − true| `pos_body` **1.00 m at a median range of 12.4 m**, p90 4.39 m
  (re-verified). Stage 9's own validity rates on random rollouts are higher
  (~85% current-gate valid / ~48% `normal_valid` at difficulty 0.0) than the
  privileged-guidance probe above — different flight distribution, same noise
  model.

  The **visibility half of P3 is measurable from data already on disk and is not blocked
  on Claire's detector** — `perception/label.py` already projects gate corners from truth
  pose, and `perception/detect.py --score` already scores against those labels. What does
  still need a real detector is the error model of a PnP fit: normal validity, hangar
  false positives.
* **Course generator, vertical prior.** `EnvConfig.alt_revert = 0.3` (was 0.9)
  mean-reverts altitude inside a ceiling band. Lowering it lengthens consecutive
  downhill runs only slightly (at difficulty 1.0, runs of ≥6 segments go from
  ~0.1% → ~0.3% of courses). The hard limit on drop magnitude is the hangar band
  itself: ceilings draw from 9–16 m, so max gate-to-gate altitude span is ~12 m —
  VQ1's monotone **24 m descent over 140 m** is geometrically unreachable until
  `ceiling_range_m` / clearances change, regardless of `alt_revert`. There is
  still no dedicated sustained-slope prior. The measured map also has a 39.4 m
  segment against the easy draw's 34 m ceiling, and its turns **alternate**
  (+5.7 / −12.1 / +15.6 / −19.4°) where the generator draws a persistent sign with
  a flip probability. Caveat: this is VQ1's map and VQ2 winds more, so it
  calibrates the vertical prior and the segment range only — it must **not** be
  used to pull turn magnitudes down.
* **No fixed evaluation course.** Every episode is generated and thrown away: right for
  training, useless for validation, because no surrogate lap time is comparable to a live
  one. The measured 6-gate map is the only course that can be flown in both places.
* **Attention is a stub.** `attention.py` with `R_COMMIT_M = 6.0` as a placeholder and
  `EnvConfig.attention_factory = None`. Meant to be Claire's real module; a behavioural
  mismatch there is a transfer risk on par with the noise model.

---

## 2a. Nothing completes a course, and fidelity is not why

Measured 2026-08-01, and this is the fact that governs whether the surrogate is usable
as a training target:

| controller | setting | completion | gates of ~20 |
|---|---|---|---|
| PPO `run1`, 5 046 272 steps | difficulty 0.0 | **0%** on all 77 updates | ~2.0 |
| `policies/baseline.py` via `evalsuite/run_policy.py` | difficulty 0.0, 16 seeds | **0/16** | 0.44 |
| same baseline, 24 seeds | difficulty 0.0 | **0/24** | 0.42 |
| self-check stage 4 scripted flight | hard course | **0 of 211 episodes** | — |

**Ablation says the unmeasured halves are not the cause.** Re-running the baseline over
24 seeds with one thing changed at a time:

| environment | completion | gates |
|---|---|---|
| training defaults | 0/24 | 0.42 |
| collision margin inflation removed (`margin_range_m = 0`) | 0/24 | 0.71 |
| near-perfect detections (long range, `p_detect` 0.995, no dropout/FP/latency) | 0/24 | 0.25 |
| both | 0/24 | 0.88 |

Handing the policy almost perfect vision and removing the margin moves it from 0.42 to
0.88 gates out of ~20. **The guessed noise model and the collision margin are transfer
risks, not the blocker.**

Two things that are *contributory* and worth knowing:

* **The gate-pass tolerance is tight against the miss distribution.** Passing requires
  `lat + sphere_r ≤ 0.75 m` with `sphere_r = 0.214 + margin` (`env.py:600`). At
  difficulty 0.0 the margin draws `U(0.10, 0.22)`, so the usable window is **0.32–0.44 m**
  of lateral miss, and it must be hit ~20 times consecutively. The repo's own probe
  comments record gate-plane misses of 0.4–0.9 m — the typical miss is at or past the
  tolerance. At difficulty 1.0 the window narrows to 0.14 m.
* **Episode starts are not the problem.** At reset, 88.2% of envs have the current gate
  ahead of the aircraft and 80.4% have it inside the frustum, so episodes are not
  systematically spawning facing the wrong way. (`_probe_signs.py` seed 0 shows a gate
  19–25 m *behind* that is never acquired, but that seed is in the unlucky ~12%, not the
  norm.)

**The root cause is not isolated.** What is established is the negative: it is not the
plant, not the detection noise, and not the collision margin. It sits somewhere in the
controllers, the reward, or the course/termination geometry, and finding it is the
gating task — not more fidelity work. (T4's time-limit truncation bootstrap now
consumes `terminal_obs`; that is a harness correctness fix after `run1`, not an
explanation of zero completions — see `CONTROL_STATE_AGENTS.md`.)

---

## 3. Doc drift found on disk

| claim | disk, 2026-08-01 |
|---|---|
| `HANDOFF_SURROGATE.md`: "`surrogate/selfcheck.py` 15/15" | **17/17**, over 9 stages |
| `HANDOFF_SURROGATE.md` / `README.md`: lap-slow session has 2937 JPEGs | `frames.csv` has 2937 rows; **1235 JPEGs are on this machine** (42%). The rest are on the other laptop. Also present: `20260731-233219` 2397 frames, `20260801-005715` 508 |
| `HANDOFF_SURROGATE.md` §Doc sync: "`SYSID.md` not updated, still reads as pre-flight" | Already updated in the working tree (uncommitted) — card 2 is marked FLOWN |
| `TRAINING_ARCHITECTURE.md`: P3 is the hand-specified fallback | Still true — planned method is measured; current is `noise.py` fallback (see that doc's planned-vs-current callouts) |
| `TRAINING_ARCHITECTURE.md` P1 "still open": thrust clamp + joint `kz` | **Retired by card 2** — measured knot table; `kz` from apex arcs (this file §1). Architecture doc updated. |
| Older agent state: PPO ignores `terminal_obs` | **Stale** — `ppo._truncation_values` bootstraps timeouts (out of P1/P2 scope; noted because §2a cites `run1`) |

The frame count matters: a visibility measurement over the lap-slow session runs on 1235
frames here, not 2937, unless the other laptop's copy is retrieved first.

---

## 4. How to re-verify

```powershell
python pilot/control/plant.py                   # expect PASS, 30.99 == 30.99
python pilot/control/surrogate/selfcheck.py     # expect 17/17 checks passed -- PASS
```

Full sysid order, if refitting rather than checking:

```powershell
python pilot/control/sysid_frames.py  <all session dirs>
python pilot/control/sysid_latency.py <all session dirs>
python pilot/control/sysid_apex.py
python pilot/control/sysid_fit.py               # writes plant.json
python pilot/control/sysid_replay.py            # the gate -- nothing trains until this passes
python pilot/control/plant.py
```

**Pass the frame referee every session directory, never card 2 alone.** Its margin
depends on attitude variety: all sessions pooled score 2.6× (PASS), card 1 alone 3.2×,
card 2 alone 1.0× (FAIL) — with the *same winning signs* in all three. A narrow subset
failing means "this data cannot tell the conventions apart", not "the convention is
wrong".

```powershell
$s = (Get-ChildItem pilot\sessions -Directory).FullName
python pilot/control/sysid_frames.py @s
```
