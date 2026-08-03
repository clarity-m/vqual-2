# Handoff — network architecture work, 2026-08-02/03

Everything below is on the current working line (`worktree-vq2-reward-eval`, commits
`4e346ed` → `489ab41`). **No worktrees remain**: `worktree-vq2-course-variations` and
`worktree-vq2-architecture` are both fully merged into HEAD and their directories are
gone. Work in `pilot/` directly.

Detail lives in `NETWORK_ARCHITECTURE.md`; this is the orientation page.

---

## The one-paragraph version

The network is fine — a conventional actor-critic with no defects found. The problems
were in **what it is asked to infer**: it cannot perceive its own vertical rate (nothing
in the 73-D vector reports it), and it was being made to rediscover
`thrust ∝ 1/(cos roll · cos pitch)` from reward, a term smaller than its own exploration
noise. Three opt-in changes address that. All three are wired end to end, recorded in the
checkpoint, and proven to produce identical commands in training and deployment. **None
has been trained to convergence** — that is the open work.

## The three flags

All default OFF. A run without them is bit-identical to before.

| flag | what it does | measured |
|---|---|---|
| `--vertical-rate` | appends 4 derived channels: leaky integrals of world-frame vertical acceleration | R² 0.630 vs 0.460 predicting true v_z |
| `--thrust-residual` | `u[2]=0` means "hold altitude" at any attitude, instead of the net learning the trig | corrects a term 0.24–0.43× the policy's own exploration σ |
| `--frame-offsets 32,16,8,4,2,0` | same 6 frames spread over 0.58 s instead of 0.111 s; identical width and parameter count | R² 0.460 → 0.562 |

Suggested first run, given both target the floor-strike failure that dominated run1
(55.5% of episodes):

```
python pilot/control/train/train.py --env surrogate --name <run> \
    --total-steps 200000000 --n-envs 2048 --n-steps 128 \
    --curriculum-window 1000 --gates-per-episode 22 --ent-coef 0.001 \
    --vertical-rate --thrust-residual
```

`--frame-offsets` is the weakest of the three and is a free add; it is not the fix.

## Why not a GRU

Asked and answered by measurement, so nobody re-opens it. Predicting true vertical
velocity, linear probe (a lower bound for both a stack and a GRU):

| input | inputs | R² |
|---|---|---|
| 6 contiguous frames (the default) | 438 | 0.460 |
| 6 dilated frames, 0.70 s | 438 | 0.562 |
| **32 frames at FULL RATE, same span** | **2336** | **0.556** |
| **1 frame + the 4 derived channels** | **77** | **0.630** |

Full-rate access to the window — exactly what recurrence buys over a stack — **gains
nothing**. What was missing is a *nonlinear* step stacking cannot perform (rotate body
specific force to world frame by attitude, add gravity, integrate — bilinear in
attitude×accel). So: we need recurrent **state**, not a **learned** one. Four scalars with
fixed time constants beat all 438 inputs of the shipping stack.

## What guards the dangerous failure

A checkpoint that scores well and flies *differently* — because deployment stacked frames
in another order or applied another thrust map — is the failure this area invites. Both
would look entirely normal.

`train/tests/test_train_deploy_agreement.py` drives **six** combinations of
{consecutive, dilated} × {affine, residual} × {plain, vertical-rate} through the training
path and through `RLPolicy`, and requires identical physical commands. All six at
**0.000e+00**. It also asserts a legacy checkpoint carries none of the new keys and loads
with every default. **Run it after touching `framestack.py`, `derived.py`, `actionmap.py`,
`ppo.py` or `policies/rl.py`.**

Checkpoints self-describe: `frame_offsets`, `thrust_residual`, `vertical_rate` are written
only when in use, so a default run still produces exactly the seven original keys.

## Gotchas worth knowing before editing

* **`interface.py` did not move and must not.** The derived channels are computed *from*
  fields the policy already receives (roll 54, pitch 55, accel 51–53). The network sees
  77-D; the contract stays 73-D. `ppo._raw_obs` is deliberately the **un-augmented** 73-D
  vector so the residual map's indices stay valid — do not "helpfully" augment it.
* **Order matters:** derive → augment → normalize → stack. The normalizer must fit the
  derived channels, and the stack must carry their history. Both `ppo.py` and
  `policies/rl.py` do it in that order; they must keep agreeing.
* **`CONVENTIONS.md` had the pitch/framing relation backwards** and is now corrected:
  pitching **down** brings a co-altitude gate *toward* frame centre (v 296 → 180); pitching
  **up** pushes it out of frame at about +9.4°. `surrogate/camera.py`'s docstring **still
  carries the old, wrong claim** — that file belongs to the environment work. Given
  CLAUDE.md names `CONVENTIONS.md` the single source for sign facts, this is worth
  finishing.
* Under `--thrust-residual` the commanded thrust varies with attitude even when the
  policy's output is constant, so `env.py`'s jerk penalty reads that compensation as jerk.
  Magnitude ~0.00013/step against a ~0.09 progress signal — negligible, but it is there if
  `k_jerk` is ever raised substantially.

## Findings handed to the environment work

**Spawn — already fixed by that session, verified here 2026-08-03.** `EnvConfig
.start_pitch_rad = -0.349` (−20°) now applies to race starts (`env.py:539`). Measured on
VQ2 forward starts: median pitch **−20.1°**, gate 0 at image row **174** of 360, **100% in
frustum** (was 89% at the old ~0° spawn), and start speed **0.00 m/s** — so the platform is
modelled too. Both findings this review raised are closed. Nothing to do.

One consequence worth carrying: `vel_bearing`/`speed` are drag-derived and therefore
**invalid at rest**, so at t=0 the aircraft genuinely has no velocity signal, and
`FrameStack.reset` fills all six slots with that. The derived vertical-rate channels start
at zero there too — correct, but it means the first ~0.3 s of every episode is flown with
no rate information of any kind.

Still open on the environment side:

* **The noise model is optimistic in three measured ways** (`perception-error.md`):
  ρ(0.1 s) is 0.954 modelled vs **0.43 measured**; `p_detect` 0.62–0.92 modelled vs
  **0.53–0.60 measured** with 0.57 s p90 blackouts; and the size-fallback range bias is
  **−3.5 m short**, not long as documented — which inverts the frame-strike prediction to
  *early* braking.

## Open, ranked

1. **Train it.** Nothing here has been trained to convergence. The flags are safe and
   measured on observability, not on task performance.
2. **The vertical-rate estimate is bounded, not solved.** R² 0.630, not 1.0. A pure
   integrator drifts on accelerometer bias (hence the leaky form), and per
   `perception-error.md` the vision channels are too noisy to correct that drift. Better
   altitude control, not a solved problem.
3. **`camera.py`'s docstring** still contradicts the corrected `CONVENTIONS.md`.
4. **Entropy bonus has zero gradient into either trunk**, because `log_std` is
   state-independent — exploration is a fixed isotropic blob the policy cannot modulate by
   situation. Consistent with `--ent-coef` 0.005 → 0.001 buying real accuracy; that lever
   is not exhausted.

## A correction this review made to itself

An intermediate claim here — that gate-position error was strongly correlated in time
(ρ≈0.95) and that differencing would therefore gain 3–10× — **was wrong**. It trusted a
surrogate parameter that had no measurement behind it. Measured reality is ρ(0.1 s) = 0.43,
so differencing gains 1.07×: essentially nothing. The original white-noise estimate was
approximately right. Recorded because the same mistake is easy to repeat: **the surrogate's
noise parameters are guesses with reasons, not measurements**, and `perception-error.md` is
the authority where the two disagree.
