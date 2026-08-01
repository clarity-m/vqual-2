# control/ — cold-start handoff

For Alex (kongalex@umich.edu). Written **2026-08-01** (verified on disk this session).
Working tree is **uncommitted**. Repo root handoff at `HANDOFF.md` is the older
project-wide doc; **this file is the control-pipeline entry point.**

## 1. Situation

| milestone | when (PST) | meaning |
|---|---|---|
| **T4 go/no-go** | **2026-08-02 06:00** | Is surrogate PPO training producing a policy worth live eval? |
| **Deadline** | **2026-08-03 06:00** | Ship an `interface.Policy` that beats the baseline |

**Architecture** (`TRAINING_ARCHITECTURE.md`, `control/README.md`) was updated for the
card-2 plant refit and the full offline pipeline (P2 surrogate → T4 PPO → E5 eval → D6
deploy). **This session built most of that pipeline on disk** — surrogate, train harness,
policies, evalsuite — but nothing is committed and several pieces are untested or broken
in ways the verification commands below expose.

The deliverable is still one thing: a **reactive or learned policy** that beats the
baseline on the live sim. Everything else is means.

---

## 2. What is built (verified on disk)

Run from the **repo root** (`c:\Users\Alex\vqual-2`):

```powershell
python pilot/control/surrogate/selfcheck.py      # P2 surrogate
python pilot/control/train/selftest.py             # T4 harness (~1 min CPU)
python pilot/control/policies/selfcheck.py         # baseline/supervisor unit checks
python pilot/control/evalsuite/run_policy.py --seeds 0-9 --difficulty 0.0   # E5 smoke
python pilot/control/plant.py                      # plant model self-check
python pilot/control/sysid_replay.py               # replay gate (any refit)
```

### Verification results (2026-08-01, this machine)

| command | result |
|---|---|
| `surrogate/selfcheck.py` | **17/17 PASS** (~47 s). Throughput ~46k env-steps/s @ 1024 envs. |
| `train/selftest.py` | **ALL CHECKS PASSED** (~74 s). Curriculum `progress_gate_scale` stall path verified. |
| `policies/selfcheck.py` | **FAIL** — hover thrust 0.3852 ≠ `HOVER_THRUST` 0.27; then `KeyError: 'climb'` (`baseline.last` uses `a_up`, selfcheck expects `climb`). |
| `evalsuite/run_policy.py` baseline, 3 seeds | **Runs** — 0% completion, 100% collision rate (policy not tuned yet). |

### Directory inventory

| path | status | notes |
|---|---|---|
| `surrogate/` | **Complete** | 11 modules: `env`, `single`, `actions`, `course`, `detect`, `noise`, `camera`, `attention`, `vmath`, `selfcheck`. |
| `train/` | **Complete harness** | `ppo`, `curriculum`, `network`, `framestack`, `normalize`, `actionmap`, `train.py`, `testenv`, `selftest`. |
| `policies/` | **Present, partial** | `baseline`, `supervisor`, `envelope`, `rl` (checkpoint loader), `selfcheck` (broken vs baseline). |
| `evalsuite/` | **Present, partial** | `run_policy`, `select`, `stress`; `_probe_*` helpers. Baseline eval runs; no winning policy yet. |
| `sysid_*.py`, `plant.py`, `plant.json` | **Done** | Card-2 refit landed 2026-08-01. |
| `train/checkpoints/smoke256_*` | **Smoke only** | 393k steps, 0% completion, negative returns — format is real, policy is not. |
| `train-followup/` | **Does not exist** | — |

---

## 3. Shared contracts (do not drift)

### Environment

- **`EnvConfig`** — all surrogate knobs (difficulty, noise draws, reward weights, …).
  Construct with `VecSurrogate(n_envs, config, seed)` or `SingleSurrogate(config, seed)`.
- **`VecSurrogate`** — vectorized training env; auto-resets; `obs_dim=73`, `act_dim=3`.
- **`SingleSurrogate`** — one env for eval/deployment; same physics, scalar info dict.

Import surface:

```python
from pilot.control.surrogate import VecSurrogate, SingleSurrogate, EnvConfig, policy_to_action
```

### Action envelope (`surrogate/actions.py`)

| constant | value | note |
|---|---|---|
| `RATE_CAP_RPS` | **2.75** | Not `interface.MAX_RATE_RPS` (6.0) — that is extrapolation. |
| `THRUST_FLOOR` | **0.10** | Matches lowest measured thrust table knot. **Do not lower unilaterally.** |
| `THRUST_CEIL` | 1.0 | |

**`policy_to_action(u)`** — sole squashing home: `u ∈ [-1,1]³` → physical
`[roll_rate, pitch_rate, thrust]`. Train harness, eval suite, and `policies/rl.py` all
import this; never reimplement.

### Checkpoint keys (`train/network.py`)

Every `.pt` checkpoint must carry exactly:

```
{model, obs_mean, obs_std, frame_stack, net_config, env_config, step}
```

- `obs_mean` / `obs_std`: float32 `[73]` — **per single frame**, not stacked.
- Pipeline order: **`normalize → stack`** (same in deployment).
- Sidecar `.json`: same minus `model`, plus `harness` block
  (`action_map_source`, `env_kind`, `seed`, …).
- Restore env: `restore_env_config(ckpt["env_config"])` — handles nested `noise` dict.

Reference artifact: `train/checkpoints/smoke256_final.pt` + `.json`.

### Curriculum / reward shaping

- **`progress_gate_scale`** — optional `EnvConfig` field (default 1.0). When `< 1.0`,
  weakens dense progress reward near gates only (`env._near_gate_mask()`). **Curriculum
  uses it**: on stall (completion stuck near zero), decrements toward
  `progress_gate_scale_min` (0.4) by `progress_gate_scale_step` (0.15). Verified in
  `train/selftest.py` and `surrogate/selfcheck.py` test 7.
- **Never weaken `k_collision`** — only progress near gates and `speed_cap` demote.

### `terminal_obs`

- **`VecSurrogate.step`** puts `info["terminal_obs"]` shape `[n, 73]` at done indices —
  the **ending episode's last observation**, before auto-reset. Test 8 in
  `surrogate/selfcheck.py` PASS.
- **`ppo.py` does NOT consume `terminal_obs` yet.** GAE bootstraps every `done` at zero
  (documented in `ppo.py` header and `train/README.md`). Mild value bias, accepted for
  now. Wiring `terminal_obs` into `_gae` is a ~5-line improvement when prioritized.
- **`SingleSurrogate`** still needs a shim in `evalsuite/run_policy.py` for array-valued
  info (scalarizes only true scalars).

---

## 4. Plant note (card-2 refit)

Refitted **2026-08-01** on card-2 data (`control/README.md` for full numbers):

- **Thrust** — 21-knot measured table (not affine + clamp). Hover **0.270** throttle.
  Deadband below 0.05; table is thin in 0.50–0.82 (240 samples).
- **`kz = 0.0436`** — from apex arcs alone (thrust parked, `w → 0`). Old joint fit was
  ~20% low. Forward-speed dependence (~7%) not modeled; body-lift `c·u²` is measured but
  not in `plant.json`.
- **`THRUST_FLOOR = 0.10`** aligns with the table floor the fit actually saw. Lowering it
  lets PPO exploit a region the plant never measured.

Re-verify after any plant change:

```powershell
python pilot/control/sysid_apex.py
python pilot/control/sysid_fit.py
python pilot/control/plant.py
python pilot/control/sysid_replay.py
python pilot/control/surrogate/selfcheck.py
```

**Frame referee:** pass **all** session dirs, not card-2 alone (card-2-only fails at 1.0×
margin — attitude variety too thin; winning signs identical).

---

## 5. Open / resume-next (priority order)

### Blocking T4 go/no-go (2026-08-02 06:00 PST)

1. **Fix `policies/selfcheck.py` ↔ `baseline.py` drift** — update checks for current
   `last` keys (`a_up` not `climb`) and reconcile hover-trim behaviour (level target
   outputs 0.385 thrust vs `HOVER_THRUST` 0.27).
2. **Tune baseline on surrogate** — `evalsuite/run_policy.py` shows 0% completion /
   immediate collisions at difficulty 0.0. Baseline is the submission floor; it must fly
   before RL upside matters.
3. **Run real training** — only `smoke256` (393k steps, 0% completion, ~−8 ep return)
   exists. Need a multi-million-step run with curriculum; watch `*_log.csv` for completion
   rate and gates/ep.
4. **Optional: wire `terminal_obs` into `ppo._gae`** — reduces truncation bias once
   learning is underway.

### Blocking deadline quality

5. **P3 detection noise** — `noise.py` ranges are guessed; visibility probe shows
   `max_range_m` too pessimistic (gates visible to 30–45 m, fallback cuts at 14–30).
   Measure from `20260731-204841-vq1-lap-slow` frames + projected gate centres (no Claire
   blocker for visibility half).
6. **Course generator** — cannot produce sustained one-way descent; segment/turn priors
   off measured 6-gate map. Land fixed eval course from measured crossings.
7. **E5 selection loop** — `evalsuite/select.py` + `stress.py` exist but need a candidate
   checkpoint that completes anything.
8. **D6 deploy path** — `policies/rl.py` loads checkpoints; end-to-end test:
   `run_policy.py --policy rl --ckpt smoke256_final --supervisor`.

### Hygiene

9. **Commit** — entire `surrogate/`, `train/`, `policies/`, `evalsuite/` stack is
   untracked.
10. **Doc sync** — `TRAINING_ARCHITECTURE.md` header still says P2/T4 "do not exist";
    `SYSID.md` still reads pre-card-2. Update when stabilizing.

### Explicitly do not

- **Do not fly card-2 maneuver D** — velocity-stream artefact, not fixable by re-flying.
- **Do not lower `THRUST_FLOOR`** without new sub-0.10 thrust measurements.
- **Do not edit `pilot/interface.py` unilaterally** — propose changes.

---

## 6. Doc pointers

Read in this order for a cold start:

| doc | path | why |
|---|---|---|
| Frozen contract | `pilot/interface.py` | Header block first — signs, obs layout, policy API. |
| Control ground truth | `pilot/control/README.md` | Sign traps, plant refit, open surrogate/plant items. |
| Training pipeline | `pilot/control/TRAINING_ARCHITECTURE.md` | Stage map P0→E5, reward design, eval protocol. |
| T4 harness | `pilot/control/train/README.md` | Checkpoint format, curriculum rules, silent-failure modes. |
| Sign single source | `pilot/CONVENTIONS.md` | Every axis/frame fact; change signs here first. |
| Measured facts | `pilot/NOTES.md` | Camera tilt, gate geometry, rates. |
| Data collection | `pilot/SYSID.md` | Card 1/2 maneuvers (stale — card 2 flown but doc not updated). |

---

## Quick status matrix

```
P0 data          DONE (card 2 A/B in git; no clean lap)
P1 plant         DONE + refit 2026-08-01
P2 surrogate     DONE (17/17 selfcheck)
P3 noise         FALLBACK (guessed; visibility measurable off disk)
T4 train         HARNESS DONE; real training NOT DONE (smoke only)
E5 eval          WIRED; baseline untuned
D6 deploy        rl.py present; untested end-to-end
```

**Path:** `pilot/control/HANDOFF.md`
**Policies:** yes (`baseline`, `supervisor`, `envelope`, `rl`, `selfcheck`)
**train-followup:** **no** (directory does not exist)
