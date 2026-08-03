# CONTROL PIPELINE STATE — FOR AGENTS

**Generated:** 2026-08-01 from disk inspection of `pilot/control/` (source, `plant.json`, checkpoint sidecars, training CSVs).  
**Do not treat older `HANDOFF*.md` / stale header lines as authority.** Prefer this file + the code.  
**P1/P2 detail:** `STATE_PLANT_SURROGATE.md` beside this file — same date, scoped to plant + surrogate, every number verified by executing the self-checks.  
**Course generation (2026-08-02, NEWER than this file):** `STATE_VQ2_COURSE.md` — the measured VQ2 map is now wired into the surrogate (`surrogate/vq2course.py`, opt-in via `vq2_frac`). It supersedes any statement here or in `TRAINING_ARCHITECTURE.md` that courses are always procedural, and carries a spec audit of the collision model.

**Deliverable:** an `interface.Policy` that beats the reactive baseline on the live sim.  
**Contract:** `pilot/interface.py` (frozen; propose changes, do not edit unilaterally).  
**Signs:** `pilot/CONVENTIONS.md` is the single source. Link layer owns the sim rate mirror (`SIGN_* = -1`). No sign flip above the link layer.

---

## 0. Status matrix (disk truth)

| Stage | Path | Status |
|---|---|---|
| P0 data | sessions referenced by `plant.json` meta | Fit set + hold-out recorded in plant meta. Sessions dirs may be machine-local. |
| P1 plant | `plant.json`, `plant.py`, `sysid_*.py` | **DONE** — card-2 refit on disk |
| P2 surrogate | `surrogate/` | **BUILT** — 9 selfcheck stages in `selfcheck.py` (8 PASS/FAIL + report-only #9) |
| P3 noise | `surrogate/noise.py` | **FALLBACK only** — hand-specified ranges, not measured |
| T4 train | `train/` | **Harness complete** (truncation bootstrap + `--resume`); `run1` ~5.05M steps, **0% completion throughout** |
| E5 eval | `evalsuite/` | **Wired**; probes + suite present; no winning policy artifact |
| D6 deploy | `policies/rl.py`, `supervisor.py` | **Code present**; end-to-end live path not evidenced on disk |

---

## 1. Directory inventory

```
pilot/control/
  plant.json, plant.py, plantfit.py          # fitted plant + dead-end record
  sysid_frames.py, sysid_latency.py,
  sysid_apex.py, sysid_fit.py, sysid_replay.py, sysid_data.py
  surrogate/     # P2 env
  train/         # T4 PPO
  policies/      # baseline, supervisor, envelope, rl, selfcheck
  evalsuite/     # run_policy, select, stress, _probe_*
```

### Checkpoint artifacts (`train/checkpoints/`)

| Artifact | Role |
|---|---|
| `run1_s1048576` … `run1_s5046272` (+ `.pt` binary, `.json` sidecar) | Archived every ~1M steps |
| `run1.json` / `run1.pt` | Rolling latest (sidecar on disk at **step 4587520**; archive ahead at **5046272**) |
| `run1_log.csv` | 77 updates, steps 65536 → 5046272 |
| `smoke256_final.*`, `smoke256_log.csv` | 393216-step smoke |
| `selftest_final.*` | testenv harness proof (obs_dim=8) |

`.pt` files exist on disk (binary); load via `train/network.load_checkpoint`.  
**No `*_resume.pt` beside `run1` archives** — those sidecars are written by the current saver; `run1` predates them. Continuing `run1` with `--resume` therefore takes the weights-only / degraded path unless you first re-save under the new code.

---

## 2. Plant (`plant.json` on disk)

| Field | Value |
|---|---|
| `kx` / `ky` / `kz` | 0.04867 / 0.04963 / **0.04359** |
| `kz_source` | `"sysid_apex arcs"` |
| `kz_spread` | 1.6e-4 |
| `kz_terminal` | [0.04347] |
| `thrust_knots` | 21 knots, throttle 0.0002 → 1.0; full T/m ≈ **51.67**; idle ≈ **0.35** m/s² |
| `rate_gain` | [0.970, 0.963, 0.904] |
| `rate_delay` / `thrust_delay` | 0.01 / 0.015 s |
| `rate_tau_max` | 0.01 |
| fit R² | [0.9975, 0.9974, 0.9311] |
| holdout R² | [0.9936, 0.9945, **0.9023**] |
| `n_samples` | 23261 |

**Fit sessions:** `20260731-150712`, `143025`, `144815`, `20260801-004843`, `20260801-005059`  
**Hold-out:** `20260731-131305`  
**Excluded:** `130744`, `233219` (stale velocity stream), `005518` (duplicate of B)

**Sysid order (code):** frames → latency → apex → fit → replay → `plant.py` self-check.

**Model:** body-axis quadratic drag; thrust = `np.interp` over knots (no clamp); rates via `SIGN_RATE = -1` only inside `plant.rates_ned()` for recording-convention inputs. Surrogate feeds **canonical NED** into the integrator and must **not** call `rates_ned()` on already-NED cmds (`env.py` + selfcheck physics test).

**Open plant gaps (from code/comments, not re-fitted):** forward-speed dependence of `kz` / optional body-lift `c·u²` not in `plant.json`; mid-table 0.50–0.82 thinner than ends.

---

## 3. Surrogate contracts

### Import surface (`surrogate/__init__.py`)

```python
from pilot.control.surrogate import (
    VecSurrogate, SingleSurrogate, EnvConfig, policy_to_action,
    RATE_CAP_RPS, THRUST_FLOOR, THRUST_CEIL, NoiseParams, ...
)
```

### Action envelope (`surrogate/actions.py`) — sole squash home

| Constant | Value | Note |
|---|---|---|
| `RATE_CAP_RPS` | **2.75** | Not `interface.MAX_RATE_RPS` (6.0) |
| `THRUST_FLOOR` | **0.10** | Training guard post card-2 (table measured below 0.10) |
| `THRUST_CEIL` | 1.0 | |
| `policy_to_action(u)` | `u∈[-1,1]³` → physical rates + thrust | `u=0` → thrust **0.55**, not hover |
| `thrust_to_unit` | inverse for hover bias init | |

Mirrored copy in `policies/envelope.py` (`RATE_CAP_RPS=2.75`, `THRUST_FLOOR=0.10`); selfcheck asserts no drift.

### `EnvConfig` defaults (`surrogate/env.py`)

Critical fields:

- Curriculum: `difficulty=0.5`, `speed_cap=1.0` (training starts 0.0 / 0.5 via CLI)
- `decision_hz_range=(45,65)`, `n_gates_range=(18,22)`, `substeps=3`
- Course: seg 26–34 easy / 18–26 hard; turn 20→80°; elev 8→22°; `alt_revert=0.3` (was 0.9; hangar band ~12 m still caps drop vs VQ1’s 24 m)
- Plant rand: drag/thrust ±15%, rate_gain ±10%, delay ×(0.7–1.4)
- Collision: `margin_range_m=(0.10,0.40)`, sphere 0.214 m
- Reward: `k_progress=1`, `k_cross=5`, `k_collision=8`, `k_corridor=8`, `k_finish=10`, `k_jerk=0.02`, `k_nogate=0.01`, `k_time=0.01`, `approach_d_m=2.0`, `progress_gate_scale=1.0`
- Starts: probs `(0.50, 0.25, 0.12, 0.08, 0.05)` = normal / mid / hover / corridor-offset / no-gate
- `r_commit_m=6.0` (placeholder), `attention_factory=None` → stub

### Step contract

```python
obs = env.reset()                          # [n,73] float32
obs, rew, done, info = env.step(physical)  # physical [n,3]: roll_rate, pitch_rate, thrust (canonical NED)
```

**info keys:** `gates_passed`, `collision`, `corridor_exit`, `timeout`, `finished`, `episode_return`, `episode_length`, **`terminal_obs`** `[n,73]` (pre-reset ending obs where done).

**Yaw:** not in action; `attention.py` stub + azimuth servo under `AUTO_ATTENTION`. Swap via `config.attention_factory`.

### Attention (`attention.py`)

**Parameterized stub.** `R_COMMIT_M = 6.0` TBD. Meant to be replaced with Claire’s real module. Transfer risk if left.

### Noise (`noise.py`)

P3 fallback. `max_range_m=(14,30)`, `p_detect=(0.62,0.92)`, bursty dropout, etc. All ranges guessed. Difficulty scales toward pessimistic end. **Not measured.**

### `SingleSurrogate` (`single.py`)

Thin adapter over `VecSurrogate(1)`. Scalarizes info with `e.item()` only when `size==1`; **`terminal_obs` stays array**. Evalsuite still has a defensive shim if older `single.py` raises.

### Selfcheck (`surrogate/selfcheck.py`)

Tests 1–8 assert; 9 report-only. Run: `python pilot/control/surrogate/selfcheck.py`

---

## 4. Training harness (`train/`)

### CLI defaults (`train.py`)

```
--env surrogate --total-steps 20000000 --n-envs 256 --n-steps 128
--frame-stack 6 --hidden 256,256 --gamma 0.99 --gae-lambda 0.95
--lr 3e-4 --ent-coef 0.005 --target-kl 0.03
--difficulty-start 0.0 --speed-cap-start 0.5 --checkpoint-every 1000000
--promote-rate 0.70 --demote-rate 0.25
--resume None   # opt-in continuation; see Resume below
```

`--total-steps` is an **absolute** target (not “steps this session”).

### Pipeline order

**Deploy / inference:** `normalize → stack → network → policy_to_action → clamp_action`  
**Collect (train):** `stack.get → net.act → policy_to_action → ×speed_cap(rates) → env.step` then, on done, trunc bootstrap from `terminal_obs`; then `obs_norm.update → stack.push`. Env also `clip_physical` on entry.

### Checkpoint keys (`network.py`) — exact

```
{model, obs_mean, obs_std, frame_stack, net_config, env_config, step}
```

- `obs_mean`/`obs_std`: float32 `[73]` per **single** frame  
- Sidecar `.json`: same minus `model`, plus `harness`  
- Restore: `restore_env_config(ckpt["env_config"])` (re-types nested `noise`)  
- Resume sidecar `<stem>_resume.pt`: optimizer, RMS counts, curriculum counters, RNGs (`RESUME_FORMAT=1`) — **not** part of the 7-key deploy contract

### PPO (`ppo.py`)

- Standard clipped PPO; `gamma=0.99`, `gae_lambda=0.95`
- **Time-limit truncations bootstrap from `V(s_T)`** via `info["terminal_obs"]` + `info["timeout"]` (`_truncation_values` → GAE). Reconstructs the terminal frame stack from the pre-step stack + terminal obs.
- **Genuine terminals** (collision / corridor_exit / finished / completed) still bootstrap at **0**; if a flag coincides with timeout, the terminal wins.
- Envs that omit both keys fall back to bootstrap-every-done-at-zero.
- Scale: at `run1`’s ~0.94 collision rate, truncations are a small fraction of endings — correctness fix, not a completion fix. See `train/README.md`.

### Curriculum (`curriculum.py`)

- Promote at rolling completion ≥0.70; demote ≤0.25  
- Stall (40 updates, no promote, rate < promote): lower `progress_gate_scale` by 0.15 toward min 0.4; back off `speed_cap`; **never** touch `k_collision`  
- Time penalty on only above ~80% completion (off below ~60%)  
- Env rebuild on config change (rate-limited by `hold_updates`)  
- Demotion never lowers `difficulty` / `speed_cap` below their start values

### Resume (`train.py --resume`)

- Opt-in: `--resume NAME_OR_PATH` (e.g. `run1_s5046272`). Does **not** auto-resume.
- With matching `<stem>_resume.pt`: restores optimizer, obs+reward statistics (counts), curriculum counters, RNGs; appends to `*_log.csv`.
- Without sidecar: weights + obs mean/std from the `.pt`, RMS count seeded to `DEGRADED_OBS_COUNT=1e5`, loud banner for lost Adam/curriculum/RNG state.
- Frame-stack / hidden sizes follow the checkpoint when they disagree with CLI flags.

### Selftest (`train/selftest.py`)

Seven stages: frame stack, obs norm, hover bias, curriculum stall path, PPO on `testenv`, checkpoint round-trip, **truncation bootstrap**, **resume round-trip** (full sidecar + bare `.pt`). Run: `python pilot/control/train/selftest.py`

---

## 5. Empirical training: `run1`

From `train/checkpoints/run1_log.csv` (authoritative for metrics):

| Metric | Observation |
|---|---|
| Steps | 65 536 → **5 046 272** (77 updates) |
| `completion_rate` | **0.0 every row** |
| `difficulty` / `speed_cap` | stuck **0.0 / 0.5** (never promoted) |
| `gates` (mean) | ~1.4–2.6 |
| `collision_rate` | ~0.99 → dip ~0.61 @ ~1.57M → back to ~0.94 @ 5M |
| `ep_return` | ~−3 → trough ~−8.5 → recovery to ~+1…+3 |
| `progress_gate_scale` | not in CSV; sidecar `run1_s5046272.json` has **0.85** (one stall step from 1.0) |
| Harness | `seed=1`, `action_map_source=surrogate`, frame_stack=6, hidden 256×256 |

**Verdict:** learning some survival/return shaping; **not learning to finish courses.** Curriculum cannot advance without completions.

`smoke256`: 393k steps, same pattern (0% completion, negative returns).

---

## 6. Policies

### Baseline (`policies/baseline.py`)

Reactive 3-DoF under `AUTO_ATTENTION`:

- **Roll:** `b_err = bearing(target) − vel_bearing` (velocity-referenced)  
- **Pitch:** speed trim + frame guard (nose-down when target near −9.4° edge)  
- **Thrust:** velocity-referenced climb `w_cmd = V·sin(e_horizon)`, leaky `w_est` (`w_tau=4`), accel feedback, **hover trim integrator**  
- State: `trim`, `w_est`  
- Defaults include `v_cruise=6.5`, `k_bearing=1.1`, `k_w=2.0`, `approach_d=2.0`, …  
- `last` keys: `a_des`, `a_up`, `w_cmd`, `w_est`, `e_horizon`, … (**not** `climb`)  
- Tune via `gains_from_str("k_roll=8,v_cruise=5.5")`

### Selfcheck (`policies/selfcheck.py`)

Updated for accel / climb-rate cascade (`accel_for`, `a_up`, trim, frame guard, supervisor, envelope drift). Run: `python pilot/control/policies/selfcheck.py`  
Older handoffs claiming KeyError `'climb'` / hover 0.385 are **stale relative to current selfcheck**.

### Supervisor (`supervisor.py`)

Collision-timer / episode-counter trigger → level + hover → hand back. Resets inner on activate.

### RL (`rl.py`)

`Observation → to_vector → ObsNormalizer → FrameStack → infer(deterministic) → policy_to_action → clamp_action`. Uses training sibling modules only.

### Envelope (`envelope.py`)

`clamp_action`; copies `RATE_CAP_RPS`/`THRUST_FLOOR` from surrogate.

---

## 7. Evalsuite

| File | Role |
|---|---|
| `run_policy.py` | Primary eval; `--policy baseline|rl`, `--ckpt`, `--supervisor`, seeds, difficulty |
| `select.py` | Rank ckpts by high-percentile completion + mean time |
| `stress.py` | Same eval at `decision_hz_range=(28,32)` |
| `_probe_baseline.py` | Classify terminal why (FLOOR/CEILING/GATE FRAME) |
| `_probe_vertical.py` | Inner/outer vertical loop diagnostics |
| `_probe_accel.py` | `a_up` recovery vs truth |
| `_probe_signs.py` | Obs vs privileged body offsets |
| `_probe_sweep.py` | Gain coordinate descent; SEEDS=0..23, DIFFS=(0.0,0.35) |

**No scored winning-policy JSON on disk.** Probe scripts encode prior qualitative findings in comments (standing `e_horizon` aim-low before climb-rate fix; gate-plane misses 0.4–0.9 m).

---

## 8. Shared gotchas (code-enforced)

1. **Sign silence:** wrong body↔world in plant → fits, mistunes all gains. Replay gate before trusting.  
2. **Surrogate +cmd ≡ plant.Sim(−cmd)** for recording-convention plant path.  
3. **`policy_to_action` is the only squash** — do not reimplement.  
4. **Normalize before stack** on the deploy / inference path (`obs_mean`/`obs_std` stay 73-D).  
5. **Never weaken `k_collision`.** Stall → `progress_gate_scale` / `speed_cap` only.  
6. **Do not lower `THRUST_FLOOR` silently** — one deliberate commit + retrain.  
7. **Do not fly excluded D sessions** — velocity-stream artifact.  
8. **Frame referee:** pass **all** session dirs; card-2-only fails margin for lack of attitude variety.  
9. **Truncation ≠ terminal:** PPO bootstraps timeouts from `terminal_obs`; collisions/exits/finishes stay V=0.  
10. **Attention stub + P3 fallback** = transfer risk flags at model selection.  
11. **`--resume` without `*_resume.pt` is degraded** — do not treat it as a clean continuation.

---

## 9. Verify commands (repo root)

```powershell
python pilot/control/plant.py
python pilot/control/sysid_replay.py
python pilot/control/surrogate/selfcheck.py
python pilot/control/train/selftest.py
python pilot/control/policies/selfcheck.py
python pilot/control/evalsuite/run_policy.py --policy baseline --seeds 0-9 --difficulty 0.0
python pilot/control/evalsuite/run_policy.py --policy rl --ckpt run1_s5046272 --seeds 0-9 --difficulty 0.0
```

Training continue (opt-in resume; absolute `--total-steps`):

```powershell
python pilot/control/train/train.py --env surrogate --total-steps 20000000 `
  --n-envs 256 --frame-stack 6 --name run1 --seed 1 --resume run1_s5046272
```

Fresh run (no `--resume`):

```powershell
python pilot/control/train/train.py --env surrogate --total-steps 20000000 `
  --n-envs 256 --frame-stack 6 --name run2 --seed 1
```

---

## 10. Priority work queue (inferred from disk)

1. **Make baseline complete gates** on easy surrogate seeds (`evalsuite` / `_probe_sweep`). Floor for ship.  
2. **Diagnose run1 0% completion** — reward horizon / gamma, collision geometry / gate-pass tolerance, detection fallback, attention stub, progress shaping. Truncation bootstrap is already wired; do not expect it alone to unlock completions.  
3. **Measured P3 visibility** from existing frames + gate projections (not blocked on Claire for visibility half).  
4. **Course generator:** optional sustained-descent prior (beyond weaker `alt_revert`); optional fixed 6-gate eval course.  
5. **E5 select + D6** once any checkpoint or baseline shows nonzero completion.  
6. Replace attention stub when Claire’s module exists.

---

## 11. Doc drift warning

| Claim in older docs | Disk |
|---|---|
| `policies/selfcheck` broken on `climb` / 0.385 thrust | Selfcheck rewritten for `a_up` / climb-rate |
| P2/T4 “do not exist” | Full trees present |
| `THRUST_FLOOR` because unmeasured | Table measured; floor is training guard (`actions.py`) |
| PPO terminal obs unrecoverable / unused | Env emits `terminal_obs`; **PPO bootstraps timeouts from it** (`_truncation_values`) |
| “train does not auto-resume / no load logic” | True that it is not silent; **`--resume` + `*_resume.pt` exist** |
| Discount-factor note under T4 | Referenced in `TRAINING_ARCHITECTURE.md` header; **no such section body found** |
| `run1` nonexistent | ~5M steps logged, 0% completion |
| `HANDOFF_SURROGATE.md`: surrogate selfcheck "15/15" | **17/17** over 9 stages (re-verified 2026-08-01) |
| `HANDOFF_SURROGATE.md` / `README.md`: lap-slow session has 2937 JPEGs | `frames.csv` has 2937 rows; **1235 JPEGs on this machine** (42%). Also on disk: `233219` 2397, `005715` 508 |
| `HANDOFF_SURROGATE.md` §Doc sync: `SYSID.md` "not updated" | `SYSID.md` marks card 2 FLOWN in the working tree (uncommitted) |

**Authority order:** code + `plant.json` + checkpoint sidecars/logs > this file > architecture/handoff markdown.  
`TRAINING_ARCHITECTURE.md` is the design-intent doc and now marks planned vs current in place (noise, attention, discount hypothesis, etc.).
