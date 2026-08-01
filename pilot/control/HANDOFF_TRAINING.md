# control/ — RL training pipeline cold-start handoff

For Alex (kongalex@umich.edu). Written **2026-08-01** from disk inspection and
verification commands run this session. Working tree is **uncommitted**. Do **not** treat
`HANDOFF.md` as authoritative — it may have been overwritten.

## 1. Situation

| milestone | when (PST) | meaning |
|---|---|---|
| **T4 go/no-go** | **2026-08-02 06:00** | Is surrogate PPO training producing a policy worth live eval? |
| **Ship deadline** | **2026-08-03 06:00** | Deliver an `interface.Policy` that beats the baseline on the live sim |

**Architecture plan:** `TRAINING_ARCHITECTURE.md` (deadline 2026-08-03 06:00 PST).

The deliverable is one thing: a **reactive or learned policy** that beats the baseline.
Everything under `pilot/control/` is means toward that.

**Reality check:** the P2 surrogate and T4 harness are built and their self-checks pass.
The baseline policy, evalsuite smoke, and surrogate PPO smoke run do **not** yet show a
flyable floor. `TRAINING_ARCHITECTURE.md`'s header still says "P2/P3/T4 do not" exist —
that line is **stale**; the code on disk is ahead of the doc header.

---

## 2. What is built + verify commands + observed results

Run from repo root (`c:\Users\Alex\vqual-2`):

```powershell
python pilot/control/surrogate/selfcheck.py
python pilot/control/train/selftest.py
python pilot/control/policies/selfcheck.py
python pilot/control/evalsuite/run_policy.py --policy baseline --seeds 0-2 --difficulty 0.2 --max-steps 500
```

### Observed this session

| command | result | notes |
|---|---|---|
| `surrogate/selfcheck.py` | **PASS** (17/17, ~41 s) | Vector equiv, physics vs `plant.Sim`, camera, scripted flight, throughput ~54k env-steps/s @ n=1024, `SingleSurrogate==VecSurrogate`, `progress_gate_scale`, `terminal_obs`, detection stats (report-only) |
| `train/selftest.py` | **PASS** (ALL CHECKS PASSED, ~24 s) | Frame stack, obs norm, hover bias init, curriculum stall path (`progress_gate_scale` → 0.40), PPO learns `testenv` stub, checkpoint round-trip |
| `policies/selfcheck.py` | **FAIL** (exit 1, crashed) | Horizontal steering PASS; vertical FAIL `level target -> HOVER_THRUST` got **0.3852** not **0.27**; then **KeyError `'climb'`** — `baseline.py` now logs `a_des`/`a_up`, selfcheck still expects `climb` |
| `evalsuite/run_policy.py` baseline smoke | **runs, 0% completion** | 3 seeds @ difficulty 0.2: 0 gates, 100% collision rate, ~120–210 steps/episode |

Optional surrogate-training smoke already on disk: `train/checkpoints/smoke256_final.pt`
(393 216 steps, 73-D). Its log shows **0% completion** throughout and negative mean returns
(≈ −8 by step 393k, ~68% collision rate).

### Directory map (what exists)

| path | role |
|---|---|
| `surrogate/` | P2 vector env (`VecSurrogate`), `SingleSurrogate`, procedural course, synthetic detections, fallback noise model, action envelope |
| `train/` | T4 PPO harness (`train.py` entry), curriculum, checkpoint I/O |
| `policies/` | `baseline.py`, `rl.py` (checkpoint wrapper), `envelope.py`, `supervisor.py`, `selfcheck.py` |
| `evalsuite/` | E5: `run_policy.py`, `select.py`, `stress.py`, `_probe_*.py` |
| `plant.json` / `plant.py` | P1 fitted plant (validated 2026-07-31) |
| `train/checkpoints/` | `selftest_final.*` (testenv harness), `smoke256_final.*` (surrogate smoke) |

---

## 3. Shared contracts (as implemented)

### Surrogate env API (`surrogate/env.py`)

```python
obs = env.reset()                          # [n, 73] float32
obs, rew, done, info = env.step(physical)  # physical [n, 3]: roll_rate, pitch_rate, thrust
```

- **Actions in `step`:** PHYSICAL units in canonical body NED. Network outputs go through
  `policy_to_action` first.
- **Yaw:** not in the action vector; driven by `attention.py` under `AUTO_ATTENTION`.
- **Sign rule:** no mirror above the link layer. Surrogate `+cmd` must match
  `plant.Sim(-cmd)` (verified in selfcheck test 2).
- **`info` keys:** `gates_passed`, `collision`, `corridor_exit`, `timeout`, **`finished`**
  (curriculum completion source), `episode_return`, `episode_length`, **`terminal_obs`**
  (`[n, 73]` — ending episode obs before auto-reset; meaningful only where `done`).

### Action map (`surrogate/actions.py` — sole squashing home)

| constant | value | note |
|---|---|---|
| `RATE_CAP_RPS` | **2.75** | Training envelope; **not** `interface.MAX_RATE_RPS` (6.0) |
| `THRUST_FLOOR` | **0.10** | Conservative training guard (see §4) |
| `THRUST_CEIL` | **1.0** | |

`policy_to_action(u)` — `u ∈ [-1,1]³` → `(roll_rate, pitch_rate, thrust)` physical.
Order: roll, pitch, thrust. Thrust is affine on `[THRUST_FLOOR, THRUST_CEIL]`; `u=0`
→ thrust **0.55**, not hover. Thrust head bias init targets `interface.HOVER_THRUST`
(0.27) via `thrust_to_unit` / `actionmap.invert_thrust`.

`train/actionmap.py` imports the real mapping, warns on fallback, records
`action_map_source` in checkpoint sidecar.

### PPO harness (`train/`)

- **Pipeline order:** `normalize → stack → network → policy_to_action → env.step`.
- **Curriculum axes:** `difficulty`, `speed_cap`; flips `enable_time_penalty` above ~80%
  completion; on stall weakens **`progress_gate_scale`** (default 1.0, min 0.4, step 0.15)
  and backs off `speed_cap` — **never** touches collision penalty.
- **`progress_gate_scale`:** `EnvConfig` field; scales dense progress symmetrically within
  `2 × approach_d_m` (4 m) of gate approach points. Verified in surrogate selfcheck §7 and
  train selftest §4.

### Checkpoint format (`train/network.py`)

`.pt` dict with **exactly** these keys:

| key | type |
|---|---|
| `model` | `ActorCritic` state_dict |
| `obs_mean` | float32 `[obs_dim]` (73 for surrogate) |
| `obs_std` | float32 `[obs_dim]` |
| `frame_stack` | int k |
| `net_config` | ActorCritic constructor kwargs |
| `env_config` | EnvConfig fields (JSON-flattened) |
| `step` | int env steps |

Sidecar `<name>.json` = same minus `model`, plus optional `harness` block (provenance only).

Reload env config with noise typed correctly:

```python
from pilot.control.train.train import restore_env_config
cfg = restore_env_config(ckpt["env_config"])
```

Reference checkpoints:

- `checkpoints/selftest_final.pt` — **testenv** stub, obs_dim=8 (harness proof only)
- `checkpoints/smoke256_final.pt` — **surrogate**, obs_dim=73, 393k steps, real format

### Deployment (`policies/rl.py`)

Checkpoint → frozen `ObsNormalizer` → `FrameStack` → `ActorCritic.infer(deterministic)` →
`policy_to_action` → `interface.Action(yaw_mode=AUTO_ATTENTION)`. Uses training sibling
code, not reimplemented.

---

## 4. Plant card-2 / THRUST_FLOOR

`plant.json` on disk (card-2 refit landed):

- **`kz_source`:** `"sysid_apex arcs"` (joint-fit collinearity addressed per architecture)
- **`thrust_knots`:** measured table from throttle **0.0002** through 1.0 (20 knots); idle
  thrust ≈ **0.35 m/s²** at the lowest knot; hand clamp below 0.10 is **gone**

`actions.py` module doc (lines 18–25) states explicitly: the original "unmeasured below 0.10"
reason **has expired**; **`THRUST_FLOOR = 0.10` is now a conservative training guard**, not a
physics requirement. Lowering it (~0.02 on evidence) is a deliberate one-commit + retrain
change — do not change silently mid-project.

`TRAINING_ARCHITECTURE.md` §P1 and T4 still mention the old hand-clamp in places; trust
`plant.json` and `actions.py` over those stale sentences.

---

## 5. Broken / incomplete (honest)

1. **`policies/selfcheck.py` FAIL** — baseline vertical unit checks out of sync with
   refactored accelerometer-based thrust loop; crashes on missing `last['climb']`.
2. **Baseline does not fly the surrogate** — evalsuite smoke: 0/3 seeds pass a gate;
   100% collision. This is the submission floor and it is not there yet.
3. **Surrogate PPO smoke (`smoke256`)** — 393k steps, 0% completion, negative returns;
   not a usable policy checkpoint.
4. **`terminal_obs` not wired into PPO** — `VecSurrogate.step` provides
   `info["terminal_obs"]` (verified selfcheck §8), but `ppo.py` bootstraps every `done` at
   zero. Module docstrings in `ppo.py` and `train/README.md` still claim the terminal obs
   "is not recoverable" — **contradicts the env**; wiring into `_gae` is a small fix.
5. **`attention.py` is a STUB** — `surrogate/__init__.py` says "meant to be replaced" with
   Claire's real attention module; transfer risk if left stubbed.
6. **P3 noise model is fallback only** — `surrogate/noise.py` is hand-specified pessimistic
   randomization, not measured from perception pipeline. Architecture allows training on
   fallback but flags it at model selection.
7. **`TRAINING_ARCHITECTURE.md` header stale** — claims course generator / P3 / T4 "do not"
   exist; disk has full surrogate + train harness. Body sections are partly updated (P3
   half-measured note 2026-08-01) but header and some P1/T4 thrust-floor wording lag.
8. **`SingleSurrogate` + `terminal_obs` shim** — `evalsuite/run_policy.py` carries a
   workaround because `single.py` scalarizes 73-vectors with `.item()` (2026-08-01 note).
9. **No committed git history** for any of `pilot/control/` pipeline files (untracked).

---

## 6. Recommended next actions

Priority order for the T4 go/no-go (~24 h):

1. **Fix baseline to fly** — tune/gain-fix `policies/baseline.py` until evalsuite shows
   nonzero completion on easy seeds (`--difficulty 0.0–0.2`), then update `selfcheck.py`
   to match the accelerometer thrust loop (`a_des`/`a_up`, not `climb`).
2. **Run evalsuite baseline sweep** — `run_policy.py` over 20+ seeds; `stress.py` for
   28–32 Hz decision rate; establish the floor number PPO must beat.
3. **Launch real T4 training** — `python pilot/control/train/train.py --env surrogate
   --total-steps 20000000 --n-envs 256 --frame-stack 6 --name run1`; watch
   `completion_rate` in `_log.csv` and curriculum promotions.
4. **Optional quick win:** wire `info["terminal_obs"]` into `ppo._gae` for done indices
   (reduces truncation bias; env already provides the data).
5. **Before ship:** E5 `select.py` over checkpoints vs baseline; deploy best via
   `policies/rl.py` + `supervisor.py`; live sim eval (serialized ~15–25 runs/session).
6. **Do not lower `THRUST_FLOOR`** without a deliberate commit and retrain (see §4).

---

## 7. Doc pointers

| doc | use |
|---|---|
| `TRAINING_ARCHITECTURE.md` | Stage definitions P0–T4, E5, D6; reward/curriculum intent (**verify header against disk**) |
| `control/README.md` | Sign traps, plant validation gate, surrogate open items |
| `pilot/interface.py` | Frozen contract: 73-D obs, `HOVER_THRUST=0.27`, action/yaw modes |
| `pilot/CONVENTIONS.md` | Every sign, frame, axis fact — single source |
| `train/README.md` | Harness CLI, checkpoint format, three silent-failure modes |
| `surrogate/__init__.py` | Module map, P3 fallback vs measured, attention stub note |

**Not authoritative:** `HANDOFF.md` (may be stale/overwritten).

---

## Quick grep anchors

```
terminal_obs       env.py provides; ppo.py does NOT consume
progress_gate_scale  env.py EnvConfig + curriculum.py stall handler
RATE_CAP_RPS=2.75    surrogate/actions.py (not interface 6.0)
THRUST_FLOOR=0.10    surrogate/actions.py (training guard post card-2)
policy_to_action     sole action squashing home
```
