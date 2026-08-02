# train/ — T4, the PPO harness

Stage T4 of `../TRAINING_ARCHITECTURE.md`. Trains a 3-DoF policy (roll rate, pitch rate,
thrust — yaw stays with the `AUTO_ATTENTION` servo) against the vectorized surrogate in
`../surrogate/`, and writes checkpoints the eval/deployment side reloads.

Requires `torch` (CPU is enough — verified on 2.10.0+cpu) and `numpy`. Nothing else.

## Run it

    # harness verification, no surrogate needed, ~1 min CPU
    python pilot/control/train/selftest.py

    # real training
    python pilot/control/train/train.py --env surrogate --total-steps 20000000 \
        --n-envs 256 --n-steps 128 --frame-stack 6 --name run1

    # the harness on its own stub env, if you want to watch it learn something trivial
    python pilot/control/train/train.py --env testenv --total-steps 400000 \
        --n-envs 64 --n-steps 64 --no-curriculum

`--help` lists every knob. `--env-kwarg KEY=VALUE` sets an arbitrary extra `EnvConfig`
field (python literal value), e.g. `--env-kwarg decision_hz_range=(28.0,34.0)` for the
low-rate stress runs E5 wants; unknown fields are reported and ignored rather than
crashing, so this harness does not break when the surrogate's config grows.

## Training on the measured VQ2 course

    python pilot/control/train/train.py --env surrogate --total-steps 20000000 \
        --n-envs 256 --n-steps 128 --frame-stack 6 --name vq2_run1 \
        --env-kwarg vq2_frac=0.8

`vq2_frac` is the share of episodes flown on the **measured** 17-gate VQ2 layout
(`../course/`) instead of a procedurally generated one. It defaults to `0` — the surrogate
is unchanged until you ask for it. Rationale, and the full measured-vs-assumed split, is in
`../surrogate/vq2course.py`; the short version is that the generator's shortest segment is
18 m while 10 of VQ2's 16 race edges are shorter than that, and the corners that decide
this course pair a sharp turn with a ~10 m exit — a combination the generator never draws.

### Training on VQ2 only

`vq2_frac=1.0` is a legitimate strategy — the course is fixed and deterministic (spec
§3.5), attempts are unlimited and ranking is on time (§9.4), so overfitting to it is the
point. Three things change, and one is easy to miss:

    # train
    --env-kwarg vq2_frac=1.0
    # rank checkpoints on the SAME course, or selection is meaningless
    python pilot/control/evalsuite/select.py --ckpt ... --vq2-frac 1.0

1. **Eval must match.** `select.py` / `run_policy.py` / `stress.py` default to procedural
   courses. Ranking a VQ2-only checkpoint against courses it never trained on selects the
   wrong checkpoint, quietly. Pass `--vq2-frac 1.0`.
2. **`difficulty` loses half its meaning.** It no longer anneals course geometry — that is
   now fixed — so it only drives perception noise, the collision margin and the corridor
   radius. `speed_cap` is unaffected. The curriculum still works, it just anneals less.
3. **You give up the gate-seeking safety net.** `course/README.md` warns that a whole leg
   can rotate by 90° if its mod-90 quadrant came from the sketch rather than a shared
   measurement — "a discrete failure a Gaussian cannot express", and the sampled envelope
   is "a lower bound". A pure-VQ2 policy has memorized a track that may be wrong in a way
   no amount of sampling covers. `vq2_frac=0.9` buys the insurance for ~10% of the run.

`vq2_alt_hypothesis_p` (default 0.5) covers the contested 1-2 edge: the accepted 8.32 m
rests on five rows from one flight and a refused 34-row channel reads 13.0 m. Every VQ2
episode flies 1→2, so the rival lands in ~25% of pool courses rather than betting the run
on one reading.

Keep some procedural share otherwise. The map cannot help a policy that is lost, so
gate-seeking has to survive alongside the track.

Other knobs (all `--env-kwarg`): `vq2_yaw_mode` (`'mixed'` randomizes across the three
disagreeing gate-yaw hypotheses, `'bisector'` fixes them), `vq2_pool_size`,
`vq2_floor_clear_m`, `vq2_headroom_m`, `vq2_tilt_deg`.

**On Colab:** open `colab_vq2.ipynb`. It clones the repo, runs both verification suites,
symlinks `checkpoints/` to Drive so a disconnect does not lose the run, and launches
training with the knobs as form fields. Use a CPU runtime — the env is vectorized NumPy and
dominates the step, so a GPU only speeds the PPO update. The win from Colab is running
several configs in parallel, not one run faster.

## Files

| file | what |
|---|---|
| `network.py` | 2x256 tanh MLP, separate policy/value trunks, tanh-squashed Gaussian head; thrust bias initialized at `interface.HOVER_THRUST`; `save_checkpoint` / `load_checkpoint` |
| `framestack.py` | the last k observations, oldest first. Shared with deployment |
| `normalize.py` | running obs mean/std (frozen into the checkpoint) + reward scaling |
| `ppo.py` | GAE(λ), clipped objective, clipped value loss, entropy bonus, minibatch epochs, grad clipping, KL early stop |
| `curriculum.py` | difficulty and `speed_cap` schedules on rolling completion rate; `enable_time_penalty` flip |
| `actionmap.py` | resolves `surrogate/actions.policy_to_action`, with a documented fallback and a divergence warning |
| `train.py` | entry point: CLI, curriculum loop, logging, checkpointing |
| `testenv.py` | throwaway 2-D point-mass stub exposing the `VecSurrogate` API. Test only |
| `selftest.py` | the verification above: stacking, normalization, hover init, curriculum, PPO learning, checkpoint round-trip |
| `colab_vq2.ipynb` | Colab runner: clone, verify, checkpoints to Drive, train on the measured VQ2 course |

## The three things that are easy to get silently wrong

**The observation pipeline order is `normalize -> stack`, not `stack -> normalize`.** That
is what makes `obs_mean` / `obs_std` 73-D and independent of k. Deployment must do the
same, in the same order, with the same class.

**The action mapping has exactly one home: `surrogate/actions.policy_to_action`.** This
harness imports it and never reimplements it (the fallback in `actionmap.py` exists only
so the harness runs before that module lands, warns loudly, and records
`action_map_source` in the checkpoint sidecar). `speed_cap` is applied HERE, on the two
rate channels only — capping thrust would move the trim point, which is not what "cap
commanded aggressiveness" means.

**Never weaken the collision penalty.** The env owns reward; the only reward-shaping lever
this harness pulls is `enable_time_penalty` (on only above ~80% completion) and, on a
detected stall, a *request* to weaken the progress term near gates via an optional
`progress_gate_scale` field — applied only if `EnvConfig` has one, ignored with a note
otherwise. Demotion in the curriculum can only undo prior promotions: `speed_cap` and
`difficulty` never go below their start values, because completion is legitimately zero
early in a run and shrinking authority there is the wrong direction.

## Checkpoint format

`checkpoints/<name>.pt` is a `torch.save` of a dict with exactly these keys:

| key | type |
|---|---|
| `model` | `state_dict` of `network.ActorCritic` |
| `obs_mean` | float32 `[73]` |
| `obs_std` | float32 `[73]` |
| `frame_stack` | int k |
| `net_config` | dict of `ActorCritic` constructor kwargs |
| `env_config` | dict of the `EnvConfig` fields used — valid as `EnvConfig(**d)` |
| `step` | int env steps |

`checkpoints/<name>.json` is the same minus `model`, plus a `harness` block (action-map
source, env kind, seed) for inspection without torch. Written on three schedules:
`<name>.pt` rolling latest, `<name>_s<step>.pt` every `--checkpoint-every` steps (this is
what E5 ranks over), `<name>_final.pt` at the end. Per-update metrics go to
`checkpoints/<name>_log.csv`.

Load side:

    from pilot.control.train.network import load_checkpoint
    from pilot.control.train.framestack import FrameStack
    from pilot.control.train.normalize import ObsNormalizer

    model, ckpt = load_checkpoint("pilot/control/train/checkpoints/run1_final.pt")
    norm = ObsNormalizer.from_arrays(ckpt["obs_mean"], ckpt["obs_std"])   # frozen
    stack = FrameStack(model.obs_dim, ckpt["frame_stack"], n_envs=1)
    stack.reset(norm(obs73))                     # at Policy.reset()
    u = model.infer(stack.push(norm(obs73)))     # [-1,1]^3
    physical = policy_to_action(u)               # surrogate/actions.py

`checkpoints/smoke256_final.pt` is a real 73-D checkpoint from a 400k-step smoke run — a
few minutes of training and it flies like it, but the format is the real thing, so develop
loaders against it.

`env_config` is stored JSON-flattened: tuples come back as lists and `EnvConfig.noise`
comes back as a plain dict, so `EnvConfig(**d)` alone would leave `noise` un-typed. Use

    from pilot.control.train.train import restore_env_config
    cfg = restore_env_config(ckpt["env_config"])   # -> a real EnvConfig, noise included

## Choices worth knowing about

* **Stacking, not GRU.** The architecture allows either; stacking has no hidden state to
  carry across a deployment reset or to get wrong in the recovery handback, which is worth
  more than the parameter saving under this deadline. `FrameStack` fills all k slots on
  reset rather than zero-padding, so episode starts are in-distribution.
* **Pre-tanh log-probabilities.** The sampled pre-tanh value is stored and scored, so PPO
  ratios are plain Gaussian ones; the tanh Jacobian cancels in the ratio and the entropy
  bonus uses the Gaussian entropy. Standard practice and numerically well-behaved.
* **Time-limit truncations bootstrap from `V(s_T)`; genuine terminals bootstrap at zero.**
  The env auto-resets, but the ending observation is not lost — it arrives as
  `info['terminal_obs']`, which `surrogate/env.py` emits for exactly this purpose.
  `ppo._truncation_values` reconstructs the terminal frame stack from the pre-step stack
  plus that observation and adds `gamma * V(s_T)` to the TD error at `info['timeout']`
  steps. A collision, corridor exit or finish still bootstraps at zero and wins when it
  coincides with a timeout, since `env.py` ORs the flags. An env that publishes neither
  key falls back to bootstrapping every `done` at zero.

  This README previously recorded the terminal observation as unrecoverable and accepted
  the bias. It was recoverable the whole time; the note is why nobody looked. Scale, so
  the fix is not oversold: at `run1`'s ~0.94 collision rate, truncations are at most ~6%
  of endings, and this is a correctness fix rather than a completion fix.
* **Curriculum changes rebuild the env** (config is a constructor argument; mutating it in
  place would not be guaranteed to reach precomputed internal state). Rebuilds are rate
  limited by `--curriculum-hold` updates and reseeded each time so courses stay fresh.
* **Completion rate** comes from `info['finished']`, which the surrogate provides, so it is
  exact. If that key ever disappears the harness falls back to
  `gates_passed >= info['n_gates']` and then to the approximate
  `gates_passed >= n_gates_range[0] and not collision`, warning loudly — on an 18–22 gate
  range the approximation counts a 22-gate course that reached gate 18 as complete, which
  would let the curriculum promote on a lie. The log line prints which definition is in
  use at startup.
