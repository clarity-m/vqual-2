"""T4 entry point. Run from the repo root:

    python pilot/control/train/train.py --env surrogate --total-steps 20000000
    python pilot/control/train/train.py --env testenv  --total-steps 300000   # harness test

Everything is a CLI flag; nothing is read from a config file. Checkpoints land in
`pilot/control/train/checkpoints/<name>.pt` with the JSON sidecar the eval/deployment
sibling reads.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pilot.control.train.derived import N_DERIVED  # noqa: E402
from pilot.control.train.actionmap import (ResidualThrustMap,  # noqa: E402
                                          resolve_action_map)
from pilot.control.train.curriculum import (  # noqa: E402
    Curriculum,
    CurriculumConfig,
    make_completion_fn,
)
from pilot.control.train.network import (  # noqa: E402
    ActorCritic,
    load_checkpoint,
    save_checkpoint,
)
from pilot.control.train.normalize import ObsNormalizer, RewardScaler  # noqa: E402
from pilot.control.train.ppo import PPO, PPOConfig  # noqa: E402
from pilot.interface import HOVER_THRUST, OBS_DIM  # noqa: E402

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


# --- environment construction -----------------------------------------------------------


def _set_if_present(cfg, name: str, value, warned: set) -> None:
    """Set an optional EnvConfig field, or warn once that the env has no such knob."""
    if hasattr(cfg, name):
        setattr(cfg, name, value)
    elif name not in warned:
        warned.add(name)
        print(f"[train] note: EnvConfig has no '{name}' field; request ignored", flush=True)


_WARNED_FIELDS: set = set()


def build_env(kind: str, n_envs: int, seed: int, spec: dict, extra: dict):
    """Construct the vectorized env for the current curriculum spec."""
    if kind == "surrogate":
        from pilot.control.surrogate.env import EnvConfig, VecSurrogate  # noqa: WPS433

        cfg = EnvConfig.curriculum(spec["difficulty"], spec["speed_cap"])
        maker = VecSurrogate
    elif kind == "testenv":
        from pilot.control.train.testenv import TestEnvConfig, VecPointMass

        cfg = TestEnvConfig.curriculum(spec["difficulty"], spec["speed_cap"])
        maker = VecPointMass
    else:
        raise ValueError(f"unknown --env {kind!r}")

    _set_if_present(cfg, "enable_time_penalty", spec["enable_time_penalty"], _WARNED_FIELDS)
    if spec.get("gates_per_episode") is not None:
        _set_if_present(cfg, "gates_per_episode", spec["gates_per_episode"], _WARNED_FIELDS)
    if spec.get("progress_gate_scale", 1.0) != 1.0:
        _set_if_present(cfg, "progress_gate_scale", spec["progress_gate_scale"], _WARNED_FIELDS)
    for k, v in extra.items():
        _set_if_present(cfg, k, v, _WARNED_FIELDS)

    return maker(n_envs=n_envs, config=cfg, seed=seed), cfg


def env_config_dict(cfg) -> dict:
    """The EnvConfig fields, JSON-safe, reconstructable as EnvConfig(**d)."""
    if dataclasses.is_dataclass(cfg):
        raw = dataclasses.asdict(cfg)
    else:
        raw = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}

    def conv(v):
        if isinstance(v, (tuple, list)):
            return [conv(x) for x in v]
        if isinstance(v, np.ndarray):
            return conv(v.tolist())
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    return {k: conv(v) for k, v in raw.items()}


def noise_scale_at(args, step: int):
    """The sensor-noise scale for this global step, or None if no ramp was asked for.

    Linear from `--noise-scale-start` at `--noise-ramp-from` to 1.0 (the full measured
    noise model) at `--noise-ramp-to`, flat outside. Steps are ABSOLUTE, which is what
    makes the ramp survive a resume: the scale is a pure function of `global_step`, so an
    interrupted run rejoins the schedule where it left off rather than restarting it.
    """
    if args.noise_ramp_to is None:
        return None if args.noise_scale_start >= 1.0 else float(args.noise_scale_start)
    a = int(args.noise_ramp_from or 0)
    b = int(args.noise_ramp_to)
    s0 = float(args.noise_scale_start)
    if b <= a:
        return 1.0 if step >= b else s0
    f = (float(step) - a) / float(b - a)
    return float(min(max(s0 + (1.0 - s0) * min(max(f, 0.0), 1.0), 0.0), 1.0))


def restore_env_config(d: dict, cls=None):
    """Rebuild an EnvConfig from a checkpoint's `env_config` dict.

    The checkpoint stores JSON-safe values, so tuples arrive as lists and nested
    dataclasses (`EnvConfig.noise`) arrive as plain dicts — `cls(**d)` alone would leave
    `noise` a dict and fail at the first attribute access. This restores both.
    """
    if cls is None:
        from pilot.control.surrogate.env import EnvConfig as cls  # noqa: N806

    proto = cls()
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        current = getattr(proto, f.name, None)
        if dataclasses.is_dataclass(current) and isinstance(v, dict):
            inner = {
                k: (tuple(x) if isinstance(x, list) else x)
                for k, x in v.items()
                if k in {g.name for g in dataclasses.fields(type(current))}
            }
            v = type(current)(**inner)
        elif isinstance(current, tuple) and isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)


# --- resume ------------------------------------------------------------------------------
#
# The seven-key checkpoint contract in network.py is frozen — `policies/rl.py` and the
# evalsuite code against it — so nothing here adds keys to the `.pt`. Everything a training
# run needs to continue and a deployment does not (optimizer moments, the normalizer's
# sample COUNT, the curriculum's counters, RNG) goes in a separate `<stem>_resume.pt`,
# written in the same call as the checkpoint so the two can never drift apart.

RESUME_FORMAT = 1

# Weight given to obs statistics recovered from a `.pt` alone, which carries mean and std
# but not the count they were computed from. Large enough that the restored statistics are
# not immediately washed out, small enough that they keep adapting — unlike
# `ObsNormalizer.from_arrays`, which freezes and fabricates 1e6 for the deployment path.
DEGRADED_OBS_COUNT = 1e5


def resume_sidecar_path(pt_path: Path) -> Path:
    return pt_path.with_name(f"{pt_path.stem}_resume.pt")


def resolve_checkpoint(spec: str) -> Path:
    """Accept a run name (`run1_s5046272`), a stem, or a path, with or without `.pt`."""
    p = Path(spec)
    seen, cands = set(), []
    for c in (p, p.with_suffix(".pt"), CHECKPOINT_DIR / p.name,
              CHECKPOINT_DIR / f"{p.name}.pt", CHECKPOINT_DIR / p.with_suffix(".pt").name):
        if c.suffix == ".pt" and str(c) not in seen:
            seen.add(str(c))
            cands.append(c)
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"--resume {spec!r}: no checkpoint found. Tried:\n  "
        + "\n  ".join(str(c) for c in cands)
    )


def _resume_state(curriculum, update: int, next_archive: int, n_rebuilds: int) -> dict:
    """The parts of the loop's state `_save` cannot read off the agent."""
    return {
        "curriculum": curriculum.state_dict(),
        "update": int(update),
        "next_archive": int(next_archive),
        "n_rebuilds": int(n_rebuilds),
    }


def _print_resume_banner(args, ckpt_path, ckpt_state, curriculum, degraded_levels,
                         global_step, start_update, total_updates, steps_per_update,
                         appending) -> None:
    """Say exactly what came back and what did not. A resume that quietly drops the
    optimizer moments and the curriculum window looks identical to one that does not,
    right up until the loss curve does something inexplicable."""
    remaining = max(0, total_updates - start_update)
    print(f"[resume] from {ckpt_path}", flush=True)
    print(f"[resume]   step {global_step:,} of {args.total_steps:,} target "
          f"-> {remaining} more updates ({remaining * steps_per_update:,} steps)")
    print(f"[resume]   curriculum: {curriculum.summary()} "
          f"progress_gate_scale={curriculum.progress_gate_scale:.2f}")
    print(f"[resume]   log {args.name}_log.csv: {'appending' if appending else 'new file'}")
    if ckpt_state is not None:
        print("[resume]   restored: weights, optimizer, obs+reward statistics, "
              "curriculum counters, RNG")
        return
    print(f"[resume]   NO RESUME SIDECAR beside {ckpt_path.name} "
          f"(expected {resume_sidecar_path(ckpt_path).name})")
    print("[resume]   restored ...... weights; obs mean/std from the .pt "
          f"(count seeded to {DEGRADED_OBS_COUNT:.0e}, unfrozen)")
    if degraded_levels:
        levels = " ".join(f"{k}={v}" for k, v in degraded_levels.items())
        print(f"[resume]   recovered ..... {levels} (from the checkpoint's env_config)")
    print("[resume]   LOST .......... Adam moments, reward-scaler statistics, "
          "curriculum window and counters, RNG")
    print("[resume]   The first updates will be noisier than a clean continuation.",
          flush=True)


def n_gates_hint(cfg) -> int | None:
    rng = getattr(cfg, "n_gates_range", None)
    try:
        return int(rng[0])
    except Exception:
        return None


# --- CLI ---------------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T4 PPO training harness")
    p.add_argument("--env", default="surrogate", choices=("surrogate", "testenv"))
    p.add_argument("--name", default="ppo")
    p.add_argument("--total-steps", type=int, default=20_000_000,
                   help="ABSOLUTE step target, not steps-to-add: resuming a 5M-step run "
                        "with 20000000 runs ~15M more, not 20M more")
    p.add_argument("--resume", default=None, metavar="NAME_OR_PATH",
                   help="continue from a checkpoint, e.g. --resume run1_s5046272. Uses the "
                        "matching <stem>_resume.pt when present; without it, weights and "
                        "obs statistics are restored and the loss is reported loudly")
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--frame-stack", type=int, default=6)
    p.add_argument("--frame-offsets", type=str, default=None,
                   help="comma-separated frame lags in DECISION STEPS, e.g. "
                        "'32,16,8,4,2,0'. Must have --frame-stack entries and include 0. "
                        "Default (unset) is consecutive steps, i.e. 0.111 s at k=6 and "
                        "55 Hz, which holds only ~3.3 distinct camera frames.")
    p.add_argument("--vertical-rate", action="store_true",
                   help="add 4 derived vertical-rate channels (leaky integrals of "
                        "world-frame vertical acceleration). interface.Observation is "
                        "unchanged at 73-D; the network sees 77-D. Measured R2 against "
                        "true v_z: 0.630, vs 0.460 for the shipping frame stack.")
    p.add_argument("--thrust-residual", action="store_true",
                   help="make the thrust action a residual about "
                        "hover/(cos roll cos pitch), so u[2]=0 holds altitude at any "
                        "attitude instead of the network learning that term from reward.")
    p.add_argument("--thrust-residual-span", type=float, default=0.30,
                   help="authority of the thrust residual either side of hover.")
    p.add_argument("--hidden", default="256,256")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--anneal-lr", action="store_true")
    # 0.99 at a 45-65 Hz decision rate is a ~1.8 s horizon, shorter than the ~2.8 s
    # between gates at cruise. Measured on `run1`: the aircraft committed to a ~2.1 m/s^2
    # sink and hit the floor ~2.4 s later, by which point the -8 terminal was discounted
    # to 0.27 of face value -- so "keep flying" was worth ~10 against a crash costing ~18,
    # and the policy was nearly indifferent to dying. At 0.997 (~6 s) those become ~33 and
    # ~41. This is the cheapest way to strengthen the effective collision penalty without
    # touching `k_collision`, which the architecture forbids weakening.
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03, help="<=0 disables the early stop")
    p.add_argument("--log-std-init", type=float, default=-0.5)
    p.add_argument("--no-reward-scaling", action="store_true")

    p.add_argument("--gates-per-episode", type=int, default=3,
                   help="STARTING episode length in gates; the curriculum grows it from "
                        "here. Each episode is cut after K gates as a truncation, not a "
                        "finish, and the course is still generated at full length. The "
                        "old behaviour is --gates-per-episode 22, which at 18-22 gates "
                        "makes completion per-gate-rate^20 and is why run1 logged 0.000 "
                        "for 77 straight updates")
    p.add_argument("--noise-scale-start", type=float, default=1.0,
                   help="sensor-noise strength at --noise-ramp-from. 1.0 is the measured "
                        "model (default, unchanged behaviour); 0.0 is a PERFECT sensor. "
                        "Architecture P3 warns clean detections are exploitable, so this "
                        "is for ramping, not for parking a run at.")
    p.add_argument("--noise-ramp-from", type=int, default=None,
                   help="ABSOLUTE global step where the noise ramp begins (default 0)")
    p.add_argument("--noise-ramp-to", type=int, default=None,
                   help="ABSOLUTE global step where noise reaches full strength. Absolute "
                        "so the ramp survives a resume instead of restarting.")
    p.add_argument("--difficulty-start", type=float, default=0.0)
    p.add_argument("--speed-cap-start", type=float, default=0.5)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--promote-rate", type=float, default=0.70)
    p.add_argument("--demote-rate", type=float, default=0.25)
    p.add_argument("--time-penalty-rate", type=float, default=0.80)
    p.add_argument("--curriculum-window", type=int, default=200)
    p.add_argument("--curriculum-hold", type=int, default=5)

    p.add_argument("--checkpoint-every", type=int, default=1_000_000,
                   help="env steps between archived checkpoints")
    p.add_argument("--log-every", type=int, default=1, help="updates between log lines")
    p.add_argument("--env-kwarg", action="append", default=[],
                   metavar="KEY=VALUE", help="extra EnvConfig field, python literal value")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _parse_env_kwargs(items) -> dict:
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--env-kwarg needs KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


# --- training ----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    extra_env = _parse_env_kwargs(args.env_kwarg)
    # The env's potential-based shaping must telescope against the SAME discount PPO
    # uses, or `gamma*PHI(s') - PHI(s)` stops being policy-preserving. An explicit
    # --env-kwarg still wins, so the coupling can be broken deliberately.
    extra_env.setdefault("gamma_shaping", args.gamma)
    hidden = tuple(int(h) for h in str(args.hidden).replace(" ", "").split(",") if h)

    # --- resume: load before anything is built, so the checkpoint governs the shapes ----
    ckpt = ckpt_net = ckpt_state = None
    ckpt_path = None
    if args.resume:
        ckpt_path = resolve_checkpoint(args.resume)
        ckpt_net, ckpt = load_checkpoint(ckpt_path, map_location=args.device, eval_mode=False)
        sidecar = resume_sidecar_path(ckpt_path)
        if sidecar.exists():
            ckpt_state = torch.load(sidecar, map_location=args.device, weights_only=False)
        # The weights decide the architecture; a CLI flag that disagrees is a mistake, not
        # an instruction. Loudly follow the checkpoint rather than crash on a stale default.
        ck_stack = int(ckpt.get("frame_stack", args.frame_stack))
        ck_hidden = tuple(int(h) for h in ckpt["net_config"].get("hidden", hidden))
        if ck_stack != args.frame_stack:
            print(f"[resume] --frame-stack {args.frame_stack} -> {ck_stack} (from checkpoint)",
                  flush=True)
            args.frame_stack = ck_stack
        if ck_hidden != hidden:
            print(f"[resume] --hidden {list(hidden)} -> {list(ck_hidden)} (from checkpoint)",
                  flush=True)
            hidden = ck_hidden

    curriculum = Curriculum(
        CurriculumConfig(
            difficulty_start=args.difficulty_start,
            speed_cap_start=args.speed_cap_start,
            promote_rate=args.promote_rate,
            demote_rate=args.demote_rate,
            time_penalty_on=args.time_penalty_rate,
            time_penalty_off=max(0.0, args.time_penalty_rate - 0.2),
            window=args.curriculum_window,
            hold_updates=args.curriculum_hold,
            gates_start=args.gates_per_episode,
            frozen=args.no_curriculum,
        )
    )

    # Curriculum state comes back before the env is built, so the env is constructed at the
    # difficulty the run had actually reached rather than at the CLI start values.
    n_rebuilds = 0
    degraded_levels: dict = {}
    if ckpt_state is not None and "curriculum" in ckpt_state:
        curriculum.load_state_dict(ckpt_state["curriculum"])
        n_rebuilds = int(ckpt_state.get("n_rebuilds", 0))
    elif ckpt is not None:
        env_c = ckpt.get("env_config") or {}
        degraded_levels = curriculum.seed_from(
            difficulty=env_c.get("difficulty"),
            speed_cap=env_c.get("speed_cap"),
            progress_gate_scale=env_c.get("progress_gate_scale"),
            enable_time_penalty=env_c.get("enable_time_penalty"),
        )

    env_seed = args.seed * 100003
    env, env_cfg = build_env(args.env, args.n_envs, env_seed + 7919 * n_rebuilds,
                             curriculum.env_spec(), extra_env)
    obs_dim = int(getattr(env, "obs_dim"))
    act_dim = int(getattr(env, "act_dim"))
    if args.env == "surrogate" and (obs_dim != OBS_DIM or act_dim != 3):
        raise RuntimeError(f"surrogate reports obs_dim={obs_dim} act_dim={act_dim}, "
                           f"expected {OBS_DIM} and 3")

    action_map = resolve_action_map()
    frame_offsets = None
    if args.frame_offsets:
        frame_offsets = [int(x) for x in str(args.frame_offsets).replace(" ", "").split(",") if x != ""]
    thrust_residual = None
    if args.thrust_residual:
        action_map = ResidualThrustMap(action_map, hover_thrust=HOVER_THRUST,
                                       span=args.thrust_residual_span)
        thrust_residual = {"hover_thrust": HOVER_THRUST,
                           "span": float(action_map.span),
                           "min_cos": float(action_map.min_cos)}
    # Under the residual map u[2]=0 already IS hover, so the head needs no bias at all.
    thrust_bias = (0.0 if thrust_residual is not None
                   else action_map.thrust_pre_tanh_bias(HOVER_THRUST))

    # The NETWORK's input width, which exceeds the env's when derived channels are on.
    # Everything downstream of the augmentation -- normalizer, stack, PPO buffers -- sizes
    # off this; only the env itself still speaks the raw 73-D contract.
    net_obs_dim = obs_dim + (N_DERIVED if args.vertical_rate else 0)

    if ckpt_net is not None:
        if ckpt_net.obs_dim != net_obs_dim or ckpt_net.act_dim != act_dim:
            extra = (f" (env reports {obs_dim} plus {N_DERIVED} derived channels)"
                     if args.vertical_rate else "")
            raise RuntimeError(
                f"--resume {ckpt_path}: checkpoint network is "
                f"obs_dim={ckpt_net.obs_dim} act_dim={ckpt_net.act_dim}, but this run "
                f"expects {net_obs_dim}/{act_dim}{extra}. Resuming across a different "
                f"observation contract would load weights into the wrong channels."
            )
        net = ckpt_net.to(args.device)
        net.train()
        # `_frame_offsets` and `_thrust_residual` describe the architecture the loaded
        # WEIGHTS were trained under, so on resume the checkpoint's values win over the
        # CLI's; overwriting them would label the network as something it is not, which
        # is the same class of error the dimension check above exists to catch. They are
        # only filled in here when the checkpoint predates them.
        if getattr(net, "_frame_offsets", None) is None:
            net._frame_offsets = frame_offsets
        if getattr(net, "_thrust_residual", None) is None:
            net._thrust_residual = thrust_residual
        if getattr(net, "_vertical_rate", None) is None:
            net._vertical_rate = bool(args.vertical_rate)
    else:
        net = ActorCritic(
            obs_dim=net_obs_dim,
            frame_stack=args.frame_stack,
            act_dim=act_dim,
            hidden=hidden,
            log_std_init=args.log_std_init,
            thrust_bias=thrust_bias,
            hover_thrust=HOVER_THRUST,
            thrust_floor=action_map.thrust_floor,
        ).to(args.device)
        net._frame_offsets = frame_offsets
        net._thrust_residual = thrust_residual
        net._vertical_rate = bool(args.vertical_rate)

    ppo_cfg = PPOConfig(
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        lr=args.lr,
        epochs=args.epochs,
        n_minibatches=args.minibatches,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl if args.target_kl and args.target_kl > 0 else None,
    )
    obs_norm = ObsNormalizer(net_obs_dim)
    rew_scaler = RewardScaler(args.n_envs, gamma=args.gamma)
    rew_scaler.enabled = not args.no_reward_scaling

    agent = PPO(
        net=net,
        obs_dim=net_obs_dim,
        frame_stack=args.frame_stack,
        n_envs=args.n_envs,
        cfg=ppo_cfg,
        action_map=action_map,
        frame_offsets=frame_offsets,
        vertical_rate=args.vertical_rate,
        obs_norm=obs_norm,
        rew_scaler=rew_scaler,
        device=args.device,
        speed_cap=curriculum.speed_cap,
        completion_fn=make_completion_fn(n_gates_hint(env_cfg), verbose=not args.quiet),
    )
    agent.set_env(env)

    steps_per_update = args.n_steps * args.n_envs
    total_updates = max(1, args.total_steps // steps_per_update)
    next_archive = args.checkpoint_every
    start_update = 0

    if ckpt is not None:
        agent.global_step = int(ckpt.get("step", 0))
        if ckpt_state is not None:
            agent.opt.load_state_dict(ckpt_state["opt"])
            agent.obs_norm.rms.load_state_dict(ckpt_state["obs_rms"])
            agent.rew_scaler.rms.load_state_dict(ckpt_state["rew_rms"])
            agent.global_step = int(ckpt_state.get("global_step", agent.global_step))
            torch.set_rng_state(ckpt_state["torch_rng"].cpu().to(torch.uint8))
            np.random.set_state(ckpt_state["numpy_rng"])
        else:
            # Weights-only. `obs_mean`/`obs_std` are in the .pt but the count they came
            # from is not, so it is seeded rather than invented at 1e6 — and left
            # unfrozen, so the statistics keep tracking.
            rms = agent.obs_norm.rms
            rms.mean = np.asarray(ckpt["obs_mean"], dtype=np.float64).reshape(-1).copy()
            std = np.asarray(ckpt["obs_std"], dtype=np.float64).reshape(-1)
            rms.var = np.maximum(std ** 2 - agent.obs_norm.epsilon, 0.0)
            rms.count = DEGRADED_OBS_COUNT

        # Derived from the step count, not from a stored update index, so a resume with a
        # different --n-envs / --n-steps still lands in the right place on the schedule.
        start_update = int(agent.global_step // steps_per_update)
        if args.checkpoint_every > 0:
            next_archive = ((agent.global_step // args.checkpoint_every) + 1) * args.checkpoint_every
        agent.speed_cap = curriculum.speed_cap

    resumed_log = ckpt is not None
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CHECKPOINT_DIR / f"{args.name}_log.csv"
    log_fields = [
        "step", "update", "sps", "ep_return", "ep_len", "gates", "collision_rate",
        "collisions_per_ep", "gate_rate", "gates_per_episode", "noise_scale",
        "completion_rate", "difficulty", "speed_cap", "time_penalty", "policy_loss",
        "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var", "reward_scale",
    ]
    # Append on resume: opening "w" here would silently destroy the history of the run
    # being continued (run1_log.csv is 77 updates of the only training evidence there is).
    append = resumed_log and log_path.exists() and log_path.stat().st_size > 0
    if append:
        # Adopt the EXISTING header rather than this build's. A column added since the run
        # started would otherwise write rows one field wider than the header they sit
        # under, which reads as a silent off-by-one in every column after it. Dropping the
        # new column for the rest of a resumed run is the recoverable failure.
        with log_path.open("r", newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), None)
        if existing:
            missing = [c for c in log_fields if c not in existing]
            if missing and not args.quiet:
                print(f"[train] {log_path.name}: appending under its original header; "
                      f"not logging {', '.join(missing)}", flush=True)
            log_fields = existing
    log_file = log_path.open("a" if append else "w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields, extrasaction="ignore")
    if not append:
        log_writer.writeheader()

    if not args.quiet:
        print(
            f"[train] env={args.env} obs_dim={obs_dim} act_dim={act_dim} "
            f"stack={args.frame_stack} n_envs={args.n_envs} n_steps={args.n_steps} "
            f"updates={total_updates} action_map={action_map.source} "
            f"thrust_bias={thrust_bias:+.4f} -> thrust {action_map.thrust_of(np.tanh(thrust_bias)):.3f}",
            flush=True,
        )
    if ckpt is not None:
        _print_resume_banner(args, ckpt_path, ckpt_state, curriculum, degraded_levels,
                             agent.global_step, start_update, total_updates,
                             steps_per_update, append)

    history: list[dict] = []
    t_start = time.perf_counter()
    update = start_update

    for update in range(start_update + 1, total_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / total_updates
            for g in agent.opt.param_groups:
                g["lr"] = args.lr * frac

        # Sensor-noise ramp. Anchored on the ABSOLUTE global step, not on where this
        # process happened to start, so a resume lands at the same point on the ramp that
        # an uninterrupted run would have -- a ramp measured from the resume point would
        # restart every disconnect and the policy would see clean detections again.
        # Mutating the live config is enough: `_reset_envs` re-draws noise per episode.
        ns = noise_scale_at(args, agent.global_step)
        if ns is not None:
            env_cfg.noise_scale = ns

        stats = agent.collect()
        losses = agent.update()
        curriculum.record(stats.completions, stats.gate_passes, stats.gate_attempts)

        if curriculum.step():
            n_rebuilds += 1
            spec = curriculum.env_spec()
            env, env_cfg = build_env(
                args.env, args.n_envs, env_seed + 7919 * n_rebuilds, spec, extra_env
            )
            agent.set_env(env)
            agent.speed_cap = spec["speed_cap"]
            if not args.quiet:
                print(f"[curriculum] update {update}: {curriculum.summary()}", flush=True)

        row = {
            "step": agent.global_step,
            "update": update,
            "sps": round(stats.sps, 1),
            "ep_return": round(float(np.mean(stats.episode_returns)), 3) if stats.episode_returns else "",
            "ep_len": round(float(np.mean(stats.episode_lengths)), 1) if stats.episode_lengths else "",
            "gates": round(float(np.mean(stats.gates_passed)), 2) if stats.gates_passed else "",
            "collision_rate": round(float(np.mean(stats.collisions)), 3) if stats.collisions else "",
            # Collisions per episode. Prefer this to `collision_rate` whenever contact is
            # non-terminal: the flag above is sampled on `done` and reads ~0 in that mode.
            "collisions_per_ep": (round(float(np.mean(stats.collision_counts)), 3)
                                  if stats.collision_counts else ""),
            "gate_rate": round(curriculum.gate_rate_raw, 4),
            "noise_scale": round(float(getattr(env_cfg, "noise_scale", 1.0)), 4),
            "gates_per_episode": curriculum.gates_per_episode,
            "completion_rate": round(curriculum.completion_rate_raw, 3),
            "difficulty": round(curriculum.difficulty, 3),
            "speed_cap": round(curriculum.speed_cap, 3),
            "time_penalty": int(curriculum.enable_time_penalty),
            "policy_loss": round(losses["policy_loss"], 5),
            "value_loss": round(losses["value_loss"], 5),
            "entropy": round(losses["entropy"], 4),
            "approx_kl": round(losses["approx_kl"], 5),
            "clip_frac": round(losses["clip_frac"], 4),
            "explained_var": round(losses["explained_var"], 4),
            "reward_scale": round(losses["reward_scale"], 4),
        }
        history.append(row)
        log_writer.writerow(row)
        log_file.flush()

        if not args.quiet and (update % max(1, args.log_every) == 0 or update == total_updates):
            print(
                f"[{update:5d}/{total_updates}] step={row['step']:>9} "
                f"fps={row['sps']:>8} ret={row['ep_return']!s:>8} "
                f"gates={row['gates']!s:>6} "
                # `coll` is the terminal-flag rate while contact ends an episode, and
                # collisions-per-episode once it does not. Same column, right meaning.
                f"coll={(row['collisions_per_ep'] if row['collisions_per_ep'] != '' else row['collision_rate'])!s:>5} "
                f"grate={row['gate_rate']:.3f} compl={row['completion_rate']:.2f} "+ (f"nz={row['noise_scale']:.2f} " if row['noise_scale'] < 1.0 else "") + "| "
                f"{curriculum.summary()} | "
                f"pl={row['policy_loss']:+.4f} vl={row['value_loss']:.4f} "
                f"ent={row['entropy']:.3f} kl={row['approx_kl']:.4f} "
                f"ev={row['explained_var']:+.3f}",
                flush=True,
            )

        if agent.global_step >= next_archive:
            next_archive += args.checkpoint_every
            _save(args, agent, net, env_cfg, action_map, tag=f"_s{agent.global_step}",
                  resume_state=_resume_state(curriculum, update, next_archive, n_rebuilds))
        if update % 10 == 0 or update == total_updates:
            _save(args, agent, net, env_cfg, action_map,  # rolling latest
                  resume_state=_resume_state(curriculum, update, next_archive, n_rebuilds))

    final = _save(args, agent, net, env_cfg, action_map, tag="_final",
                  resume_state=_resume_state(curriculum, update, next_archive, n_rebuilds))
    log_file.close()

    elapsed = time.perf_counter() - t_start
    # Steps run in THIS session, not the absolute count: dividing a resumed run's 5M total
    # by the few minutes it just spent reports a fictional throughput.
    ran = max(0, update - start_update) * steps_per_update
    if not args.quiet:
        print(
            f"[train] done: {ran} steps in {elapsed:.1f}s "
            f"({ran / max(elapsed, 1e-9):.0f} FPS overall); "
            f"total {agent.global_step}; checkpoint {final}",
            flush=True,
        )
    return {"history": history, "checkpoint": str(final), "curriculum": curriculum,
            "agent": agent, "elapsed_s": elapsed}


def _save(args, agent: PPO, net: ActorCritic, env_cfg, action_map, tag: str = "",
          resume_state: dict | None = None) -> Path:
    pt_path = save_checkpoint(
        CHECKPOINT_DIR / f"{args.name}{tag}",
        net,
        agent.obs_norm.mean,
        agent.obs_norm.std,
        agent.stack.k,
        env_config_dict(env_cfg),
        agent.global_step,
        frame_offsets=getattr(net, "_frame_offsets", None),
        thrust_residual=getattr(net, "_thrust_residual", None),
        vertical_rate=bool(getattr(net, "_vertical_rate", False)),
        extra_json={
            "env_kind": args.env,
            "action_map_source": action_map.source,
            "rate_cap_rps": action_map.rate_cap,
            "thrust_floor": action_map.thrust_floor,
            "obs_clip": agent.obs_norm.clip,
            "run_name": args.name,
            "seed": args.seed,
        },
    )
    if resume_state is not None:
        # Same call as the checkpoint, deliberately: a resume sidecar that can be newer or
        # older than the weights beside it is worse than none at all.
        torch.save(
            {
                "format": RESUME_FORMAT,
                "global_step": int(agent.global_step),
                "opt": agent.opt.state_dict(),
                "obs_rms": agent.obs_norm.rms.state_dict(),
                "rew_rms": agent.rew_scaler.rms.state_dict(),
                "rew_enabled": bool(agent.rew_scaler.enabled),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "args": {k: v for k, v in vars(args).items()},
                **resume_state,
            },
            resume_sidecar_path(pt_path),
        )
    return pt_path


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except ImportError as exc:
        if args.env == "surrogate":
            print(
                f"[train] cannot import the surrogate environment ({exc}).\n"
                f"        pilot/control/surrogate/ is built by a sibling agent; until it "
                f"lands, verify the harness with:\n"
                f"        python pilot/control/train/train.py --env testenv "
                f"--total-steps 300000 --n-envs 64 --n-steps 64",
                file=sys.stderr,
            )
            return 2
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
