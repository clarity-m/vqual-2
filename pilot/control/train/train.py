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

from pilot.control.train.actionmap import (ResidualThrustMap,  # noqa: E402
                                          resolve_action_map)
from pilot.control.train.curriculum import (  # noqa: E402
    Curriculum,
    CurriculumConfig,
    make_completion_fn,
)
from pilot.control.train.network import ActorCritic, save_checkpoint  # noqa: E402
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
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--frame-stack", type=int, default=6)
    p.add_argument("--frame-offsets", type=str, default=None,
                   help="comma-separated frame lags in DECISION STEPS, e.g. "
                        "'32,16,8,4,2,0'. Must have --frame-stack entries and include 0. "
                        "Default (unset) is consecutive steps, i.e. 0.111 s at k=6 and "
                        "55 Hz, which holds only ~3.3 distinct camera frames.")
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
    p.add_argument("--gamma", type=float, default=0.99)
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
    hidden = tuple(int(h) for h in str(args.hidden).replace(" ", "").split(",") if h)

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
            frozen=args.no_curriculum,
        )
    )

    env_seed = args.seed * 100003
    env, env_cfg = build_env(args.env, args.n_envs, env_seed, curriculum.env_spec(), extra_env)
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

    net = ActorCritic(
        obs_dim=obs_dim,
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
    obs_norm = ObsNormalizer(obs_dim)
    rew_scaler = RewardScaler(args.n_envs, gamma=args.gamma)
    rew_scaler.enabled = not args.no_reward_scaling

    agent = PPO(
        net=net,
        obs_dim=obs_dim,
        frame_stack=args.frame_stack,
        n_envs=args.n_envs,
        cfg=ppo_cfg,
        action_map=action_map,
        frame_offsets=frame_offsets,
        obs_norm=obs_norm,
        rew_scaler=rew_scaler,
        device=args.device,
        speed_cap=curriculum.speed_cap,
        completion_fn=make_completion_fn(n_gates_hint(env_cfg), verbose=not args.quiet),
    )
    agent.set_env(env)

    steps_per_update = args.n_steps * args.n_envs
    total_updates = max(1, args.total_steps // steps_per_update)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CHECKPOINT_DIR / f"{args.name}_log.csv"
    log_fields = [
        "step", "update", "sps", "ep_return", "ep_len", "gates", "collision_rate",
        "completion_rate", "difficulty", "speed_cap", "time_penalty", "policy_loss",
        "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var", "reward_scale",
    ]
    log_file = log_path.open("w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    if not args.quiet:
        print(
            f"[train] env={args.env} obs_dim={obs_dim} act_dim={act_dim} "
            f"stack={args.frame_stack} n_envs={args.n_envs} n_steps={args.n_steps} "
            f"updates={total_updates} action_map={action_map.source} "
            f"thrust_bias={thrust_bias:+.4f} -> thrust {action_map.thrust_of(np.tanh(thrust_bias)):.3f}",
            flush=True,
        )

    history: list[dict] = []
    n_rebuilds = 0
    next_archive = args.checkpoint_every
    t_start = time.perf_counter()

    for update in range(1, total_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / total_updates
            for g in agent.opt.param_groups:
                g["lr"] = args.lr * frac

        stats = agent.collect()
        losses = agent.update()
        curriculum.record(stats.completions)

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
                f"gates={row['gates']!s:>6} coll={row['collision_rate']!s:>5} "
                f"compl={row['completion_rate']:.2f} | {curriculum.summary()} | "
                f"pl={row['policy_loss']:+.4f} vl={row['value_loss']:.4f} "
                f"ent={row['entropy']:.3f} kl={row['approx_kl']:.4f} "
                f"ev={row['explained_var']:+.3f}",
                flush=True,
            )

        if agent.global_step >= next_archive:
            next_archive += args.checkpoint_every
            _save(args, agent, net, env_cfg, action_map, tag=f"_s{agent.global_step}")
        if update % 10 == 0 or update == total_updates:
            _save(args, agent, net, env_cfg, action_map)  # rolling latest

    final = _save(args, agent, net, env_cfg, action_map, tag="_final")
    log_file.close()

    elapsed = time.perf_counter() - t_start
    if not args.quiet:
        print(
            f"[train] done: {agent.global_step} steps in {elapsed:.1f}s "
            f"({agent.global_step / max(elapsed, 1e-9):.0f} FPS overall); "
            f"checkpoint {final}",
            flush=True,
        )
    return {"history": history, "checkpoint": str(final), "curriculum": curriculum,
            "agent": agent, "elapsed_s": elapsed}


def _save(args, agent: PPO, net: ActorCritic, env_cfg, action_map, tag: str = "") -> Path:
    return save_checkpoint(
        CHECKPOINT_DIR / f"{args.name}{tag}",
        net,
        agent.obs_norm.mean,
        agent.obs_norm.std,
        agent.stack.k,
        env_config_dict(env_cfg),
        agent.global_step,
        frame_offsets=getattr(net, "_frame_offsets", None),
        thrust_residual=getattr(net, "_thrust_residual", None),
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
