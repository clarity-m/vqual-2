"""E5: fly an `interface.Policy` over held-out course seeds on the surrogate.

    python pilot/control/evalsuite/run_policy.py --seeds 0-19 --difficulty 0.2
    python pilot/control/evalsuite/run_policy.py --policy rl --ckpt run3_step2M \
        --seeds 100-149 --supervisor --json out/run3.json

One seed is one held-out draw: `SingleSurrogate(config, seed)` seeds the course AND the
per-episode randomization (plant parameters, detection noise, control period, collision
margin) together, so a seed is a whole world, not just a track. Nothing here is
stochastic outside that seed, so a seed list plus a config is a reproducible number.

WHAT COUNTS AS A RESULT. The surrogate's info dict is not part of the frozen contract,
so the numbers that matter are read from the OBSERVATION, whose `RaceObs` is frozen:
gates passed from `active_gate_index`, collisions from `collision_episodes`, time from
`race_time_s`. The info dict is consulted only for termination and for a completion
flag, tolerantly, under any of the names a reasonable implementation might use. Episodes
can start mid-course (P2 randomizes starts so the recovery handback is in distribution),
so gates passed is a DELTA from the first observation, never the raw index.

Completion rate is the primary number, mean time the secondary one, in that order,
because the leaderboard counts completed runs first and fast runs second.
"""

import argparse
import json
import os
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot.control.policies.baseline import BaselinePolicy, gains_from_str  # noqa: E402
from pilot.control.policies.supervisor import RecoverySupervisor            # noqa: E402

DONE_KEYS = ("done", "terminated", "termination", "episode_done", "finished_episode")
WIN_KEYS = ("finished", "completed", "complete", "success", "course_complete")
# In the order they are reported, most specific first.
REASON_FLAGS = (("finished", "completed"), ("collision", "collision"),
                ("corridor_exit", "corridor"), ("timeout", "timeout"),
                ("crashed", "collision"), ("out_of_bounds", "corridor"),
                ("truncated", "truncated"))
GATE_KEYS = ("gates_passed", "gates", "n_passed")


# -- the surrogate, imported late so this module loads before the sibling exists ----
def load_surrogate():
    try:
        from pilot.control.surrogate.single import SingleSurrogate
        from pilot.control.surrogate.env import EnvConfig
    except Exception as exc:
        raise RuntimeError(
            "pilot/control/surrogate is not importable yet (%s: %s). The evaluation "
            "suite is wired against SingleSurrogate(config, seed) / EnvConfig."
            % (type(exc).__name__, exc))
    return SingleSurrogate, EnvConfig


class _Env:
    """`SingleSurrogate`, with a shim around one transient sibling bug.

    2026-08-01: `VecSurrogate.step` grew an info entry `terminal_obs` of shape
    [n, 73], and `SingleSurrogate.step` scalarizes every info value with `v[0].item()`,
    which raises on a 73-vector. The shim re-does the sibling's OWN two-line adapter
    (their `vec.step` plus their `build_observation`) and scalarizes only what is
    actually scalar -- it duplicates no dynamics, no detections and no attention. It
    prints once and stops being used the moment `single.py` handles the array itself.
    """

    warned = False

    def __init__(self, config, seed):
        SingleSurrogate, _ = load_surrogate()
        self.env = SingleSurrogate(config, seed)
        self.shim = False

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if not self.shim:
            try:
                return self.env.step(action)
            except (ValueError, TypeError) as exc:
                self.shim = True
                if not _Env.warned:
                    _Env.warned = True
                    print("  note: SingleSurrogate.step raised %s(%s); using the "
                          "evalsuite info shim" % (type(exc).__name__, exc))
        from pilot.control.surrogate.env import build_observation
        import numpy as np
        a = np.array([[float(action.roll_rate), float(action.pitch_rate),
                       float(action.thrust)]], dtype=np.float32)
        _, rew, done, info = self.env.vec.step(a)
        obs = build_observation(self.env.vec._fields, 0)
        out = {k: (v[0].item() if getattr(v, "ndim", 1) == 1 else v[0])
               for k, v in info.items()}
        out["reward"] = float(rew[0])
        out["done"] = bool(done[0])
        return obs, out

    def __getattr__(self, name):
        return getattr(self.env, name)


def override(cfg, **kw):
    """Set fields on an `EnvConfig` without caring whether it is frozen."""
    kw = {k: v for k, v in kw.items() if v is not None}
    if not kw:
        return cfg
    import dataclasses
    try:
        return dataclasses.replace(cfg, **kw)
    except Exception:
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg


def build_config(difficulty=0.2, speed_cap=1.0, decision_hz=None, n_gates=None,
                 time_penalty=None):
    _, EnvConfig = load_surrogate()
    cfg = EnvConfig.curriculum(float(difficulty), float(speed_cap))
    return override(cfg, decision_hz_range=decision_hz, n_gates_range=n_gates,
                    enable_time_penalty=time_penalty)


def config_dict(cfg):
    import dataclasses
    try:
        return dataclasses.asdict(cfg)
    except Exception:
        return {"repr": repr(cfg)}


# -- one episode -------------------------------------------------------------------
def _flag(info, keys):
    for k in keys:
        if k in info and bool(info[k]):
            return True
    return False


def run_episode(policy, config, seed, max_steps=5000):
    """Fly one seed to termination.

    THE OBSERVATION RETURNED BY THE TERMINATING STEP BELONGS TO THE NEXT EPISODE.
    `VecSurrogate` auto-resets, which is right for PPO and a trap for scoring: read the
    race packet after a crash and you get the fresh episode's spawn index, which on a
    randomized mid-course start is a fictitious handful of free gates. So the episode's
    numbers come from the info dict (computed pre-reset) and, failing that, from the
    last observation handed out BEFORE the terminating step.
    """
    env = _Env(config, seed)
    obs = env.reset()
    policy.reset()

    gate0 = int(obs.race.active_gate_index)
    coll0 = int(obs.race.collision_episodes)
    race0 = float(obs.race.race_time_s)
    sim_s = 0.0
    steps = 0
    ended = False
    info = {}
    last = obs
    t0 = time.time()

    for steps in range(1, max_steps + 1):
        last = obs
        obs, raw = env.step(policy(last))
        info = dict(raw) if isinstance(raw, dict) else {}
        sim_s += float(last.dt_s)
        if _flag(info, DONE_KEYS) or _flag(info, WIN_KEYS):
            ended = True
            break
        if int(last.race.active_gate_index) >= int(last.race.n_gates_total) > 0:
            ended = True
            info.setdefault("finished", True)
            break

    final = last if ended else obs
    race = final.race
    completed = _flag(info, WIN_KEYS)
    reason = next((label for key, label in REASON_FLAGS if bool(info.get(key))),
                  "max_steps" if not ended else "terminated")
    # `gates_passed` is the ABSOLUTE active index, and episodes can start mid-course, so
    # the episode's own achievement is always a delta from the spawn index.
    gates = next((int(info[k]) for k in GATE_KEYS if k in info),
                 int(race.active_gate_index)) - gate0
    race_s = float(race.race_time_s) - race0
    return dict(
        seed=int(seed),
        completed=bool(completed),
        gates=int(gates),
        gates_needed=max(int(race.n_gates_total) - gate0, 1),
        gate_index=gate0 + int(gates),
        start_index=gate0,
        n_gates=int(race.n_gates_total),
        time_s=round(race_s if race_s > 1e-9 else sim_s, 3),
        collisions=int(race.collision_episodes) - coll0,
        recoveries=int(getattr(policy, "recoveries", 0)),
        steps=int(steps),
        reason=reason,
        wall_s=round(time.time() - t0, 2),
    )


# -- a sweep of seeds ---------------------------------------------------------------
def evaluate(policy, config, seeds, max_steps=5000, progress=None):
    out = []
    for s in seeds:
        r = run_episode(policy, config, s, max_steps=max_steps)
        out.append(r)
        if progress:
            progress(r)
    return out


def summarize(results):
    if not results:
        return {}
    done = [r for r in results if r["completed"]]
    frac = [r["gates"] / max(r.get("gates_needed", r["n_gates"]), 1) for r in results]
    s = dict(
        n=len(results),
        completion_rate=round(len(done) / len(results), 4),
        gates_mean=round(statistics.fmean(r["gates"] for r in results), 2),
        gates_median=statistics.median(r["gates"] for r in results),
        course_frac_mean=round(statistics.fmean(frac), 4),
        collisions_mean=round(statistics.fmean(r["collisions"] for r in results), 3),
        crash_rate=round(sum(1 for r in results if r["reason"] in
                             ("collision", "corridor")) / len(results), 4),
        steps_mean=round(statistics.fmean(r["steps"] for r in results), 1),
        time_mean_completed=(round(statistics.fmean(r["time_s"] for r in done), 3)
                             if done else None),
        time_mean_all=round(statistics.fmean(r["time_s"] for r in results), 3),
    )
    return s


def print_results(name, results, summary, per_seed=True):
    if per_seed:
        print("\n%-6s %-4s %5s %8s %6s %6s %7s  %s"
              % ("seed", "done", "gates", "time_s", "coll", "recov", "steps", "reason"))
        for r in results:
            print("%-6d %-4s %5d %8.2f %6d %6d %7d  %s"
                  % (r["seed"], "yes" if r["completed"] else "no", r["gates"],
                     r["time_s"], r["collisions"], r["recoveries"], r["steps"],
                     r["reason"]))
    print("\n%s over %d seeds" % (name, summary.get("n", 0)))
    print("  completion rate   %.3f" % summary.get("completion_rate", 0.0))
    print("  gates passed      mean %.2f  median %s"
          % (summary.get("gates_mean", 0.0), summary.get("gates_median", 0)))
    print("  course fraction   %.3f" % summary.get("course_frac_mean", 0.0))
    tm = summary.get("time_mean_completed")
    print("  time (completed)  %s" % ("%.2f s" % tm if tm is not None else "-- none --"))
    print("  collisions/run    %.2f" % summary.get("collisions_mean", 0.0))
    print("  crash rate        %.3f" % summary.get("crash_rate", 0.0))


# -- CLI ------------------------------------------------------------------------------
def parse_seeds(spec):
    """`"0-19,30,40-42"` -> a list of ints, in the order written."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_pair(spec):
    if spec is None:
        return None
    a, _, b = str(spec).partition(",")
    return (float(a), float(b))


def make_policy(kind="baseline", ckpt=None, gains="", supervisor=False, device="cpu"):
    """(policy, name). `gains` tunes the baseline from the command line."""
    if kind == "baseline":
        policy = BaselinePolicy(**gains_from_str(gains))
        name = "baseline" + (" [%s]" % gains if gains else "")
    elif kind == "rl":
        if not ckpt:
            raise SystemExit("--policy rl needs --ckpt")
        from pilot.control.policies.rl import RLPolicy
        policy = RLPolicy(ckpt, device=device)
        name = "rl:%s" % os.path.basename(policy.path)
        print("  %s" % policy.info)
    else:
        raise SystemExit("unknown --policy %r" % kind)
    if supervisor:
        policy = RecoverySupervisor(policy)
        name += " +supervisor"
    return policy, name


def add_common_args(ap):
    ap.add_argument("--seeds", default="0-19", help="e.g. 0-19,30,40-42")
    ap.add_argument("--difficulty", type=float, default=0.2)
    ap.add_argument("--speed-cap", type=float, default=1.0)
    ap.add_argument("--decision-hz", default=None, help="override, e.g. 45,65")
    ap.add_argument("--n-gates", default=None, help="override, e.g. 18,22")
    ap.add_argument("--time-penalty", dest="time_penalty", action="store_true",
                    default=None)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--quiet", action="store_true", help="summary only")


def add_policy_args(ap):
    ap.add_argument("--policy", default="baseline", choices=("baseline", "rl"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gains", default="", help='baseline tuning, "k_roll=7,v_cruise=6"')
    ap.add_argument("--supervisor", action="store_true",
                    help="wrap in the D6 recovery supervisor")
    ap.add_argument("--device", default="cpu")


def write_json(path, blob):
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    print("\nwrote %s" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_policy_args(ap)
    add_common_args(ap)
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    cfg = build_config(args.difficulty, args.speed_cap,
                       decision_hz=parse_pair(args.decision_hz),
                       n_gates=parse_pair(args.n_gates),
                       time_penalty=args.time_penalty)
    policy, name = make_policy(args.policy, args.ckpt, args.gains, args.supervisor,
                               args.device)
    print("%s | difficulty %.2f speed_cap %.2f | %d seeds"
          % (name, args.difficulty, args.speed_cap, len(seeds)))

    results = evaluate(policy, cfg, seeds, max_steps=args.max_steps)
    summary = summarize(results)
    print_results(name, results, summary, per_seed=not args.quiet)
    write_json(args.json, dict(policy=name, config=config_dict(cfg), seeds=seeds,
                               results=results, summary=summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
