"""Throwaway: coordinate sweep over baseline gains against a fixed seed block.

    python pilot/control/evalsuite/_probe_sweep.py score "k_roll=8"          # one point
    python pilot/control/evalsuite/_probe_sweep.py sweep k_bearing 1.1,1.6,2.2,3.0
    python pilot/control/evalsuite/_probe_sweep.py descend                   # full pass

The score is the fraction of the course flown, averaged over seeds, with completion as
the tiebreak -- NOT raw gates passed, because episodes start at randomized points on
courses of different lengths and a raw count rewards the seeds that happened to start
late. Seeds are held fixed across every comparison, so two scores are paired.
"""

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

from pilot.control.policies.baseline import BaselinePolicy, gains_from_str  # noqa: E402
from pilot.control.policies.supervisor import RecoverySupervisor            # noqa: E402
from pilot.control.evalsuite.run_policy import build_config, _Env           # noqa: E402

SEEDS = list(range(24))
DIFFS = (0.0, 0.35)
MAX_STEPS = 5000


def fly(seed, gains, difficulty, supervisor=False, max_steps=MAX_STEPS):
    env = _Env(build_config(difficulty, 1.0), seed)
    obs = env.reset()
    p = BaselinePolicy(**gains)
    pol = RecoverySupervisor(p) if supervisor else p
    pol.reset()
    start = int(obs.race.active_gate_index)
    need = max(1, int(obs.race.n_gates_total) - start)
    for _ in range(max_steps):
        obs, info = env.step(pol(obs))
        if info["done"]:
            gates = int(info["gates_passed"]) - start
            return (min(1.0, gates / need), bool(info["finished"]),
                    float(info["episode_length"]))
    return 0.0, False, 0.0


def _job(arg):
    gains, s, d, sup = arg
    return fly(s, gains, d, sup)[:2]


_POOL = [None]


def _pool():
    """One process pool for the whole run. 48 episodes at ~2.6 s each is 2 minutes
    serial, which is too slow to search in; the episodes are independent, so they go
    wide. Workers are single-threaded numpy either way."""
    if _POOL[0] is None:
        import concurrent.futures as cf
        _POOL[0] = cf.ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))
    return _POOL[0]


def score(gains, seeds=SEEDS, diffs=DIFFS, supervisor=False, quiet=True):
    t0 = time.perf_counter()
    args = [(gains, s, d, supervisor) for d in diffs for s in seeds]
    res = list(_pool().map(_job, args, chunksize=2))
    fracs = [r[0] for r in res]
    wins = [r[1] for r in res]
    out = dict(frac=float(np.mean(fracs)), win=float(np.mean(wins)),
               secs=time.perf_counter() - t0)
    if not quiet:
        print("  frac %.4f  win %.3f  (%.0f s)" % (out["frac"], out["win"], out["secs"]))
    return out


def show(label, out, base=None):
    d = "" if base is None else "  %+.4f" % (out["frac"] - base)
    print("  %-34s frac %.4f  win %.3f%s  [%.0fs]"
          % (label, out["frac"], out["win"], d, out["secs"]), flush=True)


AXES = [
    ("k_bearing", [1.1, 1.6, 2.2, 3.0]),
    ("k_roll", [4.0, 6.0, 9.0, 13.0]),
    ("bank_max", [0.50, 0.62, 0.75, 0.90]),
    ("v_cruise", [4.5, 5.5, 6.5, 8.0]),
    ("v_min", [1.8, 2.5, 3.2]),
    ("turn_slow", [1.0, 1.6, 2.4, 3.5]),
    ("k_w", [1.2, 2.0, 3.0, 4.5]),
    ("w_tau", [2.0, 4.0, 8.0]),
    ("climb_a_max", [4.0, 6.0, 9.0]),
    ("approach_d", [1.0, 2.0, 3.5, 5.0]),
    ("commit_m", [0.8, 1.5, 2.5, 4.0]),
    ("k_pitch", [3.0, 5.0, 8.0]),
    ("k_speed", [0.03, 0.06, 0.12]),
    ("k_frame", [1.0, 2.0, 3.5]),
    ("conf_min", [0.06, 0.12, 0.25]),
    ("stale_max_s", [0.6, 0.9, 1.4]),
    ("k_trim", [0.0, 0.5, 1.5, 3.0]),
    ("k_acc", [0.4, 0.7, 1.1]),
]


def descend(base, passes=2):
    cur = dict(base)
    best = score(cur)
    show("start %r" % cur, best)
    for p in range(passes):
        print("\n--- pass %d ---" % (p + 1))
        for name, vals in AXES:
            trials = []
            for v in vals:
                if abs(cur.get(name, _default(name)) - v) < 1e-12:
                    continue
                cand = dict(cur, **{name: v})
                out = score(cand)
                trials.append((out, v))
                show("%s=%g" % (name, v), out, best["frac"])
            if trials:
                out, v = max(trials, key=lambda t: (t[0]["frac"], t[0]["win"]))
                if out["frac"] > best["frac"] + 1e-6:
                    best, cur[name] = out, v
                    print("    -> take %s=%g  (frac %.4f)" % (name, v, best["frac"]))
        print("\npass %d best: frac %.4f win %.3f\n  %s"
              % (p + 1, best["frac"], best["win"],
                 ",".join("%s=%g" % kv for kv in sorted(cur.items()))))
    return cur, best


_PROTO = BaselinePolicy()


def _default(name):
    return float(getattr(_PROTO, name))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    base = gains_from_str(sys.argv[2]) if len(sys.argv) > 2 and mode != "sweep" else {}
    if mode == "score":
        show(str(base) or "defaults", score(base))
    elif mode == "sweep":
        name, vals = sys.argv[2], [float(v) for v in sys.argv[3].split(",")]
        base = gains_from_str(sys.argv[4]) if len(sys.argv) > 4 else {}
        b = score(base)
        show("base", b)
        for v in vals:
            show("%s=%g" % (name, v), score(dict(base, **{name: v})), b["frac"])
    elif mode == "many":
        specs = [s for s in sys.argv[2:]]
        b = score({})
        show("defaults", b)
        for s in specs:
            show(s or "defaults", score(gains_from_str(s)), b["frac"])
    elif mode == "descend":
        cur, best = descend(base, passes=int(sys.argv[3]) if len(sys.argv) > 3 else 2)
        print("\nFINAL  frac %.4f  win %.3f" % (best["frac"], best["win"]))
        print(",".join("%s=%g" % kv for kv in sorted(cur.items())))
    else:
        raise SystemExit("modes: score | sweep | descend")
