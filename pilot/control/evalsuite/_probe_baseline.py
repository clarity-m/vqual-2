"""Throwaway: watch baseline episodes and classify what they hit. Not part of the suite."""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

from pilot.control.policies.baseline import BaselinePolicy, gains_from_str  # noqa: E402
from pilot.control.evalsuite.run_policy import build_config, _Env           # noqa: E402


def fly(seed, gains="", trace=0, difficulty=0.2, max_steps=4000):
    env = _Env(build_config(difficulty, 1.0), seed)
    obs = env.reset()
    p = BaselinePolicy(**gains_from_str(gains))
    p.reset()
    ts = env.true_state()
    start = ts["active_gate"]
    cues = {"gate": 0, "ribbon": 0, "blind": 0}
    zmin, zmax = 1e9, -1e9
    for i in range(1, max_steps + 1):
        a = p(obs)
        cues[p.last["cue"]] += 1
        prev = env.true_state()
        dt = obs.dt_s
        obs, info = env.step(a)
        # `env` auto-resets on done, so the post-step true state is the NEXT episode's
        # spawn. The terminal point has to be extrapolated from the last real one.
        end = prev["p"] + prev["v"] * dt
        alt, ceil, floor = -prev["p"][2], -prev["z_ceil"], prev["sphere_r"]
        zmin, zmax = min(zmin, alt), max(zmax, alt)
        if trace and i % trace == 0:
            L = p.last
            print("%5d alt %6.2f (ceil %5.2f) sp %5.2f | %-6s w%.2f b%+.2f ef%+.2f "
                  "eh%+.2f v%4.1f | %+5.2f %+5.2f %.3f"
                  % (i, alt, ceil, np.linalg.norm(prev["v"]), L["cue"], L["weight"],
                     L["b_tgt"], L["e_frame"], L["e_horizon"], L["v_tgt"],
                     a.roll_rate, a.pitch_rate, a.thrust))
        if info["done"]:
            # Where did the gate plane get crossed? Decompose the miss into the gate's
            # own horizontal and vertical axes: that says which loop is at fault.
            miss = ""
            gp, gn = prev["gate_pos"], prev["gate_nrm"]
            for k in range(max(prev["active_gate"] - 1, 0),
                           min(prev["active_gate"] + 3, len(gp))):
                n = gn[k]
                d0 = float((prev["p"] - gp[k]) @ n)
                d1 = float((end - gp[k]) @ n)
                if d0 < 0.0 <= d1:
                    t = -d0 / max(d1 - d0, 1e-9)
                    x = prev["p"] + t * (end - prev["p"]) - gp[k]
                    x = x - (x @ n) * n
                    up = np.array([0.0, 0.0, -1.0])
                    side = np.cross(up, n)
                    side /= max(np.linalg.norm(side), 1e-9)
                    miss = "g%d side%+.2f up%+.2f |%.2f|" % (
                        k, float(x @ side), float(x @ up), float(np.linalg.norm(x)))
            why = ("finished" if info["finished"] else
                   "corridor" if info["corridor_exit"] else
                   "timeout" if info["timeout"] else "collision")
            if why == "collision":
                if -end[2] <= floor + 0.05:
                    why = "FLOOR"
                elif -end[2] >= ceil - 0.05:
                    why = "CEILING"
                else:
                    why = "GATE FRAME" if miss else "collision?"
            return dict(seed=seed, why=why, gates=int(info["gates_passed"]) - start,
                        need=int(prev["n_gates"]) - start, steps=i,
                        t=float(prev["t_s"]), alt=alt, ceil=ceil,
                        zmin=zmin, zmax=zmax, miss=miss,
                        cue="g%d/r%d/b%d" % (cues["gate"], cues["ribbon"], cues["blind"]))
    return dict(seed=seed, why="max_steps", gates=0, need=0, steps=max_steps, t=0.0,
                alt=0, ceil=0, zmin=zmin, zmax=zmax, miss="", cue="")


if __name__ == "__main__":
    gains = sys.argv[1] if len(sys.argv) > 1 else ""
    spec = sys.argv[2] if len(sys.argv) > 2 else "8"
    seeds = ([int(x.lstrip("s")) for x in spec.split(",")]
             if "," in spec or spec.startswith("s") else list(range(int(spec))))
    trace = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print("gains: %r" % gains)
    print("%5s %-11s %5s %5s %6s %6s %-26s %s"
          % ("seed", "why", "gate", "need", "t_s", "alt", "miss at the plane",
             "cue g/r/b"))
    rows = []
    for s in seeds:
        r = fly(s, gains, trace)
        rows.append(r)
        print("%5d %-11s %5d %5d %6.2f %6.2f %-26s %s"
              % (r["seed"], r["why"], r["gates"], r["need"], r["t"], r["alt"],
                 r["miss"], r["cue"]))
    print("\nfinished %d/%d | gates mean %.2f | %s"
          % (sum(1 for r in rows if r["why"] == "finished"), len(rows),
             np.mean([r["gates"] for r in rows]),
             ", ".join("%s %d" % (w, sum(1 for r in rows if r["why"] == w))
                       for w in ("FLOOR", "CEILING", "GATE FRAME", "corridor",
                                 "timeout", "finished"))))
