"""Throwaway: is the vertical loop tracking, and if not, which term is short?

`_probe_baseline.py` says the misses at the gate plane are 0.4-0.9 m and part of that is
vertical, while the trace shows `e_horizon` sitting at +0.2..+0.4 rad for whole episodes
-- a standing aim-low. That is either (a) the accel loop not delivering the commanded
`a_des`, or (b) the accel loop delivering it fine and the OUTER loop being wrong, because
proportional-on-angle with no rate term is an undamped double integrator.

This prints both at once: commanded vs achieved vertical acceleration (the inner loop),
and elevation error vs true climb rate (the outer one).
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

from pilot.control.policies.baseline import BaselinePolicy, gains_from_str  # noqa: E402
from pilot.control.evalsuite.run_policy import build_config, _Env           # noqa: E402


def fly(seed, gains="", difficulty=0.2, max_steps=4000, trace=0):
    env = _Env(build_config(difficulty, 1.0), seed)
    obs = env.reset()
    p = BaselinePolicy(**gains_from_str(gains))
    p.reset()
    prev = env.true_state()
    rows = []
    for i in range(1, max_steps + 1):
        a = p(obs)
        dt = obs.dt_s
        v0 = prev["v"].copy()
        obs, info = env.step(a)
        ts = env.true_state()
        if info["done"]:
            break
        a_up_true = -(ts["v"][2] - v0[2]) / dt
        g = prev["gate_pos"][prev["active_gate"]]
        rows.append((p.last["e_horizon"], p.last["a_des"], p.last["a_up"], a_up_true,
                     -prev["v"][2], -prev["p"][2], -g[2],
                     float(np.linalg.norm(g - prev["p"])), a.thrust, p.last["trim"]))
        prev = ts
        if trace and i % trace == 0:
            r = rows[-1]
            print("%5d eh%+.3f a_des%+6.2f a_up%+6.2f (true%+6.2f) w%+5.2f "
                  "alt%6.2f gate%6.2f rng%6.1f thr%.3f trim%.3f" % ((i,) + r))
    return np.array(rows) if rows else np.zeros((0, 10))


if __name__ == "__main__":
    gains = sys.argv[1] if len(sys.argv) > 1 else ""
    seeds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,4,5,6").split(",")]
    trace = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print("gains: %r" % gains)
    all_rows = []
    for s in seeds:
        r = fly(s, gains, trace=trace)
        all_rows.append(r)
        if len(r):
            print("seed %2d  n%4d | e_horizon mean%+6.3f rms%5.3f | a_des mean%+6.2f | "
                  "a_up mean%+6.2f | est-true rms %5.2f | climb mean%+5.2f"
                  % (s, len(r), r[:, 0].mean(), np.sqrt((r[:, 0] ** 2).mean()),
                     r[:, 1].mean(), r[:, 3].mean(),
                     np.sqrt(((r[:, 2] - r[:, 3]) ** 2).mean()), r[:, 4].mean()))
    R = np.vstack([r for r in all_rows if len(r)])
    print("\npooled %d samples" % len(R))
    print("  inner loop  commanded a_des %+6.3f   achieved a_up %+6.3f   "
          "tracking error %+6.3f (rms %.3f)"
          % (R[:, 1].mean(), R[:, 3].mean(), (R[:, 3] - R[:, 1]).mean(),
             np.sqrt(((R[:, 3] - R[:, 1]) ** 2).mean())))
    print("  estimator   a_up est vs true rms %.3f m/s^2, corr %.4f"
          % (np.sqrt(((R[:, 2] - R[:, 3]) ** 2).mean()),
             np.corrcoef(R[:, 2], R[:, 3])[0, 1]))
    print("  outer loop  e_horizon mean %+6.3f rad (%.1f deg), rms %.3f"
          % (R[:, 0].mean(), np.degrees(R[:, 0].mean()),
             np.sqrt((R[:, 0] ** 2).mean())))
    print("  altitude    own mean %5.2f m, active gate mean %5.2f m, deficit %+5.2f m"
          % (R[:, 5].mean(), R[:, 6].mean(), (R[:, 5] - R[:, 6]).mean()))
    print("  climb rate  mean %+5.2f m/s, rms %.2f | thrust mean %.3f, trim mean %.3f"
          % (R[:, 4].mean(), np.sqrt((R[:, 4] ** 2).mean()),
             R[:, 8].mean(), R[:, 9].mean()))
