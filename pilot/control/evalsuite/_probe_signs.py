"""Throwaway: does the surrogate's OBSERVATION agree with its own world truth?

Not a test of my policy -- a referee on the boundary between us. Computes the body-frame
gate offset and the roll/pitch from privileged state and compares with what
`build_observation` published. Anything but a small noise residual is a sign problem in
one of us, and the architecture says suspect the surrogate first.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

from pilot.control.plant import quat_to_rot                      # noqa: E402
from pilot.control.policies.baseline import BaselinePolicy       # noqa: E402
from pilot.control.evalsuite.run_policy import build_config      # noqa: E402
from pilot.control.surrogate.single import SingleSurrogate       # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
env = SingleSurrogate(build_config(0.2, 1.0), seed)
obs = env.reset()
p = BaselinePolicy()

print("%5s | %-28s %-28s | %-16s %-16s"
      % ("step", "gate0 pos_body TRUTH", "gate0 pos_body OBSERVED",
         "roll/pitch TRUE", "roll/pitch OBS"))
for i in range(1, 61):
    ts = env.true_state()
    R = quat_to_rot(ts["q"])                       # body -> world
    k = ts["active_gate"]
    rel = R.T @ (ts["gate_pos"][k] - ts["p"])      # world -> body
    sr = np.arctan2(R[2, 1], R[2, 2])
    sp = np.arcsin(-np.clip(R[2, 0], -1, 1))
    g = next((gg for gg in obs.gates if gg.valid and gg.index == k), None)
    seen = "-- not detected --"
    if g is not None:
        seen = "%7.2f %7.2f %7.2f s%.2f" % (*g.pos_body, g.staleness_s)
    print("%5d | %7.2f %7.2f %7.2f  el%+.2f  %-28s | %+6.2f %+6.2f    %+6.2f %+6.2f"
          % (i, rel[0], rel[1], rel[2], np.arctan2(-rel[2], np.hypot(rel[0], rel[1])),
             seen, sr, sp, obs.own.roll_rad, obs.own.pitch_rad))
    obs, info = env.step(p(obs))
    if info["done"]:
        print("done", {kk: v for kk, v in info.items() if kk != "reward"})
        break
