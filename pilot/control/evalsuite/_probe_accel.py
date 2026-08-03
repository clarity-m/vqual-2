"""Throwaway: is world-vertical acceleration recoverable from the permitted stream?

    a_world = R_wb @ specific_force + [0,0,g],  and the z ROW of R_wb is
    [-sin(pitch), cos(pitch) sin(roll), cos(pitch) cos(roll)] -- YAW-FREE.

So the one component of world acceleration the policy is allowed to know is the vertical
one, from `own.accel` plus the gravity-derived roll/pitch. Scored here against the
surrogate's true dv_z/dt before any control law is built on it.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

from pilot.control.policies.baseline import BaselinePolicy, G     # noqa: E402
from pilot.control.evalsuite.run_policy import build_config, _Env  # noqa: E402

env = _Env(build_config(0.2, 1.0), 3)
obs = env.reset()
p = BaselinePolicy()
prev_v = env.true_state()["v"].copy()
err, mag, n = [], [], 0
for i in range(1, 400):
    dt = obs.dt_s
    a = p(obs)
    o = obs.own
    r, q = float(o.roll_rad), float(o.pitch_rad)
    row = np.array([-np.sin(q), np.cos(q) * np.sin(r), np.cos(q) * np.cos(r)])
    a_up_est = -(row @ np.asarray(o.accel)) - G
    obs, info = env.step(a)
    v = env.true_state()["v"]
    if info["done"]:
        break
    a_up_true = -(v[2] - prev_v[2]) / dt
    prev_v = v.copy()
    err.append(a_up_est - a_up_true)
    mag.append(abs(a_up_true))
    n += 1
    if i % 40 == 0:
        print("%4d  est %+7.3f  true %+7.3f  err %+6.3f   (roll %+.2f pitch %+.2f)"
              % (i, a_up_est, a_up_true, a_up_est - a_up_true, r, q))

err = np.array(err)
print("\n%d samples | rms error %.3f m/s^2 | rms signal %.3f | corr %.4f"
      % (n, np.sqrt((err ** 2).mean()), np.sqrt((np.array(mag) ** 2).mean()),
         np.corrcoef(err + np.array(mag) * 0, err)[0, 0] if n > 2 else 0.0))
print("VERDICT: %s" % ("usable" if np.sqrt((err ** 2).mean()) < 1.0 else "NOT usable"))
