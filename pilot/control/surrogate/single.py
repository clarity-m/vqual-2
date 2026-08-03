"""One instance, at the `interface.Observation` / `interface.Action` level.

This is what an `interface.Policy` is scored on -- the reactive PID baseline, the deployed
RL checkpoint, anything. `VecSurrogate` is what PPO trains on. They must agree, so
`SingleSurrogate` is a thin adapter over `VecSurrogate(1, ...)` rather than a
reimplementation: there is exactly one set of dynamics, one detection model and one
attention module in this package, and no second copy that can drift.

    env = SingleSurrogate(EnvConfig.curriculum(0.3, 0.7), seed=0)
    obs = env.reset()                        # interface.Observation
    obs, info = env.step(policy(obs))        # interface.Action in, Observation out

`Action.yaw_rate` is IGNORED under `YawMode.AUTO_ATTENTION`, exactly as the live stack
ignores it: yaw comes from the attention servo. Under `YawMode.POLICY` it is still ignored
here -- the surrogate only models the delegated case, which is the architecture's decision
(T4: "Yaw stays with AUTO_ATTENTION"). A policy that takes the axis back cannot be
evaluated on this surrogate without a change here, and that change should be deliberate.

Rates are clipped into the training envelope (`actions.RATE_CAP_RPS`, and the config's
`speed_cap`), not `interface.MAX_RATE_RPS`: outside it the rate-loop model is
extrapolation. `Action.clipped()` uses the interface's wider limit and is not what runs.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import interface  # noqa: E402

from env import EnvConfig, VecSurrogate, build_observation  # noqa: E402


class SingleSurrogate:
    """One race, `Observation` in / `Action` out. Same dynamics, noise and attention."""

    def __init__(self, config=None, seed=0):
        self.cfg = config if config is not None else EnvConfig()
        self.vec = VecSurrogate(1, self.cfg, seed)
        self._last = None

    # -- gym-ish ------------------------------------------------------------
    def reset(self):
        self.vec.reset()
        self._last = build_observation(self.vec._fields, 0)
        return self._last

    def step(self, action):
        """`interface.Action` -> (`interface.Observation`, info dict of scalars)."""
        a = np.array([[float(action.roll_rate), float(action.pitch_rate),
                       float(action.thrust)]], dtype=np.float32)
        _, rew, done, info = self.vec.step(a)
        self._last = build_observation(self.vec._fields, 0)
        # Unwrap the one env: per-env scalars become Python scalars, but per-env VECTORS
        # (`terminal_obs` is [n, 73]) stay arrays.
        out = {}
        for k, v in info.items():
            e = v[0]
            out[k] = e.item() if getattr(e, "size", 1) == 1 else e
        out["reward"] = float(rew[0])
        out["done"] = bool(done[0])
        return self._last, out

    # -- conveniences -------------------------------------------------------
    @property
    def obs(self):
        return self._last

    def obs_vector(self):
        """The 73-D vector for the current observation, via the vectorised fast path."""
        return self.vec._encode(self.vec._fields)[0]

    def true_state(self):
        """PRIVILEGED world state. For scoring and debugging only -- never for a policy."""
        v = self.vec
        return dict(p=v.p[0].copy(), v=v.v[0].copy(), q=v.q[0].copy(),
                    active_gate=int(v.active[0]), n_gates=int(v.n_gates[0]),
                    gate_pos=v.g_pos[0].copy(), gate_nrm=v.g_nrm[0].copy(),
                    corridor_r=float(v.corridor_r[0]), z_ceil=float(v.z_ceil[0]),
                    sphere_r=float(v.sphere_r[0]), t_s=float(v.t_s[0]))


def default_action(thrust=interface.HOVER_THRUST):
    """A level, hovering action -- the D6 recovery supervisor's output, for tests."""
    return interface.Action(roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0, thrust=thrust,
                            yaw_mode=interface.YawMode.AUTO_ATTENTION)
