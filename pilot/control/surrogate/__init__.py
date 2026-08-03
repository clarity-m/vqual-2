"""P2 -- the surrogate training environment.

    from pilot.control.surrogate import VecSurrogate, EnvConfig, policy_to_action

    cfg = EnvConfig.curriculum(difficulty=0.3, speed_cap=0.6)
    env = VecSurrogate(n_envs=1024, config=cfg, seed=0)
    obs = env.reset()                                    # [1024, 73] float32
    obs, rew, done, info = env.step(policy_to_action(u)) # u is [1024, 3] in [-1, 1]

Modules, in dependency order:

    actions.py     the action envelope: RATE_CAP_RPS = 2.75, THRUST_FLOOR = 0.10
    vmath.py       batched quaternion algebra, bit-identical to plant.py's
    camera.py      the 640x360 / 20 deg-up pinhole model from CONVENTIONS.md
    course.py      procedural courses; the world frame lives here and stops here
    noise.py       the P3 FALLBACK detection-noise model (hand-specified, pessimistic)
    detect.py      synthetic detections on the ~30 fps camera clock
    attention.py   attention + AUTO_ATTENTION yaw servo -- a STUB, meant to be replaced
    env.py         VecSurrogate
    single.py      SingleSurrogate, at the interface.Observation level
    selfcheck.py   python3 pilot/control/surrogate/selfcheck.py

Read `../TRAINING_ARCHITECTURE.md` (P2/P3) before changing anything here, and
`../../CONVENTIONS.md` before touching a sign.
"""

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.dirname(_HERE), _os.path.dirname(_os.path.dirname(_HERE))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from actions import (RATE_CAP_RPS, THRUST_CEIL, THRUST_FLOOR, clip_physical,  # noqa: E402
                     policy_to_action, thrust_to_unit)
from attention import AttentionPolicy, AttnInput, AttnOutput, R_COMMIT_M  # noqa: E402
from env import EnvConfig, VecSurrogate, build_observation  # noqa: E402
from noise import NoiseParams  # noqa: E402
from single import SingleSurrogate  # noqa: E402

__all__ = ["RATE_CAP_RPS", "THRUST_FLOOR", "THRUST_CEIL", "policy_to_action",
           "thrust_to_unit", "clip_physical", "EnvConfig", "VecSurrogate",
           "SingleSurrogate", "build_observation", "NoiseParams", "AttentionPolicy",
           "AttnInput", "AttnOutput", "R_COMMIT_M"]
