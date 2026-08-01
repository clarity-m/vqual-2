"""The command envelope every policy in this package emits into.

`interface.Action.clipped()` clips to `MAX_RATE_RPS` = 6.0, which is the *simulator's*
limit, not the model's. The plant fit only ever saw about +-2.5..3 rad/s (the levelling
assist clamps at 3.0 and hand flying rarely exceeded it), so a command beyond that is
extrapolation on a model that has no data there -- exactly the class of artefact the
architecture flags for the low-throttle clamp. The training sibling caps at the same
place: `surrogate/actions.py` `RATE_CAP_RPS` / `THRUST_FLOOR`. THOSE ARE THE SOURCE;
the values here are a copy so that `policies/` imports without the surrogate, and
`selfcheck.py` fails loudly if the two ever drift apart.

`clamp_action` also scrubs non-finite values. A NaN reaching the link layer becomes a
NaN rate command in flight; a policy that has been handed a NaN observation should fall
back to level-and-hover instead, which is cheap insurance for the price of one isfinite.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot import interface  # noqa: E402

RATE_CAP_RPS = 2.75   # must equal surrogate/actions.py RATE_CAP_RPS
THRUST_FLOOR = 0.10   # must equal surrogate/actions.py THRUST_FLOOR
THRUST_CEIL = 1.0


def _finite(x, fallback):
    x = float(x)
    return x if np.isfinite(x) else float(fallback)


def clamp_action(roll_rate, pitch_rate, thrust,
                 yaw_rate=0.0, yaw_mode=interface.YawMode.AUTO_ATTENTION):
    """Build an `interface.Action` inside the fitted envelope. Signs untouched."""
    return interface.Action(
        roll_rate=float(np.clip(_finite(roll_rate, 0.0), -RATE_CAP_RPS, RATE_CAP_RPS)),
        pitch_rate=float(np.clip(_finite(pitch_rate, 0.0), -RATE_CAP_RPS, RATE_CAP_RPS)),
        yaw_rate=float(np.clip(_finite(yaw_rate, 0.0), -RATE_CAP_RPS, RATE_CAP_RPS)),
        thrust=float(np.clip(_finite(thrust, interface.HOVER_THRUST),
                             THRUST_FLOOR, THRUST_CEIL)),
        yaw_mode=yaw_mode,
    )


def in_envelope(action):
    """True when an action respects the envelope. Used by `selfcheck.py`."""
    return (abs(action.roll_rate) <= RATE_CAP_RPS + 1e-9
            and abs(action.pitch_rate) <= RATE_CAP_RPS + 1e-9
            and abs(action.yaw_rate) <= RATE_CAP_RPS + 1e-9
            and THRUST_FLOOR - 1e-9 <= action.thrust <= THRUST_CEIL + 1e-9
            and all(np.isfinite(action.to_vector())))
