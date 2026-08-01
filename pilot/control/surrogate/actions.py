"""The action envelope: what a policy is allowed to command, and in what units.

Two numbers here, and both are deliberately tighter than `interface.py`'s limits.

**`RATE_CAP_RPS = 2.75`, not `interface.MAX_RATE_RPS = 6.0`.** 6.0 rad/s is what the
link layer will encode; 2.75 is what the plant fit actually saw. The rate-loop gains in
`plant.json` (0.970 / 0.963 / 0.904) were identified from doublets and taps whose
envelope tops out near 3 rad/s, so a policy commanding 6 is being scored against an
extrapolation of the inner loop rather than a measurement of it. `TRAINING_ARCHITECTURE.md`
T4 says "squashed to the envelope the plant fit actually saw, ~+-2.5-3 rad/s"; 2.75 is the
middle of that. Nothing stops the deployed policy from being re-capped later -- this is
the training envelope, not a safety limit.

**`THRUST_FLOOR = 0.10`.** The reason `TRAINING_ARCHITECTURE.md` T4 gives for this floor
is that the thrust curve was unmeasured below throttle 0.10 and hand-clamped there, so a
policy living down there would be exploiting a fit artefact rather than physics.

*That reason has since expired and the floor has not.* Card 2 was flown: `plant.json` now
carries a measured knot table with reads at throttle 0.0002, 0.05, 0.10, 0.15 and 0.20,
the hand clamp is gone, and idle thrust is a measured +0.35 m/s^2. The floor is therefore
now a conservative training guard, not a correctness requirement, and it could be lowered
to ~0.02 on the evidence. It stays at 0.10 because the training harness, the baseline and
the deployment wrapper are all coded against this constant and a silent change to the
action envelope mid-project is not worth the descent authority it buys. Lower it
deliberately, in one commit, with a retrain -- not incidentally.

`policy_to_action` is the only place the squashing convention lives, so the PPO harness,
the baseline and the deployed policy cannot drift apart on it.
"""

import numpy as np

# The envelope the plant fit saw. NOT interface.MAX_RATE_RPS.
RATE_CAP_RPS = 2.75

# The plant is unmeasured below this throttle. See the module docstring.
THRUST_FLOOR = 0.10

# Thrust ceiling: the full-throttle read is a measurement, so 1.0 is honest.
THRUST_CEIL = 1.0


def policy_to_action(u):
    """Map squashed network outputs to physical units.

    `u` is `[..., 3]` in [-1, 1] -- the tanh outputs (roll_rate, pitch_rate, thrust), in
    that order. Returns the same shape:

        [..., 0]  roll_rate  rad/s in +-RATE_CAP_RPS   (canonical body NED: >0 right wing down)
        [..., 1]  pitch_rate rad/s in +-RATE_CAP_RPS   (canonical body NED: >0 nose up)
        [..., 2]  thrust     in [THRUST_FLOOR, THRUST_CEIL]

    Yaw is absent by construction: it belongs to the AUTO_ATTENTION servo, which the
    surrogate drives from the attention module rather than from the action vector.

    Thrust is mapped affinely from [-1, 1] onto [floor, ceil], so u = 0 is the midpoint
    0.55 rather than hover. The T4 note about initialising the thrust head near
    `interface.HOVER_THRUST` = 0.27 is a *bias initialisation* on the network, computed
    with `thrust_to_unit` below; it is not a change to this mapping.
    """
    u = np.clip(np.asarray(u, dtype=np.float32), -1.0, 1.0)
    out = np.empty_like(u)
    out[..., 0] = u[..., 0] * RATE_CAP_RPS
    out[..., 1] = u[..., 1] * RATE_CAP_RPS
    out[..., 2] = THRUST_FLOOR + 0.5 * (u[..., 2] + 1.0) * (THRUST_CEIL - THRUST_FLOOR)
    return out


def thrust_to_unit(thrust):
    """Inverse of the thrust leg -- the pre-tanh value whose output is `thrust`.

    Use it to bias-initialise the thrust head near hover:
    `atanh(thrust_to_unit(interface.HOVER_THRUST))`, guarded away from +-1.
    """
    t = np.clip(np.asarray(thrust, dtype=np.float64), THRUST_FLOOR, THRUST_CEIL)
    return 2.0 * (t - THRUST_FLOOR) / (THRUST_CEIL - THRUST_FLOOR) - 1.0


def clip_physical(a, speed_cap=1.0):
    """Clamp a PHYSICAL action `[..., 3]` into the envelope, with the speed curriculum.

    `VecSurrogate.step` applies this to whatever it is handed, so a policy that ignores
    `policy_to_action` still cannot drive the plant outside the fitted region.
    """
    a = np.asarray(a, dtype=np.float32)
    cap = np.float32(RATE_CAP_RPS * float(speed_cap))
    out = np.empty_like(a)
    out[..., 0] = np.clip(a[..., 0], -cap, cap)
    out[..., 1] = np.clip(a[..., 1], -cap, cap)
    out[..., 2] = np.clip(a[..., 2], THRUST_FLOOR, THRUST_CEIL)
    return out
