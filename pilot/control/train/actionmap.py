"""Resolves the ONE action mapping shared by training and deployment.

`pilot/control/surrogate/actions.py` owns `policy_to_action`: tanh outputs in [-1,1]^3
-> physical (roll_rate, pitch_rate, thrust). The deployment wrapper calls the identical
function, so any divergence between the two sides is a silent policy killer. This module
therefore *imports* it whenever it exists and only falls back to a local reference
implementation of the documented contract when the surrogate package is absent — which
happens while the sibling module is still being written, and when training against the
stub environment in `testenv.py`.

Which source was used is recorded in the checkpoint's `env_config['action_map_source']`,
so a checkpoint trained on the fallback is identifiable after the fact.

The speed curriculum's `speed_cap` is applied by THIS harness on top of the mapping
(rates only, never thrust) — see `ppo.PPO._to_physical`.
"""

from __future__ import annotations

import warnings

import numpy as np

# Contract values, mirrored from surrogate/actions.py. Used only by the fallback and by
# the agreement check below; the imported module's values win when it exists.
RATE_CAP_RPS_DEFAULT = 2.75
THRUST_FLOOR_DEFAULT = 0.10


def _fallback_policy_to_action(u, rate_cap: float = RATE_CAP_RPS_DEFAULT,
                               thrust_floor: float = THRUST_FLOOR_DEFAULT) -> np.ndarray:
    """[-1,1]^3 -> (roll_rate, pitch_rate, thrust), the documented affine mapping."""
    a = np.asarray(u, dtype=np.float32)
    single = a.ndim == 1
    a = np.atleast_2d(a)
    a = np.clip(a, -1.0, 1.0)
    out = np.empty_like(a)
    out[:, 0] = a[:, 0] * rate_cap
    out[:, 1] = a[:, 1] * rate_cap
    out[:, 2] = thrust_floor + (1.0 - thrust_floor) * 0.5 * (a[:, 2] + 1.0)
    return out[0] if single else out


class ActionMap:
    """Callable wrapper: `physical = action_map(u)` for u of shape [n, 3] or [3]."""

    def __init__(self, fn, source: str, rate_cap: float, thrust_floor: float) -> None:
        self.fn = fn
        self.source = source
        self.rate_cap = float(rate_cap)
        self.thrust_floor = float(thrust_floor)
        self._batch_ok = self._probe_batch()

    def _probe_batch(self) -> bool:
        try:
            out = np.asarray(self.fn(np.zeros((2, 3), dtype=np.float32)), dtype=np.float32)
            return out.shape == (2, 3)
        except Exception:
            return False

    def __call__(self, u: np.ndarray) -> np.ndarray:
        a = np.asarray(u, dtype=np.float32)
        single = a.ndim == 1
        a2 = np.atleast_2d(a)
        if self._batch_ok:
            out = np.asarray(self.fn(a2), dtype=np.float32).reshape(a2.shape)
        else:
            out = np.stack([np.asarray(self.fn(row), dtype=np.float32).reshape(3) for row in a2])
        out = np.ascontiguousarray(out, dtype=np.float32)
        return out[0] if single else out

    def thrust_of(self, u_thrust: float) -> float:
        """Physical thrust produced by a scalar tanh output on the thrust channel."""
        return float(self(np.array([0.0, 0.0, float(u_thrust)], dtype=np.float32))[2])

    def thrust_pre_tanh_bias(self, thrust: float) -> float:
        """Pre-tanh bias for the network's thrust output that maps to `thrust`.

        This is what `network.ActorCritic(thrust_bias=...)` wants: the head's raw output,
        BEFORE the tanh squash. Derived from the real mapping rather than assumed, so a
        curved or re-scaled `policy_to_action` still initializes the policy at hover.
        """
        u = float(np.clip(self.invert_thrust(thrust), -0.999999, 0.999999))
        return 0.5 * float(np.log((1.0 + u) / (1.0 - u)))

    def invert_thrust(self, thrust: float) -> float:
        """tanh-space value (i.e. the squashed action u[2]) whose mapped thrust equals
        `thrust`, found by bisection."""
        target = float(thrust)
        lo, hi = -1.0, 1.0
        f_lo, f_hi = self.thrust_of(lo), self.thrust_of(hi)
        if not (min(f_lo, f_hi) - 1e-6 <= target <= max(f_lo, f_hi) + 1e-6):
            raise ValueError(f"thrust {target} outside mapping range [{f_lo}, {f_hi}]")
        ascending = f_hi >= f_lo
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = self.thrust_of(mid)
            if (f_mid < target) == ascending:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def resolve_action_map(require_surrogate: bool = False) -> ActionMap:
    """Import the surrogate's mapping, else the documented fallback."""
    try:
        from pilot.control.surrogate.actions import policy_to_action  # type: ignore
        from pilot.control.surrogate import actions as _actions  # type: ignore

        rate_cap = float(getattr(_actions, "RATE_CAP_RPS", RATE_CAP_RPS_DEFAULT))
        thrust_floor = float(getattr(_actions, "THRUST_FLOOR", THRUST_FLOOR_DEFAULT))
        amap = ActionMap(policy_to_action, "surrogate", rate_cap, thrust_floor)
        _warn_on_disagreement(amap)
        return amap
    except ImportError:
        if require_surrogate:
            raise
        warnings.warn(
            "pilot.control.surrogate.actions not importable — using the local reference "
            "action mapping (rates +-%.2f rad/s, thrust [%.2f, 1.0]). Re-train or at "
            "least re-verify once the surrogate module lands."
            % (RATE_CAP_RPS_DEFAULT, THRUST_FLOOR_DEFAULT),
            RuntimeWarning,
            stacklevel=2,
        )
        return ActionMap(_fallback_policy_to_action, "fallback",
                         RATE_CAP_RPS_DEFAULT, THRUST_FLOOR_DEFAULT)


def _warn_on_disagreement(amap: ActionMap, tol: float = 1e-4) -> None:
    """Loud warning if the imported mapping differs from the documented contract.

    Not an error: the surrogate owns the mapping and wins. But a change there silently
    invalidates in-flight checkpoints, so it must not pass unremarked.
    """
    probe = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [0.5, -0.5, -0.62]],
        dtype=np.float32,
    )
    got = amap(probe)
    want = _fallback_policy_to_action(probe, amap.rate_cap, amap.thrust_floor)
    if not np.allclose(got, want, atol=tol):
        warnings.warn(
            "surrogate policy_to_action disagrees with the documented affine contract; "
            "using the surrogate's version (it owns the mapping). max abs diff %.4g"
            % float(np.max(np.abs(got - want))),
            RuntimeWarning,
            stacklevel=3,
        )
