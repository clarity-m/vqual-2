"""D6's deployment wrapper: a frozen checkpoint behind `interface.Policy`.

    Observation -> to_vector() -> ObsNormalizer(frozen mean/std) -> FrameStack(k)
                -> ActorCritic.infer(deterministic) -> u in [-1,1]^3
                -> policy_to_action -> interface.Action(yaw_mode=AUTO_ATTENTION)

Every stage of that chain is the TRAINING SIBLING'S OWN CODE, called the way
`train/README.md` specifies, and none of it is reimplemented here. That is the whole
design rule for this file. Each stage is a place where deployment can silently disagree
with training and produce a policy that scored well and flies badly:

  * `ObsNormalizer` clips normalized observations at +-10 as well as centring them, so
    hand-rolling `(v - mean) / std` would quietly widen the input distribution on
    exactly the outlier steps that matter.
  * `FrameStack` orders frames OLDEST FIRST and fills all k slots on the first push.
    A reversed stack is a valid-looking tensor of the right shape and a different
    policy.
  * `policy_to_action` is the squash and the envelope training actually applied.
  * `infer(deterministic=True)` returns the tanh of the MEAN, not a sample. Racing wants
    the mode; sampling at race time adds variance for nothing.

The statistics ship with the weights and are never recomputed here.

torch and the sibling modules are imported lazily inside `__init__`, so `policies/`
stays importable -- and `selfcheck.py` stays runnable -- on a machine with neither.
Anything wrong with the environment surfaces as one clear exception at construction
time rather than an AttributeError mid-flight.
"""

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot import interface                                    # noqa: E402
from pilot.control.policies.envelope import clamp_action       # noqa: E402

CKPT_DIR = os.path.join(_ROOT, "pilot", "control", "train", "checkpoints")


def _ckpt_speed_cap(ckpt, default=1.0):
    """The `speed_cap` this checkpoint trained under, from its `env_config`.

    Missing or unparseable means "no curriculum recorded", which is 1.0 -- the same
    thing `EnvConfig.speed_cap` defaults to.
    """
    try:
        v = float((ckpt.get("env_config") or {}).get("speed_cap", default))
    except (TypeError, ValueError):
        return float(default)
    return float(np.clip(v, 1e-3, 1.0))


def _apply_speed_cap(phys, speed_cap):
    """Scale the two RATE channels, matching `train/ppo.py:_to_physical`.

    Thrust is deliberately untouched: capping it would move the aircraft's trim point,
    which is not what "cap commanded aggressiveness" means.
    """
    out = np.array(phys, dtype=np.float64, copy=True)
    out[..., :2] *= float(speed_cap)
    return out


def _need(what, exc):
    return RuntimeError(
        "RLPolicy needs %s (%s: %s). The baseline in this package needs neither torch "
        "nor the training sibling; if this is a machine that only has to fly the "
        "baseline, use it." % (what, type(exc).__name__, exc))


class RLPolicy(interface.Policy):
    """A trained checkpoint, wrapped for the live stack and for `evalsuite`."""

    def __init__(self, ckpt_path, device="cpu"):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise _need("pytorch", exc)
        try:
            from pilot.control.train.network import load_checkpoint
            from pilot.control.train.framestack import FrameStack
            from pilot.control.train.normalize import ObsNormalizer
        except Exception as exc:
            raise _need("pilot/control/train (network, framestack, normalize)", exc)
        try:
            from pilot.control.surrogate.actions import policy_to_action
        except Exception as exc:
            raise _need("surrogate/actions.py:policy_to_action, the exact action "
                        "mapping training applied -- RLPolicy will not substitute its "
                        "own", exc)

        self.path = self._resolve(ckpt_path)
        self.model, ckpt = load_checkpoint(self.path, map_location=device)
        self.device = device
        self.step = int(ckpt.get("step", -1))
        self.env_config = ckpt.get("env_config")
        self.k = int(ckpt["frame_stack"])
        self.obs_dim = int(getattr(self.model, "obs_dim", interface.OBS_DIM))
        if self.obs_dim != interface.OBS_DIM:
            raise RuntimeError("checkpoint network takes %d-D observations, OBS_DIM is %d"
                               % (self.obs_dim, interface.OBS_DIM))

        self.norm = ObsNormalizer.from_arrays(ckpt["obs_mean"], ckpt["obs_std"])

        # Both of these describe the input pipeline / action mapping the checkpoint was
        # TRAINED under and are not recoverable from the weights. Absent = the original
        # behaviour, so pre-existing checkpoints are unaffected.
        self.frame_offsets = ckpt.get("frame_offsets")
        self.stack = FrameStack(self.obs_dim, self.k, n_envs=1,
                                offsets=self.frame_offsets)

        self.thrust_residual = ckpt.get("thrust_residual")
        if self.thrust_residual:
            try:
                from pilot.control.train.actionmap import (PITCH_INDEX, ROLL_INDEX,
                                                           ActionMap, ResidualThrustMap)
                from pilot.control.surrogate import actions as _actions
            except Exception as exc:
                raise _need("pilot/control/train/actionmap.py (ResidualThrustMap)", exc)
            base = ActionMap(policy_to_action, "surrogate",
                             float(getattr(_actions, "RATE_CAP_RPS", 2.75)),
                             float(getattr(_actions, "THRUST_FLOOR", 0.10)))
            self._residual = ResidualThrustMap(
                base,
                hover_thrust=float(self.thrust_residual["hover_thrust"]),
                span=float(self.thrust_residual.get("span", 0.30)),
                min_cos=float(self.thrust_residual.get("min_cos", 0.5)))
            self._roll_i, self._pitch_i = ROLL_INDEX, PITCH_INDEX
        else:
            self._residual = None

        self._to_action = policy_to_action
        self._FrameStack = FrameStack

        # THE SPEED CURRICULUM IS PART OF THE ACTION MAPPING, and it lives in the
        # checkpoint, not in `policy_to_action`. `train/ppo.py:_to_physical` maps the
        # network's tanh output through `policy_to_action` and THEN scales the two rate
        # channels by the curriculum's `speed_cap`; this wrapper used to stop after the
        # first half. A checkpoint that trained its whole life at speed_cap 0.5 -- which
        # every run1 checkpoint did, the curriculum never having promoted -- was therefore
        # flown at TWICE the rate authority it ever saw, in evaluation and in deployment
        # alike. It is stored in the checkpoint, so read it rather than assume 1.0.
        self.speed_cap = _ckpt_speed_cap(ckpt)
        self.info = ("ckpt %s | step %d | k=%d | offsets %s | thrust %s | hidden %s"
                     % (os.path.basename(self.path), self.step, self.k,
                        self.frame_offsets or "consecutive",
                        "residual" if self._residual else "affine",
                        ckpt.get("net_config", {}).get("hidden")))
        self.reset()

    @staticmethod
    def _resolve(p):
        for cand in (p, str(p) + ".pt", os.path.join(CKPT_DIR, str(p)),
                     os.path.join(CKPT_DIR, str(p) + ".pt")):
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError("no checkpoint at %r (also looked in %s)" % (p, CKPT_DIR))

    # -- interface.Policy -------------------------------------------------------
    def reset(self):
        """Drop the history. The next push refills all k slots from one observation."""
        self.stack.clear()

    def __call__(self, obs):
        v = np.asarray(obs.to_vector(), dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.stack.push(self.norm(v[None, :]))
        u = np.asarray(self.model.infer(x, deterministic=True),
                       dtype=np.float64).reshape(-1)[:3]
        if self._residual is not None:
            # RAW roll/pitch, before normalization -- training reads the same unnormalized
            # channels (ppo._raw_obs). Normalizing here would apply the compensation at a
            # different attitude than the one the aircraft is actually at.
            phys = np.asarray(self._residual(u.astype(np.float32),
                                             roll=v[self._roll_i],
                                             pitch=v[self._pitch_i]),
                              dtype=np.float64).reshape(-1)
        else:
            phys = np.asarray(self._to_action(u), dtype=np.float64).reshape(-1)
        if phys.size < 3:
            raise RuntimeError("policy_to_action returned %d values, expected 3 "
                               "(roll rate, pitch rate, thrust)" % phys.size)
        phys = _apply_speed_cap(phys, self.speed_cap)
        return clamp_action(phys[0], phys[1], phys[2],
                            yaw_rate=0.0, yaw_mode=interface.YawMode.AUTO_ATTENTION)


class RLBatchPolicy:
    """`RLPolicy` over `n` lanes at once, for the batched evaluator.

    NOT a second implementation: it is constructed FROM an `RLPolicy` and reuses that
    instance's model, its frozen `ObsNormalizer`, its `policy_to_action` and its
    `speed_cap`. The only thing rebuilt is the `FrameStack`, which has to be `n` wide.
    Anything that would make the batch behave differently from the scalar wrapper has to
    be changed in `RLPolicy` first, where both paths pick it up.

    Consumes `[n, 73]` observation vectors and returns `[n, 3]` PHYSICAL actions --
    the surrogate's `step` signature, not `interface.Action`.
    """

    def __init__(self, policy, n):
        self.p = policy
        self.n = int(n)
        self.stack = policy._FrameStack(policy.obs_dim, policy.k, n_envs=self.n)
        self.info = policy.info + " | batch x%d" % self.n
        self.reset()

    def reset(self, n=None):
        if n is not None and int(n) != self.n:
            self.n = int(n)
            self.stack = self.p._FrameStack(self.p.obs_dim, self.p.k, n_envs=self.n)
        self.stack.clear()

    def act(self, obs_matrix, done=None):
        v = np.asarray(obs_matrix, dtype=np.float32).reshape(self.n, self.p.obs_dim)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.stack.push(self.p.norm(v)) if done is None else \
            self.stack.push(self.p.norm(v), done=done)
        u = np.asarray(self.p.model.infer(x, deterministic=True), dtype=np.float64)
        u = u.reshape(self.n, -1)[:, :3]
        return _apply_speed_cap(np.asarray(self.p._to_action(u), dtype=np.float64),
                                self.p.speed_cap)


def sidecar(path):
    """The JSON twin of a checkpoint: everything but the weights, readable without torch."""
    base = os.path.splitext(str(path))[0]
    for cand in (base + ".json", str(path) + ".json",
                 os.path.join(CKPT_DIR, os.path.basename(base) + ".json")):
        if os.path.isfile(cand):
            with open(cand) as fh:
                return json.load(fh)
    return {}
