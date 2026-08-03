"""Policy/value network and the checkpoint format.

Self-contained on purpose: the deployment wrapper in `pilot/control/policies/` imports
THIS module to rebuild a network from a checkpoint, so nothing here may depend on the
surrogate, on PPO, or on anything else that only exists at training time.

Architecture (per T4 in TRAINING_ARCHITECTURE.md):
  * input  = frame stack of the last k normalized 73-D observations, flattened
  * trunk  = 2 x 256 tanh MLP, separate trunks for policy and value
  * policy = state-independent-log-std Gaussian in pre-tanh space, squashed by tanh into
             [-1,1]^3, then mapped to physical units by surrogate/actions.py
  * action order is (roll_rate, pitch_rate, thrust) — yaw belongs to the attention servo
  * the thrust head's output bias is initialized so the initial mean thrust lands on
    interface.HOVER_THRUST after the mapping

On the tanh-squashed Gaussian and PPO: the sampled PRE-TANH value is what gets stored and
scored, so log-probabilities are plain Gaussian ones. The tanh Jacobian term is identical
in the numerator and denominator of the PPO ratio and cancels; entropy is likewise the
Gaussian entropy of the pre-tanh distribution, which is the usual practice and is
well-behaved (the exact squashed entropy has no closed form).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Reference interface constants rather than copying them: HOVER_THRUST has drifted once
# already (0.25 -> 0.27) and a stale copy here would silently mis-initialize the thrust
# head of every future run.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from pilot.interface import HOVER_THRUST, OBS_DIM  # noqa: E402

DEFAULT_FRAME_STACK = 6
DEFAULT_THRUST_FLOOR = 0.10  # mirrors surrogate/actions.THRUST_FLOOR; see actionmap.py
ACT_DIM = 3
THRUST_INDEX = 2


def hover_bias_pre_tanh(hover_thrust: float = HOVER_THRUST,
                        thrust_floor: float = DEFAULT_THRUST_FLOOR) -> float:
    """Pre-tanh bias whose tanh maps to hover under the documented affine thrust map.

    train.py overrides this by inverting the *actual* imported mapping numerically
    (`actionmap.ActionMap.invert_thrust`); this closed form keeps the module standalone.
    """
    frac = (float(hover_thrust) - float(thrust_floor)) / (1.0 - float(thrust_floor))
    u = 2.0 * frac - 1.0
    u = float(np.clip(u, -0.999, 0.999))
    return 0.5 * math.log((1.0 + u) / (1.0 - u))


def _mlp(in_dim: int, hidden: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    return nn.Sequential(*layers), d


def _orthogonal(module: nn.Module, gain: float) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    """Tanh-squashed Gaussian policy + value head over a stacked observation."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        frame_stack: int = DEFAULT_FRAME_STACK,
        act_dim: int = ACT_DIM,
        hidden: tuple[int, ...] | list[int] = (256, 256),
        log_std_init: float = -0.5,
        thrust_bias: float | None = None,
        hover_thrust: float = HOVER_THRUST,
        thrust_floor: float = DEFAULT_THRUST_FLOOR,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.frame_stack = int(frame_stack)
        self.act_dim = int(act_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.log_std_init = float(log_std_init)
        self.hover_thrust = float(hover_thrust)
        self.thrust_floor = float(thrust_floor)
        self.thrust_bias = (
            float(thrust_bias)
            if thrust_bias is not None
            else hover_bias_pre_tanh(hover_thrust, thrust_floor)
        )
        self.in_dim = self.obs_dim * self.frame_stack

        self.pi_trunk, d_pi = _mlp(self.in_dim, self.hidden)
        self.vf_trunk, d_vf = _mlp(self.in_dim, self.hidden)
        self.mu = nn.Linear(d_pi, self.act_dim)
        self.v = nn.Linear(d_vf, 1)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), self.log_std_init))

        _orthogonal(self.pi_trunk, math.sqrt(2.0))
        _orthogonal(self.vf_trunk, math.sqrt(2.0))
        nn.init.orthogonal_(self.mu.weight, 0.01)  # near-deterministic initial mean
        nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)
        with torch.no_grad():
            self.mu.bias[THRUST_INDEX] = self.thrust_bias

    # --- config / checkpoint plumbing -------------------------------------------------

    @property
    def net_config(self) -> dict:
        """Exactly the constructor kwargs needed to rebuild this network."""
        return {
            "obs_dim": self.obs_dim,
            "frame_stack": self.frame_stack,
            "act_dim": self.act_dim,
            "hidden": list(self.hidden),
            "log_std_init": self.log_std_init,
            "thrust_bias": self.thrust_bias,
            "hover_thrust": self.hover_thrust,
            "thrust_floor": self.thrust_floor,
        }

    # --- forward paths ----------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mu(self.pi_trunk(x))
        value = self.v(self.vf_trunk(x)).squeeze(-1)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std, value

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.vf_trunk(x)).squeeze(-1)

    def act(self, x: torch.Tensor, deterministic: bool = False):
        """Sample an action.

        Returns (z, u, logp, value): `z` is the pre-tanh sample stored for PPO, `u` is the
        squashed action in [-1,1]^3 handed to the action mapping.
        """
        mean, log_std, value = self.forward(x)
        std = log_std.exp()
        z = mean if deterministic else mean + std * torch.randn_like(mean)
        logp = self._gauss_logp(z, mean, log_std)
        return z, torch.tanh(z), logp, value

    def evaluate(self, x: torch.Tensor, z: torch.Tensor):
        mean, log_std, value = self.forward(x)
        logp = self._gauss_logp(z, mean, log_std)
        entropy = (log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum(-1)
        return logp, entropy, value

    @staticmethod
    def _gauss_logp(z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_std)
        return (-0.5 * ((z - mean) ** 2 / var) - log_std - 0.5 * math.log(2.0 * math.pi)).sum(-1)

    # --- deployment convenience --------------------------------------------------------

    @torch.no_grad()
    def infer(self, stacked_obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """numpy in, numpy out: [n, obs_dim*k] (or [obs_dim*k]) -> u in [-1,1]^3.

        Feed the result to surrogate/actions.policy_to_action to get physical units.
        """
        a = np.asarray(stacked_obs, dtype=np.float32)
        single = a.ndim == 1
        x = torch.as_tensor(np.atleast_2d(a))
        _, u, _, _ = self.act(x, deterministic=deterministic)
        out = u.cpu().numpy().astype(np.float32)
        return out[0] if single else out


# --- checkpoint format ----------------------------------------------------------------
#
# torch.save of a dict with EXACTLY these keys (the eval/deployment sibling codes against
# it):
#   'model'       state_dict
#   'obs_mean'    float32 [73]
#   'obs_std'     float32 [73]
#   'frame_stack' int k
#   'net_config'  dict of ActorCritic constructor kwargs
#   'env_config'  dict of the EnvConfig fields used
#   'step'        int
# plus a JSON sidecar with everything except 'model'.

CHECKPOINT_KEYS = ("model", "obs_mean", "obs_std", "frame_stack", "net_config", "env_config", "step")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def save_checkpoint(
    path: str | Path,
    model: ActorCritic,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    frame_stack: int,
    env_config: dict,
    step: int,
    net_config: dict | None = None,
    extra_json: dict | None = None,
) -> Path:
    """Write `<path>.pt` and the `<path>.json` sidecar. Returns the .pt path.

    `extra_json` goes ONLY into the sidecar, under the key 'harness' — provenance for
    humans (which action mapping was used, which env, the seed). The .pt keeps exactly
    the seven contract keys so that `EnvConfig(**ckpt['env_config'])` stays valid.
    """
    path = Path(path)
    if path.suffix == ".pt":
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)

    mean = np.asarray(obs_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(obs_std, dtype=np.float32).reshape(-1)
    if mean.shape != std.shape:
        raise ValueError(f"obs_mean/obs_std shape mismatch: {mean.shape} vs {std.shape}")
    if mean.size != model.obs_dim:
        raise ValueError(f"obs stats are {mean.size}-D, network expects {model.obs_dim}-D")

    ckpt = {
        "model": model.state_dict(),
        "obs_mean": mean,
        "obs_std": std,
        "frame_stack": int(frame_stack),
        "net_config": _json_safe(net_config if net_config is not None else model.net_config),
        "env_config": _json_safe(env_config),
        "step": int(step),
    }
    pt_path = path.with_suffix(".pt")
    torch.save(ckpt, pt_path)

    sidecar = {k: _json_safe(v) for k, v in ckpt.items() if k != "model"}
    if extra_json:
        sidecar["harness"] = _json_safe(extra_json)
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return pt_path


def load_checkpoint(path: str | Path, map_location: str = "cpu", eval_mode: bool = True):
    """Rebuild (model, checkpoint_dict) from a checkpoint written by save_checkpoint."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    missing = [k for k in CHECKPOINT_KEYS if k not in ckpt]
    if missing:
        raise KeyError(f"checkpoint {path} is missing keys {missing}")
    cfg = dict(ckpt["net_config"])
    cfg["hidden"] = tuple(cfg.get("hidden", (256, 256)))
    model = ActorCritic(**cfg)
    model.load_state_dict(ckpt["model"])
    if eval_mode:
        model.eval()
    return model, ckpt
