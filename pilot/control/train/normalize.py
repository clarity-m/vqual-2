"""Running observation normalization and reward scaling.

Mandatory per the architecture, not tuning: the 73-D vector mixes metres in the tens,
m/s^2 up to ~50, radians, and one-hot flags, and PPO is fragile to that spread.

Statistics are accumulated over RAW observations (73-D, pre-stacking) so that the frozen
checkpoint carries `obs_mean`/`obs_std` of shape [73] regardless of the frame-stack depth.
Deployment reloads them with `ObsNormalizer.from_arrays()` and never updates them.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Chan et al. parallel/batched variance accumulation, in float64."""

    def __init__(self, shape: tuple[int, ...] | int, epsilon: float = 1e-4) -> None:
        shape = (int(shape),) if isinstance(shape, int) else tuple(shape)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot)
        self.mean = new_mean
        self.var = np.maximum(m2 / tot, 0.0)
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64).copy()
        self.var = np.asarray(d["var"], dtype=np.float64).copy()
        self.count = float(d["count"])


class ObsNormalizer:
    """(x - mean) / std, clipped. Update during training, freeze into the checkpoint."""

    def __init__(self, obs_dim: int, clip: float = 10.0, epsilon: float = 1e-6) -> None:
        self.obs_dim = int(obs_dim)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.rms = RunningMeanStd(self.obs_dim)
        self.frozen = False

    def update(self, obs: np.ndarray) -> None:
        if not self.frozen:
            self.rms.update(obs)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        a = np.asarray(obs, dtype=np.float32)
        mean = self.mean
        std = self.std
        out = (a - mean) / std
        return np.clip(out, -self.clip, self.clip, out=out).astype(np.float32, copy=False)

    def freeze(self) -> None:
        self.frozen = True

    @property
    def mean(self) -> np.ndarray:
        return self.rms.mean.astype(np.float32)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.rms.var + self.epsilon).astype(np.float32)

    @classmethod
    def from_arrays(cls, mean, std, obs_dim: int | None = None, clip: float = 10.0) -> "ObsNormalizer":
        """Rebuild a frozen normalizer from checkpoint arrays (deployment path)."""
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        std = np.asarray(std, dtype=np.float64).reshape(-1)
        if mean.shape != std.shape:
            raise ValueError(f"mean/std shape mismatch: {mean.shape} vs {std.shape}")
        if obs_dim is not None and mean.size != obs_dim:
            raise ValueError(f"expected obs_dim {obs_dim}, got {mean.size}")
        n = cls(mean.size, clip=clip)
        n.rms.mean = mean.copy()
        n.rms.var = np.maximum(np.square(std) - n.epsilon, 0.0)
        n.rms.count = 1e6
        n.freeze()
        return n


class RewardScaler:
    """Divide rewards by the running std of the discounted return.

    Keeps the value-function target in a sane range without shifting the reward's zero
    point (which would silently change the sign of terminal penalties).
    """

    def __init__(self, n_envs: int, gamma: float = 0.99, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.n_envs = int(n_envs)
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.rms = RunningMeanStd(1)
        self._ret = np.zeros(self.n_envs, dtype=np.float64)
        self.enabled = True

    def reset(self) -> None:
        self._ret[:] = 0.0

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        r = np.asarray(reward, dtype=np.float64).reshape(-1)
        d = np.asarray(done).reshape(-1).astype(bool)
        if not self.enabled:
            return r.astype(np.float32)
        self._ret = self._ret * self.gamma + r
        self.rms.update(self._ret[:, None])
        scale = float(np.sqrt(self.rms.var[0] + self.epsilon))
        self._ret[d] = 0.0
        out = r / max(scale, 1e-8)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    @property
    def scale(self) -> float:
        return float(np.sqrt(self.rms.var[0] + self.epsilon))
