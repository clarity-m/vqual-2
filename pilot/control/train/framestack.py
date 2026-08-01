"""Observation stacking — the policy's short memory.

The architecture requires memory because synthetic (and real) detections drop out and
coast: a memoryless policy flies blind through every dropout. Stacking is chosen over a
GRU for deadline safety — no hidden state to carry, checkpoint or get wrong at
deployment.

THIS CLASS IS SHARED WITH DEPLOYMENT. The layout below is part of the checkpoint
contract; the wrapper in `pilot/control/policies/` must stack with this same class so
that the network sees the same channel order it was trained on.

Layout of `get()`:  [n_envs, k * obs_dim], frames ordered OLDEST FIRST, so the most
recent observation always occupies the last `obs_dim` columns.

Normalization happens BEFORE stacking (see `normalize.py`), so the buffer holds
normalized frames.
"""

from __future__ import annotations

import numpy as np


class FrameStack:
    """Rolling stack of the last `k` observations for `n_envs` parallel envs.

    Deployment use (single stream):

        fs = FrameStack(73, k)
        fs.reset(obs0)              # accepts shape [73] or [1, 73]
        x = fs.push(obs_t)          # -> [1, 73*k], feed straight to the net
    """

    def __init__(self, obs_dim: int, k: int, n_envs: int = 1, dtype=np.float32) -> None:
        if obs_dim < 1 or k < 1 or n_envs < 1:
            raise ValueError(f"bad FrameStack shape: obs_dim={obs_dim} k={k} n_envs={n_envs}")
        self.obs_dim = int(obs_dim)
        self.k = int(k)
        self.n_envs = int(n_envs)
        self.dtype = dtype
        self._buf = np.zeros((self.n_envs, self.k, self.obs_dim), dtype=dtype)
        self._started = False

    @property
    def stacked_dim(self) -> int:
        return self.k * self.obs_dim

    def _as_batch(self, obs) -> np.ndarray:
        a = np.asarray(obs, dtype=self.dtype)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape != (self.n_envs, self.obs_dim):
            raise ValueError(
                f"expected obs shape ({self.n_envs}, {self.obs_dim}), got {tuple(a.shape)}"
            )
        return a

    def reset(self, obs) -> np.ndarray:
        """Begin a fresh episode: every slot holds `obs`.

        Filling rather than zero-padding means the first k-1 steps of an episode are not
        drawn from a distribution the policy never sees again mid-episode.
        """
        a = self._as_batch(obs)
        self._buf[:] = a[:, None, :]
        self._started = True
        return self.get()

    def push(self, obs, done=None) -> np.ndarray:
        """Append the observation at time t.

        `done` flags envs whose episode ended on the transition INTO `obs` — under the
        auto-resetting VecSurrogate that `obs` is already the first frame of a new
        episode, so those stacks are refilled instead of carrying history across the
        episode boundary.
        """
        a = self._as_batch(obs)
        if not self._started:
            return self.reset(a)
        if self.k > 1:
            self._buf[:, :-1] = self._buf[:, 1:]
        self._buf[:, -1] = a
        if done is not None:
            d = np.asarray(done).reshape(-1).astype(bool)
            if d.shape != (self.n_envs,):
                raise ValueError(f"expected done shape ({self.n_envs},), got {d.shape}")
            if d.any():
                self._buf[d] = a[d][:, None, :]
        return self.get()

    def get(self) -> np.ndarray:
        """Current stacked view, copied so callers can hold it across pushes."""
        return self._buf.reshape(self.n_envs, self.stacked_dim).copy()

    def clear(self) -> None:
        self._buf[:] = 0.0
        self._started = False
