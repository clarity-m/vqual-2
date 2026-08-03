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

WHICH FRAMES (`offsets`)
------------------------
By default the k frames are CONSECUTIVE decision steps. At 55 Hz that is 0.111 s for
k=6 — but the camera runs at 30.05 Hz, so those six frames contain only ~3.3 distinct
detections, and with `p_detect` at 0.62-0.92 often fewer. Most of the input width is
spent re-reading the same detection.

`offsets` lets the same k frames be spread over a longer history: lags in decision
steps, e.g. `[32, 16, 8, 4, 2, 0]` spans 0.58 s at the SAME k x obs_dim input width and
the same parameter count. Measured by linear probe against true vertical velocity (a
lower bound on what the MLP can extract):

    offsets                span     R2      RMSE
    [0]                    0.000 s  0.417   1.68 m/s
    [5,4,3,2,1,0]          0.110 s  0.460   1.62 m/s   <- default
    [10,8,6,4,2,0]         0.220 s  0.492   1.58 m/s
    [32,16,8,4,2,0]        0.704 s  0.562   1.48 m/s
    [64,32,16,8,4,0]       1.409 s  0.588   1.44 m/s

Real but modest, and it does NOT make vertical rate well observed — the perception-derived
channels carry almost none of that signal at any span (see NETWORK_ARCHITECTURE.md).
Worth taking because it costs nothing, not because it solves the problem.

Default is None = consecutive, which reproduces the previous behaviour exactly.
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

    def __init__(self, obs_dim: int, k: int, n_envs: int = 1, dtype=np.float32,
                 offsets=None) -> None:
        if obs_dim < 1 or k < 1 or n_envs < 1:
            raise ValueError(f"bad FrameStack shape: obs_dim={obs_dim} k={k} n_envs={n_envs}")
        self.obs_dim = int(obs_dim)
        self.k = int(k)
        self.n_envs = int(n_envs)
        self.dtype = dtype

        if offsets is None:
            off = list(range(self.k - 1, -1, -1))       # oldest first: k-1 ... 1, 0
        else:
            off = [int(o) for o in offsets]
            if len(off) != self.k:
                raise ValueError(f"offsets has {len(off)} entries, k is {self.k}")
            if min(off) != 0:
                raise ValueError(f"offsets must include 0 (the current frame), got {off}")
            if len(set(off)) != len(off):
                raise ValueError(f"offsets must be distinct, got {off}")
            off = sorted(off, reverse=True)             # oldest first
        self.offsets = tuple(off)
        self.depth = max(self.offsets) + 1

        # Ring buffer rather than a shift: at n_envs=2048 and depth 65 a per-step shift
        # would copy ~39 MB every decision step to move data that has not changed.
        self._buf = np.zeros((self.depth, self.n_envs, self.obs_dim), dtype=dtype)
        self._head = 0          # index of the MOST RECENT frame
        self._started = False

    @property
    def stacked_dim(self) -> int:
        return self.k * self.obs_dim

    @property
    def span_steps(self) -> int:
        """History covered, in decision steps."""
        return max(self.offsets)

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

        Filling rather than zero-padding means the first `depth-1` steps of an episode are
        not drawn from a distribution the policy never sees again mid-episode.
        """
        a = self._as_batch(obs)
        self._buf[:] = a[None, :, :]
        self._head = 0
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
        self._head = (self._head + 1) % self.depth
        self._buf[self._head] = a
        if done is not None:
            d = np.asarray(done).reshape(-1).astype(bool)
            if d.shape != (self.n_envs,):
                raise ValueError(f"expected done shape ({self.n_envs},), got {d.shape}")
            if d.any():
                self._buf[:, d, :] = a[None, d, :]
        return self.get()

    def get(self) -> np.ndarray:
        """Current stacked view, copied so callers can hold it across pushes."""
        idx = [(self._head - o) % self.depth for o in self.offsets]
        return np.concatenate([self._buf[i] for i in idx], axis=1)

    def clear(self) -> None:
        self._buf[:] = 0.0
        self._head = 0
        self._started = False
