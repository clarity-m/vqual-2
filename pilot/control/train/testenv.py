"""Throwaway stub environment: a 2-D point mass that must reach a sequence of targets.

Purpose is verification of THIS harness, independent of the surrogate: it exposes exactly
the `VecSurrogate` API (`obs_dim`, `act_dim`, `reset`, `step`, auto-reset, the same info
keys) and consumes PHYSICAL actions from the same `policy_to_action` mapping, so PPO,
frame stacking, normalization, the curriculum and the checkpoint path all run over the
identical code path they will run over against the real environment.

It is deliberately not a flight model. Three properties are on purpose:
  * mixed observation scales (a pseudo-acceleration channel of order 50 against one-hot
    flags), so a normalization bug shows up as a failure to learn rather than as nothing;
  * a shaped term on the THRUST channel (reward peaks at thrust 0.6, not at the hover
    initialization of 0.27), so a dead third action dimension is visible;
  * `gates_passed` / `collision` / `completed` semantics matching the racing env, so the
    curriculum's completion accounting is exercised.

Not imported by anything that ships.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 0.05
ACC_MAX = 12.0          # m/s^2 at full commanded "rate"
DRAG = 0.25
REACH_BONUS = 5.0
COMPLETE_BONUS = 10.0
COLLISION_PENALTY = -5.0
BOUND_M = 12.0
MAX_STEPS = 200
BEST_THRUST = 0.6
RATE_REF_RPS = 2.75     # matches surrogate/actions.RATE_CAP_RPS, to normalize the input


@dataclass
class TestEnvConfig:
    """Mirrors the fields of `surrogate.env.EnvConfig` that this harness touches."""

    difficulty: float = 0.0
    speed_cap: float = 1.0
    decision_hz_range: tuple[float, float] = (45.0, 65.0)
    n_gates_range: tuple[int, int] = (3, 3)
    enable_time_penalty: bool = False
    progress_gate_scale: float = 1.0

    @classmethod
    def curriculum(cls, difficulty: float, speed_cap: float) -> "TestEnvConfig":
        return cls(difficulty=float(np.clip(difficulty, 0.0, 1.0)),
                   speed_cap=float(np.clip(speed_cap, 1e-3, 1.0)))


class VecPointMass:
    """Vectorized point-mass reach task with the VecSurrogate interface."""

    def __init__(self, n_envs: int, config: TestEnvConfig | None = None, seed: int = 0) -> None:
        self.n_envs = int(n_envs)
        self.config = config if config is not None else TestEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.obs_dim = 8
        self.act_dim = 3
        self.n_targets = int(self.config.n_gates_range[0])

        n = self.n_envs
        self.pos = np.zeros((n, 2), dtype=np.float64)
        self.vel = np.zeros((n, 2), dtype=np.float64)
        self.acc = np.zeros((n, 2), dtype=np.float64)
        self.targets = np.zeros((n, self.n_targets, 2), dtype=np.float64)
        self.t_idx = np.zeros(n, dtype=np.int64)
        self.gates = np.zeros(n, dtype=np.int64)
        self.steps = np.zeros(n, dtype=np.int64)
        self.ep_return = np.zeros(n, dtype=np.float64)
        self.last_thrust = np.full(n, 0.27, dtype=np.float64)
        self.prev_dist = np.zeros(n, dtype=np.float64)

        self.info_gates = np.zeros(n, dtype=np.float64)
        self.info_collision = np.zeros(n, dtype=bool)
        self.info_return = np.zeros(n, dtype=np.float64)
        self.info_length = np.zeros(n, dtype=np.float64)
        self.info_completed = np.zeros(n, dtype=bool)

    # --- helpers -----------------------------------------------------------------------

    @property
    def _tol(self) -> float:
        return 0.75 - 0.25 * float(self.config.difficulty)

    def _spawn(self, idx: np.ndarray) -> None:
        k = idx.size
        if k == 0:
            return
        self.pos[idx] = self.rng.normal(0.0, 0.5, size=(k, 2))
        self.vel[idx] = 0.0
        self.acc[idx] = 0.0
        self.t_idx[idx] = 0
        self.gates[idx] = 0
        self.steps[idx] = 0
        self.ep_return[idx] = 0.0
        self.last_thrust[idx] = 0.27
        # Harder courses put the targets further apart and turn more sharply.
        radius = 3.0 + 2.0 * float(self.config.difficulty)
        prev = self.pos[idx].copy()
        for j in range(self.n_targets):
            ang = self.rng.uniform(-np.pi, np.pi, size=k)
            r = radius * self.rng.uniform(0.8, 1.2, size=k)
            prev = prev + np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
            self.targets[idx, j] = prev
        self.prev_dist[idx] = self._dist(idx)

    def _dist(self, idx: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(self.n_envs) if idx is None else np.asarray(idx).reshape(-1)
        tgt = self.targets[rows, self.t_idx[rows]]
        return np.linalg.norm(tgt - self.pos[rows], axis=-1)

    def _obs(self) -> np.ndarray:
        tgt = self.targets[np.arange(self.n_envs), self.t_idx]
        rel = tgt - self.pos
        dist = np.linalg.norm(rel, axis=-1)
        o = np.stack(
            [
                rel[:, 0],
                rel[:, 1],
                dist,
                self.vel[:, 0],
                self.vel[:, 1],
                np.linalg.norm(self.acc, axis=-1) * 4.0,  # order-50 channel
                self.t_idx / max(self.n_targets, 1),
                self.last_thrust,
            ],
            axis=1,
        )
        noise = 0.05 * float(self.config.difficulty)
        if noise > 0.0:
            o = o + self.rng.normal(0.0, noise, size=o.shape) * np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 0.0, 0.0]
            )
        return o.astype(np.float32)

    # --- API ---------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._spawn(np.arange(self.n_envs))
        self.info_gates[:] = 0.0
        self.info_collision[:] = False
        self.info_return[:] = 0.0
        self.info_length[:] = 0.0
        self.info_completed[:] = False
        return self._obs()

    def step(self, actions: np.ndarray):
        a = np.asarray(actions, dtype=np.float64).reshape(self.n_envs, self.act_dim)
        cmd = np.clip(a[:, :2] / RATE_REF_RPS, -1.0, 1.0)
        thrust = np.clip(a[:, 2], 0.0, 1.0)
        self.last_thrust = thrust

        self.acc = ACC_MAX * cmd - DRAG * self.vel * np.abs(self.vel)
        self.vel = self.vel + DT * self.acc
        self.pos = self.pos + DT * self.vel
        self.steps += 1

        dist = self._dist()
        reward = 1.0 * (self.prev_dist - dist)
        reward -= 0.5 * np.abs(thrust - BEST_THRUST)
        if self.config.enable_time_penalty:
            reward -= 0.02

        reached = dist < self._tol
        reward += REACH_BONUS * reached
        self.gates = self.gates + reached.astype(np.int64)
        self.t_idx = np.minimum(self.t_idx + reached.astype(np.int64), self.n_targets - 1)
        completed = self.gates >= self.n_targets
        reward += COMPLETE_BONUS * completed

        collision = np.linalg.norm(self.pos, axis=-1) > BOUND_M
        reward += COLLISION_PENALTY * collision
        timeout = self.steps >= MAX_STEPS
        done = completed | collision | timeout

        self.ep_return += reward
        self.prev_dist = self._dist()

        if done.any():
            d = np.flatnonzero(done)
            self.info_gates[d] = self.gates[d]
            self.info_collision[d] = collision[d]
            self.info_completed[d] = completed[d]
            self.info_return[d] = self.ep_return[d]
            self.info_length[d] = self.steps[d]
            self._spawn(d)

        info = {
            "gates_passed": self.info_gates.copy(),
            "collision": self.info_collision.copy(),
            "episode_return": self.info_return.copy(),
            "episode_length": self.info_length.copy(),
            "completed": self.info_completed.copy(),
            "n_gates": np.full(self.n_envs, self.n_targets, dtype=np.float64),
        }
        return self._obs(), reward.astype(np.float32), done, info
