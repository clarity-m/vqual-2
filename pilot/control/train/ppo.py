"""PPO: vectorized rollout collection + clipped-objective updates.

Standard settings, deliberately: T4 is the only RL-specific stage and the deadline is
real. GAE(lambda), clipped surrogate objective, clipped value loss, entropy bonus,
several minibatch epochs per rollout, global gradient-norm clipping, optional KL early
stop.

The environment contract (`surrogate/env.VecSurrogate`) is auto-resetting, so the
observation returned by `step` for a done env already belongs to the next episode (the
frame stack is refilled for those envs, see framestack.py). The ending observation is not
lost, though: it arrives out of band as `info['terminal_obs']`, which is exactly why
`surrogate/env.py` emits it. So:

  * a TIME-LIMIT truncation (`info['timeout']`) bootstraps from `V(s_T)`, reconstructing
    the terminal frame stack from the pre-step stack plus the terminal observation;
  * a genuine terminal — collision, corridor exit, finish — bootstraps at zero, and wins
    when it coincides with a timeout on the same step (env.py combines the flags with OR);
  * an environment that reports neither key falls back to bootstrapping every done at
    zero, which is what `testenv` did before it published the flags.

An earlier version of this file recorded that the terminal observation was "not
recoverable" and accepted the resulting value bias. That was never true of this env, and
the note is why it went unnoticed: it read as a considered trade-off rather than a
dropped hand-off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .framestack import FrameStack
from .network import ActorCritic
from .normalize import ObsNormalizer, RewardScaler


@dataclass
class PPOConfig:
    n_steps: int = 128          # rollout length per env
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    n_minibatches: int = 4
    max_grad_norm: float = 0.5
    clip_vloss: bool = True
    normalize_adv: bool = True
    target_kl: float | None = 0.03  # None disables the early stop


@dataclass
class RolloutStats:
    steps: int = 0
    sps: float = 0.0
    raw_reward_mean: float = 0.0
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[float] = field(default_factory=list)
    gates_passed: list[float] = field(default_factory=list)
    collisions: list[bool] = field(default_factory=list)
    # Collisions PER EPISODE, from the env's own counter. `collisions` above is the
    # terminal `collision` flag sampled on `done`, which only means anything while
    # `EnvConfig.collision_terminates` is True -- with contact non-terminal an episode
    # ends on a timeout or a corridor exit, and the flag reads ~0 no matter how many
    # times the aircraft hit something. That is exactly the regime where collisions are
    # the thing worth watching, so count them properly.
    collision_counts: list[float] = field(default_factory=list)
    completions: list[bool] = field(default_factory=list)
    # Active-gate plane crossings and clean passes, per finished episode. Their pooled
    # ratio is the per-gate accuracy the curriculum promotes on: whole-course completion
    # is that ratio raised to the gate count, which stays at 0.000 long after the ratio
    # has started moving, and is why `run1` never promoted in 77 updates.
    gate_passes: list[float] = field(default_factory=list)
    gate_attempts: list[float] = field(default_factory=list)


def info_array(info: dict, key: str, n_envs: int) -> np.ndarray | None:
    """Pull a per-env array out of the env's info dict, tolerantly."""
    if not isinstance(info, dict) or key not in info:
        return None
    v = info[key]
    try:
        a = np.asarray(v).reshape(-1)
    except Exception:
        return None
    return a if a.size == n_envs else None


class PPO:
    def __init__(
        self,
        net: ActorCritic,
        obs_dim: int,
        frame_stack: int,
        n_envs: int,
        cfg: PPOConfig,
        action_map,
        obs_norm: ObsNormalizer | None = None,
        rew_scaler: RewardScaler | None = None,
        device: str = "cpu",
        speed_cap: float = 1.0,
        completion_fn=None,
    ) -> None:
        self.net = net
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.n_envs = int(n_envs)
        self.action_map = action_map
        self.speed_cap = float(speed_cap)
        self.completion_fn = completion_fn

        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
        self.stack = FrameStack(self.obs_dim, frame_stack, self.n_envs)
        self.obs_norm = obs_norm if obs_norm is not None else ObsNormalizer(self.obs_dim)
        self.rew_scaler = (
            rew_scaler if rew_scaler is not None else RewardScaler(self.n_envs, gamma=cfg.gamma)
        )

        T, N, D = cfg.n_steps, self.n_envs, self.stack.stacked_dim
        A = net.act_dim
        self.b_obs = np.zeros((T, N, D), dtype=np.float32)
        self.b_act = np.zeros((T, N, A), dtype=np.float32)  # pre-tanh samples
        self.b_logp = np.zeros((T, N), dtype=np.float32)
        self.b_val = np.zeros((T, N), dtype=np.float32)
        self.b_rew = np.zeros((T, N), dtype=np.float32)
        self.b_done = np.zeros((T, N), dtype=np.float32)
        # gamma * V(s_T) where a time limit cut the episode; zero at genuine terminals and
        # at every non-terminal step, so it can be added to delta unconditionally.
        self.b_trunc_val = np.zeros((T, N), dtype=np.float32)

        self.env = None
        self._needs_reset = True
        self.global_step = 0

    # --- env plumbing ------------------------------------------------------------------

    def set_env(self, env) -> None:
        """(Re)attach an environment. The curriculum rebuilds it when config changes."""
        if getattr(env, "obs_dim", self.obs_dim) != self.obs_dim:
            raise ValueError(f"env.obs_dim={env.obs_dim} but harness expects {self.obs_dim}")
        if getattr(env, "act_dim", self.net.act_dim) != self.net.act_dim:
            raise ValueError(f"env.act_dim={env.act_dim} but network has {self.net.act_dim}")
        self.env = env
        self._needs_reset = True

    def _reset(self) -> None:
        obs = np.asarray(self.env.reset(), dtype=np.float32)
        self.obs_norm.update(obs)
        self.stack.reset(self.obs_norm(obs))
        self.rew_scaler.reset()
        self._needs_reset = False

    def _to_physical(self, u: np.ndarray) -> np.ndarray:
        """Squashed action -> physical units, then apply the speed curriculum.

        `speed_cap` scales the two RATE channels only. Capping thrust would move the
        aircraft's trim point and is not what "cap commanded aggressiveness" means.
        """
        a = self.action_map(u)
        a = np.array(a, dtype=np.float32, copy=True).reshape(u.shape)
        a[:, :2] *= self.speed_cap
        return a

    # --- rollout -----------------------------------------------------------------------

    @torch.no_grad()
    def collect(self) -> RolloutStats:
        if self.env is None:
            raise RuntimeError("no environment attached: call set_env() first")
        if self._needs_reset:
            self._reset()

        cfg = self.cfg
        stats = RolloutStats()
        raw_rew_sum = 0.0
        t0 = time.perf_counter()

        for t in range(cfg.n_steps):
            s = self.stack.get()
            x = torch.as_tensor(s, device=self.device)
            z, u, logp, value = self.net.act(x)
            u_np = u.cpu().numpy().astype(np.float32)

            obs, rew, done, info = self.env.step(self._to_physical(u_np))
            obs = np.asarray(obs, dtype=np.float32)
            rew = np.asarray(rew, dtype=np.float32).reshape(-1)
            done = np.asarray(done).reshape(-1).astype(bool)

            self.b_obs[t] = s
            self.b_act[t] = z.cpu().numpy()
            self.b_logp[t] = logp.cpu().numpy()
            self.b_val[t] = value.cpu().numpy()
            self.b_rew[t] = self.rew_scaler(rew, done)
            self.b_done[t] = done.astype(np.float32)
            raw_rew_sum += float(rew.mean())

            # Before the stack rolls: `s` is still the pre-step stack, which is what the
            # terminal frame has to be appended to.
            self.b_trunc_val[t] = 0.0
            if done.any():
                tv = self._truncation_values(s, info, done)
                if tv is not None:
                    self.b_trunc_val[t] = tv

            self.obs_norm.update(obs)
            self.stack.push(self.obs_norm(obs), done=done)

            if done.any():
                self._record_episodes(stats, info, done)

        self.global_step += cfg.n_steps * self.n_envs
        stats.steps = cfg.n_steps * self.n_envs
        stats.raw_reward_mean = raw_rew_sum / max(cfg.n_steps, 1)
        stats.sps = stats.steps / max(time.perf_counter() - t0, 1e-9)
        return stats

    @torch.no_grad()
    def _truncation_values(self, s: np.ndarray, info: dict, done: np.ndarray) -> np.ndarray | None:
        """`gamma * V(s_T)` for envs a time limit cut off; zero everywhere else.

        Returns None when the environment reports no truncation information at all, which
        restores the old bootstrap-at-zero behaviour rather than guessing.
        """
        term = info.get("terminal_obs") if isinstance(info, dict) else None
        timeout = info_array(info, "timeout", self.n_envs)
        if term is None or timeout is None:
            return None

        trunc = done & timeout.astype(bool)
        # A genuine terminal wins: env.py ORs the flags, so one step can be both a
        # collision and a timeout, and that episode's future value really is zero.
        # "finished" is the surrogate's name for it, "completed" is testenv's.
        # `gate_timeout` is a genuine terminal, not a time limit: the policy failed to
        # reach its gate, and its future really is worth zero. The surrogate used to OR it
        # into `timeout`, so this bootstrapped a stuck policy with the value of the course
        # it was never going to fly.
        for key in ("collision", "corridor_exit", "finished", "completed", "gate_timeout"):
            flag = info_array(info, key, self.n_envs)
            if flag is not None:
                trunc &= ~flag.astype(bool)
        idx = np.flatnonzero(trunc)
        if idx.size == 0:
            return None

        term = np.asarray(term, dtype=np.float32).reshape(self.n_envs, self.obs_dim)
        # Roll one frame into the pre-step stack. The layout is oldest-first with the most
        # recent observation last (framestack.py), so this is exactly the stack the policy
        # would have seen at s_T. `s[:, D:]` is empty at k == 1, which is correct.
        d = self.obs_dim
        stacked = np.concatenate([s[:, d:], self.obs_norm(term)], axis=1)

        # `obs_norm` is applied but deliberately not updated with the terminal frames: the
        # normalizer's update stream stays identical to before this change, so the effect
        # of the bootstrap is attributable to the bootstrap alone.
        v = self.net.value(torch.as_tensor(stacked[idx], device=self.device)).cpu().numpy()
        out = np.zeros(self.n_envs, dtype=np.float32)
        out[idx] = self.cfg.gamma * v.astype(np.float32)
        return out

    def _record_episodes(self, stats: RolloutStats, info: dict, done: np.ndarray) -> None:
        n = self.n_envs
        ep_ret = info_array(info, "episode_return", n)
        ep_len = info_array(info, "episode_length", n)
        # `gates_this_episode` where the env reports it: `gates_passed` is the ABSOLUTE
        # gate index and 25% of episodes spawn mid-course, so it credits the policy with
        # the spawn offset. At update 1 of `run1` a freshly initialised network logged
        # 2.05 "gates", which is exactly the mean offset and nothing the policy did.
        gates = info_array(info, "gates_this_episode", n)
        if gates is None:
            gates = info_array(info, "gates_passed", n)
        coll = info_array(info, "collision", n)
        gp = info_array(info, "gate_passes", n)
        ga = info_array(info, "gate_attempts", n)
        if gp is not None and ga is not None:
            stats.gate_passes += [float(v) for v in gp[done]]
            stats.gate_attempts += [float(v) for v in ga[done]]
        if ep_ret is not None:
            stats.episode_returns += [float(v) for v in ep_ret[done]]
        if ep_len is not None:
            stats.episode_lengths += [float(v) for v in ep_len[done]]
        if gates is not None:
            stats.gates_passed += [float(v) for v in gates[done]]
        if coll is not None:
            stats.collisions += [bool(v) for v in coll[done]]
        cn = info_array(info, "collision_episodes", n)
        if cn is not None:
            stats.collision_counts += [float(v) for v in cn[done]]
        if self.completion_fn is not None:
            stats.completions += [bool(v) for v in self.completion_fn(info, done)]

    # --- update ------------------------------------------------------------------------

    def _gae(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        with torch.no_grad():
            last_val = (
                self.net.value(torch.as_tensor(self.stack.get(), device=self.device))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        adv = np.zeros_like(self.b_rew)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(cfg.n_steps)):
            nonterminal = 1.0 - self.b_done[t]
            next_val = last_val if t == cfg.n_steps - 1 else self.b_val[t + 1]
            # `b_trunc_val` is nonzero only where `nonterminal` is 0, so exactly one of the
            # two bootstrap terms fires. The recursion still cuts on `nonterminal`: the
            # episode boundary is real even when the value bootstraps across it.
            delta = (self.b_rew[t] + cfg.gamma * next_val * nonterminal
                     + self.b_trunc_val[t] - self.b_val[t])
            last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
            adv[t] = last_gae
        return adv, adv + self.b_val

    def update(self) -> dict:
        cfg = self.cfg
        adv, ret = self._gae()

        D = self.stack.stacked_dim
        A = self.net.act_dim
        obs = torch.as_tensor(self.b_obs.reshape(-1, D), device=self.device)
        act = torch.as_tensor(self.b_act.reshape(-1, A), device=self.device)
        logp_old = torch.as_tensor(self.b_logp.reshape(-1), device=self.device)
        val_old = torch.as_tensor(self.b_val.reshape(-1), device=self.device)
        adv_t = torch.as_tensor(adv.reshape(-1), device=self.device)
        ret_t = torch.as_tensor(ret.reshape(-1), device=self.device)

        batch = obs.shape[0]
        mb_size = max(batch // max(cfg.n_minibatches, 1), 1)
        idx = np.arange(batch)
        rng = np.random

        pg_losses, v_losses, entropies, kls, clipfracs = [], [], [], [], []
        stop = False
        for _ in range(cfg.epochs):
            rng.shuffle(idx)
            for start in range(0, batch, mb_size):
                mb = torch.as_tensor(idx[start : start + mb_size], device=self.device)
                logp, entropy, value = self.net.evaluate(obs[mb], act[mb])
                log_ratio = logp - logp_old[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    kls.append(float(approx_kl))
                    clipfracs.append(float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean()))

                mb_adv = adv_t[mb]
                if cfg.normalize_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()

                if cfg.clip_vloss:
                    v_clipped = val_old[mb] + torch.clamp(
                        value - val_old[mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        (value - ret_t[mb]) ** 2, (v_clipped - ret_t[mb]) ** 2
                    ).mean()
                else:
                    v_loss = 0.5 * ((value - ret_t[mb]) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                pg_losses.append(pg_loss.detach().item())
                v_losses.append(v_loss.detach().item())
                entropies.append(ent.detach().item())

            if cfg.target_kl is not None and kls and abs(kls[-1]) > cfg.target_kl:
                stop = True
                break

        y = ret.reshape(-1)
        var_y = float(np.var(y))
        explained_var = (
            float(1.0 - np.var(y - self.b_val.reshape(-1)) / var_y) if var_y > 1e-12 else 0.0
        )
        return {
            "policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clip_frac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_var": explained_var,
            "early_stop": bool(stop),
            "adv_std": float(np.std(adv)),
            "reward_scale": float(self.rew_scaler.scale),
        }
