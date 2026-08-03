"""E5, vectorised: score a policy over many held-out draws as LANES of one surrogate.

`run_policy.run_episode` drives `SingleSurrogate`, which is `VecSurrogate(1, ...)`. That
is the reference implementation and it stays -- but at n=1 every NumPy op pays full
dispatch overhead with nothing to amortise it against, and `build_observation` rebuilds
~30 dataclasses per step. Measured on this repo: 657 steps/s single versus 33,366
env-steps/s at n=1024. Model selection over 20 seeds x 3 draws is ~8 minutes per
candidate serial, against training runs that finish in twenty. The GPU does not help --
the surrogate is NumPy.

    from pilot.control.evalsuite.batch import evaluate_batch
    results = evaluate_batch(policy, config, n_lanes=64, seed=0)
    summary = run_policy.summarize(results)          # same result dicts

## WHAT A LANE IS, AND WHY IT IS NOT A SEED

`SingleSurrogate(cfg, seed)` seeds one whole world. The batched path CANNOT reproduce
those per-seed worlds, and does not pretend to: `VecSurrogate` draws every lane's course,
noise and plant randomisation from ONE shared `default_rng(seed)` in a single vectorised
call, so lane `i` of a batch is a different world from `SingleSurrogate(cfg, i)`. This is
checked, not assumed -- see `agrees_with_reference` below, which compares DISTRIBUTIONS.

What the batched path does guarantee is the property model selection actually needs:

  * **reproducible** -- `(config, n_lanes, seed)` fixes the whole draw list, so the same
    command gives the same numbers;
  * **shared across candidates** -- every candidate flying the same `(config, n_lanes,
    seed)` flies the SAME lane-`i` world, which is exactly what `select.hard_subset`
    assumes when it calls a draw "hard" because several candidates failed it.

The second property is why **only the first episode per lane is scored**. `VecSurrogate`
auto-resets, and a lane's SECOND course is drawn from the shared stream at whatever point
that lane happened to terminate -- which depends on the policy. Episode 1 is drawn during
`reset()`, before any policy action, so it is policy-independent; episode 2 onward is not.
Lanes are therefore frozen as they finish and their later flying is ignored.
"""

import time

import numpy as np

from pilot.control.evalsuite.run_policy import REASON_FLAGS, steps_for

# Info keys the surrogate publishes, in the order `run_policy.REASON_FLAGS` reports them.
_TERMINAL_KEYS = ("finished", "collision", "corridor_exit", "gate_timeout", "timeout")


def _load_vec():
    from pilot.control.surrogate.env import EnvConfig, VecSurrogate
    return VecSurrogate, EnvConfig


def _reason(flags):
    for key, label in REASON_FLAGS:
        if flags.get(key):
            return label
    return "terminated"


def evaluate_batch(policy, config, n_lanes, seed=0, max_steps=None, progress=None):
    """Fly `n_lanes` independent draws at once. Returns `run_episode`-shaped dicts.

    `policy` is the BATCH protocol, not `interface.Policy`:
        `reset(n)`                          -- drop all per-lane history
        `act(obs[n, 73], done[n]) -> [n, 3]`  -- PHYSICAL actions (rad/s, rad/s, thrust)
    `policies/rl.RLBatchPolicy` implements it over a checkpoint. A scalar
    `interface.Policy` can be adapted with `ScalarBatchAdapter` below, which keeps the
    vectorised env but loses most of the speedup to the Python loop.
    """
    VecSurrogate, _ = _load_vec()
    if max_steps is None:
        max_steps = steps_for(config)

    env = VecSurrogate(int(n_lanes), config, int(seed))
    obs = np.asarray(env.reset(), dtype=np.float32)
    n = env.n
    # A policy that needs the env itself (a scalar one rebuilding `interface.Observation`
    # per lane) asks for it here rather than being handed a global.
    if hasattr(policy, "bind"):
        policy.bind(env)
    policy.reset(n)

    # Snapshot the episode's own constants BEFORE any step: `n_gates`, the decision
    # period and the spawn index are all re-drawn by the auto-reset, so reading them
    # later would describe the NEXT episode.
    start_index = env.active.copy()
    n_gates = env.n_gates.copy()
    dt = env.dt_dec.copy()

    live = np.ones(n, dtype=bool)
    sim_s = np.zeros(n)
    steps = np.zeros(n, dtype=np.int64)
    results = [None] * n
    done_prev = np.zeros(n, dtype=bool)
    t0 = time.time()

    for _ in range(int(max_steps)):
        a = policy.act(obs, done=done_prev)
        obs, _rew, done, info = env.step(np.asarray(a, dtype=np.float64))
        obs = np.asarray(obs, dtype=np.float32)
        done = np.asarray(done).reshape(-1).astype(bool)

        sim_s += dt * live
        steps += live
        newly = done & live
        if newly.any():
            gates_abs = np.asarray(info["gates_passed"]).reshape(-1)
            coll_n = np.asarray(info.get("collision_episodes",
                                         np.zeros(n))).reshape(-1)
            flags = {k: np.asarray(info[k]).reshape(-1).astype(bool)
                     for k in _TERMINAL_KEYS if k in info}
            for i in np.flatnonzero(newly):
                results[i] = _lane_result(
                    i, gates_abs[i], coll_n[i], start_index[i], n_gates[i],
                    sim_s[i], steps[i], {k: bool(v[i]) for k, v in flags.items()})
                if progress:
                    progress(results[i])
        live &= ~done
        done_prev = done
        if not live.any():
            break

    # Lanes still flying when the cap hit: report them, do not drop them.
    for i in np.flatnonzero(live):
        results[i] = _lane_result(i, env.active[i], 0, start_index[i], n_gates[i],
                                  sim_s[i], steps[i], {}, reason="max_steps")

    wall = time.time() - t0
    for r in results:
        r["wall_s"] = round(wall / max(n, 1), 4)   # amortised: one batch, n lanes
    return results


def _lane_result(i, gates_abs, coll_n, start_index, n_gates, sim_s, steps, flags,
                 reason=None):
    gates = int(gates_abs) - int(start_index)
    return dict(
        seed=int(i),                       # a LANE index, not a SingleSurrogate seed
        completed=bool(flags.get("finished", False)),
        gates=int(gates),
        gates_needed=max(int(n_gates) - int(start_index), 1),
        gate_index=int(gates_abs),
        start_index=int(start_index),
        n_gates=int(n_gates),
        time_s=round(float(sim_s), 3),
        collisions=max(int(coll_n), 0),
        recoveries=0,
        steps=int(steps),
        reason=reason or _reason(flags),
        wall_s=0.0,
    )


class ScalarBatchAdapter:
    """Run a scalar `interface.Policy` over the lanes of a batched env.

    The env is vectorised but the policy is not, so this keeps only the env's share of
    the speedup -- for the PID baseline the Python loop dominates and the win is small.
    It exists so the batched harness can score the baseline at all, not to make it fast.
    Vectorising `BaselinePolicy` is a separate job.
    """

    def __init__(self, factory):
        self.factory = factory
        self.env = None
        self.pols = []

    def bind(self, env):
        """Called by `evaluate_batch` with the env it owns."""
        self.env = env
        return self

    def reset(self, n):
        self.pols = [self.factory() for _ in range(int(n))]

    def act(self, obs_matrix, done=None):
        from pilot.control.surrogate.env import build_observation
        env = self.env
        n = len(self.pols)
        out = np.zeros((n, 3))
        if done is not None:
            for i in np.flatnonzero(np.asarray(done).reshape(-1).astype(bool)):
                self.pols[i].reset()
        for i in range(n):
            a = self.pols[i](build_observation(env._fields, i))
            out[i] = (a.roll_rate, a.pitch_rate, a.thrust)
        return out


def agrees_with_reference(config, seed=0, n_lanes=64, tol=0.15, verbose=True):
    """Check the batched path against `run_episode` on the SAME policy.

    Per-seed equality is impossible (see the module docstring), so this compares the two
    paths as ESTIMATORS: fly the reference path over `n_lanes` seeds and the batched path
    over `n_lanes` lanes, and require the aggregate rates to agree within `tol`. A sign
    error, a dropped termination flag or a mis-wired action mapping moves these far more
    than sampling noise does.
    """
    from pilot.control.evalsuite.run_policy import evaluate, summarize
    from pilot.control.policies.baseline import BaselinePolicy

    ref = summarize(evaluate(BaselinePolicy(), config, list(range(n_lanes))))
    bat = summarize(evaluate_batch(ScalarBatchAdapter(BaselinePolicy), config,
                                   n_lanes=n_lanes, seed=seed))

    keys = ("completion_rate", "crash_rate", "course_frac_mean")
    ok = True
    for k in keys:
        a, b = float(ref.get(k) or 0.0), float(bat.get(k) or 0.0)
        good = abs(a - b) <= tol
        ok &= good
        if verbose:
            print("  %-18s reference %.3f   batched %.3f   %s"
                  % (k, a, b, "ok" if good else "MISMATCH"))
    if verbose:
        print("  %-18s reference %.2f   batched %.2f"
              % ("gates_mean", ref.get("gates_mean", 0.0), bat.get("gates_mean", 0.0)))
    return bool(ok)
