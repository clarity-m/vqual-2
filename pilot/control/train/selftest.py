"""Harness verification, independent of the surrogate. Run from the repo root:

    python pilot/control/train/selftest.py

Checks, in order:
  1. FrameStack layout, episode-boundary refill, single-observation (deployment) path.
  2. ObsNormalizer statistics and the frozen-reload path.
  3. The initial mean thrust lands on interface.HOVER_THRUST after the action mapping.
  4. PPO learns the `testenv` point-mass task — mean episode return in the last fifth of
     training must beat the first fifth by a clear margin, and all losses must stay finite.
  5. The checkpoint round-trips: contract keys present, shapes right, reloaded network
     reproduces the trained network's action bit-for-bit, JSON sidecar readable.
  6. Time-limit truncations bootstrap from V(s_T) and genuine terminals do not, including
     the case where a collision and a timeout land on the same step.
  7. A run resumes: it continues the step count and the curriculum rather than restarting,
     appends to its log instead of truncating it, and degrades honestly to weights-only
     when no resume sidecar exists.

Takes ~2 minutes on CPU. Exits non-zero on the first failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pilot.control.train import train as train_mod  # noqa: E402
from pilot.control.train.actionmap import resolve_action_map  # noqa: E402
from pilot.control.train.framestack import FrameStack  # noqa: E402
from pilot.control.train.network import ActorCritic, load_checkpoint  # noqa: E402
from pilot.control.train.normalize import ObsNormalizer  # noqa: E402
from pilot.interface import HOVER_THRUST  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def test_framestack() -> None:
    print("[1] frame stack")
    fs = FrameStack(obs_dim=4, k=3, n_envs=2)
    o1 = np.array([[1, 1, 1, 1], [9, 9, 9, 9]], dtype=np.float32)
    s = fs.reset(o1)
    check("reset fills every slot", s.shape == (2, 12) and np.allclose(s[0], 1.0))
    o2 = np.array([[2, 2, 2, 2], [8, 8, 8, 8]], dtype=np.float32)
    s = fs.push(o2)
    check("newest frame is last", np.allclose(s[0, -4:], 2.0) and np.allclose(s[0, :4], 1.0))
    o3 = np.array([[3, 3, 3, 3], [7, 7, 7, 7]], dtype=np.float32)
    s = fs.push(o3, done=np.array([False, True]))
    check("done env history is dropped", np.allclose(s[1], 7.0))
    check("live env history is kept", np.allclose(s[0], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]))

    single = FrameStack(obs_dim=4, k=3, n_envs=1)
    single.reset(np.arange(4, dtype=np.float32))
    out = single.push(np.arange(4, dtype=np.float32) + 10.0)
    check("single-observation deployment path", out.shape == (1, 12) and np.allclose(out[0, -4:], [10, 11, 12, 13]))


def test_normalize() -> None:
    print("[2] observation normalization")
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 20.0, size=(4000, 7)).astype(np.float32)
    n = ObsNormalizer(7)
    for i in range(0, 4000, 100):
        n.update(x[i : i + 100])
    check("mean recovered", np.allclose(n.mean, x.mean(0), atol=0.2), f"max err {np.abs(n.mean - x.mean(0)).max():.3f}")
    check("std recovered", np.allclose(n.std, x.std(0), rtol=0.05), f"max rel err {np.abs(n.std / x.std(0) - 1).max():.3f}")
    z = n(x[:500])
    check("output is standardized", abs(float(z.mean())) < 0.1 and abs(float(z.std()) - 1.0) < 0.1)
    frozen = ObsNormalizer.from_arrays(n.mean, n.std, obs_dim=7)
    check("frozen reload matches", np.allclose(frozen(x[:100]), n(x[:100]), atol=1e-5))
    before = frozen.mean.copy()
    frozen.update(x[:100] * 100.0)
    check("frozen normalizer ignores updates", np.allclose(frozen.mean, before))


def test_hover_bias() -> None:
    print("[3] thrust head initialization")
    amap = resolve_action_map()
    net = ActorCritic(obs_dim=8, frame_stack=4, act_dim=3,
                      thrust_bias=amap.thrust_pre_tanh_bias(HOVER_THRUST))
    x = torch.zeros(64, 8 * 4)
    mean, _, _ = net.forward(x)
    u = torch.tanh(mean).detach().numpy()
    phys = amap(u.astype(np.float32))
    thrust = float(np.mean(phys[:, 2]))
    check(f"initial mean thrust ~= HOVER_THRUST ({HOVER_THRUST})", abs(thrust - HOVER_THRUST) < 0.01,
          f"got {thrust:.4f} (action map: {amap.source})")
    rates = np.abs(phys[:, :2]).max()
    check("initial rates near zero", rates < 0.2, f"max |rate| {rates:.4f} rad/s")


def test_curriculum() -> None:
    print("[4] curriculum schedules")
    from pilot.control.train.curriculum import Curriculum, CurriculumConfig

    base = dict(window=40, min_episodes=20, hold_updates=1, stall_updates=20)

    never = Curriculum(CurriculumConfig(speed_cap_start=0.5, **base))
    for _ in range(300):
        never.record([False] * 10)
        never.step()
    check("no completions: difficulty stays at the start", never.difficulty == 0.0)
    check("no completions: speed_cap never drops below the start", never.speed_cap >= 0.5,
          f"speed_cap={never.speed_cap:.2f}")
    check("no completions: time penalty stays off", not never.enable_time_penalty)
    # The stall detector still has to FIRE -- that half is load-bearing and is what backs
    # speed_cap off. What it no longer does is touch `progress_gate_scale`: that response
    # assumed a stall means the policy is diving at gates, whereas the measured cause was
    # that completion over 18-22 gates is per-gate-rate^20 and unreachable. It is also
    # the only shaping term that is not potential-based, i.e. the only one that can move
    # the optimal policy rather than just the learning dynamics.
    check("stall detector still fires when completion never moves", never.n_stalls > 0,
          f"n_stalls={never.n_stalls}")
    check("stall leaves the progress term alone", never.progress_gate_scale == 1.0,
          f"progress_gate_scale={never.progress_gate_scale:.2f} after {never.n_stalls} stalls")

    always = Curriculum(CurriculumConfig(speed_cap_start=0.5, **base))
    for _ in range(400):
        always.record([True] * 10)
        always.step()
    check("full completion: difficulty anneals to 1.0", always.difficulty >= 0.999,
          f"{always.difficulty:.2f}")
    check("full completion: speed_cap relaxes to 1.0", always.speed_cap >= 0.999,
          f"{always.speed_cap:.2f}")
    check("full completion: time penalty turns on", always.enable_time_penalty)
    check("full completion: a saturated schedule is not treated as a stall",
          always.progress_gate_scale == 1.0 and always.n_stalls == 0,
          f"progress_gate_scale={always.progress_gate_scale:.2f} stalls={always.n_stalls}")

    middling = Curriculum(CurriculumConfig(speed_cap_start=0.5, **base))
    for i in range(300):
        middling.record([i % 2 == 0] * 10)  # ~50%, between demote and promote
        middling.step()
    check("50% completion: schedules hold", middling.difficulty == 0.0 and middling.speed_cap == 0.5,
          middling.summary())
    check("50% completion: time penalty stays off (not reliable yet)",
          not middling.enable_time_penalty)


def test_ppo_learns() -> dict:
    print("[5] PPO on the point-mass stub (this is the slow one)")
    args = train_mod.parse_args(
        [
            "--env", "testenv",
            "--name", "selftest",
            "--total-steps", "400000",
            "--n-envs", "64",
            "--n-steps", "64",
            "--frame-stack", "4",
            "--hidden", "256,256",
            "--minibatches", "4",
            "--epochs", "4",
            "--lr", "3e-4",
            "--ent-coef", "0.003",
            "--speed-cap-start", "1.0",
            "--no-curriculum",
            "--log-every", "10",
            "--seed", "1",
        ]
    )
    out = train_mod.run(args)
    hist = out["history"]
    rets = [float(r["ep_return"]) for r in hist if r["ep_return"] != ""]
    n = max(len(rets) // 5, 1)
    first, last = float(np.mean(rets[:n])), float(np.mean(rets[-n:]))
    check(
        "mean episode return improves",
        last > first + 5.0,
        f"first fifth {first:.2f} -> last fifth {last:.2f} over {len(rets)} updates",
    )
    gates = [float(r["gates"]) for r in hist if r["gates"] != ""]
    check(
        "targets reached per episode improves",
        float(np.mean(gates[-n:])) > float(np.mean(gates[:n])) + 0.3,
        f"{float(np.mean(gates[:n])):.2f} -> {float(np.mean(gates[-n:])):.2f} of 3",
    )
    finite = all(
        np.isfinite([r["policy_loss"], r["value_loss"], r["entropy"], r["approx_kl"]]).all()
        for r in hist
    )
    check("all losses finite", finite)
    print(
        f"       curve: "
        + " ".join(f"{v:.1f}" for v in rets[:: max(len(rets) // 12, 1)])
        + f"   ({out['elapsed_s']:.0f}s, {hist[-1]['sps']:.0f} FPS)"
    )
    return out


def test_checkpoint(out: dict) -> None:
    print("[6] checkpoint format")
    path = Path(out["checkpoint"])
    check("checkpoint written", path.exists(), str(path))
    if not path.exists():
        return
    raw = torch.load(path, map_location="cpu", weights_only=False)
    want = {"model", "obs_mean", "obs_std", "frame_stack", "net_config", "env_config", "step"}
    check("exact contract keys", set(raw.keys()) == want, f"got {sorted(raw.keys())}")
    check("obs_mean float32 [obs_dim]", raw["obs_mean"].dtype == np.float32 and raw["obs_mean"].shape == (8,))
    check("obs_std float32 [obs_dim]", raw["obs_std"].dtype == np.float32 and raw["obs_std"].shape == (8,))
    check("frame_stack is int", isinstance(raw["frame_stack"], int) and raw["frame_stack"] == 4)
    check("step is int", isinstance(raw["step"], int) and raw["step"] > 0)
    check("env_config carries difficulty/speed_cap",
          {"difficulty", "speed_cap", "enable_time_penalty"} <= set(raw["env_config"]))

    model, ckpt = load_checkpoint(path)
    x = np.random.default_rng(3).normal(size=(5, 8 * 4)).astype(np.float32)
    trained = out["agent"].net
    a_new = model.infer(x)
    with torch.no_grad():
        _, u, _, _ = trained.act(torch.as_tensor(x), deterministic=True)
    check("reloaded network reproduces actions", np.allclose(a_new, u.numpy(), atol=1e-6),
          f"max diff {np.abs(a_new - u.numpy()).max():.2e}")

    side = path.with_suffix(".json")
    check("json sidecar exists", side.exists())
    if side.exists():
        j = json.loads(side.read_text(encoding="utf-8"))
        check("sidecar has everything but the model",
              "model" not in j and {"obs_mean", "obs_std", "frame_stack", "net_config", "env_config", "step"} <= set(j))
        check("sidecar records the action map source",
              j.get("harness", {}).get("action_map_source") in ("surrogate", "fallback"),
              str(j.get("harness", {}).get("action_map_source")))


def test_truncation() -> None:
    print("[7] time-limit truncation bootstrap")
    from pilot.control.train.ppo import PPO, PPOConfig
    from pilot.control.train.testenv import TestEnvConfig, VecPointMass

    n_envs, k, obs_dim = 4, 3, 8
    net = ActorCritic(obs_dim=obs_dim, frame_stack=k, act_dim=3, hidden=(32, 32))
    cfg = PPOConfig(n_steps=256, gamma=0.9, gae_lambda=0.95)
    agent = PPO(net=net, obs_dim=obs_dim, frame_stack=k, n_envs=n_envs, cfg=cfg,
                action_map=resolve_action_map())
    agent.set_env(VecPointMass(n_envs, TestEnvConfig(), seed=0))
    agent._reset()

    # --- which endings bootstrap, and which do not ---------------------------------
    s = agent.stack.get()
    term = np.linspace(-1.0, 1.0, n_envs * obs_dim, dtype=np.float32).reshape(n_envs, obs_dim)
    done = np.array([True, True, True, False])
    info = {
        "terminal_obs": term,
        "timeout":   np.array([True, True, False, True]),   # 3 timed out but is not done
        "collision": np.array([False, True, False, False]),  # 1 also hit a wall
        "completed": np.zeros(n_envs, dtype=bool),
    }
    tv = agent._truncation_values(s, info, done)
    check("bootstraps the truncation", tv is not None and tv[0] != 0.0,
          f"env0 -> {None if tv is None else tv[0]:.4f}")
    check("a coinciding collision wins over the timeout", tv is not None and tv[1] == 0.0)
    check("a non-timeout done stays at zero", tv is not None and tv[2] == 0.0)
    check("a timeout without done stays at zero", tv is not None and tv[3] == 0.0)

    rolled = np.concatenate([s[:, obs_dim:], agent.obs_norm(term)], axis=1)
    with torch.no_grad():
        expect = cfg.gamma * float(net.value(torch.as_tensor(rolled[[0]])).item())
    check("value is gamma * V(s_T) on the rolled stack",
          tv is not None and abs(float(tv[0]) - expect) < 1e-6,
          f"{float(tv[0]):.6f} vs {expect:.6f}")

    check("no timeout key -> legacy bootstrap at zero",
          agent._truncation_values(s, {"terminal_obs": term}, done) is None)
    check("no terminal_obs key -> legacy bootstrap at zero",
          agent._truncation_values(s, {"timeout": info["timeout"]}, done) is None)

    # --- the term reaches the TD error --------------------------------------------
    for buf in (agent.b_rew, agent.b_val, agent.b_done, agent.b_trunc_val):
        buf[:] = 0.0
    t0 = 5
    agent.b_done[t0] = 1.0
    agent.b_rew[t0] = 1.0
    adv_zero, _ = agent._gae()
    agent.b_trunc_val[t0] = 5.0
    adv_boot, _ = agent._gae()
    check("without the term the TD error is reward only", np.allclose(adv_zero[t0], 1.0),
          f"{adv_zero[t0][0]:.4f}")
    check("with it the TD error picks up gamma*V(s_T)", np.allclose(adv_boot[t0], 6.0),
          f"{adv_boot[t0][0]:.4f}")
    check("non-terminal steps are untouched",
          np.allclose(adv_zero[t0 + 1:], adv_boot[t0 + 1:]))

    # --- and it is wired into collect() on the real env ----------------------------
    for buf in (agent.b_trunc_val,):
        buf[:] = 0.0
    agent.collect()          # 256 steps > testenv MAX_STEPS, so timeouts must fire
    nz = np.flatnonzero(agent.b_trunc_val.reshape(-1))
    check("collect() populates the buffer from the live env", nz.size > 0,
          f"{nz.size} truncated transitions in {agent.b_trunc_val.size}")
    check("buffer is nonzero only where an episode ended",
          bool(np.all(agent.b_done.reshape(-1)[nz] == 1.0)))


def test_resume() -> None:
    print("[8] resume round-trip")
    name = "selftest_resume"
    per_update = 64 * 32
    base = ["--env", "testenv", "--name", name, "--n-envs", "32", "--n-steps", "64",
            "--seed", "3", "--quiet"]
    log_path = train_mod.CHECKPOINT_DIR / f"{name}_log.csv"
    if log_path.exists():
        log_path.unlink()

    out1 = train_mod.run(train_mod.parse_args(base + ["--total-steps", str(4 * per_update)]))
    ep1 = out1["curriculum"].n_episodes
    side = train_mod.resume_sidecar_path(Path(out1["checkpoint"]))
    check("resume sidecar written beside the checkpoint", side.exists(), side.name)
    blob = torch.load(side, map_location="cpu", weights_only=False)
    check("sidecar carries what the .pt cannot",
          {"opt", "obs_rms", "rew_rms", "curriculum", "torch_rng", "numpy_rng",
           "global_step", "n_rebuilds"} <= set(blob),
          f"format {blob.get('format')}")
    check("optimizer moments are in it", len(blob["opt"].get("state", {})) > 0)
    check("obs statistics carry their sample count",
          float(blob["obs_rms"]["count"]) > 1.0, f"count={blob['obs_rms']['count']:.0f}")

    # --- continue it ----------------------------------------------------------------
    out2 = train_mod.run(train_mod.parse_args(
        base + ["--total-steps", str(8 * per_update), "--resume", f"{name}_final"]))
    first = out2["history"][0]
    check("continues rather than restarting", first["update"] == 5 and first["step"] == 5 * per_update,
          f"first row update={first['update']} step={first['step']}")
    check("runs the remaining updates, not another full run", len(out2["history"]) == 4,
          f"{len(out2['history'])} updates")
    check("curriculum counters carried across", out2["curriculum"].n_episodes > 1.5 * ep1,
          f"{ep1} -> {out2['curriculum'].n_episodes} episodes")

    rows = [r for r in log_path.read_text(encoding="utf-8").splitlines() if r.strip()]
    heads = [r for r in rows if r.startswith("step,")]
    check("log appended under a single header", len(heads) == 1 and len(rows) == 9,
          f"{len(rows) - 1} data rows, {len(heads)} header(s)")

    # --- and the weights-only path, which is what run1 on disk actually needs --------
    side.unlink()
    out3 = train_mod.run(train_mod.parse_args(
        base + ["--total-steps", str(12 * per_update), "--resume", f"{name}_final"]))
    check("resumes from a bare .pt with no sidecar",
          out3["history"][0]["update"] == 9, f"update {out3['history'][0]['update']}")
    check("weights-only resume still recovers the obs statistics",
          float(out3["agent"].obs_norm.rms.count) >= train_mod.DEGRADED_OBS_COUNT)


def main() -> int:
    print("=== T4 harness self-test ===", flush=True)
    test_framestack()
    test_normalize()
    test_hover_bias()
    test_curriculum()
    out = test_ppo_learns()
    test_checkpoint(out)
    test_truncation()
    test_resume()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
