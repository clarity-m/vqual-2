"""Train/deploy agreement for the two new architecture options.

The failure this guards against is silent: a checkpoint that scores well in training and
flies a DIFFERENT policy live, because the deployment wrapper stacked frames in another
order or applied another thrust mapping. So: save a checkpoint the way train.py does,
reload it the way RLPolicy does, and require identical physical commands.
"""
import os, sys, tempfile
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)
from pilot import interface
from pilot.control.train.network import ActorCritic, save_checkpoint, load_checkpoint
from pilot.control.train.framestack import FrameStack
from pilot.control.train.derived import VerticalRate, augment, N_DERIVED
from pilot.control.train.normalize import ObsNormalizer
from pilot.control.train.actionmap import (resolve_action_map, ResidualThrustMap,
                                           ROLL_INDEX, PITCH_INDEX)
from pilot.control.policies.rl import RLPolicy

D = interface.OBS_DIM
K = 6
rng = np.random.default_rng(0)


def make_obs(i):
    """A realistic-ish observation with a non-trivial attitude."""
    o = interface.Observation()
    o.own.roll_rad = float(0.30 * np.sin(i * 0.11))
    o.own.pitch_rad = float(-0.25 + 0.20 * np.cos(i * 0.07))
    o.own.gyro = rng.normal(0, 0.2, 3)
    o.own.accel = np.array([0.4, -0.2, -9.7]) + rng.normal(0, 0.1, 3)
    o.own.speed_est_mps = 7.0 + rng.normal(0, 0.5)
    o.own.vel_valid = True
    g = o.gates[0]
    g.valid = True; g.index = 3; g.confidence = 0.8
    g.pos_body = np.array([9.0 - 0.05 * i, 0.3, -0.4]) + rng.normal(0, 0.05, 3)
    o.race.active_gate_index = 3
    return o


def run_case(name, offsets, residual, vrate=False):
    amap = resolve_action_map()
    tr = None
    if residual:
        amap = ResidualThrustMap(amap, hover_thrust=interface.HOVER_THRUST, span=0.30)
        tr = {"hover_thrust": interface.HOVER_THRUST, "span": amap.span,
              "min_cos": amap.min_cos}
    net_dim = D + (N_DERIVED if vrate else 0)
    net = ActorCritic(obs_dim=net_dim, frame_stack=K, hidden=(32, 32),
                      thrust_bias=0.0 if residual else None)
    net.eval()

    obs_list = [make_obs(i) for i in range(40)]
    vecs = [np.asarray(o.to_vector(), np.float32) for o in obs_list]
    norm = ObsNormalizer(net_dim)
    vr_fit = VerticalRate(n_envs=1) if vrate else None
    for v in vecs:
        a = v[None, :]
        if vr_fit is not None:
            a = augment(a, vr_fit.step(a))
        norm.update(a)
    norm.freeze()

    # ---- TRAINING-SIDE: FrameStack + net + action map, as ppo does it ----------
    fs = FrameStack(net_dim, K, n_envs=1, offsets=offsets)
    vr = VerticalRate(n_envs=1) if vrate else None
    if vr is not None:
        vr.reset()
    train_cmds = []
    for v in vecs:
        raw = v[None, :]
        if vr is not None:
            raw = augment(raw, vr.step(raw))
        x = fs.push(norm(raw))
        u = net.infer(x, deterministic=True).reshape(-1)[:3]
        if residual:
            a = amap(u.astype(np.float32), roll=v[ROLL_INDEX], pitch=v[PITCH_INDEX])
        else:
            a = amap(u)
        train_cmds.append(np.asarray(a, np.float64).reshape(-1))
    train_cmds = np.array(train_cmds)

    # ---- save exactly as train.py does, reload exactly as RLPolicy does -------
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t")
        save_checkpoint(p, net, norm.mean, norm.std, K, {"difficulty": 0.0}, 1234,
                        frame_offsets=offsets, thrust_residual=tr, vertical_rate=vrate)
        pol = RLPolicy(p + ".pt")
        deploy_cmds = np.array([np.array(
            [(a := pol(o)).roll_rate, a.pitch_rate, a.thrust]) for o in obs_list])
        info = pol.info

    d = np.abs(train_cmds - deploy_cmds).max()
    print("  %-28s max|train-deploy| = %.3e   %s" % (name, d, "OK" if d < 1e-6 else "MISMATCH"))
    if d >= 1e-6:
        print("     train[:3]\n", train_cmds[:3], "\n     deploy[:3]\n", deploy_cmds[:3])
    return d < 1e-6, info


print("TRAIN vs DEPLOY agreement (identical commands required)")
ok = []
o1, i1 = run_case("baseline (unchanged)", None, False); ok.append(o1)
o2, i2 = run_case("dilated stack", [32, 16, 8, 4, 2, 0], False); ok.append(o2)
o3, i3 = run_case("residual thrust", None, True); ok.append(o3)
o4, i4 = run_case("dilated + residual", [32, 16, 8, 4, 2, 0], True); ok.append(o4)
o5, i5 = run_case("vertical rate", None, False, vrate=True); ok.append(o5)
o6, i6 = run_case("all three", [32, 16, 8, 4, 2, 0], True, vrate=True); ok.append(o6)
print()
print("  checkpoint self-describes:", i6.split("| step")[1].strip())

# old checkpoints must still load and behave as before
print()
net = ActorCritic(obs_dim=D, frame_stack=K, hidden=(32, 32))
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "old")
    save_checkpoint(p, net, np.zeros(D, np.float32), np.ones(D, np.float32), K, {}, 7)
    ck = torch.load(p + ".pt", weights_only=False)
    print("  legacy checkpoint has no new keys:",
          "frame_offsets" not in ck and "thrust_residual" not in ck)
    pol = RLPolicy(p + ".pt")
    print("  legacy checkpoint loads and defaults correctly:",
          pol.frame_offsets is None and pol._residual is None and pol._vrate is None)

print()
print("ALL CASES AGREE:", all(ok))
raise SystemExit(0 if all(ok) else 1)
