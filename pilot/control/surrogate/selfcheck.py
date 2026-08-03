"""P2 self-check. Repo style: prints PASS or FAIL per stage, exits non-zero on any FAIL.

    python3 pilot/control/surrogate/selfcheck.py

Six tests, in the order a failure would matter:

1. **Vector equivalence.** The fast 73-D encoder against `interface.Observation.to_vector()`
   on randomised states, EXACTLY. The encoder is the one thing every consumer of this
   package depends on and the one thing a hand-written index slip would silently corrupt.
2. **Physics.** Hover holds altitude, a cut throttle descends, and the integrator
   reproduces `plant.Sim` to float precision -- with the -1 rate mirror applied exactly
   once, in the right place. The control that gives that test teeth is printed too: feed
   `plant.Sim` the UNmirrored command and it must diverge.
3. **Camera.** Body-forward lands at v = 296 of 360, i.e. BELOW image centre, and gates
   behind or outside the frustum are never visible.
4. **A flight.** A scripted guidance controller flies a gentle course, crosses gates,
   `active_gate_index` advances and the return is positive.
5. **Throughput** at n_envs = 1024.
6. **Single == Vec.** Same seed, same config, same actions, same trajectory.
7. **`progress_gate_scale`.** Two envs on a bit-identical trajectory, scale 1.0 against
   0.0: the reward may differ only on near-gate steps, and the crossing count may not
   differ at all.
8. **`terminal_obs`.** At a done index it is the ending episode's observation, not the
   fresh one; at every other index it is the observation being returned.
9. **Detection statistics**, report only. Nothing measured exists to assert against until
   P3 lands, but "the detector silently emits nothing" and "the detector is clean" are
   both easy to ship and both fatal, and neither shows up as a FAIL above.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import interface  # noqa: E402
import plant as plant_mod  # noqa: E402

import actions as act  # noqa: E402
import camera  # noqa: E402
import vmath  # noqa: E402
from attention import AttnOutput  # noqa: E402
from env import EnvConfig, VecSurrogate, build_observation  # noqa: E402
from single import SingleSurrogate  # noqa: E402

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%-28s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
    return ok


# =========================================================================
# 1 -- vector equivalence
# =========================================================================
def test_vector_equivalence():
    print("\n1. vector equivalence -- fast encoder vs interface.Observation.to_vector()")
    env = VecSurrogate(48, EnvConfig.curriculum(0.8, 1.0), seed=11)
    rng = np.random.default_rng(3)
    worst = 0.0
    checked = 0
    obs = env.reset()
    for k in range(60):
        u = rng.uniform(-1, 1, (env.n, 3)).astype(np.float32)
        obs, _, _, _ = env.step(act.policy_to_action(u))
        if k % 12 != 0:
            continue
        for i in range(env.n):
            ref = build_observation(env._fields, i).to_vector()
            d = float(np.max(np.abs(ref - obs[i])))
            worst = max(worst, d)
            checked += 1
    ok = worst == 0.0 and obs.shape == (48, interface.OBS_DIM) and obs.dtype == np.float32
    return report("vector equivalence", ok,
                  "%d observations, max |fast - dataclass| = %.1e, shape %s %s"
                  % (checked, worst, obs.shape, obs.dtype))


# =========================================================================
# 2 -- physics
# =========================================================================
def _bare_env(seed=0, hz=40.0):
    """One env with every randomisation off and dt_sub pinned to plant.Sim's 1/120 s."""
    cfg = EnvConfig(difficulty=0.0, randomize_plant=False,
                    decision_hz_range=(hz, hz), substeps=3)
    env = VecSurrogate(1, cfg, seed=seed)
    return env


def test_physics():
    print("\n2. physics -- hover, throttle cut, and plant.Sim equivalence")
    p = plant_mod.Plant.load(os.path.join(os.path.dirname(_HERE), "plant.json"))
    ok_all = True

    # -- hover holds altitude ------------------------------------------------
    env = _bare_env()
    env.p[:] = np.array([0.0, 0.0, -5.0])
    env.v[:] = 0.0
    env.q[:] = np.array([1.0, 0.0, 0.0, 0.0])
    thr = p.hover_throttle()
    for _ in range(int(3.0 * 40)):
        env._substep(np.zeros((1, 3)), np.full(1, thr))
        env._substep(np.zeros((1, 3)), np.full(1, thr))
        env._substep(np.zeros((1, 3)), np.full(1, thr))
    dz_hover = float(env.p[0, 2] + 5.0)
    ok = abs(dz_hover) < 0.6
    ok_all &= report("  hover holds altitude", ok,
                     "throttle %.3f (plant hover), 3.0 s, altitude change %+.3f m"
                     % (thr, -dz_hover))

    # -- throttle cut descends ----------------------------------------------
    env = _bare_env()
    env.p[:] = np.array([0.0, 0.0, -20.0])
    env.v[:] = 0.0
    env.q[:] = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(int(1.5 * 120)):
        env._substep(np.zeros((1, 3)), np.full(1, act.THRUST_FLOOR))
    drop = float(env.p[0, 2] + 20.0)
    vz = float(env.v[0, 2])
    ok = drop > 1.0 and vz > 1.0
    ok_all &= report("  throttle cut descends", ok,
                     "1.5 s at the %.2f floor: fell %.2f m, w = %+.2f m/s (down)"
                     % (act.THRUST_FLOOR, drop, vz))

    # -- the integrator equals plant.Sim, mirror applied exactly once --------
    env = _bare_env()
    env.dt_sub[:] = 1.0 / 120.0
    env.rate_k[:] = max(1, int(round(p.rate_delay * 120)))
    env.thr_k[:] = max(1, int(round(p.thrust_delay * 120)))
    env.p[:] = 0.0
    env.v[:] = np.array([6.0, -2.0, 1.0])
    env.q[:] = vmath.quat_from_euler(np.array([0.15]), np.array([-0.25]), np.array([0.6]))[0]
    sim = plant_mod.Sim(p, dt=1 / 120.0)
    sim.reset(p=env.p[0], v=env.v[0], q=env.q[0])
    sim_bad = plant_mod.Sim(p, dt=1 / 120.0)
    sim_bad.reset(p=env.p[0], v=env.v[0], q=env.q[0])

    rng = np.random.default_rng(7)
    worst = 0.0
    worst_bad = 0.0
    for k in range(400):
        cmd = np.array([1.2 * np.sin(0.07 * k), -0.8 * np.cos(0.05 * k),
                        0.4 * np.sin(0.11 * k)])
        thr = 0.25 + 0.15 * np.sin(0.03 * k) + 0.02 * rng.normal()
        env._substep(cmd[None, :], np.array([thr]))
        # plant.Sim consumes cmd.csv-convention rates and applies the -1 mirror itself,
        # so the canonical NED command has to be mirrored on the way IN. Zero flips or
        # two flips is the silent sign error README.md warns about.
        sim.step(-cmd, thr)
        sim_bad.step(cmd, thr)
        worst = max(worst, float(np.max(np.abs(env.p[0] - sim.p))),
                    float(np.max(np.abs(env.v[0] - sim.v))),
                    float(np.max(np.abs(env.q[0] - sim.q))))
        worst_bad = max(worst_bad, float(np.max(np.abs(env.p[0] - sim_bad.p))))
    ok = worst < 1e-9 and worst_bad > 1.0
    ok_all &= report("  == plant.Sim (mirror once)", ok,
                     "400 steps: max |surrogate - Sim(-cmd)| = %.2e m/(m/s)/quat;"
                     " control Sim(+cmd) diverges %.1f m" % (worst, worst_bad))

    # -- terminal climb, the closed form plant.py itself checks ---------------
    env = _bare_env()
    env.dt_sub[:] = 1.0 / 120.0
    env.p[:] = 0.0
    env.v[:] = 0.0
    env.q[:] = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(7200):
        env._substep(np.zeros((1, 3)), np.ones(1))
    climb = -float(env.v[0, 2])
    closed = float(np.sqrt((p.thrust(1.0) - plant_mod.G) / p.kz))
    ok = abs(climb - closed) < 0.05 * closed
    ok_all &= report("  terminal climb", ok,
                     "integrated %.2f m/s vs closed form %.2f m/s" % (climb, closed))
    return ok_all


# =========================================================================
# 3 -- camera
# =========================================================================
def test_camera():
    print("\n3. camera -- the 20 deg UP tilt, and what that does to the lower frame edge")
    ok_all = True

    u, v, z = camera.project(np.array([10.0, 0.0, 0.0]))
    ok = abs(u - 320.0) < 1e-6 and abs(v - 296.46) < 0.1 and z > 0
    ok_all &= report("  forward -> v = 296", ok,
                     "gate dead ahead at own altitude: u %.2f, v %.2f of 360 (BELOW"
                     " centre 180)" % (u, v))

    lo, hi = camera.vertical_span_rad()
    ok = abs(np.degrees(hi) - 49.4) < 0.1 and abs(np.degrees(lo) + 9.4) < 0.1
    ok_all &= report("  vertical span", ok,
                     "+%.1f deg / %.1f deg about body-forward"
                     % (np.degrees(hi), np.degrees(lo)))

    behind = np.array([[-10.0, 0.0, 0.0], [-3.0, 1.0, -1.0], [-30.0, 5.0, 2.0]])
    ok = not np.any(camera.in_frustum(behind))
    ok_all &= report("  behind is invisible", ok, "3 gates behind the drone, none in frustum")

    # Just outside each edge, and just inside, on the tight lower edge.
    d = 12.0
    below = np.array([d * np.cos(np.radians(-11.0)), 0.0, -d * np.sin(np.radians(-11.0))])
    inside = np.array([d * np.cos(np.radians(-8.0)), 0.0, -d * np.sin(np.radians(-8.0))])
    side_out = np.array([d, d * 1.2, 0.0])
    above = np.array([d * np.cos(np.radians(52.0)), 0.0, -d * np.sin(np.radians(52.0))])
    ok = (not camera.in_frustum(below)) and camera.in_frustum(inside) \
        and (not camera.in_frustum(side_out)) and (not camera.in_frustum(above))
    ok_all &= report("  frustum edges", ok,
                     "-11 deg out, -8 deg in, +52 deg out, 50 deg off-axis out")

    # The calibrated rangefinder, spec-exact on the 1500 mm aperture.
    px = camera.size_px_from_range(20.0, 1.0)
    r = camera.range_from_size_px(px)
    ok = abs(px - 24.0) < 1e-6 and abs(r - 20.0) < 1e-6
    ok_all &= report("  480/gate_px rangefinder", ok,
                     "20 m -> %.2f px -> %.3f m (fronto-parallel)" % (px, r))
    return ok_all


# =========================================================================
# 4 -- a scripted flight
# =========================================================================
class _GateAttention:
    """Attention stub that points the nose at the TRUE active gate.

    Privileged, and used only to keep this test about the dynamics and the course rather
    than about the detector's dropout draw. It also exercises the swap path that the real
    attention module will use (`EnvConfig.attention_factory`).
    """

    def __init__(self):
        self.env = None

    def reset(self, n=1, mask=None):
        pass

    def step_batch(self, det, dt):
        n = det.valid.shape[0]
        if self.env is None:
            d = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        else:
            e = self.env
            g = e.g_pos[e._rows, np.clip(e.active, 0, e.n_slots_course - 1)]
            r = vmath.quat_to_rot(e.q)
            d = vmath.unit(np.einsum('nji,nj->ni', r, g - e.p))
        az = np.arctan2(d[:, 1], d[:, 0])
        return AttnOutput(kind=np.full(n, int(interface.Attn.GATE_CURRENT)),
                          target_dir_body=d, target_gate_index=det.gate_index[:, 0],
                          yaw_rate=np.clip(2.5 * az, -3.0, 3.0))


def scripted_action(env, cruise=9.0):
    """PRIVILEGED guidance: fly at the true approach point at `cruise` m/s.

    A cascaded velocity -> tilt -> body-rate controller, i.e. what the reactive baseline
    does with detections instead of truth. Here it exists only to prove the surrogate is
    flyable end to end and that the reward accumulates on a flown course.
    """
    a = np.clip(env.active, 0, env.n_slots_course - 1)
    gp = env.g_pos[env._rows, a]
    gn = env.g_nrm[env._rows, a]
    tgt = gp - gn * env.cfg.approach_d_m
    s = np.einsum('ni,ni->n', env.p - gp, gn)
    tgt = np.where((s > -env.cfg.approach_d_m)[:, None], gp + gn * 6.0, tgt)

    v_des = cruise * vmath.unit(tgt - env.p)
    a_des = 2.2 * (v_des - env.v)
    f_world = a_des - np.array([0.0, 0.0, plant_mod.G])       # thrust must carry gravity
    T = np.linalg.norm(f_world, axis=1)
    zb = -f_world / np.maximum(T, 1e-6)[:, None]

    r = vmath.quat_to_rot(env.q)
    _, _, yaw = vmath.euler_from_rot(r)
    cy, sy = np.cos(yaw), np.sin(yaw)
    xp = zb[:, 0] * cy + zb[:, 1] * sy
    yp = -zb[:, 0] * sy + zb[:, 1] * cy
    roll_d = np.arcsin(np.clip(-yp, -0.9, 0.9))
    pitch_d = np.arctan2(xp, np.maximum(zb[:, 2], 0.2))

    roll, pitch, _ = vmath.euler_from_rot(r)
    rr = 4.0 * _wrap(roll_d - roll)
    pr = 4.0 * _wrap(pitch_d - pitch)

    thr = env.throttle_for_thrust(T)
    return np.stack([rr, pr, thr], axis=1).astype(np.float32)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _fly(cfg, seed, steps, n=16):
    """Fly `n` envs for `steps` decisions and return what happened. Auto-reset is left on."""
    at = _GateAttention()
    cfg = cfg.replace(attention_factory=lambda: at)
    env = VecSurrogate(n, cfg, seed=seed)
    at.env = env
    env.reset()
    best = 0
    rets, gates, why = [], [], dict(collision=0, corridor_exit=0, timeout=0, finished=0)
    rew_sum = 0.0
    for _ in range(steps):
        _, rew, done, info = env.step(scripted_action(env))
        rew_sum += float(np.sum(rew))
        best = max(best, int(np.max(info["gates_passed"])))
        if np.any(done):
            rets.extend(info["episode_return"][done].tolist())
            gates.extend(info["gates_passed"][done].tolist())
            for k in why:
                why[k] += int(np.count_nonzero(info[k] & done))
    return dict(best=best, rets=rets, gates=gates, why=why,
                per_step=rew_sum / (steps * n), n_done=len(rets))


def test_flight():
    print("\n4. a scripted flight -- gentle course, gates crossed, positive return")
    base = EnvConfig.curriculum(0.0, 1.0).replace(
        start_probs=(1.0, 0.0, 0.0, 0.0, 0.0))     # gate-following starts only
    ok_all = True

    straight = base.replace(turn_deg_easy=0.0, elev_deg_easy=0.0, turn_min_frac=0.0)
    r = _fly(straight, seed=5, steps=2500, n=16)
    ok = r["best"] >= 3 and r["per_step"] > 0.0
    ok_all &= report("  straight course", ok,
                     "%d gates crossed, mean reward %+.3f/step, %d episodes ended %s"
                     % (r["best"], r["per_step"], r["n_done"], r["why"]))

    r = _fly(base, seed=6, steps=2500, n=16)
    ok = r["best"] >= 3 and r["per_step"] > 0.0
    ok_all &= report("  winding gentle course", ok,
                     "%d gates crossed, mean reward %+.3f/step, %d episodes ended %s"
                     % (r["best"], r["per_step"], r["n_done"], r["why"]))

    r = _fly(EnvConfig.curriculum(1.0, 1.0), seed=7, steps=2500, n=16)
    ok_all &= report("  hard course (report only)", True,
                     "%d gates crossed, mean reward %+.3f/step, %d episodes ended %s"
                     % (r["best"], r["per_step"], r["n_done"], r["why"]))
    return ok_all


# =========================================================================
# 5 -- throughput
# =========================================================================
def test_throughput(n=1024, steps=200):
    print("\n5. throughput")
    env = VecSurrogate(n, EnvConfig.curriculum(0.6, 1.0), seed=1)
    env.reset()
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    a = act.policy_to_action(u)
    for _ in range(10):
        env.step(a)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(a)
    dt = time.perf_counter() - t0
    sps = n * steps / dt
    return report("throughput", sps > 0,
                  "n_envs %d: %.0f env-steps/s (%.1f s for %d x %d)"
                  % (n, sps, dt, n, steps))


# =========================================================================
# 6 -- single == vec
# =========================================================================
def test_single_matches_vec():
    print("\n6. SingleSurrogate vs VecSurrogate -- same seed, same actions")
    cfg = EnvConfig.curriculum(0.7, 0.8)
    vec = VecSurrogate(1, cfg, seed=99)
    sgl = SingleSurrogate(cfg, seed=99)
    ov = vec.reset()
    os_ = sgl.reset()
    worst = float(np.max(np.abs(ov[0] - os_.to_vector())))
    rng = np.random.default_rng(4)
    for _ in range(300):
        u = rng.uniform(-1, 1, (1, 3)).astype(np.float32)
        a = act.policy_to_action(u)
        ov, _, _, _ = vec.step(a)
        obs, _ = sgl.step(interface.Action(roll_rate=float(a[0, 0]),
                                           pitch_rate=float(a[0, 1]),
                                           yaw_rate=0.0, thrust=float(a[0, 2])))
        worst = max(worst, float(np.max(np.abs(ov[0] - obs.to_vector()))))
    pos = float(np.max(np.abs(vec.p[0] - sgl.vec.p[0])))
    ok = worst == 0.0 and pos < 1e-12
    return report("single == vec", ok,
                  "300 steps: max |obs difference| = %.1e, |position difference| = %.1e m"
                  % (worst, pos))


# =========================================================================
# 7 -- progress_gate_scale
# =========================================================================
def test_progress_gate_scale():
    print("\n7. progress_gate_scale -- weakens dense progress near gates and nothing else")
    base = EnvConfig.curriculum(0.0, 1.0).replace(
        turn_deg_easy=0.0, elev_deg_easy=0.0, turn_min_frac=0.0,
        start_probs=(1.0, 0.0, 0.0, 0.0, 0.0))
    at1, at0 = _GateAttention(), _GateAttention()
    e1 = VecSurrogate(8, base.replace(attention_factory=lambda: at1), seed=21)
    e0 = VecSurrogate(8, base.replace(progress_gate_scale=0.0,
                                      attention_factory=lambda: at0), seed=21)
    at1.env, at0.env = e1, e0
    e1.reset()
    e0.reset()

    traj = 0.0
    near_diff = off_diff = 0.0
    n_near = n_off = n_skip = 0
    ret1 = ret0 = 0.0
    cross1 = cross0 = 0
    done_same = True
    for _ in range(1500):
        a = scripted_action(e1)                 # states are identical, so e1 speaks for both
        _, r1, d1, i1 = e1.step(a)
        _, r0, d0, i0 = e0.step(a)
        traj = max(traj, float(np.max(np.abs(e1.p - e0.p))))
        done_same &= bool(np.array_equal(d1, d0))
        cross1 = max(cross1, int(np.max(i1["gates_passed"])))
        cross0 = max(cross0, int(np.max(i0["gates_passed"])))
        ret1 += float(np.sum(r1))
        ret0 += float(np.sum(r0))
        if np.any(d1 | d0):
            # after a reset `_near_gate_mask` describes the NEW episode, not the step
            # whose reward was just paid
            n_skip += 1
            continue
        near = e1._near_gate_mask()
        diff = np.abs(r1.astype(np.float64) - r0.astype(np.float64))
        n_near += int(near.sum())
        n_off += int((~near).sum())
        near_diff = max(near_diff, float(np.max(diff[near], initial=0.0)))
        off_diff = max(off_diff, float(np.max(diff[~near], initial=0.0)))

    ok = (traj == 0.0 and done_same and off_diff == 0.0 and near_diff > 0.0
          and cross1 == cross0 and n_near > 0 and ret0 < ret1)
    return report("progress_gate_scale", ok,
                  "scale 1.0 vs 0.0 on an identical trajectory (max drift %.1e m):"
                  " reward differs by up to %.4f on %d near-gate steps and by %.1e on"
                  " %d others; both crossed %d gates; return %+.1f -> %+.1f"
                  % (traj, near_diff, n_near, off_diff, n_off, cross1, ret1, ret0))


# =========================================================================
# 8 -- terminal_obs
# =========================================================================
# Race channel offsets in the 73-D vector: 3 gates x 11 = 33, ribbon 15, own 14.
I_ACTIVE, I_TGATE, I_TCOLL = 62, 63, 64


def test_terminal_obs():
    print("\n8. terminal_obs -- the ending episode's last observation survives auto-reset")
    cfg = EnvConfig.curriculum(0.6, 1.0).replace(start_probs=(1.0, 0.0, 0.0, 0.0, 0.0))
    env = VecSurrogate(64, cfg, seed=31)
    env.reset()
    rng = np.random.default_rng(12)

    n_done = n_coll = 0
    off_worst = 0.0        # where not done, terminal_obs must BE the returned observation
    same_at_done = 0       # where done, it must differ from the fresh episode's
    not_fresh = 0          # the returned obs at a done index must be a fresh episode
    stale_coll = 0         # terminal_obs must carry the ENDING episode's collision clock
    shape_ok = True
    for _ in range(900):
        u = rng.uniform(-1, 1, (64, 3)).astype(np.float32)
        obs, _, done, info = env.step(act.policy_to_action(u))
        t = info["terminal_obs"]
        shape_ok &= (t.shape == (64, interface.OBS_DIM) and t.dtype == np.float32)
        if np.any(~done):
            off_worst = max(off_worst, float(np.max(np.abs(t[~done] - obs[~done]))))
        if not np.any(done):
            continue
        n_done += int(done.sum())
        same_at_done += int(np.count_nonzero(np.all(t[done] == obs[done], axis=1)))
        # A normal-start episode begins at gate 0, with no gate yet timed and no
        # collision on record, so the post-reset observation is pinned exactly.
        fresh = obs[done]
        not_fresh += int(np.count_nonzero((fresh[:, I_ACTIVE] != 0.0)
                                          | (fresh[:, I_TGATE] != 0.0)
                                          | (fresh[:, I_TCOLL] != 5.0)))
        # ...whereas an episode that ended in a collision has t_since_collision_s == 0,
        # which is only visible if terminal_obs really was captured before the reset.
        coll = info["collision"][done]
        n_coll += int(coll.sum())
        stale_coll += int(np.count_nonzero(t[done][coll, I_TCOLL] != 0.0))

    ok = (shape_ok and off_worst == 0.0 and same_at_done == 0 and not_fresh == 0
          and stale_coll == 0 and n_done > 50 and n_coll > 20)
    return report("terminal_obs", ok,
                  "%d terminations (%d collisions), shapes %s: max |terminal - obs| where"
                  " not done %.1e, done indices equal to the fresh obs %d, post-reset obs"
                  " not a fresh episode %d, terminal t_since_collision != 0 %d"
                  % (n_done, n_coll, "ok" if shape_ok else "WRONG", off_worst,
                     same_at_done, not_fresh, stale_coll))


# =========================================================================
# 9 -- what the detections actually look like (report only)
# =========================================================================
def detection_stats(difficulty, steps=1200, n=32, seed=3):
    """No assertion: there is nothing measured to assert against until P3 lands.

    Printed because "the detector silently emits nothing" and "the detector is clean" are
    both easy to ship and both fatal, and neither shows up in a PASS/FAIL above.
    """
    cfg = EnvConfig.curriculum(difficulty, 1.0).replace(start_probs=(1.0, 0, 0, 0, 0))
    at = _GateAttention()
    env = VecSurrogate(n, cfg.replace(attention_factory=lambda: at), seed=seed)
    at.env = env
    env.reset()
    tot = v0 = nv = pv = fp = unchanged = rib = 0
    stale = 0.0
    err, rngs = [], []
    prev = None
    for _ in range(steps):
        env.step(scripted_action(env))
        f = env._fields
        valid = f["gate_valid"][:, 0]
        tot += n
        v0 += int(valid.sum())
        nv += int(f["gate_nvalid"][:, 0].sum())
        pv += int(f["gate_pvalid"][:, 0].sum())
        fp += int((env.det.is_fp & f["gate_valid"]).sum())
        rib += int(f["rb_valid"].sum())
        stale += float(f["gate_stale"][valid, 0].sum())
        g = env.g_pos[env._rows, np.clip(env.active, 0, env.n_slots_course - 1)]
        true_b = np.einsum('nji,nj->ni', vmath.quat_to_rot(env.q), g - env.p)
        sel = valid & ~env.det.is_fp[:, 0]
        if np.any(sel):
            err.extend(np.linalg.norm(f["gate_pos"][sel, 0] - true_b[sel], axis=1).tolist())
            rngs.extend(np.linalg.norm(true_b[sel], axis=1).tolist())
        cur = f["gate_pos"][:, 0].copy()
        if prev is not None:
            unchanged += int(np.sum(np.all(cur == prev, axis=1) & valid))
        prev = cur
    e, rr = np.array(err), np.array(rngs)
    print("  difficulty %.1f" % difficulty)
    print("    current gate valid %5.1f%%   normal_valid %5.1f%%   pose_valid %5.1f%%"
          % (100 * v0 / tot, 100 * nv / tot, 100 * pv / tot))
    print("    staleness mean %.3f s   detection unchanged since last step %5.1f%%"
          % (stale / max(v0, 1), 100 * unchanged / max(v0, 1)))
    print("    false tracks %5.2f%% of valid slots   ribbon valid %5.1f%%"
          % (100 * fp / max(3 * tot, 1), 100 * rib / tot))
    print("    |reported - true| pos_body: median %.2f m, p90 %.2f m at median range"
          " %.1f m" % (np.median(e), np.percentile(e, 90), np.median(rr)))


def main():
    print("P2 surrogate self-check")
    print("=" * 72)
    test_vector_equivalence()
    test_physics()
    test_camera()
    test_flight()
    test_throughput()
    test_single_matches_vec()
    test_progress_gate_scale()
    test_terminal_obs()
    print("\n9. detection statistics (report only -- nothing measured to assert against)")
    detection_stats(0.0)
    detection_stats(1.0)
    print("=" * 72)
    ok = all(RESULTS)
    print("%d/%d checks passed -- %s" % (sum(RESULTS), len(RESULTS),
                                         "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
