"""P2 -- the vectorised surrogate. Thousands of instances, no pixels, 73-D observations.

    obs = env.reset()                        # [n, 73] float32
    obs, rew, done, info = env.step(a)       # a is [n, 3] in PHYSICAL units

`info` carries `gates_passed`, `collision`, `corridor_exit`, `timeout`, `finished`,
`episode_return`, `episode_length` and `terminal_obs`. The last four are meaningful only
where `done` is True. `terminal_obs` is `[n, 73]` float32: the ending episode's final
observation, captured before auto-reset overwrote the state, so a value function can
bootstrap a timeout-truncated episode from V(s_T) instead of from zero.

`a` is (roll_rate rad/s, pitch_rate rad/s, thrust 0..1) in **canonical body NED**.
`actions.policy_to_action` maps a network's tanh outputs into it. Yaw is NOT in the action
vector: it belongs to the `AUTO_ATTENTION` servo, which this env drives from the attention
module over the synthetic detections (`attention.py`).

## The sign rule, because this is where a sign error is silent

`CONVENTIONS.md`: the simulator's -1 rate mirror covers commanded rates and the gyro, it
is applied **in the link layer only**, and no file above the link layer may contain a sign
flip. This file is above the link layer. So:

* actions are canonical NED, the integrated body rate is `rate_gain * commanded` with
  **no** mirror, and the emitted `SelfObs.gyro` is the canonical rate;
* `plant.Plant.rates_ned()` is deliberately NOT called -- it exists to convert
  `cmd.csv`-convention rates (what the recordings hold) into NED, and the surrogate's
  rates are already NED. `selfcheck.py` pins this: the surrogate driven with `+cmd` must
  reproduce `plant.Sim` driven with `-cmd`, exactly. Applying the mirror zero times or
  twice is the failure `README.md` calls "great on the surrogate, bad in the sim".

## The world frame exists only in here

Position, heading and gate geometry are used for reward and collision tests and are never
encoded into the observation. Reward is computed from TRUE state on purpose: a reward that
followed the noisy detections would silently change target on every dropout
(`TRAINING_ARCHITECTURE.md` T4).

## Timing

Decisions at a per-episode randomised 45-65 Hz; three internal substeps per decision, so
the integrator runs at 135-195 Hz -- at or above the architecture's 120 Hz floor, and an
exact integer subdivision of the decision period, which keeps the command-delay ring
buffers honest. Detections tick on their own ~30 fps camera clock.
"""

import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import interface  # noqa: E402
import plant as plant_mod  # noqa: E402

import actions as act  # noqa: E402
import attention as attn_mod  # noqa: E402
import camera  # noqa: E402
import course  # noqa: E402
import detect  # noqa: E402
import noise as noise_mod  # noqa: E402
import vmath  # noqa: E402

G = plant_mod.G
N_SLOTS = interface.N_GATES
N_RIBBON = interface.N_RIBBON
OBS_DIM = interface.OBS_DIM

# Look-ahead distances, in metres along the course path, at which the cyan corridor is
# sampled. Roughly "the next 5 gates" (NOTES.md) at ~28 m spacing, front-loaded.
RIBBON_LOOKAHEAD = np.array([4.0, 8.0, 14.0, 22.0, 32.0, 45.0])

# Drone bounding sphere: 280 x 280 x 160 mm (spec 3.7) -> half-diagonal 0.214 m.
DRONE_SPHERE_M = 0.214

# `interface.RaceObs.collision_episodes` counts contact EPISODES, not contact samples:
# two contacts closer together than this are the same episode. The number is the
# interface's own (a parked drone emits ~250 contact messages a second).
COLLISION_GAP_S = 0.5

# The drag coefficient the ON-BOARD speed estimator believes, i.e. the published fit.
# Deliberately NOT the per-episode randomised value: the aircraft only ever knows the
# number in plant.json, and the mismatch is exactly what domain randomisation must expose.
K_DRAG_PUBLISHED = 0.0492

_POSE_HIST = 24         # decision steps of pose history, for detection latency
_CMD_RING = 8           # substeps of command history, for the plant's transport delays


@dataclass
class EnvConfig:
    """Everything the surrogate randomises or can be told to stop randomising.

    `difficulty` and `speed_cap` are the two curriculum axes from
    `TRAINING_ARCHITECTURE.md` T4: difficulty anneals gentle wide courses + low noise into
    tight winding turns + the full fallback noise model; speed_cap caps commanded
    aggressiveness so exploration survives the collision terminations early on.
    """

    # -- the curriculum ------------------------------------------------------
    difficulty: float = 0.5           # 0 = gentle wide + low noise, 1 = tight + full noise
    speed_cap: float = 1.0            # multiplier on actions.RATE_CAP_RPS, in (0, 1]

    # -- required by the contract -------------------------------------------
    decision_hz_range: tuple = (45.0, 65.0)
    n_gates_range: tuple = (18, 22)
    enable_time_penalty: bool = False

    # -- course --------------------------------------------------------------
    # ~28 m per gate: 2.76 s/station at a racing 8-12 m/s, and VQ1's 167 m / 6 gates.
    seg_len_easy: tuple = (26.0, 34.0)
    seg_len_hard: tuple = (18.0, 26.0)
    turn_deg_easy: float = 20.0
    turn_deg_hard: float = 80.0
    turn_min_frac: float = 0.25
    turn_flip_p: float = 0.35
    elev_deg_easy: float = 8.0
    elev_deg_hard: float = 22.0       # the ~20 deg descent that hides gates below -9.4 deg
    alt_revert: float = 0.9
    ceiling_range_m: tuple = (9.0, 16.0)
    floor_clear_m: float = 1.8
    ceil_clear_m: float = 1.0
    corridor_easy_m: float = 16.0
    corridor_hard_m: float = 10.0

    # -- dynamics ------------------------------------------------------------
    substeps: int = 3
    plant_path: str = None            # defaults to ../plant.json
    randomize_plant: bool = True
    drag_jitter: float = 0.15         # +-15%, deliberately wider than the fit residuals
    thrust_jitter: float = 0.15
    rate_gain_jitter: float = 0.10
    delay_jitter: tuple = (0.7, 1.4)

    # -- collision -----------------------------------------------------------
    margin_range_m: tuple = (0.10, 0.40)   # architecture T4: the main transfer trick
    gate_outer_m: float = course.GATE_OUTER_M

    # -- reward --------------------------------------------------------------
    approach_d_m: float = 2.0         # d in "pos + d*normal". TBD in the architecture.
    k_progress: float = 1.0
    progress_clip: float = 3.0
    k_cross: float = 5.0
    k_collision: float = 8.0
    k_corridor: float = 8.0
    k_finish: float = 10.0
    k_jerk: float = 0.02
    k_nogate: float = 0.01
    k_time: float = 0.01
    # The curriculum's stall response, mandated by TRAINING_ARCHITECTURE.md T4: "If
    # completion stalls, weaken the progress term near gates; do not weaken the collision
    # penalty." A policy that stalls is usually one that dives at the gate because dense
    # progress pays right up to the plane, and clips the frame. Scaling progress down
    # inside the last few metres leaves the crossing bonus as the dominant local signal,
    # which pays for centring rather than for closing distance.
    #
    # "Near a gate" is `|p - approach_point| < 2 * approach_d_m` (4 m by default). The
    # approach point sits `approach_d_m` before the gate, so the zone runs from ~6 m out
    # to ~2 m past the plane -- about 0.4 s of flight at cruise, i.e. the committed part
    # of the approach. It is derived from `approach_d_m` rather than being its own knob so
    # that moving the guidance target moves the zone with it.
    #
    # Scales progress SYMMETRICALLY: weakening only the positive half would pay full
    # credit for closing and a discount for backing off, which is a shaping loophole.
    # 1.0 = no effect. Touches neither the crossing bonus nor the collision penalty.
    progress_gate_scale: float = 1.0

    # -- episode -------------------------------------------------------------
    max_time_s: float = 120.0
    gate_timeout_s: float = 12.0
    # normal, mid-course, hover, corridor-offset, no-gate-visible. The last three cover
    # D6's recovery handback, which lands at hover with an arbitrary corridor offset and
    # possibly nothing in frustum.
    start_probs: tuple = (0.50, 0.25, 0.12, 0.08, 0.05)
    cruise_speed_mps: float = 10.0
    start_speed_frac: tuple = (0.35, 1.05)

    # -- perception ----------------------------------------------------------
    noise: noise_mod.NoiseParams = field(default_factory=noise_mod.NoiseParams)
    r_commit_m: float = attn_mod.R_COMMIT_M     # TBD, see attention.py
    attention_factory: object = None            # () -> object with reset/step_batch

    @classmethod
    def curriculum(cls, difficulty, speed_cap, **overrides):
        """The two-axis curriculum. Everything else stays at its default unless overridden."""
        d = float(np.clip(difficulty, 0.0, 1.0))
        s = float(np.clip(speed_cap, 1e-3, 1.0))
        return cls(difficulty=d, speed_cap=s, **overrides)

    def replace(self, **kw):
        return replace(self, **kw)


class VecSurrogate:
    """`n_envs` independent races, stepped together in NumPy."""

    obs_dim = OBS_DIM
    act_dim = 3

    def __init__(self, n_envs, config=None, seed=0):
        self.n = int(n_envs)
        self.cfg = config if config is not None else EnvConfig()
        self.rng = np.random.default_rng(seed)

        path = self.cfg.plant_path or os.path.join(os.path.dirname(_HERE), "plant.json")
        self.plant = plant_mod.Plant.load(path)

        n, S = self.n, int(self.cfg.n_gates_range[1]) + course.RUNOUT
        self.n_slots_course = S

        self.p = np.zeros((n, 3))
        self.v = np.zeros((n, 3))
        self.q = np.zeros((n, 4))
        self.q[:, 0] = 1.0

        self.g_pos = np.zeros((n, S, 3))
        self.g_nrm = np.zeros((n, S, 3))
        self.g_nrm[:, :, 0] = 1.0
        self.n_gates = np.full(n, self.cfg.n_gates_range[0], dtype=np.int64)
        self.start_p = np.zeros((n, 3))
        self.z_ceil = np.full(n, -12.0)
        self.corridor_r = np.full(n, 14.0)

        self.active = np.zeros(n, dtype=np.int64)
        self.t_s = np.zeros(n)
        self.t_gate = np.zeros(n)
        self.t_coll = np.full(n, 1e3)
        self.coll_n = np.zeros(n, dtype=np.int64)
        self.ep_ret = np.zeros(n)
        self.ep_len = np.zeros(n, dtype=np.int64)

        self.dt_dec = np.full(n, 1.0 / 55.0)
        self.dt_sub = self.dt_dec / self.cfg.substeps

        self.kx = np.full(n, self.plant.kx)
        self.ky = np.full(n, self.plant.ky)
        self.kz = np.full(n, self.plant.kz)
        # Thrust is randomised as a GAIN on `plant.thrust()` rather than by perturbing the
        # curve's coefficients, so the surrogate depends only on the plant's public API.
        # The curve has already changed shape once (affine -> quadratic -> measured knot
        # table) and will change again if card 2 is reflown.
        self.thrust_gain = np.ones(n)
        knots = np.asarray(self.plant.thrust_knots, dtype=float)
        self._knot_x, self._knot_y = knots[:, 0], knots[:, 1]
        self.rgain = np.tile(np.asarray(self.plant.rate_gain, dtype=float), (n, 1))
        self.rate_k = np.ones(n, dtype=np.int64)
        self.thr_k = np.full(n, 2, dtype=np.int64)

        self._cbuf = np.zeros((_CMD_RING, n, 3))
        self._tbuf = np.zeros((_CMD_RING, n))
        self._chead = 0
        self._phist = np.zeros((_POSE_HIST, n, 3))
        self._qhist = np.zeros((_POSE_HIST, n, 4))
        self._qhist[:, :, 0] = 1.0
        self._thist = np.zeros((_POSE_HIST, n))
        self._phead = 0
        self.lat_k = np.zeros(n, dtype=np.int64)

        self.cam_acc = np.zeros(n)
        self.cam_period = np.full(n, 1.0 / 30.05)
        self.t_cam = np.zeros(n)

        self.sphere_r = np.full(n, DRONE_SPHERE_M + 0.2)
        self.gyro_true = np.zeros((n, 3))
        self.accel_body = np.zeros((n, 3))
        self.gyro_b = np.zeros((n, 3))
        self.accel_b = np.zeros((n, 3))

        self.u_prev = np.zeros((n, 3))
        self.rem_prev = np.zeros(n)

        self.nz = {k: np.zeros(n) for k in noise_mod.SAMPLED}
        self.det = detect.Detector(n)
        self.rb_on = np.zeros(n, dtype=bool)

        mk = self.cfg.attention_factory or (
            lambda: attn_mod.AttentionPolicy(r_commit_m=self.cfg.r_commit_m))
        self.attn = mk()
        self.attn.reset(n)
        self._yaw_cmd = np.zeros(n)
        self._fields = None
        self._rows = np.arange(n)

        self.reset()

    # =====================================================================
    # reset
    # =====================================================================
    def reset(self):
        self._reset_envs(np.ones(self.n, dtype=bool))
        self._fields = self._observation_fields()
        return self._encode(self._fields)

    def _reset_envs(self, mask):
        m = int(np.count_nonzero(mask))
        if m == 0:
            return
        cfg, rng = self.cfg, self.rng

        c = course.generate(rng, m, cfg)
        self.g_pos[mask] = c["pos"]
        self.g_nrm[mask] = c["nrm"]
        self.n_gates[mask] = c["n_gates"]
        self.start_p[mask] = c["start_p"]
        self.z_ceil[mask] = c["z_ceil"]
        self.corridor_r[mask] = c["corridor_r"]

        nz = noise_mod.sample(cfg.noise, rng, m, cfg.difficulty)
        for k, val in nz.items():
            self.nz[k][mask] = val

        # -- plant, per episode, deliberately wider than the fit residuals -----
        if cfg.randomize_plant:
            j = lambda w: rng.uniform(1 - w, 1 + w, size=m)      # noqa: E731
            self.kx[mask] = self.plant.kx * j(cfg.drag_jitter)
            self.ky[mask] = self.plant.ky * j(cfg.drag_jitter)
            self.kz[mask] = self.plant.kz * j(cfg.drag_jitter)
            self.thrust_gain[mask] = j(cfg.thrust_jitter)
            self.rgain[mask] = (np.asarray(self.plant.rate_gain)[None, :]
                                * rng.uniform(1 - cfg.rate_gain_jitter,
                                              1 + cfg.rate_gain_jitter, size=(m, 3)))
            rd = self.plant.rate_delay * rng.uniform(*cfg.delay_jitter, size=m)
            td = self.plant.thrust_delay * rng.uniform(*cfg.delay_jitter, size=m)
            self.sphere_r[mask] = DRONE_SPHERE_M + rng.uniform(
                cfg.margin_range_m[0],
                cfg.margin_range_m[0] + (cfg.margin_range_m[1] - cfg.margin_range_m[0])
                * (0.4 + 0.6 * float(np.clip(cfg.difficulty, 0, 1))), size=m)
        else:
            self.kx[mask] = self.plant.kx
            self.ky[mask] = self.plant.ky
            self.kz[mask] = self.plant.kz
            self.thrust_gain[mask] = 1.0
            self.rgain[mask] = np.asarray(self.plant.rate_gain)[None, :]
            rd = np.full(m, self.plant.rate_delay)
            td = np.full(m, self.plant.thrust_delay)
            self.sphere_r[mask] = DRONE_SPHERE_M + cfg.margin_range_m[0]

        hz = rng.uniform(cfg.decision_hz_range[0], cfg.decision_hz_range[1], size=m)
        dt = 1.0 / hz
        self.dt_dec[mask] = dt
        sub = dt / cfg.substeps
        self.dt_sub[mask] = sub
        self.rate_k[mask] = np.clip(np.round(rd / sub), 1, _CMD_RING - 1).astype(np.int64)
        self.thr_k[mask] = np.clip(np.round(td / sub), 1, _CMD_RING - 1).astype(np.int64)

        self.cam_period[mask] = 1.0 / self.nz["cam_fps"][mask]
        self.cam_acc[mask] = 0.0
        self.t_cam[mask] = 0.0
        self.lat_k[mask] = np.clip(np.round(self.nz["latency_s"][mask] / dt),
                                   0, _POSE_HIST - 1).astype(np.int64)

        self.gyro_b[mask] = rng.normal(0.0, 1.0, (m, 3)) * self.nz["gyro_bias"][mask][:, None]
        self.accel_b[mask] = rng.normal(0.0, 1.0, (m, 3)) * self.nz["accel_bias"][mask][:, None]

        # -- the start state ---------------------------------------------------
        mode = rng.choice(5, size=m, p=np.asarray(cfg.start_probs, dtype=float)
                          / float(np.sum(cfg.start_probs)))
        ng = self.n_gates[mask]
        a0 = np.where(mode == 1, rng.integers(0, np.maximum(ng - 3, 1)), 0)

        gp = self.g_pos[mask][np.arange(m), a0]
        gn = self.g_nrm[mask][np.arange(m), a0]
        back = rng.uniform(0.45, 0.95, size=m) * c["seg_len"]
        p0 = gp - gn * back[:, None]

        # lateral/vertical offset from the corridor centreline
        up = np.zeros((m, 3))
        up[:, 2] = -1.0
        side = vmath.unit(np.cross(gn, up))
        vert = vmath.unit(np.cross(side, gn))
        off_s = rng.normal(0.0, 1.0, m) * np.where(mode == 3, 0.45, 0.10) * self.corridor_r[mask]
        off_v = rng.normal(0.0, 1.0, m) * np.where(mode == 3, 0.30, 0.07) * self.corridor_r[mask]
        p0 = p0 + side * off_s[:, None] + vert * off_v[:, None]

        alt_lo = cfg.floor_clear_m + 0.6
        p0[:, 2] = np.minimum(p0[:, 2], -alt_lo)
        p0[:, 2] = np.maximum(p0[:, 2], self.z_ceil[mask] + 1.2)

        speed = np.where(mode == 2, rng.uniform(0.0, 1.2, m),
                         rng.uniform(*cfg.start_speed_frac, size=m) * cfg.cruise_speed_mps)
        v0 = gn * speed[:, None]

        yaw_path = np.arctan2(gn[:, 1], gn[:, 0])
        yaw = yaw_path + rng.normal(0.0, 0.12, m)
        yaw = np.where(mode == 2, rng.uniform(-np.pi, np.pi, m), yaw)
        away = rng.uniform(1.75, np.pi, m) * rng.choice([-1.0, 1.0], size=m)
        yaw = np.where(mode == 4, yaw_path + away, yaw)
        roll = rng.normal(0.0, 0.06, m) * np.where(mode >= 2, 0.4, 1.0)
        pitch = rng.normal(0.0, 0.06, m) * np.where(mode >= 2, 0.4, 1.0)
        q0 = vmath.quat_from_euler(roll, pitch, yaw)

        self.p[mask] = p0
        self.v[mask] = v0
        self.q[mask] = q0
        self.active[mask] = a0
        self.t_s[mask] = 0.0
        self.t_gate[mask] = 0.0
        # A hover / no-gate start is the D6 handback: the supervisor gives control back
        # shortly after a collision episode, so t_since_collision_s must look like it.
        # `collision_episodes` is seeded to match: a start that claims a collision just
        # happened must also have COUNTED it, or the two halves of the same event
        # disagree and the supervisor's two trigger branches see different worlds.
        self.t_coll[mask] = np.where(mode >= 2, rng.uniform(0.2, 2.5, m), 1e3)
        self.coll_n[mask] = np.where(mode >= 2, 1, 0)
        self.ep_ret[mask] = 0.0
        self.ep_len[mask] = 0
        self.gyro_true[mask] = 0.0
        self.accel_body[mask] = np.array([0.0, 0.0, -G])
        self.u_prev[mask] = 0.0
        self.rb_on[mask] = rng.random(m) < self.nz["ribbon_duty"][mask]

        self._cbuf[:, mask] = 0.0
        self._tbuf[:, mask] = 0.0
        self._phist[:, mask] = p0
        self._qhist[:, mask] = q0
        self._thist[:, mask] = 0.0

        self.det.clear(mask)
        self.attn.reset(self.n, mask=mask)
        # One camera frame at t=0 so the first observation is not blank. Zero latency for
        # this frame only -- there is no history to be late about.
        self.det.tick(self.rng, self.nz, self.p, self.q, self.g_pos, self.g_nrm,
                      self.active, self.n_gates, self.t_s, mask)
        self.rem_prev[mask] = self._remaining()[mask]

    # =====================================================================
    # step
    # =====================================================================
    def step(self, actions):
        cfg = self.cfg
        a = act.clip_physical(np.asarray(actions, dtype=np.float32).reshape(self.n, 3),
                              cfg.speed_cap)
        rates = np.empty((self.n, 3))
        rates[:, 0] = a[:, 0]
        rates[:, 1] = a[:, 1]
        rates[:, 2] = self._yaw_cmd          # AUTO_ATTENTION: yaw is not the policy's
        thr = a[:, 2].astype(np.float64)

        collided = np.zeros(self.n, dtype=bool)
        advanced = np.zeros(self.n, dtype=bool)

        for _ in range(cfg.substeps):
            p0 = self.p.copy()
            self._substep(rates, thr)
            hit, passed = self._gate_geometry(p0, self.p)
            collided |= hit
            newly = passed & ~advanced & ~collided
            self.active = self.active + newly.astype(np.int64)
            self.det.shift(newly)
            advanced |= newly

        # Floor and ceiling are contacts too, and must be known BEFORE the collision clock
        # is updated. Testing them further down -- after `t_coll` had already been
        # advanced -- left `info['collision']` saying yes while the observation's
        # `t_since_collision_s` said no, for every floor strike. Two of this file's own
        # outputs disagreeing about the same event is the kind of silent inconsistency the
        # D6 recovery supervisor keys off.
        floor_hit = self.p[:, 2] > -self.sphere_r
        ceil_hit = self.p[:, 2] < self.z_ceil + self.sphere_r
        collided |= floor_hit | ceil_hit

        dt = self.dt_dec
        self.t_s = self.t_s + dt
        self.t_gate = np.where(advanced, 0.0, self.t_gate + dt)
        # `interface.RaceObs`: COLLISION is a contact SAMPLE, not a crash, and the counter
        # reports EPISODES separated by a gap > COLLISION_GAP_S -- never message counts.
        # Every collision terminates in this surrogate, so in practice the gate is always
        # open and the counter steps 0 -> 1 on the terminal step; the rule is written out
        # anyway so the semantics stay right if contact ever stops being terminal.
        self.coll_n = self.coll_n + (collided & (self.t_coll + dt > COLLISION_GAP_S))
        self.t_coll = np.where(collided, 0.0, self.t_coll + dt)
        self.ep_len = self.ep_len + 1

        # -- pose history and the camera clock --------------------------------
        self._phead = (self._phead + 1) % _POSE_HIST
        self._phist[self._phead] = self.p
        self._qhist[self._phead] = self.q
        self._thist[self._phead] = self.t_s

        self.cam_acc = self.cam_acc + dt
        tick = self.cam_acc >= self.cam_period
        self.cam_acc = np.where(tick, self.cam_acc - self.cam_period, self.cam_acc)
        if np.any(tick):
            i = (self._phead - self.lat_k) % _POSE_HIST
            self.det.tick(self.rng, self.nz, self._phist[i, self._rows],
                          self._qhist[i, self._rows], self.g_pos, self.g_nrm,
                          self.active, self.n_gates, self._thist[i, self._rows], tick)
        self.t_cam = np.where(tick, self.t_s, self.t_cam)

        # -- termination -------------------------------------------------------
        finished = self.active >= self.n_gates
        corridor = self._corridor_exit() & ~finished
        timeout = (self.t_s > cfg.max_time_s) | (self.t_gate > cfg.gate_timeout_s)
        done = collided | corridor | timeout | finished

        # -- reward, from TRUE state ------------------------------------------
        rem = self._remaining()
        # On the step that crosses a gate, `rem_prev` measures the OLD target and `rem`
        # the NEW one, ~28 m further off. Differencing them would charge a large negative
        # progress for the one event the reward most wants to encourage, cancelling most
        # of the crossing bonus. The target moved; the aircraft did not regress.
        prog = np.where(advanced | finished, 0.0,
                        np.clip(self.rem_prev - rem, -cfg.progress_clip, cfg.progress_clip))
        if cfg.progress_gate_scale != 1.0:
            prog = prog * np.where(self._near_gate_mask(), cfg.progress_gate_scale, 1.0)
        u = np.stack([a[:, 0] / act.RATE_CAP_RPS, a[:, 1] / act.RATE_CAP_RPS,
                      act.thrust_to_unit(a[:, 2])], axis=1)
        rew = cfg.k_progress * prog
        rew = rew + cfg.k_cross * advanced
        rew = rew + cfg.k_finish * finished
        rew = rew - cfg.k_collision * collided
        rew = rew - cfg.k_corridor * corridor
        rew = rew - cfg.k_jerk * np.sum((u - self.u_prev) ** 2, axis=1)
        rew = rew - cfg.k_nogate * (~self._gate_in_frustum())
        if cfg.enable_time_penalty:
            rew = rew - cfg.k_time
        self.u_prev = u
        self.rem_prev = rem
        self.ep_ret = self.ep_ret + rew

        info = dict(gates_passed=self.active.copy(),
                    collision=collided.copy(),
                    # Post-increment, pre-reset -- the count the ENDING episode finished
                    # with. Scoring must read it here: the observation handed back by a
                    # terminating step belongs to the next episode, and the last
                    # observation handed out before it predates the contact.
                    collision_episodes=self.coll_n.copy(),
                    corridor_exit=corridor.copy(),
                    timeout=timeout.copy(),
                    finished=finished.copy(),
                    episode_return=self.ep_ret.astype(np.float32),
                    episode_length=self.ep_len.copy())

        # Assemble the post-step observation BEFORE any reset. For an env that is ending
        # this IS the terminal observation, and auto-reset would otherwise destroy it --
        # which forces a PPO harness to bootstrap timeout-truncated episodes at V = 0, a
        # value-estimation bias on episodes that did not actually fail. For every other
        # env it is simply the observation to return, so a step with nothing done costs
        # exactly what it did before.
        fields = self._observation_fields()
        terminal_obs = self._encode(fields)
        info["terminal_obs"] = terminal_obs

        if np.any(done):
            yaw_pre = self._yaw_cmd
            self._reset_envs(done)
            # Re-assemble, and splice in ONLY the rows that were reset. `dt_advance` is
            # zeroed elsewhere so the second pass does not advance the ribbon burst state
            # or the attention sweep of envs that are still flying: an env's observation
            # must never depend on whether some OTHER env happened to end this step.
            fresh = self._observation_fields(dt_advance=np.where(done, self.dt_dec, 0.0))
            fields = {k: np.where(done.reshape((-1,) + (1,) * (v.ndim - 1)), fresh[k], v)
                      for k, v in fields.items()}
            self._yaw_cmd = np.where(done, self._yaw_cmd, yaw_pre)
            obs = self._encode(fields)
        else:
            # Same array as `info["terminal_obs"]`, deliberately: nothing was reset, so
            # they hold identical values and `terminal_obs` is meaningless at every index
            # anyway. On steps that DO reset, the two are distinct arrays.
            obs = terminal_obs
        self._fields = fields
        return obs, rew.astype(np.float32), done, info

    # -- physics ------------------------------------------------------------
    def _substep(self, rates, thr):
        """One integration step. Identical algebra to `plant.Sim.step`, minus the mirror."""
        self._chead = (self._chead + 1) % _CMD_RING
        self._cbuf[self._chead] = rates
        self._tbuf[self._chead] = thr
        ri = (self._chead - self.rate_k) % _CMD_RING
        ti = (self._chead - self.thr_k) % _CMD_RING
        r_use = self._cbuf[ri, self._rows]
        t_use = self._tbuf[ti, self._rows]

        r_wb = vmath.quat_to_rot(self.q)
        v_body = vmath.rot_apply_t(r_wb, self.v)
        k = np.stack([self.kx, self.ky, self.kz], axis=1)
        f_body = -k * v_body * np.abs(v_body)
        f_body[:, 2] -= self.thrust_gain * self.plant.thrust(t_use)
        a_world = vmath.rot_apply(r_wb, f_body)
        a_world[:, 2] += G

        dt = self.dt_sub[:, None]
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        # NO SIGN FLIP. See the module docstring.
        omega = self.rgain * r_use
        self.q = vmath.quat_integrate(self.q, omega, self.dt_sub)
        self.gyro_true = omega
        self.accel_body = f_body

    # -- geometry -----------------------------------------------------------
    def _slot_gates(self):
        idx = self.active[:, None] + np.arange(N_SLOTS)[None, :]
        idx_c = np.clip(idx, 0, self.n_slots_course - 1)
        gp = np.take_along_axis(self.g_pos, idx_c[:, :, None], axis=1)
        gn = np.take_along_axis(self.g_nrm, idx_c[:, :, None], axis=1)
        return idx, gp, gn, idx < self.n_gates[:, None]

    def _gate_geometry(self, p0, p1):
        """Plane crossings over the segment p0->p1. Returns (collision, passed_active)."""
        _, gp, gn, exists = self._slot_gates()
        d0 = np.einsum('nsi,nsi->ns', p0[:, None, :] - gp, gn)
        d1 = np.einsum('nsi,nsi->ns', p1[:, None, :] - gp, gn)
        cross = exists & (d0 < 0.0) & (d1 >= 0.0)
        span = d1 - d0
        t = np.clip(-d0 / np.where(np.abs(span) < 1e-12, 1.0, span), 0.0, 1.0)
        x = p0[:, None, :] + t[:, :, None] * (p1 - p0)[:, None, :]
        rel = x - gp
        lat = np.linalg.norm(rel - np.einsum('nsi,nsi->ns', rel, gn)[:, :, None] * gn, axis=2)

        r = self.sphere_r[:, None]
        inner = 0.5 * course.GATE_INNER_M
        outer = 0.5 * self.cfg.gate_outer_m
        # The sphere clips the frame: through the plane, outside the clear aperture, and
        # not so wide that it missed the 2700 mm structure entirely.
        hit = cross & (lat + r > inner) & (lat - r < outer)
        passed = cross[:, 0] & (lat[:, 0] + r[:, 0] <= inner)
        return np.any(hit, axis=1), passed

    def _corridor_exit(self):
        a = np.clip(self.active, 0, self.n_slots_course - 1)
        ga = self.g_pos[self._rows, a]
        gb = self.g_pos[self._rows, np.clip(a + 1, 0, self.n_slots_course - 1)]
        prev = np.where((self.active <= 0)[:, None], self.start_p,
                        self.g_pos[self._rows, np.clip(a - 1, 0, self.n_slots_course - 1)])
        d = np.minimum(course.point_segment_distance(self.p, prev, ga),
                       course.point_segment_distance(self.p, ga, gb))
        return d > self.corridor_r

    def _remaining(self):
        """Distance still to fly to the true current gate, via its approach point.

        Blended so it is continuous through the approach point: far out it drives at
        `gate + d*normal`, and inside `d` it drives at the gate centre. Privileged by
        design -- computed from world state, never from `pos_body`/`normal_valid`.
        """
        a = np.clip(self.active, 0, self.n_slots_course - 1)
        gp = self.g_pos[self._rows, a]
        gn = self.g_nrm[self._rows, a]
        dd = self.cfg.approach_d_m
        ap = gp - gn * dd
        s = np.einsum('ni,ni->n', self.p - gp, gn)
        w = np.clip(-s / max(dd, 1e-6), 0.0, 1.0)
        rem = (w * (np.linalg.norm(self.p - ap, axis=1) + dd)
               + (1.0 - w) * np.linalg.norm(self.p - gp, axis=1))
        return np.where(self.active >= self.n_gates, 0.0, rem)

    def _near_gate_mask(self):
        """Where `progress_gate_scale` applies: inside `2*approach_d_m` of the approach point.

        Same privileged world state the rest of the reward uses. See `EnvConfig`.
        """
        a = np.clip(self.active, 0, self.n_slots_course - 1)
        ap = (self.g_pos[self._rows, a]
              - self.g_nrm[self._rows, a] * self.cfg.approach_d_m)
        return (np.linalg.norm(self.p - ap, axis=1) < 2.0 * self.cfg.approach_d_m) \
            & (self.active < self.n_gates)

    def _gate_in_frustum(self):
        """TRUE frustum test on the next three gates -- the reward's visibility term."""
        _, gp, gn, exists = self._slot_gates()
        r_wb = vmath.quat_to_rot(self.q)
        pos_b = np.einsum('nji,nsj->nsi', r_wb, gp - self.p[:, None, :])
        return np.any(exists & camera.in_frustum(pos_b), axis=1)

    # =====================================================================
    # observation
    # =====================================================================
    def _observation_fields(self, dt_advance=None):
        """Everything the 73-D vector and the `Observation` dataclasses are built from.

        Both encoders read this one dict, in float64, so `single.py` and the fast path
        cannot disagree -- which is what `selfcheck.py` test 1 checks.

        `dt_advance` overrides the time step used by the two sub-processes that carry
        state across calls (the ribbon's on/off burst and the attention sweep). It is
        zeroed per-env by `step()` when it has to assemble the observation a second time
        after a reset, so those envs' clocks tick once per step and not twice. The
        observation's own `dt_s` is always the true decision period.
        """
        n = self.n
        nz = self.nz
        rng = self.rng
        adv = self.dt_dec if dt_advance is None else dt_advance
        idx, _, _, exists = self._slot_gates()

        alive, stale = self.det.read(self.t_s, nz["max_coast_s"])
        g_valid = alive
        g_pos = np.where(g_valid[:, :, None], self.det.pos, 0.0)
        g_nvalid = self.det.nvalid & g_valid
        g_nrm = np.where(g_nvalid[:, :, None], self.det.nrm, 0.0)
        g_pvalid = self.det.pvalid & g_valid
        g_conf = np.where(g_valid, self.det.conf, 0.0)
        g_index = np.where(g_valid, idx, -1)

        # -- ribbon ------------------------------------------------------------
        r_wb = vmath.quat_to_rot(self.q)
        pts = self._ribbon_points()
        d_body = np.einsum('nji,nsj->nsi', r_wb, pts - self.p[:, None, :])
        bear = np.arctan2(d_body[:, :, 1], d_body[:, :, 0])
        elev = np.arctan2(-d_body[:, :, 2], np.hypot(d_body[:, :, 0], d_body[:, :, 1]))
        rn = nz["ribbon_noise_rad"][:, None]
        bear = bear + rng.normal(0.0, 1.0, (n, N_RIBBON)) * rn
        elev = elev + rng.normal(0.0, 1.0, (n, N_RIBBON)) * rn
        # Two-state burst process: the corridor is absent from long stretches, so a
        # per-frame coin flip would be far too easy to see through.
        sw = nz["ribbon_switch_hz"] * adv
        duty = nz["ribbon_duty"]
        on_to_off = sw * (1.0 - duty)
        off_to_on = sw * duty
        u = rng.random(n)
        self.rb_on = np.where(self.rb_on, u > on_to_off, u < off_to_on)
        # The ribbon is only visible when it is actually in front of the camera.
        rb_valid = self.rb_on & camera.in_frustum(d_body[:, 0])
        rb_pix = np.where(rb_valid, nz["ribbon_pixfrac"], 0.0)
        rb_stale = np.maximum(self.t_s - self.t_cam, 0.0)

        # -- own state ---------------------------------------------------------
        gyro = self.gyro_true + self.gyro_b + rng.normal(0.0, 1.0, (n, 3)) * nz["gyro_sigma"][:, None]
        accel = self.accel_body + self.accel_b + rng.normal(0.0, 1.0, (n, 3)) * nz["accel_sigma"][:, None]
        roll_t, pitch_t, _ = vmath.euler_from_rot(r_wb)
        a_h = np.hypot(accel[:, 0], accel[:, 1])
        # Gravity roll/pitch degrade under racing acceleration: the gravity vector is only
        # separable from specific force when linear acceleration is small.
        conf = (np.clip(1.0 - a_h / nz["attitude_ref_a"], 0.02, 1.0)
                * np.clip(1.0 - np.linalg.norm(gyro, axis=1) / 6.0, 0.15, 1.0))
        asig = nz["attitude_sigma_rad"] * (1.0 + 6.0 * (1.0 - conf))
        roll = roll_t + rng.normal(0.0, 1.0, n) * asig
        pitch = pitch_t + rng.normal(0.0, 1.0, n) * asig

        vel_bearing = np.arctan2(-accel[:, 1], -accel[:, 0])
        vel_valid = a_h > nz["align_min_accel"]
        speed = np.sqrt(np.maximum(a_h, 0.0) / K_DRAG_PUBLISHED)
        speed_conf = np.clip(1.0 - nz["speed_resid_mps2"] / np.maximum(a_h, 1e-3), 0.0, 1.0)
        speed = np.where(vel_valid, speed, 0.0)
        speed_conf = np.where(vel_valid, speed_conf, 0.0)

        f = dict(
            gate_valid=g_valid, gate_pos=g_pos, gate_nrm=g_nrm, gate_nvalid=g_nvalid,
            gate_pvalid=g_pvalid, gate_conf=g_conf, gate_stale=stale, gate_index=g_index,
            gate_rsig=np.where(g_valid, self.det.rsig, 0.0),
            gate_size=np.where(g_valid, self.det.size, 0.0),
            rb_valid=rb_valid, rb_bear=bear, rb_elev=elev, rb_pix=rb_pix,
            rb_stale=rb_stale,
            gyro=gyro, accel=accel, roll=roll, pitch=pitch, att_conf=conf,
            vel_bearing=vel_bearing, vel_valid=vel_valid,
            speed=speed, speed_conf=speed_conf,
            active=self.active.copy(), n_gates=self.n_gates.copy(),
            t_gate=self.t_gate.copy(), race_t=self.t_s.copy(),
            t_coll=self.t_coll.copy(), coll_n=self.coll_n.copy(),
            armed=np.ones(n, dtype=bool),
            t_s=self.t_s.copy(), dt_s=self.dt_dec.copy(),
        )

        det_in = attn_mod.AttnInput(
            valid=g_valid, pos_body=g_pos, staleness_s=stale, normal_valid=g_nvalid,
            confidence=g_conf, gate_index=g_index, ribbon_valid=rb_valid,
            ribbon_bearings=bear, ribbon_elevs=elev, active_gate_index=self.active)
        out = self.attn.step_batch(det_in, adv)
        self._yaw_cmd = out.yaw_rate
        f["att_kind"] = out.kind
        f["att_dir"] = out.target_dir_body
        f["att_target"] = out.target_gate_index
        return f

    def _ribbon_points(self):
        """Course path points at increasing look-ahead, from the drone's own position."""
        n = self.n
        a = np.clip(self.active, 0, self.n_slots_course - 1)
        nodes = [self.p]
        for j in range(4):
            nodes.append(self.g_pos[self._rows, np.clip(a + j, 0, self.n_slots_course - 1)])
        nodes = np.stack(nodes, axis=1)                       # [n, 5, 3]
        seg = np.linalg.norm(np.diff(nodes, axis=1), axis=2)  # [n, 4]
        cum = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, axis=1)], axis=1)
        out = np.empty((n, N_RIBBON, 3))
        for j, s0 in enumerate(RIBBON_LOOKAHEAD):
            s = np.minimum(s0, cum[:, -1] - 1e-6)
            i = np.clip(np.sum(cum[:, :-1] <= s[:, None], axis=1) - 1, 0, 3)
            t = (s - cum[self._rows, i]) / np.maximum(seg[self._rows, i], 1e-9)
            out[:, j] = (nodes[self._rows, i]
                         + t[:, None] * (nodes[self._rows, i + 1] - nodes[self._rows, i]))
        return out

    def _encode(self, f):
        """The fast path to `interface.Observation.to_vector()`. Same order, same masking."""
        n = self.n
        cols = []
        for j in range(N_SLOTS):
            val = f["gate_valid"][:, j][:, None]
            nv = (f["gate_valid"][:, j] & f["gate_nvalid"][:, j])[:, None]
            cols.append(np.where(val, f["gate_pos"][:, j], 0.0))
            cols.append(np.where(nv, f["gate_nrm"][:, j], 0.0))
            cols.append(np.stack([
                f["gate_valid"][:, j].astype(np.float64),
                f["gate_nvalid"][:, j].astype(np.float64),
                f["gate_pvalid"][:, j].astype(np.float64),
                np.where(f["gate_valid"][:, j], f["gate_conf"][:, j], 0.0),
                np.where(f["gate_valid"][:, j], np.minimum(f["gate_stale"][:, j], 5.0), 5.0),
            ], axis=1))
        rv = f["rb_valid"][:, None]
        cols.append(np.where(rv, f["rb_bear"], 0.0))
        cols.append(np.where(rv, f["rb_elev"], 0.0))
        cols.append(np.stack([
            f["rb_valid"].astype(np.float64),
            np.where(f["rb_valid"], f["rb_pix"], 0.0),
            np.where(f["rb_valid"], np.minimum(f["rb_stale"], 5.0), 5.0)], axis=1))
        cols.append(f["gyro"])
        cols.append(f["accel"])
        cols.append(np.stack([f["roll"], f["pitch"], f["att_conf"]], axis=1))
        cols.append(np.stack([np.sin(f["vel_bearing"]), np.cos(f["vel_bearing"]),
                              f["vel_valid"].astype(np.float64)], axis=1))
        cols.append(np.stack([f["speed"], f["speed_conf"]], axis=1))
        cols.append(np.stack([
            f["active"] / np.maximum(f["n_gates"], 1),
            np.minimum(f["t_gate"], 30.0),
            np.minimum(f["t_coll"], 5.0),
            f["armed"].astype(np.float64)], axis=1))
        cols.append(f["att_dir"])
        cols.append(np.stack([(f["att_kind"] == k).astype(np.float64)
                              for k in (interface.Attn.SEARCH, interface.Attn.GATE_CURRENT,
                                        interface.Attn.GATE_NEXT, interface.Attn.RIBBON)],
                             axis=1))
        out = np.concatenate(cols, axis=1).astype(np.float32)
        assert out.shape == (n, OBS_DIM), out.shape
        return out

    def throttle_for_thrust(self, t_mps2):
        """Invert this episode's thrust curve: the throttle that yields `T/m` `[n]`.

        The measured knot table is monotone, so `np.interp` inverts it directly. Used by
        anything that reasons in accelerations (the baseline's altitude loop, the
        self-check's scripted guidance) rather than in throttle units.
        """
        t = np.asarray(t_mps2, dtype=float) / np.maximum(self.thrust_gain, 1e-6)
        return np.interp(t, self._knot_y, self._knot_x)

    def observation(self, i):
        """Env `i` as an `interface.Observation` dataclass, from the same float64 fields."""
        return build_observation(self._fields, i)


def build_observation(f, i):
    """`interface.Observation` for env `i` of a fields dict. Shared with `single.py`."""
    gates = []
    for j in range(N_SLOTS):
        valid = bool(f["gate_valid"][i, j])
        gates.append(interface.GateObs(
            valid=valid,
            index=int(f["gate_index"][i, j]),
            pos_body=np.array(f["gate_pos"][i, j], dtype=float),
            normal_body=np.array(f["gate_nrm"][i, j], dtype=float),
            normal_valid=bool(f["gate_nvalid"][i, j]),
            pose_valid=bool(f["gate_pvalid"][i, j]),
            range_sigma_m=float(f["gate_rsig"][i, j]),
            confidence=float(f["gate_conf"][i, j]),
            staleness_s=float(f["gate_stale"][i, j]),
            size_px=float(f["gate_size"][i, j])))
    ribbon = interface.RibbonObs(
        valid=bool(f["rb_valid"][i]),
        bearings_rad=np.array(f["rb_bear"][i], dtype=float),
        elevs_rad=np.array(f["rb_elev"][i], dtype=float),
        pixel_fraction=float(f["rb_pix"][i]),
        staleness_s=float(f["rb_stale"][i]))
    own = interface.SelfObs(
        gyro=np.array(f["gyro"][i], dtype=float),
        accel=np.array(f["accel"][i], dtype=float),
        roll_rad=float(f["roll"][i]), pitch_rad=float(f["pitch"][i]),
        attitude_conf=float(f["att_conf"][i]),
        vel_bearing_rad=float(f["vel_bearing"][i]),
        vel_valid=bool(f["vel_valid"][i]),
        speed_est_mps=float(f["speed"][i]), speed_conf=float(f["speed_conf"][i]))
    race = interface.RaceObs(
        active_gate_index=int(f["active"][i]), n_gates_total=int(f["n_gates"][i]),
        t_since_gate_s=float(f["t_gate"][i]), race_time_s=float(f["race_t"][i]),
        armed=bool(f["armed"][i]), t_since_collision_s=float(f["t_coll"][i]),
        collision_episodes=int(f["coll_n"][i]))
    att = interface.Attention(
        kind=interface.Attn(int(f["att_kind"][i])),
        target_dir_body=np.array(f["att_dir"][i], dtype=float),
        target_gate_index=int(f["att_target"][i]))
    return interface.Observation(t_s=float(f["t_s"][i]), dt_s=float(f["dt_s"][i]),
                                 gates=gates, ribbon=ribbon, own=own, race=race,
                                 attention=att)
