"""The reactive baseline: proportional guidance off the gate approach point.

This is the submission floor. It is the thing that flies if PPO never trains, so it is
written to be boring and legible rather than clever. It carries exactly ONE piece of
state -- the hover trim, below -- and no filters, no memory of the target, nothing else
that can wind up or diverge. Every number in it is a constructor argument, because it
will be tuned against the surrogate in minutes and against the live sim in a handful of
runs.

THE THREE LOOPS, and why each axis is assigned the way it is
------------------------------------------------------------
Under `YawMode.AUTO_ATTENTION` the policy owns roll, pitch and thrust; yaw belongs to
the attention servo. That leaves a clean assignment:

  roll   <- horizontal steering. Bank is the ONLY way to move the trajectory sideways
            (yaw is a free gimbal and produces a skid, not a turn).
  thrust <- vertical steering: climb toward the target's elevation above the HORIZON.
  pitch  <- forward speed, minus whatever the camera's lower frame edge demands.

STEER ON THE VELOCITY BEARING, NOT THE NOSE BEARING
---------------------------------------------------
The obvious law -- bank proportional to the target's bearing -- is wrong here, and
wrong in a way that hides. Under `AUTO_ATTENTION` the yaw servo is already driving the
attended target's bearing to zero, so `bearing_rad` collapses toward 0 no matter where
the aircraft is actually going. A nose-referenced pursuit therefore stops banking while
still translating sideways past the gate.

`SelfObs.vel_bearing_rad` is the fix and it is free: horizontal body accelerometer
components can only be drag (thrust is along body -z by definition), drag is
antiparallel to airspeed, so `atan2(-ay, -ax)` is the direction of travel relative to
the nose -- algebraic, no integration (NOTES.md). The guidance error is then

    b_err = bearing(target) - vel_bearing

with both terms nose-referenced, so the unknown heading cancels. Near hover drag goes
to zero and the bearing is noise; the producer clears `vel_valid` there and the law
falls back to nose-referenced pursuit, which is correct at low speed.

WHICH WAY PITCH MOVES A TARGET IN THE FRAME  -- derived, not assumed
--------------------------------------------------------------------
Body NED, pitch angle `th` (nose up positive), a target dead ahead and level with us at
distance D. R_wb = Ry(th), so

    p_body = Ry(th)^T (D,0,0) = (D cos th, 0, D sin th)
    elev   = atan2(-p_z, hypot(p_x,p_y)) = -th

    *** NOSE UP LOWERS THE TARGET IN THE FRAME. NOSE DOWN RAISES IT. ***

The camera is tilted 20 deg UP and the frame spans +49.4 deg to -9.4 deg about
body-forward, so the lower edge is 9.4 deg away and the upper edge 49.4 deg: the low
edge is the binding one (interface.py, CONVENTIONS.md). Combining the two facts, the
manoeuvre that loses a gate out of the bottom of the frame is PITCHING UP -- which is
also how a quad slows down. That is the real coupling on this course: braking blinds
you. The frame guard below therefore adds NOSE-DOWN when the target approaches the
-9.4 deg edge, and nose-up only near +49.4 deg where nothing else ever goes.

  (NOTE FOR THE ARCHITECTURE OWNER: interface.py and NOTES.md both say "pitching down
  to accelerate pushes it lower still". For a body-fixed camera the algebra above says
  the opposite, and the descent case those sentences also cite -- gates below the
  aircraft -- needs nose-DOWN to frame. The code follows the algebra. If a live run
  ever shows gates leaving the frame while pitching down, this comment is where to
  start, because that would mean the camera tilt is negative and the binding edge is
  the upper one.)

THE VERTICAL LOOP IS HORIZON-REFERENCED
----------------------------------------
`GateObs.elev_rad` is body-referenced, so at a 22 deg nose-down cruise trim a target
level with us reads +22 deg. Feeding that to thrust would command a permanent climb.
The climb loop therefore uses `elev_body + own.pitch_rad`, which is the target's
elevation above the horizon to first order (the identity `elev = -th` above), with
`pitch_rad` read off gravity. The frame guard, by contrast, is a statement about the
camera and stays body-referenced. Getting those two mixed up is a silent altitude bias.

DEGRADED INPUTS
---------------
Per `GateObs`'s documented degradation order:
  * `normal_valid` False -> no approach point; steer at the gate centre.
  * `pose_valid` False   -> range is biased LONG; shrink it by `range_sigma_m` before
                            deciding how much approach offset is still safe.
  * track coasting       -> `valid` stays True and `staleness_s` grows. Authority and
                            speed decay linearly from `stale_soft_s` to `stale_max_s`,
                            past which the gate is not used at all.
  * no gate              -> the ribbon, at reduced speed. It is a pre-turn prior and
                            measurably absent for long stretches, never load-bearing.
  * nothing              -> wings level, cruise trim for `v_min`, hover thrust. Slow
                            down and stay flat so attention can re-acquire.

THE CLIMB LOOP IS VELOCITY-REFERENCED TOO  (2026-08-01)
--------------------------------------------------------
The roll loop closes on where the aircraft is GOING, not where it is pointing, for the
reason above. The climb loop used to close on elevation alone -- `a_des = k * e_horizon`
-- and that is proportional control commanding an ACCELERATION, i.e. an undamped double
integrator. It has no equilibrium at the target: elevation error goes to zero at the
moment the aircraft is climbing hardest, so it sails through and comes back. Measured on
the surrogate it held `e_horizon` at 0.23 rad rms, 13 degrees of permanent wander, and
that is a large share of the vertical miss at the gate plane.

The fix is the same law the roll loop uses, one axis over: null the angle between the
velocity vector and the target, not the angle to the target.

    w_cmd = V * e_horizon        the climb rate that points the velocity AT the target
    a_des = k_w * (w_cmd - w)    close on it

which is stable, has the right equilibrium (flying straight at the target), and scales
its own gain with speed instead of being tuned for one cruise.

That needs vertical speed, and `SelfObs` has none -- the drag bearing is horizontal only,
because thrust acts along body -z and swamps the vertical drag term the other two axes
are read from. But `vertical_accel` below IS an exact algebraic read of world-vertical
acceleration, so vertical speed is one integration away. A pure integral would drift over
a whole lap, so it is LEAKY (`w_tau`, 4 s): bounded by construction, cannot run away, and
scored against surrogate truth at 0.38 m/s rms on a 1.53 m/s signal (corr 0.97) -- an
undrifting integral scores 0.29, so the leak costs almost nothing. `w_tau = 0` disables
the estimator entirely and restores the old proportional law with `k_climb_a = k_w * V`.

The leak biases a sustained climb low by ~0.1 m/s, which is the hover trim's job to
absorb, and it is the second piece of state in this file for the same reason as the
first: the alternative is a standing error, not a simpler policy.

WHAT IS DELIBERATELY MISSING
----------------------------
No derivative term on the attitude loops. The rate loop is nearly ideal (gain 0.90-0.97,
tau < 10 ms), so angle <- rate is a clean integrator and proportional control on it is a
stable first-order lag. Gyro damping is available as `k_gyro` but defaults to 0.0, for
one reason: `SelfObs.gyro`'s convention is stated one way by interface.py ("no file
above the link layer may contain a sign flip", i.e. canonical NED) and the other way by
control/plant.py's docstring. A wrong-signed damping term is positive feedback, and the
loop does not need it, so the default declines the bet.

THE ONE INTEGRATOR: HOVER TRIM
------------------------------
`HOVER_THRUST` is a measured constant of ONE aircraft on ONE day -- it has already moved
from 0.55 to 0.25 to 0.27 -- and both the surrogate and the real sim present a thrust
gain that differs from the fit. Training randomizes it by +-10-20% deliberately.
A fixed hover feedforward therefore leaves a standing vertical acceleration of a metre
or two per second squared, and a proportional climb loop converts that into a STANDING
elevation error of bias / (dT/dthrottle * k_climb): at 10% of hover and k_climb = 0.15
that is 0.2 rad, twenty degrees of permanent aim-low. Measured on the surrogate before
this term existed, and it is why the baseline flew into gate frames from below.

So the hover feedforward is estimated instead of assumed: a slow integrator on the
horizon-referenced elevation error, hard-clamped to [0.7, 1.6] x HOVER_THRUST, updated
only while a cue is actually valid (no reference, no integration). It is the only
unknown constant in the vertical loop and the only state in this file. `k_trim = 0`
disables it and restores the pure feedforward.
"""

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot import interface                                    # noqa: E402
from pilot.control.policies.envelope import clamp_action       # noqa: E402

G = 9.81
DRAG_KX = 0.0487        # body-x quadratic drag, fitted 2026-07-31, held-out R^2 0.993

FRAME_LOW_RAD = math.radians(-9.4)    # lower edge of the frame about body-forward
FRAME_HIGH_RAD = math.radians(49.4)   # upper edge


def wrap(a):
    """Angle to (-pi, pi]."""
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


class BaselinePolicy(interface.Policy):
    """Proportional guidance to `pos_body + d * normal_body`. Stateless.

    Every argument is a gain or a limit; none of them is measured, all of them are
    starting points. `self.last` carries the internals of the most recent call for
    tuning and for `selfcheck.py`.
    """

    def __init__(self,
                 # --- guidance target -------------------------------------------
                 approach_d=2.0,          # m along the gate normal, the approach point
                 commit_m=1.5,            # inside this range, steer at the centre
                 conf_min=0.12,           # ignore detections below this confidence
                 conf_full=0.55,          # full authority at or above this confidence
                 conf_floor_w=0.35,       # authority left at conf_min
                 stale_soft_s=0.25,       # full authority out to here
                 stale_max_s=0.90,        # zero authority here; gate unusable beyond
                 ribbon_index=1,          # which ribbon look-ahead sample to steer on
                 ribbon_speed_frac=0.60,  # speed allowance when flying the ribbon
                 # --- roll: bearing -> bank -> rate -----------------------------
                 k_bearing=1.1,           # rad of bank per rad of bearing error
                 bank_max=0.50,           # rad (29 deg): 5.5 m/s^2 of lateral accel
                 k_roll=4.0,              # 1/s on bank error. dt*k < 1 at 30 Hz
                 # --- pitch: speed + frame guard --------------------------------
                 v_cruise=6.5,            # m/s on a clear line
                 v_min=2.5,               # m/s when blind or fully coasting
                 turn_slow=1.6,           # speed shed per rad of bearing error
                 k_speed=0.06,            # rad of pitch per m/s of speed error
                 k_pitch=5.0,             # 1/s on pitch error
                 pitch_down_max=0.55,     # rad
                 pitch_up_max=0.30,       # rad; braking blinds, so keep it small
                 elev_floor_rad=math.radians(-5.0),   # 4.4 deg of margin on -9.4
                 elev_ceil_rad=math.radians(40.0),    # 9.4 deg of margin on +49.4
                 k_frame=2.0,             # rad of pitch per rad of frame violation
                 # --- thrust: climb, commanded as an ACCELERATION ---------------
                 k_w=2.0,                 # 1/s on climb-rate error
                 w_tau=4.0,               # s, leak on the climb-rate estimator; 0 = off
                 w_max=6.0,               # m/s cap on the commanded climb rate
                 v_ref_min=3.0,           # m/s floor under V in w_cmd = V * elevation
                 k_climb_a=6.0,           # fallback m/s^2 per rad when w_tau = 0
                 climb_a_max=4.0,         # m/s^2
                 a_per_thrust=59.6,       # fitted dT/dthrottle; 1/this converts back
                 k_acc=0.70,              # accel-error feedback, dimensionless
                 comp_max=1.8,            # cap on the 1/(cos roll cos pitch) term
                 k_trim=0.5,              # hover-trim integrator, 1/s
                 trim_min=0.19,           # 0.7 * HOVER_THRUST
                 trim_max=0.43,           # 1.6 * HOVER_THRUST
                 # --- misc --------------------------------------------------------
                 k_gyro=0.0,              # rate damping; see the module docstring
                 drag_kx=DRAG_KX):
        self.approach_d = float(approach_d)
        self.commit_m = float(commit_m)
        self.conf_min = float(conf_min)
        self.conf_full = float(conf_full)
        self.conf_floor_w = float(conf_floor_w)
        self.stale_soft_s = float(stale_soft_s)
        self.stale_max_s = float(stale_max_s)
        self.ribbon_index = int(ribbon_index)
        self.ribbon_speed_frac = float(ribbon_speed_frac)
        self.k_bearing = float(k_bearing)
        self.bank_max = float(bank_max)
        self.k_roll = float(k_roll)
        self.v_cruise = float(v_cruise)
        self.v_min = float(v_min)
        self.turn_slow = float(turn_slow)
        self.k_speed = float(k_speed)
        self.k_pitch = float(k_pitch)
        self.pitch_down_max = float(pitch_down_max)
        self.pitch_up_max = float(pitch_up_max)
        self.elev_floor_rad = float(elev_floor_rad)
        self.elev_ceil_rad = float(elev_ceil_rad)
        self.k_frame = float(k_frame)
        self.k_w = float(k_w)
        self.w_tau = float(w_tau)
        self.w_max = float(w_max)
        self.v_ref_min = float(v_ref_min)
        self.k_climb_a = float(k_climb_a)
        self.climb_a_max = float(climb_a_max)
        self.a_per_thrust = float(a_per_thrust)
        self.k_acc = float(k_acc)
        self.comp_max = float(comp_max)
        self.k_trim = float(k_trim)
        self.trim_min = float(trim_min)
        self.trim_max = float(trim_max)
        self.k_gyro = float(k_gyro)
        self.drag_kx = float(drag_kx)
        self.reset()

    # -- interface.Policy ------------------------------------------------------
    def reset(self):
        self.last = {}
        self.trim = interface.HOVER_THRUST
        self.w_est = 0.0

    def __call__(self, obs):
        own = obs.own
        cue, b_tgt, e_tgt, e_frame, w = self._cue(obs)

        # --- roll: close the angle between where we are going and where the target is
        vel_b = float(own.vel_bearing_rad) if own.vel_valid else 0.0
        b_err = wrap(b_tgt - vel_b)
        bank_des = w * float(np.clip(self.k_bearing * b_err, -self.bank_max, self.bank_max))
        roll_rate = self.k_roll * (bank_des - float(own.roll_rad)) \
            - self.k_gyro * float(own.gyro[0])

        # --- pitch: trim for the speed we want, then keep the target in frame
        v_tgt = self._speed_target(cue, w, b_err)
        pitch_ff = -math.atan2(self.drag_kx * v_tgt * v_tgt, G)
        pitch_fb = -self.k_speed * (v_tgt - float(own.speed_est_mps)) \
            * float(np.clip(own.speed_conf, 0.0, 1.0))
        pitch_des = pitch_ff + pitch_fb
        if e_frame < self.elev_floor_rad:
            pitch_des += self.k_frame * (e_frame - self.elev_floor_rad)   # nose DOWN
        elif e_frame > self.elev_ceil_rad:
            pitch_des += self.k_frame * (e_frame - self.elev_ceil_rad)    # nose up
        pitch_des = float(np.clip(pitch_des, -self.pitch_down_max, self.pitch_up_max))
        pitch_rate = self.k_pitch * (pitch_des - float(own.pitch_rad)) \
            - self.k_gyro * float(own.gyro[1])

        # --- thrust: hold the vertical component through the bank, then climb
        e_horizon = e_tgt + float(own.pitch_rad)
        cc = math.cos(float(own.roll_rad)) * math.cos(float(own.pitch_rad))
        cc = max(cc, 1.0 / self.comp_max)
        comp = 1.0 + float(np.clip(own.attitude_conf, 0.0, 1.0)) * (1.0 / cc - 1.0)
        a_up = self.vertical_accel(own)
        dt = max(float(obs.dt_s), 0.0)
        if self.w_tau > 0.0:
            self.w_est = float(np.clip(
                self.w_est * max(0.0, 1.0 - dt / self.w_tau) + a_up * dt, -25.0, 25.0))
            # Climb rate that would put the velocity vector on the target, then close on
            # it. `w` gates the DEMAND, not the damping: with no cue there is nothing to
            # aim at, but arresting an inherited climb is exactly "stay flat".
            v_ref = max(self._finite(own.speed_est_mps), self.v_ref_min)
            w_cmd = w * float(np.clip(v_ref * math.sin(e_horizon),
                                      -self.w_max, self.w_max))
            a_raw = self.k_w * (w_cmd - self.w_est)
        else:
            w_cmd = w * self.k_climb_a * e_horizon
            a_raw = w_cmd
        a_des = float(np.clip(a_raw, -self.climb_a_max, self.climb_a_max))
        a_err = a_des - a_up
        per_a = 1.0 / max(self.a_per_thrust, 1e-6)
        trim = self.trim
        thrust = trim * comp + per_a * (a_des + self.k_acc * a_err)
        # No reference, no integration: the trim estimates a hover offset against a cue
        # it can actually see, and winding it against pure damping is windup.
        if w > 0.0:
            self.trim = float(np.clip(trim + self.k_trim * per_a * a_err * dt,
                                      self.trim_min, self.trim_max))

        act = clamp_action(roll_rate, pitch_rate, thrust,
                           yaw_rate=0.0, yaw_mode=interface.YawMode.AUTO_ATTENTION)
        self.last = dict(cue=cue, weight=w, b_tgt=b_tgt, b_err=b_err, e_tgt=e_tgt,
                         e_frame=e_frame, e_horizon=e_horizon, bank_des=bank_des,
                         pitch_des=pitch_des, v_tgt=v_tgt, comp=comp, a_des=a_des,
                         a_up=a_up, w_cmd=w_cmd, w_est=self.w_est, trim=trim, action=act)
        return act

    @staticmethod
    def _finite(x, default=0.0):
        x = float(x)
        return x if math.isfinite(x) else default

    @staticmethod
    def vertical_accel(own):
        """Upward acceleration in m/s^2, from the accelerometer and gravity. Yaw-free.

            a_world = R_wb @ specific_force + [0, 0, g]

        and the z ROW of R_wb is [-sin(pitch), cos(pitch) sin(roll), cos(pitch) cos(roll)]
        -- it contains no yaw. So the ONE component of world acceleration that is
        recoverable from permitted telemetry is the vertical one, which is exactly the
        one this loop needs. Nothing is integrated and nothing can diverge; it is the
        same class of algebraic read as the drag bearing.

        Measured against the surrogate's true dv_z/dt: 0.38 m/s^2 rms, sign confirmed.
        """
        r, q = float(own.roll_rad), float(own.pitch_rad)
        row = (-math.sin(q), math.cos(q) * math.sin(r), math.cos(q) * math.cos(r))
        a = np.asarray(own.accel, dtype=float)
        return float(np.clip(-(row[0] * a[0] + row[1] * a[1] + row[2] * a[2]) - G,
                             -25.0, 25.0))

    # -- guidance --------------------------------------------------------------
    def _cue(self, obs):
        """(cue, target bearing, target elevation, framing elevation, authority).

        The framing elevation is the GATE's own elevation, not the approach point's:
        the thing that has to stay inside the frame is the detection.
        """
        gate, w = self._pick_gate(obs)
        if gate is not None:
            t = self._approach_point(gate)
            b = float(np.arctan2(t[1], t[0]))
            e = float(np.arctan2(-t[2], np.hypot(t[0], t[1])))
            return "gate", b, e, gate.elev_rad, w

        rb = obs.ribbon
        if rb.valid and len(rb.bearings_rad) and len(rb.elevs_rad):
            i = min(self.ribbon_index, len(rb.bearings_rad) - 1, len(rb.elevs_rad) - 1)
            b = float(rb.bearings_rad[i])
            e = float(rb.elevs_rad[i])
            return "ribbon", b, e, e, self._coast_weight(rb.staleness_s)

        # Blind. Zero elevations keep the frame guard quiet, which is what we want:
        # there is nothing to frame, so pitch belongs entirely to the speed loop.
        return "blind", 0.0, 0.0, 0.0, 0.0

    def _pick_gate(self, obs):
        """The gate being flown, correlated with `race.active_gate_index`."""
        cands = []
        for g in obs.gates:
            if not g.valid or g.confidence < self.conf_min:
                continue
            w = self._coast_weight(g.staleness_s)
            if w <= 0.0:
                continue
            cands.append((g, w * self._conf_weight(g.confidence)))
        if not cands:
            return None, 0.0

        active = int(obs.race.active_gate_index)
        known = [(g, w) for g, w in cands if g.index >= 0]
        if known:
            for g, w in known:
                if g.index == active:
                    return g, w
            ahead = [(g, w) for g, w in known if g.index > active]
            if ahead:
                return min(ahead, key=lambda gw: gw[0].index)
            # Everything visible is already behind us. The ribbon is a better cue than
            # a gate we have crossed.
            return None, 0.0
        return cands[0]   # indices unknown: trust the producer's list order

    def _approach_point(self, gate):
        """`pos_body + d * normal_body`, with d shrunk to nothing on final approach."""
        p = np.asarray(gate.pos_body, dtype=float)
        if not gate.normal_valid:
            return p
        n = np.asarray(gate.normal_body, dtype=float)
        nn = float(np.linalg.norm(n))
        if nn < 1e-6:
            return p
        n = n / nn
        # The producer signs the normal TOWARD the camera, so it must oppose the
        # line of sight. If it does not, the producer is broken and the approach point
        # would sit on the far side of the gate; fall back rather than flip a sign.
        if float(n @ p) > 0.0:
            return p
        rng = float(np.linalg.norm(p))
        if not gate.pose_valid:
            rng = max(0.0, rng - abs(float(gate.range_sigma_m)))
        d = min(self.approach_d, max(0.0, rng - self.commit_m))
        return p + d * n

    def _conf_weight(self, confidence):
        """Authority scaled by the producer's confidence.

        The named false-positive risk on this course is the white ceiling lights
        (NOTES.md), and the surrogate models them as plausible tracks at confidence
        0.15-0.50 living 2-9 frames -- while a real gate at 25 m, near the edge of
        detection range, comes in around 0.45. No single-frame threshold separates
        those, deliberately. So the answer is not a better filter: it is that a
        marginal cue steers gently and a confident one steers hard, which turns a
        phantom into a small heading error instead of a bank.
        """
        if self.conf_full <= self.conf_min:
            return 1.0
        f = (float(confidence) - self.conf_min) / (self.conf_full - self.conf_min)
        return self.conf_floor_w + (1.0 - self.conf_floor_w) * float(np.clip(f, 0.0, 1.0))

    def _coast_weight(self, staleness_s):
        s = float(staleness_s)
        if s <= self.stale_soft_s:
            return 1.0
        if s >= self.stale_max_s:
            return 0.0
        return (self.stale_max_s - s) / (self.stale_max_s - self.stale_soft_s)

    def _speed_target(self, cue, w, b_err):
        if cue == "blind":
            return self.v_min
        v = self.v_min + (self.v_cruise - self.v_min) / (1.0 + self.turn_slow * abs(b_err))
        if cue == "ribbon":
            v = self.v_min + (v - self.v_min) * self.ribbon_speed_frac
        return self.v_min + (v - self.v_min) * w


def gains_from_str(spec):
    """`"k_roll=7,v_cruise=6.5"` -> kwargs. Lets the eval CLIs tune without an edit."""
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part or part == "-":
            continue
        k, _, v = part.partition("=")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise ValueError("bad gain spec %r: expected name=number" % part)
    return out
