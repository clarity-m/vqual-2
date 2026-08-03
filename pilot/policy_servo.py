"""policy_servo.py -- the scripted fallback: a visual-servo Policy over the frozen
interface. No training, no surrogate, no map coordinates -- it consumes exactly what
`interface.Observation` publishes and steers at the current gate.

DESIGN, and why it is shaped this way:

* Completion beats speed. Scoring ranks on time, but a lap that finishes beats any lap
  that does not, the cap is 8 minutes, and Claire's gentle hand-flown lap took 210 s.
  Everything here is tuned toward "slow and certain": speed comes from a tilt schedule
  whose ceiling is deliberately low, and anything ambiguous means "level out and let
  drag brake".

* Two nested loops, both proportional. The outer loop turns the gate's position in a
  gravity-levelled frame into a roll/pitch ATTITUDE target; the inner loop turns
  attitude error into the body-rate command the interface wants. The inner gain (4.0
  rad/s per rad) and clamp (3.0 rad/s) are teleop's LEVEL_GAIN / LEVEL_MAX_RATE -- the
  values flight-proven by the levelling assist, not new guesses.

* Speed control is open-loop through drag, because speed is not observable (speed_est
  is stubbed at 0). At a held pitch tilt t the airframe settles at
  |v| = sqrt(g*tan(t)/k) with k = 0.0425/m (the fitted drag model): 5 deg -> ~4.5 m/s,
  10 deg -> ~6.4 m/s. Commanding tilt IS commanding terminal speed, with the plant
  itself closing the loop. The schedule maps range to tilt, so the aircraft slows on
  final without any speed feedback.

* Yaw belongs to the producer (YawMode.AUTO_ATTENTION). The whole policy is 3-DoF:
  roll, pitch, thrust. When nothing is tracked, the producer's SEARCH sweep is also
  what re-finds a gate; this policy's job in that state is only to hold level and
  hover, presenting a stable camera.

* All outputs are CANONICAL signs (interface.py header). The harness's link layer owns
  the -1 mirror; nothing here may flip a sign.

Gains are constants below, grouped and commented -- they will be tuned live against the
sim with Claire watching, so every one of them states its unit and its first-guess
provenance.
"""

from __future__ import annotations

import math

import numpy as np

from interface import (Action, HOVER_THRUST, Observation, Policy, YawMode)

# --- inner attitude loop (flight-proven values from teleop's levelling assist) --------
G_ATT = 4.0               # rad/s of body rate per rad of attitude error (LEVEL_GAIN)
RATE_MAX = 3.0            # rad/s clamp on commanded roll/pitch rates (LEVEL_MAX_RATE)

# --- outer steering loop --------------------------------------------------------------
K_BANK = 1.2              # rad of bank target per rad of levelled bearing error
BANK_MAX = math.radians(25.0)   # never ask for more bank than this
BRG_ADVANCE = math.radians(35.0)  # advance (pitch forward) only when this well aimed;
                                  # beyond it, hold position and let yaw bring the nose
                                  # around first -- prevents orbiting the gate

# --- speed via the drag-limited tilt schedule ----------------------------------------
TILT_MAX = math.radians(10.0)   # ~6.4 m/s terminal; the "racing" ceiling of a fallback
TILT_MIN = math.radians(4.0)    # ~4 m/s; floor while advancing, so we never stall out
R_SLOW = 6.0             # m; inside this, taper the tilt toward TILT_MIN
R_FAST = 18.0            # m; beyond this, full TILT_MAX

# --- vertical channel -----------------------------------------------------------------
K_CLIMB = 0.030           # thrust offset per metre of height error to the aim point
T_CLIMB_MAX = 0.10        # clamp on the CLIMB side. The ceiling is far away.
T_DESC_MAX = 0.05         # clamp on the DESCENT side (~-1.9 m/s^2). History, one flight
                          # each way (2026-08-02): -0.10 undamped dug into the floor
                          # before gate 1; -0.03 with full damping could not get DOWN to
                          # gate 0 and flew over it. The floor still kills, so descent
                          # keeps less authority than climb -- but it must be enough to
                          # shed ~3 m across one approach.
K_VZD = 0.030             # symmetric vertical damping, thrust per m/s of WASHED-OUT
                          # climb rate (bias-free by construction; see ClimbDamper).
                          # Loop check at hover 0.264: P 1.1 s^-2, D 1.1 s^-1, zeta 0.53.
T_VZD_MAX = 0.05
K_VZ = 0.025              # thrust per m/s of estimated climb rate (the damper). At
                          # 2 m/s of climb this pulls 0.05 of thrust ~ 1.8 m/s^2 down.
T_VZ_MAX = 0.08           # clamp on the damper's authority (DOWN only, see below)
VZ_TAU = 3.0              # s; leak on the climb-rate integrator (drift containment)
VZ_DEADBAND = 0.8         # m/s; ignore climb readings smaller than this. MEASURED
                          # (20260802-195732): this sim's accel under-reads |a| in
                          # flight by 0.1-0.5 m/s^2, a phantom "descent" that biases
                          # vz positive by up to ~1.5 m/s. Acting inside the band
                          # means acting on the artifact.
G_MPS2 = 9.81
VZ_BIAS_TAU = 10.0        # s; the slow tracker that absorbs the accel bias
TILT_COMP_MAX = 1.5       # cap on the 1/cos(tilt) hover compensation (teleop's cap)
AIM_UP_M = 0.2            # aim above the gate centre: fly-high is the safe error on
                          # this course (floor-adjacent gates 0/3/4/5; ceiling far).
                          # 0.2 m spends ~40% of the 0.536 m vertical pass margin.

# --- degradation ----------------------------------------------------------------------
STALE_STEER_S = 1.0       # steer on a coasted track up to this old; then hold level
CONF_MIN = 0.05           # below this attitude_conf the level-frame transform is a
                          # guess; freeze the outer loop and hold the last attitude
                          # target rather than chase a bad frame
R_COMMIT = 4.0            # m; inside this the pass is ballistic (producer hands
                          # attention to the next gate at the same radius) -- hold the
                          # current attitude and thrust, change nothing


def _to_level(v_body: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    """Body -> gravity-levelled frame (yaw-free NED: x along body-forward's horizontal
    projection, y right, z down). v_lvl = Ry(-pitch) @ Rx(-roll) @ v_body for frame
    rotations in the 3-2-1 convention with yaw omitted."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])       # undo roll
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])       # undo pitch
    return ry @ (rx @ np.asarray(v_body, float))


class ClimbDamper:
    """Climb rate from permitted streams: levelled specific force plus gravity,
    leaky-integrated. First flight (2026-08-02) showed why hover thrust alone cannot
    hover: after the takeoff burst the climb rate simply persists -- thrust equal to
    weight is zero NET force, and vertical drag is too weak to bleed it. Nothing in
    VQ2 observes altitude, but climb RATE is derivable the same way roll/pitch are
    (quality, not capability -- the standard NOTES.md already applies to levelling).
    The leak (VZ_TAU) contains accelerometer-bias drift at the price of reading a
    steady climb ~VZ_TAU*az low; for a damper that trade is right."""

    def __init__(self):
        self.vz = 0.0                # m/s, NED: positive DOWN (leaky, bias-poisoned)
        self.bias = 0.0              # slow tracker of vz's DC content (the artifact)
        self._t = None

    def update(self, own, t_s: float) -> float:
        dt = 0.0 if self._t is None else min(0.1, max(0.0, t_s - self._t))
        self._t = t_s
        if dt > 0.0 and own.attitude_conf >= CONF_MIN:
            a_lvl = _to_level(own.accel, own.roll_rad, own.pitch_rad)
            az = float(a_lvl[2]) + G_MPS2         # net accel, down-positive; 0 at hover
            self.vz += az * dt
        if dt > 0.0:
            self.vz *= math.exp(-dt / VZ_TAU)
            self.bias += (self.vz - self.bias) * (1.0 - math.exp(-dt / VZ_BIAS_TAU))
        return self.vz

    @property
    def vz_hp(self) -> float:
        """Washed-out climb rate: the accel artifact is low-frequency, so vz minus its
        own slow average is bias-free over the seconds that matter for damping."""
        return self.vz - self.bias

    def damping(self) -> float:
        """SYMMETRIC vertical damping off the washout -- safe both ways because the
        bias is subtracted. Descending (vz_hp > 0) => positive offset (more thrust)."""
        return min(T_VZD_MAX, max(-T_VZD_MAX, K_VZD * self.vz_hp))

    def thrust_offset(self) -> float:
        """Thrust CUT on clear climbs; never an increase. First flight showed the
        phantom-descent artifact drives any symmetric version to sit at its up-clamp,
        FEEDING a climb instead of damping it. Down-only with a deadband keeps the
        useful half (runaway-climb protection) and amputates the harmful one; slow
        sinks are left to the gate-height term, which has a referee (vision)."""
        vz_clear = min(0.0, self.vz + VZ_DEADBAND)     # <0 only on real climbs
        return max(-T_VZ_MAX, K_VZ * vz_clear)


class ServoPolicy(Policy):
    """Steer at the current gate; hover level when blind. See module docstring."""

    def __init__(self, tilt_max=TILT_MAX, bank_max=BANK_MAX, hover=HOVER_THRUST):
        self.tilt_max = float(tilt_max)
        self.bank_max = float(bank_max)
        self.hover = float(hover)
        self.reset()

    def reset(self) -> None:
        self._hold = (0.0, 0.0)      # last attitude target, for freeze states
        self._mode = "init"          # HUD/diagnostic string, no control effect
        self._damper = ClimbDamper()

    # -- pieces, separable for testing ------------------------------------------------

    def _tilt_for_range(self, rng: float) -> float:
        """Range -> forward tilt (positive number, radians). Linear taper R_SLOW..R_FAST."""
        f = (rng - R_SLOW) / (R_FAST - R_SLOW)
        return TILT_MIN + (self.tilt_max - TILT_MIN) * min(1.0, max(0.0, f))

    def _thrust(self, roll: float, pitch: float, climb_err_m: float) -> float:
        """Hover, tilt-compensated off the ESTIMATED attitude; P on height with
        floor-respecting asymmetric clamps; washout damping; runaway cut.

        The damper must not FIGHT a commanded descent: symmetric damping opposes any
        sink, which rate-limited the previous flight to ~1 m/s and contributed to
        sailing over gate 0. During a deliberate descent its add-thrust side is capped
        low; its cut-thrust side (arresting a dive) keeps full authority always."""
        ct = math.cos(roll) * math.cos(pitch)
        comp = TILT_COMP_MAX if ct <= 1.0 / TILT_COMP_MAX else 1.0 / ct
        off = min(T_CLIMB_MAX, max(-T_DESC_MAX, K_CLIMB * climb_err_m))
        d = self._damper.damping()
        if climb_err_m < -0.5 and d > 0.015:
            d = 0.015
        return min(1.0, max(0.0, self.hover * comp + off + d
                            + self._damper.thrust_offset()))

    # -- the policy --------------------------------------------------------------------

    def __call__(self, obs: Observation) -> Action:
        own = obs.own
        gate = obs.gates[0]
        self._damper.update(own, obs.t_s)

        # Attitude estimate is load-bearing for every branch below. If it is junk,
        # do not chase anything: command rates toward the LAST attitude target we
        # trusted (usually near-level) at reduced authority.
        if own.attitude_conf < CONF_MIN:
            self._mode = "att-blind"
            roll_t, pitch_t = self._hold
            return self._track(own, 0.5 * G_ATT * 0.0 + roll_t, pitch_t, 0.0)

        usable = gate.valid and gate.staleness_s <= STALE_STEER_S
        if not usable:
            # Blind: nose-DOWN a touch and sink slowly while the yaw sweep runs.
            # Level-and-hover proved to be a deadlock (hop 2, 20260802-205619): the
            # frame spans +49.4/-9.4 about body-forward, so a level platform can only
            # ever re-find gates NEAR ITS OWN ALTITUDE -- and the residual hover-thrust
            # bias climbs, so a blind drone rises until nothing is ever in frame again
            # (it was photographing the ceiling grid for 40 s). Slight nose-down tips
            # the searchable band onto the course, and the sink bias (~-0.15 m/s^2)
            # makes altitude converge toward the gates instead of away from them.
            self._mode = "search"
            self._hold = (0.0, 0.0)
            a = self._track(own, 0.0, math.radians(-4.0), 0.0)
            return Action(a.roll_rate, a.pitch_rate, a.yaw_rate,
                          max(0.0, a.thrust - 0.004), a.yaw_mode)

        p_lvl = _to_level(gate.pos_body, own.roll_rad, own.pitch_rad)
        rng = float(np.linalg.norm(gate.pos_body))
        brg = math.atan2(p_lvl[1], p_lvl[0])          # >0: gate to the RIGHT
        climb_err = -float(p_lvl[2]) + AIM_UP_M        # >0: gate ABOVE us (z is down)

        if rng < R_COMMIT:
            # Ballistic. Attention has already moved on; changing anything now only
            # adds attitude rate during the pass. Freeze what got us here.
            self._mode = "commit"
            roll_t, pitch_t = self._hold
            return self._track(own, roll_t, pitch_t, 0.0)

        roll_t = min(self.bank_max, max(-self.bank_max, K_BANK * brg))
        if abs(brg) <= BRG_ADVANCE:
            self._mode = "track"
            pitch_t = -self._tilt_for_range(rng)      # nose DOWN (negative) advances
        else:
            # Badly aimed: do not translate toward where we are not looking. Level
            # pitch (drag brakes), keep the bank so the turn continues, let the yaw
            # servo finish bringing the nose around.
            self._mode = "aim"
            pitch_t = 0.0
        self._hold = (roll_t, pitch_t)
        return self._track(own, roll_t, pitch_t, climb_err)

    def _track(self, own, roll_t: float, pitch_t: float, climb_err: float) -> Action:
        """Inner loop: attitude target -> body rates, plus the thrust channel."""
        roll_rate = min(RATE_MAX, max(-RATE_MAX, G_ATT * (roll_t - own.roll_rad)))
        pitch_rate = min(RATE_MAX, max(-RATE_MAX, G_ATT * (pitch_t - own.pitch_rad)))
        return Action(roll_rate=roll_rate, pitch_rate=pitch_rate, yaw_rate=0.0,
                      thrust=self._thrust(own.roll_rad, own.pitch_rad, climb_err),
                      yaw_mode=YawMode.AUTO_ATTENTION).clipped()


# ---------------------------------------------------------------------------------------
# Self test: sign sanity on synthetic observations. Cheap, offline, and exactly the
# class of error (a mirrored axis) that has burned this project before.

def selftest():
    from interface import GateObs

    p = ServoPolicy()

    def obs_with(pos, roll=0.0, pitch=0.0, conf=1.0, valid=True):
        o = Observation()
        o.own.roll_rad, o.own.pitch_rad, o.own.attitude_conf = roll, pitch, conf
        g = GateObs(valid=valid, index=1, pos_body=np.array(pos, float))
        o.gates[0] = g
        return o

    # Gate dead ahead, 15 m: advance -> nose-down pitch rate (negative), no roll.
    p.reset()
    a = p(obs_with([15.0, 0.0, 0.0]))
    assert a.pitch_rate < 0 and abs(a.roll_rate) < 1e-9, (a.roll_rate, a.pitch_rate)

    # Gate to the RIGHT -> positive (right-wing-down) roll rate. Canonical.
    p.reset()
    a = p(obs_with([12.0, 6.0, 0.0]))
    assert a.roll_rate > 0, a.roll_rate

    # Gate ABOVE (z negative in body NED) -> thrust above the level hover point.
    p.reset()
    a = p(obs_with([15.0, 0.0, -4.0]))
    assert a.thrust > HOVER_THRUST, a.thrust

    # Already banked right at the target bank -> roll rate ~ 0, not still positive.
    p.reset()
    o = obs_with([12.0, 6.0, 0.0], roll=BANK_MAX)
    assert abs(p(o).roll_rate) < G_ATT * BANK_MAX, p(o).roll_rate

    # Nothing seen -> level + hover: rates oppose current attitude, thrust ~ hover*comp.
    p.reset()
    a = p(obs_with([0, 0, 0], roll=0.3, pitch=-0.2, valid=False))
    assert a.roll_rate < 0 and a.pitch_rate > 0, (a.roll_rate, a.pitch_rate)

    # Attitude blind -> does not chase the gate.
    p.reset()
    a = p(obs_with([12.0, 6.0, 0.0], conf=0.0))
    assert abs(a.roll_rate) < 1e-9, a.roll_rate

    # Commit zone -> holds, and thrust stays put (no climb chase inside 4 m).
    p.reset()
    p(obs_with([15.0, 0.0, 0.0]))          # establish a hold
    a = p(obs_with([3.0, 0.5, -0.5]))
    assert p._mode == "commit", p._mode

    print("policy_servo selftest OK")


if __name__ == "__main__":
    selftest()
