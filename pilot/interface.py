"""
vqual-2 — the frozen contract between PERCEPTION and CONTROL.

Both halves import this module and nothing else from each other. Perception fills an
Observation; control returns an Action. The surrogate and the live sim present the
identical interface, so a policy that runs against one runs against the other unchanged.

Nothing here knows about the world frame, absolute position, or absolute yaw. That is
deliberate and structural, not an omission: VQ2 spec 9.3 blocks ATTITUDE,
LOCAL_POSITION_NED, ODOMETRY and gate geometry, and absolute yaw is unrecoverable from
what remains (gravity is symmetric about it, no magnetometer). Every quantity below is
body-referenced or an elapsed time.


================================ SIGN CONVENTIONS =================================

Canonical body NED, textbook:  x forward,  y right,  z down.

    roll_rate  > 0  ->  right wing down
    pitch_rate > 0  ->  nose up
    yaw_rate   > 0  ->  nose right
    bearing    > 0  ->  target is RIGHT of the nose
    elevation  > 0  ->  target is ABOVE the nose

Measured state of the simulator, flight-confirmed via teleop:

    roll_rate   INVERTED vs the convention above  (SIGN_ROLL = -1)
    pitch_rate  to spec
    yaw_rate    to spec

The one correction lives in the link layer ONLY, applied when encoding a MAVLink
message. No file above the link layer may contain a sign flip.

This is cheap insurance, not a crisis: an acro sign error is SELF-ANNOUNCING -- the
drone visibly rolls the wrong way within a second of the first flight, which is exactly
how the roll inversion was found and fixed. Body rates in, body-frame perception out;
nothing in the flight stack ever converts to a world frame.

    *** WHERE SIGN ERRORS ARE ACTUALLY DANGEROUS: THE SURROGATE FIT. ***

That is the only place a world frame appears. A sign error in the body->world conversion
produces a model that is PLAUSIBLE -- it fits the recordings and yields sensible
trajectories -- while silently mistuning every gain trained against it. It then presents
as "great on the surrogate, bad in the sim", which reads like a sim-to-real gap and
sends you off tuning the noise model instead.

Mandatory before any gain is tuned on the surrogate: replay recorded commands through
the fitted model and compare predicted vs. recorded IMU. A wrong sign diverges
immediately. Offline, no sim time, turns the invisible error visible.


==================================== GEOMETRY =====================================

Camera shares the body origin, tilted 20 deg UP. Pinhole, no distortion:
640 x 360, cx/cy = 320/180, fx = fy = 320.

The spec calls 90 deg "VFoV"; it is the HORIZONTAL FoV. True vertical FoV is 58.7 deg,
so the frame spans +49.4 deg (up) to -9.4 deg (down) about body-forward. That narrow
lower edge is the binding constraint on this course -- a gate at own altitude renders
LOW, and pitching down to accelerate pushes it lower still.

Gate inner aperture is 1500 mm square (spec-exact), which makes apparent size a
calibrated rangefinder:  range_m = 320 * 1.5 / gate_px  =  480 / gate_px.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

# Gates carried in the observation: the one being flown plus lookahead.
N_GATES = 3
# Ribbon direction samples, at increasing look-ahead distance.
N_RIBBON = 6

OBS_DIM = 70

HOVER_THRUST = 0.25  # measured in flight, NOT the 0.55 originally guessed
MAX_RATE_RPS = 6.0
GATE_INNER_M = 1.5
FOCAL_PX = 320.0


class Attn(IntEnum):
    """What perception is currently pointing the camera at."""

    SEARCH = 0  # nothing held; sweeping
    GATE_CURRENT = 1
    GATE_NEXT = 2
    RIBBON = 3


class YawMode(IntEnum):
    """Who drives yaw.

    Quadrotor yaw does not disturb the flight path -- it produces a skid, nose one way
    and momentum the other. That makes it a FREE GIMBAL: the camera can be pointed
    anywhere at no trajectory cost, and roll alone owns the trajectory.

    AUTO_ATTENTION delegates yaw to a fixed servo that drives Attention.target_dir_body
    to the frame centre, leaving the policy a 3-DoF problem (roll, pitch, thrust).
    Recommended default -- a learned policy should not have to rediscover "point the
    camera at the thing you are looking at". POLICY takes the fourth axis back.
    """

    AUTO_ATTENTION = 0
    POLICY = 1


@dataclass
class GateObs:
    """One gate, in the body frame.

    Degrades in a defined order rather than lying. Full PnP on the known 1500 mm square
    gives position AND normal. When the aperture is too small or too head-on to
    disambiguate, normal_valid goes False and position survives. When the quad fit fails
    entirely, position comes from centroid + apparent size. When nothing is seen, valid
    stays True while the track coasts and staleness_s grows -- the consumer decides how
    stale is too stale.
    """

    valid: bool = False
    index: int = -1  # absolute race gate index, to correlate with RaceObs

    pos_body: np.ndarray = field(default_factory=lambda: np.zeros(3))  # metres, x/y/z

    # Unit vector along the gate axis, pointing back OUT toward the approach side.
    # This is what makes "enter at the normal" expressible: the guidance target is a
    # virtual approach point at pos_body + d * normal_body, not the gate centre itself.
    normal_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal_valid: bool = False

    confidence: float = 0.0  # 0..1
    staleness_s: float = 0.0  # 0.0 == observed in the current frame
    size_px: float = 0.0

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.pos_body))

    @property
    def bearing_rad(self) -> float:
        return float(np.arctan2(self.pos_body[1], self.pos_body[0]))

    @property
    def elev_rad(self) -> float:
        h = float(np.hypot(self.pos_body[0], self.pos_body[1]))
        return float(np.arctan2(-self.pos_body[2], h))


@dataclass
class RibbonObs:
    """The cyan guidance corridor, as direction samples at increasing look-ahead.

    NOT LOAD-BEARING, by design constraint: it is measurably absent from long stretches
    of real flight. A policy must fly without it. It is a prior for pre-turning into
    corners before the next gate is visible, not a primary cue.
    """

    valid: bool = False
    bearings_rad: np.ndarray = field(default_factory=lambda: np.zeros(N_RIBBON))
    elevs_rad: np.ndarray = field(default_factory=lambda: np.zeros(N_RIBBON))
    pixel_fraction: float = 0.0  # cyan share of frame; the dropout detector
    staleness_s: float = 0.0


@dataclass
class SelfObs:
    """Own state, restricted to what VQ2 actually permits.

    There is no position and no yaw field here, and there never will be.
    """

    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s, measured
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s^2, specific force

    # Roll and pitch from the gravity vector. Directly observable; degrades under
    # racing accelerations, hence the confidence.
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    attitude_conf: float = 0.0

    # Direction of travel relative to the nose, from drag.
    #
    # Thrust acts along body -z BY DEFINITION, so it contributes exactly zero to the
    # horizontal body accelerometer components. In flight those can only be measuring
    # drag, and drag is antiparallel to airspeed:  (ax, ay) prop. -(vx, vy)_body.
    # An instantaneous algebraic read of a permitted sensor -- no integration, no filter,
    # nothing to diverge. Invalid near hover, where drag -> 0 and the bearing is noise.
    vel_bearing_rad: float = 0.0
    vel_valid: bool = False

    # Speed from drag magnitude, once the drag coefficient is fit from recordings.
    speed_est_mps: float = 0.0
    speed_conf: float = 0.0


@dataclass
class RaceObs:
    """The race-status packet: the only externally-sourced spatial fact VQ2 leaves us.

    active_gate_index advances when a gate is genuinely crossed, computed by the sim.
    Use it to close the loop BEHIND the aircraft -- vision looks forward, this confirms
    the crossing -- so no vision budget is spent watching a gate we are already
    committed to and can no longer influence.
    """

    active_gate_index: int = 0
    n_gates_total: int = 20
    t_since_gate_s: float = 0.0
    race_time_s: float = 0.0
    armed: bool = False

    # COLLISION is a contact SAMPLE, not a crash: a parked drone emits ~250/second.
    # Episodes (gap > 0.5 s), never message counts.
    t_since_collision_s: float = 1e3
    collision_episodes: int = 0


@dataclass
class Attention:
    """Where perception wants the camera, and why.

    Provided to the policy as context, and consumed directly by the yaw servo under
    YawMode.AUTO_ATTENTION. The handoff rule: track the current gate until
    range < R_COMMIT, then transfer to the next gate or the ribbon, because inside a few
    metres the aircraft is ballistic and nothing commanded changes the outcome.
    """

    kind: Attn = Attn.SEARCH
    target_dir_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    target_gate_index: int = -1


@dataclass
class Observation:
    t_s: float = 0.0
    dt_s: float = 0.02
    gates: list[GateObs] = field(default_factory=lambda: [GateObs() for _ in range(N_GATES)])
    ribbon: RibbonObs = field(default_factory=RibbonObs)
    own: SelfObs = field(default_factory=SelfObs)
    race: RaceObs = field(default_factory=RaceObs)
    attention: Attention = field(default_factory=Attention)

    def to_vector(self) -> np.ndarray:
        """Flat fixed-size encoding for a learned policy. Length == OBS_DIM.

        Angles that can wrap are encoded as (sin, cos); bounded ones stay raw. Invalid
        entries are zeroed with their flag cleared, so "not seen" is representable
        rather than being faked as "seen at the origin".
        """
        v: list[float] = []
        for g in self.gates:
            v += list(g.pos_body) + list(g.normal_body)
            v += [float(g.valid), float(g.normal_valid), g.confidence, min(g.staleness_s, 5.0)]
        v += list(self.ribbon.bearings_rad) + list(self.ribbon.elevs_rad)
        v += [float(self.ribbon.valid), self.ribbon.pixel_fraction, min(self.ribbon.staleness_s, 5.0)]
        o = self.own
        v += list(o.gyro) + list(o.accel)
        v += [o.roll_rad, o.pitch_rad, o.attitude_conf]
        v += [np.sin(o.vel_bearing_rad), np.cos(o.vel_bearing_rad), float(o.vel_valid)]
        v += [o.speed_est_mps, o.speed_conf]
        r = self.race
        v += [r.active_gate_index / max(r.n_gates_total, 1), min(r.t_since_gate_s, 30.0),
              min(r.t_since_collision_s, 5.0), float(r.armed)]
        a = self.attention
        v += list(a.target_dir_body)
        v += [float(a.kind == k) for k in (Attn.SEARCH, Attn.GATE_CURRENT, Attn.GATE_NEXT, Attn.RIBBON)]
        out = np.asarray(v, dtype=np.float32)
        assert out.size == OBS_DIM, f"OBS_DIM is {OBS_DIM}, encoder produced {out.size}"
        return out


@dataclass
class Action:
    """Acro output. Encoded into SET_ATTITUDE_TARGET by the link layer.

    Body rates are body-referenced BY CONSTRUCTION -- a roll rate is a roll rate
    regardless of where north is -- which is what lets the whole stack avoid ever
    estimating a heading.

    ATTITUDE_IGNORE stays SET. The sim does honour an attitude quaternion, but a
    quaternion encodes absolute yaw, and commanding absolute yaw is reading it -- a
    compass obtained through the actuator instead of the blocked telemetry stream.
    Ruled out of bounds 2026-07-30; not to be reopened for a smoother inner loop.
    """

    roll_rate: float = 0.0  # rad/s, canonical signs above
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0  # ignored under YawMode.AUTO_ATTENTION
    thrust: float = HOVER_THRUST  # 0..1
    yaw_mode: YawMode = YawMode.AUTO_ATTENTION

    def clipped(self) -> "Action":
        return Action(
            roll_rate=float(np.clip(self.roll_rate, -MAX_RATE_RPS, MAX_RATE_RPS)),
            pitch_rate=float(np.clip(self.pitch_rate, -MAX_RATE_RPS, MAX_RATE_RPS)),
            yaw_rate=float(np.clip(self.yaw_rate, -MAX_RATE_RPS, MAX_RATE_RPS)),
            thrust=float(np.clip(self.thrust, 0.0, 1.0)),
            yaw_mode=self.yaw_mode,
        )

    def to_vector(self) -> np.ndarray:
        return np.array([self.roll_rate, self.pitch_rate, self.yaw_rate, self.thrust], dtype=np.float32)

    @staticmethod
    def from_vector(v: np.ndarray, yaw_mode: YawMode = YawMode.AUTO_ATTENTION) -> "Action":
        return Action(float(v[0]), float(v[1]), float(v[2]), float(v[3]), yaw_mode).clipped()


class Policy:
    """Everything below the measurement vector. PID, MPC or learned -- the surrogate
    scores them identically, so the choice is an implementation detail of this class.
    """

    def reset(self) -> None:
        ...

    def __call__(self, obs: Observation) -> Action:
        raise NotImplementedError
