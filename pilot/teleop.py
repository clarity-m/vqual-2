"""
Manual teleop + recorder for the AI-GP simulator, VQ2 spec (VADR-TS-003 iss. 00.03).

Purpose: let a human fly the course by keyboard while every stream the simulator
still exposes is recorded to disk, so that later offline work has ground truth to
referee against.

VQ2 blocks ATTITUDE, LOCAL_POSITION_NED, ODOMETRY and gate info (spec 9.3), so
there is NO pose telemetry to record or display. What is left:

    HEARTBEAT               liveness / armed flag
    HIGHRES_IMU             accel + gyro (body frame == IMU frame, spec 3.8)
    TIMESYNC                sim clock offset
    COLLISION               1001 = gate, 1002 = environment
    ACTUATOR_OUTPUT_STATUS  motor outputs
    ENCAPSULATED_DATA id=1  race status -> active_gate_index is the ONLY
                            trustworthy gate-crossing signal available
    UDP :5600               640x360 JPEG camera stream @30 Hz

The recording is the point. A hand-flown lap gives a frame sequence stamped with
an independently-sourced gate counter; that pairing is what makes it possible to
measure a perception pipeline without grading it against itself.

CONTROL: body rates only (SET_ATTITUDE_TARGET), like an acro quad.

The sim ignores the coordinate_frame field on velocity setpoints - vx/vy are
always world axes - so velocity control silently requires a heading, and VQ2
blocks every message that carries one. Yaw cannot be recovered from the IMU
either (gravity is symmetric about it, and there is no magnetometer), so any
heading would be unbounded dead reckoning. Body rates need no heading at all:
a roll rate is a roll rate regardless of where north is. Velocity mode was
removed rather than left as a trap, so what is hand-flown here matches what the
autonomous pilot will command.

Usage
    python3 pilot/teleop.py                 # fly, record, live camera view
    python3 pilot/teleop.py --no-record     # fly only
    python3 pilot/teleop.py --no-view       # no cv2 window (lower jitter)
    python3 pilot/teleop.py --listen        # record only, send nothing

Constraints worth knowing before you run it:
  * Nothing else may hold UDP 14550. An autonomous pilot and this script cannot
    run at the same time.
  * Key state is read with a global hook, so keystrokes reach the simulator
    window too. Keep the sim focused for the video, but expect its own hotkeys
    to fire as well.
  * Spec 4.4 caps the command rate at <100 Hz; this sends at 50 Hz.
  * Spec 7: a hand-flown run is not a valid competitive attempt. Fly Training.
"""

import argparse
import collections
import json
import math
import os
import queue
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
from pymavlink import mavutil

try:
    import keyboard
except ImportError:
    sys.exit("teleop needs the 'keyboard' package:  pip install keyboard")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

MAVLINK_IP = "127.0.0.1"
MAVLINK_PORT = 14550
VISION_IP = "0.0.0.0"
VISION_PORT = 5600

CONTROL_HZ = 50           # spec 4.4: command rate < 100 Hz
HEARTBEAT_HZ = 2          # spec 4.4: minimum heartbeat rate
TIMESYNC_HZ = 10

MAVLINK_CMD_SIM_RESET = 31000

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID = 2

COLLISION_NAMES = {1001: "gate", 1002: "environment"}

# A drone resting on the ground emits COLLISION continuously (~200/s observed),
# so a message is a contact sample, not a crash. Consecutive contacts closer
# together than this are folded into one episode for display and events; every
# individual sample still lands in collisions.csv.
COLLISION_EPISODE_GAP = 0.5

# --- body rate limits (rad/s) and collective thrust ---
ACRO_ROLL = -2.5
ACRO_PITCH = 2.5
ACRO_YAW = 2.0
# Two different jobs, previously conflated in one constant:
#   THRUST_START  what the throttle sits at on arm / reset. 0.0 is the safe choice and
#                 matches a real transmitter, where you arm with the stick down.
#   THRUST_HOVER  the measured hover point. Used by the F10 snap and as the value
#                 auto-takeoff settles to. Setting this to 0 disables auto-takeoff,
#                 because "take off then settle to zero thrust" is just a drop.
THRUST_START = 0.0
THRUST_HOVER = 0.27      # measured in flight; adjust if it drifts up or down
THRUST_SLEW = 0.5         # per second, while a throttle key is held

BOOST = 2.0               # left ctrl
PRECISION = 0.35          # left alt

# --- align-to-velocity ("point the camera where I am going") --------------------------
# Thrust acts along body -z BY DEFINITION, so it contributes exactly zero to the
# body-frame horizontal accelerometer components. Whatever attitude we are at, ax and
# ay can only be measuring non-thrust forces - in flight, drag - and drag is
# antiparallel to velocity through the air:
#
#     (ax, ay) ~ -(vx, vy)_body        =>    bearing = atan2(-ay, -ax)
#
# So the direction of travel relative to the nose is an instantaneous algebraic read of
# one permitted sensor. No integration, no filter, no drift, no fusion. Hold the align
# key and we yaw until that bearing is zero.
#
# Two honest limits: it is AIRspeed, so a wind field would bias it (the sim appears to
# have none), and near hover drag goes to zero and the bearing becomes noise - hence the
# magnitude gate below. At a steady 20 deg nose-down, drag balances g*tan(20) = 3.6
# m/s^2, so in fast flight the signal is large and clean.
ALIGN_MIN_ACCEL = 0.30    # m/s^2 of horizontal specific force before we believe it
ALIGN_GAIN = 2.0          # rad/s of yaw per rad of bearing error
ALIGN_MAX_RATE = 2.0      # rad/s

# --- levelling assist (our own angle mode) --------------------------------------------
# The sim ships ACRO / ANGLE / ARCADE / GPS flight modes, but this build exposes only
# graphics and sound in its menu and nothing in Input.ini or the .sav names a mode, so
# ANGLE is unreachable. This is angle mode built on OUR side instead: the stick commands
# a bank ANGLE, an outer loop turns the angle error into a body RATE, and the rate goes
# out the same acro path as everything else.
#
# Three consequences worth being explicit about, because they are the reason this is
# preferable to asking the sim for angle mode even if the sim would give it:
#
#   * cmd.csv keeps recording real body-rate commands, so cmd -> response system ID
#     needs no reconstruction. Nothing about the data changes.
#   * No quaternion is transmitted and ATTITUDE_IGNORE stays set, so absolute yaw never
#     enters the protocol. Yaw is untouched here - it stays pure acro.
#   * It self-gates to the VQ1 rig. The outer loop needs truth attitude, which VQ2 does
#     not send, so under VQ2 it cannot engage even if the key is pressed.
#
# DATA-COLLECTION ONLY. This closes a loop around ground truth, which the raced pilot may
# never do. It lives in teleop, which is not the pilot, and interface.Policy cannot see it.
#
# Caveat for whoever fits a model to assisted flight: the commands are now generated from
# the state by this controller, so command and state are correlated through it. That
# biases an open-loop fit in a way that looks clean. Keep stick input live on top (it
# offsets the target, so it does excite the loop) and prefer the acro doublet sessions
# for identifying the rate loop itself.
LEVEL_MAX_ANGLE = math.radians(35.0)   # full stick = this much bank / pitch
LEVEL_GAIN = 4.0                       # rad/s of body rate per rad of angle error
LEVEL_MAX_RATE = 3.0                   # rad/s, clamp on the outer loop's output
LEVEL_MAX_AGE = 0.25                   # s; older truth than this and the assist drops out

# --- throttle behaviour under the levelling assist -------------------------------------
# Two separate fixes, both only active while levelling (plain acro is untouched):
#
# TILT COMPENSATION. Thrust acts along body -z, so only its vertical component holds the
# drone up: at a tilt of theta you need THRUST_HOVER / cos(theta) to stay level. At 35 deg
# bank that is 1.22x, so an uncompensated hover setting sags every time you turn - which is
# most of what makes a lap hard to hold altitude through. cos(theta) = cos(roll)*cos(pitch),
# both of which the assist already has. Clamped, because 1/cos runs away near 90 deg.
#
# RETURN TO HOVER. Plain teleop's throttle is a held value that stays where you leave it,
# which pairs badly with an angle-mode stick: you end up trimming throttle constantly. Here
# the throttle keys slew AWAY from the (compensated) hover point and it returns the moment
# you let go, so throttle becomes an offset rather than an absolute.
#
# Both need truth attitude, so like the rest of the assist they cannot engage under VQ2.
# Neither observes altitude - nothing here is an altitude hold, and letting go returns you
# to hover THRUST, not to a hover. Vertical speed is not measured, so drift remains yours
# to trim.
TILT_COMP_MAX = 1.6            # ceiling on the 1/cos(tilt) factor (~51 deg of tilt)
THRUST_STICK_EPS = 0.02        # |axis| below this counts as "let go"

# Sign that converts a physical (NED) body rate into this sim's mirrored command
# convention. Derived from the constants the stick path already uses rather than written
# as a bare -1, so the mirror stays recorded in exactly one place per axis:
#   roll  line is  roll  =  roll_ax * ACRO_ROLL      -> mirror carried by ACRO_ROLL's sign
#   pitch line is  pitch = -pitch_ax * ACRO_PITCH    -> mirror carried by the explicit minus
RATE_SIGN_ROLL = math.copysign(1.0, ACRO_ROLL)
RATE_SIGN_PITCH = math.copysign(1.0, -ACRO_PITCH)

# --- auto takeoff on arm --------------------------------------------------------------
# Open loop, because nothing in VQ2 observes altitude or vertical speed. It unsticks the
# pad and settles to hover thrust; it does NOT hold height, so expect to trim.
TAKEOFF_THRUST = 0.40     # UNVERIFIED - above THRUST_HOVER, below anything violent
TAKEOFF_S = 1.0

SLEW_PER_S = 12.0         # command ramp, in units of the per-axis limit per second

# --- heading (diagnostic only) --------------------------------------------------------
# Body rates are body-referenced by construction, so flying needs no heading at all.
# We still integrate zgyro and record it, because it is the only orientation signal
# VQ2 leaves and it is worth having alongside the frames when working on vision-based
# yaw later. It drifts without bound and nothing observes it - never steer by it.
YAW_GYRO_SIGN = 1.0       # UNVERIFIED. If yawing right makes heading fall, flip this.
HEADING_MAX_GAP_S = 0.5   # ignore an IMU dt longer than this (stall / reconnect)

# SET_ATTITUDE_TARGET.type_mask bit 16 is unused by the standard enum; the sim
# reads it as "body_rates really are rad/s" rather than the legacy scaling.
ATTITUDE_TARGET_TYPEMASK_DCL_BODY_RATES_RADS = 16
RATES_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
    | ATTITUDE_TARGET_TYPEMASK_DCL_BODY_RATES_RADS
)

# --------------------------------------------------------------------------------------
# Key bindings
#
# The keyboard hook is global and NOT exclusive: every key pressed here is also
# delivered to the simulator window. The simulator is a game and binds ordinary
# game keys - SPACE restarts the run (confirmed: "SpaceBar" is bound in the pak,
# and pressing it mid-flight resets), and ESC / R / number keys are the usual
# menu-and-restart suspects.
#
# So the binding rule is:
#   * continuous axes -> letter / arrow keys. If the sim echoes them the worst
#     case is a spectator camera twitch, which is cosmetic.
#   * one-shot commands -> function keys F5-F12, which games essentially never
#     bind, because these are the ones whose collisions destroy a flight.
#
# If a collision does turn up, rebind here; nothing else reads raw key names.
# --------------------------------------------------------------------------------------

KEYS_AXIS = {                     # axis: (positive key, negative key)
    "pitch":    ("w", "s"),       # w = nose down = forward
    "roll":     ("d", "a"),
    "throttle": ("up", "down"),
    "yaw":      ("e", "q"),
}
KEYS_MODIFIER = {"boost": "ctrl", "precision": "alt", "align": "c"}
KEYS_COMMAND = {                  # action: key
    "arm":       "f5",
    "disarm":    "f6",
    "zero_head": "f7",
    "quit":      "f8",
    "reset":     "f9",
    "hover":     "f10",
    "level":     "f11",
    "marker":    "f12",
}

KEYMAP = """
  sticks (body rates)              commands (function keys)
  W/S   pitch fwd / back           F5   arm
  A/D   roll right / left          F6   disarm
  Q/E   yaw left / right           F7   zero heading readout
  UP    throttle up                F8   quit (disarms)
  DOWN  throttle down              F9   sim reset
                                   F10  throttle back to hover
  C     align nose to velocity     F11  levelling assist on/off (VQ1 only)
  LCTRL boost   LALT precision      F12  drop a marker

  Throttle is a held value, not a stick position: it stays where you leave it.
  It sits at 0 on arm and after a reset, like a real transmitter. Arming runs a
  short takeoff burst and settles at hover thrust; throttle-down cancels it.
  Release the other keys and the rates go to zero, which is not a hover - the
  drone keeps whatever attitude it had.

  Hold C to yaw the nose onto the direction of travel, read from drag in the
  accelerometer. It is gated near hover, where there is no drag to read.

  F11 toggles the levelling assist: the sticks command a bank ANGLE instead of a
  rate, and releasing them returns to level instead of holding the attitude. It
  needs the VQ1 truth stream, so under VQ2 the key does nothing and says so. Body
  rates are still what goes on the wire, so recordings stay ordinary acro data.

  Keys reach the simulator too. Commands sit on F-keys because the sim binds
  SPACE to restart; axes sit on letters because an echoed letter is harmless.
"""


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def velocity_bearing(accel):
    """
    Bearing of travel relative to the nose, from the accelerometer alone.

    Returns (bearing_rad, magnitude) with bearing positive to the right, or
    (None, magnitude) when there is too little horizontal specific force to be
    meaningful - which is the case in a hover, where drag vanishes.

    See ALIGN_MIN_ACCEL above for why this works with no fusion: thrust is along
    body -z, so it cannot appear in ax/ay, leaving drag - which opposes velocity.
    """
    if accel is None:
        return None, 0.0
    ax, ay = accel[0], accel[1]
    mag = math.hypot(ax, ay)
    if mag < ALIGN_MIN_ACCEL:
        return None, mag
    return math.atan2(-ay, -ax), mag


class HeadingReadout:
    """Relative heading since the last zero, in radians, positive nose-right.

    Diagnostic only - nothing in the control path reads this. It exists so the
    recording carries an orientation trace and so the gyro sign can be checked
    by eye in flight.
    """

    def __init__(self):
        self._zero = 0.0

    def value(self, heading_gyro):
        return wrap_pi(heading_gyro - self._zero)

    def zero(self, heading_gyro):
        self._zero = heading_gyro


# --------------------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------------------

class Recorder:
    """Session directory of CSVs + raw JPEGs. All writes go through one thread so
    the UDP receive loops never block on disk."""

    def __init__(self, root, save_frames=True):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(root, stamp)
        self.frame_dir = os.path.join(self.dir, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.save_frames = save_frames

        self._q = queue.Queue(maxsize=4096)
        self._files = {}
        self._dropped = 0
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=False)
        self._thread.start()

        self._open("imu", "imu.csv",
                   "t_wall_ns,time_usec,xacc,yacc,zacc,xgyro,ygyro,zgyro,"
                   "abs_pressure,pressure_alt,temperature")
        self._open("race", "race.csv",
                   "t_wall_ns,sim_boot_time_ms,race_start_boot_time_ms,"
                   "race_finish_time_ns,active_gate_index,last_gate_race_time")
        self._open("cmd", "cmd.csv",
                   "t_wall_ns,armed,heading_rad,roll_rate,pitch_rate,yaw_rate,thrust")
        self._open("frames", "frames.csv",
                   "t_recv_wall_ns,frame_id,sim_time_ns,jpeg_size,file")
        self._open("actuators", "actuators.csv",
                   "t_wall_ns,time_usec,m0,m1,m2,m3")
        self._open("collisions", "collisions.csv",
                   "t_wall_ns,what,threat_level,impulse,episode")

        # --- ground-truth streams -----------------------------------------------
        # VQ2 blocks all three (spec 9.3), so under VQ2 these files stay empty and
        # cost nothing. The VQ1 simulator still sends them, and its physics and gate
        # dimensions are identical to VQ2 (all three spec revisions diffed) -- which
        # makes VQ1 a ground-truth rig for VQ2 development: fly the SAME rate-only
        # control path and get true pose back to referee against.
        #
        # Used to CHECK estimators offline, never to feed the pilot. The VQ2 pilot
        # consumes permitted streams only.
        #
        # Recorded unconditionally rather than behind a flag: one code path for both
        # sims, and if VQ2 ever leaks one of these it shows up in the data instead of
        # being silently discarded.
        self._open("attitude", "attitude.csv",
                   "t_wall_ns,time_boot_ms,roll,pitch,yaw,"
                   "rollspeed,pitchspeed,yawspeed")
        self._open("position", "position.csv",
                   "t_wall_ns,time_boot_ms,x,y,z,vx,vy,vz")
        self._open("odometry", "odometry.csv",
                   "t_wall_ns,time_usec,x,y,z,qw,qx,qy,qz,vx,vy,vz,"
                   "rollspeed,pitchspeed,yawspeed")

    def _open(self, key, name, header):
        f = open(os.path.join(self.dir, name), "w", encoding="utf-8", newline="\n")
        f.write(header + "\n")
        self._files[key] = f

    # -- producers (any thread) -------------------------------------------------
    def row(self, key, *fields):
        self._put(("row", key, fields))

    def event(self, kind, **payload):
        payload["kind"] = kind
        payload["t_wall_ns"] = time.time_ns()
        self._put(("event", None, payload))

    def frame(self, frame_id, sim_time_ns, jpeg_bytes):
        # Under --no-frames the JPEG is not written, so the `file` column must not
        # name one: it previously recorded a filename that never existed, which
        # reads as "the frame is there" to anything consuming the CSV later.
        # Timing and size still carry the frame-rate information worth keeping.
        name = "%08d.jpg" % frame_id if self.save_frames else ""
        self.row("frames", time.time_ns(), frame_id, sim_time_ns,
                 len(jpeg_bytes), name)
        if self.save_frames:
            self._put(("frame", name, bytes(jpeg_bytes)))

    def _put(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1

    # -- consumer ---------------------------------------------------------------
    def _writer_loop(self):
        events = open(os.path.join(self.dir, "events.jsonl"), "w",
                      encoding="utf-8", newline="\n")
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                if not self._running:
                    break
                continue
            kind, key, data = item
            if kind == "row":
                self._files[key].write(",".join(str(x) for x in data) + "\n")
            elif kind == "event":
                events.write(json.dumps(data) + "\n")
                events.flush()
            elif kind == "frame":
                with open(os.path.join(self.frame_dir, key), "wb") as fh:
                    fh.write(data)
        events.close()
        for f in self._files.values():
            f.close()

    def close(self):
        self._running = False
        self._thread.join(timeout=10.0)
        if self._dropped:
            print("WARNING: recorder dropped %d items (disk too slow)" % self._dropped)


class NullRecorder:
    dir = "(not recording)"

    def row(self, *a, **k):
        pass

    def event(self, *a, **k):
        pass

    def frame(self, *a, **k):
        pass

    def close(self):
        pass


# --------------------------------------------------------------------------------------
# MAVLink receive
# --------------------------------------------------------------------------------------

class Telemetry:
    """Receives everything VQ2 still publishes. Holds only the latest values;
    the durable copy is the recorder's."""

    def __init__(self, conn, rec):
        self.conn = conn
        self.rec = rec
        self.lock = threading.Lock()

        self.armed = False
        self.gyro = (0.0, 0.0, 0.0)
        self.accel = (0.0, 0.0, 0.0)
        self.active_gate = -1
        self.race_start_ms = -1
        self.race_time_s = 0.0
        self.last_gate_time = -1.0
        self.collisions = 0            # contact episodes
        self.contact_samples = 0       # raw COLLISION messages
        self._last_contact_t = 0.0
        self.last_collision = None
        self.clock_offset_ns = None
        self.imu_count = 0
        self.heading_gyro = 0.0        # integrated zgyro, radians, relative
        self.gyro_live = False         # has zgyro ever been non-zero?
        # True once any blocked-under-VQ2 stream arrives, i.e. we are on the VQ1
        # sim and this recording carries ground truth. Surfaced in the HUD so a
        # truth run is never mistaken for an ordinary one after the fact.
        self.truth_seen = False
        # (roll, pitch, t_wall) with the per-axis sign correction already applied.
        # None under VQ2, which is what gates the levelling assist off there.
        self.truth_att = None
        self._last_imu_us = None

        self._track_chunks = {}
        self._expected_chunks = {}

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=False)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)

    def _loop(self):
        while self._running:
            try:
                msg = self.conn.recv_match(blocking=False)
            except ConnectionResetError:
                print("\nWARNING: MAVLink port reset; telemetry stopped.")
                return
            if msg is None:
                time.sleep(0.001)
                continue

            t = msg.get_type()
            if t == "BAD_DATA":
                continue

            if t == "HEARTBEAT":
                armed = bool(msg.base_mode
                             & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                with self.lock:
                    if armed != self.armed:
                        self.rec.event("armed_changed", armed=armed)
                    self.armed = armed

            elif t == "HIGHRES_IMU":
                self._on_imu(msg)

            elif t == "TIMESYNC":
                # Sim echoes our request with its own clock in tc1.
                if msg.tc1 != 0 and msg.ts1 != 0:
                    with self.lock:
                        self.clock_offset_ns = msg.tc1 - msg.ts1

            elif t == "ACTUATOR_OUTPUT_STATUS":
                self.rec.row("actuators", time.time_ns(), msg.time_usec,
                             msg.actuator[0], msg.actuator[1],
                             msg.actuator[2], msg.actuator[3])

            elif t == "COLLISION":
                self._on_collision(msg)

            # --- ground truth: present under VQ1, blocked under VQ2 ---
            elif t == "ATTITUDE":
                self.truth_seen = True
                self.rec.row("attitude", time.time_ns(), msg.time_boot_ms,
                             msg.roll, msg.pitch, msg.yaw,
                             msg.rollspeed, msg.pitchspeed, msg.yawspeed)
                # Corrected truth for the levelling assist. The VQ1 truth streams are
                # each wrong on a DIFFERENT axis (NOTES.md, refereed against gravity on
                # a parked drone): roll is right in ATTITUDE, pitch is right in ODOMETRY
                # and inverted here. Yaw is not used - the assist never touches yaw.
                with self.lock:
                    self.truth_att = (msg.roll, -msg.pitch, time.time())

            elif t == "LOCAL_POSITION_NED":
                self.truth_seen = True
                self.rec.row("position", time.time_ns(), msg.time_boot_ms,
                             msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz)

            elif t == "ODOMETRY":
                self.truth_seen = True
                q = msg.q  # w, x, y, z
                self.rec.row("odometry", time.time_ns(), msg.time_usec,
                             msg.x, msg.y, msg.z,
                             q[0], q[1], q[2], q[3],
                             msg.vx, msg.vy, msg.vz,
                             msg.rollspeed, msg.pitchspeed, msg.yawspeed)

            elif t == "ENCAPSULATED_DATA":
                self._on_encapsulated(msg)

            elif t == "DATA_TRANSMISSION_HANDSHAKE":
                self._track_chunks[msg.width] = {}
                self._expected_chunks[msg.width] = msg.packets

    def _on_imu(self, msg):
        # Integrate zgyro into a relative heading. This is the only orientation
        # signal VQ2 leaves us; it drifts and nothing here can correct it.
        with self.lock:
            self.accel = (msg.xacc, msg.yacc, msg.zacc)
            self.gyro = (msg.xgyro, msg.ygyro, msg.zgyro)
            self.imu_count += 1
            if abs(msg.zgyro) > 1e-3:
                self.gyro_live = True
            last = self._last_imu_us
            self._last_imu_us = msg.time_usec
            if last is not None and msg.time_usec > last:
                dt = (msg.time_usec - last) / 1e6
                if dt <= HEADING_MAX_GAP_S:
                    self.heading_gyro = wrap_pi(
                        self.heading_gyro + YAW_GYRO_SIGN * msg.zgyro * dt)
        self.rec.row("imu", time.time_ns(), msg.time_usec,
                     msg.xacc, msg.yacc, msg.zacc,
                     msg.xgyro, msg.ygyro, msg.zgyro,
                     msg.abs_pressure, msg.pressure_alt, msg.temperature)

    def _on_collision(self, msg):
        # horizontal_minimum_delta carries impulse magnitude (kg m/s), not a
        # distance, despite the field name.
        name = COLLISION_NAMES.get(msg.id, "id=%s" % msg.id)
        now = time.time()
        with self.lock:
            self.contact_samples += 1
            new_episode = (now - self._last_contact_t) > COLLISION_EPISODE_GAP
            if new_episode:
                self.collisions += 1
            self._last_contact_t = now
            self.last_collision = (now, name, msg.threat_level)
            episode = self.collisions
        self.rec.row("collisions", time.time_ns(), name, msg.threat_level,
                     msg.horizontal_minimum_delta, episode)
        if new_episode:
            self.rec.event("collision_episode", episode=episode, what=name,
                           threat_level=msg.threat_level,
                           impulse=msg.horizontal_minimum_delta)

    def _on_encapsulated(self, msg):
        payload = bytes(msg.data)
        if not payload:
            return
        if payload[0] == ENCAPSULATED_RACE_STATUS_MSG_ID:
            self._on_race_status(payload)
        elif payload[0] == ENCAPSULATED_TRACK_INFO_MSG_ID:
            # Track info still arrives but its gate fields are nulled under VQ2
            # (spec 9.3). Reassembled and dropped; do not build anything on it.
            self._reassemble_track(msg, payload)

    def _on_race_status(self, payload):
        (_id, sim_boot_ms, race_start_ms, race_finish_ns,
         active_gate, last_gate_time) = struct.unpack_from("<BQqqIq", payload)
        now = time.time_ns()
        self.rec.row("race", now, sim_boot_ms, race_start_ms, race_finish_ns,
                     active_gate, last_gate_time)
        with self.lock:
            prev_gate = self.active_gate
            self.active_gate = active_gate
            self.race_start_ms = race_start_ms
            self.last_gate_time = last_gate_time
            if race_start_ms is not None and race_start_ms >= 0:
                self.race_time_s = max(0.0, (sim_boot_ms - race_start_ms) / 1000.0)
            else:
                self.race_time_s = 0.0
        # The gate counter is sourced independently of anything we compute, so a
        # change here is the one gate-crossing fact worth trusting.
        if active_gate != prev_gate:
            self.rec.event("gate_advance", from_gate=prev_gate, to_gate=active_gate,
                           sim_boot_time_ms=sim_boot_ms,
                           last_gate_race_time=last_gate_time)
            if prev_gate >= 0:
                print("\n[gate] %d -> %d  (t=%.2fs)"
                      % (prev_gate, active_gate, self.race_time_s))
        if race_finish_ns is not None and race_finish_ns > 0:
            self.rec.event("race_finished", race_finish_time_ns=race_finish_ns)

    def _reassemble_track(self, msg, payload):
        _id, transfer_id = struct.unpack_from("<BH", payload)
        if transfer_id not in self._expected_chunks:
            return
        self._track_chunks[transfer_id][msg.seqnr] = payload[3:]
        if len(self._track_chunks[transfer_id]) == self._expected_chunks[transfer_id]:
            chunks = self._track_chunks.pop(transfer_id)
            self._expected_chunks.pop(transfer_id)
            full = b"".join(chunks[i] for i in range(len(chunks)))
            num_gates, = struct.unpack_from("<H", full)
            self.rec.event("track_info", num_gates=num_gates,
                           note="geometry nulled under VQ2 telemetry restrictions")

    def snapshot(self):
        with self.lock:
            return dict(armed=self.armed, gyro=self.gyro, accel=self.accel,
                        active_gate=self.active_gate, race_time_s=self.race_time_s,
                        collisions=self.collisions, contacts=self.contact_samples,
                        last_collision=self.last_collision,
                        imu_count=self.imu_count,
                        heading_gyro=self.heading_gyro, gyro_live=self.gyro_live,
                        truth_seen=self.truth_seen, truth_att=self.truth_att)


# --------------------------------------------------------------------------------------
# Vision receive
# --------------------------------------------------------------------------------------

class VisionRX:
    """Reassembles the 640x360 JPEG stream (spec 4.6). Hands the newest decoded
    frame to the main thread; cv2 windows are the main thread's business."""

    HEADER_FMT = "<IHHIIQ"

    def __init__(self, rec, decode=True):
        self.rec = rec
        self.decode = decode
        self.lock = threading.Lock()
        self.latest = None            # (frame_id, sim_time_ns, BGR image)
        self.frames_done = 0
        self.frames_dropped = 0
        self.duplicate_packets = 0
        self._last_fps_t = time.time()
        self._last_fps_n = 0
        self.fps = 0.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.sock.settimeout(0.5)
        self.sock.bind((VISION_IP, VISION_PORT))

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=False)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self.sock.close()

    def _loop(self):
        header_sz = struct.calcsize(self.HEADER_FMT)
        pending = {}
        # The sim delivers each chunk more than once (observed: every frame
        # arriving twice, which reads as 60 fps against the spec's 30 Hz).
        # Without this, duplicates refill a slot after it completes and the
        # frame is recorded twice.
        done = collections.deque(maxlen=512)
        done_set = set()
        while self._running:
            try:
                packet, _addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if len(packet) < header_sz:
                continue

            (frame_id, chunk_id, total_chunks, jpeg_size,
             payload_size, sim_time_ns) = struct.unpack(
                self.HEADER_FMT, packet[:header_sz])
            payload = packet[header_sz:header_sz + payload_size]

            if frame_id in done_set:
                self.duplicate_packets += 1
                continue

            slot = pending.setdefault(frame_id, {"chunks": {}, "total": total_chunks,
                                                 "size": jpeg_size,
                                                 "time": sim_time_ns})
            slot["chunks"][chunk_id] = payload

            if len(slot["chunks"]) < slot["total"]:
                continue

            pending.pop(frame_id)
            if len(done) == done.maxlen:
                done_set.discard(done[0])
            done.append(frame_id)
            done_set.add(frame_id)

            jpeg = b"".join(slot["chunks"][i] for i in range(slot["total"]))
            if len(jpeg) != slot["size"]:
                self.frames_dropped += 1
                continue

            self.rec.frame(frame_id, slot["time"], jpeg)
            self._tick_fps()

            if self.decode:
                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is not None:
                    with self.lock:
                        self.latest = (frame_id, slot["time"], img)

            # Any frame older than the one we just completed lost packets.
            for stale in [f for f in pending if f < frame_id - 4]:
                pending.pop(stale)
                self.frames_dropped += 1

    def _tick_fps(self):
        self.frames_done += 1
        now = time.time()
        if now - self._last_fps_t >= 1.0:
            self.fps = (self.frames_done - self._last_fps_n) / (now - self._last_fps_t)
            self._last_fps_t = now
            self._last_fps_n = self.frames_done

    def take(self):
        with self.lock:
            return self.latest


# --------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------

class Sticks:
    """Keyboard -> four axes, slew limited so a keypress is a ramp rather than a
    step. Axes are normalized -1..1; the flight mode decides what they mean."""

    def __init__(self):
        self.axes = [0.0, 0.0, 0.0, 0.0]   # pitch, roll, throttle, yaw
        self.align = False                 # align-to-velocity key held?
        self._prev_edges = {a: False for a in KEYS_COMMAND}

    @staticmethod
    def _held(name):
        try:
            return keyboard.is_pressed(name)
        except Exception:
            # Unknown key name on this layout, or a transient hook error. A
            # control that cannot be read must read as "not pressed".
            return False

    def _axis(self, pos, neg):
        return (1.0 if self._held(pos) else 0.0) - (1.0 if self._held(neg) else 0.0)

    def poll(self, dt):
        """Returns (axes, scale, actions_triggered)."""
        target = [
            self._axis(*KEYS_AXIS["pitch"]),
            self._axis(*KEYS_AXIS["roll"]),
            self._axis(*KEYS_AXIS["throttle"]),
            self._axis(*KEYS_AXIS["yaw"]),
        ]
        step = SLEW_PER_S * dt
        for i, want in enumerate(target):
            delta = want - self.axes[i]
            if delta > step:
                delta = step
            elif delta < -step:
                delta = -step
            self.axes[i] += delta

        scale = 1.0
        if self._held(KEYS_MODIFIER["boost"]):
            scale *= BOOST
        if self._held(KEYS_MODIFIER["precision"]):
            scale *= PRECISION
        self.align = self._held(KEYS_MODIFIER["align"])

        actions = []
        for action, key in KEYS_COMMAND.items():
            now = self._held(key)
            if now and not self._prev_edges[action]:
                actions.append(action)
            self._prev_edges[action] = now
        return list(self.axes), scale, actions


# --------------------------------------------------------------------------------------
# Command output
# --------------------------------------------------------------------------------------

class Pilot:
    def __init__(self, conn, rec, boot_ms, listen_only=False,
                 hover=THRUST_HOVER):
        self.conn = conn
        self.rec = rec
        self.boot_ms = boot_ms
        self.listen_only = listen_only
        self.thrust = THRUST_START
        self.last_cmd = (0.0, 0.0, 0.0, 0.0)
        self.takeoff_until = 0.0
        self.aligning = False
        self.align_bearing = None
        self.align_mag = 0.0
        self.auto_takeoff = True
        self.levelling = False
        self.level_err = (0.0, 0.0)
        # The hover point is a MEASURED quantity, so it lives on the instance and is
        # settable with --hover. THRUST_HOVER is only its default. Everything that
        # needs a hover thrust reads self.hover; nothing re-states the number.
        self.hover = hover
        self.hover_ref = hover

    # -- one-shot commands ------------------------------------------------------
    def arm(self, armed=True):
        if self.listen_only:
            return
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1 if armed else 0, 0, 0, 0, 0, 0, 0)
        self.rec.event("arm_command", armed=armed)
        if armed and self.auto_takeoff and self.hover > 0.0:
            self.takeoff_until = time.time() + TAKEOFF_S
            self.rec.event("auto_takeoff", thrust=TAKEOFF_THRUST,
                           seconds=TAKEOFF_S)
        else:
            self.takeoff_until = 0.0

    def sim_reset(self):
        if self.listen_only:
            return
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0)
        self.thrust = THRUST_START
        self.takeoff_until = 0.0
        self.rec.event("sim_reset")

    def heartbeat(self):
        self.conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def timesync_request(self):
        # Argument order follows the vendor example: (tc1=client time, ts1=0).
        self.conn.mav.timesync_send(time.time_ns(), 0)

    # -- streamed setpoints -----------------------------------------------------
    def send(self, axes, scale, dt, armed, heading=0.0, align=False, accel=None,
             level=False, truth_att=None):
        """Body rates + collective thrust. No frame, no heading, no rotation:
        a roll rate is a roll rate regardless of where north is, which is the
        whole reason this is the only control path teleop offers."""
        if self.listen_only:
            return
        pitch_ax, roll_ax, thr_ax, yaw_ax = axes

        roll = roll_ax * ACRO_ROLL * scale
        pitch = -pitch_ax * ACRO_PITCH * scale   # NED: nose down is negative
        yaw = yaw_ax * ACRO_YAW * scale

        # Levelling assist: the stick becomes an angle demand on roll and pitch, and an
        # outer P loop turns the angle error into the body rate we were going to send
        # anyway. Yaw and throttle are untouched - yaw stays acro because levelling it
        # would mean holding a heading, and holding a heading means knowing one.
        self.levelling = False
        if level and truth_att is not None:
            t_roll, t_pitch, t_stamp = truth_att
            if time.time() - t_stamp <= LEVEL_MAX_AGE:
                self.levelling = True
                # Stick deflection is a target ANGLE now, not a rate. `scale` still
                # applies, so precision/boost trim how much bank a full deflection asks
                # for, which is the same thing they meant before.
                want_roll = roll_ax * LEVEL_MAX_ANGLE * scale
                want_pitch = -pitch_ax * LEVEL_MAX_ANGLE * scale
                self.level_err = (want_roll - t_roll, want_pitch - t_pitch)
                p_roll = max(-LEVEL_MAX_RATE, min(LEVEL_MAX_RATE,
                                                  LEVEL_GAIN * self.level_err[0]))
                p_pitch = max(-LEVEL_MAX_RATE, min(LEVEL_MAX_RATE,
                                                   LEVEL_GAIN * self.level_err[1]))
                # p_* are physical NED rates; the RATE_SIGN_* constants carry them into
                # the sim's mirrored command convention.
                roll = p_roll * RATE_SIGN_ROLL
                pitch = p_pitch * RATE_SIGN_PITCH

                # Thrust needed to hold altitude at this tilt. Uses the ACHIEVED
                # attitude, not the demanded one, so it compensates the bank you are
                # actually at rather than the one you asked for.
                ctilt = math.cos(t_roll) * math.cos(t_pitch)
                comp = TILT_COMP_MAX if ctilt <= 1.0 / TILT_COMP_MAX else 1.0 / ctilt
                self.hover_ref = min(1.0, self.hover * comp)

        # Align to velocity: yaw until the direction of travel is under the nose.
        # The bearing is physical (body frame, from the accelerometer), so the
        # desired yaw RATE is physical too - dividing by ACRO_YAW converts it into
        # the same units the stick uses and inherits that constant's sign, keeping
        # the sim's yaw convention in exactly one place.
        self.align_bearing, self.align_mag = velocity_bearing(accel)
        self.aligning = bool(align and self.align_bearing is not None)
        if self.aligning:
            # rate is physical: positive means "yaw the nose right".
            rate = max(-ALIGN_MAX_RATE,
                       min(ALIGN_MAX_RATE, ALIGN_GAIN * self.align_bearing))
            yaw = rate * math.copysign(1.0, ACRO_YAW)

        # Auto takeoff: hold a raised thrust to unstick the pad, then settle to
        # hover. Open loop - nothing here observes altitude. Any throttle-down
        # input cancels it, so it can never fight the pilot.
        if self.takeoff_until:
            if thr_ax < -0.05 or time.time() >= self.takeoff_until:
                self.takeoff_until = 0.0
                self.thrust = self.hover
            else:
                self.thrust = TAKEOFF_THRUST
        elif self.levelling and abs(thr_ax) < THRUST_STICK_EPS:
            # Let go under the assist: snap straight to the tilt-compensated hover point.
            #
            # This used to ease back with a time constant. Flight-tested 2026-07-31: the
            # step is not felt, because thrust reaches velocity through mass and drag,
            # which is itself a first-order lag - the airframe already supplies the
            # smoothing. A filter here would put a second lag in series with it.
            #
            # It is also the better choice for the data. A step is what excites a plant;
            # a pre-smoothed command shares its shape with the response, which is exactly
            # what makes a command->thrust lag hard to bracket (plantfit.py failed on that
            # and on thrust R^2 ~ 0.05). Sharp edges in cmd.csv are worth keeping.
            #
            # Nothing here observes altitude - this returns to hover THRUST, not to hover.
            self.thrust = min(1.0, max(0.0, self.hover_ref))
        else:
            # Throttle is a held value, not a stick deflection - it integrates
            # while a key is down and stays put when released.
            self.thrust = min(1.0, max(0.0,
                                       self.thrust + thr_ax * THRUST_SLEW * dt))

        self._send_rates(roll, pitch, yaw, self.thrust)
        self.last_cmd = (roll, pitch, yaw, self.thrust)
        self.rec.row("cmd", time.time_ns(), int(armed), "%.4f" % heading,
                     *self.last_cmd)

    def _send_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.conn.mav.set_attitude_target_send(
            int(time.time() * 1000) - self.boot_ms,
            self.conn.target_system, self.conn.target_component,
            RATES_MASK,
            [1.0, 0.0, 0.0, 0.0],       # ignored
            roll_rate, pitch_rate, yaw_rate, thrust)

    def stop(self):
        """Zero rates at zero thrust. This is a cut, not a hover - on exit we
        want the drone inert, and zero rates alone would leave it coasting at
        whatever attitude and thrust it had."""
        if self.listen_only:
            return
        self.thrust = 0.0
        self._send_rates(0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------------------

WINDOW = "AI-GP teleop"


def draw_hud(img, pilot, tel, vision, marker_count, heading_now):
    view = img.copy()
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.rectangle(view, (0, h - 20), (w, h), (0, 0, 0), -1)

    a, b, c, d = pilot.last_cmd
    line1 = "RATES r%+.2f p%+.2f y%+.2f  THR %.2f%s%s  hdg %+4.0f%s" % (
        a, b, c, d,
        (" LEVEL hov%.2f" % pilot.hover_ref) if pilot.levelling else "",
        " TAKEOFF" if pilot.takeoff_until else "",
        math.degrees(heading_now),
        "" if tel["gyro_live"] else " (no gyro!)")
    if pilot.align_bearing is not None:
        line1 += "  vel %+4.0f%s" % (math.degrees(pilot.align_bearing),
                                     " ALIGN" if pilot.aligning else "")
    else:
        line1 += "  vel --"
    if tel.get("truth_seen"):
        line1 += "  [TRUTH]"   # VQ1 sim: pose telemetry is arriving and being recorded

    line2 = "%s  gate %s  t %.1fs  contacts %d  %.0f fps" % (
        "ARMED" if tel["armed"] else "disarmed",
        tel["active_gate"] if tel["active_gate"] >= 0 else "-",
        tel["race_time_s"], tel["collisions"], vision.fps)

    colour = (0, 200, 255) if tel["armed"] else (160, 160, 160)
    cv2.putText(view, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                cv2.LINE_AA)
    cv2.putText(view, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                cv2.LINE_AA)

    hit = tel["last_collision"]
    if hit and time.time() - hit[0] < 1.5:
        cv2.putText(view, "COLLISION: %s" % hit[1], (8, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if marker_count:
        cv2.putText(view, "markers %d" % marker_count, (w - 130, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return view


def console_line(pilot, tel, vision, heading_now):
    a, b, c, d = pilot.last_cmd
    vel = ("%+4.0f" % math.degrees(pilot.align_bearing)
           if pilot.align_bearing is not None else "  --")
    sys.stdout.write(
        "\r%-8s r%+.2f p%+.2f y%+.2f thr%.2f%-8s hdg%+4.0f vel%s%-6s gate %-3s "
        "t%6.1f contact %-3d cam %4.1f fps imu %-6d " % (
            "ARMED" if tel["armed"] else "disarmed",
            a, b, c, d,
            " TAKEOFF" if pilot.takeoff_until else "",
            math.degrees(heading_now), vel,
            " ALIGN" if pilot.aligning else "",
            tel["active_gate"] if tel["active_gate"] >= 0 else "-",
            tel["race_time_s"], tel["collisions"], vision.fps, tel["imu_count"]))
    sys.stdout.flush()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="AI-GP manual teleop and recorder")
    ap.add_argument("--ip", default=MAVLINK_IP)
    ap.add_argument("--port", type=int, default=MAVLINK_PORT)
    ap.add_argument("--sessions", default=os.path.join(os.path.dirname(__file__),
                                                       "sessions"))
    ap.add_argument("--no-record", action="store_true",
                    help="fly without writing anything to disk")
    ap.add_argument("--no-frames", action="store_true",
                    help="record telemetry but not the JPEG stream")
    ap.add_argument("--no-view", action="store_true",
                    help="skip the cv2 window (less jitter in the control loop)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="exit cleanly after N seconds (0 = until the quit key)")
    ap.add_argument("--hover", type=float, default=THRUST_HOVER,
                    help="measured hover thrust 0..1 (default %(default)s). F10 snaps "
                         "to it, auto-takeoff settles to it, and the levelling assist "
                         "returns to it tilt-compensated.")
    ap.add_argument("--listen", action="store_true",
                    help="record only: send no setpoints, no arm/disarm, no reset. "
                         "Safe to attach to a flight already in progress.")
    args = ap.parse_args()

    rec = NullRecorder() if args.no_record else Recorder(args.sessions,
                                                         save_frames=not args.no_frames)

    print("Binding MAVLink on udpin:%s:%d ..." % (args.ip, args.port))
    try:
        conn = mavutil.mavlink_connection("udpin:%s:%d" % (args.ip, args.port))
    except OSError as e:
        sys.exit("could not bind UDP %d (%s). Another client is probably holding "
                 "it - only one thing can talk to the sim at a time." % (args.port, e))

    print("Waiting for heartbeat (start the sim and enter a Training flight) ...")
    conn.wait_heartbeat()
    print("Connected to system %d, component %d"
          % (conn.target_system, conn.target_component))

    boot_ms = int(time.time() * 1000)
    tel = Telemetry(conn, rec)
    vision = VisionRX(rec, decode=not args.no_view)
    pilot = Pilot(conn, rec, boot_ms, listen_only=args.listen,
                  hover=args.hover)
    sticks = Sticks()
    heading = HeadingReadout()

    rec.event("session_start", control="body_rates", ip=args.ip, port=args.port,
              listen_only=args.listen)
    if args.listen:
        print("\nLISTEN ONLY - recording; no commands will be sent.\n")
    else:
        print(KEYMAP)
    print("recording to: %s" % rec.dir)
    if not args.listen:
        print("Body-rate control only. This flies like an acro quad: the drone holds\n"
              "whatever attitude you leave it in, so it will NOT self-level and\n"
              "releasing the keys is not a hover. Throttle starts at %.2f.\n"
              % pilot.hover)
        print("Press %s to arm, %s to cut and quit.\n"
              % (KEYS_COMMAND["arm"].upper(), KEYS_COMMAND["quit"].upper()))

    if not args.no_view:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 540)

    period = 1.0 / CONTROL_HZ
    next_tick = time.perf_counter()
    last_hb = 0.0
    last_ts = 0.0
    last_console = 0.0
    last_loop = time.perf_counter()
    markers = 0
    level_on = False
    running = True
    deadline = time.perf_counter() + args.duration if args.duration else None

    try:
        while running:
            now = time.perf_counter()
            if deadline and now >= deadline:
                break
            dt = min(0.1, now - last_loop)
            last_loop = now

            axes, scale, actions = sticks.poll(dt)
            snap = tel.snapshot()

            for action in actions:
                if action == "arm":
                    pilot.arm(True)
                    print("\n[arm]")
                elif action == "disarm":
                    pilot.arm(False)
                    print("\n[disarm]")
                elif action == "reset":
                    pilot.sim_reset()
                    # The reset re-places the drone, so any integrated heading
                    # is now measuring from a pose that no longer exists.
                    heading.zero(snap["heading_gyro"])
                    print("\n[sim reset] (heading zeroed)")
                elif action == "hover":
                    pilot.thrust = pilot.hover
                    rec.event("thrust_to_hover", thrust=pilot.hover)
                    print("\n[throttle] %.2f" % pilot.hover)
                elif action == "zero_head":
                    heading.zero(snap["heading_gyro"])
                    rec.event("heading_zeroed")
                    print("\n[heading zeroed]")
                elif action == "level":
                    if snap["truth_att"] is None:
                        print("\n[level] unavailable: no ATTITUDE stream "
                              "(VQ2 blocks it - this needs the VQ1 build)")
                    else:
                        level_on = not level_on
                        rec.event("level_assist", on=level_on)
                        print("\n[level] %s" % ("ON" if level_on else "OFF"))
                elif action == "marker":
                    markers += 1
                    rec.event("marker", index=markers,
                              active_gate=snap["active_gate"],
                              race_time_s=snap["race_time_s"])
                    print("\n[marker %d] gate=%s t=%.2f"
                          % (markers, snap["active_gate"], snap["race_time_s"]))
                elif action == "quit":
                    running = False

            head_now = heading.value(snap["heading_gyro"])
            pilot.send(axes, scale, dt, snap["armed"], head_now,
                       align=sticks.align, accel=snap["accel"],
                       level=level_on, truth_att=snap["truth_att"])

            wall = time.time()
            if args.listen:
                pass          # truly silent: not even heartbeat or timesync
            elif wall - last_hb >= 1.0 / HEARTBEAT_HZ:
                pilot.heartbeat()
                last_hb = wall
            if not args.listen and wall - last_ts >= 1.0 / TIMESYNC_HZ:
                pilot.timesync_request()
                last_ts = wall

            if not args.no_view:
                latest = vision.take()
                if latest is not None:
                    cv2.imshow(WINDOW, draw_hud(latest[2], pilot, snap, vision,
                                                markers, head_now))
                if cv2.waitKey(1) & 0xFF == 27:
                    running = False

            if wall - last_console >= 0.25:
                console_line(pilot, snap, vision, head_now)
                last_console = wall

            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()   # fell behind; resync

    except KeyboardInterrupt:
        pass
    finally:
        if args.listen:
            print("\nStopping: flushing recording.")
        else:
            print("\nStopping: neutral setpoint, disarm, flush.")
            for _ in range(10):
                pilot.stop()
                time.sleep(0.02)
            pilot.arm(False)
            time.sleep(0.1)
        rec.event("session_end", frames=vision.frames_done,
                  frames_dropped=vision.frames_dropped,
                  duplicate_packets=vision.duplicate_packets,
                  contact_samples=tel.contact_samples,
                  contact_episodes=tel.collisions)
        vision.stop()
        tel.stop()
        if not args.no_view:
            cv2.destroyAllWindows()
        rec.close()
        print("frames received %d (incomplete %d, duplicate packets %d)"
              % (vision.frames_done, vision.frames_dropped,
                 vision.duplicate_packets))
        print("contacts: %d episodes from %d COLLISION messages"
              % (tel.collisions, tel.contact_samples))
        print("session: %s" % rec.dir)


if __name__ == "__main__":
    main()
