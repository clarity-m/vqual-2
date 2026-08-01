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

THROTTLE: an offset, not an absolute. Holding UP/DOWN slews thrust exactly as it
always has; releasing them snaps thrust to the MEASURED hover point (THRUST_HOVER,
provenance at the constant). ACRO is the only reachable flight mode, so the thrust
channel is fully manual and there is exactly one correct setting for it - snapping
to that setting is what makes a long lap flyable by hand. It is NOT an altitude
hold and deliberately not a loop of any kind: VQ2 blocks position and velocity and
integrated accelerometer height drifts, so there is no altitude measurement to
close on. --no-thrust-snap restores the old hold-where-you-leave-it throttle.

The sim ignores the coordinate_frame field on velocity setpoints - vx/vy are
always world axes - so velocity control silently requires a heading, and VQ2
blocks every message that carries one. Yaw cannot be recovered from the IMU
either (gravity is symmetric about it, and there is no magnetometer), so any
heading would be unbounded dead reckoning. Body rates need no heading at all:
a roll rate is a roll rate regardless of where north is. Velocity mode was
removed rather than left as a trap, so what is hand-flown here matches what the
autonomous pilot will command.

INPUT SHAPING: a keyboard axis is 0 or 1, so without shaping every input is full
deflection and a gentle correction can only be spelled as a tap. --slow-lap adds
expo, a hard rate cap and a softer stick ramp, which is what a map-building lap
wants; the default --shaping full is arithmetically identical to every session
recorded before shaping existed. Whichever is used is written to events.jsonl,
because cmd.csv feeds a plant fit and a run flown under different shaping is a
different experiment.

Usage
    python3 pilot/teleop.py                 # fly, record, live camera view
    python3 pilot/teleop.py --slow-lap      # gentler handling for a mapping lap
    python3 pilot/teleop.py --no-record     # fly only
    python3 pilot/teleop.py --no-view       # no cv2 window (lower jitter)
    python3 pilot/teleop.py --listen        # record only, send nothing
    python3 pilot/teleop.py --no-thrust-snap  # old throttle: holds where you leave it

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
#   THRUST_HOVER  the measured hover point. Used by the F10 snap, by the release snap
#                 below, and as the value auto-takeoff settles to. Setting this to 0
#                 disables auto-takeoff, because "take off then settle to zero thrust"
#                 is just a drop.
THRUST_START = 0.0

# MEASURED, not assumed. An IMU sample is a hover sample when |a| is within 0.35 of g
# (not accelerating) and gravity lies along body -z (not banked); at that instant thrust
# is exactly balancing weight, so the thrust commanded alongside it IS the hover point.
#
#   Claire, session 20260731-222724: 221 such samples -> 0.266 (p10 0.20, p90 0.285).
#
# Re-measured independently here across all 24 sessions that carry both imu.csv and a
# cmd.csv with a thrust column (pilot/hovercheck.py), adding two filters the first pass
# did not have -- armed, and thrust >= 0.05, because a drone SITTING ON THE PAD also
# reads steady and level at whatever thrust the stick is at, and that is what the 0.20
# p10 above is: pad samples, not flight. Pooled n=17329, median 0.265; the four
# longest sessions land on 0.265, 0.265, 0.267, 0.272. So 0.266 is confirmed rather
# than replaced, and the honest spread is ~0.265-0.272, not 0.20-0.285.
THRUST_HOVER = 0.266
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

# --- roll/pitch from the IMU, so the assist works under VQ2 (--imu-level) --------------
# The assist above needs an attitude, and VQ2 sends none. But roll and pitch ARE
# observable under VQ2: gravity is a vector the accelerometer can see, and CONVENTIONS.md
# lists gravity-near-stationary as a valid referee for exactly those two axes. NOTES.md
# already ruled this in bounds when it rejected absolute yaw -- "Roll/pitch self-levelling
# -- no [it does not give information no permitted stream provides]; gravity is observable;
# quality gain, not capability gain". Yaw stays out, and is untouched here as everywhere.
#
# The accelerometer alone is not enough, and --imu-tilt-comp is the evidence: gravity is
# only readable when not accelerating, which rejects 40-78% of frames with gaps up to 5 s,
# and rejects them preferentially when banked. So this is a complementary filter -- the
# gyro carries attitude through the manoeuvre, the accelerometer trims the drift out
# whenever it is trustworthy. Textbook, and the same structure a real flight controller
# uses.
#
# MEASURED against VQ1 truth (roll = ATTITUDE.roll, pitch = ODOMETRY.pitch), replaying
# six sessions' imu.csv through this exact filter (pilot/test_imu_level.py, 2026-08-01).
# Median |error| / p90, degrees, at LEVEL_IMU_TAU = 2.0:
#
#     20260731-204841-vq1-lap-slow   roll 1.36 / 3.71    pitch 1.34 / 3.96
#     20260731-195307 (lap, resets)  roll 0.67 / 3.12    pitch 1.12 / 4.86
#     20260731-131305 (mixed)        roll 0.03 / 6.53    pitch 0.22 / 6.62
#     20260731-150712 (excursions)   roll 0.03 / 17.4    pitch 1.08 / 9.72
#     20260731-143025 (rate doublets)roll 3.56 / 29.8    pitch 3.93 / 11.7
#
# ACCURACY IS A FUNCTION OF HOW HARD YOU FLY, and that is the whole story: on the two
# lap-like sessions -- the profile a slow mapping lap actually is -- it holds 1-2 deg
# median and better than 5 deg at p90. On the deliberate rate-doublet card it degrades to
# p90 30 deg, because sustained high rate is precisely when the accelerometer has nothing
# to say. Fly it slowly, as intended, and it is a good attitude; throw it around and it is
# not. It is opt-in for that reason.
#
# THE SIGNS ARE MEASURED, not assumed. Sweeping both hypotheses against truth is decisive
# by two orders of magnitude: gyro -1 with accel +1 gives roll median 0.11 deg, and every
# other combination gives 180 deg (roll) or 26-31 deg (pitch). So the gyro carries the
# mirror CONVENTIONS.md documents, and THE ACCELEROMETER AXES ARE CANONICAL -- which
# CONVENTIONS.md lists under "Still unverified". This is the referee it was missing, and
# it is an outside one: VQ1 truth pose is not derived from HIGHRES_IMU.
#
# Nothing extra is recorded. imu.csv already holds every input, so any estimate this makes
# is reproducible offline from the recording, and cmd.csv keeps its existing columns.
LEVEL_IMU_TAU = 2.0            # s; accelerometer trim rate. Measured, see the table.
LEVEL_IMU_ACCEL_TOL = 1.0      # m/s^2 of ||a| - g| before a sample counts as gravity.
                               # Looser than TILT_IMU_ACCEL_TOL (0.5) on purpose: a slow
                               # complementary trim can average a noisier gate, and 1.0
                               # is the value every number in the table was measured at.
LEVEL_IMU_MAX_TILT = 1.4       # rad; tan() blows up at 90 deg, so clamp before it does

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

# --- snap to hover on release (plain acro, VQ2's only mode) ----------------------------
# The same "throttle is an offset, not an absolute" behaviour as the levelling assist,
# but with no attitude input at all, so it works under VQ2 where the assist cannot engage.
# Hold a throttle key and thrust integrates exactly as it always did; release it and
# thrust goes to THRUST_HOVER on the next tick.
#
# Why the whole feature is worth one line of code: in ACRO the thrust channel is fully
# manual (NOTES.md -> Gotchas: no other flight mode is reachable in this build), so
# holding altitude down a 17-gate course means continuously re-trimming a value that has
# exactly one correct setting. This snaps to that setting.
#
# What it is NOT, and must not be mistaken for: an altitude hold. Nothing in VQ2 observes
# altitude or vertical speed -- position and velocity are blocked and double-integrating
# the accelerometer drifts -- so this returns to hover THRUST, not to a hover. Drift is
# still the pilot's to trim, and that is deliberate: no altitude measurement means any
# feedback loop here would be closing on a fiction, and a loop with no valid measurement
# oscillates. A constant cannot.
#
# One timing detail worth knowing before it surprises someone: the throttle AXIS is
# slew-limited by Sticks at SLEW_PER_S = 12/s, so after a physical key release the axis
# takes ~80 ms to fall from 1.0 through THRUST_STICK_EPS. Thrust keeps integrating for
# that ~80 ms and then snaps. That is inherited from the existing stick path, not added
# here, and 80 ms of 0.5/s slew is 0.04 of thrust -- but it is why the snap looks like it
# fires a frame or two late rather than instantly.
#
# Default ON (Claire, 2026-08-01: hand-holding thrust was blocking a full training lap).
# --no-thrust-snap restores the old hold-where-you-leave-it throttle, which is what the
# acro system-ID recordings were flown with.
THRUST_SNAP_DEFAULT = True

# --- optional: tilt compensation from the accelerometer, DEFAULT OFF -------------------
# Hover thrust rises as hover / cos(tilt), and under VQ2 the accelerometer is the only
# thing that sees tilt: with no linear acceleration the accel vector IS gravity, so
# cos(tilt) = -az / |a|. The levelling assist already does this from truth attitude; this
# would do it from a permitted stream.
#
# It is off by default because the measurement is unavailable exactly when it matters.
# The estimate is only valid when the drone is not manoeuvring, which is the same
# condition as "not banked":
#
#   accepted samples (||a| - g| <= 0.5) are 45% / 60% / 22% of frames in the three
#   longest sessions, with gaps of up to 5.0 s between accepted samples; and among the
#   accepted in-flight samples the implied tilt has median 2.0 deg and p90 4.5-5.8 deg.
#
# That last number is the killer: the gate passes near-level frames almost exclusively,
# so the compensation it computes is ~1.00 nearly always and stale whenever a real bank
# is on. It is shipped because it degrades to exactly the flat snap -- a rejected or
# stale sample decays the factor back to 1.0 rather than holding a wrong one -- and
# because that makes it cheap to A/B in flight. If it ever proves useful, the numbers
# above are what it has to beat.
TILT_IMU_ACCEL_TOL = 0.5       # m/s^2; ||a| - g| above this and the sample is not gravity
TILT_IMU_TAU = 0.5             # s, low-pass on cos(tilt); slower than any real gust
TILT_IMU_MAX_AGE = 0.5         # s without an accepted sample -> fall back to flat hover
G = 9.81

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

# --- input shaping --------------------------------------------------------------------
# A keyboard has no stick travel: a key is 0 or 1, so every input is full deflection and
# the only way to ask for a small correction is to tap. On a 17-gate winding course that
# is the whole workload. Three knobs, all FEEDFORWARD - nothing here reads a measurement,
# so nothing here can oscillate:
#
#   expo        soft centre. out = (1-e)*x + e*x^3. Only bites while the axis is between
#               0 and 1, which on a keyboard is exactly the slew ramp after a keypress -
#               so a tap becomes gentle and a held key still reaches full rate.
#   max_rate    hard clamp in rad/s on every rate that goes on the wire, applied AFTER
#               boost, align and the levelling assist. A clamp is used rather than a
#               scale on ACRO_* so that LCTRL boost cannot defeat it, and so that the
#               recorded limit is one number rather than a product of four.
#   stick_slew  replaces SLEW_PER_S for this run: how fast the axis itself ramps.
#
# THE PRESET GOES IN events.jsonl. cmd.csv feeds a plant fit, and a session flown under
# different shaping is a different experiment - identical stick work produces a different
# command trace. `full` is arithmetically identical to the pre-shaping code (expo 0 is an
# early return, max_rate 0 disables the clamp, stick_slew == SLEW_PER_S), which
# test_shaping.py asserts tick by tick against the old algebra.
#
# `slow` numbers, and why: max_rate 1.4 rad/s is 80 deg/s against acro's 143, which is
# still far more than a lap needs - the VQ1 slow lap (20260731-204841) was flown at a true
# median 2.9 m/s and its p99 commanded rate is well under this. expo 0.55 puts half stick
# at 59% of linear. Neither is tuned in flight; they are starting points chosen to be
# obviously gentler, and the cap is the part that matters.
SHAPING_PRESETS = {
    "full": dict(expo=0.0, max_rate=0.0, stick_slew=SLEW_PER_S),
    "slow": dict(expo=0.55, max_rate=1.4, stick_slew=5.0),
}

# --- ground speed from drag (VQ2 has no velocity telemetry at all) ---------------------
# Same algebra the align-to-velocity code uses for DIRECTION, read for MAGNITUDE. Thrust
# is along body -z, so the horizontal accelerometer components carry only drag, and drag
# grows with airspeed:
#
#     |(ax, ay)| = DRAG_K * v^2      =>     v = sqrt(|(ax,ay)| / DRAG_K)
#
# MEASURED against LOCAL_POSITION_NED on the VQ1 build, which has identical physics and
# does stream truth velocity: 35 203 in-flight IMU samples across 9 sessions, binned by
# true horizontal speed (pilot/_scratch_measure.py, 2026-08-01).
#
#     v m/s     2     3     4     5     6     7     8    12    15
#     |a| p50  0.19  0.38  0.68  1.04  1.65  1.99  2.60  4.46  8.04
#     |a|/v^2  .048  .042  .043  .042  .046  .041  .041  .031  .036
#
# So 0.041 holds to ~10% from 2 to 8 m/s, which is the entire slow-lap regime; it reads
# low above 10 m/s, i.e. it UNDER-reports exactly when you are already too fast, which is
# the safe direction for a warning. Inverting the calibration reproduces the truth
# medians: |a| 0.5 -> 3.5 m/s (measured 3.3), 1.2 -> 5.4 (4.9), 2.0 -> 7.0 (7.0).
#
# TWO REAL LIMITS. It is AIRspeed (the sim appears windless), and ON THE GROUND it reads
# the pad reaction instead of drag - the launch pad is inclined 17.8 deg, so a parked
# drone shows a rock-steady |(ax,ay)| = g*sin(17.8) = 3.00, which inverts to a wholly
# fictional 8.6 m/s. Hence the readout is gated on armed-and-flying.
DRAG_K = 0.041

# Advisory only, nothing acts on these. Cruise on this course is 5.8 m/s (the station
# ruler: 2.76 s/station x 15.97 m/station, NOTES.md + perception/NOTES.md), so 6 is
# "you are at race pace, not map pace".
#
# HONEST GAP: no recording pairs a measured speed with a station-read coverage figure.
# 20260731-222724 is the 38%-coverage session and its cmd.csv never shows armed, so its
# IMU cannot be gated to flight; 20260730-174527 (1%) is dominated by pad samples. The
# 38-vs-1% coverage split is real and speed is the stated cause, but the NUMBER below is
# chosen from the cruise pace, not fitted to coverage.
SPEED_MAP_OK = 6.0        # m/s; above this the HUD goes amber
SPEED_MAP_BAD = 9.0       # m/s; red

# --- steadiness (this is what a gate HEIGHT costs) -------------------------------------
# With no pose stream, gate height comes from gravity in the accelerometer, and the
# accelerometer only reads gravity when the aircraft is not accelerating. Same gate the
# hover measurement uses: ||a| - g| <= 0.35. perception/NOTES.md reports 4 of 5 labelled
# frames unusable for height on 20260731-222724 for exactly this reason ("a calmer session
# is the place to get heights"), so the pilot needs to see it WHILE flying, not afterwards.
#
# Measured in-flight acceptance at this tolerance: 39% on the slow VQ1 lap with a worst
# gap of 3.7 s, 61% on 20260730-215221. So a few seconds without a steady sample is
# normal; ten is a stretch of course that will yield no heights.
STEADY_TOL = 0.35         # m/s^2 of ||a| - g|
STEADY_WARN_S = 10.0      # seconds without a steady sample before the HUD complains

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
    # W is nose UP, not nose down. The old comment here said "w = nose down = forward"
    # and was simply wrong -- CONVENTIONS.md settled it against the pilot, the strongest
    # referee there is. The SIGNS are deliberately left alone: flipping them changes the
    # feel of every session ever recorded and wants doing on purpose, not hours before a
    # lap. Only the labels are corrected, here and in KEYMAP.
    "pitch":    ("w", "s"),       # w = nose UP (flies backward), s = nose down
    "roll":     ("d", "a"),       # d = roll right, a = roll left
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
  W     nose UP  (flies BACKWARD)  F5   arm
  S     nose down (flies forward)  F6   disarm
  A/D   roll left / roll right     F7   zero heading readout
  Q/E   yaw RIGHT / yaw LEFT       F8   quit (disarms)
  UP    throttle up                F9   sim reset
  DOWN  throttle down              F10  throttle back to hover
  C     align nose to velocity     F11  levelling assist on/off (VQ1 only)
  LCTRL boost   LALT precision     F12  drop a marker

  Q/E and W/S read backwards from the letters, and always have. That is the sim's
  mirrored rate convention, measured against the pilot, not a typo (CONVENTIONS.md).
  The keys are NOT being flipped hours before a lap - the labels above are what
  they actually do.

  Throttle is an OFFSET, not an absolute: hold UP/DOWN to move it, let go and it
  snaps to the measured hover point. It sits at 0 on arm and after a reset, like
  a real transmitter. Arming runs a short takeoff burst and settles at hover
  thrust; throttle-down cancels it. --no-thrust-snap gives back the old throttle
  that stays where you leave it.

  The snap is hover THRUST, not an altitude hold - nothing in VQ2 measures
  height, so you will still drift and still have to trim.

  Release the other keys and the rates go to zero, which is not a hover - the
  drone keeps whatever attitude it had.

  Hold C to yaw the nose onto the direction of travel, read from drag in the
  accelerometer. It is gated near hover, where there is no drag to read.

  F11 toggles the levelling assist: the sticks command a bank ANGLE instead of a
  rate, and releasing them returns to level instead of holding the attitude - so
  A/D and W/S become strafe and forward/back that recover on their own. Body rates
  are still what goes on the wire, so recordings stay ordinary acro data.

  The assist needs an attitude. On VQ1 it uses the truth stream. On VQ2 there is
  none, so start with --imu-level and it estimates roll and pitch from gravity in
  the IMU instead (1-2 deg on lap-like flying, worse the harder you fly - so fly
  it slowly). Yaw is not estimated and never levelled. Without that flag F11 says
  so and does nothing, exactly as before.

  Keys reach the simulator too. Commands sit on F-keys because the sim binds
  SPACE to restart; axes sit on letters because an echoed letter is harmless.
"""


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Shaping:
    """How a keypress becomes a rate command. Feedforward only - it reads no
    measurement, so it cannot oscillate.

    Exists because a keyboard axis is 0 or 1: without shaping every input is full
    deflection, and a small correction can only be spelled as a tap. See
    SHAPING_PRESETS for what the three knobs do and why `full` must stay
    arithmetically identical to the pre-shaping code.
    """

    def __init__(self, name, expo=0.0, max_rate=0.0, stick_slew=SLEW_PER_S):
        self.name = name
        self.expo = float(expo)
        self.max_rate = float(max_rate)      # rad/s; 0 = no cap
        self.stick_slew = float(stick_slew)

    @classmethod
    def preset(cls, name):
        return cls(name, **SHAPING_PRESETS[name])

    def curve(self, x):
        """Soft centre. Identity when expo is 0 -- an early return, not a
        multiply by one, so `full` is bit-for-bit the old arithmetic."""
        if self.expo <= 0.0:
            return x
        return (1.0 - self.expo) * x + self.expo * x * x * x

    def clamp(self, rate):
        """Cap on what goes on the wire. Applied last, so LCTRL boost, the
        align loop and the levelling assist are all inside it."""
        if self.max_rate <= 0.0:
            return rate
        return max(-self.max_rate, min(self.max_rate, rate))

    def as_dict(self):
        """Recorded verbatim in events.jsonl. cmd.csv feeds a plant fit, and a
        run flown under different shaping is a different experiment, so the
        limits that produced a command trace travel with it."""
        return dict(preset=self.name, expo=self.expo,
                    max_rate_rad_s=self.max_rate,
                    stick_slew_per_s=self.stick_slew,
                    acro_roll=ACRO_ROLL, acro_pitch=ACRO_PITCH,
                    acro_yaw=ACRO_YAW, boost=BOOST, precision=PRECISION)

    def describe(self):
        if self.expo <= 0.0 and self.max_rate <= 0.0 \
                and self.stick_slew == SLEW_PER_S:
            return "shaping: %s (unshaped - identical to every earlier session)" % self.name
        return ("shaping: %s   expo %.2f   max rate %s   stick ramp %.1f/s"
                % (self.name, self.expo,
                   ("%.2f rad/s (%.0f deg/s)"
                    % (self.max_rate, math.degrees(self.max_rate)))
                   if self.max_rate > 0 else "uncapped",
                   self.stick_slew))


class TiltEstimator:
    """Roll and pitch from HIGHRES_IMU alone, in PHYSICAL NED (not the sim's
    mirrored command convention). Yaw is deliberately absent - it is not
    observable from gravity and is out of bounds besides.

    Complementary filter: integrate the gyro for the fast path, trim toward the
    accelerometer's gravity direction whenever the accelerometer is actually
    reading gravity. See LEVEL_IMU_TAU above for the measured error against VQ1
    truth, and for why the signs applied here are measured rather than assumed.

    Feeds the levelling assist, which is a P loop on the result - so the one thing
    that would make this dangerous is lag inside the loop. There is none on the
    fast path: the gyro term is instantaneous and the accelerometer trim runs at
    tau = 2 s, far slower than any airframe mode. That ordering is what keeps a
    complementary filter from oscillating, and it is why this is not a PID.
    """

    def __init__(self, tau=LEVEL_IMU_TAU, accel_tol=LEVEL_IMU_ACCEL_TOL):
        self.tau = tau
        self.accel_tol = accel_tol
        self.roll = 0.0
        self.pitch = 0.0
        self.seeded = False        # False until the first trustworthy accel sample
        self.accepted = 0
        self.samples = 0

    def update(self, gyro, accel, dt):
        """gyro/accel exactly as HIGHRES_IMU reports them. Returns (roll, pitch)."""
        self.samples += 1
        # The gyro is mirrored (CONVENTIONS.md); the accelerometer is not. Both
        # settled by the sweep against VQ1 truth documented at LEVEL_IMU_TAU.
        p, q, r = (-gyro[0], -gyro[1], -gyro[2])
        if dt > 0.0:
            tt = math.tan(max(-LEVEL_IMU_MAX_TILT,
                              min(LEVEL_IMU_MAX_TILT, self.pitch)))
            self.roll += dt * (p + q * math.sin(self.roll) * tt
                               + r * math.cos(self.roll) * tt)
            self.pitch += dt * (q * math.cos(self.roll) - r * math.sin(self.roll))

        ax, ay, az = accel
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if mag > 1e-6 and abs(mag - G) <= self.accel_tol:
            self.accepted += 1
            roll_a = math.atan2(-ay, -az)
            pitch_a = math.atan2(ax, math.hypot(ay, az))
            if not self.seeded:
                # Seed rather than blend, so the estimate starts correct instead of
                # walking to the truth from zero. It matters on the pad, which is
                # inclined 17.8 deg nose-down (NOTES.md) - starting from level would
                # mean starting 17.8 deg wrong.
                self.roll, self.pitch, self.seeded = roll_a, pitch_a, True
            elif dt > 0.0:
                alpha = 1.0 - math.exp(-dt / self.tau)
                self.roll += alpha * wrap_pi(roll_a - self.roll)
                self.pitch += alpha * wrap_pi(pitch_a - self.pitch)
        return self.roll, self.pitch


def speed_from_drag(accel):
    """Airspeed estimate in m/s from horizontal specific force, or None.

    VQ2 blocks every velocity stream, so this is the only speed number available
    at all. Calibration and its error bars are at DRAG_K. Returns None when there
    is no accelerometer sample; the caller is responsible for not showing it on
    the ground, where the inclined pad reads a fictional 8.6 m/s.
    """
    if accel is None:
        return None
    return math.sqrt(math.hypot(accel[0], accel[1]) / DRAG_K)


def is_steady(accel, tol=STEADY_TOL):
    """True when the accelerometer is reading gravity and nothing else, i.e. when
    this instant could yield a gravity-referenced gate height. Same test the
    hover measurement uses."""
    if accel is None:
        return False
    mag = math.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)
    return abs(mag - G) <= tol


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
        # (x, y, z) world position exactly as reported. VQ1 only. See the HUD note.
        self.truth_pos = None
        self._last_imu_us = None
        # (roll, pitch, t_wall) estimated from the IMU, in the same physical NED
        # convention as truth_att, so the levelling assist can consume either. Runs
        # unconditionally and costs a few flops per IMU sample; whether the assist is
        # ALLOWED to use it is --imu-level's business, not this class's.
        self.tilt = TiltEstimator()
        self.imu_att = None

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
                # Held for the HUD as well as the recorder. Shown RAW, exactly as the
                # sim reports it, with no sign applied -- the whole point of the readout
                # is to let a human compare a reported number against a motion they can
                # see, and a "helpful" correction here would destroy that.
                with self.lock:
                    self.truth_pos = (msg.x, msg.y, msg.z)

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
            dt = 0.0
            if last is not None and msg.time_usec > last:
                dt = (msg.time_usec - last) / 1e6
                if dt <= HEADING_MAX_GAP_S:
                    self.heading_gyro = wrap_pi(
                        self.heading_gyro + YAW_GYRO_SIGN * msg.zgyro * dt)
                else:
                    dt = 0.0     # stall or reconnect: do not integrate across it
            # Updated here rather than on the control tick so it runs at the IMU's own
            # rate (61 Hz on the VM) with the sim's own dt, which is what the offline
            # validation against VQ1 truth replayed.
            roll, pitch = self.tilt.update(self.gyro, self.accel, dt)
            self.imu_att = (roll, pitch, time.time())
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
                        truth_seen=self.truth_seen, truth_att=self.truth_att,
                        truth_pos=self.truth_pos, imu_att=self.imu_att,
                        tilt_seeded=self.tilt.seeded)


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

    def __init__(self, slew=SLEW_PER_S):
        self.axes = [0.0, 0.0, 0.0, 0.0]   # pitch, roll, throttle, yaw
        self.align = False                 # align-to-velocity key held?
        # Per-run, because the slow-lap preset softens it -- but on the THREE RATE
        # AXES ONLY. The throttle axis keeps SLEW_PER_S no matter what the preset
        # says, because the snap-to-hover release edge is timed off it: the axis has
        # to fall through THRUST_STICK_EPS before the snap fires, and thrust keeps
        # integrating meanwhile (~80 ms, 0.04 of thrust, documented at
        # THRUST_SNAP_DEFAULT). A 5/s ramp would stretch that to 200 ms and 0.1 of
        # thrust. The snap is measured and not yet flight-tested; shaping has no
        # business changing its timing as a side effect.
        self.slew = float(slew)
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
        for i, want in enumerate(target):
            step = (SLEW_PER_S if i == 2 else self.slew) * dt   # i == 2 is throttle
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
                 hover=THRUST_HOVER, thrust_snap=THRUST_SNAP_DEFAULT,
                 tilt_comp_imu=False, shaping=None):
        self.shaping = shaping if shaping is not None else Shaping.preset("full")
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
        self.level_source = "truth"    # or "imu" when running off the tilt estimator
        self.level_err = (0.0, 0.0)
        # The hover point is a MEASURED quantity, so it lives on the instance and is
        # settable with --hover. THRUST_HOVER is only its default. Everything that
        # needs a hover thrust reads self.hover; nothing re-states the number.
        self.hover = hover
        self.hover_ref = hover
        self.thrust_snap = thrust_snap
        self.tilt_comp_imu = tilt_comp_imu
        self.snapped = False           # HUD: is thrust currently held at hover_ref?
        self._cos_tilt = 1.0           # low-passed cos(tilt) from the accelerometer
        self._cos_tilt_age = 1e9       # seconds since the last ACCEPTED accel sample

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

    def _imu_tilt_comp(self, accel, dt):
        """1/cos(tilt) from the accelerometer, or 1.0 when it cannot be trusted.

        Returns 1.0 -- i.e. the plain flat hover point -- whenever no sample has been
        ACCEPTED recently, so every failure mode of this estimator lands on the
        behaviour we would have had without it. See TILT_IMU_* above for why that
        matters: on this data the gate rejects most manoeuvring frames, which are
        precisely the banked ones, so "no recent sample" is the common case and has to
        be the safe one.
        """
        dt = max(0.0, dt)
        # Age is accumulated from the loop's own dt rather than read off the wall
        # clock, so the gate is tied to control ticks actually taken.
        self._cos_tilt_age += dt
        if accel is not None:
            mag = math.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)
            if mag > 1e-6 and abs(mag - G) <= TILT_IMU_ACCEL_TOL:
                # Gravity points along body -z when level, so -az/|a| is cos(tilt).
                # (Confirmed against the parked drone: az ~ -9.34 with |a| = 9.81.)
                ct = min(1.0, max(0.0, -accel[2] / mag))
                alpha = 1.0 - math.exp(-dt / TILT_IMU_TAU)
                self._cos_tilt += alpha * (ct - self._cos_tilt)
                self._cos_tilt_age = 0.0
        if self._cos_tilt_age > TILT_IMU_MAX_AGE:
            self._cos_tilt = 1.0
            return 1.0
        if self._cos_tilt <= 1.0 / TILT_COMP_MAX:
            return TILT_COMP_MAX
        return 1.0 / self._cos_tilt

    # -- streamed setpoints -----------------------------------------------------
    def send(self, axes, scale, dt, armed, heading=0.0, align=False, accel=None,
             level=False, truth_att=None, att_source="truth"):
        """Body rates + collective thrust. No frame, no heading, no rotation:
        a roll rate is a roll rate regardless of where north is, which is the
        whole reason this is the only control path teleop offers."""
        if self.listen_only:
            return
        pitch_ax, roll_ax, thr_ax, yaw_ax = axes

        # Expo on the three RATE axes. Throttle is deliberately left raw: it is not a
        # deflection, it is an integrator with a measured snap target, and softening its
        # centre would change the hover behaviour rather than the feel.
        pitch_ax = self.shaping.curve(pitch_ax)
        roll_ax = self.shaping.curve(roll_ax)
        yaw_ax = self.shaping.curve(yaw_ax)

        roll = roll_ax * ACRO_ROLL * scale
        pitch = -pitch_ax * ACRO_PITCH * scale   # NED: nose down is negative
        yaw = yaw_ax * ACRO_YAW * scale

        # Levelling assist: the stick becomes an angle demand on roll and pitch, and an
        # outer P loop turns the angle error into the body rate we were going to send
        # anyway. Yaw and throttle are untouched - yaw stays acro because levelling it
        # would mean holding a heading, and holding a heading means knowing one.
        # Hover reference for THIS tick. Recomputed every tick rather than left where
        # the last branch put it: hover_ref used to be written only inside the levelling
        # block, so switching the assist off left a tilt-compensated value latched.
        self.hover_ref = self.hover
        if self.tilt_comp_imu:
            self.hover_ref = min(1.0, self.hover * self._imu_tilt_comp(accel, dt))

        self.levelling = False
        if level and truth_att is not None:
            t_roll, t_pitch, t_stamp = truth_att
            if time.time() - t_stamp <= LEVEL_MAX_AGE:
                self.levelling = True
                self.level_source = att_source
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
                #
                # Under --imu-level this is where tilt compensation finally works on
                # VQ2, and it is strictly better than --imu-tilt-comp: that one reads
                # the raw accelerometer, which goes stale exactly when banked, whereas
                # the estimator carries attitude through the manoeuvre on the gyro.
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
            self.snapped = False
            if thr_ax < -0.05 or time.time() >= self.takeoff_until:
                self.takeoff_until = 0.0
                self.thrust = self.hover
            else:
                self.thrust = TAKEOFF_THRUST
        elif ((self.levelling or self.thrust_snap)
                and abs(thr_ax) < THRUST_STICK_EPS):
            # Let go: snap straight to the hover point. Under the levelling assist that
            # point is tilt-compensated from truth attitude; in plain acro (all of VQ2)
            # it is the flat measured constant, or the accelerometer-derived one if
            # --imu-tilt-comp is on.
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
            if not self.snapped:
                # One event per release edge, not one per tick: the release instant is
                # what a later reader of events.jsonl wants to line up against cmd.csv.
                self.rec.event("thrust_snap", thrust=self.hover_ref,
                               tilt_comp=self.hover_ref / self.hover
                               if self.hover > 0 else 1.0,
                               source="level" if self.levelling else "acro")
            self.snapped = True
            self.thrust = min(1.0, max(0.0, self.hover_ref))
        else:
            # Throttle is a held value, not a stick deflection - it integrates
            # while a key is down and stays put when released. UNCHANGED by the snap:
            # while a key is down this is the only branch that runs, so held-key
            # behaviour is exactly what it has always been.
            self.snapped = False
            self.thrust = min(1.0, max(0.0,
                                       self.thrust + thr_ax * THRUST_SLEW * dt))

        # The cap is applied LAST, on its way to the wire, so nothing upstream can get
        # around it: boost, the align loop and the levelling assist are all inside it.
        # With max_rate 0 (the `full` preset) this is an early return and the value is
        # untouched.
        roll = self.shaping.clamp(roll)
        pitch = self.shaping.clamp(pitch)
        yaw = self.shaping.clamp(yaw)

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


def flight_quality(tel, steady_age):
    """The two things that silently spoil a mapping recording, as display strings.

    SPEED, because station-number reads are a function of how fast the columns go
    past (38% of frames on a slow session, 1% on a fast one -- perception/NOTES.md),
    and VQ2 has no velocity telemetry, so the drag estimate at DRAG_K is the only
    speed number that exists. Blank while disarmed: on the ground the accelerometer
    reads the 17.8 deg pad incline, which inverts to a fictional 8.6 m/s.

    STEADY, because a gate height comes from gravity in the accelerometer, which is
    only readable when the aircraft is not accelerating.

    Advisory. Nothing acts on either value.
    """
    if not tel["armed"]:
        return "", ""
    v = speed_from_drag(tel["accel"])
    spd = "" if v is None else "  v~%.1f%s" % (
        v, "!!" if v > SPEED_MAP_BAD else "!" if v > SPEED_MAP_OK else "")
    if steady_age is None:
        stead = "  steady --"
    elif steady_age < 0.5:
        stead = "  STEADY"
    else:
        stead = "  steady %.0fs%s" % (steady_age,
                                      " !" if steady_age > STEADY_WARN_S else "")
    return spd, stead


def draw_hud(img, pilot, tel, vision, marker_count, heading_now, steady_age=None):
    view = img.copy()
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.rectangle(view, (0, h - 20), (w, h), (0, 0, 0), -1)

    a, b, c, d = pilot.last_cmd
    line1 = "RATES r%+.2f p%+.2f y%+.2f  THR %.2f%s%s  hdg %+4.0f%s" % (
        a, b, c, d,
        (" LEVEL[%s] hov%.2f" % (pilot.level_source, pilot.hover_ref))
        if pilot.levelling
        else (" HOV%.2f" % pilot.hover_ref) if pilot.snapped else "",
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
    # World position, RAW. VQ1 only -- under VQ2 there is no pose stream and this stays
    # blank, which is itself the correct readout.
    if tel.get("truth_pos") is not None:
        line2 += "  NED %+.1f %+.1f %+.1f" % tel["truth_pos"]
    spd, stead = flight_quality(tel, steady_age)
    line2 += spd + stead

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


def console_line(pilot, tel, vision, heading_now, steady_age=None):
    a, b, c, d = pilot.last_cmd
    vel = ("%+4.0f" % math.degrees(pilot.align_bearing)
           if pilot.align_bearing is not None else "  --")
    spd, stead = flight_quality(tel, steady_age)
    sys.stdout.write(
        "\r%-8s r%+.2f p%+.2f y%+.2f thr%.2f%-8s hdg%+4.0f vel%s%-6s gate %-3s "
        "t%6.1f contact %-3d cam %4.1f fps imu %-6d%-9s%-12s " % (
            "ARMED" if tel["armed"] else "disarmed",
            a, b, c, d,
            " TAKEOFF" if pilot.takeoff_until else "",
            math.degrees(heading_now), vel,
            (" LEVEL" if pilot.levelling else
             " ALIGN" if pilot.aligning else ""),
            tel["active_gate"] if tel["active_gate"] >= 0 else "-",
            tel["race_time_s"], tel["collisions"], vision.fps, tel["imu_count"],
            spd, stead))
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
    ap.add_argument("--no-thrust-snap", action="store_true",
                    help="restore the old throttle: a held value that stays where you "
                         "leave it. By default, releasing the throttle keys snaps thrust "
                         "to the hover point. Use this for acro system-ID recordings, "
                         "which were flown with the old behaviour.")
    ap.add_argument("--imu-tilt-comp", action="store_true",
                    help="EXPERIMENTAL, off by default: raise the snap target by "
                         "1/cos(tilt) read from the accelerometer. The reading is only "
                         "valid when not manoeuvring, i.e. mostly when not banked, so it "
                         "reads ~1.00 almost always; it falls back to flat hover.")
    ap.add_argument("--imu-level", action="store_true",
                    help="let the F11 levelling assist run on roll/pitch estimated from "
                         "HIGHRES_IMU, so it works under VQ2 where there is no ATTITUDE "
                         "stream. Roll/pitch from gravity is in bounds (NOTES.md); yaw is "
                         "not estimated and not touched. Measured against VQ1 truth at "
                         "1-2 deg median on lap-like flying, worse the harder you fly.")
    ap.add_argument("--shaping", choices=sorted(SHAPING_PRESETS), default="full",
                    help="how a keypress becomes a rate command (default %(default)s). "
                         "'full' is exactly the pre-shaping behaviour every earlier "
                         "session was flown with. 'slow' adds expo, a hard rate cap and "
                         "a softer stick ramp for map-building laps. The preset is "
                         "written to events.jsonl either way.")
    ap.add_argument("--slow-lap", action="store_true",
                    help="the one flag for a slow mapping lap: --shaping slow plus "
                         "--imu-level. It ARMS the levelling assist; F11 still has to "
                         "engage it, so this cannot change how the aircraft flies "
                         "without a deliberate keypress.")
    ap.add_argument("--expo", type=float, default=None,
                    help="override the preset's expo, 0..1 (0 = linear)")
    ap.add_argument("--max-rate", type=float, default=None,
                    help="override the preset's rate cap in rad/s (0 = uncapped). "
                         "Applied after boost and after any assist.")
    ap.add_argument("--stick-slew", type=float, default=None,
                    help="override the preset's stick ramp, in units of full "
                         "deflection per second. Rate axes only; throttle keeps %.1f."
                         % SLEW_PER_S)
    ap.add_argument("--listen", action="store_true",
                    help="record only: send no setpoints, no arm/disarm, no reset. "
                         "Safe to attach to a flight already in progress.")
    args = ap.parse_args()

    if args.slow_lap:
        args.imu_level = True
    shaping = Shaping.preset("slow" if args.slow_lap else args.shaping)
    if args.expo is not None:
        shaping.expo = args.expo
    if args.max_rate is not None:
        shaping.max_rate = args.max_rate
    if args.stick_slew is not None:
        shaping.stick_slew = args.stick_slew
    if (args.expo, args.max_rate, args.stick_slew) != (None, None, None):
        # A hand-tuned run is not the preset it started from, and events.jsonl must not
        # imply otherwise -- the preset name is what a later reader will group sessions by.
        shaping.name = shaping.name + "+custom"

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
                  hover=args.hover, thrust_snap=not args.no_thrust_snap,
                  tilt_comp_imu=args.imu_tilt_comp, shaping=shaping)
    sticks = Sticks(slew=shaping.stick_slew)
    heading = HeadingReadout()

    rec.event("session_start", control="body_rates", ip=args.ip, port=args.port,
              listen_only=args.listen, hover=pilot.hover,
              thrust_snap=pilot.thrust_snap, imu_tilt_comp=pilot.tilt_comp_imu,
              imu_level=args.imu_level, shaping=shaping.as_dict())
    # Written a SECOND time under its own kind as well as inside session_start, so that
    # anything scanning events.jsonl for what shaped the commands finds it by name
    # without having to know that session_start carries a nested dict.
    rec.event("input_shaping", **shaping.as_dict())
    if args.listen:
        print("\nLISTEN ONLY - recording; no commands will be sent.\n")
    else:
        print(KEYMAP)
    print(shaping.describe())
    if args.imu_level and not args.listen:
        print("IMU levelling is ARMED: press %s to engage the assist off gravity-\n"
              "estimated roll/pitch. Fly it slowly - the estimate degrades with how\n"
              "hard you fly (1-2 deg on lap-like flying, p90 30 deg on rate doublets).\n"
              "Yaw is not estimated and is never levelled."
              % KEYS_COMMAND["level"].upper())
    print("recording to: %s" % rec.dir)
    if not args.listen:
        print("Body-rate control only. This flies like an acro quad: the drone holds\n"
              "whatever attitude you leave it in, so it will NOT self-level and\n"
              "releasing the keys is not a hover. Throttle starts at %.2f.\n"
              % THRUST_START)
        if pilot.thrust_snap:
            print("Throttle SNAP is on: release UP/DOWN and thrust goes to %.3f (the\n"
                  "measured hover point). That is hover THRUST, not an altitude hold -\n"
                  "nothing in VQ2 observes height, so drift is still yours to trim.%s\n"
                  % (pilot.hover,
                     "\nIMU tilt compensation is ON (experimental)."
                     if pilot.tilt_comp_imu else ""))
        else:
            print("Throttle snap DISABLED: throttle is a held value that stays where\n"
                  "you leave it.\n")
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
    last_steady = None        # wall time of the last gravity-only accelerometer sample
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
                    src = ("truth" if snap["truth_att"] is not None
                           else "imu" if (args.imu_level and snap["imu_att"] is not None
                                          and snap["tilt_seeded"]) else None)
                    if src is None:
                        print("\n[level] unavailable: no ATTITUDE stream (VQ2 blocks "
                              "it). Restart with --imu-level to estimate roll/pitch "
                              "from the IMU instead.")
                    else:
                        level_on = not level_on
                        rec.event("level_assist", on=level_on, source=src)
                        print("\n[level] %s (%s attitude)"
                              % ("ON" if level_on else "OFF", src))
                elif action == "marker":
                    markers += 1
                    rec.event("marker", index=markers,
                              active_gate=snap["active_gate"],
                              race_time_s=snap["race_time_s"])
                    print("\n[marker %d] gate=%s t=%.2f"
                          % (markers, snap["active_gate"], snap["race_time_s"]))
                elif action == "quit":
                    running = False

            # Which attitude the levelling assist is allowed to close on. VQ1 truth
            # always wins when it exists - it is a measurement, not an estimate - and the
            # IMU estimate is only offered when --imu-level says so, so the assist's
            # behaviour on every previously recorded session is unchanged.
            level_att, level_src = snap["truth_att"], "truth"
            if level_att is None and args.imu_level and snap["imu_att"] is not None:
                level_att, level_src = snap["imu_att"], "imu"

            head_now = heading.value(snap["heading_gyro"])
            pilot.send(axes, scale, dt, snap["armed"], head_now,
                       align=sticks.align, accel=snap["accel"],
                       level=level_on, truth_att=level_att, att_source=level_src)

            wall = time.time()
            if is_steady(snap["accel"]):
                last_steady = wall
            steady_age = None if last_steady is None else wall - last_steady

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
                                                markers, head_now, steady_age))
                if cv2.waitKey(1) & 0xFF == 27:
                    running = False

            if wall - last_console >= 0.25:
                console_line(pilot, snap, vision, head_now, steady_age)
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
