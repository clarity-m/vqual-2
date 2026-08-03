"""autopilot.py -- the live closed-loop harness: producer -> Policy -> link layer.

This is the missing piece between the frozen interface and the sim. It reuses teleop's
proven plumbing wholesale (Telemetry, VisionRX, Pilot, Recorder -- imported, not
forked), adds the producer on its own thread per PRODUCER.md's wiring notes, and runs
whatever `interface.Policy` is asked for at CONTROL_HZ. Any policy that runs here runs
against the surrogate unchanged, and vice versa -- that equivalence is the whole point
of the interface.

THREADS
  telemetry (teleop.Telemetry)  -- MAVLink RX; a subclass here also buffers raw
                                   HIGHRES_IMU rows for the producer.
  vision    (teleop.VisionRX)   -- JPEG reassembly + decode.
  perception (this file)        -- paces the producer at the camera rate: fresh frame
                                   when one arrived, frame=None tick otherwise so IMU
                                   integration and coasting continue. Publishes the
                                   latest Observation; never blocks the control loop.
  main                          -- 50 Hz control: read latest obs, call the policy,
                                   yaw servo, link layer, HUD. All sends happen here.

SIGNS. Policy output is CANONICAL (interface.py). The link layer here applies the -1
mirror on all three rates -- the same convention teleop's constants carry -- in exactly
one place (_wire). cmd.csv records the WIRE values, same as teleop, so plant-fit
tooling reads both kinds of session identically.

SAFETY / DQ. Commands are teleop's F-keys (KEYS_COMMAND, same global hook), so the
bindings and the muscle memory carry over unchanged -- and, unlike a cv2-window key,
the abort works regardless of which window has focus, which matters when Claire is
watching the sim window. F-keys are the ones games essentially never bind (teleop's
own rationale), so leaking to the sim is safe. The submission rule is "no human
interaction during a timed run": after F5, hands off; F8 is the abort (cut + disarm),
which forfeits the run rather than corrupting it. ESC in the cv2 window also aborts.

    python3 pilot/autopilot.py                    # ServoPolicy, gatenet on
    python3 pilot/autopilot.py --no-net           # contour-only producer (no torch)
    python3 pilot/autopilot.py --policy hover     # inner-loop-only: level + hover
    F5 arm+takeoff | F6 disarm | F8 quit | F9 sim reset | F11 hold toggle | F12 marker
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import teleop as T
from interface import Action, HOVER_THRUST, YawMode
from policy_servo import ClimbDamper, ServoPolicy, G_ATT, RATE_MAX, TILT_COMP_MAX

# The one place the canonical->wire mirror lives in this file. Roll/pitch reuse
# teleop's derived constants; yaw's is stated here because teleop keeps its yaw sign
# inside ACRO_YAW's usage rather than a named constant. CONVENTIONS.md: all three
# mirrored, wire = -canonical.
RATE_SIGN_ROLL = T.RATE_SIGN_ROLL      # -1.0
RATE_SIGN_PITCH = T.RATE_SIGN_PITCH    # -1.0
RATE_SIGN_YAW = -1.0

PERCEPTION_HZ = 30.0      # producer pacing; fresh frames gate themselves by frame_id
OBS_FAILSAFE_S = 0.75     # no Observation newer than this -> level-and-hover failsafe
# Softer than teleop's 0.40 x 1.0 s unstick: the first hover test (2026-08-02) showed
# the burst's climb rate persists after handover -- thrust equal to weight is zero net
# force. The ClimbDamper now bleeds it, but there is no reason to inject 2 m/s of climb
# on purpose either.
TAKEOFF_THRUST = 0.34
TAKEOFF_S = 0.8


class TelemetryTap(T.Telemetry):
    """teleop.Telemetry plus a drainable buffer of raw HIGHRES_IMU rows.

    The producer wants every raw row since its last step (its ImuFilter owns the gyro
    mirror and its own integration); Telemetry only keeps the latest. Buffered AFTER
    super()._on_imu so the recorder/HUD path is untouched, under a separate lock so we
    never lock-invert against Telemetry's own.
    """

    def __init__(self, conn, rec):
        self._imu_lock = threading.Lock()
        self._imu_rows = collections.deque(maxlen=4096)
        super().__init__(conn, rec)

    def _on_imu(self, msg):
        super()._on_imu(msg)
        with self._imu_lock:
            self._imu_rows.append((time.time_ns(),
                                   msg.xacc, msg.yacc, msg.zacc,
                                   msg.xgyro, msg.ygyro, msg.zgyro))

    def drain_imu(self):
        with self._imu_lock:
            rows = list(self._imu_rows)
            self._imu_rows.clear()
        return rows


class PerceptionLoop:
    """Runs the producer at camera pace on its own thread; publishes the latest obs."""

    def __init__(self, vision, tap, use_net=True, use_compass=True):
        from producer import Producer
        self.prod = Producer(use_net=use_net, use_compass=use_compass)
        self.vision = vision
        self.tap = tap
        self.lock = threading.Lock()
        self.latest = None            # (Observation, wall_time)
        self.steps = 0
        self.fresh_frames = 0
        self.last_ms = 0.0
        self._last_frame_id = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=False)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=3.0)

    def take(self):
        with self.lock:
            return self.latest

    def _race_dict(self, snap):
        last = snap.get("last_collision")
        return dict(active_gate_index=max(0, snap["active_gate"]),
                    race_time_s=snap["race_time_s"],
                    armed=snap["armed"],
                    n_gates_total=17,
                    t_since_collision_s=(time.time() - last[0]) if last else 1e3,
                    collision_episodes=snap["collisions"])

    def _loop(self):
        period = 1.0 / PERCEPTION_HZ
        next_tick = time.perf_counter()
        while self._running:
            frame = None
            latest = self.vision.take()
            if latest is not None and latest[0] != self._last_frame_id:
                self._last_frame_id = latest[0]
                frame = latest[2]
                self.fresh_frames += 1
            rows = self.tap.drain_imu()
            snap = self.tap.snapshot()
            t0 = time.perf_counter()
            try:
                obs = self.prod.step(frame, rows, self._race_dict(snap), time.time())
            except Exception as e:              # a perception fault must read as
                obs = None                      # "stale obs" downstream, never as a
                print("\nPERCEPTION ERROR: %r" % (e,))  # dead control loop
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            if obs is not None:
                self.steps += 1
                with self.lock:
                    self.latest = (obs, time.time())
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()


class CommandKeys:
    """teleop's one-shot commands, verbatim: same KEYS_COMMAND table, same global-hook
    mechanism, same edge detection as teleop.Sticks. 'hover' and 'zero_head' are
    teleop-specific and not polled; 'level' maps to the hold toggle, its closest
    semantic relative (both mean "stop chasing things and sit level")."""

    ACTIONS = ("arm", "disarm", "quit", "reset", "level", "marker")

    def __init__(self):
        self._prev = {a: False for a in self.ACTIONS}

    def poll(self):
        import keyboard
        out = []
        for a in self.ACTIONS:
            try:
                held = keyboard.is_pressed(T.KEYS_COMMAND[a])
            except Exception:
                held = False          # unreadable key must read as "not pressed"
            if held and not self._prev[a]:
                out.append(a)
            self._prev[a] = held
        return out


class HoverPolicy:
    """Inner loop only: hold level, hover thrust, climb-rate damped. The first thing
    to fly, and the failsafe body -- if this cannot hold altitude, no policy above it
    can be trusted."""

    def __init__(self, hover=HOVER_THRUST):
        self.hover = float(hover)
        self.reset()

    def reset(self):
        self._damper = ClimbDamper()

    def __call__(self, obs):
        own = obs.own
        self._damper.update(own, obs.t_s)
        ct = math.cos(own.roll_rad) * math.cos(own.pitch_rad)
        comp = TILT_COMP_MAX if ct <= (1.0 / TILT_COMP_MAX) else 1.0 / ct
        return Action(
            roll_rate=max(-RATE_MAX, min(RATE_MAX, G_ATT * (0.0 - own.roll_rad))),
            pitch_rate=max(-RATE_MAX, min(RATE_MAX, G_ATT * (0.0 - own.pitch_rad))),
            yaw_rate=0.0,
            thrust=min(1.0, max(0.0, self.hover * comp + self._damper.damping()
                                + self._damper.thrust_offset())),
            yaw_mode=YawMode.AUTO_ATTENTION).clipped()


def make_policy(name, hover=HOVER_THRUST):
    if name == "servo":
        return ServoPolicy(hover=hover)
    if name == "hover":
        return HoverPolicy(hover=hover)
    sys.exit("unknown --policy %r (servo, hover)" % name)


def _overlay(img, state, snap, per, action, obs_age):
    h = img.shape[0]
    lines = [
        "%s  %s" % (state.upper(), "ARMED" if snap["armed"] else "disarmed"),
        "gate %d  race %.1fs  col %d" % (snap["active_gate"], snap["race_time_s"],
                                         snap["collisions"]),
        "producer %.0f ms  obs age %.2fs" % (per.last_ms, obs_age if obs_age else -1),
    ]
    if action is not None:
        lines.append("cmd r%+.2f p%+.2f y%+.2f t%.2f" % (
            action.roll_rate, action.pitch_rate, action.yaw_rate, action.thrust))
    o = per.take()
    if o is not None:
        g = o[0].gates[0]
        mode = getattr(per, "_policy_mode", "")
        if g.valid:
            lines.append("g%d rng %.1fm stale %.2fs %s" % (
                g.index, g.range_m, g.staleness_s, mode))
        else:
            lines.append("no gate  %s" % mode)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, h - 8 - 18 * (len(lines) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (8, h - 8 - 18 * (len(lines) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 80), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="AI-GP autonomous flight harness")
    ap.add_argument("--ip", default=T.MAVLINK_IP)
    ap.add_argument("--port", type=int, default=T.MAVLINK_PORT)
    ap.add_argument("--sessions", default=os.path.join(os.path.dirname(__file__),
                                                       "sessions"))
    ap.add_argument("--policy", default="servo", help="servo (default) or hover")
    ap.add_argument("--no-net", action="store_true",
                    help="contour-only producer (no torch import)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--no-frames", action="store_true")
    # 0.264, not teleop's 0.266: bisected live 2026-08-02 (0.266 climbs, 0.262 sinks).
    ap.add_argument("--hover-thrust", type=float, default=0.264)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="exit cleanly after N seconds (0 = until ESC)")
    ap.add_argument("--arm", action="store_true",
                    help="arm and take off immediately (headless / timed runs)")
    ap.add_argument("--reset-first", action="store_true",
                    help="send a sim reset before anything else (recover from a crash "
                         "without touching the sim window)")
    args = ap.parse_args()

    rec = (T.NullRecorder() if args.no_record
           else T.Recorder(args.sessions, save_frames=not args.no_frames))

    print("Binding MAVLink on udpin:%s:%d ..." % (args.ip, args.port))
    try:
        conn = T.mavutil.mavlink_connection("udpin:%s:%d" % (args.ip, args.port))
    except OSError as e:
        sys.exit("could not bind UDP %d (%s); another client holds it." % (args.port, e))
    print("Waiting for heartbeat ...")
    conn.wait_heartbeat()
    print("Connected to system %d, component %d"
          % (conn.target_system, conn.target_component))

    boot_ms = int(time.time() * 1000)
    tel = TelemetryTap(conn, rec)
    vision = T.VisionRX(rec, decode=True)
    print("Loading producer%s ..." % (" (no net)" if args.no_net else ""))
    per = PerceptionLoop(vision, tel, use_net=not args.no_net)
    pilot = T.Pilot(conn, rec, boot_ms, hover=args.hover_thrust)
    pilot.auto_takeoff = False        # takeoff is this file's state machine, not send()'s
    policy = make_policy(args.policy, hover=args.hover_thrust)
    holder = HoverPolicy(hover=args.hover_thrust)   # persistent: its damper must integrate

    keys = CommandKeys()
    rec.event("session_start", control="autopilot", policy=args.policy,
              use_net=not args.no_net, hover=args.hover_thrust,
              perception_hz=PERCEPTION_HZ, control_hz=T.CONTROL_HZ,
              obs_failsafe_s=OBS_FAILSAFE_S,
              keys_command={a: T.KEYS_COMMAND[a] for a in CommandKeys.ACTIONS})
    print("recording to: %s" % rec.dir)
    print("commands (teleop bindings, any window focused):\n"
          "  F5 arm+takeoff   F6 disarm   F8 quit   F9 sim reset\n"
          "  F11 hold (level+hover) toggle   F12 marker   [ESC in cv2 window: abort]")

    win = "AI-GP autopilot"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    period = 1.0 / T.CONTROL_HZ
    next_tick = time.perf_counter()
    last_hb = last_ts = last_console = 0.0
    state = "idle"                    # idle | takeoff | fly | hold
    takeoff_until = 0.0
    running = True
    deadline = time.perf_counter() + args.duration if args.duration else None
    if args.reset_first:
        pilot.sim_reset()
        time.sleep(1.5)
    want_arm = args.arm

    def wire(a: Action, armed):
        """Canonical Action -> the sim's mirrored convention -> the wire + cmd.csv."""
        r = a.roll_rate * RATE_SIGN_ROLL
        p = a.pitch_rate * RATE_SIGN_PITCH
        y = a.yaw_rate * RATE_SIGN_YAW
        pilot._send_rates(r, p, y, a.thrust)
        rec.row("cmd", time.time_ns(), int(armed), "%.4f" % 0.0, r, p, y, a.thrust)

    try:
        while running:
            now = time.perf_counter()
            if deadline and now >= deadline:
                break
            snap = tel.snapshot()

            if want_arm:
                want_arm = False
                pilot.arm(True)
                policy.reset()
                state, takeoff_until = "takeoff", time.time() + TAKEOFF_S
                rec.event("autopilot_state", state=state)
                print("\n[arm + takeoff]")

            latest = per.take()
            obs_age = (time.time() - latest[1]) if latest else None
            action = None

            if state == "takeoff":
                if time.time() >= takeoff_until:
                    state = "fly"
                    rec.event("autopilot_state", state=state)
                    print("\n[policy engaged: %s]" % args.policy)
                else:
                    att = snap["imu_att"]
                    r_rate = p_rate = 0.0
                    if att is not None:
                        r_rate = max(-RATE_MAX, min(RATE_MAX, G_ATT * (0.0 - att[0])))
                        p_rate = max(-RATE_MAX, min(RATE_MAX, G_ATT * (0.0 - att[1])))
                    action = Action(r_rate, p_rate, 0.0, TAKEOFF_THRUST,
                                    yaw_mode=YawMode.POLICY)

            if state in ("fly", "hold") and action is None:
                if latest is None or obs_age > OBS_FAILSAFE_S:
                    # Perception stale or dead: hold level off the IMU estimate.
                    att = snap["imu_att"] or (0.0, 0.0, 0.0)
                    action = Action(
                        max(-RATE_MAX, min(RATE_MAX, G_ATT * (0.0 - att[0]))),
                        max(-RATE_MAX, min(RATE_MAX, G_ATT * (0.0 - att[1]))),
                        0.0, HOVER_THRUST, yaw_mode=YawMode.POLICY)
                elif state == "hold":
                    action = holder(latest[0])
                else:
                    action = policy(latest[0])
                    per._policy_mode = getattr(policy, "_mode", "")
                if action.yaw_mode == YawMode.AUTO_ATTENTION and latest is not None:
                    y = per.prod.auto_yaw_rate(latest[0])
                    action = Action(action.roll_rate, action.pitch_rate, y,
                                    action.thrust, YawMode.AUTO_ATTENTION)

            if action is not None and not pilot.listen_only:
                wire(action.clipped(), snap["armed"])

            wall = time.time()
            if wall - last_hb >= 1.0 / T.HEARTBEAT_HZ:
                pilot.heartbeat()
                last_hb = wall
            if wall - last_ts >= 1.0 / T.TIMESYNC_HZ:
                pilot.timesync_request()
                last_ts = wall

            latest_frame = vision.take()
            if latest_frame is not None:
                img = _overlay(latest_frame[2].copy(), state, snap, per, action,
                               obs_age)
                cv2.imshow(win, img)
            if (cv2.waitKey(1) & 0xFF) == 27:
                running = False
            for act in keys.poll():
                if act == "arm":
                    want_arm = True
                elif act == "disarm":
                    pilot.stop()
                    pilot.arm(False)
                    state = "idle"
                    rec.event("autopilot_state", state=state)
                    print("\n[disarm]")
                elif act == "quit":
                    running = False
                elif act == "reset":
                    pilot.sim_reset()
                    policy.reset()
                    state = "idle"
                    rec.event("autopilot_state", state=state)
                    print("\n[sim reset]")
                elif act == "level" and state in ("fly", "hold"):
                    state = "hold" if state == "fly" else "fly"
                    rec.event("autopilot_state", state=state)
                    print("\n[%s]" % state)
                elif act == "marker":
                    rec.event("marker", active_gate=snap["active_gate"],
                              race_time_s=snap["race_time_s"])
                    print("\n[marker] gate=%s t=%.2f"
                          % (snap["active_gate"], snap["race_time_s"]))

            if wall - last_console >= 0.5:
                o = per.take()
                g = o[0].gates[0] if o else None
                print("\r%-7s gate=%d %s prod=%.0fms fps=%.0f      "
                      % (state, snap["active_gate"],
                         ("rng=%.1f stale=%.2f" % (g.range_m, g.staleness_s))
                         if (g and g.valid) else "no-gate",
                         per.last_ms, vision.fps), end="")
                last_console = wall

            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping: neutral setpoint, disarm, flush.")
        for _ in range(10):
            pilot.stop()
            time.sleep(0.02)
        pilot.arm(False)
        time.sleep(0.1)
        rec.event("session_end", frames=vision.frames_done,
                  producer_steps=per.steps, fresh_frames=per.fresh_frames)
        per.stop()
        vision.stop()
        tel.stop()
        cv2.destroyAllWindows()
        rec.close()
        try:
            print("producer timing:", per.prod.timing_report())
        except Exception:
            pass
        print("session: %s" % rec.dir)


if __name__ == "__main__":
    main()
