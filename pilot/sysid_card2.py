"""TEMPORARY — Card 2 from SYSID.md. Delete once the gaps are flown and refit.

Automates the scriptable parts (preflight referee gate, A apex reads, optional B
terminal runs) and can hand sticks over for C (finished lap) and D (re-fly 130744).

    python3 pilot/sysid_card2.py                  # preflight session, then A
    python3 pilot/sysid_card2.py --only A
    python3 pilot/sysid_card2.py --only B --tall
    python3 pilot/sysid_card2.py --only C         # manual, JPEGs kept
    python3 pilot/sysid_card2.py --only D         # manual free flight
    python3 pilot/sysid_card2.py --all --tall     # preflight + A + B, then C, then D

VQ1 build, Training flight, nothing else on UDP 14550. Levelling assist stays OFF
during apex (plain acro throttle hold). F8 aborts.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

# Reuse teleop's link, recorder, and stick path rather than re-deriving signs.
import teleop as t

# Apex table from SYSID.md Card 2 §A. peak is where the UP ramp lands; hold is
# hands-off time after the test throttle is parked. Times are open-loop starting
# points from the plant under test — valid wherever the apex actually lands.
APEX = [
    # (test_thrust, peak_thrust, hold_s, envelope_m, note)
    (0.05, 0.50, 1.5, 6.4, ""),
    (0.10, 0.50, 1.5, 5.5, ""),
    (0.15, 0.50, 1.5, 5.9, ""),
    (0.20, 0.50, 1.5, 7.2, ""),
    (0.00, 0.60, 1.5, 14.1, "tall"),
]
APEX_REPS = 2
SETTLE_S = 2.0
LEVEL_S = 2.5
PREFLIGHT_S = 20.0
BETWEEN_MANEUVER_S = 1.5


class Runner:
    def __init__(self, conn, rec, tel, vision, pilot, hover):
        self.conn = conn
        self.rec = rec
        self.tel = tel
        self.vision = vision
        self.pilot = pilot
        self.hover = hover
        self.thrust = hover
        self.markers = 0
        self._period = 1.0 / t.CONTROL_HZ
        self._next = time.perf_counter()
        self._last_hb = 0.0
        self._last_ts = 0.0
        self._abort = False

    def _check_abort(self):
        try:
            if t.keyboard.is_pressed(t.KEYS_COMMAND["quit"]):
                self._abort = True
        except Exception:
            pass
        return self._abort

    def marker(self, label):
        self.markers += 1
        snap = self.tel.snapshot()
        self.rec.event("marker", index=self.markers, label=label,
                       active_gate=snap["active_gate"],
                       race_time_s=snap["race_time_s"],
                       thrust=self.thrust)
        print("\n[marker %d] %s  thr=%.2f" % (self.markers, label, self.thrust))

    def _send(self, roll, pitch, yaw, thrust):
        self.thrust = min(1.0, max(0.0, thrust))
        self.pilot.thrust = self.thrust
        self.pilot._send_rates(roll, pitch, yaw, self.thrust)
        snap = self.tel.snapshot()
        self.pilot.last_cmd = (roll, pitch, yaw, self.thrust)
        self.rec.row("cmd", time.time_ns(), int(snap["armed"]),
                     "%.4f" % 0.0, roll, pitch, yaw, self.thrust)
        wall = time.time()
        if wall - self._last_hb >= 1.0 / t.HEARTBEAT_HZ:
            self.pilot.heartbeat()
            self._last_hb = wall
        if wall - self._last_ts >= 1.0 / t.TIMESYNC_HZ:
            self.pilot.timesync_request()
            self._last_ts = wall

    def _pace(self):
        self._next += self._period
        sleep = self._next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            self._next = time.perf_counter()

    def hold(self, seconds, thrust=None, roll=0.0, pitch=0.0, yaw=0.0):
        """Hold rates/thrust for `seconds`. thrust=None keeps the current value."""
        if thrust is not None:
            self.thrust = thrust
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            if self._check_abort():
                return False
            self._send(roll, pitch, yaw, self.thrust)
            self._pace()
        return True

    def slew_thrust(self, target, rates=(0.0, 0.0, 0.0)):
        """Ramp collective at THRUST_SLEW (same as the UP/DOWN keys)."""
        target = min(1.0, max(0.0, target))
        roll, pitch, yaw = rates
        while abs(self.thrust - target) > 1e-3:
            if self._check_abort():
                return False
            step = t.THRUST_SLEW * self._period
            if self.thrust < target:
                self.thrust = min(target, self.thrust + step)
            else:
                self.thrust = max(target, self.thrust - step)
            self._send(roll, pitch, yaw, self.thrust)
            self._pace()
        self.thrust = target
        self._send(roll, pitch, yaw, self.thrust)
        self._pace()
        return True

    def level(self, seconds=LEVEL_S):
        """Brief outer-loop level using VQ1 truth. Assist OFF afterwards."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            if self._check_abort():
                return False
            snap = self.tel.snapshot()
            att = snap["truth_att"]
            roll = pitch = 0.0
            if att is not None and time.time() - att[2] <= t.LEVEL_MAX_AGE:
                err_r, err_p = -att[0], -att[1]
                p_roll = max(-t.LEVEL_MAX_RATE, min(t.LEVEL_MAX_RATE,
                                                    t.LEVEL_GAIN * err_r))
                p_pitch = max(-t.LEVEL_MAX_RATE, min(t.LEVEL_MAX_RATE,
                                                     t.LEVEL_GAIN * err_p))
                roll = p_roll * t.RATE_SIGN_ROLL
                pitch = p_pitch * t.RATE_SIGN_PITCH
            self._send(roll, pitch, 0.0, self.hover)
            self._pace()
        return True

    def takeoff(self):
        print("arm + takeoff -> hover %.2f" % self.hover)
        self.pilot.auto_takeoff = True
        self.pilot.arm(True)
        # Drive the same takeoff profile teleop uses, then freeze at hover.
        end = time.time() + t.TAKEOFF_S + 0.5
        while time.time() < end:
            if self._check_abort():
                return False
            thr = t.TAKEOFF_THRUST if time.time() < self.pilot.takeoff_until else self.hover
            self._send(0.0, 0.0, 0.0, thr)
            self._pace()
        self.pilot.takeoff_until = 0.0
        self.pilot.auto_takeoff = False
        return self.hold(SETTLE_S, thrust=self.hover)

    def cut(self):
        for _ in range(10):
            self._send(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.02)
        self.pilot.arm(False)
        time.sleep(0.1)


def connect(args):
    """One MAVLink bind for the whole script — port 14550 is exclusive."""
    print("Binding MAVLink on udpin:%s:%d ..." % (args.ip, args.port))
    try:
        conn = t.mavutil.mavlink_connection("udpin:%s:%d" % (args.ip, args.port))
    except OSError as e:
        sys.exit("could not bind UDP %d (%s). Another client is holding it."
                 % (args.port, e))
    print("Waiting for heartbeat ...")
    conn.wait_heartbeat()
    print("Connected to system %d" % conn.target_system)
    return conn, int(time.time() * 1000)


def open_session(args, conn, boot_ms, save_frames):
    rec = t.Recorder(args.sessions, save_frames=save_frames)
    tel = t.Telemetry(conn, rec)
    vision = t.VisionRX(rec, decode=False)
    pilot = t.Pilot(conn, rec, boot_ms, hover=args.hover)
    pilot.auto_takeoff = False
    run = Runner(conn, rec, tel, vision, pilot, args.hover)
    rec.event("session_start", control="sysid_card2", ip=args.ip, port=args.port,
              save_frames=save_frames)
    deadline = time.time() + 5.0
    while time.time() < deadline and not tel.snapshot()["truth_seen"]:
        time.sleep(0.05)
    if not tel.snapshot()["truth_seen"]:
        print("WARNING: no ATTITUDE/POSITION/ODOMETRY yet — is this the VQ1 build?")
    print("recording to: %s" % rec.dir)
    return run


def close_session(run, label):
    run.rec.event("session_end", label=label, markers=run.markers,
                  frames=run.vision.frames_done)
    run.vision.stop()
    run.tel.stop()
    run.rec.close()
    print("session closed: %s" % run.rec.dir)
    return run.rec.dir


def run_referee(session_dir):
    script = os.path.join(os.path.dirname(__file__), "control", "sysid_frames.py")
    print("\n--- sysid_frames.py on %s ---" % session_dir)
    proc = subprocess.run([sys.executable, script, session_dir])
    return proc.returncode == 0


def maneuver_preflight(run):
    """20 s of attitude variety, then the caller closes and referees."""
    print("\n=== PREFLIGHT (~%.0fs roll/pitch/yaw) ===" % PREFLIGHT_S)
    if not run.takeoff():
        return False
    if not run.level():
        return False
    run.marker("preflight_start")
    # Gentle doublets so the referee has attitude variety (SYSID.md).
    phases = [
        (0.6, 0.8, 0.0, 0.0),
        (0.6, -0.8, 0.0, 0.0),
        (0.4, 0.0, 0.0, 0.0),
        (0.6, 0.0, 0.8, 0.0),
        (0.6, 0.0, -0.8, 0.0),
        (0.4, 0.0, 0.0, 0.0),
        (0.6, 0.0, 0.0, 0.7),
        (0.6, 0.0, 0.0, -0.7),
        (0.4, 0.0, 0.0, 0.0),
    ]
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < PREFLIGHT_S:
        dur, rr, pr, yr = phases[i % len(phases)]
        i += 1
        if not run.hold(dur, thrust=run.hover, roll=rr, pitch=pr, yaw=yr):
            return False
    run.marker("preflight_end")
    return run.hold(1.0, thrust=run.hover)


def maneuver_apex(run, include_zero):
    print("\n=== A. APEX READS ===")
    print("plain acro, levelling OFF, throttle parked before w crosses zero")
    if not run.takeoff():
        return False
    rows = [r for r in APEX if include_zero or r[4] != "tall"]
    for test, peak, hold_s, envelope, _note in rows:
        for rep in range(1, APEX_REPS + 1):
            label = "apex_thr%.2f_rep%d" % (test, rep)
            print("\n-- %s  (clear ~%.0fm floor+ceiling) --" % (label, envelope))
            if not run.level():
                return False
            if not run.hold(SETTLE_S, thrust=run.hover):
                return False
            run.marker(label + "_start")
            if not run.slew_thrust(peak):
                return False
            if not run.slew_thrust(test):
                return False
            # Throttle must be STOPPED here — that is the whole point.
            if not run.hold(hold_s, thrust=test):
                return False
            run.marker(label + "_end")
            if not run.slew_thrust(run.hover):
                return False
            if not run.level():
                return False
            if not run.hold(BETWEEN_MANEUVER_S, thrust=run.hover):
                return False
    print("\nA done (%d markers)." % run.markers)
    return True


def maneuver_terminal(run):
    print("\n=== B. VERTICAL TERMINAL RUNS ===")
    print("needs ~27 m clear up / ~23 m clear down. Abort with F8 if the ceiling bites.")
    if not run.takeoff():
        return False
    if not run.level():
        return False
    run.marker("terminal_up_start")
    if not run.slew_thrust(1.0):
        return False
    # Hold full throttle long enough that a 27 m hangar can show a plateau.
    if not run.hold(4.0, thrust=1.0):
        return False
    run.marker("terminal_up_end")
    if not run.slew_thrust(0.0):
        return False
    run.marker("terminal_down_start")
    if not run.hold(4.0, thrust=0.0):
        return False
    run.marker("terminal_down_end")
    if not run.slew_thrust(run.hover):
        return False
    if not run.level():
        return False
    print("B done. Report whether vertical speed visibly plateaued.")
    return True


def maneuver_manual(run, label, seconds, hint):
    """Hand sticks to the human for C / D. Same session, same recorder."""
    print("\n=== %s (manual, ~%ds) ===" % (label, int(seconds)))
    print(hint)
    print("Keys as in teleop. F12 = marker. F8 = done with this maneuver.\n")
    sticks = t.Sticks()
    if not run.tel.snapshot()["armed"]:
        if not run.takeoff():
            return False
    run.marker(label + "_start")
    end = time.perf_counter() + seconds
    last_loop = time.perf_counter()
    while time.perf_counter() < end:
        now = time.perf_counter()
        dt = min(0.1, now - last_loop)
        last_loop = now
        axes, scale, actions = sticks.poll(dt)
        for action in actions:
            if action == "marker":
                run.marker(label + "_manual")
            elif action == "hover":
                run.thrust = run.hover
            elif action == "quit":
                print("\n[manual] ended early")
                run.marker(label + "_end")
                return True
            elif action == "reset":
                run.pilot.sim_reset()
                run.thrust = t.THRUST_START
                print("\n[sim reset]")
            elif action == "arm":
                run.pilot.arm(True)
            elif action == "disarm":
                run.pilot.arm(False)
        # Drive through Pilot.send so stick throttle slew matches teleop.
        snap = run.tel.snapshot()
        run.pilot.send(axes, scale, dt, snap["armed"], heading=0.0,
                       align=sticks.align, accel=snap["accel"],
                       level=False, truth_att=None)
        run.thrust = run.pilot.thrust
        wall = time.time()
        if wall - run._last_hb >= 1.0 / t.HEARTBEAT_HZ:
            run.pilot.heartbeat()
            run._last_hb = wall
        if wall - run._last_ts >= 1.0 / t.TIMESYNC_HZ:
            run.pilot.timesync_request()
            run._last_ts = wall
        run._pace()
    run.marker(label + "_end")
    return True


def parse_args():
    ap = argparse.ArgumentParser(description="TEMPORARY Card 2 sysid flight runner")
    ap.add_argument("--ip", default=t.MAVLINK_IP)
    ap.add_argument("--port", type=int, default=t.MAVLINK_PORT)
    ap.add_argument("--sessions", default=os.path.join(os.path.dirname(__file__),
                                                       "sessions"))
    ap.add_argument("--hover", type=float, default=t.THRUST_HOVER)
    ap.add_argument("--only", choices=["preflight", "A", "B", "C", "D"],
                    help="run a single card section")
    ap.add_argument("--all", action="store_true",
                    help="preflight + A + B + C + D (B still needs --tall)")
    ap.add_argument("--tall", action="store_true",
                    help="hangar is tall enough for B and for apex thr=0.00")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the 20 s referee gate (not recommended)")
    ap.add_argument("--include-zero", action="store_true",
                    help="include apex thr=0.00 (needs ~14 m envelope)")
    return ap.parse_args()


def plan(args):
    if args.only:
        return [args.only]
    steps = []
    if not args.skip_preflight:
        steps.append("preflight")
    steps.append("A")
    if args.all and args.tall:
        steps.append("B")
    elif args.all and not args.tall:
        print("note: --all without --tall skips B (hangar height unknown)")
    if args.all:
        steps.extend(["C", "D"])
    return steps


def main():
    args = parse_args()
    steps = plan(args)
    include_zero = args.include_zero or args.tall
    print("Card 2 plan: %s" % " -> ".join(steps))
    print("F8 aborts. Levelling assist stays OFF for A/B.\n")

    conn, boot_ms = connect(args)

    # Preflight is its own session so the referee sees a closed, flushed recording.
    if "preflight" in steps:
        run = open_session(args, conn, boot_ms, save_frames=False)
        ok = False
        try:
            ok = maneuver_preflight(run)
        except KeyboardInterrupt:
            pass
        finally:
            run.cut()
            session = close_session(run, "preflight")
        if not ok:
            sys.exit("preflight aborted")
        if not run_referee(session):
            sys.exit("PREFLIGHT FAIL — do not fly the rest. See SYSID.md Card 2.")
        print("PREFLIGHT PASS\n")
        steps = [s for s in steps if s != "preflight"]
        if not steps:
            return

    # A/B share a no-frames session; C wants frames — split if both are planned.
    ab = [s for s in steps if s in ("A", "B")]
    rest = [s for s in steps if s in ("C", "D")]

    if ab:
        run = open_session(args, conn, boot_ms, save_frames=False)
        ok = True
        try:
            for step in ab:
                if step == "A":
                    ok = maneuver_apex(run, include_zero=include_zero)
                elif step == "B":
                    ok = maneuver_terminal(run)
                if not ok:
                    break
        except KeyboardInterrupt:
            ok = False
        finally:
            run.cut()
            close_session(run, "AB")
        if not ok:
            sys.exit("A/B aborted")

    for step in rest:
        save_frames = (step == "C")
        run = open_session(args, conn, boot_ms, save_frames=save_frames)
        ok = True
        try:
            if step == "C":
                ok = maneuver_manual(
                    run, "lap", 120.0,
                    "Fly a conservative finished lap. Collision/reset -> F12 and retry.\n"
                    "JPEGs are being kept.")
            elif step == "D":
                ok = maneuver_manual(
                    run, "refly_130744", 30.0,
                    "Re-fly ~20 s of early free flight up to ~30 m/s (session 130744).\n"
                    "Then re-run: python3 pilot/control/sysid_frames.py <this session>")
        except KeyboardInterrupt:
            ok = False
        finally:
            run.cut()
            session = close_session(run, step)
        if not ok:
            sys.exit("%s aborted" % step)
        if step == "D":
            run_referee(session)

    print("\nCard 2 script finished. Next: refit (SYSID.md §After).")


if __name__ == "__main__":
    main()
