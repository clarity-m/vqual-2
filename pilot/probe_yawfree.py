"""
Probe: can we command attitude (roll/pitch) while leaving YAW as a body rate?

Why this matters. The sim honours an attitude quaternion (see probe_angle.py), which
would give self-levelling for free - the FC closes the attitude loop with its own
estimate. But a quaternion also encodes ABSOLUTE yaw, and commanding absolute yaw is
equivalent to reading it: we would know our world heading, which VQ2 deliberately
withholds. That was ruled out on spirit-of-the-spec grounds (see NOTES.md, DECISION
2026-07-30).

The variant that leaks nothing is attitude for roll/pitch with yaw commanded as a
body RATE, because a rate carries no absolute information. If the sim honours that, we
get self-levelling honestly. If not, rate control stays the only path.

NOTE ON WHAT THIS PROBE ITSELF DOES: it necessarily sends quaternions, which contain a
yaw component, in order to find out how the FC treats them. That is characterisation,
not use - nothing here feeds a competitive run, and the outcome we are hoping for is
precisely the mode in which the quaternion's yaw is NOT authoritative.

METHOD
Same shape as probe_angle.py, which this borrows its reset protocol from: stay on the
ground with thrust below hover, reset to the pad before every trial so conditions are
identical, interleave conditions, and test the SIGN of the response rather than its
magnitude - a disturbance cannot fake a sign that follows the command.

Read the YAW row of the mixer. On an X quad the counter-rotating diagonals are
FL+BR against FR+BL, so:

    yaw asymmetry := (FL + BR) - (FR + BL)          actuator[0..3] = FL,FR,BL,BR

Independent second observable: zgyro, which measures body yaw rate directly. Ground
friction may prevent actual rotation, in which case zgyro stays ~0 and the motor
channel carries the answer alone - that is why the mixer metric is primary here.

If the FC is simultaneously fighting to hold the quaternion's absolute yaw, that torque
is COMMON MODE - identical in the +rate and -rate trials - so paired differencing
removes it. That is the whole reason this design can work on a drone whose yaw the FC
may be actively holding.

CONDITIONS (each run at +yaw_rate and -yaw_rate, paired)
  control  ACRO mask 144 (attitude ignored)     - proves the metric can see yaw at all
  test_A   mask 16  (attitude active, all body rates active)
  test_B   mask 19  (attitude active, roll+pitch rates ignored, yaw rate active)
           = the canonical PX4-style "attitude plus yaw rate"

VERDICT RULE, fixed before running:
  * control must split by sign with every pair agreeing, else INCONCLUSIVE.
  * a test condition HONOURS the yaw rate iff its paired differences all agree in sign
    and point the same way as the control.
  * if neither test splits, yaw is locked to the quaternion and self-levelling is
    unavailable to us honestly -> stay rate-only.
"""

import argparse
import json
import math
import os
import statistics
import struct
import threading
import time

from pymavlink import mavutil

DCL_RADS_BIT = 16
ACRO_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE | DCL_RADS_BIT
ANGLE_ALL_MASK = DCL_RADS_BIT                      # attitude active, all rates active
ANGLE_YAWRATE_MASK = (DCL_RADS_BIT
                      | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
                      | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE)

SEND_HZ = 50
PROBE_THRUST = 0.20
YAW_RATE_TEST = 1.5        # rad/s
REPS = 3

MAVLINK_CMD_SIM_RESET = 31000
RESET_EVENT_WAIT_S = 6.0
RESET_MIN_WAIT_S = 3.0
RESET_MAX_WAIT_S = 10.0
RESET_STABLE_HOLD_S = 0.5
RESET_GUARD_S = 0.5
IDLE_MOTOR = 0.05
IDLE_TOL = 0.02
STILL_GYRO = 0.05
PAD_ACCEL = (-2.9993, 0.0014, -9.3403)
PAD_TOL = 0.35

MEASURE_S = 2.0
MEASURE_SKIP_S = 0.6


class Listener:
    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
        self.motors = (0.0, 0.0, 0.0, 0.0)
        self.gyro = (0.0, 0.0, 0.0)
        self.accel = None
        self.race_start = None
        self.samples = []
        self.gyro_samples = []
        self.t0 = time.time()
        self.collecting = False
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                msg = self.conn.recv_match(blocking=False)
            except Exception:
                continue
            if msg is None:
                time.sleep(0.001)
                continue
            t = msg.get_type()
            if t == "ACTUATOR_OUTPUT_STATUS":
                m = (msg.actuator[0], msg.actuator[1],
                     msg.actuator[2], msg.actuator[3])
                with self.lock:
                    self.motors = m
                    if self.collecting:
                        self.samples.append((time.time() - self.t0, m))
            elif t == "HIGHRES_IMU":
                with self.lock:
                    self.gyro = (msg.xgyro, msg.ygyro, msg.zgyro)
                    self.accel = (msg.xacc, msg.yacc, msg.zacc)
                    if self.collecting:
                        self.gyro_samples.append(
                            (time.time() - self.t0, msg.zgyro))
            elif t == "ENCAPSULATED_DATA":
                payload = bytes(msg.data)
                if payload and payload[0] == 1:
                    (_i, _b, race_start, _f, _g,
                     _l) = struct.unpack_from("<BQqqIq", payload)
                    with self.lock:
                        self.race_start = race_start

    @staticmethod
    def yaw_asym(m):
        """Diagonal split: the yaw row of an X-quad mixer."""
        return (m[0] + m[3]) - (m[1] + m[2])

    def start(self):
        with self.lock:
            self.samples = []
            self.gyro_samples = []
            self.t0 = time.time()
            self.collecting = True

    def finish(self):
        with self.lock:
            self.collecting = False
            return list(self.samples), list(self.gyro_samples)

    def stop(self):
        self._running = False


class Sender:
    def __init__(self, conn, boot_ms):
        self.conn = conn
        self.boot_ms = boot_ms

    def _t(self):
        return int(time.time() * 1000) - self.boot_ms

    def send(self, mask, q, yaw_rate, thrust):
        self.conn.mav.set_attitude_target_send(
            self._t(), self.conn.target_system, self.conn.target_component,
            mask, q, 0.0, 0.0, yaw_rate, thrust)

    def arm(self, on):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1 if on else 0, 0, 0, 0, 0, 0, 0)

    def sim_reset(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0)


def _ready(listener):
    with listener.lock:
        accel, gyro, motors = listener.accel, listener.gyro, listener.motors
    if accel is None:
        return False
    on_pad = all(abs(accel[i] - PAD_ACCEL[i]) < PAD_TOL for i in range(3))
    still = math.sqrt(sum(x * x for x in gyro)) < STILL_GYRO
    idle = all(abs(m - IDLE_MOTOR) < IDLE_TOL for m in motors)
    return on_pad and still and idle


def reset_to_pad(sender, listener):
    """Reset and wait for it to actually land. See probe_angle.py for why the
    discrete event matters: pose alone is already true before the reset takes."""
    for _ in range(10):
        sender.send(ACRO_MASK, [1.0, 0.0, 0.0, 0.0], 0.0, 0.0)
        time.sleep(0.02)
    sender.arm(False)
    time.sleep(0.2)

    with listener.lock:
        before = listener.race_start
    sender.sim_reset()
    t0 = time.time()

    saw_event = False
    while time.time() - t0 < RESET_EVENT_WAIT_S:
        time.sleep(0.02)
        with listener.lock:
            now = listener.race_start
        if before is None or now != before:
            saw_event = True
            break
    while time.time() - t0 < RESET_MIN_WAIT_S:
        time.sleep(0.02)
    stable_since = None
    while time.time() - t0 < RESET_MAX_WAIT_S:
        time.sleep(0.02)
        if _ready(listener):
            stable_since = stable_since or time.time()
            if time.time() - stable_since >= RESET_STABLE_HOLD_S:
                break
        else:
            stable_since = None
    ready = stable_since is not None and saw_event
    time.sleep(RESET_GUARD_S)
    sender.arm(True)
    time.sleep(0.4)
    return ready


def one_trial(label, mask, sign, sender, listener, thrust):
    ready = reset_to_pad(sender, listener)
    q = [1.0, 0.0, 0.0, 0.0]          # level; its yaw component is what is under test
    listener.start()
    t_end = time.time() + MEASURE_S
    while time.time() < t_end:
        sender.send(mask, q, sign * YAW_RATE_TEST, thrust)
        time.sleep(1.0 / SEND_HZ)
    motors, gyros = listener.finish()

    late = [Listener.yaw_asym(m) for t, m in motors if t >= MEASURE_SKIP_S]
    gz = [g for t, g in gyros if t >= MEASURE_SKIP_S]
    asym = statistics.fmean(late) if late else None
    zg = statistics.fmean(gz) if gz else None
    print("    %-16s %+d : yaw_asym %+.4f   zgyro %s%s"
          % (label, sign, asym if asym is not None else float("nan"),
             "%+.3f rad/s" % zg if zg is not None else "n/a",
             "" if ready else "   [WARN: not pad-ready]"))
    return {"asym": asym, "zgyro": zg, "ready": ready}


def analyse(name, pos, neg, log):
    n = min(len(pos), len(neg))
    diffs = [pos[i]["asym"] - neg[i]["asym"] for i in range(n)]
    gd = [pos[i]["zgyro"] - neg[i]["zgyro"] for i in range(n)
          if pos[i]["zgyro"] is not None and neg[i]["zgyro"] is not None]
    res = {
        "pos": [t["asym"] for t in pos], "neg": [t["asym"] for t in neg],
        "pos_gz": [t["zgyro"] for t in pos], "neg_gz": [t["zgyro"] for t in neg],
        "paired_diffs": diffs,
        "diff_mean": statistics.fmean(diffs) if diffs else None,
        "all_agree": bool(diffs) and (all(d > 0 for d in diffs)
                                      or all(d < 0 for d in diffs)),
        "gz_diff_mean": statistics.fmean(gd) if gd else None,
        "gz_all_agree": bool(gd) and (all(d > 0 for d in gd)
                                      or all(d < 0 for d in gd)),
    }
    print("  %-22s paired diff %+.4f  signs agree %s   | zgyro diff %s agree %s"
          % (name, res["diff_mean"], res["all_agree"],
             "%+.3f" % res["gz_diff_mean"] if res["gz_diff_mean"] is not None
             else "n/a", res["gz_all_agree"]))
    log[name] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--thrust", type=float, default=PROBE_THRUST)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "evidence",
        "2026-07-30-yawfree.json"))
    args = ap.parse_args()

    print("Connecting to udpin:%s:%d ..." % (args.ip, args.port))
    conn = mavutil.mavlink_connection("udpin:%s:%d" % (args.ip, args.port))
    conn.wait_heartbeat()
    print("Connected to system %d.\n" % conn.target_system)

    listener = Listener(conn)
    sender = Sender(conn, int(time.time() * 1000))
    log = {"yaw_rate": YAW_RATE_TEST, "thrust": args.thrust, "reps": REPS,
           "acro_mask": ACRO_MASK, "angle_all_mask": ANGLE_ALL_MASK,
           "angle_yawrate_mask": ANGLE_YAWRATE_MASK}
    print("masks: acro=%d  attitude+allrates=%d  attitude+yawrate=%d"
          % (ACRO_MASK, ANGLE_ALL_MASK, ANGLE_YAWRATE_MASK))
    time.sleep(1.0)

    conds = [("control", ACRO_MASK), ("test_A_all", ANGLE_ALL_MASK),
             ("test_B_yawrate", ANGLE_YAWRATE_MASK)]
    results = {name: {"pos": [], "neg": []} for name, _ in conds}

    try:
        for rep in range(REPS):
            print("  --- rep %d ---" % rep)
            for name, mask in conds:
                for sign, key in ((+1, "pos"), (-1, "neg")):
                    results[name][key].append(
                        one_trial(name, mask, sign, sender, listener,
                                  args.thrust))
        print()
        analysed = {name: analyse(name, results[name]["pos"],
                                  results[name]["neg"], log)
                    for name, _ in conds}
    finally:
        print("\n  cutting and disarming ...")
        for _ in range(20):
            sender.send(ACRO_MASK, [1.0, 0.0, 0.0, 0.0], 0.0, 0.0)
            time.sleep(0.02)
        sender.arm(False)
        time.sleep(0.3)
        listener.stop()

    print("\n" + "=" * 72)
    ctl = analysed["control"]
    if not ctl["all_agree"]:
        verdict = "INCONCLUSIVE"
        reason = ("positive control failed: commanding +/- yaw RATE in acro did not "
                  "split the yaw mixer channel by sign, so this metric cannot see "
                  "yaw and the test conditions mean nothing")
    else:
        same = lambda r: r["all_agree"] and ((r["diff_mean"] > 0)
                                             == (ctl["diff_mean"] > 0))
        a, b = analysed["test_A_all"], analysed["test_B_yawrate"]
        if same(a) or same(b):
            verdict = "YAW RATE HONOURED ALONGSIDE ATTITUDE"
            reason = ("attitude active and the yaw rate still commands the yaw "
                      "channel (A %+.4f agree %s | B %+.4f agree %s) against a "
                      "control of %+.4f. Self-levelling is available without "
                      "commanding absolute yaw."
                      % (a["diff_mean"], a["all_agree"], b["diff_mean"],
                         b["all_agree"], ctl["diff_mean"]))
        else:
            verdict = "YAW LOCKED TO THE QUATERNION"
            reason = ("with attitude active the commanded yaw rate had no "
                      "sign-consistent effect (A %+.4f agree %s | B %+.4f agree %s) "
                      "while acro did (%+.4f). Using attitude would mean commanding "
                      "absolute yaw, which is ruled out - stay rate-only."
                      % (a["diff_mean"], a["all_agree"], b["diff_mean"],
                         b["all_agree"], ctl["diff_mean"]))
    log["verdict"], log["reason"] = verdict, reason
    print("VERDICT: %s\n   %s" % (verdict, reason))
    print("=" * 72)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
