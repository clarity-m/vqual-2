"""
Probe: does the sim honour an attitude quaternion in SET_ATTITUDE_TARGET?

If it does, "angle mode" is available: the sim's flight controller would close the
attitude loop with its own attitude estimate - exactly the information VQ2 withholds
- letting us command absolute bank angles without knowing our own attitude.

METHOD - stays on the ground, and tests SIGN, not magnitude
The FC runs its loop whether or not the drone is airborne, and ACTUATOR_OUTPUT_STATUS
publishes all four motors. A commanded roll appears as differential thrust between the
left and right pairs. We arm, hold thrust well below hover so the drone never lifts,
and read the motors.

    Motor order (vendor example): actuator[0..3] = FL, FR, BL, BR
    Roll asymmetry := (FL + BL) - (FR + BR)      positive = left pair harder

An earlier version of this probe compared the asymmetry under an attitude command
against a quiescent no-command baseline, and "found" the quaternion was honoured.
That was wrong. Commanding a roll RATE tips the drone on the ground; afterwards the
rate loop fights the rocking and produces asymmetry of the same size with NO command
at all (measured: |asym| 0.45 while commanding zero). Magnitude above a quiescent
floor therefore proves nothing.

So this version tests the one thing a disturbance cannot fake: does the asymmetry
follow the SIGN of what we asked for? Each trial commands +A and -A and pairs them.
A disturbance is indifferent to which way we asked the drone to bank; a working
attitude controller is not.

  positive control  +/- roll RATE      must show a clean sign split, otherwise the
                                       measurement channel cannot see sign at all
                                       and the real test is uninterpretable
  test              +/- roll ANGLE via quaternion, ATTITUDE_IGNORE cleared

Every trial begins with MAVLINK_CMD_SIM_RESET, which restores the drone to its pad
exactly - verified reproducible to the last decimal of the resting accelerometer
reading, (-2.9993, 0.0014, -9.3403). Waiting for the airframe to "settle" instead
never converged, because a disturbed drone on the ground does not go quiet while
armed. Identical initial conditions per trial is what makes the paired comparison
mean anything.

Conditions are interleaved (+rate, -rate, +angle, -angle, repeat) so that any drift
in the sim over the run cannot masquerade as a condition effect.

VERDICT RULE, fixed before running:
  * positive control must split by sign with every paired difference agreeing, else
    INCONCLUSIVE.
  * quaternion HONOURED iff the +A and -A means are separated in the same direction
    as the control, and every paired difference agrees in sign.
  * otherwise IGNORED.
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

ATTITUDE_TARGET_TYPEMASK_DCL_BODY_RATES_RADS = 16
ACRO_MASK = (mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
             | ATTITUDE_TARGET_TYPEMASK_DCL_BODY_RATES_RADS)
# ATTITUDE_IGNORE cleared -> quaternion active. Body-rate fields are sent as zero
# and stay active, so if the quaternion is ignored the FC simply holds zero rate,
# which is exactly the null we are testing against.
ANGLE_MASK = ATTITUDE_TARGET_TYPEMASK_DCL_BODY_RATES_RADS

SEND_HZ = 50
PROBE_THRUST = 0.20        # below any plausible hover point
ROLL_RATE_TEST = 1.0       # rad/s, positive control
ROLL_ANGLE_TEST = 20.0     # degrees, the actual test
REPS = 3                   # repetitions of the interleaved condition block

MAVLINK_CMD_SIM_RESET = 31000
# The reset is not instantaneous. Detecting "ready" from pose alone does NOT work:
# the drone never leaves the pad during these probes, so pad-pose + still + idle is
# already true the instant the reset is sent, and the check passes on the OLD state
# before the reset has taken effect (observed live: ~1350 ms early, worse than the
# fixed wait it replaced). So we first wait for a discrete event proving the reset
# landed - race_start_boot_time_ms changing in the race-status packet - and only
# then look at pose.
RESET_EVENT_WAIT_S = 6.0    # wait for race_start to change
RESET_MIN_WAIT_S = 3.0      # floor, even if the event never arrives
RESET_MAX_WAIT_S = 10.0
RESET_STABLE_HOLD_S = 0.5   # pose conditions must hold this long after the event
RESET_GUARD_S = 0.5         # extra margin before arming
IDLE_MOTOR = 0.05
IDLE_TOL = 0.02
STILL_GYRO = 0.05

MEASURE_S = 2.0
MEASURE_SKIP_S = 0.6        # drop the leading transient from the mean
# Canonical resting accel on the pad; a trial starting far from this did not reset
# cleanly and is flagged rather than silently averaged in.
PAD_ACCEL = (-2.9993, 0.0014, -9.3403)
PAD_TOL = 0.35


class Listener:
    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
        self.motors = (0.0, 0.0, 0.0, 0.0)
        self.gyro = (0.0, 0.0, 0.0)
        self.accel = None
        self.samples = []
        self.t0 = time.time()
        self.collecting = False
        self.race_start = None
        self.active_gate = -1
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
            elif t == "ENCAPSULATED_DATA":
                payload = bytes(msg.data)
                if payload and payload[0] == 1:
                    (_i, _boot, race_start, _fin, gate,
                     _last) = struct.unpack_from("<BQqqIq", payload)
                    with self.lock:
                        self.race_start = race_start
                        self.active_gate = gate

    @staticmethod
    def asym(m):
        return (m[0] + m[2]) - (m[1] + m[3])

    def state(self):
        with self.lock:
            g = math.sqrt(sum(x * x for x in self.gyro))
            return g, abs(self.asym(self.motors))

    def start(self):
        with self.lock:
            self.samples = []
            self.t0 = time.time()
            self.collecting = True

    def finish(self):
        with self.lock:
            self.collecting = False
            return [(t, self.asym(m)) for t, m in self.samples]

    def stop(self):
        self._running = False


class Sender:
    def __init__(self, conn, boot_ms):
        self.conn = conn
        self.boot_ms = boot_ms

    def _t(self):
        return int(time.time() * 1000) - self.boot_ms

    def rates(self, roll, thrust):
        self.conn.mav.set_attitude_target_send(
            self._t(), self.conn.target_system, self.conn.target_component,
            ACRO_MASK, [1.0, 0.0, 0.0, 0.0], roll, 0.0, 0.0, thrust)

    def attitude(self, deg, thrust):
        h = math.radians(deg) / 2.0
        q = [math.cos(h), math.sin(h), 0.0, 0.0]
        self.conn.mav.set_attitude_target_send(
            self._t(), self.conn.target_system, self.conn.target_component,
            ANGLE_MASK, q, 0.0, 0.0, 0.0, thrust)

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
    """Is the drone actually back on its pad and quiet?"""
    with listener.lock:
        accel, gyro, motors = listener.accel, listener.gyro, listener.motors
    if accel is None:
        return False
    on_pad = all(abs(accel[i] - PAD_ACCEL[i]) < PAD_TOL for i in range(3))
    still = math.sqrt(sum(x * x for x in gyro)) < STILL_GYRO
    idle = all(abs(m - IDLE_MOTOR) < IDLE_TOL for m in motors)
    return on_pad and still and idle


def reset_to_pad(sender, listener):
    """Put the drone back on its pad, wait until it is genuinely ready, arm."""
    for _ in range(10):
        sender.rates(0.0, 0.0)
        time.sleep(0.02)
    sender.arm(False)
    time.sleep(0.2)

    with listener.lock:
        race_start_before = listener.race_start
    sender.sim_reset()
    t0 = time.time()

    # 1) wait for proof the reset actually landed
    saw_event = False
    while time.time() - t0 < RESET_EVENT_WAIT_S:
        time.sleep(0.02)
        with listener.lock:
            now_start = listener.race_start
        if race_start_before is None or now_start != race_start_before:
            saw_event = True
            break
    # 2) floor the wait regardless, then require a stable pad pose
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

    with listener.lock:
        accel = listener.accel
    sender.arm(True)
    time.sleep(0.4)
    return ready, accel


def measure(send_fn, listener):
    """Returns (mean after the leading transient, mean over everything, n_used)."""
    listener.start()
    t_end = time.time() + MEASURE_S
    while time.time() < t_end:
        send_fn()
        time.sleep(1.0 / SEND_HZ)
    samples = listener.finish()
    if not samples:
        return None, None, 0
    late = [a for t, a in samples if t >= MEASURE_SKIP_S]
    allv = [a for _, a in samples]
    return (statistics.fmean(late) if late else None,
            statistics.fmean(allv), len(late))


def one_trial(label, cmd_fn, amount, sign, sender, listener, thrust):
    ready, accel0 = reset_to_pad(sender, listener)
    val, val_all, n = measure(lambda: cmd_fn(sign * amount, thrust), listener)
    with listener.lock:
        accel1 = listener.accel
    # Secondary, independent observable: did the airframe actually tilt? Roll is
    # recoverable from gravity even though VQ2 publishes no attitude.
    roll0 = math.degrees(math.atan2(-accel0[1], -accel0[2])) if accel0 else None
    roll1 = math.degrees(math.atan2(-accel1[1], -accel1[2])) if accel1 else None
    d_roll = (roll1 - roll0) if (roll0 is not None and roll1 is not None) else None
    print("    %-14s %+d : asym %+.4f (all %+.4f, n=%d)   d_roll %s%s"
          % (label, sign, val, val_all, n,
             "%+.2f deg" % d_roll if d_roll is not None else "n/a",
             "" if ready else "   [WARN: never reached pad-ready state]"))
    return {"asym": val, "asym_all": val_all, "d_roll": d_roll,
            "ready": ready, "accel0": accel0}


def analyse(name, pos, neg, log):
    res = {
        "pos": [t["asym"] for t in pos], "neg": [t["asym"] for t in neg],
        "pos_droll": [t["d_roll"] for t in pos],
        "neg_droll": [t["d_roll"] for t in neg],
        "pos_mean": statistics.fmean([t["asym"] for t in pos]) if pos else None,
        "neg_mean": statistics.fmean([t["asym"] for t in neg]) if neg else None,
    }
    n = min(len(pos), len(neg))
    if n:
        diffs = [pos[i]["asym"] - neg[i]["asym"] for i in range(n)]
        res["paired_diffs"] = diffs
        res["all_agree"] = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
        res["diff_mean"] = statistics.fmean(diffs)
        rd = [(pos[i]["d_roll"], neg[i]["d_roll"]) for i in range(n)
              if pos[i]["d_roll"] is not None and neg[i]["d_roll"] is not None]
        if rd:
            res["droll_diff_mean"] = statistics.fmean([a - b for a, b in rd])
            res["droll_all_agree"] = (all(a - b > 0 for a, b in rd)
                                      or all(a - b < 0 for a, b in rd))
        print("  %s: +mean %+.4f  -mean %+.4f  paired diff %+.4f  signs agree %s"
              % (name, res["pos_mean"], res["neg_mean"], res["diff_mean"],
                 res["all_agree"]))
        if "droll_diff_mean" in res:
            print("      tilt cross-check: d_roll diff %+.3f deg, agree %s"
                  % (res["droll_diff_mean"], res["droll_all_agree"]))
    log[name] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--thrust", type=float, default=PROBE_THRUST)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "probe_angle_result.json"))
    ap.add_argument("--sweep", action="store_true",
                    help="sweep commanded bank angle and check the achieved angle "
                         "tracks it - distinguishes a real attitude controller from "
                         "the drone simply tipping onto a mechanical stop")
    args = ap.parse_args()

    print("Connecting to udpin:%s:%d ..." % (args.ip, args.port))
    conn = mavutil.mavlink_connection("udpin:%s:%d" % (args.ip, args.port))
    conn.wait_heartbeat()
    print("Connected to system %d." % conn.target_system)

    listener = Listener(conn)
    sender = Sender(conn, int(time.time() * 1000))
    log = {"thrust": args.thrust, "reps": REPS,
           "roll_rate_test": ROLL_RATE_TEST, "roll_angle_test": ROLL_ANGLE_TEST,
           "acro_mask": ACRO_MASK, "angle_mask": ANGLE_MASK,
           "measure_s": MEASURE_S, "reset_per_trial": True}

    time.sleep(1.0)
    print("resting accel: %s\n" % (listener.accel,))

    if args.sweep:
        angles = [5.0, 10.0, 20.0, 30.0]
        print("  sweeping commanded bank angle: %s deg (both signs)\n" % angles)
        rows = []
        try:
            for a in angles:
                for sign in (+1, -1):
                    t = one_trial("angle %g" % (sign * a), sender.attitude, a,
                                  sign, sender, listener, args.thrust)
                    rows.append({"commanded": sign * a, "achieved": t["d_roll"],
                                 "asym": t["asym"], "ready": t["ready"]})
        finally:
            for _ in range(20):
                sender.rates(0.0, 0.0)
                time.sleep(0.02)
            sender.arm(False)
            time.sleep(0.3)
            listener.stop()

        print("\n  commanded -> achieved")
        good = [r for r in rows if r["achieved"] is not None]
        for r in good:
            print("    %+6.1f deg -> %+7.2f deg   (error %+.2f)"
                  % (r["commanded"], r["achieved"],
                     r["achieved"] - r["commanded"]))
        log["sweep"] = rows
        if len(good) >= 3:
            xs = [r["commanded"] for r in good]
            ys = [r["achieved"] for r in good]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
            rms = math.sqrt(statistics.fmean([r * r for r in resid]))
            log["sweep_slope"], log["sweep_rms"] = slope, rms
            print("\n  fit: achieved = %.3f x commanded  (rms residual %.2f deg)"
                  % (slope, rms))
            if slope > 0.7 and rms < 4.0:
                print("  => achieved angle TRACKS the setpoint: a real attitude "
                      "controller, not a mechanical stop.")
            else:
                print("  => does NOT track the setpoint; the earlier result is "
                      "suspect.")
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print("\nwrote %s" % args.out)
        return

    rate_pos, rate_neg, att_pos, att_neg = [], [], [], []
    try:
        for rep in range(REPS):
            print("  --- rep %d ---" % rep)
            rate_pos.append(one_trial("rate", sender.rates, ROLL_RATE_TEST, +1,
                                      sender, listener, args.thrust))
            rate_neg.append(one_trial("rate", sender.rates, ROLL_RATE_TEST, -1,
                                      sender, listener, args.thrust))
            att_pos.append(one_trial("attitude", sender.attitude,
                                     ROLL_ANGLE_TEST, +1, sender, listener,
                                     args.thrust))
            att_neg.append(one_trial("attitude", sender.attitude,
                                     ROLL_ANGLE_TEST, -1, sender, listener,
                                     args.thrust))
        print()
        control = analyse("rate_control", rate_pos, rate_neg, log)
        test = analyse("attitude_test", att_pos, att_neg, log)
    finally:
        print("\n  cutting and disarming ...")
        for _ in range(20):
            sender.rates(0.0, 0.0)
            time.sleep(0.02)
        sender.arm(False)
        time.sleep(0.3)
        listener.stop()

    print("\n" + "=" * 72)
    verdict, reason = "INCONCLUSIVE", ""
    if not control.get("paired_diffs") or not control.get("all_agree"):
        reason = ("positive control failed: commanding +/- roll RATE did not "
                  "produce a consistent sign split, so this measurement cannot "
                  "detect sign at all and the attitude result means nothing")
    elif not test.get("paired_diffs"):
        reason = "attitude condition produced no usable trials"
    elif not test["all_agree"]:
        verdict = "QUATERNION IGNORED"
        reason = ("attitude commands did not split by sign (paired diffs %s) while "
                  "the rate control did (%+.4f) - the asymmetry seen under an "
                  "attitude command is disturbance, not response"
                  % (["%+.3f" % d for d in test["paired_diffs"]],
                     control["diff_mean"]))
    else:
        same_dir = (test["diff_mean"] > 0) == (control["diff_mean"] > 0)
        verdict = "QUATERNION HONOURED" if same_dir else "QUATERNION HONOURED (INVERTED)"
        reason = ("attitude commands split by sign, every pair agreeing: diff "
                  "%+.4f against rate-control diff %+.4f"
                  % (test["diff_mean"], control["diff_mean"]))
    log["verdict"], log["reason"] = verdict, reason
    print("VERDICT: %s\n   %s" % (verdict, reason))
    print("=" * 72)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
