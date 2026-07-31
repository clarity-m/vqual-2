"""What is this simulator actually sending?

Binds UDP 14550, counts every MAVLink message type for a few seconds, prints a
histogram. Sends NOTHING by default -- no heartbeat, no setpoints, no requests.

    python3 pilot/msgscan.py                 # 10 s, passive
    python3 pilot/msgscan.py --secs 20
    python3 pilot/msgscan.py --heartbeat     # also send a GCS heartbeat at 2 Hz

Why --heartbeat exists: some autopilots and sims only stream telemetry once they
have seen a GCS announce itself. If the passive scan is thin and the heartbeat
scan is rich, that is the answer -- and it means a purely passive "listen" mode
cannot observe the full stream.

Nothing can share UDP 14550, so stop teleop before running this.
"""

import argparse
import collections
import time

from pymavlink import mavutil

# Blocked under VQ2 (spec 9.3), present under VQ1. Their absence or presence is
# the whole question this tool exists to answer.
TRUTH = ("ATTITUDE", "LOCAL_POSITION_NED", "ODOMETRY", "ATTITUDE_QUATERNION",
         "GLOBAL_POSITION_INT", "VISION_POSITION_ESTIMATE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--heartbeat", action="store_true",
                    help="send a GCS heartbeat at 2 Hz while scanning")
    args = ap.parse_args()

    print("binding udpin:%s:%d ..." % (args.ip, args.port))
    conn = mavutil.mavlink_connection("udpin:%s:%d" % (args.ip, args.port))
    print("waiting for first packet (start/unpause the sim if this hangs) ...")
    conn.wait_heartbeat()
    print("connected: system %d component %d\n" % (conn.target_system,
                                                   conn.target_component))

    counts = collections.Counter()
    first = {}
    t0 = time.time()
    last_hb = 0.0

    while time.time() - t0 < args.secs:
        if args.heartbeat and time.time() - last_hb >= 0.5:
            conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            last_hb = time.time()

        msg = conn.recv_match(blocking=True, timeout=0.2)
        if msg is None:
            continue
        t = msg.get_type()
        if t == "BAD_DATA":
            continue
        counts[t] += 1
        first.setdefault(t, time.time() - t0)

    span = time.time() - t0
    print("=== %.1f s, heartbeat=%s ===" % (span, args.heartbeat))
    print("%-32s %8s %8s %8s" % ("message", "count", "Hz", "first@s"))
    for t, n in counts.most_common():
        mark = "  <-- TRUTH" if t in TRUTH else ""
        print("%-32s %8d %8.1f %8.1f%s" % (t, n, n / span, first[t], mark))

    seen = [t for t in TRUTH if t in counts]
    print()
    if seen:
        print("GROUND TRUTH PRESENT: %s" % ", ".join(seen))
        print("-> this is the VQ1 sim (or an unrestricted build); recordings")
        print("   from it can referee the estimators.")
    else:
        print("NO GROUND-TRUTH MESSAGES in %.0f s." % span)
        print("-> either this is the VQ2 sim, or the sim only streams pose")
        print("   during an active race. Re-run with a race started, and with")
        print("   --heartbeat, before concluding the build blocks them.")


if __name__ == "__main__":
    main()
