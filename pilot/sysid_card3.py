"""TEMPORARY — Card 3. The three gaps card 2 left in the plant. Delete once flown.

    python3 pilot/sysid_card3.py                 # preflight session, then E
    python3 pilot/sysid_card3.py --only E        # trough reads, throttle 0.50..0.80
    python3 pilot/sysid_card3.py --only F        # wide apex arcs at 0.15 / 0.20
    python3 pilot/sysid_card3.py --only G        # body-lift sweeps (own session)
    python3 pilot/sysid_card3.py --all --tall --long

Reuses card 2's link, recorder, `Runner` and preflight referee rather than re-deriving
any of it — the sign conventions are settled there and must not be restated here.

VQ1 build, Training flight, nothing else on UDP 14550. F8 aborts.

## What each section is for

**E — trough reads (throttle 0.50..0.82).** The thrust table rests on ~240 samples
between 0.50 and 0.82, which is exactly the band a policy accelerates through. The
obvious fix — hold 0.60 for half a minute and read the plateau — is not flyable: `T/m`
at 0.60 is about 33 m/s^2 against gravity's 9.81, so the drone climbs at 23 m/s^2 and
is through any hangar ceiling in two seconds.

So E inverts card 2's apex trick. Dive at low throttle to build downward speed, park the
throttle at the *high* test value, and let the drone decelerate through `w = 0`. At that
crossing `kz*w|w|` vanishes identically and `-a_z` is a direct read of `T/m`, with no
regression and no thrust/drag trade-off — the same argument as the apex arcs, run in the
other direction. Each rep costs a few seconds and a bounded envelope instead of a
runaway climb. The throttle must be STOPPED before the crossing: `sysid_apex.py` accepts
a read only where `|d(thr)/dt|` is zero, so the slew has to finish first, which is what
`lead_s` below is for.

**F — apex arcs that actually span 9 m/s.** Card 2 flew arcs at 0.15 and 0.20 but they
coasted through less than 9 m/s of `w`, and `sysid_apex.py` rejects them on that span
(`MIN_W_SPAN`) — correctly, since it excludes on the span rather than on the answer,
which is the only non-circular ordering. Only 4 of 8 arcs survived. The fix is a higher
entry peak and a longer hold so the arc sweeps a wider band of `w`. At test throttle
0.15 the net is about 5.6 m/s^2, so 2.2 s of hold spans ~12 m/s while the physical
excursion stays near 6 m, because an apex rises and falls back through itself.

**G — body-lift sweeps.** `c_lift` is the term that took held-out body-z R^2 from 0.90
to 0.95, and nothing in the existing corpus was flown to measure it. In racing flight
`u` and `w` covary through every turn, so a fit there cannot separate lift from vertical
drag. G holds a *fixed* pitch attitude and a *fixed* throttle and lets forward speed
settle, at several attitudes. With `w` near zero at equilibrium, `-a_z` reads
`T/m + c*u^2`, and `T/m` is already known from the table at `u = 0`, so the excess is
lift and nothing else. Five attitudes span |u| from ~9 to ~22 m/s, a 6x lever on `u^2`.

G is the one section that closes a loop on attitude, so it runs in its own session and
records a `level_assist` event. `sysid_fit.py` reads that event and keeps the session out
of the rate-loop fit while still using it for forces — the rate loop can only be
identified from open-loop sticks, but the force model does not care where the commands
came from, only that they were recorded, and `cmd.csv` records what actually went out.

Every number in the tables below is an open-loop starting point computed from the plant
under test, valid wherever the maneuver actually lands. Watch the envelope, not the
clock, and abort with F8 if the hangar bites.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import sysid_card2 as c2
import teleop as t

# -- E: trough reads -------------------------------------------------------------
# (test_thrust, dive_thrust, dive_s, lead_s, hold_s, envelope_m)
# `dive_s` builds downward speed at `dive_thrust`; `lead_s` is time at the test
# throttle BEFORE the crossing is expected, so the slew is over and the throttle is
# stopped when `w` passes zero; `hold_s` keeps it parked across the crossing.
TROUGH = [
    (0.55, 0.05, 1.4, 0.9, 2.0, 17.0),
    (0.60, 0.05, 1.4, 0.8, 2.0, 17.0),
    (0.65, 0.05, 1.3, 0.8, 1.8, 17.0),
    (0.70, 0.05, 1.2, 0.7, 1.8, 16.0),
    (0.80, 0.05, 1.1, 0.6, 1.6, 16.0),
]
TROUGH_REPS = 2

# -- F: wide apex arcs -----------------------------------------------------------
# (test_thrust, peak_thrust, hold_s, envelope_m). Higher peak and longer hold than
# card 2's, which is the whole point: card 2's reps were rejected on w span.
APEX_WIDE = [
    (0.15, 0.65, 2.2, 9.0),
    (0.20, 0.65, 2.4, 10.0),
]
APEX_WIDE_REPS = 2

# -- G: body-lift sweeps ---------------------------------------------------------
# (pitch_deg nose-down, thrust, settle_s, hold_s, expected_u, run_m, needs_long)
# thrust is g/cos(pitch) mapped through the current thrust table, so the drone holds
# altitude while it accelerates; expected_u is sqrt(g*tan(pitch)/kx) at that tilt.
LIFT = [
    (20.0, 0.285, 2.5, 3.0,  8.6,  40.0, False),
    (35.0, 0.313, 2.5, 3.0, 11.9,  55.0, False),
    (50.0, 0.372, 3.0, 3.0, 15.5,  75.0, True),
    (60.0, 0.435, 3.0, 3.0, 18.7,  90.0, True),
    (68.0, 0.506, 3.0, 3.0, 22.3, 105.0, True),
]
LIFT_REPS = 2

SETTLE_S = c2.SETTLE_S
BETWEEN_MANEUVER_S = c2.BETWEEN_MANEUVER_S


class Runner(c2.Runner):
    """Card 2's runner plus an attitude hold, which only section G needs."""

    def hold_pitch(self, pitch_deg, thrust, seconds):
        """Hold a nose-down pitch attitude and a fixed throttle for `seconds`.

        The same P loop as `Runner.level`, with a non-zero pitch target instead of
        zero. Roll is still driven to zero: a rolled drone puts speed into body y and
        the whole point here is to put it into body x, where `u^2` is the regressor.

        Yaw is left alone. It costs nothing in path -- that is the free-gimbal property
        the control architecture is built on -- and touching it here would only add a
        term to explain.
        """
        target = math.radians(-abs(pitch_deg))    # nose down is negative pitch in NED
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            if self._check_abort():
                return False
            snap = self.tel.snapshot()
            att = snap["truth_att"]
            roll = pitch = 0.0
            if att is not None and time.time() - att[2] <= t.LEVEL_MAX_AGE:
                err_r = -att[0]
                err_p = target - att[1]
                p_roll = max(-t.LEVEL_MAX_RATE, min(t.LEVEL_MAX_RATE,
                                                    t.LEVEL_GAIN * err_r))
                p_pitch = max(-t.LEVEL_MAX_RATE, min(t.LEVEL_MAX_RATE,
                                                     t.LEVEL_GAIN * err_p))
                roll = p_roll * t.RATE_SIGN_ROLL
                pitch = p_pitch * t.RATE_SIGN_PITCH
            self._send(roll, pitch, 0.0, thrust)
            self._pace()
        return True


def open_session(args, conn, boot_ms, save_frames, control="sysid_card3"):
    """Card 2's session opener, tagged as card 3 so the recording says what it is."""
    rec = t.Recorder(args.sessions, save_frames=save_frames)
    tel = t.Telemetry(conn, rec)
    vision = t.VisionRX(rec, decode=False)
    pilot = t.Pilot(conn, rec, boot_ms, hover=args.hover)
    pilot.auto_takeoff = False
    run = Runner(conn, rec, tel, vision, pilot, args.hover)
    rec.event("session_start", control=control, ip=args.ip, port=args.port,
              save_frames=save_frames)
    deadline = time.time() + 5.0
    while time.time() < deadline and not tel.snapshot()["truth_seen"]:
        time.sleep(0.05)
    if not tel.snapshot()["truth_seen"]:
        print("WARNING: no ATTITUDE/POSITION/ODOMETRY yet — is this the VQ1 build?")
    print("recording to: %s" % rec.dir)
    return run


def maneuver_trough(run, tall):
    """E. Dive, park a high throttle, read T/m where w crosses zero."""
    print("\n=== E. TROUGH READS (throttle 0.50..0.82) ===")
    print("plain acro, levelling OFF. Throttle is STOPPED before the crossing.")
    if not run.takeoff():
        return False
    rows = TROUGH if tall else [r for r in TROUGH if r[5] <= 16.0]
    if len(rows) < len(TROUGH):
        print("note: --tall not set, flying %d of %d rows (envelope)"
              % (len(rows), len(TROUGH)))
    for test, dive, dive_s, lead_s, hold_s, envelope in rows:
        for rep in range(1, TROUGH_REPS + 1):
            label = "trough_thr%.2f_rep%d" % (test, rep)
            print("\n-- %s  (needs ~%.0f m below and above) --" % (label, envelope))
            if not run.level():
                return False
            if not run.hold(SETTLE_S, thrust=run.hover):
                return False
            run.marker(label + "_start")
            # Build downward speed at low throttle.
            if not run.slew_thrust(dive):
                return False
            if not run.hold(dive_s, thrust=dive):
                return False
            # Get the throttle to the test value and STOP it, then coast through w = 0.
            if not run.slew_thrust(test):
                return False
            if not run.hold(lead_s + hold_s, thrust=test):
                return False
            run.marker(label + "_end")
            if not run.slew_thrust(run.hover):
                return False
            if not run.level():
                return False
            if not run.hold(BETWEEN_MANEUVER_S, thrust=run.hover):
                return False
    print("\nE done (%d markers). Report whether each trough visibly bottomed out."
          % run.markers)
    return True


def maneuver_apex_wide(run):
    """F. Card 2's apex, flown wide enough to clear MIN_W_SPAN."""
    print("\n=== F. WIDE APEX ARCS (0.15 / 0.20) ===")
    print("higher peak and longer hold than card 2 — the span is the point")
    if not run.takeoff():
        return False
    for test, peak, hold_s, envelope in APEX_WIDE:
        for rep in range(1, APEX_WIDE_REPS + 1):
            label = "apex_thr%.2f_rep%d" % (test, rep)
            print("\n-- %s  (clear ~%.0f m floor+ceiling) --" % (label, envelope))
            if not run.level():
                return False
            if not run.hold(SETTLE_S, thrust=run.hover):
                return False
            run.marker(label + "_start")
            if not run.slew_thrust(peak):
                return False
            if not run.slew_thrust(test):
                return False
            # Throttle STOPPED here — same rule as card 2.
            if not run.hold(hold_s, thrust=test):
                return False
            run.marker(label + "_end")
            if not run.slew_thrust(run.hover):
                return False
            if not run.level():
                return False
            if not run.hold(BETWEEN_MANEUVER_S, thrust=run.hover):
                return False
    print("\nF done. sysid_apex.py reports the w span each arc achieved.")
    return True


def maneuver_lift(run, long_runs):
    """G. Fixed pitch, fixed throttle, let forward speed settle. Measures c_lift."""
    print("\n=== G. BODY-LIFT SWEEPS ===")
    print("attitude hold ON for this section (recorded as level_assist)")
    run.rec.event("level_assist", on=True)
    if not run.takeoff():
        return False
    rows = LIFT if long_runs else [r for r in LIFT if not r[6]]
    if len(rows) < len(LIFT):
        print("note: --long not set, flying %d of %d rows (run length)"
              % (len(rows), len(LIFT)))
    try:
        for pitch, thrust, settle_s, hold_s, u_exp, run_m, _needs in rows:
            for rep in range(1, LIFT_REPS + 1):
                label = "lift_p%02d_rep%d" % (int(pitch), rep)
                print("\n-- %s  (~%.0f m of clear run, expect |u| ~ %.0f m/s) --"
                      % (label, run_m, u_exp))
                if not run.level():
                    return False
                if not run.hold(SETTLE_S, thrust=run.hover):
                    return False
                # Settle first, unmarked: the marked window must contain only the
                # equilibrium, not the acceleration into it, or the fit reads the
                # transient as lift.
                if not run.hold_pitch(pitch, thrust, settle_s):
                    return False
                run.marker(label + "_start")
                if not run.hold_pitch(pitch, thrust, hold_s):
                    return False
                run.marker(label + "_end")
                if not run.level():
                    return False
                if not run.hold(BETWEEN_MANEUVER_S, thrust=run.hover):
                    return False
    finally:
        run.rec.event("level_assist", on=False)
    print("\nG done. Report the |u| each row actually reached.")
    return True


def parse_args():
    ap = argparse.ArgumentParser(description="TEMPORARY Card 3 sysid flight runner")
    ap.add_argument("--ip", default=t.MAVLINK_IP)
    ap.add_argument("--port", type=int, default=t.MAVLINK_PORT)
    ap.add_argument("--sessions", default=os.path.join(os.path.dirname(__file__),
                                                       "sessions"))
    ap.add_argument("--hover", type=float, default=t.THRUST_HOVER)
    ap.add_argument("--only", choices=["preflight", "E", "F", "G"],
                    help="run a single card section")
    ap.add_argument("--all", action="store_true", help="preflight + E + F, then G")
    ap.add_argument("--tall", action="store_true",
                    help="hangar clears ~17 m each way (all of E)")
    ap.add_argument("--long", action="store_true",
                    help="hangar clears ~105 m of straight run (all of G)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the 20 s referee gate (not recommended)")
    return ap.parse_args()


def plan(args):
    if args.only:
        return [args.only]
    steps = []
    if not args.skip_preflight:
        steps.append("preflight")
    steps.append("E")
    if args.all:
        steps.extend(["F", "G"])
    return steps


def main():
    args = parse_args()
    steps = plan(args)
    print("Card 3 plan: %s" % " -> ".join(steps))
    print("F8 aborts. Levelling assist stays OFF for E and F.\n")

    conn, boot_ms = c2.connect(args)

    # Preflight is its own session so the referee sees a closed, flushed recording.
    if "preflight" in steps:
        run = open_session(args, conn, boot_ms, save_frames=False,
                           control="sysid_card3_preflight")
        ok = False
        try:
            ok = c2.maneuver_preflight(run)
        except KeyboardInterrupt:
            pass
        finally:
            run.cut()
            session = c2.close_session(run, "preflight")
        if not ok:
            sys.exit("preflight aborted")
        if not c2.run_referee(session):
            sys.exit("PREFLIGHT FAIL — do not fly the rest. See SYSID.md.")
        print("PREFLIGHT PASS\n")
        steps = [s for s in steps if s != "preflight"]
        if not steps:
            return

    # E and F are open-loop sticks and share a session. G closes a loop on attitude,
    # so it gets its own -- keeping them apart is what lets the rate loop be fitted
    # from E/F while G still contributes force samples.
    ef = [s for s in steps if s in ("E", "F")]
    if ef:
        run = open_session(args, conn, boot_ms, save_frames=False)
        ok = True
        try:
            for step in ef:
                if step == "E":
                    ok = maneuver_trough(run, args.tall)
                elif step == "F":
                    ok = maneuver_apex_wide(run)
                if not ok:
                    break
        except KeyboardInterrupt:
            ok = False
        finally:
            run.cut()
            c2.close_session(run, "+".join(ef))
        if not ok:
            sys.exit("aborted during %s" % "+".join(ef))

    if "G" in steps:
        run = open_session(args, conn, boot_ms, save_frames=False,
                           control="sysid_card3_G")
        ok = True
        try:
            ok = maneuver_lift(run, args.long)
        except KeyboardInterrupt:
            ok = False
        finally:
            run.cut()
            c2.close_session(run, "G")
        if not ok:
            sys.exit("aborted during G")

    print("\nCard 3 flown. Refit with:")
    print("  $s = (Get-ChildItem pilot\\sessions -Directory).FullName")
    print("  python pilot/control/sysid_frames.py @s")
    print("  python pilot/control/sysid_fit.py")
    print("  python pilot/control/sysid_replay.py")


if __name__ == "__main__":
    main()
