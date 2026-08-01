"""Stage 1.5: direct thrust and vertical-drag reads from the card 2 arcs.

    python3 pilot/control/sysid_apex.py                 # the card 2 sessions
    python3 pilot/control/sysid_apex.py <session> ...

Every other number in this directory comes out of a regression over racing flight, where
throttle and body-z speed move together. These come out of maneuvers built so they do
not, and that is the whole reason the flight was flown (`../SYSID.md` card 2).

**Apex arcs.** Climb, park the throttle low, coast over the top. Throttle is a constant
through the arc, so regressing `-a_z` on `w|w|` *within* one arc reads `kz` with thrust
as a plain intercept -- no thrust model involved and nothing for it to trade against.
Pooling the arcs as a fixed-effects fit (one intercept per arc, one shared `kz`) uses
every sample for `kz` while letting each arc keep its own thrust level. Those intercepts
are then direct reads of `T/m` at throttle 0.05..0.20, where the whole training set held
17 samples.

**Terminal runs.** Full throttle until the climb stops accelerating, then zero throttle
down. At the plateau `T/m = g + kz*v^2`, so with `kz` from the arcs the climb is a direct
read of `T/m` at full throttle -- the other end of the curve, equally unmeasured. The
descent is a second read of `kz` alone, since `T/m` is zero there.

The one thing that invalidates an arc is the throttle still moving when `w` crosses zero,
which re-introduces the correlation the maneuver exists to break. That is checked here
rather than assumed, and an arc that fails it is dropped rather than fitted.

**A short coast reads kz high, and the flown card contains both kinds.** The 0.05 and
0.10 arcs coast through ~10 m/s of `w` and agree on kz to four figures; the 0.15 and 0.20
arcs barely reach apex at all -- the card's own table predicts they recover at 5.3 and
0.8 m/s -- and over that stub of a lever arm the slope reads 0.06 and 0.09. So kz is
taken from the arcs that actually coasted, and the short ones contribute only their
thrust read, which does not depend on the slope. The two terminal descents referee that
choice from a completely different maneuver: thrust is zero there, so `dv/dt = g - kz*v^2`
reads kz with nothing else in it at all.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(HERE, "..", "sessions")
sys.path.insert(0, HERE)

from sysid_data import Session, interp_cols, zoh          # noqa: E402
from sysid_frames import v_body                           # noqa: E402

G = 9.81
THRUST_DELAY = 0.015    # s, measured; matches sysid_fit

CARD2 = ["20260801-004843", "20260801-005059", "20260801-005518"]

MIN_STILL = 1.0         # s the throttle must be parked for an arc to count
MAX_PARK_THROTTLE = 0.25    # an arc is a *low* throttle read; above this it is a hover
MIN_W_SPAN = 9.0        # m/s of `w` an arc must cover before its slope means anything
APEX_W = 1.0            # m/s; |w| under this makes kz*w|w| < 0.05 m/s^2, i.e. negligible
PLATEAU_S = 1.0         # s at the end of a terminal run used for the steady read
PLATEAU_TOL = 0.5       # m/s of remaining acceleration over PLATEAU_S to call it steady


def markers(path, ep):
    """(sim time, label) for every marker that falls inside this epoch."""
    out = []
    ev = os.path.join(path, "events.jsonl")
    if not os.path.exists(ev):
        return out
    t0, t1 = ep.span()
    with open(ev) as fh:
        for line in fh:
            e = json.loads(line)
            if e.get("kind") != "marker":
                continue
            t = float(ep.clock.to_sim(e["t_wall_ns"]))
            if t0 <= t <= t1:
                out.append((t, e["label"]))
    return sorted(out)


def windows(path, ep, suffix_start="_start", suffix_end="_end"):
    """Marker pairs `<stem>_start` .. `<stem>_end` as (stem, t0, t1)."""
    ms = markers(path, ep)
    out = []
    for i, (t0, lab) in enumerate(ms):
        if not lab.endswith(suffix_start):
            continue
        stem = lab[:-len(suffix_start)]
        for t1, lab2 in ms[i + 1:]:
            if lab2 == stem + suffix_end:
                out.append((stem, t0, t1))
                break
    return out


class Arc:
    """One apex coast: throttle parked, `w` sweeping through zero."""

    def __init__(self, name, t, w, fz, throttle, dt):
        self.name = name
        self.t = t
        self.w = w
        self.fz = fz                # -a_z == T/m + kz*w|w|
        self.throttle = throttle
        self.dt = dt

    @property
    def still_s(self):
        return float(self.t[-1] - self.t[0])

    @property
    def crosses(self):
        return bool(self.w.min() < 0 < self.w.max())

    @property
    def w_span(self):
        return float(self.w.max() - self.w.min())

    def direct_read(self):
        """`T/m` straight off the apex, with no regression and no drag model.

        Within |w| < APEX_W the drag term is under 0.05 m/s^2, so the accelerometer is
        reading thrust and nothing else. This is the measurement the card was written
        to manufacture; everything else here is a consistency check on it.
        """
        m = np.abs(self.w) < APEX_W
        if m.sum() < 3:
            return None, 0
        return float(np.average(self.fz[m], weights=self.dt[m])), int(m.sum())


def _longest_run(mask):
    idx = np.where(mask)[0]
    if not len(idx):
        return idx
    return max(np.split(idx, np.where(np.diff(idx) != 1)[0] + 1), key=len)


def arcs(path, ep):
    """Every apex arc in this epoch, reduced to its parked-throttle segment."""
    t = ep.imu["t"]
    fz = -ep.imu["acc"][:, 2]
    w = v_body(ep, t)[:, 2]
    thr = zoh(t - THRUST_DELAY, ep.cmd["t"], ep.cmd["thrust"])
    dt = np.gradient(t)

    out = []
    for stem, t0, t1 in windows(path, ep):
        if not stem.startswith("apex"):
            continue
        m = (t >= t0) & (t <= t1 + 0.3)
        if m.sum() < 20:
            continue
        # The measurement is only valid while the throttle is not moving. Take the
        # longest such run rather than the marker window, which includes the ramp down.
        park = (np.abs(np.gradient(thr)) < 1e-9) & (thr < MAX_PARK_THROTTLE) & m
        seg = _longest_run(park)
        if len(seg) < 10:
            continue
        out.append(Arc(stem, t[seg], w[seg], fz[seg], float(thr[seg][0]), dt[seg]))
    return out


def fit_kz(arcs_in):
    """Fixed effects: `-a_z = alpha_arc + kz * w|w|`, one alpha per arc, one shared kz.

    Returns (kz, [(name, throttle, alpha)], weighted R^2). Every sample constrains kz;
    no sample constrains it through a thrust model, because each arc's thrust is a free
    intercept rather than a function of its throttle.
    """
    n = len(arcs_in)
    rows, y, wt = [], [], []
    for i, a in enumerate(arcs_in):
        onehot = np.zeros((len(a.t), n))
        onehot[:, i] = 1.0
        rows.append(np.hstack([onehot, (a.w * np.abs(a.w))[:, None]]))
        y.append(a.fz)
        wt.append(a.dt)
    X = np.vstack(rows)
    y = np.concatenate(y)
    wt = np.concatenate(wt)
    sw = np.sqrt(wt)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    pred = X @ coef
    r2 = float(1 - np.sum(wt * (y - pred) ** 2)
               / np.sum(wt * (y - np.average(y, weights=wt)) ** 2))
    reads = [(a.name, a.throttle, float(coef[i])) for i, a in enumerate(arcs_in)]
    return float(coef[-1]), reads, r2


def per_arc_kz(a):
    """kz from a single arc on its own -- the spread over these is the honest error bar."""
    q = a.w * np.abs(a.w)
    X = np.stack([np.ones_like(q), q], 1)
    sw = np.sqrt(a.dt)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], a.fz * sw, rcond=None)
    return float(coef[1]), float(coef[0])


def intercept_at(a, kz):
    """`T/m` for one arc with kz held fixed -- valid however short the coast was."""
    return float(np.average(a.fz - kz * a.w * np.abs(a.w), weights=a.dt))


def terminal(path, ep, kz):
    """Steady reads off the vertical terminal runs, given kz.

    Climb: `T/m = g + kz*v^2` at the plateau.
    Descent: thrust is zero, so `dv/dt = g - kz*v^2` reads kz on its own -- and it does
    not need a plateau, only a known acceleration, which is why a run that ran out of
    height is still worth reading.
    """
    t = ep.imu["t"]
    vz = interp_cols(t, ep.pos["t"], ep.pos["v"])[:, 2]
    thr = zoh(t - THRUST_DELAY, ep.cmd["t"], ep.cmd["thrust"])
    out = []
    for stem, t0, t1 in windows(path, ep):
        if not stem.startswith("terminal"):
            continue
        m = (t >= t0) & (t <= t1)
        if m.sum() < 20:
            continue
        tw, vw, hw = t[m], vz[m], thr[m]
        tail = tw >= tw[-1] - PLATEAU_S
        dv = float(vw[-1] - vw[tail][0])
        speed = float(np.abs(np.mean(vw[tail])))
        throttle = float(np.median(hw[tail]))
        steady = abs(dv) < PLATEAU_TOL
        if "up" in stem:
            read = ("T/m", G + kz * speed ** 2)
        else:
            accel = dv / max(tw[-1] - tw[tail][0], 1e-6)
            read = ("kz", (G - accel) / max(speed ** 2, 1e-6))
        out.append(dict(name=stem, throttle=throttle, speed=speed, dv=dv,
                        steady=steady, kind=read[0], value=read[1]))
    return out


def collect(paths):
    """Every arc and terminal read across the given sessions."""
    all_arcs, all_term = [], []
    for path in paths:
        s = Session(path)
        for ep in s.epochs:
            if not ep.has_truth() or ep.cmd is None or ep.imu is None:
                continue
            all_arcs.extend(arcs(path, ep))
            all_term.append((path, ep))
    return all_arcs, all_term


def valid_arcs(paths):
    """Arcs that actually satisfy the maneuver's premise, and the ones that do not."""
    found, _ = collect(paths)
    good, bad = [], []
    for a in found:
        if a.still_s < MIN_STILL:
            bad.append((a, "throttle parked only %.2f s" % a.still_s))
        elif not a.crosses:
            bad.append((a, "w never crossed zero (%.1f..%.1f m/s)"
                        % (a.w.min(), a.w.max())))
        else:
            good.append(a)
    return good, bad


def measure(paths):
    """The whole card 2 read, in one call. This is what `sysid_fit` consumes.

    Returns a dict with `kz` (from the arcs that coasted far enough for a slope),
    `thrust_reads` as [(throttle, T/m)] over every valid arc plus the terminal climb,
    and the diagnostics that say whether to believe any of it.
    """
    good, bad = valid_arcs(paths)
    if not good:
        return dict(kz=None, thrust_reads=[], arcs=[], dropped=bad, terminal=[])

    wide = [a for a in good if a.w_span >= MIN_W_SPAN]
    kz, _, r2 = fit_kz(wide if len(wide) >= 2 else good)
    per_wide = [per_arc_kz(a)[0] for a in (wide if len(wide) >= 2 else good)]

    # `T/m` per arc comes from the whole coast with kz held fixed, not from the handful
    # of samples inside |w| < APEX_W. The apex read is the assumption-free one and it is
    # what the card promised, but at throttle 0.05 the drone falls through that band in a
    # fifth of a second and the read is 7-14 accelerometer samples wide; the two reps
    # there disagree by 1.3 m/s^2 while the whole-arc fits agree to 0.1. Now that kz is
    # independently pinned to four figures, using every sample in the arc costs no
    # assumption and buys the precision back. `direct_read` stays as the cross-check.
    reads = [(a.throttle, intercept_at(a, kz)) for a in good]

    term = []
    for path, ep in collect(paths)[1]:
        term.extend(terminal(path, ep, kz))
    for r in term:
        if r["kind"] == "T/m" and r["steady"]:
            reads.append((r["throttle"], r["value"]))

    kz_term = [r["value"] for r in term if r["kind"] == "kz"]
    return dict(kz=kz, thrust_reads=reads, arcs=good, wide=wide, dropped=bad,
                terminal=term, r2=r2, kz_spread=float(np.std(per_wide)),
                kz_terminal=kz_term)


def main(argv):
    paths = argv or [os.path.join(SESSIONS, s) for s in CARD2]
    paths = [p for p in paths if os.path.isdir(p)]
    if not paths:
        raise SystemExit("no session directories given or found")
    print("sessions: %s\n" % ", ".join(os.path.basename(p.rstrip("/\\"))
                                       for p in paths))

    m = measure(paths)
    for a, why in m["dropped"]:
        print("  DROPPED %-22s %s" % (a.name, why))
    if m["kz"] is None:
        print("no valid apex arc in these sessions")
        return 1
    kz = m["kz"]

    print("apex arcs   (kz taken from the ones spanning >= %.0f m/s of w)" % MIN_W_SPAN)
    print("  %-22s %5s %6s %8s %9s %9s %9s %6s"
          % ("arc", "thr", "still", "w range", "T/m apex", "T/m fit", "kz alone", "kz?"))
    for a in m["arcs"]:
        direct, n = a.direct_read()
        k_alone, _ = per_arc_kz(a)
        print("  %-22s %5.2f %5.2fs %+4.0f..%+3.0f %9s %9.3f %9.4f %6s"
              % (a.name, a.throttle, a.still_s, a.w.min(), a.w.max(),
                 "%.3f/%d" % (direct, n) if direct is not None else "-",
                 intercept_at(a, kz), k_alone,
                 "yes" if a in m["wide"] else "no"))
    print("  kz %.5f   R2 %.4f   spread %.5f over the %d arcs that coasted"
          % (kz, m["r2"], m["kz_spread"], len(m["wide"])))
    if m["kz_terminal"]:
        print("  independent referee -- terminal descent (thrust is zero there): %s"
              % ", ".join("%.5f" % k for k in m["kz_terminal"]))

    print("\nT/m reads feeding the curve (the training set held 17 samples below 0.10):")
    by_thr = {}
    for thr, tm in m["thrust_reads"]:
        by_thr.setdefault(round(thr, 3), []).append(tm)
    for thr in sorted(by_thr):
        v = by_thr[thr]
        print("  throttle %.2f   %s   mean %7.3f"
              % (thr, " ".join("%7.3f" % x for x in v), float(np.mean(v))))

    print("\nvertical terminal runs:")
    for r in m["terminal"]:
        print("  %-18s throttle %.2f  speed %5.1f m/s  residual %+5.2f m/s over %.0fs"
              "  -> %s %.4f%s"
              % (r["name"], r["throttle"], r["speed"], r["dv"], PLATEAU_S,
                 r["kind"], r["value"], "" if r["steady"] else "   (not plateaued)"))

    thrs = [t for t, _ in m["thrust_reads"]]
    agree = (not m["kz_terminal"]
             or abs(np.mean(m["kz_terminal"]) - kz) / kz < 0.05)
    ok = len(m["wide"]) >= 2 and m["kz_spread"] < 0.002 and agree
    print("\nthrottle covered by direct reads: %.2f..%.2f" % (min(thrs), max(thrs)))
    print("%s" % ("PASS -- kz is measured with no thrust parameter in the regression,\n"
                  "and the low-throttle curve has direct reads instead of an"
                  " extrapolation."
                  if ok else
                  "FAIL -- the arcs disagree with each other or with the terminal\n"
                  "descent; do not build a curve on them."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
