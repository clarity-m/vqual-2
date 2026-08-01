"""Loading, clock alignment and epoch splitting for the VQ1 system-ID recordings.

Shared by `sysid_frames.py`, `sysid_latency.py`, `sysid_fit.py` and `sysid_replay.py`.
Nothing here interprets a sign or fits a parameter; it only puts every stream on one
clock so that a lag measured downstream is physics rather than bookkeeping.

Three things it exists to get right.

**A session is not one flight.** The simulator's boot clock *restarts* at every
`SIM_RESET`, and the recordings are full of them (six in `20260731-150712`, four in
`20260731-131305`). Wall time keeps running, so a session read on the wall clock looks
continuous while actually splicing together separate runs across a teleport back to the
pad. Anything differentiated across such a seam -- velocity, attitude -- produces a
fictional several-hundred-m/s^2 sample that no force model explains. So a session is
loaded as a list of `Epoch`s, each one continuous, and every fit runs per epoch.

**One clock inside an epoch.** Telemetry rows carry a device stamp (`time_usec` on
HIGHRES_IMU / ODOMETRY / ACTUATOR_OUTPUT_STATUS, `time_boot_ms` on ATTITUDE /
LOCAL_POSITION_NED -- measured to share the sim-boot base) *and* the client's wall stamp
at receipt. `cmd.csv` carries only a wall stamp, taken immediately after the send, so
commands are mapped onto the sim clock by a line fitted per epoch to the telemetry
(wall, device) pairs. The slope earns its keep: the sim does not run at exactly
realtime, and a 0.1% error is 150 ms over a 150 s epoch -- an order of magnitude larger
than the actuator lag being measured.

**Zero-order hold for commands.** A command is held until the next one is sent, so
resampling it with `np.interp` invents ramps between samples and smears exactly the
edges that carry the response. `zoh()` holds the previous value.

Duplicated and reordered messages are dropped by device stamp; the camera is not the
only stream this simulator sends twice.

    python3 pilot/control/sysid_data.py pilot/sessions/2026*/     # inventory
"""

import csv
import json
import os

import numpy as np

RESET_JUMP = -1.0    # s of device-clock regression that means SIM_RESET, not reordering
MIN_EPOCH = 3.0      # s; shorter epochs are reset transients, not flying
SETTLE = 0.5         # s dropped after each reset (pose is mid-transition -- NOTES.md)
CONTACT_PAD = 0.25   # s cut either side of every COLLISION sample
MIN_THRUST = 0.05    # default "not parked" gate; see Epoch.flying


def load_csv(path):
    """CSV -> dict of column name to float array. Missing or empty file -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    return {key: np.array([_f(r[key]) for r in rows]) for key in rows[0]}


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def zoh(t_query, t_src, y_src):
    """Sample a held signal: the value of y_src at the last t_src <= t_query."""
    idx = np.searchsorted(t_src, t_query, side="right") - 1
    return np.asarray(y_src)[np.clip(idx, 0, len(t_src) - 1)]


def interp_cols(t_query, t_src, y_src):
    """Linear interpolation of an (N, k) array, column by column."""
    y = np.asarray(y_src)
    if y.ndim == 1:
        return np.interp(t_query, t_src, y)
    return np.stack([np.interp(t_query, t_src, y[:, i]) for i in range(y.shape[1])], 1)


def _dedup(t, cols):
    """Sort by device stamp and keep the first row of each distinct stamp."""
    order = np.argsort(t, kind="stable")
    t = t[order]
    cols = [c[order] for c in cols]
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 0
    return t[keep], [c[keep] for c in cols]


class Clock:
    """wall ns -> sim seconds, from the rows that carry both stamps.

    The intercept comes from a high quantile of the residual, not from least squares.
    A wall receipt stamp is a device stamp plus a one-sided transport delay, so the
    least-squares intercept is biased by the *mean* delay, while the packets that
    arrived fastest sit closest to the true offset. `spread_ms` is what that leaves
    unresolved, and therefore the floor on any lag measured below.
    """

    def __init__(self, wall_ns, sim_s):
        w = wall_ns / 1e9
        self.w0 = w[0]
        slope, icept = np.polyfit(w - self.w0, sim_s, 1)
        resid = sim_s - (slope * (w - self.w0) + icept)
        self.slope = float(slope)
        self.icept = float(icept + np.quantile(resid, 0.95))
        self.spread_ms = float((np.quantile(resid, 0.95) - np.quantile(resid, 0.05)) * 1e3)
        self.n = len(w)

    def to_sim(self, wall_ns):
        return self.slope * (np.asarray(wall_ns) / 1e9 - self.w0) + self.icept


class Epoch:
    """One continuous simulator run: no reset inside, one clock, one trajectory.

    Streams, each a dict on the sim clock, or None when the stream is absent (all
    three truth streams are absent under VQ2):

        cmd   t, roll_rate, pitch_rate, yaw_rate, thrust, armed   (client sends)
        imu   t, acc (N,3), gyro (N,3)                            HIGHRES_IMU
        act   t, m (N,4)                                          ACTUATOR_OUTPUT_STATUS
        pos   t, p (N,3), v (N,3)                                 LOCAL_POSITION_NED
        att   t, roll, pitch, yaw                                 ATTITUDE
        odo   t, q (N,4) wxyz, v (N,3)                             ODOMETRY

    Truth is handed over **raw**. The per-axis sign corrections are re-derived in
    `sysid_frames.py` rather than applied here, so that exactly one file in this
    directory decides what "truth" means.
    """

    def __init__(self, session, index, tables):
        self.session = session
        self.index = index
        self.name = "%s#%d" % (session, index)
        for k, v in tables.items():
            setattr(self, k, v)
        self.markers = []

    def span(self):
        ts = [d["t"] for d in self._streams() if d is not None and len(d["t"])]
        return (min(t[0] for t in ts), max(t[-1] for t in ts))

    def duration(self):
        t0, t1 = self.span()
        return t1 - t0

    def _streams(self):
        return [self.cmd, self.imu, self.act, self.pos, self.att, self.odo]

    def has_truth(self):
        return None not in (self.att, self.pos, self.odo)

    def flying(self, t, min_speed=1.0, min_thrust=MIN_THRUST):
        """Mask over sim times t: thrust commanded, off the pad, past the reset settle.

        A parked drone satisfies any force model with thrust = drag = 0, so leaving it
        in inflates every R^2 while teaching the fit nothing.

        `min_thrust` is the crude half of that test and card 2 flies straight through
        it: the apex holds park at exactly 0.05 and the terminal descent at 0.00, so the
        default gate discards the entire low-throttle measurement the card exists to
        produce. Pass `min_thrust=0.0` to keep it. That is safe because the speed gate
        is the half that actually excludes the pad -- a drone on the ground is not
        moving at 1 m/s, whatever its throttle says.

        Deliberately does *not* gate on `cmd.armed`. That column carries the HEARTBEAT
        SAFETY_ARMED flag, and the VQ1 build leaves it at 0 through an entire flight --
        epoch 0 of `20260731-150712` reaches 34.3 m/s and thrust command 0.86 with
        `armed` never once true. Filtering on it discards the widest-envelope data in
        the whole dataset.
        """
        t = np.asarray(t)
        m = (t >= self.span()[0] + SETTLE)
        if self.cmd is not None and min_thrust > 0.0:
            m &= zoh(t, self.cmd["t"], self.cmd["thrust"]) > min_thrust
        if self.pos is not None:
            m &= np.linalg.norm(interp_cols(t, self.pos["t"], self.pos["v"]), axis=1) \
                > min_speed
        return m

    def quiet(self, t, pad=CONTACT_PAD):
        """Mask over sim times t excluding contact windows.

        `COLLISION` is a contact *sample*, not a crash (NOTES.md), and a contact is a
        force this plant model has no term for -- it is also an impulse, so the 60 Hz
        velocity stream aliases it into a several-hundred-m/s^2 artefact. Both reasons
        to cut a window around every sample rather than trust an outlier gate.
        """
        t = np.asarray(t)
        if self.contacts is None or len(self.contacts) == 0:
            return np.ones(len(t), dtype=bool)
        i = np.clip(np.searchsorted(self.contacts, t), 0, len(self.contacts) - 1)
        near = np.abs(self.contacts[i] - t)
        j = np.clip(i - 1, 0, len(self.contacts) - 1)
        near = np.minimum(near, np.abs(self.contacts[j] - t))
        return near > pad

    def usable(self, t, min_speed=1.0, min_thrust=MIN_THRUST):
        return self.flying(t, min_speed, min_thrust) & self.quiet(t)


def _split_tables(tables, bounds):
    """Slice every stream into epochs by wall time, then by device time.

    The wall boundary alone is not enough: streams are received independently, so a
    packet generated before a reset can be *received* after it and land in the next
    epoch carrying a device stamp from the old run. One such straggler stretches the
    epoch's apparent duration by however long the previous run lasted. The second
    filter keeps only rows whose device stamp belongs to the epoch's own run.
    """
    out = []
    for w0, w1, d0, d1 in bounds:
        cut = {}
        for key, tbl in tables.items():
            if not tbl:
                cut[key] = None
                continue
            sel = (tbl["t_wall_ns"] >= w0) & (tbl["t_wall_ns"] < w1)
            if key in DEV_KEY:
                dev = tbl[DEV_KEY[key]] / DEV_SCALE[key]
                sel &= (dev >= d0 - 1.0) & (dev <= d1 + 1.0)
            cut[key] = {k: v[sel] for k, v in tbl.items()} if sel.any() else None
        out.append(cut)
    return out


DEV_KEY = dict(imu="time_usec", att="time_boot_ms", pos="time_boot_ms",
               odo="time_usec", act="time_usec")
DEV_SCALE = dict(imu=1e6, att=1e3, pos=1e3, odo=1e6, act=1e6)


class Session:
    """A recording directory, split into continuous epochs."""

    def __init__(self, path, min_epoch=MIN_EPOCH):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        tables = dict(imu=load_csv(os.path.join(path, "imu.csv")),
                      att=load_csv(os.path.join(path, "attitude.csv")),
                      pos=load_csv(os.path.join(path, "position.csv")),
                      odo=load_csv(os.path.join(path, "odometry.csv")),
                      act=load_csv(os.path.join(path, "actuators.csv")),
                      cmd=load_csv(os.path.join(path, "cmd.csv")),
                      col=load_csv(os.path.join(path, "collisions.csv")))
        self.resets = 0
        self.epochs = []
        if not any(tables[k] for k in DEV_KEY):
            return

        bounds = self._epoch_bounds(tables)
        self.resets = len(bounds) - 1
        markers = self._markers(path)

        for i, cut in enumerate(_split_tables(tables, bounds)):
            ep = self._build(i, cut)
            if ep is None or ep.duration() < min_epoch:
                continue
            t0, t1 = ep.span()
            ep.markers = sorted(t for t in
                                (float(ep.clock.to_sim(w)) for w in markers)
                                if t0 <= t <= t1)
            self.epochs.append(ep)

    def _epoch_bounds(self, tables):
        """Per epoch: (wall start, wall end, device start, device end).

        Boundaries come from whichever stream is densest, so this works on VQ2
        recordings (IMU only) as well as on the VQ1 truth sessions.
        """
        key = max(DEV_KEY, key=lambda k: len(tables[k].get("t_wall_ns", ())))
        tbl = tables[key]
        dev = tbl[DEV_KEY[key]] / DEV_SCALE[key]
        wall = tbl["t_wall_ns"]
        jump = list(np.where(np.diff(dev) < RESET_JUMP)[0])
        starts = [0] + [i + 1 for i in jump]
        ends = [i for i in jump] + [len(dev) - 1]
        out = []
        for n, (s, e) in enumerate(zip(starts, ends)):
            w0 = wall[s] if n == 0 else (wall[s - 1] + wall[s]) / 2
            w1 = np.inf if e == len(dev) - 1 else (wall[e] + wall[e + 1]) / 2
            out.append((w0, w1, dev[s:e + 1].min(), dev[s:e + 1].max()))
        return out

    @staticmethod
    def _markers(path):
        out = []
        ev = os.path.join(path, "events.jsonl")
        if os.path.exists(ev):
            with open(ev) as fh:
                for line in fh:
                    e = json.loads(line)
                    if e.get("kind") == "marker":
                        out.append(e["t_wall_ns"])
        return out

    def _build(self, index, cut):
        pairs_w, pairs_s = [], []
        for key in DEV_KEY:
            tbl = cut[key]
            if tbl is not None:
                pairs_w.append(tbl["t_wall_ns"])
                pairs_s.append(tbl[DEV_KEY[key]] / DEV_SCALE[key])
        if not pairs_w:
            return None
        clock = Clock(np.concatenate(pairs_w), np.concatenate(pairs_s))

        streams = {}
        for key in ("imu", "att", "pos", "odo", "act"):
            tbl = cut[key]
            if tbl is None:
                streams[key] = None
                continue
            dev = tbl[DEV_KEY[key]] / DEV_SCALE[key]
            if key == "imu":
                t, c = _dedup(dev, [tbl[k] for k in ("xacc", "yacc", "zacc",
                                                     "xgyro", "ygyro", "zgyro")])
                streams[key] = dict(t=t, acc=np.stack(c[:3], 1), gyro=np.stack(c[3:], 1))
            elif key == "att":
                t, c = _dedup(dev, [tbl[k] for k in ("roll", "pitch", "yaw")])
                streams[key] = dict(t=t, roll=c[0], pitch=c[1], yaw=c[2])
            elif key == "pos":
                t, c = _dedup(dev, [tbl[k] for k in ("x", "y", "z", "vx", "vy", "vz")])
                streams[key] = dict(t=t, p=np.stack(c[:3], 1), v=np.stack(c[3:], 1))
            elif key == "odo":
                t, c = _dedup(dev, [tbl[k] for k in ("qw", "qx", "qy", "qz",
                                                     "vx", "vy", "vz")])
                streams[key] = dict(t=t, q=np.stack(c[:4], 1), v=np.stack(c[4:], 1))
            else:
                t, c = _dedup(dev, [tbl[k] for k in ("m0", "m1", "m2", "m3")])
                streams[key] = dict(t=t, m=np.stack(c, 1))

        cmd = cut["cmd"]
        streams["cmd"] = None if cmd is None else dict(
            t=clock.to_sim(cmd["t_wall_ns"]), roll_rate=cmd["roll_rate"],
            pitch_rate=cmd["pitch_rate"], yaw_rate=cmd["yaw_rate"],
            thrust=cmd["thrust"], armed=cmd["armed"])

        col = cut["col"]
        streams["contacts"] = np.array([]) if col is None \
            else np.sort(clock.to_sim(col["t_wall_ns"]))

        ep = Epoch(self.name, index, streams)
        ep.clock = clock
        return ep

    def describe(self):
        lines = ["%s   %d reset(s), %d usable epoch(s)"
                 % (self.name, self.resets, len(self.epochs))]
        for ep in self.epochs:
            fly = ep.usable(ep.imu["t"]).sum() if ep.imu is not None else 0
            lines.append("  #%d  %6.1f s  clock x%.5f +-%4.1f ms  imu %5.1f Hz  "
                         "usable %5d  contacts %5d  truth %s  markers %d"
                         % (ep.index, ep.duration(), ep.clock.slope, ep.clock.spread_ms,
                            _hz(ep.imu), fly, len(ep.contacts),
                            "yes" if ep.has_truth() else "no ", len(ep.markers)))
        return "\n".join(lines)


def _hz(d):
    if d is None or len(d["t"]) < 2:
        return float("nan")
    return (len(d["t"]) - 1) / (d["t"][-1] - d["t"][0])


def load_epochs(paths, truth_only=True, min_epoch=MIN_EPOCH):
    """Every usable epoch across the given session directories, flattened."""
    out = []
    for p in paths:
        for ep in Session(p, min_epoch=min_epoch).epochs:
            if truth_only and not ep.has_truth():
                continue
            if ep.cmd is None or ep.imu is None:
                continue
            out.append(ep)
    return out


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(Session(p).describe())
