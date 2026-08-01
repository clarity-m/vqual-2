"""The fitted plant: what the drone does with a command. Written by `sysid_fit.py`.

Everything is **per unit mass**. Mass is not separately identifiable from an
accelerometer -- it measures specific force -- and nothing downstream needs it, so no
number in this file pretends to be a force in newtons.

The model, in body axes, with (u, v, w) the body-frame velocity:

    accelerometer_x = -kx * u|u|
    accelerometer_y = -ky * v|v|
    accelerometer_z = -T/m - kz * w|w| - c_lift * u^2
    T/m             = interp(throttle, thrust_knots)
    body rates      = rate_gain * commanded rates          (simulator convention)

Four properties worth knowing before using it.

**The thrust curve is a measured table, not a polynomial.** Card 2 made `T/m` directly
readable -- wherever `|w|` is small the drag term is under 0.2 m/s^2, so `-a_z` *is*
thrust -- and the reads that come back do not have a low-order shape. There is a
deadband below throttle 0.05 (0.35 m/s^2 at zero, 0.60 at 0.05), then a rise steepening
to about 76 m/s^2 per unit throttle near 0.6, then a flattening to 51.7 at full. Fitting
that with a quadratic overshoots full throttle by 3.9 m/s^2 and puts hover at 0.252
against the 0.27 measured in flight; the table reproduces both ends and puts hover at
0.270. Held-out body-z R^2: 0.78 quadratic, 0.90 table.

The affine curve this replaces was the reason `thrust()` needed a hand clamp -- it went
negative below throttle 0.10 and had to be clipped, which made the model predict a free
fall at throttle 0.10 where the drone actually holds 2.3 m/s^2. That was the last
unmeasured hack in the plant and it sat squarely in the regime a policy enters every
time it chops throttle into a descent.

**The drag is component-wise quadratic, not a function of |v|.** `-k * v_i * |v_i|` per
axis fits to R^2 0.999; the isotropic `-k * |v| * v_i` fits the same data to 0.80. That
is a statement about this simulator, not about aerodynamics, and it is the reason the
body frame is load-bearing: on `|v_world|` the same data gives R^2 0.11.

**Forward speed generates lift, and it is a separate term from drag.** Without
`c_lift * u^2` there is nowhere for that force to go but `kz`, which then reads 0.0443
below 2 m/s of forward speed and 0.0412 between 10 and 15 -- one coefficient describing
two different effects, biased toward whichever regime the recordings happen to contain.
Splitting them is worth +0.043 of held-out body-z R^2. The exponent is `u^2` and not
`u|u|` because lift does not reverse when the drone flies backwards; fitting `u|u|`
instead scores worse than having no term at all, which is the data saying the same thing.
Note the term vanishes at `u = 0`, so hover, terminal climb and both self-checks below
are untouched by it.

**The rate loop is very nearly ideal, and mirrored.** Commanded body rates are tracked
at gain 0.90..0.97 with a lag too short to resolve at 61 Hz, so there is almost no
rotational dynamics to model -- but the simulator's rate convention is mirrored versus
MAVLink NED on all three axes, so `rates_ned()` applies `SIGN_RATE` exactly once, here,
at the boundary. Everywhere above this file, rates stay in the simulator's convention
(which is what `cmd.csv`, the gyro and `interface.py` all use); everywhere below it, in
`step()`, they are NED. Applying that mirror zero times or twice is the sign error the
handoff warns is silent.
"""

import json

import numpy as np

G = 9.81
SIGN_RATE = -1.0    # NED body rate = SIGN_RATE * (gyro or commanded rate)


class Plant:
    """Fitted parameters plus the algebra they define. Stateless."""

    FIELDS = ("kx", "ky", "kz", "c_lift", "thrust_knots", "rate_gain",
              "rate_delay", "thrust_delay", "rate_tau_max")

    # Parameters that postdate a shipped `plant.json`. Defaulting them rather than
    # requiring them keeps every previously written file loadable -- including the
    # frozen sidecars beside training checkpoints, which are the record of what a
    # policy was actually trained against and must not be rewritten to stay readable.
    # A default of 0.0 is also the honest one: it reproduces the older model exactly.
    OPTIONAL = {"c_lift": 0.0}

    def __init__(self, **kw):
        self.meta = kw.pop("meta", {})
        for f in self.FIELDS:
            if f not in kw:
                if f in self.OPTIONAL:
                    kw[f] = self.OPTIONAL[f]
                else:
                    raise KeyError("plant parameter %r missing" % f)
            setattr(self, f, kw[f])
        self.c_lift = float(self.c_lift)
        self.rate_gain = np.asarray(self.rate_gain, dtype=float)
        knots = np.asarray(self.thrust_knots, dtype=float)
        if knots.ndim != 2 or knots.shape[1] != 2:
            raise ValueError("thrust_knots must be a list of [throttle, T/m] pairs")
        order = np.argsort(knots[:, 0])
        self._knot_x = knots[order, 0]
        self._knot_y = knots[order, 1]
        self.thrust_knots = knots[order].tolist()

    # -- serialisation ----------------------------------------------------------
    def to_json(self, path):
        blob = {f: getattr(self, f) for f in self.FIELDS}
        blob["rate_gain"] = list(map(float, self.rate_gain))
        blob["meta"] = self.meta
        with open(path, "w") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(**json.load(fh))

    # -- the model --------------------------------------------------------------
    def thrust(self, throttle):
        """T/m in m/s^2, interpolated between the measured knots.

        `np.interp` holds the end values outside the knot range, which is the safe
        behaviour here: the table spans throttle 0.00 to 1.00 and the command is clipped
        to the same interval, so the flat extension is never reached in normal use and
        cannot invent thrust if it is. No clamp at zero is needed any more -- the lowest
        knot is a measured +0.35 m/s^2 of idle thrust, so the curve is non-negative by
        construction rather than by correction.
        """
        return np.interp(np.asarray(throttle, dtype=float), self._knot_x, self._knot_y)

    def drag(self, v_body):
        v = np.asarray(v_body, dtype=float)
        k = np.array([self.kx, self.ky, self.kz])
        return -k * v * np.abs(v)

    def specific_force(self, v_body, throttle):
        """What the accelerometer would read: thrust along body -z, plus drag and lift.

        Body lift is `c_lift * u^2` along body -z, the same direction as thrust. A quad
        in fast forward flight is not a point mass: the airframe and the tilted disc
        generate a force that does not depend on throttle, and with no term for it the
        fit has nowhere to put that force but `kz`. That is visible directly -- `kz`
        reads 0.0443 below 2 m/s of forward speed and 0.0412 between 10 and 15, which is
        one coefficient being asked to describe two different things.

        It is `u^2` rather than `u|u|` because lift does not reverse when the drone flies
        backwards, and the data agrees: `u|u|` scores *worse* than having no term at all.
        """
        f = self.drag(v_body)
        u = np.asarray(v_body, dtype=float)[..., 0]
        lift = self.c_lift * u ** 2
        return f - np.array([0.0, 0.0, 1.0]) * (self.thrust(throttle) + lift)

    def rates_ned(self, cmd_rates):
        """Commanded body rates -> true NED body rates. The mirror is applied here."""
        return SIGN_RATE * self.rate_gain * np.asarray(cmd_rates, dtype=float)

    def hover_throttle(self):
        """The throttle at which T/m == g. The table is monotone, so this inverts it.

        Body lift does not enter: hovering means `u = 0`, where the term is identically
        zero. The same is true of `terminal_speed` and of both self-checks in `__main__`,
        which is why adding lift leaves the closed-form climb algebra untouched.
        """
        return float(np.interp(G, self._knot_y, self._knot_x))

    def terminal_speed(self, axis=0, tilt_deg=20.0):
        """Speed at which drag balances the horizontal component of thrust at a tilt."""
        k = (self.kx, self.ky, self.kz)[axis]
        return float(np.sqrt(G * np.tan(np.radians(tilt_deg)) / k))

    def describe(self):
        return "\n".join([
            "drag        kx %.4f  ky %.4f  kz %.4f   (m/s^2 per (m/s)^2, body axes)"
            % (self.kx, self.ky, self.kz),
            "body lift   c  %.5f   (m/s^2 per (m/s)^2 of forward speed, along body -z)"
            % self.c_lift,
            "thrust      %d measured knots, idle %.2f   hover %.3f   "
            "full %.1f m/s^2 (%.2f g)"
            % (len(self._knot_x), self.thrust(0.0), self.hover_throttle(),
               self.thrust(1.0), self.thrust(1.0) / G),
            "rate loop   gain %.3f %.3f %.3f   delay %.0f ms   tau < %.0f ms   sign %+.0f"
            % (*self.rate_gain, self.rate_delay * 1e3, self.rate_tau_max * 1e3,
               SIGN_RATE),
            "thrust lag  %.0f ms" % (self.thrust_delay * 1e3),
        ])


class Sim:
    """Steppable forward model: commands in, state out. NumPy, no simulator needed.

    State is a position and velocity in world NED plus a body attitude quaternion.
    Attitude is a quaternion rather than Euler angles because a racing quad on this
    course pitches steeply and Euler integration would gimbal-lock at 90 deg.

    Command delays are held here rather than in `Plant` so that the parameters stay a
    plain data object: `Sim` owns the state, `Plant` owns the physics.
    """

    def __init__(self, plant, dt=1 / 100.0):
        self.plant = plant
        self.dt = dt
        self.reset()

    def reset(self, p=(0, 0, 0), v=(0, 0, 0), q=(1, 0, 0, 0)):
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)
        self.q = np.array(q, dtype=float)
        self.q /= np.linalg.norm(self.q)
        nr = max(1, int(round(self.plant.rate_delay / self.dt)))
        nt = max(1, int(round(self.plant.thrust_delay / self.dt)))
        self._rate_q = [np.zeros(3)] * nr
        self._thr_q = [0.0] * nt
        return self.state()

    def state(self):
        return dict(p=self.p.copy(), v=self.v.copy(), q=self.q.copy())

    def rotation(self):
        return quat_to_rot(self.q)

    def step(self, cmd_rates, throttle):
        """One step of dt. `cmd_rates` and `throttle` are an `interface.Action`."""
        self._rate_q.append(np.asarray(cmd_rates, dtype=float))
        self._thr_q.append(float(throttle))
        rates = self._rate_q.pop(0)
        thr = self._thr_q.pop(0)

        R = self.rotation()
        v_body = R.T @ self.v
        f_body = self.plant.specific_force(v_body, thr)
        a_world = R @ f_body + np.array([0.0, 0.0, G])

        self.p = self.p + self.v * self.dt + 0.5 * a_world * self.dt ** 2
        self.v = self.v + a_world * self.dt
        self.q = quat_integrate(self.q, self.plant.rates_ned(rates), self.dt)
        return self.state()


def quat_to_rot(q):
    """Body -> world rotation from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_integrate(q, omega_body, dt):
    """Advance a quaternion by a body-frame rate vector over dt."""
    w, x, y, z = q
    p, r, s = omega_body
    dq = 0.5 * np.array([-x * p - y * r - z * s,
                         w * p + y * s - z * r,
                         w * r - x * s + z * p,
                         w * s + x * r - y * p])
    out = q + dq * dt
    return out / np.linalg.norm(out)


def quat_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


if __name__ == "__main__":
    import os
    import sys

    p = Plant.load(sys.argv[1] if len(sys.argv) > 1
                   else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "plant.json"))
    print(p.describe())

    # The integrator and the algebra have to agree, or one of them is wrong. Level
    # flight at full throttle settles where thrust minus gravity balances vertical drag.
    sim = Sim(p)
    sim.reset()
    for _ in range(6000):
        sim.step((0, 0, 0), 1.0)
    climb = -float(sim.v[2])
    closed = np.sqrt((p.thrust(1.0) - G) / p.kz)
    print("\nterminal climb   integrated %.2f m/s   closed form %.2f m/s" % (climb, closed))

    sim.reset()
    for _ in range(500):
        sim.step((0, 0, 0), p.hover_throttle())
    drift = float(sim.v[2])
    print("hover drift      %+.3f m/s after 5 s   (expect %.3f, the %.0f ms thrust "
          "delay at zero thrust)" % (drift, G * p.thrust_delay, p.thrust_delay * 1e3))

    ok = abs(climb - closed) < 0.05 * closed and abs(drift - G * p.thrust_delay) < 0.05
    print("\n%s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
