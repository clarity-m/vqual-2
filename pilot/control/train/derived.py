"""Derived observation channels — vertical rate, computed rather than learned.

THE PROBLEM. Nothing in the 73-D observation reports vertical velocity. Both velocity
channels are drag-derived and horizontal (`vel_bearing = atan2(-a_y, -a_x)`,
`speed = sqrt(a_horiz / K_DRAG)`), and `accel_z` is specific force — thrust plus gravity,
not v_z. run1 flew into the floor in 55.5% of its episodes: it was optimising correctly,
it simply could not perceive that it was sinking.

WHY STACKING DOES NOT FIX IT. Measured by linear probe against true v_z (a lower bound on
what the MLP can extract):

    input                                     inputs    R2      RMSE
    6 contiguous frames (the default)            438   0.460   1.62 m/s
    6 dilated frames, 0.70 s span                438   0.562   1.48 m/s
    32 frames at FULL RATE, same span           2336   0.556   1.49 m/s
    1 frame + the channels below                  77   0.630   1.34 m/s

Full-rate access to the window — exactly what a GRU would add over a stack — buys nothing
(0.556 vs 0.562 at 5x the width). The stack is not losing information by subsampling.
What is missing is a NONLINEAR step no amount of stacking performs: rotating body-frame
specific force into the world frame using attitude, adding gravity, and integrating.
That is bilinear in (attitude, accel), so a linear read of raw frames cannot form it and
an MLP would have to learn it from reward.

So: compute it. Four leaky integrators alone score R2 0.451 — better than all 438 inputs
of the shipping stack. This is why the architecture's "stack, don't GRU" call was right,
though for a different reason than it gives: we do need recurrent STATE (an integrator has
memory), we do not need a LEARNED one.

WHAT THIS IS NOT. `interface.Observation` does not move. These channels are derived from
fields the policy already receives — roll (54), pitch (55), accel (51-53) — using only
NOISY observed values, never truth, so training and deployment compute them identically.

HONEST LIMITS. R2 0.630 is not 1.0 and will not get there. A pure integrator drifts on
accelerometer bias, which is why the form is leaky; and per `perception-error.md` the
vision channels are too noisy to correct that drift (per-sighting bias ~0.6 m redrawn each
sighting, rho(0.1 s) = 0.43, so differencing cancels almost nothing). Bounded estimate,
substantially better altitude control, not a solved problem.
"""

from __future__ import annotations

import numpy as np

# Channel indices into the 73-D vector. Verified against `interface.Observation` by
# perturbing each field and diffing, not counted by hand.
ROLL_INDEX = 54
PITCH_INDEX = 55
ACCEL_SLICE = slice(51, 54)

G = 9.81

# Time constants, seconds. A spread rather than one value: the short ones track fast
# transients (the onset of a sink) and survive bias; the long ones integrate a sustained
# descent but drift. Handing the network all four lets it weight them by situation,
# which is the part worth learning.
DEFAULT_TAUS = (0.1, 0.3, 1.0, 3.0)

N_DERIVED = len(DEFAULT_TAUS)


def vertical_accel(obs: np.ndarray) -> np.ndarray:
    """World-frame vertical acceleration (+ = downward, matching NED) from `obs`.

    Rotates the measured body specific force onto world-down by the measured roll/pitch
    and removes gravity. Yaw is not needed and is not observable anyway — the vertical
    axis is yaw-invariant, which is the reason this particular derived channel is
    computable at all under the VQ2 sensor blocks.
    """
    a = np.atleast_2d(np.asarray(obs, dtype=np.float64))
    r = a[:, ROLL_INDEX]
    p = a[:, PITCH_INDEX]
    f = a[:, ACCEL_SLICE]
    # Third row of R_wb applied to body specific force: the world-down component.
    f_down = (-f[:, 0] * np.sin(p)
              + f[:, 1] * np.sin(r) * np.cos(p)
              + f[:, 2] * np.cos(r) * np.cos(p))
    return f_down + G


class VerticalRate:
    """Bank of leaky integrators over `vertical_accel`, one per time constant.

    Shared by training and deployment, like `FrameStack` and `ObsNormalizer` — the layout
    it produces is part of the checkpoint contract. Deployment use (single stream):

        vr = VerticalRate(n_envs=1)
        vr.reset()
        extra = vr.step(raw_obs_73)         # -> [1, 4], append to the raw vector
    """

    def __init__(self, n_envs: int = 1, taus=DEFAULT_TAUS, dt: float = 1.0 / 55.0,
                 clip: float = 20.0) -> None:
        self.n_envs = int(n_envs)
        self.taus = tuple(float(t) for t in taus)
        self.dt = float(dt)
        # A leaky integrator cannot run away, but a corrupt observation can still spike
        # it; 20 m/s is far outside anything the aircraft does.
        self.clip = float(clip)
        self._decay = np.exp(-self.dt / np.asarray(self.taus, dtype=np.float64))
        self.v = np.zeros((self.n_envs, len(self.taus)), dtype=np.float64)

    @property
    def n_channels(self) -> int:
        return len(self.taus)

    def reset(self, mask=None) -> None:
        """Zero the integrators. `mask` resets only those envs (episode boundaries).

        Zero is the right initial value: an episode starts with the aircraft at whatever
        vertical rate it has, and the integrator has no evidence yet either way.
        """
        if mask is None:
            self.v[:] = 0.0
        else:
            m = np.asarray(mask).reshape(-1).astype(bool)
            self.v[m] = 0.0

    def step(self, obs: np.ndarray, done=None) -> np.ndarray:
        """Advance one decision step and return the derived channels, [n_envs, n_taus].

        `done` marks envs whose episode ended on the transition INTO `obs`; under the
        auto-resetting VecSurrogate that observation already belongs to the next episode,
        so those integrators are zeroed BEFORE accumulating it — carrying vertical rate
        across an episode boundary would integrate two unrelated worlds together.
        """
        if done is not None:
            self.reset(done)
        a = vertical_accel(obs).reshape(-1, 1)
        self.v = self._decay[None, :] * self.v + a * self.dt
        np.clip(self.v, -self.clip, self.clip, out=self.v)
        return self.v.astype(np.float32, copy=True)

    def state_dict(self) -> dict:
        return {"taus": list(self.taus), "dt": self.dt, "clip": self.clip}


def augment(obs: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Append derived channels to a raw observation batch. [n, 73] + [n, 4] -> [n, 77]."""
    a = np.atleast_2d(np.asarray(obs, dtype=np.float32))
    e = np.atleast_2d(np.asarray(extra, dtype=np.float32))
    return np.concatenate([a, e], axis=1)
