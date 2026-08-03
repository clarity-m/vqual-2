"""D6's scripted recovery supervisor: level the aircraft after a graze, hand back.

Wraps any `interface.Policy`. Deterministic, memoryless apart from three scalars, and
nothing learned -- that is the point. An 8-minute budget against a 1-2 minute lap makes
recovery nearly free in time, so there is no reason to spend PPO samples teaching a
network to un-tumble, and every reason not to: a tumble is far off the training
distribution and is exactly where a network's output is least predictable.

TRIGGER. `RaceObs.t_since_collision_s` is a timer on the last contact SAMPLE, and a
parked drone emits ~250 of those per second, so the signal to watch is the timer being
RESET -- it decreasing -- not it being small. `collision_episodes` incrementing says the
same thing and is used as a second, equivalent trigger in case the producer only
publishes the counter.

HOLD. While active: roll and pitch driven to zero proportionally off the gravity
estimate, hover thrust, yaw left to attention. Continued contact keeps
`t_since_collision_s` pinned near zero, which keeps the supervisor active by
construction -- correct behaviour while still scraping something.

HAND BACK when the attitude is level AND either the re-acquire time has elapsed or
attention has a gate again. `max_hold_s` is a failsafe: an attitude estimate that never
converges must not strand the aircraft in a hover forever, and flying badly beats not
flying. On activation the inner policy is reset, so a stacked-observation network comes
back with a clean history rather than four frames of tumble; the handback state itself
(hover, level, possibly no gate in frustum) is covered in training by P2's randomized
episode starts.

The levelling command uses `own.roll_rad` / `own.pitch_rad`, which come from gravity and
are trustworthy precisely when the aircraft is NOT accelerating hard -- so the level
test also requires `attitude_conf`, which is the producer saying the gravity read is
currently meaningful. A "level" verdict taken from a swamped accelerometer is how a
supervisor hands back mid-tumble.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pilot import interface                                    # noqa: E402
from pilot.control.policies.envelope import clamp_action       # noqa: E402


class RecoverySupervisor(interface.Policy):
    """Scripted level-and-hover override around `inner`."""

    def __init__(self, inner,
                 trigger_s=0.25,          # a timer below this AND falling is a reset
                 reacquire_s=0.80,        # s of calm before handing back
                 level_tol_rad=0.12,      # 7 deg
                 k_level=4.0,             # 1/s on attitude error
                 conf_min=0.20,           # gravity read must mean something
                 max_hold_s=4.0,          # failsafe; flying badly beats not flying
                 hover_thrust=interface.HOVER_THRUST,
                 reset_inner=True):
        self.inner = inner
        self.trigger_s = float(trigger_s)
        self.reacquire_s = float(reacquire_s)
        self.level_tol_rad = float(level_tol_rad)
        self.k_level = float(k_level)
        self.conf_min = float(conf_min)
        self.max_hold_s = float(max_hold_s)
        self.hover_thrust = float(hover_thrust)
        self.reset_inner = bool(reset_inner)
        self.reset()

    def reset(self):
        self.active = False
        self.recoveries = 0
        self._held_s = 0.0
        self._prev_t = None
        self._episodes = None
        if self.inner is not None:
            self.inner.reset()

    def __call__(self, obs):
        race = obs.race
        t = float(race.t_since_collision_s)
        episodes = int(race.collision_episodes)

        reset_seen = (self._prev_t is None and t < self.trigger_s) \
            or (self._prev_t is not None and t < self.trigger_s and t < self._prev_t - 1e-9)
        counted = self._episodes is not None and episodes > self._episodes
        self._prev_t, self._episodes = t, episodes

        if (reset_seen or counted) and not self.active:
            self.active = True
            self._held_s = 0.0
            self.recoveries += 1
            if self.reset_inner and self.inner is not None:
                self.inner.reset()

        if self.active:
            self._held_s += max(float(obs.dt_s), 0.0)
            own = obs.own
            level = (abs(float(own.roll_rad)) <= self.level_tol_rad
                     and abs(float(own.pitch_rad)) <= self.level_tol_rad
                     and float(own.attitude_conf) >= self.conf_min)
            reacquired = t >= self.reacquire_s or any(g.valid for g in obs.gates)
            if (level and reacquired) or self._held_s >= self.max_hold_s:
                self.active = False
            else:
                return clamp_action(
                    self.k_level * (0.0 - float(own.roll_rad)),
                    self.k_level * (0.0 - float(own.pitch_rad)),
                    self.hover_thrust,
                    yaw_rate=0.0, yaw_mode=interface.YawMode.AUTO_ATTENTION)

        act = self.inner(obs)
        # The inner policy is contractually a policy, not a trusted one: re-clamp.
        return clamp_action(act.roll_rate, act.pitch_rate, act.thrust,
                            yaw_rate=act.yaw_rate, yaw_mode=act.yaw_mode)
