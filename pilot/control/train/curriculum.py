"""Curriculum: difficulty and speed schedules driven by rolling completion rate.

Per T4:
  1. difficulty — gentle wide courses and low noise first, annealed toward full measured
     noise and tight turns as completion rises;
  2. speed — `speed_cap` starts low (~0.5) so exploration survives collision
     terminations, and relaxes toward 1.0 as completion rises.

Two rules from the architecture are encoded here rather than left to judgement:
  * If completion STALLS, weaken the progress term near gates. NEVER weaken the collision
    penalty. The env owns reward, so this harness can only *request* the weaker progress
    shaping (via an optional `progress_gate_scale` EnvConfig field, applied only if that
    field exists) and back the schedules off a notch; nothing here can touch the collision
    penalty, by construction.
  * `enable_time_penalty` flips on only once completion is reliable (>80% by default):
    the leaderboard counts completed runs first, fast runs second.

"Completion" = an episode that passed all of its gates. The env's info dict is the source;
see `make_completion_fn` for the fallbacks when it does not say so directly.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .ppo import info_array

_COMPLETION_KEYS = ("completed", "complete", "success", "finished", "all_gates_passed")
_N_GATES_KEYS = ("n_gates", "n_gates_total", "gates_total", "course_n_gates")


def make_completion_fn(n_gates_hint: int | None = None, verbose: bool = True):
    """Build `fn(info, done) -> list[bool]`, one flag per finished episode.

    Resolution order, best first:
      1. an explicit completion flag in info;
      2. `gates_passed >= n_gates` when the env reports the episode's gate count;
      3. `gates_passed >= n_gates_hint and not collision`, using the low end of
         `n_gates_range` — warned about once, because with an 18-22 gate range this counts
         a 22-gate course that reached gate 18 as complete.
    """
    state = {"mode": None}

    def fn(info: dict, done: np.ndarray) -> list[bool]:
        done = np.asarray(done).reshape(-1).astype(bool)
        n = done.size
        for key in _COMPLETION_KEYS:
            arr = info_array(info, key, n)
            if arr is not None:
                _announce(state, f"info['{key}']", verbose)
                return [bool(v) for v in arr[done]]

        gates = info_array(info, "gates_passed", n)
        if gates is None:
            _announce(state, "unavailable (no 'gates_passed' in info)", verbose)
            return []

        for key in _N_GATES_KEYS:
            total = info_array(info, key, n)
            if total is not None:
                _announce(state, f"gates_passed >= info['{key}']", verbose)
                return [bool(v) for v in (gates >= total)[done]]

        if n_gates_hint is None:
            _announce(state, "unavailable (no gate total, no hint)", verbose)
            return []
        coll = info_array(info, "collision", n)
        ok = gates >= float(n_gates_hint)
        if coll is not None:
            ok = ok & (~coll.astype(bool))
        _announce(
            state,
            f"gates_passed >= {n_gates_hint} and not collision (APPROXIMATE)",
            verbose,
            warn=True,
        )
        return [bool(v) for v in ok[done]]

    return fn


def _announce(state: dict, mode: str, verbose: bool, warn: bool = False) -> None:
    if state["mode"] == mode:
        return
    state["mode"] = mode
    msg = f"[curriculum] completion rate measured as: {mode}"
    if warn:
        warnings.warn(msg + " — env does not report a completion flag or gate total",
                      RuntimeWarning, stacklevel=3)
    elif verbose:
        print(msg, flush=True)


@dataclass
class CurriculumConfig:
    difficulty_start: float = 0.0
    difficulty_step: float = 0.05
    speed_cap_start: float = 0.5
    speed_cap_step: float = 0.05
    speed_cap_min: float = 0.3
    promote_rate: float = 0.70   # rolling completion above this -> harder / faster
    demote_rate: float = 0.25    # below this -> back off one notch
    window: int = 200            # episodes in the rolling window
    min_episodes: int = 40       # before any adjustment is allowed
    hold_updates: int = 5        # minimum updates between adjustments (env rebuild cost)
    time_penalty_on: float = 0.80
    time_penalty_off: float = 0.60
    stall_updates: int = 40      # updates without promotion at low completion == stall
    progress_gate_scale_step: float = 0.15
    progress_gate_scale_min: float = 0.4
    frozen: bool = False         # pin everything at the start values


@dataclass
class Curriculum:
    cfg: CurriculumConfig = field(default_factory=CurriculumConfig)

    def __post_init__(self) -> None:
        self.difficulty = float(np.clip(self.cfg.difficulty_start, 0.0, 1.0))
        self.speed_cap = float(np.clip(self.cfg.speed_cap_start, 1e-3, 1.0))
        # Demotion undoes promotions and nothing more. The start values are already the
        # gentle end of the schedule, and completion is legitimately zero for the first
        # stretch of any run — backing off below the start there would keep shrinking the
        # policy's authority in exactly the situation where it needs more of it.
        self._difficulty_floor = self.difficulty
        self._speed_floor = max(self.speed_cap, self.cfg.speed_cap_min)
        self.enable_time_penalty = False
        self.progress_gate_scale = 1.0
        self._window: deque[bool] = deque(maxlen=self.cfg.window)
        self._since_change = 0
        self._since_promote = 0
        self.n_episodes = 0
        self.n_promotions = 0
        self.n_demotions = 0
        self.n_stalls = 0

    # --- observation of progress -------------------------------------------------------

    def record(self, completions) -> None:
        for c in completions:
            self._window.append(bool(c))
            self.n_episodes += 1

    @property
    def completion_rate(self) -> float | None:
        if len(self._window) < min(self.cfg.min_episodes, self.cfg.window):
            return None
        return float(np.mean(self._window))

    @property
    def completion_rate_raw(self) -> float:
        """Rate over whatever is in the window, for logging before it is actionable."""
        return float(np.mean(self._window)) if self._window else 0.0

    # --- schedule ----------------------------------------------------------------------

    def step(self) -> bool:
        """Advance one update. Returns True if the env config changed."""
        self._since_change += 1
        self._since_promote += 1
        if self.cfg.frozen:
            return False

        rate = self.completion_rate
        if rate is None:
            return False

        changed = self._update_time_penalty(rate)
        if self._since_change < self.cfg.hold_updates:
            return changed

        c = self.cfg
        if rate >= c.promote_rate and (self.difficulty < 1.0 or self.speed_cap < 1.0):
            # Speed first while difficulty is still low: a policy that cannot fly fast on
            # a gentle course will not learn anything from a harder one.
            if self.speed_cap < 1.0 and self.speed_cap <= self.difficulty + 0.5:
                self.speed_cap = min(1.0, self.speed_cap + c.speed_cap_step)
            else:
                self.difficulty = min(1.0, self.difficulty + c.difficulty_step)
            self._window.clear()  # the rate now describes a course distribution we left
            self._since_change = 0
            self._since_promote = 0
            self.n_promotions += 1
            return True

        if rate <= c.demote_rate and (
            self.difficulty > self._difficulty_floor or self.speed_cap > self._speed_floor
        ):
            if self.difficulty > self._difficulty_floor:
                self.difficulty = max(self._difficulty_floor, self.difficulty - c.difficulty_step)
            else:
                self.speed_cap = max(self._speed_floor, self.speed_cap - c.speed_cap_step)
            self._window.clear()
            self._since_change = 0
            self.n_demotions += 1
            return True

        # A schedule with nothing left to promote is finished, not stalled.
        saturated = self.difficulty >= 1.0 and self.speed_cap >= 1.0
        if self._since_promote >= c.stall_updates and rate < c.promote_rate and not saturated:
            return self._handle_stall() or changed

        return changed

    def _update_time_penalty(self, rate: float) -> bool:
        c = self.cfg
        if not self.enable_time_penalty and rate >= c.time_penalty_on:
            self.enable_time_penalty = True
            return True
        if self.enable_time_penalty and rate < c.time_penalty_off:
            self.enable_time_penalty = False
            return True
        return False

    def _handle_stall(self) -> bool:
        """Completion is not improving: weaken progress shaping near gates, not the
        collision penalty, and give back one notch of speed."""
        c = self.cfg
        self._since_promote = 0
        self.n_stalls += 1
        changed = False
        if self.progress_gate_scale > c.progress_gate_scale_min:
            self.progress_gate_scale = max(
                c.progress_gate_scale_min, self.progress_gate_scale - c.progress_gate_scale_step
            )
            changed = True
        if self.speed_cap > self._speed_floor:
            self.speed_cap = max(self._speed_floor, self.speed_cap - c.speed_cap_step)
            changed = True
        if changed:
            self._window.clear()
            self._since_change = 0
            print(
                f"[curriculum] STALL: progress_gate_scale={self.progress_gate_scale:.2f} "
                f"speed_cap={self.speed_cap:.2f} (collision penalty untouched)",
                flush=True,
            )
        return changed

    # --- resume ------------------------------------------------------------------------

    _STATE_FIELDS = (
        "difficulty", "speed_cap", "_difficulty_floor", "_speed_floor",
        "enable_time_penalty", "progress_gate_scale",
        "_since_change", "_since_promote",
        "n_episodes", "n_promotions", "n_demotions", "n_stalls",
    )

    def state_dict(self) -> dict:
        """Everything `__post_init__` sets, so a resumed run continues one schedule.

        The counters matter as much as the levels: `_since_promote` drives the stall
        detector, and a run resumed with it zeroed would wait a fresh `stall_updates`
        before backing off again. The window is stored as a plain list; `cfg` is not
        stored, because it comes from the CLI and a resume must be able to change it.
        """
        out = {k: getattr(self, k) for k in self._STATE_FIELDS}
        out["window"] = [bool(c) for c in self._window]
        return out

    def seed_from(self, **levels) -> dict:
        """Start the schedule at levels recovered from elsewhere, e.g. a checkpoint's
        `env_config` when no resume sidecar exists.

        Sets the demotion floors to the seeded values, matching `__post_init__`: the run
        must not be able to demote below the point it was resumed at. Returns what was
        actually applied, for the caller to report. Counters are deliberately untouched —
        this recovers levels, not history, and pretending otherwise would let the stall
        detector fire on a window that does not exist.
        """
        applied = {}
        for name in ("difficulty", "speed_cap", "progress_gate_scale", "enable_time_penalty"):
            if levels.get(name) is None:
                continue
            v = levels[name]
            setattr(self, name, bool(v) if name == "enable_time_penalty" else float(v))
            applied[name] = getattr(self, name)
        self._difficulty_floor = self.difficulty
        self._speed_floor = max(self.speed_cap, self.cfg.speed_cap_min)
        return applied

    def load_state_dict(self, d: dict) -> None:
        for k in self._STATE_FIELDS:
            if k not in d:
                continue
            cur = getattr(self, k)
            setattr(self, k, type(cur)(d[k]) if isinstance(cur, (bool, int)) else float(d[k]))
        self._window = deque(
            (bool(c) for c in d.get("window", ())), maxlen=self.cfg.window
        )

    # --- what the env should be built with ---------------------------------------------

    def env_spec(self) -> dict:
        return {
            "difficulty": round(float(self.difficulty), 4),
            "speed_cap": round(float(self.speed_cap), 4),
            "enable_time_penalty": bool(self.enable_time_penalty),
            "progress_gate_scale": round(float(self.progress_gate_scale), 4),
        }

    def summary(self) -> str:
        rate = self.completion_rate
        rate_s = "n/a" if rate is None else f"{rate:.2f}"
        return (
            f"diff={self.difficulty:.2f} speed_cap={self.speed_cap:.2f} "
            f"timepen={int(self.enable_time_penalty)} compl={rate_s} "
            f"(win {len(self._window)})"
        )
