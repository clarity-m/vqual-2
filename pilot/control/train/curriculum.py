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

    # --- the per-gate signal, which is what actually drives the schedule -------------
    #
    # Completion is the wrong control variable while completion is zero. It is per-gate
    # accuracy raised to the gate count: measured on the shipped surrogate, per-gate is
    # ~0.51, which over 18-22 gates is ~1e-6 and reads as a flat 0.000 no matter how much
    # the underlying accuracy improves. `run1` spent 77 updates pinned at difficulty 0.0
    # and speed_cap 0.5 for exactly this reason. Per-gate accuracy is roughly invariant
    # to episode length, so it keeps working as `gates_per_episode` grows.
    #
    # The thresholds are HIGHER than the completion ones on purpose: to finish 20 gates
    # at 70% you need per-gate 0.983, so 0.85 is a promotion, not an arrival.
    gate_rate_promote: float = 0.85
    gate_rate_demote: float = 0.55
    min_attempts: int = 60       # pooled gate attempts before the rate is actionable

    # --- episode length, the axis the architecture was missing -----------------------
    gates_start: int = 3
    gates_step: int = 2
    gates_max: int = 22          # >= n_gates_range[1] means "the whole course"
    hold_updates: int = 5        # minimum updates between adjustments (env rebuild cost)
    time_penalty_on: float = 0.80
    time_penalty_off: float = 0.60
    stall_updates: int = 40      # updates without promotion at low completion == stall
    # Was 0.15. The stall response weakened near-gate progress on the theory that a
    # stalled policy is one diving at gates -- but the measured cause of the stall was
    # that completion over 18-22 gates is per-gate-rate^20 and unreachable, so the
    # detector was firing on the wrong diagnosis. It is also the one shaping term that
    # is NOT potential-based, i.e. the one thing here that can actually move the optimal
    # policy. Zero disables it; the speed_cap half of the stall response still applies.
    progress_gate_scale_step: float = 0.0
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
        self.gates_per_episode = int(self.cfg.gates_start)
        self._window: deque[bool] = deque(maxlen=self.cfg.window)
        # (passes, attempts) per finished episode. Pooled, not averaged per episode: an
        # episode that attempted one gate and missed it should not weigh the same as one
        # that attempted six and passed five.
        self._gates: deque = deque(maxlen=self.cfg.window)
        self._since_change = 0
        self._since_promote = 0
        self.n_episodes = 0
        self.n_promotions = 0
        self.n_demotions = 0
        self.n_stalls = 0

    # --- observation of progress -------------------------------------------------------

    def record(self, completions, gate_passes=None, gate_attempts=None) -> None:
        for c in completions:
            self._window.append(bool(c))
            self.n_episodes += 1
        if gate_passes is not None and gate_attempts is not None:
            for p, a in zip(gate_passes, gate_attempts):
                self._gates.append((float(p), float(a)))

    @property
    def completion_rate(self) -> float | None:
        if len(self._window) < min(self.cfg.min_episodes, self.cfg.window):
            return None
        return float(np.mean(self._window))

    @property
    def gate_rate(self) -> float | None:
        """Pooled clean passes / plane crossings of the active gate. None until enough
        attempts have accumulated for the ratio to mean anything."""
        if not self._gates:
            return None
        att = sum(a for _, a in self._gates)
        if att < self.cfg.min_attempts:
            return None
        return float(sum(p for p, _ in self._gates) / max(att, 1.0))

    @property
    def gate_rate_raw(self) -> float:
        """The ratio over whatever is in the window, for logging before it is actionable."""
        att = sum(a for _, a in self._gates)
        return float(sum(p for p, _ in self._gates) / att) if att > 0 else 0.0

    def _signal(self):
        """(value, promote_threshold, demote_threshold) for the schedule.

        Per-gate accuracy when it is available, completion otherwise -- so an env that
        does not report gate counters still schedules exactly as it did before.
        """
        r = self.gate_rate
        if r is not None:
            return r, self.cfg.gate_rate_promote, self.cfg.gate_rate_demote
        return self.completion_rate, self.cfg.promote_rate, self.cfg.demote_rate

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

        c = self.cfg
        # The time penalty stays keyed to COMPLETION: it is a statement about finishing
        # races, not about gate accuracy, and the leaderboard counts completed runs first.
        comp = self.completion_rate
        changed = self._update_time_penalty(comp) if comp is not None else False

        rate, promote_at, demote_at = self._signal()
        if rate is None:
            return changed
        if self._since_change < c.hold_updates:
            return changed

        room = (self.difficulty < 1.0 or self.speed_cap < 1.0
                or self.gates_per_episode < c.gates_max)
        if rate >= promote_at and room:
            # Episode length grows on EVERY promotion, independently of the speed /
            # difficulty rotation, because per-gate accuracy is roughly invariant to how
            # many gates an episode contains -- lengthening does not corrupt the signal
            # the schedule runs on, and it is the axis that eventually reaches a full
            # course. Speed first among the other two while difficulty is still low: a
            # policy that cannot fly fast on a gentle course learns nothing from a harder.
            if self.gates_per_episode < c.gates_max:
                self.gates_per_episode = min(int(c.gates_max),
                                             self.gates_per_episode + int(c.gates_step))
            if self.speed_cap < 1.0 and self.speed_cap <= self.difficulty + 0.5:
                self.speed_cap = min(1.0, self.speed_cap + c.speed_cap_step)
            elif self.difficulty < 1.0:
                self.difficulty = min(1.0, self.difficulty + c.difficulty_step)
            self._clear_windows()  # the rate now describes a distribution we left
            self._since_change = 0
            self._since_promote = 0
            self.n_promotions += 1
            return True

        if rate <= demote_at and (
            self.difficulty > self._difficulty_floor or self.speed_cap > self._speed_floor
        ):
            if self.difficulty > self._difficulty_floor:
                self.difficulty = max(self._difficulty_floor, self.difficulty - c.difficulty_step)
            else:
                self.speed_cap = max(self._speed_floor, self.speed_cap - c.speed_cap_step)
            self._clear_windows()
            self._since_change = 0
            self.n_demotions += 1
            return True

        # A schedule with nothing left to promote is finished, not stalled.
        saturated = (self.difficulty >= 1.0 and self.speed_cap >= 1.0
                     and self.gates_per_episode >= c.gates_max)
        if self._since_promote >= c.stall_updates and rate < promote_at and not saturated:
            return self._handle_stall() or changed

        return changed

    def _clear_windows(self) -> None:
        self._window.clear()
        self._gates.clear()

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
            self._clear_windows()
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
        "enable_time_penalty", "progress_gate_scale", "gates_per_episode",
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
        out["gates"] = [[float(p), float(a)] for p, a in self._gates]
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
        self._gates = deque(
            ((float(p), float(a)) for p, a in d.get("gates", ())), maxlen=self.cfg.window
        )

    # --- what the env should be built with ---------------------------------------------

    def env_spec(self) -> dict:
        return {
            "difficulty": round(float(self.difficulty), 4),
            "speed_cap": round(float(self.speed_cap), 4),
            "enable_time_penalty": bool(self.enable_time_penalty),
            "progress_gate_scale": round(float(self.progress_gate_scale), 4),
            "gates_per_episode": int(self.gates_per_episode),
        }

    def summary(self) -> str:
        rate = self.completion_rate
        rate_s = "n/a" if rate is None else f"{rate:.2f}"
        gr = self.gate_rate
        gr_s = "n/a" if gr is None else f"{gr:.2f}"
        return (
            f"diff={self.difficulty:.2f} speed_cap={self.speed_cap:.2f} "
            f"K={self.gates_per_episode} gate_rate={gr_s} "
            f"timepen={int(self.enable_time_penalty)} compl={rate_s} "
            f"(win {len(self._window)})"
        )
