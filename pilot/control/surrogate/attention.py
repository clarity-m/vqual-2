"""Attention + the AUTO_ATTENTION yaw servo -- a PARAMETERISED STUB, not the real thing.

`TRAINING_ARCHITECTURE.md` P2 is explicit that the surrogate must run Claire's ACTUAL
attention module, because a behavioural mismatch here is a transfer risk on par with the
noise model, and that its version is frozen with every checkpoint. That module does not
exist yet. This is a stand-in with the same call signature, and it is meant to be deleted.

    HOW TO SWAP IT: construct `VecSurrogate` with `config.attention_factory` set to a
    callable returning an object with `reset(n)` and `step_batch(det, dt)`. The scalar
    `__call__(gates, ribbon, race, dt) -> interface.Attention` below is the shape the real
    module is expected to have (a list of `GateObs`, a `RibbonObs`, a `RaceObs`, a dt);
    `step_batch` is a vectorised fast path over the same logic and `__call__` is
    implemented by calling it, so the two cannot drift apart. A scalar real module can be
    adapted with a loop over envs at a throughput cost.

What the stub does:

* tracks the CURRENT gate until its range falls below `R_COMMIT`, then hands off to the
  next gate, or to the ribbon if there is no next track. Inside a few metres the aircraft
  is ballistic and nothing commanded changes the outcome, which is the whole argument for
  the handoff (`interface.Attention`).
* falls back through GATE_NEXT -> RIBBON -> SEARCH as tracks die.
* SEARCH sweeps: it holds a constant off-axis bearing and flips it periodically, which
  the azimuth servo turns into a steady yaw sweep. Sweeping the *target bearing*
  sinusoidally would instead make the servo chase its own tail.

**`R_COMMIT` is TBD.** `NOTES.md` lists it under Open: "attention handoff range unset --
needs a real approach measurement", measurable from VQ1 flying since the gates are
identical. 6.0 m is a placeholder chosen so that at a 10 m/s cruise the handoff happens
~0.6 s out, which is about when a 1.5 m gate stops fitting usefully in a 90 deg
horizontal frame. Do not read it as measured.

**The servo is azimuth-only, and that is not a simplification.** Yaw cannot change
elevation at all, and with the camera pitched 20 deg up a gate at own altitude sits below
centre at zero bearing -- so a centre-the-target objective would chase an error no yaw
command can reduce (`interface.YawMode`). Vertical framing stays with pitch/thrust in the
policy.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import interface  # noqa: E402

# TBD -- see the module docstring. Not a measurement.
R_COMMIT_M = 6.0


@dataclass
class AttnInput:
    """The synthetic detection state the attention module sees. All arrays are `[n, ...]`.

    This is deliberately the same information a real attention module gets from an
    `Observation`: the gate tracks, the ribbon, and the race packet. Nothing privileged.
    """

    valid: np.ndarray                  # [n, N_GATES] bool
    pos_body: np.ndarray               # [n, N_GATES, 3]
    staleness_s: np.ndarray            # [n, N_GATES]
    normal_valid: np.ndarray           # [n, N_GATES] bool
    confidence: np.ndarray             # [n, N_GATES]
    gate_index: np.ndarray             # [n, N_GATES] int, absolute race index
    ribbon_valid: np.ndarray           # [n] bool
    ribbon_bearings: np.ndarray        # [n, N_RIBBON]
    ribbon_elevs: np.ndarray           # [n, N_RIBBON]
    active_gate_index: np.ndarray      # [n] int


@dataclass
class AttnOutput:
    kind: np.ndarray                   # [n] int, interface.Attn
    target_dir_body: np.ndarray        # [n, 3] unit
    target_gate_index: np.ndarray      # [n] int
    yaw_rate: np.ndarray               # [n] rad/s, canonical NED (>0 nose right)


class AttentionPolicy:
    """Heuristic attention + azimuth yaw servo. Swappable; see the module docstring."""

    def __init__(self, r_commit_m=R_COMMIT_M, yaw_gain=2.5, yaw_rate_max=3.0,
                 stale_max_s=0.45, search_az_rad=0.55, search_period_s=1.6,
                 ribbon_sample=2, deadband_rad=0.015):
        self.r_commit_m = float(r_commit_m)
        self.yaw_gain = float(yaw_gain)
        self.yaw_rate_max = float(yaw_rate_max)
        self.stale_max_s = float(stale_max_s)
        self.search_az_rad = float(search_az_rad)
        self.search_period_s = float(search_period_s)
        self.ribbon_sample = int(ribbon_sample)
        self.deadband_rad = float(deadband_rad)
        self._n = 0
        self.reset(1)

    # -- state ------------------------------------------------------------------
    def reset(self, n=1, mask=None):
        """Allocate/clear per-env sweep state. `mask` resets a subset in place."""
        if mask is None or self._n != n:
            self._n = int(n)
            self._sweep_t = np.zeros(self._n)
            self._sweep_s = np.ones(self._n)
        else:
            self._sweep_t[mask] = 0.0
            self._sweep_s[mask] = 1.0

    # -- the vectorised path ----------------------------------------------------
    def step_batch(self, det, dt):
        """`AttnInput` -> `AttnOutput`. `dt` is `[n]` or a scalar, seconds."""
        n = det.valid.shape[0]
        if self._n != n:
            self.reset(n)

        rng_ = np.linalg.norm(det.pos_body, axis=2)
        fresh = det.valid & (det.staleness_s <= self.stale_max_s)
        cur_ok = fresh[:, 0]
        nxt_ok = fresh[:, 1]

        commit = cur_ok & (rng_[:, 0] < self.r_commit_m)
        prefer_next = commit | ~cur_ok
        use_next = prefer_next & nxt_ok
        use_rib = prefer_next & ~nxt_ok & det.ribbon_valid
        use_cur = ~use_next & ~use_rib & cur_ok
        search = ~use_next & ~use_rib & ~use_cur

        # SEARCH sweep: a held off-axis bearing whose sign flips, so the servo produces a
        # steady yaw rate rather than chasing an oscillating setpoint.
        self._sweep_t = self._sweep_t + np.asarray(dt, dtype=np.float64)
        flip = self._sweep_t >= self.search_period_s
        self._sweep_s = np.where(flip, -self._sweep_s, self._sweep_s)
        self._sweep_t = np.where(flip, 0.0, self._sweep_t)
        self._sweep_t = np.where(search, self._sweep_t, 0.0)

        az = self._sweep_s * self.search_az_rad
        d_search = np.stack([np.cos(az), np.sin(az), np.zeros(n)], axis=1)

        j = min(self.ribbon_sample, det.ribbon_bearings.shape[1] - 1)
        b, e = det.ribbon_bearings[:, j], det.ribbon_elevs[:, j]
        d_rib = np.stack([np.cos(e) * np.cos(b), np.cos(e) * np.sin(b), -np.sin(e)],
                         axis=1)

        d_cur = _unit(det.pos_body[:, 0])
        d_nxt = _unit(det.pos_body[:, 1])

        d = np.where(use_cur[:, None], d_cur,
                     np.where(use_next[:, None], d_nxt,
                              np.where(use_rib[:, None], d_rib, d_search)))
        d = _unit(d)

        kind = np.full(n, int(interface.Attn.SEARCH), dtype=np.int64)
        kind = np.where(use_cur, int(interface.Attn.GATE_CURRENT), kind)
        kind = np.where(use_next, int(interface.Attn.GATE_NEXT), kind)
        kind = np.where(use_rib, int(interface.Attn.RIBBON), kind)

        tgt = np.full(n, -1, dtype=np.int64)
        tgt = np.where(use_cur, det.gate_index[:, 0], tgt)
        tgt = np.where(use_next, det.gate_index[:, 1], tgt)

        yaw = self.yaw_rate_from(d)
        return AttnOutput(kind=kind, target_dir_body=d, target_gate_index=tgt,
                          yaw_rate=yaw)

    def yaw_rate_from(self, target_dir_body):
        """The AUTO_ATTENTION servo: drive the target's AZIMUTH to zero, nothing else."""
        d = np.atleast_2d(np.asarray(target_dir_body, dtype=np.float64))
        az = np.arctan2(d[:, 1], d[:, 0])
        az = np.where(np.abs(az) < self.deadband_rad, 0.0, az)
        return np.clip(self.yaw_gain * az, -self.yaw_rate_max, self.yaw_rate_max)

    # -- the scalar path: the signature the real module is expected to have -------
    def __call__(self, gates, ribbon, race, dt):
        """`(list[GateObs], RibbonObs, RaceObs, dt) -> interface.Attention`.

        Implemented by packing into a 1-env `AttnInput` and calling `step_batch`, so the
        scalar and batch behaviours are the same code by construction.
        """
        ng = len(gates)
        det = AttnInput(
            valid=np.array([[g.valid for g in gates]], dtype=bool),
            pos_body=np.array([[np.asarray(g.pos_body, dtype=np.float64) for g in gates]]),
            staleness_s=np.array([[g.staleness_s for g in gates]], dtype=np.float64),
            normal_valid=np.array([[g.normal_valid for g in gates]], dtype=bool),
            confidence=np.array([[g.confidence for g in gates]], dtype=np.float64),
            gate_index=np.array([[g.index for g in gates]], dtype=np.int64),
            ribbon_valid=np.array([ribbon.valid], dtype=bool),
            ribbon_bearings=np.asarray(ribbon.bearings_rad, dtype=np.float64)[None, :],
            ribbon_elevs=np.asarray(ribbon.elevs_rad, dtype=np.float64)[None, :],
            active_gate_index=np.array([race.active_gate_index], dtype=np.int64),
        )
        assert det.pos_body.shape == (1, ng, 3)
        out = self.step_batch(det, float(dt))
        return interface.Attention(kind=interface.Attn(int(out.kind[0])),
                                   target_dir_body=out.target_dir_body[0].copy(),
                                   target_gate_index=int(out.target_gate_index[0]))


def _unit(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.maximum(n, eps)
    # A zero vector has no direction; body-forward is the only harmless default.
    bad = (n[..., 0] < eps)
    if np.any(bad):
        out = out.copy()
        out[bad] = np.array([1.0, 0.0, 0.0])
    return out
