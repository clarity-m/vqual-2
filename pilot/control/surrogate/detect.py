"""Synthetic detections: project the true gates, then corrupt them properly.

**Detections tick on the camera's own ~30 fps clock, not per policy step.** At 45-65 Hz
decisions that means roughly every other step sees an UNCHANGED detection with
`staleness_s` grown by ~33 ms, and `TRAINING_ARCHITECTURE.md` says that rhythm is part of
what the policy must learn. Ticking detections per step would delete it.

The corruption chain, in the order `interface.GateObs` says a producer degrades:

1. **Visibility.** In frustum (the real 640x360 / 20 deg-up model, so the -9.4 deg lower
   edge bites), in range, not edge-on, and the gate has to exist.
2. **Dropout**, bursty rather than iid -- see `noise.py` for why that matters.
3. **Latency.** Modelled at the input: a tick uses the pose from `latency_s` ago, so the
   measurement genuinely describes where things WERE. This is exact rather than
   approximate, and it costs one pose ring buffer instead of a payload ring buffer.
4. **Staleness accumulation while tracks coast.** A dead track keeps its last body-frame
   estimate -- which goes stale in the aircraft's frame as it flies, exactly as a real
   coasting tracker does -- and `staleness_s` grows until `max_coast_s`, then `valid`
   drops. `staleness_s` INCLUDES the latency, because it is the age of the information.
5. **Position and normal noise growing with range and obliquity.**
6. **`normal_valid` failing near head-on** -- the PnP tilt degeneracy. Not noise: a
   geometric failure that fires on final approach, when the approach point matters most.
7. **`pose_valid=False` fallback with LONG-biased range**, from `480/gate_px` on a gate
   that projects narrower than fronto-parallel. Used exactly when the fit that would
   correct it has failed.
8. **False positives** -- white ceiling lights are the named risk, and the first gate
   blooms. A false track occupies an EMPTY slot for a few frames with a plausible pose.

Nothing here is measured. Every parameter comes from `noise.py`'s P3 fallback and is
drawn per episode from a range.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import camera  # noqa: E402
import vmath  # noqa: E402

N_SLOTS = 3          # interface.N_GATES: current gate + lookahead


class Detector:
    """Track state for `n` envs x 3 slots. Slot j is the gate at `active_gate_index + j`."""

    def __init__(self, n):
        self.n = int(n)
        z = lambda *s: np.zeros((self.n,) + s)          # noqa: E731
        self.valid = np.zeros((n, N_SLOTS), dtype=bool)
        self.pos = z(N_SLOTS, 3)
        self.nrm = z(N_SLOTS, 3)
        self.nvalid = np.zeros((n, N_SLOTS), dtype=bool)
        self.pvalid = np.zeros((n, N_SLOTS), dtype=bool)
        self.rsig = z(N_SLOTS)
        self.conf = z(N_SLOTS)
        self.size = z(N_SLOTS)
        self.stamp = np.full((n, N_SLOTS), -1e3)
        self.is_fp = np.zeros((n, N_SLOTS), dtype=bool)
        self.burst = np.zeros((n, N_SLOTS), dtype=np.int64)
        self.fp_left = np.zeros((n, N_SLOTS), dtype=np.int64)

    _FIELDS = ("valid", "pos", "nrm", "nvalid", "pvalid", "rsig", "conf", "size",
               "stamp", "is_fp", "burst", "fp_left")

    def clear(self, mask):
        """Wipe tracks for the envs in `mask` (episode reset)."""
        for f in self._FIELDS:
            a = getattr(self, f)
            a[mask] = -1e3 if f == "stamp" else 0

    def shift(self, adv):
        """`active_gate_index` advanced: slot j inherits slot j+1, slot 2 is cleared."""
        if not np.any(adv):
            return
        for f in self._FIELDS:
            a = getattr(self, f)
            a[adv, 0] = a[adv, 1]
            a[adv, 1] = a[adv, 2]
            a[adv, 2] = -1e3 if f == "stamp" else 0

    # -- the camera tick --------------------------------------------------------
    def tick(self, rng, nz, p_cap, q_cap, gate_pos, gate_nrm, active, n_gates,
             t_cap, tick_mask):
        """One camera frame. `p_cap`/`q_cap` are the pose at CAPTURE time (latency)."""
        n = self.n
        idx = active[:, None] + np.arange(N_SLOTS)[None, :]        # [n, 3]
        idx_c = np.clip(idx, 0, gate_pos.shape[1] - 1)
        exists = idx < n_gates[:, None]

        gp = np.take_along_axis(gate_pos, idx_c[:, :, None], axis=1)      # [n,3,3]
        gn = np.take_along_axis(gate_nrm, idx_c[:, :, None], axis=1)

        r_wb = vmath.quat_to_rot(q_cap)
        rel = gp - p_cap[:, None, :]
        pos_b = np.einsum('nji,nsj->nsi', r_wb, rel)
        # interface.py: normal_body points TOWARD the camera. A gate not yet crossed is
        # approached from its own approach side, so that is -travel_direction.
        nrm_b = np.einsum('nji,nsj->nsi', r_wb, -gn)

        rng_t = np.linalg.norm(pos_b, axis=2)
        dir_b = pos_b / np.maximum(rng_t, 1e-9)[:, :, None]
        cos_obl = np.abs(np.einsum('nsi,nsi->ns', dir_b, nrm_b))
        size_t = camera.size_px_from_range(rng_t, cos_obl)

        c = lambda k: nz[k][:, None]                                # noqa: E731
        visible = (exists & camera.in_frustum(pos_b)
                   & (rng_t < c("max_range_m")) & (rng_t > c("min_range_m"))
                   & (cos_obl > c("min_cos_visible")))

        range_fac = np.clip(1.0 - 0.6 * (rng_t / c("max_range_m")) ** 2, 0.05, 1.0)
        obl_fac = np.clip(0.35 + 0.65 * cos_obl, 0.0, 1.0)
        p_hit = c("p_detect") * range_fac * obl_fac

        in_burst = self.burst > 0
        detected = visible & ~in_burst & (rng.random((n, N_SLOTS)) < p_hit)
        # Bursts are entered only when the gate was otherwise detectable, so a burst is a
        # tracker failure rather than a restatement of "out of view".
        start = visible & ~in_burst & ~detected & (rng.random((n, N_SLOTS)) < c("burst_p"))
        blen = np.maximum(1, np.round(rng.uniform(0.35, 1.0, (n, N_SLOTS))
                                      * c("burst_frames"))).astype(np.int64)
        new_burst = np.where(start, blen, np.maximum(self.burst - 1, 0))

        # -- the measurement ---------------------------------------------------
        sigma_lat = c("pos_noise_floor_m") + c("pos_noise_frac") * rng_t
        sigma_rng = rng_t * c("range_noise_frac") * (1.0 + c("oblique_gain") * (1.0 - cos_obl))

        pose_ok = (detected & (size_t > c("pose_min_px"))
                   & (rng.random((n, N_SLOTS)) > c("p_pose_fail")))

        r_pose = rng_t + rng.normal(0.0, 1.0, (n, N_SLOTS)) * sigma_rng
        r_size = (camera.range_from_size_px(size_t) * c("pose_fail_bias")
                  + rng.normal(0.0, 1.0, (n, N_SLOTS)) * sigma_rng * c("pose_fail_sigma_mult"))
        r_meas = np.maximum(np.where(pose_ok, r_pose, r_size), 0.3)

        e = rng.normal(0.0, 1.0, (n, N_SLOTS, 3))
        e = e - np.einsum('nsi,nsi->ns', e, dir_b)[:, :, None] * dir_b
        pos_m = dir_b * r_meas[:, :, None] + sigma_lat[:, :, None] * e

        # The tilt ambiguity separates only when the projected quad is measurably
        # trapezoidal: `size_px * sin(obliquity)` is that separation, in pixels.
        sin_obl = np.sqrt(np.maximum(1.0 - cos_obl * cos_obl, 0.0))
        nvalid = (pose_ok & (size_t * sin_obl > c("normal_tilt_min_px"))
                  & (size_t > c("normal_min_px")))
        sigma_n = c("normal_sigma_rad") * (1.0 + 2.0 * cos_obl)
        nrm_m = vmath.unit(nrm_b + sigma_n[:, :, None] * rng.normal(0.0, 1.0, (n, N_SLOTS, 3)))

        conf = np.clip(c("conf_base") * np.clip(size_t / 25.0, 0.25, 1.0)
                       * (0.4 + 0.6 * cos_obl)
                       + rng.normal(0.0, 1.0, (n, N_SLOTS)) * c("conf_sigma"), 0.03, 1.0)
        conf = np.where(pose_ok, conf, conf * 0.7)

        rsig = np.where(pose_ok, sigma_rng,
                        sigma_rng * c("pose_fail_sigma_mult")
                        + np.abs(r_size - rng_t)) * c("sigma_report_mult")

        # -- false positives ---------------------------------------------------
        stale_now = t_cap[:, None] - self.stamp
        dead = ~self.valid | (stale_now > c("max_coast_s"))
        cam_dt = 1.0 / c("cam_fps")
        fp_new = (~detected & dead & (self.fp_left <= 0)
                  & (rng.random((n, N_SLOTS)) < c("fp_rate_hz") * cam_dt))
        az = rng.uniform(-0.70, 0.70, (n, N_SLOTS))
        el = rng.uniform(-0.14, 0.75, (n, N_SLOTS))
        r_fp = rng.uniform(6.0, 25.0, (n, N_SLOTS))
        d_fp = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)],
                        axis=2)
        pos_fp = d_fp * r_fp[:, :, None]
        conf_fp = rng.uniform(0.15, 0.50, (n, N_SLOTS))
        size_fp = camera.size_px_from_range(r_fp, 1.0) * rng.uniform(0.6, 1.4, (n, N_SLOTS))
        fp_life = np.round(c("fp_frames")).astype(np.int64)

        fp_hold = (~detected & (self.fp_left > 0) & self.is_fp)

        # -- commit ------------------------------------------------------------
        m = tick_mask[:, None]
        write = detected & m
        wfp = fp_new & m
        hold = fp_hold & m

        self.pos = np.where(write[:, :, None], pos_m,
                            np.where(wfp[:, :, None], pos_fp, self.pos))
        self.nrm = np.where(write[:, :, None], nrm_m,
                            np.where(wfp[:, :, None], 0.0, self.nrm))
        self.nvalid = np.where(write, nvalid, np.where(wfp, False, self.nvalid))
        self.pvalid = np.where(write, pose_ok, np.where(wfp, False, self.pvalid))
        self.rsig = np.where(write, rsig, np.where(wfp, r_fp * 0.5, self.rsig))
        self.conf = np.where(write, conf, np.where(wfp, conf_fp, self.conf))
        self.size = np.where(write, size_t, np.where(wfp, size_fp, self.size))
        self.is_fp = np.where(write, False, np.where(wfp, True, self.is_fp))
        # A false track is dropped by the tracker when its lifetime runs out, rather than
        # being allowed to coast for another `max_coast_s`. Otherwise a rare false
        # positive turns into a common one purely through the coasting rule.
        expire = tick_mask[:, None] & self.is_fp & (self.fp_left <= 0) & ~wfp
        self.valid = (self.valid | write | wfp) & ~expire
        self.stamp = np.where(write | wfp | hold, t_cap[:, None], self.stamp)
        self.fp_left = np.where(wfp, fp_life,
                                np.where(hold, self.fp_left - 1,
                                         np.where(write, 0, self.fp_left)))
        self.burst = np.where(tick_mask[:, None], new_burst, self.burst)

    # -- what the observation sees ---------------------------------------------
    def read(self, t_now, max_coast_s):
        """(valid, staleness) after coasting. A track past `max_coast_s` is dropped."""
        stale = np.maximum(t_now[:, None] - self.stamp, 0.0)
        alive = self.valid & (stale <= max_coast_s[:, None])
        return alive, np.where(alive, stale, 0.0)
