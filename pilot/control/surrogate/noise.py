"""P3 FALLBACK detection-noise model -- hand-specified, deliberately pessimistic.

`TRAINING_ARCHITECTURE.md` P3 is a *measured* model: run Claire's perception pipeline
over recorded frames, score against VQ1 ground truth, randomise over the measurement
uncertainty. It does not exist yet and training does not wait on it. This module is the
stated fallback: "worse than anything measured, far better than clean", with wide
per-episode randomisation.

**Every number below is a guess with a reason, not a measurement.** They are written as
RANGES, and an episode draws a point from each range at reset -- never a point estimate,
because a policy will exploit any regularity in synthetic detections and a fixed dropout
rate is a regularity. When P3 lands, replace the ranges here and nothing else changes:
`env.py` only ever reads the sampled arrays.

The bias is deliberately toward pessimism. A policy trained on this and flown against a
better real pipeline is over-cautious; the reverse is a crash. Model selection flags any
checkpoint trained on the fallback (architecture P3).

## Where each range comes from

| parameter | range | why this range |
|---|---|---|
| `max_range_m` | 14-30 | a 1.5 m gate at 30 m is 16 px wide; at 14 m, 34 px. Real detectors on 640x360 lose a 16 px box in hangar dark long before geometry says they should. Deliberately shorter than the ~28 m gate spacing so long stretches have NOTHING in view |
| `p_detect` | 0.62-0.92 | per-frame hit rate on a gate that is in frustum, in range and not occluded. VQ1's orange mask is near-solid on the start frame (2.0% of pixels, mean S 202) so the honest number is likely >0.9; 0.62 is the pessimistic tail |
| `burst_p`, `burst_frames` | 0.006-0.035 / 2-14 | dropouts are BURSTY, not iid: a tracker that loses a gate loses it for a run of frames. iid dropout at the same mean rate is far easier -- one good frame is always ~3 frames away -- and would train a policy that never needs memory |
| `latency_s` | 0.030-0.130 | measured floor: frames trail capture by ~38 ms of JPEG encode plus UDP (`NOTES.md`), and detection + PnP is on top. The upper end covers a loaded VM |
| `max_coast_s` | 0.35-1.10 | how long a track stays `valid` while coasting on a frozen body-frame estimate before the producer gives up. Short = more honest dropouts, long = more confidently wrong data |
| `pos_noise_frac` | 0.010-0.045 | lateral (bearing) error as a fraction of range. Bearing is the STRONG axis of a monocular fit: 1 px at 320 focal is 3 mrad |
| `pos_noise_floor_m` | 0.04-0.16 | pixel quantisation and corner-fit jitter at close range |
| `range_noise_frac` | 0.035-0.120 | depth is the WEAK axis: `range = 480/gate_px` differentiates to `dr/r = -dpx/px`, so 1 px on a 24 px box is 4% |
| `oblique_gain` | 1.0-3.0 | multiplies range noise by `1 + gain*(1-cos)`. An oblique square's corner fit is ill-conditioned along the viewing axis |
| `normal_tilt_min_px` | 1.5-6.0 | the PnP tilt degeneracy, modelled where it actually lives. The two tilt solutions separate only when the projected quad shows measurable trapezoidal distortion, whose scale is `size_px * sin(obliquity)` -- so the failure is joint in range and obliquity, not a fixed angle. A fixed `cos` threshold was the first cut and was wrong in both directions: it killed `normal_valid` outright on a gentle course (every approach is near head-on) while allowing it on a distant oblique gate that is 16 px wide. This is not noise, it is geometry, and it bites hardest on final approach when the approach point matters most |
| `normal_min_px`, `pose_min_px` | 12-22 / 18-40 | a quad fit needs resolvable corners. Below these the producer degrades in the documented order: pose first, then normal, then position |
| `p_pose_fail` | 0.04-0.28 | residual PnP failures not explained by size alone (bloom, partial occlusion, motion blur). Drives the `pose_valid=False` + long-biased-range path |
| `pose_fail_bias` | 1.00-1.30 | EXTRA multiplicative long bias on top of the geometric `1/(0.5+0.5cos)` narrowing, for the frames where the box was clipped or bloomed |
| `sigma_report_mult` | 0.55-1.70 | the producer's `range_sigma_m` is its own ESTIMATE of its error and is itself miscalibrated. A policy that trusts `range_sigma_m` as truth is trusting a number nobody has validated |
| `fp_rate_hz` | 0.05-0.70 | white ceiling lights are the named false-positive risk (`NOTES.md`), and the first gate blooms. A false positive occupies an empty slot for a few frames with a plausible-looking pose |
| `fp_frames` | 2-9 | how long a false track survives before the tracker drops it |
| `ribbon_duty` | 0.25-0.75 | the cyan corridor is "absent from long stretches of real flight", so it is modelled as a two-state burst process, not as a per-frame coin flip |
| `ribbon_noise_rad` | 0.02-0.10 | the ribbon is a fat blob; its direction samples are coarse |
| `gyro_sigma`, `accel_sigma` | 0.004-0.020 / 0.02-0.10 | parked-recording levels: `|a|` reads 9.8100 at rest and the three truth streams agree to five decimals (`CONVENTIONS.md`), so the IMU noise floor is genuinely small. Biases are drawn once per episode |
| `attitude_ref_a` | 6-16 | horizontal specific force at which gravity-derived roll/pitch confidence has decayed to ~0. At 20 deg nose-down, drag reads 3.6 m/s^2 |
| `speed_resid_mps2` | 0.20-0.55 | the drag fit's residual scale, which is what `speed_conf` is tied to. Held-out R^2 0.993/0.994 on horizontal drag over speeds to 34 m/s |

`difficulty` scales the *harmful* half of every range toward its pessimistic end (see
`sample`), so `difficulty=0` is a gentle-but-not-clean world and `difficulty=1` is the
full fallback. Clean detections are a bug, so even at difficulty 0 nothing is noiseless.
"""

from dataclasses import dataclass, field


def _r(lo, hi):
    return (float(lo), float(hi))


@dataclass
class NoiseParams:
    """Per-episode sampling ranges. Every field is `(low, high)`."""

    # visibility / dropout
    max_range_m: tuple = _r(14.0, 30.0)
    min_range_m: tuple = _r(0.6, 1.2)
    min_cos_visible: tuple = _r(0.10, 0.30)     # edge-on gates disappear
    p_detect: tuple = _r(0.62, 0.92)
    burst_p: tuple = _r(0.006, 0.035)
    burst_frames: tuple = _r(2.0, 14.0)
    latency_s: tuple = _r(0.030, 0.130)
    max_coast_s: tuple = _r(0.35, 1.10)

    # geometry error
    pos_noise_frac: tuple = _r(0.010, 0.045)
    pos_noise_floor_m: tuple = _r(0.04, 0.16)
    range_noise_frac: tuple = _r(0.035, 0.120)
    oblique_gain: tuple = _r(1.0, 3.0)
    normal_sigma_rad: tuple = _r(0.04, 0.18)

    # degradation thresholds
    normal_tilt_min_px: tuple = _r(1.5, 6.0)
    normal_min_px: tuple = _r(12.0, 22.0)
    pose_min_px: tuple = _r(18.0, 40.0)
    p_pose_fail: tuple = _r(0.04, 0.28)
    pose_fail_bias: tuple = _r(1.00, 1.30)
    pose_fail_sigma_mult: tuple = _r(2.0, 4.5)
    sigma_report_mult: tuple = _r(0.55, 1.70)

    # confidence
    conf_base: tuple = _r(0.55, 0.95)
    conf_sigma: tuple = _r(0.03, 0.15)

    # false positives
    fp_rate_hz: tuple = _r(0.05, 0.70)
    fp_frames: tuple = _r(2.0, 9.0)

    # ribbon
    ribbon_duty: tuple = _r(0.25, 0.75)
    ribbon_switch_hz: tuple = _r(0.10, 0.55)
    ribbon_noise_rad: tuple = _r(0.02, 0.10)
    ribbon_pixfrac: tuple = _r(0.004, 0.030)

    # own state
    gyro_sigma: tuple = _r(0.004, 0.020)
    gyro_bias: tuple = _r(0.0, 0.010)
    accel_sigma: tuple = _r(0.02, 0.10)
    accel_bias: tuple = _r(0.0, 0.08)
    attitude_sigma_rad: tuple = _r(0.004, 0.020)
    attitude_ref_a: tuple = _r(6.0, 16.0)
    align_min_accel: tuple = _r(0.25, 0.70)
    speed_resid_mps2: tuple = _r(0.20, 0.55)

    # camera clock: 30.05 fps deduped (NOTES.md); jitter covers the dedup edge
    cam_fps: tuple = _r(29.4, 30.7)

    # Which direction of each range is "worse". Fields not listed are symmetric and are
    # drawn from the full range at every difficulty.
    worse_high: tuple = field(default=(
        "burst_p", "burst_frames", "latency_s", "pos_noise_frac", "pos_noise_floor_m",
        "range_noise_frac", "oblique_gain", "normal_sigma_rad", "normal_min_px",
        "normal_tilt_min_px", "pose_min_px", "p_pose_fail", "pose_fail_bias",
        "pose_fail_sigma_mult",
        "fp_rate_hz", "fp_frames", "gyro_sigma", "gyro_bias", "accel_sigma",
        "accel_bias", "attitude_sigma_rad", "ribbon_noise_rad", "speed_resid_mps2",
        "min_range_m", "min_cos_visible", "align_min_accel",
    ))
    worse_low: tuple = field(default=(
        "max_range_m", "p_detect", "max_coast_s", "conf_base",
        "ribbon_duty", "attitude_ref_a", "ribbon_pixfrac",
    ))


# Fields that are ranges to be sampled (everything except the two policy tuples).
SAMPLED = tuple(f for f in NoiseParams.__dataclass_fields__
                if f not in ("worse_high", "worse_low"))


# The value of each parameter in a PERFECT sensor, used as the origin of the `scale`
# interpolation below. It is not uniformly zero: `p_detect` is a probability whose ideal is
# 1, `pose_fail_bias` and the `*_mult` fields are multipliers whose neutral value is 1, and
# the `*_min_px` fields are thresholds whose ideal is 0 (never binds). Where the ideal is
# not obvious from the semantics, the good end of the field's own measured range is used
# rather than inventing a value beyond anything observed.
#
# `cam_fps` is deliberately absent: the camera frame rate is PHYSICAL, not error. A clean
# sensor still runs at 30 fps, and collapsing it would hand the policy a continuous-time
# detector that cannot exist.
_CLEAN = {
    "max_range_m": 120.0, "min_range_m": 0.0, "min_cos_visible": 0.0,
    "p_detect": 1.0, "burst_p": 0.0, "burst_frames": 0.0,
    "latency_s": 0.0, "max_coast_s": 1.10,
    "pos_noise_frac": 0.0, "pos_noise_floor_m": 0.0, "range_noise_frac": 0.0,
    "oblique_gain": 1.0, "normal_sigma_rad": 0.0,
    "normal_tilt_min_px": 0.0, "normal_min_px": 0.0, "pose_min_px": 0.0,
    "p_pose_fail": 0.0, "pose_fail_bias": 1.0, "pose_fail_sigma_mult": 1.0,
    "sigma_report_mult": 1.0,
    "conf_base": 1.0, "conf_sigma": 0.0,
    "fp_rate_hz": 0.0, "fp_frames": 0.0,
    "ribbon_duty": 1.0, "ribbon_switch_hz": 0.0, "ribbon_noise_rad": 0.0,
    "ribbon_pixfrac": 0.030,
    "gyro_sigma": 0.0, "gyro_bias": 0.0, "accel_sigma": 0.0, "accel_bias": 0.0,
    "attitude_sigma_rad": 0.0, "attitude_ref_a": 16.0, "align_min_accel": 0.0,
    "speed_resid_mps2": 0.0,
}

# Adding a field to NoiseParams without a clean value would silently leave it at full
# strength at scale 0, i.e. a "clean" sensor that is not clean. Fail at import instead.
_missing = set(SAMPLED) - set(_CLEAN) - {"cam_fps"}
if _missing:
    raise RuntimeError(f"noise._CLEAN is missing a clean value for: {sorted(_missing)}")


def sample(params, rng, n, difficulty, scale=1.0):
    """Draw one point per episode from every range. Returns `{name: float64 [n]}`.

    `difficulty` in [0, 1] narrows each range toward its pessimistic end rather than
    scaling a magnitude: at difficulty 0 the draw covers the gentle 60% of the range, at
    difficulty 1 the whole of it. The gentle end is still noisy -- difficulty alone never
    reaches zero noise.

    `scale` in [0, 1] is the separate axis that does. Each range is interpolated from its
    perfect-sensor value (`_CLEAN`) toward the configured range, so `scale=0` is a NOISELESS
    sensor, `scale=1` is exactly the behaviour before this argument existed, and anything
    between is a partially degraded one. `cam_fps` is never scaled.

    Training at scale 0 for any length of time is a KNOWN RISK, not a free win --
    architecture P3 is explicit that "clean detections are a bug: a policy will exploit any
    regularity in synthetic tracks". A policy that converges on a perfect sensor learns to
    trust `pos_body` exactly and has to unlearn it. This exists to be ramped, not parked.
    """
    d = float(min(max(difficulty, 0.0), 1.0))
    s = float(min(max(scale, 0.0), 1.0))
    out = {}
    for name in SAMPLED:
        lo, hi = getattr(params, name)
        if s < 1.0 and name in _CLEAN:
            c = _CLEAN[name]
            lo, hi = c + s * (lo - c), c + s * (hi - c)
        span = hi - lo
        if name in params.worse_high:
            # gentle end is `lo`; open the window from 60% to 100% of the range
            sub_hi = lo + span * (0.55 + 0.45 * d)
            sub_lo = lo
        elif name in params.worse_low:
            sub_lo = hi - span * (0.55 + 0.45 * d)
            sub_hi = hi
        else:
            sub_lo, sub_hi = lo, hi
        out[name] = rng.uniform(sub_lo, sub_hi, size=n)
    return out
