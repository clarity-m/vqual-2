"""The camera model, straight out of `CONVENTIONS.md`. No pixels are rendered.

    640 x 360, cx/cy = 320/180, fx = fy = 320, no distortion, body origin, 20 deg UP tilt.
    body(FRD) -> camera(x right, y down, z forward) = RELABEL @ Ry(-20 deg)

**The tilt is negative and that is the whole trap.** A camera pointing up must put
body-forward BELOW image centre, at

    v = 180 + 320 * tan(20 deg) = 296   of 360

so the vertical span about body-forward is +49.4 deg (up) to -9.4 deg (down), not the
symmetric +-29.4 deg a naive reading gives. That narrow lower edge is the binding
constraint on this course: a gate at own altitude renders low, and pitching down to
accelerate pushes it lower still. `selfcheck.py` asserts the 296 directly, because a
sign flip here would produce a surrogate in which gates are visible exactly when they
are not, and the policy would learn to look the wrong way.

The horizontal half-angle is atan(320/320) = 45 deg, i.e. the 90 deg the spec calls
"VFoV" and which is really the HORIZONTAL FoV.
"""

import numpy as np

FX = 320.0
FY = 320.0
CX = 320.0
CY = 180.0
W_PX = 640
H_PX = 360
TILT_DEG = 20.0

GATE_INNER_M = 1.5


def body_to_cam_matrix(tilt_deg=TILT_DEG):
    """RELABEL @ Ry(-tilt). Returns the 3x3 body->camera rotation."""
    th = np.radians(-tilt_deg)
    ry = np.array([[np.cos(th), 0.0, np.sin(th)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(th), 0.0, np.cos(th)]])
    relabel = np.array([[0.0, 1.0, 0.0],    # cam x  = body y  (right)
                        [0.0, 0.0, 1.0],    # cam y  = body z  (down)
                        [1.0, 0.0, 0.0]])   # cam z  = body x  (forward)
    return relabel @ ry


R_CB = body_to_cam_matrix()


def project(pos_body):
    """`[..., 3]` body-frame points -> (u, v, z_cam). z_cam <= 0 means behind the lens."""
    c = np.asarray(pos_body, dtype=np.float64) @ R_CB.T
    z = c[..., 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return CX + FX * c[..., 0] / zs, CY + FY * c[..., 1] / zs, z


def in_frustum(pos_body, margin_px=0.0):
    """True where the point projects inside the image AND in front of the lens."""
    u, v, z = project(pos_body)
    return ((z > 1e-3) & (u >= -margin_px) & (u <= W_PX + margin_px)
            & (v >= -margin_px) & (v <= H_PX + margin_px))


def vertical_span_rad():
    """(lower, upper) elevation limits about body-forward: (-9.4 deg, +49.4 deg)."""
    t = np.radians(TILT_DEG)
    upper = t + np.arctan(CY / FY)
    lower = t - np.arctan((H_PX - CY) / FY)
    return float(lower), float(upper)


def size_px_from_range(range_m, cos_obliquity):
    """Apparent width of the 1500 mm inner aperture, as a detector would fit it.

    Fronto-parallel gives the spec-exact `size_px = 480 / range_m`, i.e. the calibrated
    monocular rangefinder. An oblique gate projects NARROWER, so a size-only range read
    off this is biased LONG -- and it is used exactly when the PnP fit that would correct
    it has failed. The `0.5 + 0.5*cos` blend is a modelling choice, not a measurement:
    the longest fitted edge foreshortens less than the full quad does, so the true
    narrowing sits between `cos` (full foreshortening) and 1 (none). Replace with the
    measured curve when P3 lands.
    """
    r = np.maximum(np.asarray(range_m, dtype=np.float64), 1e-3)
    return FX * GATE_INNER_M / r * (0.5 + 0.5 * np.abs(cos_obliquity))


def range_from_size_px(size_px):
    """The consumer-side inverse: `range_m = 480 / gate_px`, spec-exact and long-biased."""
    return FX * GATE_INNER_M / np.maximum(np.asarray(size_px, dtype=np.float64), 1e-6)
