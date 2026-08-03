"""Skylight compass: per-frame heading (mod 90 deg) from the ceiling light grid.

WHY THIS WORKS. VQ2 blocks every pose stream; gravity (HIGHRES_IMU accel, canonical per
CONVENTIONS.md) gives roll and pitch, and yaw has been unobservable since day one. Claire's
observation closes the gap: the ceiling lights are square panels forming a flat grid
parallel to the ground. A horizontal line with direction (cos t, sin t, 0) in the
gravity-levelled body frame projects into the image as a segment whose interpretation
plane (the plane through the camera centre and the segment) must CONTAIN that direction.
So every detected light-edge segment votes for one heading angle, mod 180; the grid's two
orthogonal families fold that to mod 90. No vanishing-point intersection is needed -- each
segment is an independent measurement, which is what makes the estimate robust to a
handful of bad segments.

DEFINITIONS (all angles degrees):
  * levelled body frame: z along gravity (down), x = body-forward projected horizontal.
  * theta_mod90 = angle of a grid line family measured in that frame, atan2 over (x, y)
    with y to the right -- i.e. positive theta is a line rotated toward body-right.
  * psi = (-theta) mod 90 = the BODY's heading in the grid frame. Yawing nose-right
    (positive canonical NED yaw rate) INCREASES psi. skylight_validate.py measures this
    sign against the gyro rather than trusting the derivation.

FALSE POSITIVES, each with its filter, all refusals:
  * floor reflections of the lights (the floor is glossy; Claire's frames show them) --
    rejected by GRAVITY: every accepted component must sit entirely ABOVE the levelled
    horizon (min elevation HORIZON_MARGIN_DEG). The ceiling is above the camera, its
    mirror image is below, and gravity tells them apart with no appearance argument.
  * white gate decoration (AI-GP wordmark, checkerboard) shares the S<50 & V>200 mask --
    rejected by ORANGE SURROUND: decoration is paint on an orange frame, so its dilated
    neighbourhood is orange; a ceiling light's is dark.
  * station signage is GREY (V 95-195, perception/NOTES.md), already below V_MIN.
  * active-gate bloom -- white core inside an orange frame; the surround test kills it.

The lights image as thin bright square OUTLINES (fill ~0.10), and far rows merge into
elongated strips. Neither is a problem: the estimator consumes edge SEGMENTS, not quads,
and a merged strip's long edges lie along the same grid family as the squares' edges.

    # per-session pass: heading per frame + gyro-based mod-90 unwrap
    python3 skylight.py <session-dir> [--out skylight_<name>.csv] [--stride 1]

Importable:  heading(frame_img, gravity_vec) -> (theta_mod90, confidence_dict)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402

# --- light mask (perception/NOTES.md colour table: white lights S<50 & V>200) ----------
SAT_MAX = 50
VAL_MIN = 200

MIN_AREA_PX = 12          # below this a component's edges are too short to vote
MIN_SEG_PX = 6.0          # minimum edge-segment length
APPROX_EPS_PX = 1.6       # polygon approximation tolerance on contours
HORIZON_MARGIN_DEG = 2.0  # entire component must sit this far above the levelled horizon
ORANGE_FRAC_MAX = 0.06    # dilated-surround orange fraction above this = gate decoration
SURROUND_PAD_PX = 5

# vote quality
MIN_COND = 0.20           # |horizontal part of interpretation-plane normal|; below this
                          # the segment is near-degenerate (plane nearly horizontal)
GRAV_TOL = 2.0            # |a|-9.81 gate for accepting the accelerometer as gravity.
                          # Looser than mapvq2's 0.35 on purpose: a 1-2 deg tilt error
                          # costs about as much heading error, and a compass that answers
                          # on 95% of frames beats one that answers on 40%.

# confidence gate for the session-level unwrap: frames failing this coast on the gyro
CONF_MIN_SEG = 4
CONF_MAX_SPREAD_DEG = 12.0

_R_CB = L.body_to_cam()          # body(FRD) -> camera(right,down,forward)
_R_BC = _R_CB.T


def level_rotation(g_body):
    """Rows of R: levelled-frame axes in body coordinates. z = gravity(down),
    x = body-forward projected horizontal. Degenerate only at |pitch| = 90."""
    z = np.asarray(g_body, float)
    z = z / np.linalg.norm(z)
    ex = np.array([1.0, 0.0, 0.0])
    x = ex - (ex @ z) * z
    n = np.linalg.norm(x)
    if n < 1e-6:
        return None
    x /= n
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _rays_cam(pts):
    """Nx2 pixels -> Nx3 unit rays in the CAMERA frame."""
    p = np.asarray(pts, float).reshape(-1, 2)
    r = np.stack([(p[:, 0] - L.CX) / L.FX, (p[:, 1] - L.CY) / L.FY,
                  np.ones(len(p))], axis=1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _orange_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (((h <= 12) | (h >= 170)) & (s > 110) & (v > 110)).astype(np.uint8)


def light_components(img, R_lb, debug=None):
    """Accepted ceiling-light components -> list of (contours, bbox).

    Each element: (list of Nx2 contour arrays in image coords, (x, y, w, h)).
    `debug`, if a list, collects (bbox, reason) for every rejection.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] < SAT_MAX) & (hsv[:, :, 2] > VAL_MIN)).astype(np.uint8)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    orange = None
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        bbox = (int(x), int(y), int(w), int(h))
        if a < MIN_AREA_PX:
            continue
        # GRAVITY FILTER: entire component above the levelled horizon. The lowest pixel
        # of the bbox is the binding one; its ray's up-component decides. This is what
        # rejects the glossy-floor reflections, and it is appearance-free.
        corners = [(x, y + h), (x + w, y + h)]      # bottom corners: lowest elevation
        rl = (_rays_cam(corners) @ _R_BC.T) @ R_lb.T
        elev = np.degrees(np.arcsin(np.clip(-rl[:, 2], -1, 1)))
        if elev.min() < HORIZON_MARGIN_DEG:
            if debug is not None:
                debug.append((bbox, 'below-horizon %.1f' % elev.min()))
            continue
        # ORANGE SURROUND: gate decoration is white paint on an orange frame.
        if orange is None:
            orange = _orange_mask(img)
        x0, y0 = max(0, x - SURROUND_PAD_PX), max(0, y - SURROUND_PAD_PX)
        x1 = min(img.shape[1], x + w + SURROUND_PAD_PX)
        y1 = min(img.shape[0], y + h + SURROUND_PAD_PX)
        region = orange[y0:y1, x0:x1]
        own = (lab[y0:y1, x0:x1] == i)
        denom = max(int((~own).sum()), 1)
        ofrac = float(region[~own].sum()) / denom
        if ofrac > ORANGE_FRAC_MAX:
            if debug is not None:
                debug.append((bbox, 'orange-surround %.2f' % ofrac))
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c.reshape(-1, 2) + np.array([x, y]) for c in cnts if len(c) >= 3]
        if cs:
            out.append((cs, bbox))
    return out


def segment_votes(contours, R_lb):
    """Contour edges -> [(theta_deg_mod90, weight), ...].

    Each polygon edge of length >= MIN_SEG_PX defines an interpretation plane through the
    camera centre; if the 3-D line is horizontal, its direction (cos t, sin t, 0) in the
    levelled frame is orthogonal to that plane's normal, so t = atan2(-n_x, n_y). Weight =
    segment length x conditioning (|horizontal part of the normal|): a plane that is
    nearly horizontal constrains the heading weakly and is down-weighted smoothly on top
    of the MIN_COND refusal.
    """
    votes = []
    R = R_lb @ _R_BC       # camera -> levelled
    for c in contours:
        approx = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2),
                                  APPROX_EPS_PX, True).reshape(-1, 2)
        k = len(approx)
        if k < 2:
            continue
        for j in range(k):
            p0, p1 = approx[j], approx[(j + 1) % k]
            length = float(np.hypot(*(p1 - p0)))
            if length < MIN_SEG_PX:
                continue
            r = _rays_cam([p0, p1])
            n_cam = np.cross(r[0], r[1])
            nn = np.linalg.norm(n_cam)
            if nn < 1e-12:
                continue
            n_lev = R @ (n_cam / nn)
            cond = float(np.hypot(n_lev[0], n_lev[1]))
            if cond < MIN_COND:
                continue
            t = math.degrees(math.atan2(-n_lev[0], n_lev[1])) % 90.0
            votes.append((t, length * cond))
    return votes


def _circular_median_mod90(votes):
    """Weighted circular median of angles mod 90 -> (theta, spread_deg).

    Median rather than mean because a minority of segments are not grid lines at all
    (bloom edges, a clipped contour corner) and a mean lets them drag. Spread is the
    weighted median absolute deviation, folded.
    """
    th = np.array([v[0] for v in votes])
    w = np.array([v[1] for v in votes])
    # coarse mode via the vector mean on 4*theta, then refine with a local median
    ang = np.deg2rad(th * 4.0)
    mx = float((w * np.cos(ang)).sum())
    my = float((w * np.sin(ang)).sum())
    centre = math.degrees(math.atan2(my, mx)) / 4.0 % 90.0
    d = (th - centre + 45.0) % 90.0 - 45.0
    o = np.argsort(d)
    cw = np.cumsum(w[o])
    med = d[o][int(np.searchsorted(cw, 0.5 * cw[-1]))]
    theta = (centre + med) % 90.0
    dd = np.abs((th - theta + 45.0) % 90.0 - 45.0)
    oo = np.argsort(dd)
    spread = dd[oo][int(np.searchsorted(np.cumsum(w[oo]), 0.5 * w.sum()))]
    return float(theta), float(spread)


def heading(frame_img, gravity_vec, debug=None):
    """Per-frame heading from the ceiling light grid.

    frame_img   : BGR image (640x360)
    gravity_vec : unit gravity DOWN in the body frame (labelgates.gravity_at convention)
    returns (theta_mod90_deg, conf) where conf = {n_comp, n_seg, weight, spread_deg,
    psi_mod90}. theta is None when no usable light edges are found.
    """
    R_lb = level_rotation(gravity_vec)
    if R_lb is None:
        return None, {'n_comp': 0, 'n_seg': 0, 'weight': 0.0, 'spread_deg': None,
                      'psi_mod90': None}
    comps = light_components(frame_img, R_lb, debug=debug)
    votes = []
    for contours, _bbox in comps:
        votes.extend(segment_votes(contours, R_lb))
    if not votes:
        return None, {'n_comp': len(comps), 'n_seg': 0, 'weight': 0.0,
                      'spread_deg': None, 'psi_mod90': None}
    theta, spread = _circular_median_mod90(votes)
    conf = {'n_comp': len(comps), 'n_seg': len(votes),
            'weight': float(sum(v[1] for v in votes)), 'spread_deg': spread,
            'psi_mod90': (-theta) % 90.0}
    return theta, conf


# ---------------------------------------------------------------------------------------
# session-level pass


class Imu:
    """imu.csv with the CONVENTIONS.md signs applied: gyro is mirrored (x -1), the
    accelerometer is canonical (no sign).

    GRAVITY IS FILTERED, NOT INSTANTANEOUS. The first build used -a/|a| per frame with a
    loose magnitude gate, and the gyro-vs-grid plot showed exactly what that costs:
    20-40 deg heading spikes during translation bursts that the gyro (correctly) does not
    see -- the accelerometer reads specific force, and a 2 m/s^2 push tilts the fake
    vertical by ~12 deg. The fix is the same complementary filter CONVENTIONS.md already
    validated for --imu-level (roll/pitch median 1.12/1.54 deg against VQ1 truth):
    propagate gravity through the body with the canonical gyro (g' = -omega x g), and pull
    toward the accelerometer only as hard as its magnitude looks like gravity.
    """

    def __init__(self, session):
        rows = L.load_csv(os.path.join(session, 'imu.csv'))
        self.t = np.array([float(r['t_wall_ns']) for r in rows])
        self.a = np.array([[float(r['xacc']), float(r['yacc']), float(r['zacc'])]
                           for r in rows])
        self.w = -np.array([[float(r['xgyro']), float(r['ygyro']), float(r['zgyro'])]
                            for r in rows])          # canonical body rates
        self.g = self._filter_gravity()

    def _filter_gravity(self):
        g = np.zeros_like(self.a)
        cur = -self.a[0] / max(np.linalg.norm(self.a[0]), 1e-9)
        t_prev = self.t[0]
        for i in range(len(self.t)):
            dt = min(max((self.t[i] - t_prev) / 1e9, 0.0), 0.1)
            t_prev = self.t[i]
            cur = cur - np.cross(self.w[i], cur) * dt
            cur /= max(np.linalg.norm(cur), 1e-9)
            n = float(np.linalg.norm(self.a[i]))
            err = abs(n - 9.81)
            # Small on purpose. A quad's accelerometer reads the THRUST axis during a
            # sustained dash -- |a| stays near 9.81 while the direction tilts with the
            # body -- so magnitude gating alone cannot catch it. The sim gyro is
            # near-noiseless (|w_z| p50 = 1e-7 rad/s parked), so the filter leans on
            # integration and takes the accelerometer with a ~2 s time constant only
            # when the magnitude looks like gravity. Swept: alpha = 0.05 left 5-15 deg
            # heading wobble during dashes; 0.008 removed it without visible drift.
            alpha = 0.008 if err < 0.35 else (0.001 if err < 1.5 else 0.0001)
            ga = -self.a[i] / max(n, 1e-9)
            cur = (1.0 - alpha) * cur + alpha * ga
            cur /= max(np.linalg.norm(cur), 1e-9)
            g[i] = cur
        return g

    def gravity(self, t_ns, tol=None):
        """Filtered unit gravity DOWN in the body frame at t_ns, + |a| for reporting.
        None only when there is no IMU sample within 0.1 s."""
        i = int(np.argmin(np.abs(self.t - t_ns)))
        n = float(np.linalg.norm(self.a[i]))
        if abs(self.t[i] - t_ns) > 0.1e9:
            return None, n
        return self.g[i], n

    def yaw_lev_integral(self, t0_ns, t1_ns):
        """Integral of the levelled (about-gravity) yaw rate over [t0, t1], degrees.
        Positive = nose right (canonical NED)."""
        if t1_ns <= t0_ns:
            return 0.0
        i0 = int(np.searchsorted(self.t, t0_ns))
        i1 = int(np.searchsorted(self.t, t1_ns))
        idx = range(max(i0 - 1, 0), min(i1 + 1, len(self.t) - 1))
        total = 0.0
        for i in idx:
            ta, tb = max(self.t[i], t0_ns), min(self.t[i + 1], t1_ns)
            if tb <= ta:
                continue
            total += float(self.g[i] @ self.w[i]) * (tb - ta) / 1e9
        return math.degrees(total)


def run_session(session, out_csv=None, stride=1, quiet=False):
    """Per-frame heading + gyro-unwrapped continuous heading -> CSV.

    The unwrap: psi (mod 90) is trusted only on confident frames (>= CONF_MIN_SEG
    segments, spread <= CONF_MAX_SPREAD_DEG); between confident frames the levelled gyro
    yaw integral picks the mod-90 branch and carries the estimate. psi_unwrapped is
    blended 100% toward the measurement on confident frames, so gyro drift never
    accumulates past one confident sighting.
    """
    name = os.path.basename(os.path.normpath(session))
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    frames = frames[::stride]
    imu = Imu(session)
    rows = []
    psi_u = None
    t_prev = None
    for k, r in enumerate(frames):
        t = float(r['t_recv_wall_ns'])
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is None:
            continue
        g, gmag = imu.gravity(t)
        theta, conf = (None, {'n_comp': 0, 'n_seg': 0, 'weight': 0.0,
                              'spread_deg': None, 'psi_mod90': None}) \
            if g is None else heading(img, g)
        dpsi = imu.yaw_lev_integral(t_prev, t) if t_prev is not None else 0.0
        confident = (theta is not None and conf['n_seg'] >= CONF_MIN_SEG
                     and conf['spread_deg'] is not None
                     and conf['spread_deg'] <= CONF_MAX_SPREAD_DEG)
        if psi_u is not None:
            psi_u += dpsi
        if confident:
            psi = conf['psi_mod90']
            if psi_u is None:
                psi_u = psi
            else:
                psi_u += (psi - psi_u + 45.0) % 90.0 - 45.0
        rows.append({
            'frame_id': r['frame_id'], 'file': r['file'], 't_recv_wall_ns': int(t),
            'gmag': round(gmag, 3),
            'theta_mod90': round(theta, 3) if theta is not None else '',
            'psi_mod90': round(conf['psi_mod90'], 3) if conf['psi_mod90'] is not None else '',
            'n_comp': conf['n_comp'], 'n_seg': conf['n_seg'],
            'weight': round(conf['weight'], 1),
            'spread_deg': round(conf['spread_deg'], 2) if conf['spread_deg'] is not None else '',
            'confident': int(confident),
            'psi_unwrapped': round(psi_u, 3) if psi_u is not None else '',
        })
        t_prev = t
        if not quiet and k % 500 == 0:
            print(f'  {k}/{len(frames)}', flush=True)
    if out_csv is None:
        out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f'skylight_{name}.csv')
    import csv as _csv
    with open(out_csv, 'w', newline='') as fh:
        wr = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    nc = sum(r['confident'] for r in rows)
    if not quiet:
        print(f'{name}: {len(rows)} frames, {nc} confident ({100*nc/len(rows):.0f}%) '
              f'-> {out_csv}')
    return out_csv, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--out', default=None)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()
    run_session(args.session, args.out, args.stride)


if __name__ == '__main__':
    main()
