"""Metric calibration of the ceiling light grid, and the floor-grid alignment check.

GRID PITCH. One frame gives light positions only up to the unknown ceiling height H above
the camera: a light ray r (levelled) hits the ceiling plane at H * (r_x/-r_z, r_y/-r_z).
So per frame the grid SPACING is measurable in units of H. The missing metre comes from
the strafe session (20260801-114948, gates 15/16 in view throughout): a static gate's PnP
position gives the camera's metric displacement between frames (in the grid frame, using
the skylight heading), and the same displacement is measured by the lights in units of H.
The ratio is H; pitch = spacing_ratio x H.

FLOOR ALIGNMENT. The floor markings (yellow lines) lie on the horizontal FLOOR plane, so
the same segment->heading math applies to them, just below the horizon instead of above.
If the floor-line families land on the light-grid families, the bay/column grid is
parallel to the light grid -- which is what makes 'Station N faces the aisle' usable for
quadrant disambiguation later.

    python3 skylight_pitch.py            # both measurements
"""

from __future__ import annotations

import collections
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402
import skylight as SK  # noqa: E402
import mapvq2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.abspath(os.path.join(HERE, '..', 'sessions'))
STRAFE = os.path.join(SESS, '20260801-114948-vq2-strafing-15-16')


def wrap90(x):
    return (np.asarray(x, float) + 45.0) % 90.0 - 45.0


def light_centres_grid(img, R_lb, theta):
    """Accepted single-light components -> (grid-frame xy in units of H, image centre).

    Grid frame: levelled frame rotated by -theta so the grid axes are the coordinate
    axes (mod 90). Positions are camera-relative, ceiling at height H=1.
    """
    comps = SK.light_components(img, R_lb)
    ct, st = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    Rz = np.array([[ct, st], [-st, ct]])       # levelled -> grid (rotate by -theta)
    out = []
    for contours, (x, y, w, h) in comps:
        if not (6 <= w <= 240 and 6 <= h <= 160):
            continue
        # border-clipped light: its bbox centre lags the true centre as the light
        # enters/leaves the frame -- a systematic motion bias, not noise. Refuse.
        if x <= 2 or y <= 2 or x + w >= L.W - 2 or y + h >= L.H - 2:
            continue
        area = sum(cv2.contourArea(c.astype(np.float32)) for c in contours[:1])
        if area < 20:
            continue
        c_img = np.array([x + w / 2.0, y + h / 2.0])
        r_lev = (SK._rays_cam([c_img])[0] @ SK._R_BC.T) @ R_lb.T
        if r_lev[2] > -0.15:      # too close to the horizon: huge lever arm, skip
            continue
        p = np.array([r_lev[0] / -r_lev[2], r_lev[1] / -r_lev[2]])
        out.append((Rz @ p, c_img))
    return out


def grid_spacing(session=STRAFE, stride=5, limit=4000):
    """Histogram of consecutive gaps between light centres along each grid axis, over
    many frames -> spacing in units of H (per axis)."""
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    imu = SK.Imu(session)
    gaps = {0: [], 1: []}
    for r in frames[:limit:stride]:
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is None:
            continue
        g, _n = imu.gravity(float(r['t_recv_wall_ns']))
        if g is None:
            continue
        R_lb = SK.level_rotation(g)
        theta, conf = SK.heading(img, g)
        if theta is None or conf['n_seg'] < SK.CONF_MIN_SEG or \
                conf['spread_deg'] > 6.0:
            continue
        pts = [p for p, _c in light_centres_grid(img, R_lb, theta)]
        if len(pts) < 3:
            continue
        P = np.array(pts)
        for ax in (0, 1):
            # lights in one ROW share the other coordinate; group first, then gap
            other = P[:, 1 - ax]
            order = np.argsort(other)
            groups, cur = [], [order[0]]
            for j in order[1:]:
                if abs(other[j] - other[cur[-1]]) < 0.12:
                    cur.append(j)
                else:
                    groups.append(cur)
                    cur = [j]
            groups.append(cur)
            for grp in groups:
                if len(grp) < 2:
                    continue
                v = np.sort(P[grp, ax])
                gaps[ax].extend(np.diff(v).tolist())
    out = {}
    for ax in (0, 1):
        v = np.array([x for x in gaps[ax] if 0.05 < x < 3.0])
        if len(v) < 20:
            out[ax] = (None, len(v))
            continue
        # fundamental spacing: mode of the histogram's first peak. Multiples (a skipped
        # light) contaminate the tail, so take the mode of values under 2x the p20.
        base = np.percentile(v, 20)
        vv = v[v < 2.0 * base]
        hist, edges = np.histogram(vv, bins=40)
        k = int(np.argmax(hist))
        peak = 0.5 * (edges[k] + edges[k + 1])
        sel = vv[np.abs(vv - peak) < 0.25 * peak]
        out[ax] = (float(np.median(sel)), len(sel))
    return out


def height_from_strafe(session=STRAFE, stride=1, baseline_s=0.8, min_net=0.25,
                       max_steps=120):
    """Ceiling height H above the camera, from metric camera motion (gate PnP) vs
    light-grid motion (units of H) over the strafe.

    V2, after the first attempt returned MAD 4.5 m with negative p10. Two faults, both of
    the measurement-setup kind: (i) matching lights between frames ~1 s apart ALIASES on
    a periodic grid -- the camera moves about half a grid cell per second, so nearest-
    neighbour matches snap to the wrong light and bias the motion toward zero; (ii) gate
    PnP jitter over a 1 s baseline is comparable to the baseline itself. Fix: match
    lights only frame-to-frame (motion per frame << cell), CHAIN the increments so the
    light-side displacement never aliases, and take the gate-side displacement between
    window ENDPOINTS (smoothed over +/-7 frames) so intermediate PnP noise telescopes
    away."""
    obs = _strafe_obs(session)
    return _height_windows(obs, min_net, max_steps)


_OBS_CACHE = {}


def _strafe_obs(session):
    """Per-frame extraction only (the expensive pass), cached per session."""
    if session in _OBS_CACHE:
        return _OBS_CACHE[session]
    S = mapvq2.Session(session)
    imu = SK.Imu(session)
    obs = []       # per usable frame: t, psi_lev (theta), cam-from-gate (levelled, m),
    #                light constellation in grid units, gate range
    for i, r in enumerate(S.frames):
        det = [d for d in S.det.get(r['file'], []) if mapvq2.clean(d)]
        if not det:
            obs.append(None)
            continue
        d = max(det, key=lambda d: d['size_px'])
        t = float(r['t_recv_wall_ns'])
        g, _n = imu.gravity(t)
        if g is None:
            obs.append(None)
            continue
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is None:
            obs.append(None)
            continue
        R_lb = SK.level_rotation(g)
        theta, conf = SK.heading(img, g)
        if theta is None or conf['n_seg'] < SK.CONF_MIN_SEG or conf['spread_deg'] > 6.0:
            obs.append(None)
            continue
        v_lev = R_lb @ np.asarray(d['pos_body'], float)
        # store LEVELLED coordinates + theta; the grid rotation is applied pairwise so
        # the mod-90 branch can be matched between the two frames (theta = 1 deg and
        # theta = 89 deg are one grid, but naive per-frame rotation would swap the axes)
        lights = light_centres_grid(img, R_lb, 0.0)
        obs.append({'t': t, 'theta': theta, 'gate_lev': v_lev[:2], 'lights': lights,
                    'range': float(d['range_m']), 'centre': d['centre']})
    _OBS_CACHE[session] = obs
    return obs


def _height_windows(obs, min_net, max_steps):
    def rz(th):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        return np.array([[c, s], [-s, c]])

    # continuous theta branch along the whole session, so grid axes never swap mid-chain
    idx = [i for i, o in enumerate(obs) if o]
    if not idx:
        return []
    cont = {}
    th = obs[idx[0]]['theta']
    for i in idx:
        th = th + float(wrap90(obs[i]['theta'] - th))
        cont[i] = th

    # frame-to-frame light motion, chained (aliasing-free: per-frame motion << cell)
    step_mv = {}          # i -> median light motion from frame i to next usable frame
    for a_i in range(len(idx) - 1):
        i, j = idx[a_i], idx[a_i + 1]
        if (obs[j]['t'] - obs[i]['t']) > 0.2e9:
            continue
        RA, RB = rz(cont[i]), rz(cont[j])
        moves = []
        for pa_lev, _ca in obs[i]['lights']:
            pa = RA @ pa_lev
            best, bd = None, 1e9
            for pb_lev, _cb in obs[j]['lights']:
                pb = RB @ pb_lev
                d2 = float(np.linalg.norm(pa - pb))
                if d2 < bd:
                    best, bd = pb, d2
            if best is not None and bd < 0.08:
                moves.append(best - pa)
        if len(moves) >= 2:
            step_mv[i] = (j, np.median(np.array(moves), axis=0))

    def gate_grid_smooth(a_i):
        """Gate position in grid coords, averaged over +/-7 usable frames."""
        pts = []
        for b_i in range(max(0, a_i - 7), min(len(idx), a_i + 8)):
            k = idx[b_i]
            pts.append(rz(cont[k]) @ obs[k]['gate_lev'])
        return np.median(np.array(pts), axis=0)

    ratios = []
    for a_i in range(0, len(idx) - 10, 10):
        # extend the chain until the light constellation has moved >= 0.25 units of H
        net = np.zeros(2)
        b_i = a_i
        ok = True
        while b_i < len(idx) - 1:
            i = idx[b_i]
            if i not in step_mv or step_mv[i][0] != idx[b_i + 1]:
                ok = False
                break
            net = net + step_mv[i][1]
            b_i += 1
            if np.linalg.norm(net) >= min_net or (b_i - a_i) > max_steps:
                break
        if not ok or np.linalg.norm(net) < 0.5 * min_net:
            continue
        # gate continuity across the window (same physical gate at both ends)
        rA, rB = obs[idx[a_i]]['range'], obs[idx[b_i]]['range']
        if abs(rA - rB) > 8.0:
            continue
        dcam_m = gate_grid_smooth(a_i) - gate_grid_smooth(b_i)
        if np.linalg.norm(dcam_m) < 0.8:
            continue
        num = float(dcam_m @ -net)
        den = float(net @ net)
        ratios.append({'H': num / den, 'base_m': float(np.linalg.norm(dcam_m)),
                       'n_steps': b_i - a_i})
    return ratios


def floor_alignment(session='20260801-144858-vqual2-lap-pausing', stride=10, limit=None):
    """Angle between the yellow floor-line families and the light-grid heading, per
    frame, folded mod 90. Median near 0 (or 45) says the bay grid is (dis)aligned."""
    sess = os.path.join(SESS, session)
    frames = [r for r in L.load_csv(os.path.join(sess, 'frames.csv')) if r['file']]
    imu = SK.Imu(sess)
    diffs = []
    for r in frames[:limit:stride]:
        img = cv2.imread(os.path.join(sess, 'frames', r['file']))
        if img is None:
            continue
        g, _n = imu.gravity(float(r['t_recv_wall_ns']))
        if g is None:
            continue
        R_lb = SK.level_rotation(g)
        theta, conf = SK.heading(img, g)
        if theta is None or conf['n_seg'] < 8 or conf['spread_deg'] > 4.0:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ym = ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] > 90)
              & (hsv[:, :, 2] > 70)).astype(np.uint8) * 255
        segs = cv2.HoughLinesP(ym, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=4)
        if segs is None:
            continue
        R = R_lb @ SK._R_BC
        votes = []
        for x0, y0, x1, y1 in segs.reshape(-1, 4):
            pts = SK._rays_cam([(x0, y0), (x1, y1)])
            rl = pts @ R.T
            # floor lines must be BELOW the horizon (the mirror of the light filter)
            elev = np.degrees(np.arcsin(np.clip(-rl[:, 2], -1, 1)))
            if elev.max() > -2.0:
                continue
            n_cam = np.cross(pts[0], pts[1])
            nn = np.linalg.norm(n_cam)
            if nn < 1e-12:
                continue
            n_lev = R @ (n_cam / nn)
            cond = float(np.hypot(n_lev[0], n_lev[1]))
            if cond < SK.MIN_COND:
                continue
            tt = math.degrees(math.atan2(-n_lev[0], n_lev[1])) % 90.0
            votes.append((tt, np.hypot(x1 - x0, y1 - y0) * cond))
        if len(votes) < 3:
            continue
        tf, spread = SK._circular_median_mod90(votes)
        if spread > 6.0:
            continue
        diffs.append(float(wrap90(tf - theta)))
    return np.array(diffs)


def main():
    print('=' * 78)
    print('FLOOR-GRID ALIGNMENT (yellow floor lines vs light grid, mod 90):')
    d = floor_alignment()
    if len(d):
        print(f'  n={len(d)} frames: median {np.median(d):+.2f} deg, '
              f'MAD {np.median(np.abs(d - np.median(d))):.2f}, p90(|.|) '
              f'{np.percentile(np.abs(d), 90):.2f}')
    else:
        print('  no usable frames')

    print('=' * 78)
    print('GRID SPACING in units of ceiling height H (strafe session):')
    sp = grid_spacing()
    for ax, (s, n) in sp.items():
        print(f'  axis {ax}: spacing/H = {s if s is None else round(s, 4)}   (n={n})')

    print('=' * 78)
    print('CEILING HEIGHT H above camera, gate-PnP vs light-motion (strafe):')
    ratios = height_from_strafe()
    if ratios:
        H = np.array([x['H'] for x in ratios])
        med = float(np.median(H))
        print(f'  n={len(H)} frame pairs: H median {med:.2f} m, '
              f'MAD {np.median(np.abs(H - med)):.2f}, p10 {np.percentile(H, 10):.2f}, '
              f'p90 {np.percentile(H, 90):.2f}')
        for ax, (s, n) in sp.items():
            if s:
                print(f'  -> grid pitch axis {ax}: {s * med:.2f} m')
    else:
        print('  no usable frame pairs')


if __name__ == '__main__':
    main()
