"""Render skylight detections: accepted light components (green, with levelled grid
directions drawn), rejections coloured by reason (red = below horizon i.e. reflection,
orange = gate decoration). Read back by eye before trusting anything downstream.

    python3 skylight_render.py <session> <frame.jpg> [<frame.jpg> ...] [--out sheet.png]
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
import skylight as SK  # noqa: E402


def render_one(session, fname, imu=None):
    frames = {r['file']: r for r in L.load_csv(os.path.join(session, 'frames.csv'))
              if r['file']}
    if imu is None:
        imu = SK.Imu(session)
    r = frames[fname]
    img = cv2.imread(os.path.join(session, 'frames', fname))
    t = float(r['t_recv_wall_ns'])
    g, gmag = imu.gravity(t)
    vis = img.copy()
    if g is None:
        cv2.putText(vis, f'NO GRAVITY |a|={gmag:.1f}', (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return vis
    R_lb = SK.level_rotation(g)
    rejects = []
    comps = SK.light_components(img, R_lb, debug=rejects)
    votes = []
    for contours, bbox in comps:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        for c in contours:
            cv2.polylines(vis, [c.astype(np.int32).reshape(-1, 1, 2)], True,
                          (0, 200, 0), 1)
        votes.extend(SK.segment_votes(contours, R_lb))
    for bbox, reason in rejects:
        x, y, w, h = bbox
        col = (0, 0, 255) if reason.startswith('below') else (0, 140, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 1)
        cv2.putText(vis, reason.split()[0], (x, max(8, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
    # levelled horizon line: pixels where the levelled ray elevation is zero, drawn so
    # the up-side filter is visible in the render
    R = R_lb @ SK._R_BC
    for u in range(0, L.W, 4):
        # solve for v where levelled z-component of the ray = 0 (approx by scan)
        vv = np.arange(0, L.H, 2, dtype=float)
        rays = SK._rays_cam(np.stack([np.full_like(vv, u), vv], axis=1))
        z = (rays @ R.T)[:, 2]
        j = np.argmin(np.abs(z))
        if abs(z[j]) < 0.02:
            cv2.circle(vis, (u, int(vv[j])), 1, (255, 255, 0), -1)
    theta, conf = SK.heading(img, g)
    if theta is not None:
        # draw the two levelled grid directions through each accepted component centre,
        # projected back into the image (short horizontal 3-D segments at the ray point)
        for contours, bbox in comps:
            x, y, w, h = bbox
            c = np.array([x + w / 2.0, y + h / 2.0])
            rc = SK._rays_cam([c])[0]
            r_lev = (R @ rc)
            for t_ang in (theta, theta + 90.0):
                d = np.array([math.cos(math.radians(t_ang)),
                              math.sin(math.radians(t_ang)), 0.0])
                pts = []
                for s in (-0.12, 0.12):
                    p_lev = r_lev + s * d
                    p_cam = R.T @ p_lev
                    if p_cam[2] <= 1e-6:
                        break
                    pts.append((int(L.CX + L.FX * p_cam[0] / p_cam[2]),
                                int(L.CY + L.FY * p_cam[1] / p_cam[2])))
                if len(pts) == 2:
                    cv2.line(vis, pts[0], pts[1], (255, 0, 255), 1, cv2.LINE_AA)
        txt = (f'theta={theta:.1f} psi={conf["psi_mod90"]:.1f} '
               f'nseg={conf["n_seg"]} spread={conf["spread_deg"]:.1f}')
    else:
        txt = f'no heading (ncomp={conf["n_comp"]})'
    cv2.putText(vis, txt, (6, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(vis, fname, (480, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('frames', nargs='+')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    imu = SK.Imu(args.session)
    tiles = [render_one(args.session, f, imu) for f in args.frames]
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * L.H, cols * L.W, 3), np.uint8)
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sheet[rr * L.H:(rr + 1) * L.H, cc * L.W:(cc + 1) * L.W] = t
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'vercheck', 'skylight_render.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, sheet)
    print('wrote', out)


if __name__ == '__main__':
    main()
