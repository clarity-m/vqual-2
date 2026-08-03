"""Draw the measured map, and put it back in front of the camera.

Two outputs, and the second is the one that can actually falsify anything:

  --map      top-down plot of the measured gate positions with the sketch Procrustes-fitted
             on top, plus the height profile. Read it back; a map nobody looked at is a
             number, not a map.

  --project  REPROJECTION. VQ2 streams no pose, so there is no camera pose to project from
             -- but there is enough to recover one per frame. Gravity from HIGHRES_IMU fixes
             roll and pitch, which leaves position (3) and heading (1); three identified
             gates in one frame give nine equations for those four unknowns. Solve that,
             then project ALL SEVENTEEN map gates into the image and look at where they
             land. Gates the solve never used are the test: if the map is right they fall on
             orange, and if a gate is misplaced it falls on empty hangar.

    python3 pilot/perception/vq2render.py --map out.png
    python3 pilot/perception/vq2render.py --project proj.png --n 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402
import mapvq2 as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_map(path=None):
    d = json.load(open(path or os.path.join(HERE, 'map_vq2.json')))
    return {int(k): np.array([v['x'], v['y'], v['z_up_m']])
            for k, v in d['gates'].items()}, d


# ---------------------------------------------------------------------------------------
# top-down plot


def plot_map(pos, meta, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ks = sorted(pos)
    P = np.array([pos[k] for k in ks])
    Psk, _m = M.sketch_xy()
    B = np.array([Psk[k] for k in ks])
    F, scale, rms, refl = M.procrustes(P[:, :2], B)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    a = ax[0]
    a.plot(B[:, 0], B[:, 1], 'o--', color='0.7', ms=9, label='sketch (map_approx.json)')
    a.plot(F[:, 0], F[:, 1], 'o-', color='crimson', ms=6,
           label='MEASURED (fitted to sketch units)')
    for j, k in enumerate(ks):
        a.annotate(str(k), B[j], color='0.5', fontsize=9, xytext=(4, 4),
                   textcoords='offset points')
        a.annotate(str(k), F[j], color='crimson', fontsize=9, xytext=(4, -10),
                   textcoords='offset points')
        a.plot([B[j, 0], F[j, 0]], [B[j, 1], F[j, 1]], '-', color='0.85', lw=0.8, zorder=0)
    a.set_xlabel('along (station units)')
    a.set_ylabel('across (station units)')
    a.set_title('measured vs sketch  (%.2f m/station, rms %.1f m, reflection %s)'
                % (1 / scale, rms / scale, 'YES' if refl else 'no'))
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a.set_aspect('equal')

    b = ax[1]
    b.plot(ks, P[:, 2], 'o-', color='crimson')
    for j, k in enumerate(ks):
        b.annotate(str(k), (k, P[j, 2]), fontsize=9, xytext=(3, 4),
                   textcoords='offset points')
    b.set_xlabel('race index')
    b.set_ylabel('height above gate 0 (m, from gravity)')
    b.set_title('vertical profile -- the sketch has none')
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print('wrote', out)


# ---------------------------------------------------------------------------------------
# reprojection


def R_from_gravity(g_body, psi):
    """Body->world rotation with world z UP, given gravity-down in body and a heading psi.

    Gravity supplies two of the three degrees of freedom outright, which is the whole reason
    a VQ2 frame can be posed at all: yaw is unobservable from the IMU (CONVENTIONS.md), so
    it is left as the one free parameter and solved for against the map.
    """
    d = g_body / np.linalg.norm(g_body)          # world -z, expressed in body
    # any body vector not parallel to d
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(a @ d)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ d) * d
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    c, s = np.cos(psi), np.sin(psi)
    # columns are world axes expressed in body: x, y, z(up) = -d
    xb = c * e1 + s * e2
    yb = -s * e1 + c * e2
    R_wb = np.stack([xb, yb, -d], axis=1)        # world -> body
    return R_wb.T                                # body -> world


def solve_pose(obs, pos):
    """obs = {gate: pos_body}, plus gravity. -> (C, R_bw, psi, rms) by sweeping heading."""
    g = obs['g']
    ks = [k for k in obs if k != 'g']
    if len(ks) < 3:
        return None
    Pw = np.array([pos[k] for k in ks])
    Pb = np.array([obs[k] for k in ks])
    best = None
    for psi in np.linspace(-np.pi, np.pi, 721):
        R = R_from_gravity(g, psi)
        C = (Pw - Pb @ R.T).mean(axis=0)
        r = float(np.sqrt(((Pw - (C + Pb @ R.T)) ** 2).sum(1).mean()))
        if best is None or r < best[3]:
            best = (C, R, psi, r)
    return best


def project_sheet(session, mapfile, out, n=6, gates_needed=3):
    pos, _meta = load_map(mapfile)
    S = M.Session(session)
    good, per_frame = M.build_tracks(S)
    ident, _c, _r = M.anchor(S, good, per_frame)
    cand = []
    for i, assigned in per_frame:
        gd = {ident[t]: d['pos_body'] for t, d in assigned.items()
              if t in ident and M.clean(d) and ident[t] in pos}
        if len(gd) < gates_needed:
            continue
        gh = S.gravity(i)
        if gh is None:
            continue
        gd['g'] = gh
        cand.append((i, gd))
    print(f'{len(cand)} frames with >= {gates_needed} identified gates and steady gravity')
    if not cand:
        return
    picks = [cand[int(round(x))] for x in np.linspace(0, len(cand) - 1, min(n, len(cand)))]
    tiles = []
    for i, gd in picks:
        sol = solve_pose(gd, pos)
        if sol is None:
            continue
        C, R, psi, rms = sol
        used = {k for k in gd if k != 'g'}
        img = cv2.imread(os.path.join(session, 'frames', S.frames[i]['file']))
        if img is None:
            continue
        R_cb = L.body_to_cam()
        for k in sorted(pos):
            pb = R.T @ (pos[k] - C)
            cam = R_cb @ pb
            if cam[2] < 0.5:
                continue
            u = L.CX + L.FX * cam[0] / cam[2]
            v = L.CY + L.FY * cam[1] / cam[2]
            if not (-40 <= u < L.W + 40 and -40 <= v < L.H + 40):
                continue
            half = L.FX * 0.75 / cam[2]
            col = (0, 255, 255) if k in used else (60, 60, 255)
            cv2.rectangle(img, (int(u - half), int(v - half)),
                          (int(u + half), int(v + half)), col, 2)
            cv2.putText(img, str(k), (int(u + half) + 3, int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(img, '%s  pose rms %.2f m   yellow=used in pose, red=PREDICTED'
                    % (S.frames[i]['file'], rms), (5, 352),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(cv2.resize(img, None, fx=1.6, fy=1.6))
    if not tiles:
        print('no tiles')
        return
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + 1) // 2
    sheet = np.zeros((rows * h, 2 * w, 3), np.uint8)
    for j, t in enumerate(tiles):
        r, c = divmod(j, 2)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    cv2.imwrite(out, sheet)
    print('wrote', out, sheet.shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapfile', default=os.path.join(HERE, 'map_vq2.json'))
    ap.add_argument('--map', default='')
    ap.add_argument('--project', default='')
    ap.add_argument('--session', default=M.PRIMARY)
    ap.add_argument('--n', type=int, default=6)
    a = ap.parse_args()
    if a.map:
        p, m = load_map(a.mapfile)
        plot_map(p, m, a.map)
    if a.project:
        project_sheet(a.session, a.mapfile, a.project, a.n)
