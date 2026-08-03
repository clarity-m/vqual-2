"""Fit a constant per-gate 3D position error from the image-space offsets.

If a gate's surveyed centre is wrong by a constant world vector Delta, every matched
instance's offset o = label_centre - detection_centre satisfies o ~ J(pose) @ Delta,
where J is the 2x3 Jacobian of the projection at that pose. Stacking all instances of
one gate gives an overdetermined linear system; the residual says how much of the
observed bias a constant 3D error actually explains. This uses EVERY viewpoint, unlike
the active-window refit, so it is an independent estimate of the same quantity.

IRLS with a Huber weight so merged-blob / ribbon-corrupted detections do not drag it.

    python3 pilot/perception/gatebias_delta3d.py [--csv gatebias_offsets.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402
import autolabel as A  # noqa: E402

SESSIONS = os.path.join(os.path.dirname(HERE), 'sessions')


def jacobian(gate_w, pose):
    """2x3 d(uv)/d(gate world position) at this pose."""
    p, roll, pitch, yaw = pose
    R_bw = L.euler_to_R(roll, pitch, yaw)
    R_cb = L.body_to_cam()
    R_cw = R_cb @ R_bw.T                     # world -> camera
    cam = R_cw @ (np.asarray(gate_w) - p)
    x, y, z = cam
    Juv = np.array([[L.FX / z, 0.0, -L.FX * x / z**2],
                    [0.0, L.FY / z, -L.FY * y / z**2]])
    return Juv @ R_cw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(HERE, 'gatebias_offsets.csv'))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    gates = A.load_gate_set(os.path.join(HERE, 'gates_refined.json'))

    # pose per (session, file): sim-time from frames.csv, PoseTrack per session
    sessions = sorted({r['session'] for r in rows})
    tracks, ftime = {}, {}
    for s in sessions:
        sd = os.path.join(SESSIONS, s)
        tracks[s] = L.PoseTrack(sd)
        for rec in L.load_csv(os.path.join(sd, 'frames.csv')):
            if rec['file']:
                ftime[(s, rec['file'])] = float(rec['sim_time_ns']) / 1e9

    print(f'{len(rows)} offset rows, sessions: {sessions}')
    for g in sorted({int(r['gate']) for r in rows}):
        Js, os_ = [], []
        for r in rows:
            if int(r['gate']) != g:
                continue
            s = r['session']
            fname = r['key'].split('/', 1)[1]
            t = ftime.get((s, fname))
            if t is None:
                continue
            pose = tracks[s].at_time(t)
            if pose is None:
                continue
            Js.append(jacobian(gates[g], pose))
            os_.append([float(r['dx']), float(r['dy'])])
        if len(os_) < 10:
            print(f'gate {g}: only {len(os_)} usable rows, skipped')
            continue
        Jm = np.concatenate(Js)              # (2n, 3)
        om = np.asarray(os_).reshape(-1)     # (2n,)
        w = np.ones(len(om))
        delta = np.zeros(3)
        for _ in range(8):                   # IRLS, Huber k=2 px
            W = np.repeat(w, 1)
            delta, *_ = np.linalg.lstsq(Jm * W[:, None], om * W, rcond=None)
            res = om - Jm @ delta
            aw = np.abs(res)
            w = np.where(aw <= 2.0, 1.0, np.sqrt(2.0 / aw))
        res2 = (om - Jm @ delta).reshape(-1, 2)
        pre = np.hypot(*np.asarray(os_).T)
        post = np.hypot(res2[:, 0], res2[:, 1])
        print(f'gate {g}: n={len(os_):4d}  Delta_world = '
              f'[{delta[0]:+6.2f} {delta[1]:+6.2f} {delta[2]:+6.2f}] m  '
              f'|{np.linalg.norm(delta):5.2f}|   median |off| {np.median(pre):.2f} -> '
              f'{np.median(post):.2f} px after removing it')


if __name__ == '__main__':
    main()
