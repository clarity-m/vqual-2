"""Build a relative gate map from VQ2 vision alone. No pose telemetry anywhere.

VQ2 blocks ATTITUDE, LOCAL_POSITION_NED, ODOMETRY and gate geometry, so a world-frame map
is unobtainable from telemetry. It is obtainable from the CAMERA, which the spec permits,
and information derived from a permitted sensor is in bounds -- unlike absolute yaw, which
would come from a blocked stream by way of the actuator.

THE STRUCTURE, and why there is no odometry in it:

Each detected gate is a known 1500 mm square, so solvePnP returns the FULL pose of the
camera relative to that gate. Two gates seen in the SAME FRAME therefore give their
relative transform directly:

    T_A_B  =  T_cam_A^-1 @ T_cam_B

No dead reckoning, no drift, no IMU. Measured on 20260730-215221: 80% of frames carry two
or more gates, so edges are abundant. Chaining edges gives a map that is RELATIVE by
construction, which is all the guidance ever needs -- "where is gate k+1 relative to gate
k" -- and it sidesteps global scale drift, accumulated yaw error and loop closure, none of
which a relative query can see.

IDENTITY comes from temporal tracking, not from the race packet. active_gate_index names
only the gate being flown, while a frame may hold five; and the sim renders without motion
blur, so a detection moves smoothly between 30 Hz frames and nearest-neighbour association
is reliable. Tracks are the map's nodes.

    python3 pilot/perception/mapbuild.py <session-dir> [--limit N] [--out FILE]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402
import detect as D  # noqa: E402

MATCH_PX_BASE = 40.0     # association radius floor, px
MATCH_SIZE_RATIO = 2.0   # apparent size may not jump by more than this between frames
MAX_GAP = 4              # frames a track may coast before it is closed
MIN_TRACK_LEN = 6        # tracks shorter than this are noise, not gates
MIN_EDGE_OBS = 5         # relative transforms observed fewer times than this are dropped


def pose_in_cam(d):
    """4x4 camera-from-gate transform for a detection."""
    T = np.eye(4)
    # detect.py stores body-frame vectors; undo that to get back to the camera frame,
    # because the relative transform between two gates must be taken in ONE frame and
    # the camera is the only frame both detections share.
    R_cb = L.body_to_cam()
    T[:3, 3] = R_cb @ d['pos_body']
    n = R_cb @ d['normal_body']
    n = n / max(np.linalg.norm(n), 1e-9)
    # Complete an orthonormal basis around the gate normal. The in-plane rotation is not
    # recoverable from a symmetric square anyway (4-fold ambiguous), so only the normal
    # carries information here and the other two axes are an arbitrary consistent choice.
    a = np.array([0.0, 1.0, 0.0])
    if abs(float(n @ a)) > 0.9:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, n); u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(n, u)
    T[:3, :3] = np.stack([u, v, n], axis=1)
    return T


def track(session, start, count, stride, drop_tail_s):
    """Sequential detection + nearest-neighbour association -> gate tracks.

    MUST run on CONSECUTIVE frames. Association assumes a detection moves a little
    between samples; subsampling a long session to spread coverage breaks that outright.
    First attempt here sampled every 28th frame and produced ZERO tracks -- ~0.9 s of
    gate motion at 30 Hz is far past any sane association radius. Cover a session with
    several contiguous WINDOWS, never with a stride.
    """
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    # DROP THE TAIL. The last seconds of a session are the post-run reset: the aircraft is
    # teleported and the view jumps, which fabricates enormous apparent gate motion and
    # would splice unrelated gates into one track. Claire's rule, made a default here
    # rather than a thing to remember.
    if drop_tail_s > 0 and len(frames) > 2:
        t = np.array([float(r['t_recv_wall_ns']) for r in frames]) / 1e9
        frames = [f for f, tt in zip(frames, t) if tt <= t[-1] - drop_tail_s]
    end = min(len(frames), start + count) if count else len(frames)
    idxs = list(range(start, end, stride))

    tracks = {}          # tid -> dict(last_i, centre, size, obs=[(i, det)])
    live = {}
    next_tid = 0
    per_frame = []       # (frame_index, {tid: det})

    for k, i in enumerate(idxs):
        img = cv2.imread(os.path.join(session, 'frames', frames[i]['file']))
        if img is None:
            continue
        det = D.detections(img)
        assigned = {}
        used = set()
        # Greedy nearest-neighbour, largest detections first: big gates are the reliable
        # ones and should claim their track before a small ambiguous one can steal it.
        for d in sorted(det, key=lambda x: -x['size_px']):
            best, bd = None, 1e18
            for tid, t in live.items():
                if tid in used or k - t['last_k'] > MAX_GAP:
                    continue
                dist = float(np.linalg.norm(t['centre'] - d['centre']))
                ratio = max(d['size_px'], 1e-6) / max(t['size'], 1e-6)
                if ratio > MATCH_SIZE_RATIO or ratio < 1.0 / MATCH_SIZE_RATIO:
                    continue
                # Radius scales with apparent size: a near gate sweeps many more pixels
                # per frame than a distant one, and a fixed radius suits neither.
                lim = max(MATCH_PX_BASE, 1.2 * t['size']) * max(1, k - t['last_k'])
                if dist < lim and dist < bd:
                    best, bd = tid, dist
            if best is None:
                best = next_tid
                next_tid += 1
                live[best] = {'last_k': k, 'centre': d['centre'], 'size': d['size_px'],
                              'obs': []}
                tracks[best] = live[best]
            used.add(best)
            live[best].update(last_k=k, centre=d['centre'], size=d['size_px'])
            live[best]['obs'].append((i, d))
            assigned[best] = d
        for tid in [t for t, v in live.items() if k - v['last_k'] > MAX_GAP]:
            del live[tid]
        per_frame.append((i, assigned))

    good = {t: v for t, v in tracks.items() if len(v['obs']) >= MIN_TRACK_LEN}
    return good, per_frame


def edges(good, per_frame):
    """Inter-gate DISTANCES, one measurement per co-visible frame.

    Distance, not relative transform, and the reason is worth keeping. A relative
    transform T_A_B = inv(T_A) @ T_B needs a frame attached to gate A that is consistent
    across frames, and we do not have one: a square is 4-fold symmetric so PnP's in-plane
    rotation flips as the gate turns in image, and reconstructing a basis from the normal
    alone leaves an arbitrary in-plane axis that does NOT cancel in the product -- it
    rotates the answer directly. An earlier version of this file did exactly that and
    produced a map whose gates all sat within 7 m of each other.

    |t_B - t_A| has no orientation in it. It is invariant to how the camera is held, so
    every co-visible frame measures the same scalar and they can simply be averaged. The
    price is that distances alone fix the map only up to rotation AND REFLECTION, which
    is fine here: guidance asks "how far is gate k+1 from gate k", never "where is north".
    """
    acc = collections.defaultdict(list)
    for _i, assigned in per_frame:
        ids = [t for t in assigned if t in good]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ta = L.body_to_cam() @ assigned[ids[a]]['pos_body']
                tb = L.body_to_cam() @ assigned[ids[b]]['pos_body']
                dist = float(np.linalg.norm(tb - ta))
                if np.isfinite(dist) and dist > 0.1:
                    acc[tuple(sorted((ids[a], ids[b])))].append(dist)
    out = {}
    for k, v in acc.items():
        if len(v) >= MIN_EDGE_OBS:
            arr = np.asarray(v)
            med = float(np.median(arr))
            # Median absolute deviation: the scatter of one edge measured from many
            # viewpoints. This IS the map's precision, and nothing in the estimator can
            # flatter it -- disagreeing measurements of the same fixed distance are
            # simply error.
            out[k] = (med, len(arr), float(np.median(np.abs(arr - med))))
    return out


def solve(ed):
    """Distance matrix -> 3-D coordinates by classical MDS.

    Unobserved pairs are filled by shortest path through observed ones, which is exact
    for points strung along a path and an over-estimate when it must detour. Gates on a
    course are close to a path, so it is a reasonable completion; the stress reported by
    main() is what says whether it held.
    """
    nodes = sorted({n for e in ed for n in e})
    if len(nodes) < 4:
        return {}, None, nodes
    idx = {n: i for i, n in enumerate(nodes)}
    m = len(nodes)
    Dm = np.full((m, m), np.inf)
    np.fill_diagonal(Dm, 0.0)
    for (a, b), (d, _n, _s) in ed.items():
        Dm[idx[a], idx[b]] = Dm[idx[b], idx[a]] = d
    observed = np.isfinite(Dm).copy()
    for k in range(m):                       # Floyd-Warshall completion
        Dm = np.minimum(Dm, Dm[:, k, None] + Dm[None, k, :])
    if not np.all(np.isfinite(Dm)):          # graph is disconnected; keep the big piece
        reach = np.isfinite(Dm).sum(axis=1)
        keep = np.where(reach == reach.max())[0]
        nodes = [nodes[i] for i in keep]
        Dm = Dm[np.ix_(keep, keep)]
        observed = observed[np.ix_(keep, keep)]
        m = len(nodes)
        if m < 4:
            return {}, None, nodes
    D2 = Dm ** 2
    J = np.eye(m) - np.ones((m, m)) / m
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:3]
    X = V[:, order] * np.sqrt(np.clip(w[order], 0, None))
    return {n: X[i] for i, n in enumerate(nodes)}, (Dm, observed, nodes), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=3000)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--drop-tail', type=float, default=5.0,
                    help='seconds to discard from the end (post-run reset)')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    good, per_frame = track(args.session, args.start, args.count, args.stride, args.drop_tail)
    lens = sorted((len(v['obs']) for v in good.values()), reverse=True)
    print(f'frames processed : {len(per_frame)}')
    print(f'tracks (>= {MIN_TRACK_LEN} obs): {len(good)}   lengths {lens[:12]}')

    ed = edges(good, per_frame)
    print(f'edges (>= {MIN_EDGE_OBS} obs) : {len(ed)}')
    if not ed:
        return
    dd = np.array([v[0] for v in ed.values()])
    sp = np.array([v[2] for v in ed.values()])
    print(f'  edge length : median {np.median(dd):6.1f} m   p10 {np.percentile(dd,10):.1f}'
          f'   p90 {np.percentile(dd,90):.1f}')
    print(f'  edge spread : median {np.median(sp):6.2f} m   p90 {np.percentile(sp,90):.2f}'
          f"   <- same edge, different viewpoints: the map precision")

    pos, aux, _ = solve(ed)
    print(f'\nnodes placed : {len(pos)} of {len(good)} tracks')
    if aux is not None and pos:
        Dm, observed, nodes = aux
        X = np.stack([pos[n] for n in nodes])
        R = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        iu = np.triu_indices(len(nodes), 1)
        err = np.abs(R[iu] - Dm[iu])[observed[iu]]
        if err.size:
            print(f'MDS stress on OBSERVED edges: median {np.median(err):.2f} m  '
                  f'p90 {np.percentile(err,90):.2f} m  over {err.size} pairs')
            print('  (how far the reconstruction must bend the measured distances;\n'
                  '   large stress means they are mutually inconsistent -- tracks that\n'
                  '   are not really distinct gates, or bad detections)')
    for t in sorted(pos, key=lambda t: -len(good[t]['obs']))[:12]:
        print('  track %3d  obs %4d   [%8.2f %7.2f %7.2f]'
              % (t, len(good[t]['obs']), *pos[t]))

    if args.out:
        json.dump({'nodes': {str(t): pos[t].tolist() for t in pos},
                   'n_obs': {str(t): len(good[t]['obs']) for t in pos}},
                  open(args.out, 'w'), indent=1)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
