"""SECOND INDEPENDENT REFEREE: leg odometry against the map.

The gate-PnP range-closure referee is noisy -- PnP range error is ~3% of range, which at
15 m swamps one frame of closure -- so it can only be run over windows and it yields few
independent samples on the sessions we have.

This one is far better conditioned and uses nothing the fit touched. Between two
consecutive gate crossings the aircraft flies from gate k to gate k+1, whose separation the
map MEASURED (mapedges, 20260802 strafe sessions). Integrating the drag speed over that
leg gives the PATH length. Two facts make it a test:

    path >= chord            always, with equality only for a perfectly straight leg
    path is UNDERSTATED      because the estimate carries only the in-plane component

so the ratio path/chord must sit at or slightly above 1, and it is proportional to 1/sqrt(k)
-- a coefficient wrong by 4x moves the ratio by 2x, which is impossible to miss. Nothing
here uses VQ1, pose telemetry, or the camera's range: only crossing times from the race
packet, the accelerometer, and the map's measured edge lengths.
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import dragmodel as DM   # noqa: E402

SESS = os.path.join(os.path.dirname(HERE), 'sessions')


def map_dist():
    import json
    m = json.load(open(os.path.join(HERE, 'map_vq2.json')))
    pairs = {tuple(sorted(int(x) for x in k.split('-'))): v['dist_m']
             for k, v in m.get('measured_pairs', {}).items()}
    ld = m['layout_directions_2026_08_02']['positions_m']
    P = {int(k): np.array([v['x'], v['y'], v['z_up_m']]) for k, v in ld.items()}

    def d(i, j):
        key = (min(i, j), max(i, j))
        if key in pairs:
            return pairs[key]
        if i in P and j in P:
            return float(np.linalg.norm(P[j] - P[i]))
        return None
    return d


def legs(session, k=DM.K_DRAG):
    d_map = map_dist()
    imu_t, acc = [], []
    with open(os.path.join(SESS, session, 'imu.csv'), newline='') as f:
        for r in csv.DictReader(f):
            try:
                imu_t.append(float(r['t_wall_ns']) / 1e9)
                acc.append((float(r['xacc']), float(r['yacc']), float(r['zacc'])))
            except (TypeError, ValueError):
                pass
    imu_t = np.asarray(imu_t)
    v, conf, _ = DM.inplane_batch(np.asarray(acc), k=k)
    dt = np.gradient(imu_t)

    rt, ai = [], []
    with open(os.path.join(SESS, session, 'race.csv'), newline='') as f:
        for r in csv.DictReader(f):
            try:
                rt.append(float(r['t_wall_ns']) / 1e9)
                ai.append(int(r['active_gate_index']))
            except (TypeError, ValueError):
                pass
    rt, ai = np.asarray(rt), np.asarray(ai)
    # discard everything before the final usable reset -- a reset is the index dropping,
    # and only one that leaves >= 5 s behind it counts (a session can END on a reset)
    drops = [i for i in range(1, len(ai))
             if ai[i] < ai[i - 1] and rt[-1] - rt[i] >= 5.0]
    if drops:
        keep = rt >= rt[drops[-1]]
        rt, ai = rt[keep], ai[keep]
    ch = np.nonzero(np.diff(ai))[0]
    out = []
    for a, b in zip(ch[:-1], ch[1:]):
        if ai[b + 1] != ai[a + 1] + 1:
            continue                                  # not a clean sequential advance
        t0, t1 = rt[a + 1], rt[b + 1]
        g0, g1 = ai[a + 1] - 1, ai[b + 1] - 1         # the gate just crossed at each end
        chord = d_map(g0, g1)
        if chord is None or not (0.5 < t1 - t0 < 30.0):
            continue
        m = (imu_t >= t0) & (imu_t < t1)
        if m.sum() < 10:
            continue
        path = float(np.sum(v[m] * dt[m]))
        out.append((g0, g1, t1 - t0, chord, path, path / chord))
    return out


def main():
    sessions = sys.argv[1:] or [
        '20260802-153626-vm-fast-lap-0-10-with-collisions',
        '20260801-121520-vq2-lap-0-15',
        '20260801-115735-vq2-lap-0-11',
    ]
    print('=== leg odometry (drag speed integrated) vs measured map edge ===')
    print('    path/chord must be >= 1 and is UNDERSTATED (no vertical component).')
    print('    ratio scales as 1/sqrt(k): 4x error in k -> 2x error here.')
    allr = []
    for s in sessions:
        if not os.path.isdir(os.path.join(SESS, s)):
            print('  %s: not present' % s)
            continue
        L = legs(s)
        if not L:
            print('  %-46s no clean legs' % s[:46])
            continue
        r = np.array([x[5] for x in L])
        print()
        print('  %s  (%d legs)' % (s, len(L)))
        print('    leg     dur    chord     path   ratio')
        for g0, g1, dur, chord, path, ratio in L:
            print('    %2d-%-2d %6.2f %8.2f %8.2f %7.2f' % (g0, g1, dur, chord, path, ratio))
        print('    ratio: median %.2f  p10 %.2f  p90 %.2f  frac>=0.9 %.2f'
              % (np.median(r), np.percentile(r, 10), np.percentile(r, 90),
                 float(np.mean(r >= 0.9))))
        allr.append(r)
    if allr:
        r = np.concatenate(allr)
        print()
        print('  POOLED n=%d  median ratio %.2f  (p10 %.2f, p90 %.2f)'
              % (len(r), np.median(r), np.percentile(r, 10), np.percentile(r, 90)))
        print()
        print('  sensitivity of the pooled median to k:')
        for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
            rr = []
            for s in sessions:
                if not os.path.isdir(os.path.join(SESS, s)):
                    continue
                L = legs(s, k=DM.K_DRAG * mult)
                rr += [x[5] for x in L]
            if rr:
                print('    k x %-5.2f (k = %.5f)  median path/chord = %.2f'
                      % (mult, DM.K_DRAG * mult, float(np.median(rr))))


if __name__ == '__main__':
    main()
