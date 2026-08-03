"""Verification strips: coast->reacquire events, and Claire's slot-swap regime.

Two things are drawn that the ordinary annotate() does not show, because they are exactly
the things under test:

  COAST events -- at the moment a track is re-acquired after a coast gap, the COASTED
  prediction (where the producer thought the gate was) is drawn against the NEW
  measurement. Under rotation-only coasting the prediction lags along the direction of
  travel; if the translation fix works, the two converge.

  SLOT regime -- frames with active_gate_index == 1, both lookahead slots valid and a high
  yaw rate: the configuration Claire caught the swap in. Each slot is labelled with its
  range and, next to it, the map distance it OUGHT to have from the current gate, so a
  swap is readable directly off the tile.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import producer as PR   # noqa: E402

W, H = PR.W, PR.H


def draw_coast(img, pred_uv, meas_uv, gap, miss_m, miss_px, speed, tag):
    im = img.copy()
    if pred_uv is not None:
        p = (int(pred_uv[0]), int(pred_uv[1]))
        cv2.drawMarker(im, p, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.putText(im, 'coasted', (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 165, 255), 1, cv2.LINE_AA)
    if meas_uv is not None:
        m = (int(meas_uv[0]), int(meas_uv[1]))
        cv2.drawMarker(im, m, (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(im, 'measured', (m[0] + 10, m[1] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 0), 1, cv2.LINE_AA)
    if pred_uv is not None and meas_uv is not None:
        cv2.line(im, (int(pred_uv[0]), int(pred_uv[1])),
                 (int(meas_uv[0]), int(meas_uv[1])), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(im, '%s gap %.2fs  miss %.1fm / %.0fpx  v %.1f m/s'
                % (tag, gap, miss_m, miss_px, speed), (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def sheet(tiles, path, cols=2):
    if not tiles:
        print('  no tiles for', path)
        return
    rows = (len(tiles) + cols - 1) // cols
    out = np.zeros((rows * H, cols * W, 3), np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * H:(r + 1) * H, c * W:(c + 1) * W] = im
    cv2.imwrite(path, out)
    print('  wrote %s (%d tiles)' % (path, len(tiles)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--tag', default='run')
    ap.add_argument('--no-coast-translate', action='store_true')
    ap.add_argument('--no-joint-slots', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    PR.COAST_TRANSLATE = not args.no_coast_translate
    PR.JOINT_SLOTS = not args.no_joint_slots

    sess = args.session
    if not os.path.isdir(sess):
        sess = os.path.join(os.path.dirname(HERE), 'sessions', sess)
    outdir = args.outdir or os.path.join(HERE, 'producer_runs', 'strips')
    os.makedirs(outdir, exist_ok=True)

    frames = [r for r in PR.load_csv(os.path.join(sess, 'frames.csv')) if r['file']]
    imu = PR.load_csv(os.path.join(sess, 'imu.csv'))
    race = PR.load_csv(os.path.join(sess, 'race.csv'))
    if args.skip:
        frames = frames[args.skip:]
    if args.limit:
        frames = frames[:args.limit]
    imu_t = np.array([float(r['t_wall_ns']) for r in imu])
    race_t = np.array([float(r['t_wall_ns']) for r in race]) if race else np.array([0.0])

    prod = PR.Producer()
    t0_ns = float(frames[0]['t_recv_wall_ns'])
    imu_i = 0
    coast_tiles, slot_tiles = [], []
    prev_stale = {}

    for n, fr in enumerate(frames):
        t_ns = float(fr['t_recv_wall_ns'])
        t_s = (t_ns - t0_ns) / 1e9
        j = int(np.searchsorted(imu_t, t_ns))
        rows = imu[imu_i:j]
        imu_i = j
        ri = max(0, int(np.searchsorted(race_t, t_ns)) - 1)
        active = int(race[ri]['active_gate_index']) if race else 0
        img = cv2.imread(os.path.join(sess, 'frames', fr['file']))

        # snapshot the coasted prediction BEFORE the frame is consumed
        pre = {}
        for g, tr in prod.tracks.items():
            if tr.pos is not None:
                pre[g] = (tr.pos.copy(), t_s - tr.t_obs)

        n_re = len(prod.reacq)
        obs = prod.step(img, rows, {'active_gate_index': active, 'race_time_s': 0.0,
                                    'armed': True, 'n_gates_total': 17}, t_s)

        # a reacquisition happened on this frame
        if len(prod.reacq) > n_re and img is not None and len(coast_tiles) < 6:
            gap, miss_m, miss_px, spd, cd = prod.reacq[-1]
            if gap > 0.25:
                g = active
                tr = prod.tracks.get(g)
                pred = pre.get(g)
                pred_uv = PR.project_body(pred[0]) if pred else None
                meas_uv = tr.quad.mean(0) if (tr is not None and tr.quad is not None) \
                    else (PR.project_body(tr.pos) if tr is not None and tr.pos is not None
                          else None)
                coast_tiles.append(draw_coast(img, pred_uv, meas_uv, gap, miss_m,
                                              miss_px, spd, args.tag))

        # Claire's regime: active == 1, both lookahead slots valid, high yaw
        if img is not None and active == 1 and len(slot_tiles) < 4:
            wz = abs(float(obs.own.gyro[2]))
            g1, g2 = obs.gates[1], obs.gates[2]
            if g1.valid and g2.valid and wz > 0.15:
                im = PR.annotate(img.copy(), obs, prod)
                cv2.rectangle(im, (0, 0), (W, 46), (0, 0, 0), -1)
                d12 = prod.map.dist(active, g1.index)
                d13 = prod.map.dist(active, g2.index)
                cv2.putText(im, '%s active=%d  yaw %.1f deg/s' %
                            (args.tag, active, math.degrees(wz)), (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(im, 'g%d %.1fm (map %.1f)   g%d %.1fm (map %.1f)   cur %.1fm'
                            % (g1.index, g1.range_m, d12 or -1, g2.index, g2.range_m,
                               d13 or -1, obs.gates[0].range_m if obs.gates[0].valid else -1),
                            (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1,
                            cv2.LINE_AA)
                slot_tiles.append(im)
        if n % 400 == 0:
            print('  %d/%d' % (n, len(frames)), flush=True)

    sheet(coast_tiles, os.path.join(outdir, 'coast_%s.png' % args.tag))
    sheet(slot_tiles, os.path.join(outdir, 'slots_%s.png' % args.tag))


if __name__ == '__main__':
    main()
