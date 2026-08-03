"""Template verdict checks for the 2026-08-02 strafe session (12-13-14).

Per marker window (+-1.5 s): gate detections (count/sizes/clipped), gyro steadiness,
ceiling-light co-visibility (from the skylight CSV). Per steady segment (|gyro|<0.3):
translation baseline in metres from tracked-gate PnP position expressed in the
gravity-levelled grid frame (psi_unwrapped), and drift direction for perpendicularity.
Montage of 4 marker-window frames -> vercheck/strafe_template_check.png
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detect as D          # noqa: E402
import skylight as SK       # noqa: E402
import label as L           # noqa: E402

SESS_NAME = sys.argv[1] if len(sys.argv) > 1 else '20260802-004153-vq2-strafe-12-13-14'
MONTAGE_NAME = sys.argv[2] if len(sys.argv) > 2 else 'strafe_template_check.png'
SESS = os.path.join(HERE, '..', 'sessions', SESS_NAME)
SKYCSV = os.path.join(HERE, f'skylight_{SESS_NAME}.csv')
W, H = 640, 360
PAD = 1.5e9          # marker window half-width, ns
GYRO_STEADY = 0.3    # rad/s


def load_frames():
    rows = [r for r in L.load_csv(os.path.join(SESS, 'frames.csv')) if r['file']]
    for r in rows:
        r['t'] = float(r['t_recv_wall_ns'])
    return rows


def load_markers():
    out = []
    with open(os.path.join(SESS, 'events.jsonl')) as fh:
        for line in fh:
            e = json.loads(line)
            if e.get('kind') == 'marker':
                out.append(e)
    return out


def load_sky():
    if not os.path.exists(SKYCSV):
        return None
    by_file = {}
    with open(SKYCSV) as fh:
        for r in csv.DictReader(fh):
            by_file[r['file']] = r
    return by_file


def load_gyro():
    t, w = [], []
    with open(os.path.join(SESS, 'imu.csv')) as fh:
        for r in csv.DictReader(fh):
            t.append(float(r['t_wall_ns']))
            w.append((float(r['xgyro'])**2 + float(r['ygyro'])**2
                      + float(r['zgyro'])**2) ** 0.5)
    return np.array(t), np.array(w)


def clipped(d):
    if d['source'] == 'outer':
        return True
    q = d['quad']
    return bool((q[:, 0] < 3).any() or (q[:, 0] > W - 4).any()
                or (q[:, 1] < 3).any() or (q[:, 1] > H - 4).any())


def dets_at(frames, i, cache={}):
    if i not in cache:
        img = cv2.imread(os.path.join(SESS, 'frames', frames[i]['file']))
        cache[i] = [] if img is None else D.detections(img)
    return cache[i]


def marker_report(frames, markers, sky, gt, gw):
    ft = np.array([f['t'] for f in frames])
    print('\n=== MARKER WINDOWS (+-1.5 s) ===')
    for m in markers:
        tm = m['t_wall_ns']
        lo, hi = np.searchsorted(ft, [tm - PAD, tm + PAD])
        idxs = list(range(lo, hi, 3))
        ngates, sizes, nclip, ntot, npair = [], [], 0, 0, 0
        for i in idxs:
            ds = dets_at(frames, i)
            ngates.append(len(ds))
            ntrust = 0
            for d in ds:
                sizes.append(d['size_px'])
                ntot += 1
                nclip += clipped(d)
                if (not clipped(d) and d['source'] == 'inner'
                        and d['size_px'] >= 26):
                    ntrust += 1
            npair += ntrust >= 2
        glo, ghi = np.searchsorted(gt, [tm - PAD, tm + PAD])
        wwin = gw[glo:ghi]
        # lights from skylight csv
        lit = conf = nn = 0
        if sky:
            for i in range(lo, hi):
                r = sky.get(frames[i]['file'])
                if r is None:
                    continue
                nn += 1
                lit += int(r['n_comp']) > 0
                conf += int(r['confident'])
        sizes = np.array(sizes) if sizes else np.array([0.0])
        print(f"marker {m['index']} (race_t={m['race_time_s']:.1f}s):")
        print(f"  gates/frame: mean {np.mean(ngates):.1f}, min {min(ngates)}, "
              f"max {max(ngates)} over {len(idxs)} frames")
        print(f"  sizes px: median {np.median(sizes):.0f}, "
              f"p10 {np.percentile(sizes,10):.0f}, p90 {np.percentile(sizes,90):.0f}, "
              f"max {sizes.max():.0f}  (PnP range@median ~{480/max(np.median(sizes),1e-6):.1f} m)")
        print(f"  clipped: {nclip}/{ntot} ({100*nclip/max(ntot,1):.0f}%)")
        print(f"  DECISIVE two-trusted-gate frames (>=2 inner, unclipped, "
              f">=26 px): {npair}/{len(idxs)}")
        print(f"  |gyro|: mean {wwin.mean():.3f}, max {wwin.max():.3f} rad/s "
              f"({'STEADY' if wwin.max() < GYRO_STEADY else 'NOT steady'})")
        if sky:
            print(f"  lights: in-frame {lit}/{nn} frames, compass-confident {conf}/{nn}")


def steady_segments(gt, gw, min_len_s=2.0):
    # smooth |gyro| with ~0.25 s box
    dt = np.median(np.diff(gt)) / 1e9
    k = max(1, int(0.25 / dt))
    ws = np.convolve(gw, np.ones(k) / k, mode='same')
    ok = ws < GYRO_STEADY
    segs, start = [], None
    for i, o in enumerate(ok):
        if o and start is None:
            start = i
        elif not o and start is not None:
            if (gt[i - 1] - gt[start]) / 1e9 >= min_len_s:
                segs.append((gt[start], gt[i - 1]))
            start = None
    if start is not None and (gt[-1] - gt[start]) / 1e9 >= min_len_s:
        segs.append((gt[start], gt[-1]))
    return segs


def grid_pos(det, g, psi):
    """Gate position in gravity-levelled frame rotated by psi_unwrapped (grid frame)."""
    R_lb = SK.level_rotation(g)
    if R_lb is None:
        return None
    p = R_lb @ det['pos_body']
    c, s = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


def baseline_report(frames, sky, gt, gw, markers):
    ft = np.array([f['t'] for f in frames])
    imu = SK.Imu(SESS)
    segs = steady_segments(gt, gw)
    print(f'\n=== STEADY SEGMENTS (|gyro|<{GYRO_STEADY} rad/s, >=2 s): {len(segs)} ===')
    dirs = []
    for si, (t0, t1) in enumerate(segs):
        lo, hi = np.searchsorted(ft, [t0, t1])
        idxs = list(range(lo, hi, 5))
        # track by centre proximity, keyed on previous frame's detections
        tracks = {}   # tid -> list of (t, grid_pos)
        prev = []     # (tid, centre)
        next_tid = 0
        for i in idxs:
            ds = dets_at(frames, i)
            t = frames[i]['t']
            r = sky.get(frames[i]['file']) if sky else None
            psi = float(r['psi_unwrapped']) if r and r['psi_unwrapped'] else None
            g, _ = imu.gravity(t)
            cur = []
            for d in ds:
                tid = None
                for ptid, pc in prev:
                    if np.linalg.norm(d['centre'] - pc) < 60:
                        tid = ptid
                        prev = [(a, b) for a, b in prev if a != ptid]
                        break
                if tid is None:
                    tid = next_tid
                    next_tid += 1
                cur.append((tid, d['centre']))
                # inner-source, near enough for PnP to be trusted (MIN_SIZE_PX ~ 18.5 m)
                if (psi is not None and g is not None and not clipped(d)
                        and d['source'] == 'inner' and d['size_px'] >= 26):
                    gp = grid_pos(d, g, psi)
                    if gp is not None:
                        tracks.setdefault(tid, []).append((t, gp))
            prev = cur
        best, bl = None, 0.0
        for tid, obs in tracks.items():
            if len(obs) < 6:
                continue
            tt = np.array([o[0] for o in obs]) / 1e9
            pp = np.array([o[1] for o in obs])
            span = tt[-1] - tt[0]
            if span < 1.5:
                continue
            # linear fit per axis with one outlier-rejection round
            for _ in range(2):
                coef = np.array([np.polyfit(tt, pp[:, k], 1) for k in range(3)])
                pred = np.stack([np.polyval(coef[k], tt) for k in range(3)], axis=1)
                resid = np.linalg.norm(pp - pred, axis=1)
                keep = resid < max(1.0, 3 * np.median(resid))
                if keep.all() or keep.sum() < 6:
                    break
                tt, pp = tt[keep], pp[keep]
            v = -coef[:, 0]   # aircraft velocity = -gate velocity in grid frame
            dv = v * span
            if np.linalg.norm(dv[:2]) > bl:
                bl, best = np.linalg.norm(dv[:2]), (tid, len(tt), dv)
        near = [m['index'] for m in markers if t0 - PAD <= m['t_wall_ns'] <= t1 + PAD]
        tag = f" [markers {near}]" if near else ''
        if best:
            tid, n, dv = best
            ang = np.degrees(np.arctan2(dv[1], dv[0])) % 360
            dirs.append((si, (t1 - t0) / 1e9, bl, ang))
            print(f"seg {si}: {(t0-ft[0])/1e9:6.1f}-{(t1-ft[0])/1e9:6.1f}s "
                  f"({(t1-t0)/1e9:4.1f}s){tag}  drift {bl:5.2f} m horiz, "
                  f"dz {dv[2]:+.2f} m, grid-dir {ang:5.1f} deg (n={n} obs)")
        else:
            print(f"seg {si}: {(t0-ft[0])/1e9:6.1f}-{(t1-ft[0])/1e9:6.1f}s "
                  f"({(t1-t0)/1e9:4.1f}s){tag}  no usable gate track")
    if len(dirs) >= 2:
        print('\npairwise drift-direction differences (legs >=1.5 m):')
        big = [d for d in dirs if d[2] >= 1.5]
        for i in range(len(big)):
            for j in range(i + 1, len(big)):
                dd = abs((big[i][3] - big[j][3] + 180) % 360 - 180)
                print(f"  seg {big[i][0]} vs seg {big[j][0]}: {dd:.0f} deg apart")


def montage(frames, markers, sky):
    ft = np.array([f['t'] for f in frames])
    imu = SK.Imu(SESS)
    # 4 panels: one at each marker, plus one 1 s after the busiest marker
    picks = [np.searchsorted(ft, m['t_wall_ns']) for m in markers]
    picks.append(np.searchsorted(ft, markers[0]['t_wall_ns'] + 1.0e9))
    picks = picks[:4]
    tiles = []
    for k, i in enumerate(picks):
        img = cv2.imread(os.path.join(SESS, 'frames', frames[i]['file']))
        ds = D.detections(img)
        g, _ = imu.gravity(frames[i]['t'])
        vis = img.copy()
        for d in ds:
            col = (0, 165, 255) if not clipped(d) else (0, 0, 255)
            cv2.polylines(vis, [d['quad'].astype(int)], True, col, 2)
            c = d['centre'].astype(int)
            cv2.putText(vis, f"{d['range_m']:.1f}m {d['size_px']:.0f}px",
                        (c[0] - 30, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
        nlight = 0
        if g is not None:
            R_lb = SK.level_rotation(g)
            if R_lb is not None:
                comps = SK.light_components(img, R_lb)
                for contours, bbox in comps:
                    x0, y0, x1, y1 = bbox
                    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                                  (255, 255, 0), 1)
                nlight = len(comps)
        lab = (f"marker {markers[k]['index']}" if k < len(markers)
               else f"marker {markers[0]['index']} +1s")
        cv2.putText(vis, f"{lab}  gates={len(ds)} lights={nlight}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(vis)
    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4])
    out = np.vstack([top, bot])
    path = os.path.join(HERE, 'vercheck', MONTAGE_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out)
    print(f'\nmontage -> {path}')


def main():
    frames = load_frames()
    markers = load_markers()
    sky = load_sky()
    gt, gw = load_gyro()
    if sky is None:
        print('WARNING: skylight CSV missing, light/psi checks skipped')
    marker_report(frames, markers, sky, gt, gw)
    if sky:
        baseline_report(frames, sky, gt, gw, markers)
    montage(frames, markers, sky)


if __name__ == '__main__':
    main()
