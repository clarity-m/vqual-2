"""Automatic ground-truth gate-corner labels for VQ1 recordings.

WHY THIS IS POSSIBLE. VQ2 blocks pose telemetry, but the VQ1 build streams it, and the
gates are VISUALLY IDENTICAL between the two builds (same orange AI-GP frames, same
1500 mm aperture -- `NOTES.md`). So a VQ1 frame can be labelled EXACTLY and automatically
by projecting known gate geometry through known camera pose; no hand labelling, no
detector in the loop, and therefore no circularity when the labels are used to train or
score a detector.

Everything geometric is reused from `label.py` -- `load_gates`, `gate_corners_world`,
`euler_to_R`, `body_to_cam`, `project`, `PoseTrack` -- including its yaw negation, which is
load-bearing and is documented there and in `../CONVENTIONS.md`. Nothing here re-derives a
sign.

GATE CENTRES. Default source is `gates_refined.json` (refine.py's fusion of hundreds of
detections through the pose stream, held-out centre error 1.8 px, versus 5.2 px for the raw
crossing positions in vqual-1's `gate_truth.json`). Its keys are 1..5 and are ONE-BASED:
key N is race index N-1, verified here against the position stream at each `gate_advance`
(crossing of race gate 0 lands at x=-23.3, which is truth key "1"). The VQ1 course has SIX
gates, 0..5; race gate 5 is never crossed in either session, so no truth exists for it and
it is NOT labelled. Frames containing it therefore carry a false negative.

Pass `--gates` (or `$VQ_GATES`, honoured by `label.load_gates`) to use another file.

WINDING ORDER -- IMAGE-SPACE CANONICAL, byte-for-byte the rule `labelui.html` enforces:
clockwise as seen on screen (image axes x right, y down, so signed shoelace > 0), rotated
so index 0 is the corner with the smallest x+y. It is a deterministic function of the four
points, which is what a corner-regression net needs, and under large image roll index 0 is
a DIFFERENT PHYSICAL CORNER -- intended, because the label is defined in image space.

The corners are generated in gate space first and then canonicalised:

    up    = world -z (gravity up)
    right = s * world +y,  where s = sign(gate_centre_x - camera_x)

    TL = centre + (-HALF*right, -HALF*z)      TR = centre + (+HALF*right, -HALF*z)
    BR = centre + (+HALF*right, +HALF*z)      BL = centre + (-HALF*right, +HALF*z)

`s` keeps that pre-canonical order view-dependent rather than gate-fixed, so the quad is
clockwise on screen whichever side of the gate the camera is on; every VQ1 gate shares the
world-x normal and the course runs toward -x, so `s` is -1 on every labelled instance here.
`canon()` then makes the emitted order agree with the hand labels regardless.

OUTPUT SCHEMA -- the hand-labelling UI's (`labelui.html`, "EXPORT SCHEMA"):

    {"<session>/<file>": [{"corners": [[x,y] x4], "occluded": bool,
                           "clipped": bool, "unsure": bool, "z": int}, ...], ...}

The UI keys on BASENAME because it labels one session at a time; this file keys on
`<session>/<file>` (as `labels_all.json` already does) because it spans sessions. Strip the
prefix if feeding a single session back into the UI.

  * corners are the INNER 1500 mm aperture, in ORIGINAL 640x360 pixels, and AMODAL:
    coordinates outside the image are kept, never clamped.
  * `clipped` -- at least one corner falls outside the image rect. Not a rejection: an
    amodal regressor needs these, and 46% of instances were lost to edge clipping in the
    contour detector (`detect.py`), so they are the interesting half.
  * `occluded` -- a NEARER labelled gate's quad covers >= OCCL_FRAC of this one's visible
    area. LIMITED BY CONSTRUCTION: only gate-on-gate occlusion is modelled. Hangar
    structure, columns and the cyan guidance ribbon are invisible to this test, so
    `occluded: false` means "not occluded by another gate", not "fully visible".
  * `unsure` -- always false. A projection is either emitted or rejected; it has no
    confidence axis of the kind a human labeller has. Present so both files load through
    the same reader.
  * `z` -- DRAW ORDER, matching the UI: z=0 is FURTHEST BACK, the highest z is nearest the
    camera. (The natural instinct is the opposite; the UI's rule wins.)

A frame with no labelable gate is OMITTED, not written as an empty array. In the UI's
schema an empty array is a VERIFIED NEGATIVE, and this file cannot honestly claim one:
race gate 5 has no truth, so a frame showing only gate 5 would be recorded as "reviewed,
contains no gate" -- a wrong label, not a missing one.

REJECTION RULES (an instance is emitted only if all hold):
  * pose interpolation available at the frame's `sim_time_ns` (label.py's rule: use
    sim_time_ns, NOT t_recv_wall_ns -- the latter carries ~38 ms of encode+UDP);
  * all four corners in front of the camera (z > MIN_DEPTH). A gate straddling the image
    plane projects to garbage, and there is no amodal answer to give;
  * the quad intersects the image rect at all;
  * apparent size >= MIN_SIZE_PX (default 6 px; 480/6 = 80 m).

    python3 pilot/perception/autolabel.py <session-dir> [<session-dir> ...] \
        [--out autolabels.json] [--min-px 6]
    python3 pilot/perception/autolabel.py <session-dir> ... --verify verify/ [--n 12]
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

MIN_DEPTH = 0.30        # m in front of the camera, per corner
MIN_SIZE_PX = 6.0       # longest projected edge; 480/6 px = 80 m range
OCCL_FRAC = 0.25        # fraction of visible area covered by a nearer gate -> occluded
IMG_RECT = np.array([[0.0, 0.0], [L.W, 0.0], [L.W, L.H], [0.0, L.H]], np.float32)


def corners_world(centre, cam_pos):
    """Inner-aperture corners in TL, TR, BR, BL order as seen from `cam_pos`.

    See the module docstring for the winding derivation. Gate plane is world y-z (normal
    is world x, shared by every VQ1 gate), up is world -z, right is s * world +y.
    """
    cx, cy, cz = float(centre[0]), float(centre[1]), float(centre[2])
    s = 1.0 if (cx - float(cam_pos[0])) >= 0.0 else -1.0
    h = L.HALF
    return np.array([
        [cx, cy - s * h, cz - h],   # TL
        [cx, cy + s * h, cz - h],   # TR
        [cx, cy + s * h, cz + h],   # BR
        [cx, cy - s * h, cz + h],   # BL
    ])


def canon(uv):
    """Image-space canonical winding, identical to `labelui.html`'s canon().

    Clockwise on screen (signed shoelace > 0 with y down), then rotated so index 0 is the
    min(x+y) corner. Deterministic in the four points alone, so hand labels and these
    agree corner-for-corner and a regressor never sees two orders for one picture.
    """
    p = np.asarray(uv, float)
    sl = float(sum(p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1]
                   for i in range(4)))
    if sl <= 0:
        p = p[[0, 3, 2, 1]]
    start = int(np.argmin(p[:, 0] + p[:, 1]))
    return np.roll(p, -start, axis=0)


def _poly_area(p):
    p = np.asarray(p, np.float32)
    if len(p) < 3:
        return 0.0
    return float(abs(cv2.contourArea(p)))


def _clip_to_image(quad):
    """Quad ^ image rect as a polygon (possibly empty). Both are convex."""
    q = np.asarray(quad, np.float32)
    area, inter = cv2.intersectConvexConvex(q, IMG_RECT)
    if inter is None or len(inter) < 3:
        return None
    return inter.reshape(-1, 2).astype(np.float32)


def _overlap_area(a, b):
    area, _ = cv2.intersectConvexConvex(np.asarray(a, np.float32),
                                        np.asarray(b, np.float32))
    return float(area)


def frame_instances(pose, gates, min_px=MIN_SIZE_PX):
    """All labelable gate instances for one camera pose. Returns a list of dicts with
    the output schema plus 'gate' (race index) and 'range_m' for internal use."""
    cand = []
    for gi in sorted(gates):
        pts = corners_world(gates[gi], pose[0])
        uv, z = L.project(pts, pose)
        if not np.all(np.isfinite(uv)) or np.any(z <= MIN_DEPTH):
            continue
        e = [float(np.linalg.norm(uv[(j + 1) % 4] - uv[j])) for j in range(4)]
        if max(e) < min_px:
            continue
        vis = _clip_to_image(uv.astype(np.float32))
        if vis is None:
            continue
        cand.append({
            'gate': int(gi),
            'uv': uv,
            'vis': vis,
            'vis_area': _poly_area(vis),
            'depth': float(np.mean(z)),
            'size_px': max(e),
            'range_m': float(np.linalg.norm(np.asarray(gates[gi]) - pose[0])),
        })

    # FURTHEST FIRST: z is the UI's draw order, so index 0 is the back of the stack and
    # the last element is nearest the camera.
    cand.sort(key=lambda d: -d['depth'])
    out = []
    for k, c in enumerate(cand):
        covered = 0.0
        for nearer in cand[k + 1:]:
            covered += _overlap_area(c['vis'], nearer['vis'])
        occl = bool(c['vis_area'] > 0 and covered / c['vis_area'] >= OCCL_FRAC)
        uv = canon(c['uv'])
        clipped = bool(np.any(uv[:, 0] < 0) or np.any(uv[:, 0] > L.W - 1) or
                       np.any(uv[:, 1] < 0) or np.any(uv[:, 1] > L.H - 1))
        out.append({
            'corners': [[round(float(u), 2), round(float(v), 2)] for u, v in uv],
            'occluded': occl,
            'clipped': clipped,
            'unsure': False,
            'z': k,
            'gate': c['gate'],
            'range_m': c['range_m'],
            'size_px': c['size_px'],
        })
    return out


def label_session(session, gates, min_px=MIN_SIZE_PX):
    """-> (dict keyed '<session>/<file>', stats dict)."""
    name = os.path.basename(os.path.normpath(session))
    track = L.PoseTrack(session)
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    labels, stats = {}, {'frames': len(frames), 'posed': 0, 'inst': 0, 'sizes': [],
                         'clipped': 0, 'occluded': 0, 'per_gate': {}}
    for rec in frames:
        t = float(rec['sim_time_ns']) / 1e9
        pose = track.at_time(t)
        if pose is None:
            continue
        stats['posed'] += 1
        inst = frame_instances(pose, gates, min_px)
        if not inst:
            continue
        for d in inst:
            stats['inst'] += 1
            stats['sizes'].append(d['size_px'])
            stats['clipped'] += d['clipped']
            stats['occluded'] += d['occluded']
            g = str(d['gate'])
            stats['per_gate'][g] = stats['per_gate'].get(g, 0) + 1
        labels[f'{name}/{rec["file"]}'] = [
            {'corners': d['corners'], 'occluded': d['occluded'],
             'clipped': d['clipped'], 'unsure': d['unsure'], 'z': d['z']} for d in inst]
    return labels, stats


# --------------------------------------------------------------------------------------
# verification


def verify(sessions, gates, outdir, n=12, min_px=MIN_SIZE_PX):
    """Overlay sheets + the detector cross-check.

    The cross-check is the one that can fail informatively: `detect.py` finds gates from
    ORANGE PIXELS alone, with no pose and no gate truth, so agreement between a projected
    centre and a detected centre is evidence from outside every telemetry convention. A
    few px means the projection is right; tens of px is a sign or frame error.

    NEAREST-DETECTION MATCHING NEEDS A SIZE GATE, and the gate is not a fudge. Where the
    detector MISSES a gate (recall is 0.21 at 50-100 px apparent size -- near gates run out
    of frame and the aperture stops being an enclosed contour), the nearest detection is
    some other gate metres away, and the resulting tens-of-px "error" measures detector
    recall, not projection. Requiring the matched detection to be within 2x of the
    projected apparent size separates the two, and both numbers are reported.
    """
    import detect as D
    os.makedirs(outdir, exist_ok=True)
    per = max(1, n // len(sessions))
    rows, tiles = [], []

    for session in sessions:
        name = os.path.basename(os.path.normpath(session))
        track = L.PoseTrack(session)
        frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]

        # frames worth looking at: a gate present and big enough to judge by eye
        good = []
        for i, rec in enumerate(frames):
            pose = track.at_time(float(rec['sim_time_ns']) / 1e9)
            if pose is None:
                continue
            inst = frame_instances(pose, gates, min_px)
            if inst and max(d['size_px'] for d in inst) > 25:
                good.append((i, rec, pose, inst))
        if not good:
            print(f'{name}: no frames with a gate above 25 px')
            continue

        # cross-check on a wide sample, overlays on a few
        step = max(1, len(good) // 200)
        for i, rec, pose, inst in good[::step]:
            img = cv2.imread(os.path.join(session, 'frames', rec['file']))
            if img is None:
                continue
            dets = D.detections(img)
            if not dets:
                continue
            dc = np.array([d['centre'] for d in dets])
            dsz = np.array([d['size_px'] for d in dets])
            for d in inst:
                if d['clipped'] or d['occluded'] or d['size_px'] < 15:
                    continue
                c = np.mean(np.array(d['corners']), axis=0)
                k = int(np.argmin(np.linalg.norm(dc - c, axis=1)))
                rows.append((float(np.linalg.norm(dc[k] - c)), d['size_px'], float(dsz[k])))

        sel = [good[int(round(k))] for k in
               np.linspace(0, len(good) - 1, min(per, len(good)))]
        for i, rec, pose, inst in sel:
            img = cv2.imread(os.path.join(session, 'frames', rec['file']))
            if img is None:
                continue
            for d in inst:
                uv = np.array(d['corners'])
                cv2.polylines(img, [uv.astype(np.int32).reshape(-1, 1, 2)], True,
                              (0, 0, 255) if d['clipped'] else (0, 255, 0), 2)
                # corner 0 (TL) marked, so the winding is visible in the picture
                cv2.circle(img, tuple(uv[0].astype(int)), 5, (255, 0, 255), -1)
                cv2.circle(img, tuple(uv[1].astype(int)), 4, (255, 255, 0), 2)
                c = uv.mean(axis=0).astype(int)
                cv2.putText(img, f'{d["gate"]} {d["range_m"]:.0f}m',
                            (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)
            cv2.putText(img, f'{name}/{rec["file"]}', (5, 352),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(img)

    cols = 3
    nrows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((max(nrows, 1) * L.H, cols * L.W, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * L.H:(r + 1) * L.H, c * L.W:(c + 1) * L.W] = t
    path = os.path.join(outdir, 'overlay_sheet.png')
    cv2.imwrite(path, sheet)
    print(f'wrote {path}  ({len(tiles)} tiles)')

    if not rows:
        print('detector cross-check: no comparable pairs')
        return
    a = np.array([r[0] for r in rows])
    sz = np.array([r[1] for r in rows])
    ds = np.array([r[2] for r in rows])
    m = (ds > sz / 2) & (ds < sz * 2)
    print(f'detector cross-check, all nearest matches: n={len(a)}  '
          f'median={np.median(a):.1f} px  p75={np.percentile(a, 75):.1f}  '
          f'p90={np.percentile(a, 90):.1f}')
    if m.any():
        b = a[m]
        print(f'  size-consistent only (n={m.sum()}, {100*m.mean():.0f}%): '
              f'median={np.median(b):.1f} px  p75={np.percentile(b, 75):.1f}  '
              f'p90={np.percentile(b, 90):.1f}  '
              f'relative={np.median(b / sz[m]):.3f} of gate width')
    for lo, hi in ((15, 30), (30, 60), (60, 120), (120, 1e9)):
        k = (sz >= lo) & (sz < hi)
        if k.any():
            km = k & m
            print(f'  size {lo:4.0f}-{hi:<6.0f} n={k.sum():4d}  median={np.median(a[k]):6.1f}'
                  f'   size-consistent n={km.sum():4d} '
                  f'median={np.median(a[km]) if km.any() else float("nan"):.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--gates', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'gates_refined.json'))
    ap.add_argument('--out', default='autolabels.json')
    ap.add_argument('--min-px', type=float, default=MIN_SIZE_PX)
    ap.add_argument('--verify', metavar='DIR')
    ap.add_argument('--n', type=int, default=12)
    args = ap.parse_args()

    # keys are ONE-BASED in both truth files; race index = key - 1 (docstring)
    raw = L.load_gates(args.gates)
    gates = {k - 1: v for k, v in raw.items()}
    print(f'{len(gates)} gate centres from {args.gates} -> race indices {sorted(gates)}')

    if args.verify:
        verify(args.sessions, gates, args.verify, args.n, args.min_px)
        return

    allsizes, out, total = [], {}, {'frames': 0, 'posed': 0, 'inst': 0,
                                   'clipped': 0, 'occluded': 0}
    for s in args.sessions:
        labels, st = label_session(s, gates, args.min_px)
        out.update(labels)
        allsizes += st['sizes']
        for k in total:
            total[k] += st[k]
        print(f'{os.path.basename(os.path.normpath(s))}: {st["inst"]} instances over '
              f'{len(labels)}/{st["posed"]} posed frames, per-gate {st["per_gate"]}')

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    a = np.array(allsizes) if allsizes else np.zeros(1)
    print(f'wrote {args.out}: {len(out)} frames, {total["inst"]} instances '
          f'({total["inst"] - total["clipped"]} fully in-frame, {total["clipped"]} clipped, '
          f'{total["occluded"]} gate-occluded)')
    print(f'apparent size px: min {a.min():.1f}  p10 {np.percentile(a, 10):.1f}  '
          f'median {np.median(a):.1f}  p90 {np.percentile(a, 90):.1f}  max {a.max():.1f}')


if __name__ == '__main__':
    main()
