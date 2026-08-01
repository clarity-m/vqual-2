"""Auto-label gate corners in recorded VQ1 frames.

Ground truth comes from three places, each with a different reliability, and the
differences matter:

  * gate CENTRES  -- vqual-1's gate_truth.json. These are CROSSING POSITIONS, i.e. where
    the drone went through, not where the gate centre is. Error is bounded by the inner
    half-aperture (0.75 m) and each is a single sample (n=1). Treat as an INITIALISATION,
    not as truth. refine.py replaces them once the detector can fuse many views.
  * gate ORIENTATION -- every VQ1 gate shares one normal, along world x (Claire,
    2026-07-31). So the aperture lies in the world y-z plane and there is nothing to
    estimate. This is what makes labelling possible at all from centres alone.
  * camera POSE -- LOCAL_POSITION_NED + attitude, both dense and accurate. The strong
    signal. Sign corrections below are mandatory and are the whole ballgame.

SIGN CORRECTIONS (pilot/NOTES.md). The VQ1 truth streams are each wrong on a different
axis; a label set built without these is mirrored, and mirrored silently:

    roll  =  ATTITUDE.roll        (== -ODOMETRY.roll)
    pitch =  ODOMETRY.pitch       (== -ATTITUDE.pitch)
    yaw   = -ATTITUDE.yaw         Settled 2026-07-31 by the ONE referee outside the sim:
                                  the human pilot. `e` sends yaw_rate +2.00 (KEYS_AXIS,
                                  ACRO_YAW=+2.0, no sign applied on the way out) and `e`
                                  turns the nose LEFT (Claire, observed). Nose left is
                                  DECREASING NED yaw, but session 20260731-203428 shows
                                  that tap moving ATTITUDE.yaw by +1.094. So ATTITUDE.yaw
                                  is mirrored, like everything else in this sim.

THE MIRROR DOES NOT CANCEL HERE, SO THE NEGATION ABOVE IS LOAD-BEARING.
LOCAL_POSITION_NED is plain canonical NED -- forward, right and up all read negative,
checked on the HUD against motion a human could see (Claire, 2026-07-31), and the VQ1
course descends, which matches z counting up as you go down. ATTITUDE.yaw is mirrored.
They therefore disagree, and a projection that takes yaw at face value is wrong.

HOW WRONG, measured rather than argued: project every gate under both hypotheses, keep
only frames where the two predictions differ by more than 80 px, and score each against
the ORANGE PIXEL MASK -- image evidence, outside every telemetry convention:

    yaw = +ATTITUDE.yaw   median 169.6 px from the nearest orange blob,  2% within 40 px
    yaw = -ATTITUDE.yaw   median  18.0 px,                              79% within 40 px

The 80 px separation filter is the entire point. Near frame centre both hypotheses predict
the same pixel, so unfiltered scoring buries the signal in frames that carry none. An
earlier version of this file shipped the WRONG sign and passed a visual check for exactly
that reason: every tile inspected had the gate near centre with the aircraft flying at it.
That is the same degeneracy that hid vqual-1's yaw error at its 180 deg start heading.

TIME ALIGNMENT. Use frames.csv `sim_time_ns`, not `t_recv_wall_ns`. The latter includes
JPEG encode + UDP latency (~38 ms measured), which at racing rates is tens of pixels of
label error. Both are on the same wall epoch as the telemetry timestamps.

    python3 pilot/perception/label.py <session-dir> [--sheet out.png] [--n 12]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

# Spec 3.7/3.8. Camera shares the body origin, tilted 20 deg UP.
W, H = 640, 360
FX = FY = 320.0
CX, CY = 320.0, 180.0
CAM_TILT_UP = math.radians(20.0)
GATE_INNER = 1.5
HALF = GATE_INNER / 2.0

GATE_TRUTH = 'C:/Users/USER/Projects/vqual-1/pilot/gate_truth.json'


def load_gates(path=None):
    """Gate centres. Override with $VQ_GATES to use refine.py's fused estimate.

    Kept as an explicit override rather than auto-preferring gates_refined.json, because
    scoring a detector against labels that detector produced is circular and the default
    must not silently become the circular one.
    """
    import numpy as _np
    src = path or os.environ.get('VQ_GATES') or GATE_TRUTH
    d = json.load(open(src))['consensus']
    return {int(k): _np.array([v['x'], v['y'], v['z']]) for k, v in d.items()}


def load_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def euler_to_R(roll, pitch, yaw):
    """Body->world rotation, ZYX (yaw-pitch-roll), canonical NED."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def body_to_cam():
    """Body(FRD) -> camera(x right, y down, z forward), with the 20 deg upward tilt.

    SIGN, derived not guessed. Body-forward (1,0,0) through RELABEL @ Ry(theta) lands at
    v = CY - FY*tan(theta). The camera points UP, so forward must render BELOW centre
    (v > CY), which requires theta NEGATIVE. Confirmed by convention sweep against seven
    pre-crossing frames: -20 deg beat +20 deg by 3x.

    That is also why a gate at own altitude sits low in frame -- the binding constraint
    on this course. Forward is at v = 180 + 320*tan(20) = 296 of 360.
    """
    relabel = np.array([[0.0, 1.0, 0.0],   # cam x  =  body y (right)
                        [0.0, 0.0, 1.0],   # cam y  =  body z (down)
                        [1.0, 0.0, 0.0]])  # cam z  =  body x (forward)
    c, s = math.cos(-CAM_TILT_UP), math.sin(-CAM_TILT_UP)
    tilt = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])  # Ry(-20 deg)
    return relabel @ tilt


def gate_corners_world(centre):
    """Four corners of the inner aperture. Normal is world x, so the square spans y-z."""
    cx_, cy_, cz_ = centre
    return np.array([
        [cx_, cy_ - HALF, cz_ - HALF],
        [cx_, cy_ + HALF, cz_ - HALF],
        [cx_, cy_ + HALF, cz_ + HALF],
        [cx_, cy_ - HALF, cz_ + HALF],
    ])


class PoseTrack:
    def __init__(self, session):
        pos = load_csv(os.path.join(session, 'position.csv'))
        att = load_csv(os.path.join(session, 'attitude.csv'))
        odo = load_csv(os.path.join(session, 'odometry.csv'))

        self.pt = np.array([float(r['t_wall_ns']) for r in pos]) / 1e9
        self.p = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in pos])

        self.at = np.array([float(r['t_wall_ns']) for r in att]) / 1e9
        self.roll = np.unwrap(np.array([float(r['roll']) for r in att]))
        self.yaw = np.unwrap(np.array([float(r['yaw']) for r in att]))

        self.ot = np.array([float(r['t_wall_ns']) for r in odo]) / 1e9
        q = np.array([[float(r[k]) for k in ('qw', 'qx', 'qy', 'qz')] for r in odo])
        w_, x_, y_, z_ = q.T
        self.pitch = np.unwrap(np.arcsin(np.clip(2 * (w_ * y_ - z_ * x_), -1.0, 1.0)))

    def at_time(self, t):
        if not (self.pt[0] <= t <= self.pt[-1]):
            return None
        p = np.array([np.interp(t, self.pt, self.p[:, i]) for i in range(3)])
        roll = float(np.interp(t, self.at, self.roll))
        pitch = float(np.interp(t, self.ot, self.pitch))
        # NEGATED. ATTITUDE.yaw is mirrored; world position is not. See the header.
        yaw = -float(np.interp(t, self.at, self.yaw))
        return p, roll, pitch, yaw


def project(pts_world, pose):
    """World points -> pixels. Returns (uv, depths); depth <= 0 means behind the camera."""
    p, roll, pitch, yaw = pose
    R_bw = euler_to_R(roll, pitch, yaw)          # body -> world
    R_cb = body_to_cam()                          # body -> camera
    rel_world = pts_world - p
    rel_body = rel_world @ R_bw                   # world -> body  (R_bw.T @ v)
    cam = rel_body @ R_cb.T
    z = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = CX + FX * cam[:, 0] / z
        v = CY + FY * cam[:, 1] / z
    return np.stack([u, v], axis=1), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--sheet', default='label_sheet.png')
    ap.add_argument('--n', type=int, default=12)
    args = ap.parse_args()

    gt = json.load(open(GATE_TRUTH))['consensus']
    gates = {int(k): np.array([v['x'], v['y'], v['z']]) for k, v in gt.items()}
    print(f'{len(gates)} gates from gate_truth.json: {sorted(gates)}')

    track = PoseTrack(args.session)
    frames = [r for r in load_csv(os.path.join(args.session, 'frames.csv')) if r['file']]
    ft = np.array([float(r['sim_time_ns']) for r in frames]) / 1e9

    # Pick frames on APPROACH to real crossings. Spreading picks uniformly over the
    # session mostly samples moments with no gate in view, which cannot validate
    # anything -- the check needs frames where a gate is both visible and identified.
    ok = np.where((ft >= track.pt[0]) & (ft <= track.pt[-1]))[0]
    print(f'{len(frames)} frames, {len(ok)} inside the pose window')
    ev = [json.loads(l) for l in open(os.path.join(args.session, 'events.jsonl'))]
    xt = [e['t_wall_ns'] / 1e9 for e in ev
          if e['kind'] == 'gate_advance' and e['from_gate'] >= 0]
    leads = [3.0, 1.5, 0.7]
    times = sorted(t - d for t in xt for d in leads)
    picks = []
    for t in times:
        i = int(np.argmin(np.abs(ft - t)))
        if i in ok and i not in picks:
            picks.append(i)
    picks = picks[:args.n]
    tiles = []

    for idx in picks:
        rec = frames[idx]
        pose = track.at_time(ft[idx])
        img = cv2.imread(os.path.join(args.session, 'frames', rec['file']))
        if img is None or pose is None:
            continue
        n_drawn = 0
        for gi in sorted(gates):
            uv, z = project(gate_corners_world(gates[gi]), pose)
            if np.any(z <= 0.1):
                continue
            if np.all((uv[:, 0] < -200) | (uv[:, 0] > W + 200)) or \
               np.all((uv[:, 1] < -200) | (uv[:, 1] > H + 200)):
                continue
            pts = uv.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            c = uv.mean(axis=0).astype(int)
            cv2.circle(img, tuple(c), 3, (0, 255, 255), -1)
            cv2.putText(img, str(gi), (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            n_drawn += 1
        rng = min(float(np.linalg.norm(gates[g] - pose[0])) for g in gates)
        cv2.putText(img, f't={ft[idx]-ft[ok[0]]:.1f}s  gates={n_drawn}  nearest={rng:.0f}m',
                    (6, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(img)

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * H, cols * W, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * H:(r + 1) * H, c * W:(c + 1) * W] = t
    cv2.imwrite(args.sheet, sheet)
    print(f'wrote {args.sheet}  ({len(tiles)} tiles, {rows}x{cols})')


if __name__ == '__main__':
    main()
