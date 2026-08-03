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
    area. STILL GATE-ON-GATE ONLY, and still a bool, deliberately: `gatenet.load_index`,
    `packcrops.pack` and `labelui.html` all read it as a bool today and this file must not
    silently change what they read. The honest three-way answer is the ADDITIVE
    `visibility` field below; `occluded` keeps meaning exactly what it always meant.
  * `unsure` -- always false. A projection is either emitted or rejected; it has no
    confidence axis of the kind a human labeller has. Present so both files load through
    the same reader.
  * `z` -- DRAW ORDER, matching the UI: z=0 is FURTHEST BACK, the highest z is nearest the
    camera. (The natural instinct is the opposite; the UI's rule wins.)

ADDITIVE FIELDS -- the VISIBILITY TEST (2026-08-01). Everything above projects geometry and
asks nothing of the image, so ANY gate inside the view frustum got a label, including ones
hidden behind a column, behind the hangar structure, or behind another gate. Measured on a
600-instance random sample before this test existed: 4.7% had essentially no orange anywhere
near where the projection said a gate was, and a further 1.7% had a degenerate (<2 px)
visible box. Those are label noise for the corner regressor and POISON for a confidence
head, whose entire job is "is a real gate here".

So each instance now also carries what the PIXELS say, as numbers, not just flags -- a
number can be re-thresholded later without re-running the labeller:

  * `orange_band` -- fraction of orange pixels (detect.orange_mask, reused unchanged, no new
    colour test) inside the projected gate FRAME BAND: the annulus between the 1500 mm
    aperture quad and the 2700 mm outer quad, intersected with the image rect. This is the
    primary measure because it looks exactly where orange must be if the gate is really
    there. An interior- or bbox-based measure is confounded by gate size: a near gate's
    aperture is mostly empty background and its frame lies outside its own bounding box.
  * `band_px` -- area of that band inside the image, in px^2. It is the CONFIDENCE in
    `orange_band`: a gate so clipped that almost none of its frame is on screen cannot be
    tested this way, and the code says so rather than guessing.
  * `orange_box` -- fraction of orange in the visible-aperture bbox padded by VIS_PAD.
    Secondary, kept because it is the measure the problem was first found with, and because
    it is the fallback when `band_px` is too small to judge.
  * `vis_px` -- longest side of the visible (quad ^ image rect) bbox. Below MIN_VIS_BOX_PX the
    instance is a sliver with no content to learn from.
  * `occl_frac` -- the gate-on-gate covered fraction as a NUMBER; `occluded` is this
    thresholded at OCCL_FRAC.
  * `visibility` -- the three-way verdict, a string:
        "visible"  nothing says otherwise;
        "partial"  gate-on-gate occluded, or orange support weak but present
                   (ORANGE_HIDDEN <= support < ORANGE_PARTIAL);
        "hidden"   no gate is there: support < ORANGE_HIDDEN, or vis_px < MIN_VIS_BOX_PX.
  * `reason` -- on "hidden" only, which test fired.

HARD NEGATIVES, NOT DELETIONS. "hidden" instances are written to a SECOND file
(`--out-negatives`, default `<out>_negatives.json`) in the IDENTICAL schema, rather than
dropped. A crop where geometry says "gate" and the image says otherwise is precisely the
signal that teaches a confidence head to check the pixels, and it is free -- it costs
nothing to keep and cannot be recovered later without re-running the projection.

  * Existing consumers are UNAFFECTED and need no change: they open `--out`, which now
    contains positives only, with strictly additive keys per instance.
  * A confidence head opts in by ALSO reading the negatives file. It has the same
    `{"<session>/<file>": [inst, ...]}` shape, so `gatenet.load_index()` reads it unchanged
    (point `gatenet.LABELS` at it) and `packcrops.pack` will tile it unchanged. Positives
    are class 1, negatives class 0.
  * A frame that appears ONLY in the negatives file is still not a verified negative in the
    UI's sense -- other gates may be present unlabelled -- so the two files must not be
    merged by concatenating frame keys and treating a missing key as "no gate".

  * NOT ALL NEGATIVES ARE THE SAME THING, and a confidence head must filter on `reason`.
    Measured over the two VQ1 sessions: 437 of the 454 carry `reason: "no-orange"` -- the
    projection lands on empty void or on hangar structure, no gate is there, and those are
    the true hard negatives. The other 17 carry `reason: "degenerate-visible-box"`, and a
    contact sheet of all 17 shows they are REAL gates the aircraft has just flown past,
    clipped until under 2 px of the quad is still on screen (their orange support reads
    100%, because the band lands on the gate filling the frame). They are unlearnable for a
    corner regressor -- there is no aperture left in the crop -- but teaching a confidence
    head that they are "not a gate" would be a lie of exactly the kind this test exists to
    remove. **Take `reason == "no-orange"` as class 0 and DROP the degenerate ones from both
    classes.**

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

The visibility test is NOT a rejection rule: a "hidden" instance is emitted, to the
negatives file. Nothing is thrown away.

WHAT IT ACTUALLY DID, on the two VQ1 sessions (2026-08-01):

    13008 projections -> 12554 positives + 454 hard negatives (3.49%), 423 of the
    positives flagged visibility=partial. By size band, removed/before:
    0-15 px 203/4514 (4.5%), 15-30 px 234/4082 (5.7%), 30-60 px 3/1715 (0.2%),
    60-120 px 14/1391 (1.0%), >=120 px 0/1306 (0.0%);
    clipped 32/2691 (1.2%), unclipped 422/10317 (4.1%).

The removals are SMALL and UNCLIPPED, which is the signature of gates hidden behind hangar
structure rather than of edge effects -- and it is what the sample that motivated this test
predicted. Two independent checks that the cut is not eating real gates: every one of the
13 removals above 60 px was rendered and inspected (six are gate 3 projected into empty
sky with the camera pitched up, seven are gate 2 projected into ground terrain with the
real gates plainly visible above); and the mean brightness of the frame band over the whole
removed set has p90 = 44 of 255 against a median of 215 for kept instances, i.e. the
removed set is sitting on darkness, not on unrecognised gates.

    python3 pilot/perception/autolabel.py <session-dir> [<session-dir> ...] \
        [--out autolabels.json] [--min-px 6] [--gate5]
    python3 pilot/perception/autolabel.py <session-dir> ... --verify verify/ [--n 12]
    python3 pilot/perception/autolabel.py <session-dir> ... --sweep sweepdir/
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

# --- visibility test (see the module docstring) ----------------------------------------
OUTER_RATIO = 2.7 / 1.5   # detect.GATE_OUTER_M / GATE_INNER_M -- the orange frame's extent
VIS_PAD = 0.35            # padding on the visible bbox for the secondary `orange_box`
MIN_BAND_PX = 40.0        # px^2 of frame band on screen below which orange_band is not
                          # trustworthy and orange_box is used instead. A 6 px aperture --
                          # the MIN_SIZE_PX floor -- already gives a ~76 px^2 band, so this
                          # only ever fires on gates clipped down to a sliver.
MIN_VIS_BOX_PX = 2.0      # longest visible-bbox side; below this there is no picture
# CHOSEN BY SWEEP, over all 13008 VQ1 instances (`--sweep`, 2026-08-01). The support
# distribution is BIMODAL and the cut sits in the gap, which is why the exact value barely
# matters: removals go 474 (thr 0.005) -> 478 (0.01) -> 493 (0.02) -> 502 (0.03) -> 527
# (0.05), i.e. +28 instances across a 6x range of threshold, and only then start climbing
# (569 at 0.10, 677 at 0.20). 0.02 is the middle of that plateau. Percentiles of the same
# number: p3 = 0.000, p5 = 0.192, p50 = 0.753 -- there is essentially nothing between "no
# orange at all" and "obviously a gate".
ORANGE_HIDDEN = 0.02      # support below this -> "hidden"
# The partial band is the softer call and is NOT plateau-protected: it marks the bottom ~5%
# by orange support (p5 = 0.19) as weakly evidenced. It changes no data, only a flag, and
# every input to it is stored per instance, so re-deciding it costs a re-read and not a
# re-run.
ORANGE_PARTIAL = 0.20     # support below this -> "partial". See `--sweep`.


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


def _scale_quad(q, f):
    """Quad scaled about its own centroid. Used to get the 2700 mm outer boundary from the
    1500 mm aperture -- a similarity in the gate plane, so it is only exact under weak
    perspective, which is fine for a colour-support test but would not be for geometry."""
    q = np.asarray(q, np.float32)
    c = q.mean(axis=0)
    return (q - c) * f + c


_D = None       # detect.py, imported lazily: `gatenet` imports this module for canon()
                # alone and should not pay for it.


PALE_S_MIN = 30           # saturation floor for the bloom-tolerant channel; see masks_of


def _detect():
    global _D
    if _D is None:
        import detect as _det
        _D = _det
    return _D


def orange_mask_of(bgr):
    """detect.py's mask, reused verbatim -- no second colour test to drift out of sync."""
    return _detect().orange_mask(bgr)


def masks_of(bgr):
    """-> (orange, pale). `orange` IS `detect.orange_mask`, unchanged and still primary.

    `pale` is that same test with ONE constant relaxed: the saturation floor drops from
    detect.S_MIN (110) to PALE_S_MIN (30). Hue and value bounds are detect.py's own, read
    from the module so they cannot drift apart.

    WHY IT IS NEEDED, and it was found by looking rather than reasoning. `orange_mask` is
    tuned for DETECTION, where S>110 is what keeps the white ceiling lights out -- NOTES.md
    measures orange gate S=202 against lights S=3. But a gate that is distant, bloomed or
    backlit renders PALE PINK: still orange-hued and bright, saturation pushed below 110.
    That is invisible to the primary mask, so a visibility test built on it alone calls a
    perfectly real gate "not there". Measured: of the 476 instances the primary channel
    would have removed, the band of 25 of them is >=10% bright orange-hued desaturated
    pixels, and a contact sheet of the top 24 by that fraction shows unmistakable gates --
    including a 21 px and a 25 px one with the projected aperture landing dead on.

    WHY THE RELAXATION IS SAFE HERE AND WOULD NOT BE IN detect.py. This test never SEARCHES
    the frame; it asks whether gate-like material sits in one band whose position geometry
    already fixed. A false keep therefore needs bright orange-hued material at exactly the
    projected frame position. The two known confusers are still excluded: ceiling lights sit
    at S~3, an order of magnitude below PALE_S_MIN, and the cyan ribbon is excluded on hue.
    The cost is measured below rather than assumed -- see `--sweep`."""
    D = _detect()
    orange = D.orange_mask(bgr)          # CALLED, not reimplemented -- it cannot drift
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue = (h <= D.HUE_LO) | (h >= D.HUE_HI)
    pale = cv2.morphologyEx(
        (hue & (s > PALE_S_MIN) & (v > D.V_MIN)).astype(np.uint8),
        cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return orange, pale


def orange_metrics(mask, uv, vis, pale=None):
    """What the PIXELS say about a projected instance. See the docstring's ADDITIVE FIELDS.

    `mask` is detect.orange_mask's uint8 0/1 image, `pale` the bloom-tolerant companion from
    `masks_of` (optional), `uv` the amodal aperture quad, `vis` the quad ^ image-rect
    polygon. Returns (orange_band, band_px, orange_box, vis_px, pale_band).

    fillPoly clips to the raster, so every count below is already "inside the image" and no
    separate intersection is needed."""
    inner = np.round(np.asarray(uv, np.float32)).astype(np.int32)
    outer = np.round(_scale_quad(uv, OUTER_RATIO)).astype(np.int32)
    band = np.zeros(mask.shape[:2], np.uint8)
    cv2.fillPoly(band, [outer.reshape(-1, 1, 2)], 1)
    cv2.fillPoly(band, [inner.reshape(-1, 1, 2)], 0)
    band_px = float(band.sum())
    orange_band = float((band & mask).sum()) / band_px if band_px > 0 else float('nan')

    p = np.asarray(vis, np.float32)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    vis_px = float(max(x1 - x0, y1 - y0))
    pad = VIS_PAD * max(vis_px, 1.0)
    bx0 = int(max(0, np.floor(x0 - pad)))
    by0 = int(max(0, np.floor(y0 - pad)))
    bx1 = int(min(mask.shape[1], np.ceil(x1 + pad) + 1))
    by1 = int(min(mask.shape[0], np.ceil(y1 + pad) + 1))
    sub = mask[by0:by1, bx0:bx1]
    orange_box = float(sub.mean()) if sub.size else float('nan')

    pale_band = float('nan')
    if pale is not None and band_px > 0:
        pale_band = float((band & pale).sum()) / band_px
    return orange_band, band_px, orange_box, vis_px, pale_band


def support(d):
    """The single number the verdict is taken on: the LARGEST of the three orange channels
    -- `orange_band`, the bloom-tolerant `pale_band`, and the crude `orange_box`.

    `orange_band` is the better measure almost everywhere, and taking a max with the cruder
    `orange_box` looks like weakening it. It is not, and the reason was measured, not
    reasoned about. `_scale_quad` builds the outer boundary as a similarity about the
    aperture centroid, which is only right under weak perspective; on a gate the aircraft is
    about to fly THROUGH, the band degenerates to a couple of pixel-wide slivers that can
    land in the dark gap beside the frame. Four such instances (684-1172 px apparent size,
    gates 1 and 4) measured `orange_band` 0.00-0.04 while `orange_box` said 25-39% -- and the
    frames show unmistakable gates filling the screen. A cut on the band alone eats them.

    So: orange ANYWHERE it is looked for is evidence the gate is there, and only agreement
    between both channels calls an instance empty. Cost of the max, measured over all 13008
    instances at threshold 0.02: 18 fewer removals (511 -> 493), of which the 3 largest are
    the confirmed real gates above and the rest are 7-35 px with 2.7-5.4% box orange. That
    is the right trade -- a filter that eats good data is worse than the noise it removes.

    Kept separate from `classify` so a re-threshold later reuses exactly this rule."""
    s = []
    trust_band = (d.get('band_px') or 0.0) >= MIN_BAND_PX
    for k in ('orange_band', 'pale_band'):
        v = d.get(k, np.nan)
        if trust_band and np.isfinite(v):
            s.append(float(v))
    box = d.get('orange_box', np.nan)
    if np.isfinite(box):
        s.append(float(box))
    return max(s) if s else 0.0


def classify(d, hidden=ORANGE_HIDDEN, partial=ORANGE_PARTIAL):
    """-> (visibility, reason). Pure function of the numbers already in `d`, so the whole
    sweep runs without touching an image twice."""
    if d.get('vis_px', 0.0) < MIN_VIS_BOX_PX:
        return 'hidden', 'degenerate-visible-box'
    s = support(d)
    if s < hidden:
        return 'hidden', 'no-orange'
    if d.get('occluded'):
        return 'partial', 'gate-on-gate'
    if s < partial:
        return 'partial', 'weak-orange'
    return 'visible', ''


def frame_instances(pose, gates, min_px=MIN_SIZE_PX, mask=None, pale=None):
    """All labelable gate instances for one camera pose. Returns a list of dicts with
    the output schema plus 'gate' (race index) and 'range_m' for internal use.

    `mask` is detect.orange_mask of the SAME frame. Pass it and the visibility fields are
    filled in; omit it and they are absent, which is what `--verify` wants (it is checking
    the projection, and reading a JPEG twice per frame there buys nothing)."""
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
        occl_frac = float(covered / c['vis_area']) if c['vis_area'] > 0 else 0.0
        occl = bool(occl_frac >= OCCL_FRAC)
        uv = canon(c['uv'])
        clipped = bool(np.any(uv[:, 0] < 0) or np.any(uv[:, 0] > L.W - 1) or
                       np.any(uv[:, 1] < 0) or np.any(uv[:, 1] > L.H - 1))
        d = {
            'corners': [[round(float(u), 2), round(float(v), 2)] for u, v in uv],
            'occluded': occl,
            'clipped': clipped,
            'unsure': False,
            'z': k,
            'occl_frac': round(min(occl_frac, 1.0), 4),
            'gate': c['gate'],
            'range_m': c['range_m'],
            'size_px': c['size_px'],
        }
        if mask is not None:
            ob, bpx, obox, vpx, opale = orange_metrics(mask, c['uv'], c['vis'], pale)
            d['orange_band'] = None if not np.isfinite(ob) else round(ob, 4)
            d['band_px'] = round(bpx, 1)
            d['orange_box'] = None if not np.isfinite(obox) else round(obox, 4)
            d['vis_px'] = round(vpx, 2)
            d['pale_band'] = None if not np.isfinite(opale) else round(opale, 4)
            # classify() reads these keys, and json None must not be mistaken for 0.0
            probe = dict(d)
            probe['orange_band'] = ob
            probe['orange_box'] = obox
            probe['pale_band'] = opale
            d['visibility'], reason = classify(probe)
            if reason:
                d['reason'] = reason
        out.append(d)
    return out


def scan_session(session, gates, min_px=MIN_SIZE_PX, vision=True, progress=0):
    """Yield (key, image_path, instances) per posed frame, WITH the visibility numbers and
    WITHOUT any threshold applied. The one pass every mode below shares, so `--sweep` and
    the real run can never disagree about what was measured."""
    name = os.path.basename(os.path.normpath(session))
    track = L.PoseTrack(session)
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    for i, rec in enumerate(frames):
        pose = track.at_time(float(rec['sim_time_ns']) / 1e9)
        if pose is None:
            continue
        path = os.path.join(session, 'frames', rec['file'])
        mask = None
        if vision:
            # Cheap pre-pass: project WITHOUT the mask first, so a frame with no candidate
            # never pays for a JPEG decode. Most frames have a candidate, but the ones that
            # do not are free.
            if not frame_instances(pose, gates, min_px):
                yield f'{name}/{rec["file"]}', path, []
                continue
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:                     # unreadable frame: no pixels, no verdict
                mask = None
            else:
                mask, pale = masks_of(img)
        if progress and i % progress == 0:
            print(f'  {name} {i}/{len(frames)}', flush=True)
        yield (f'{name}/{rec["file"]}', path,
               frame_instances(pose, gates, min_px, mask, pale))


UI_KEYS = ('corners', 'occluded', 'clipped', 'unsure', 'z')
# `gate` and `size_px` ride along too. Both were computed and then thrown away before, and
# both are diagnosis: a removal count concentrated on ONE gate index is evidence of a bad
# centre in the truth file rather than of occlusion, and that check is worthless if the
# label file cannot say which gate an instance was. `size_px` is exactly what
# `gatenet.load_index` recomputes from the corners.
EXTRA_KEYS = ('gate', 'size_px', 'occl_frac', 'orange_band', 'pale_band', 'band_px',
              'orange_box', 'vis_px', 'visibility', 'reason')


def emit(d, z):
    """One instance in the on-disk schema: the UI's five keys first (byte-for-byte what
    every existing consumer reads), then the additive ones."""
    out = {k: d[k] for k in UI_KEYS}
    out['z'] = z
    for k in EXTRA_KEYS:
        if k in d:
            out[k] = round(d[k], 2) if k == 'size_px' else d[k]
    return out


VERDICT_COLS = ('key', 'set', 'idx', 'gate', 'visibility', 'reason', 'support',
                'orange_band', 'pale_band', 'band_px', 'orange_box', 'vis_px', 'occl_frac',
                'size_px', 'clipped', 'occluded')


def verdict_rows(labels, negs):
    """One row per emitted instance, keyed EXACTLY as the label files are.

    JOIN KEY IS (key, set, idx): `key` is '<session>/<file>', `set` is 'pos' (the file a
    consumer opens as `--out`) or 'neg' (the hard negatives), and `idx` is the instance's
    POSITION IN THAT FILE'S LIST -- which is also its `z`, since `emit()` re-indexes z
    contiguously per list. That is the same enumeration `gatenet.load_index()` produces
    (sorted keys, then list order), so a hand triage done through the label file joins to
    this table with no guesswork and no coordinate matching.

    Written so an automated verdict and a human one are two INDEPENDENT measurements of the
    same instance. Where they disagree is where the threshold is wrong."""
    rows = []
    for setname, src in (('pos', labels), ('neg', negs)):
        for key in sorted(src):
            for i, d in enumerate(src[key]):
                probe = {k: (np.nan if d.get(k) is None else d.get(k, np.nan))
                         for k in ('orange_band', 'pale_band', 'band_px', 'orange_box')}
                rows.append({
                    'key': key, 'set': setname, 'idx': i,
                    'gate': d.get('gate', ''),
                    'visibility': d.get('visibility', ''),
                    'reason': d.get('reason', ''),
                    'support': round(support(probe), 4),
                    'orange_band': d.get('orange_band', ''),
                    'pale_band': d.get('pale_band', ''),
                    'band_px': d.get('band_px', ''),
                    'orange_box': d.get('orange_box', ''),
                    'vis_px': d.get('vis_px', ''),
                    'occl_frac': d.get('occl_frac', ''),
                    'size_px': round(float(d.get('size_px', 0.0)), 2),
                    'clipped': int(bool(d['clipped'])),
                    'occluded': int(bool(d['occluded'])),
                })
    return rows


def write_verdicts(path, rows):
    import csv
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, VERDICT_COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def label_session(session, gates, min_px=MIN_SIZE_PX, vision=True, progress=0):
    """-> (labels, negatives, stats), both dicts keyed '<session>/<file>'.

    `labels` holds the visible + partly-occluded instances; `negatives` holds the ones the
    image says are not there. Same schema, so either file loads through the same reader.
    `z` is re-indexed CONTIGUOUSLY within each output list -- it is a draw order, and a
    gappy one would render wrong in the UI. The occlusion computation upstream still saw
    every candidate, so nothing is lost by re-indexing here."""
    nframes = sum(1 for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file'])
    labels, negs = {}, {}
    stats = {'frames': nframes, 'posed': 0, 'inst': 0, 'sizes': [], 'clipped': 0,
             'occluded': 0, 'hidden': 0, 'partial': 0, 'per_gate': {}}
    for key, _path, inst in scan_session(session, gates, min_px, vision, progress):
        stats['posed'] += 1
        if not inst:
            continue
        keep = [d for d in inst if d.get('visibility', 'visible') != 'hidden']
        drop = [d for d in inst if d.get('visibility', 'visible') == 'hidden']
        for d in inst:
            stats['inst'] += 1
            stats['sizes'].append(d['size_px'])
            stats['clipped'] += d['clipped']
            stats['occluded'] += d['occluded']
            stats['hidden'] += d.get('visibility') == 'hidden'
            stats['partial'] += d.get('visibility') == 'partial'
            g = str(d['gate'])
            stats['per_gate'][g] = stats['per_gate'].get(g, 0) + 1
        if keep:
            labels[key] = [emit(d, i) for i, d in enumerate(keep)]
        if drop:
            negs[key] = [emit(d, i) for i, d in enumerate(drop)]
    stats['kept'] = sum(len(v) for v in labels.values())
    stats['dropped'] = sum(len(v) for v in negs.values())
    return labels, negs, stats


# --------------------------------------------------------------------------------------
# threshold sweep + eyeball sheets


def _tile(path, d, side=144):
    """One contact-sheet tile: the frame cropped around the instance's VISIBLE box, with
    the projected aperture drawn on it. The crop is what a human needs to answer "is a gate
    there"; the overlay is what says where the label claimed one was."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    uv = np.array(d['corners'], np.float32)
    vis = _clip_to_image(uv)
    if vis is None:
        p = np.clip(uv, [0, 0], [L.W - 1, L.H - 1])
    else:
        p = vis
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    s = max(float(max(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min())) * 3.0,
            56.0)
    a = side / s
    M = np.array([[a, 0, side / 2.0 - cx * a], [0, a, side / 2.0 - cy * a]], np.float32)
    t = cv2.warpAffine(img, M, (side, side), flags=cv2.INTER_LINEAR)
    q = (uv * a + M[:, 2]).astype(np.int32)
    cv2.polylines(t, [q.reshape(-1, 1, 2)], True, (0, 255, 255), 1)
    sup = support({k: (np.nan if d.get(k) is None else d.get(k, np.nan))
                   for k in ('orange_band', 'pale_band', 'band_px', 'orange_box')})
    cv2.putText(t, f'{100*sup:.1f}%', (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 255, 255), 1)
    cv2.putText(t, f'{d["size_px"]:.0f}px{" C" if d["clipped"] else ""}',
                (3, side - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 255), 1)
    return t


def contact_sheet(recs, path, cols=10, side=144):
    tiles = [t for t in (_tile(r['path'], r, side) for r in recs) if t is not None]
    if not tiles:
        print(f'  (no tiles for {path})')
        return
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * side, cols * side, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * side:(r + 1) * side, c * side:(c + 1) * side] = t
    cv2.imwrite(path, sheet)
    print(f'wrote {path}  ({len(tiles)} tiles)')


def sweep(sessions, gates, outdir, min_px=MIN_SIZE_PX, n_sheet=60, seed=0, reuse=False):
    """Measure the visibility threshold instead of choosing one.

    Prints how many instances each candidate threshold removes, split by size band and by
    clipped/unclipped so the change to the TRAINING DISTRIBUTION is visible and not just the
    headline count, and renders three sheets: instances just below the chosen cut (removed),
    instances just above it (kept), and a random sample of the removed set. The middle sheet
    is the one that can fail informatively -- if it is full of real gates the cut is eating
    good data, which is worse than the noise it removes."""
    os.makedirs(outdir, exist_ok=True)
    mpath = os.path.join(outdir, 'metrics.json')
    if reuse and os.path.exists(mpath):
        # The point of storing NUMBERS rather than flags: a new threshold is a re-read, not
        # a re-run. The scan is ~10 min of JPEG decode over 6598 frames; this is ~2 s.
        recs = json.load(open(mpath))
        print(f'reusing {mpath} ({len(recs)} instances) -- no frames re-read')
    else:
        recs = []
        for s in sessions:
            for key, path, inst in scan_session(s, gates, min_px, True, progress=1000):
                for d in inst:
                    r = {k: d[k] for k in ('corners', 'clipped', 'occluded', 'size_px',
                                           'occl_frac', 'gate')}
                    for k in ('orange_band', 'pale_band', 'band_px',
                              'orange_box', 'vis_px', 'visibility'):
                        r[k] = d.get(k)
                    r['path'] = path
                    r['key'] = key
                    recs.append(r)
        json.dump(recs, open(mpath, 'w'))
    n = len(recs)
    print(f'\n{n} instances measured over {len(sessions)} sessions')

    sup = np.array([support({k: (np.nan if r.get(k) is None else r[k])
                             for k in ('orange_band', 'pale_band', 'band_px', 'orange_box')})
                    for r in recs])
    size = np.array([r['size_px'] for r in recs])
    clip = np.array([bool(r['clipped']) for r in recs])
    vispx = np.array([r['vis_px'] if r['vis_px'] is not None else 0.0 for r in recs])
    bandpx = np.array([r['band_px'] if r['band_px'] is not None else 0.0 for r in recs])
    degen = vispx < MIN_VIS_BOX_PX

    print(f'\nband_px < MIN_BAND_PX ({MIN_BAND_PX:.0f}), i.e. judged on orange_box '
          f'instead: {int((bandpx < MIN_BAND_PX).sum())} ({100*(bandpx < MIN_BAND_PX).mean():.2f}%)')
    print(f'degenerate visible box (< {MIN_VIS_BOX_PX} px): {int(degen.sum())} '
          f'({100*degen.mean():.2f}%)')
    print('\nsupport (orange fraction) percentiles: ' + '  '.join(
        f'p{p}={np.percentile(sup, p):.3f}' for p in (1, 2, 5, 10, 25, 50, 75, 90)))

    bands = ((0, 15), (15, 30), (30, 60), (60, 120), (120, 1e9))
    print('\nthreshold sweep -- instances the cut REMOVES (degenerate boxes always removed)')
    hdr = (f'{"thr":>6} {"removed":>8} {"%":>6} | ' +
           ' '.join(f'{lo:.0f}-{hi:.0f}'.replace('-1e+09', '+').rjust(9) for lo, hi in bands) +
           f' | {"clipped":>9} {"unclip":>9}')
    print(hdr)
    for t in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30):
        m = degen | (sup < t)
        cells = []
        for lo, hi in bands:
            k = (size >= lo) & (size < hi)
            cells.append(f'{int((m & k).sum())}/{int(k.sum())}'.rjust(9))
        print(f'{t:6.3f} {int(m.sum()):8d} {100*m.mean():6.2f} | ' + ' '.join(cells) +
              f' | {int((m & clip).sum()):9d} {int((m & ~clip).sum()):9d}')

    rng = np.random.default_rng(seed)
    thr = ORANGE_HIDDEN
    below = np.flatnonzero((~degen) & (sup < thr))
    above = np.flatnonzero((~degen) & (sup >= thr) & (sup < thr * 3))
    for name, idx in (('removed_marginal', below[np.argsort(-sup[below])][:n_sheet]),
                      ('kept_marginal', above[np.argsort(sup[above])][:n_sheet]),
                      ('removed_random', rng.permutation(below)[:n_sheet]),
                      ('degenerate', np.flatnonzero(degen)[:n_sheet])):
        contact_sheet([recs[int(i)] for i in idx],
                      os.path.join(outdir, f'{name}.png'))
    print(f'\nsheets rendered at threshold {thr}: removed_marginal is the top {n_sheet} '
          f'instances JUST BELOW it, kept_marginal the ones just above.')


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


def load_gate_set(path, gate5=None):
    """Race-indexed gate centres. `gates_refined.json` keys are ONE-BASED (race index =
    key - 1, verified against the position stream -- see the docstring); `gate5.py`'s
    output is already race-indexed and is added only on request."""
    gates = {k - 1: v for k, v in L.load_gates(path).items()}
    if gate5:
        rec = json.load(open(gate5))
        c = rec['centre']
        gates[int(rec['gate'])] = np.array([c['x'], c['y'], c['z']], float)
    return gates


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--gates', default=os.path.join(here, 'gates_refined.json'))
    ap.add_argument('--out', default='autolabels.json')
    ap.add_argument('--out-negatives', default=None,
                    help='hard negatives (default <out>_negatives.json)')
    ap.add_argument('--min-px', type=float, default=MIN_SIZE_PX)
    ap.add_argument('--no-vision', action='store_true',
                    help='skip the visibility test entirely (old behaviour, no JPEGs read)')
    ap.add_argument('--gate5', nargs='?', const=os.path.join(here, 'gate5_recovered.json'),
                    default=None,
                    help='ALSO label VQ1 race gate 5 from gate5_recovered.json. OFF by '
                         'default: it changes the instance population, so a run with it on '
                         'is not comparable to the tables in TRAINING.md.')
    ap.add_argument('--verify', metavar='DIR')
    ap.add_argument('--sweep', metavar='DIR')
    ap.add_argument('--reuse', action='store_true',
                    help='--sweep: re-threshold from a previous run\'s metrics.json '
                         'instead of re-reading every frame')
    ap.add_argument('--verdicts', default=None,
                    help='also write a flat per-instance verdict table (CSV) joinable to '
                         'the label files on (key, set, idx). Default <out>_verdicts.csv')
    ap.add_argument('--n', type=int, default=12)
    args = ap.parse_args()

    gates = load_gate_set(args.gates, args.gate5)
    print(f'{len(gates)} gate centres from {args.gates}'
          f'{" + " + os.path.basename(args.gate5) if args.gate5 else ""}'
          f' -> race indices {sorted(gates)}')

    if args.verify:
        verify(args.sessions, gates, args.verify, args.n, args.min_px)
        return
    if args.sweep:
        sweep(args.sessions, gates, args.sweep, args.min_px, reuse=args.reuse)
        return

    negpath = args.out_negatives or (os.path.splitext(args.out)[0] + '_negatives.json')
    allsizes, out, neg = [], {}, {}
    total = {'frames': 0, 'posed': 0, 'inst': 0, 'clipped': 0, 'occluded': 0,
             'hidden': 0, 'partial': 0, 'kept': 0, 'dropped': 0}
    for s in args.sessions:
        labels, negs, st = label_session(s, gates, args.min_px, not args.no_vision,
                                         progress=2000)
        out.update(labels)
        neg.update(negs)
        allsizes += st['sizes']
        for k in total:
            total[k] += st[k]
        print(f'{os.path.basename(os.path.normpath(s))}: {st["inst"]} instances over '
              f'{st["posed"]} posed frames -> {st["kept"]} kept / {st["dropped"]} '
              f'not-visible, per-gate {st["per_gate"]}')

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    with open(negpath, 'w') as fh:
        json.dump(neg, fh, indent=1)
    vpath = args.verdicts or (os.path.splitext(args.out)[0] + '_verdicts.csv')
    rows = verdict_rows(out, neg)
    write_verdicts(vpath, rows)
    print(f'wrote {vpath}: {len(rows)} per-instance verdicts, joinable on (key, set, idx)')
    a = np.array(allsizes) if allsizes else np.zeros(1)
    print(f'wrote {args.out}: {len(out)} frames, {total["kept"]} instances '
          f'({total["clipped"]} clipped and {total["occluded"]} gate-occluded across all '
          f'{total["inst"]} projected)')
    print(f'wrote {negpath}: {len(neg)} frames, {total["dropped"]} HARD NEGATIVES '
          f'({100.0*total["dropped"]/max(total["inst"],1):.2f}% of projections); '
          f'{total["partial"]} kept instances are flagged visibility=partial')
    print(f'apparent size px (all projections): min {a.min():.1f}  '
          f'p10 {np.percentile(a, 10):.1f}  median {np.median(a):.1f}  '
          f'p90 {np.percentile(a, 90):.1f}  max {a.max():.1f}')


if __name__ == '__main__':
    main()
