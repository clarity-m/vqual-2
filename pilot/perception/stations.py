"""Read the hangar's numbered Station columns from the camera.

WHY THIS EXISTS. mapbuild.py derives gate identity from temporal tracking, and tracking
loses identity every time a gate is occluded for more than a few frames -- 59 tracks for
17 gates on 20260731-222724. The hangar hands us identity for free: every support column
carries a unique number, so a gate seen beside Station 24 is THE gate at Station 24
whether or not any track survived. NOTES.md said this before mapbuild.py was written
(":256, the Station columns are uniquely numbered, which solves data association"); this
file is that idea actually built.

Two label forms per column, both legible at the recorded 640x360:
  * a large two-digit number on the upper column face, upright, roughly fronto-parallel
  * vertical "Station N" text running down the column side

v1 reads the FACE NUMBER only. It is upright and compact, where the vertical text is
rotated by an amount that depends on viewing angle.

THE FREE ERROR CHECK: the two rows are numbered in opposite directions and a facing pair
sums to 41 (confirmed on frames 500 and 900 of 20260731-222724: 16|25 and 12|29, and
again on Claire's screenshots: 18|23, 19|22). A misread digit is therefore CATCHABLE
rather than silently placing a gate in the wrong bay.

    # 1. harvest two-digit candidates to a contact sheet, and label them by eye
    python3 stations.py <session> --harvest-pairs sheet.png --npz pairs.npz
    # 2. average the labelled glyphs into one template per digit
    python3 stations.py <session> --npz pairs.npz --build-templates station_templates.npz
    # 3. read a session and check it against the sum-to-41 invariant
    python3 stations.py <session> --report --stride 3 --limit 400

Measured 2026-07-31 on 20260731-222724: 38% of frames yield at least one station number,
16 distinct stations, reads concentrated on 24-29 where the flight actually went. On the
larger 20260730-215221 coverage falls to 8% -- that session is faster and the columns blur.
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

# Signage is desaturated against a hangar that NOTES.md measures as 66.5% below V=40 --
# but it is GREY, not white: V runs about 100-160, where NOTES.md's ceiling-light class is
# S<50 & V>200. Reusing the light threshold here found nothing at all. The gap is a gift:
# the lights are BRIGHTER than the text, so an upper bound separates the known false
# positive on colour, and shape only has to clean up the remainder.
SAT_MAX = 70
VAL_MIN = 95
VAL_MAX = 195

MIN_H, MAX_H = 7, 90          # digit height in px at 640x360
AR_LO, AR_HI = 0.28, 1.05     # width/height of a single digit
FILL_LO, FILL_HI = 0.22, 0.80  # ink fraction of the bbox; a lit panel is ~1.0
PAIR_DX = 2.2                 # neighbour within this many digit-widths, same height band


def ink_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] < SAT_MAX) & (hsv[:, :, 2] > VAL_MIN)
         & (hsv[:, :, 2] < VAL_MAX)).astype(np.uint8) * 255
    # Close 1-px gaps inside a stroke without merging neighbouring digits.
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))


def candidates(img):
    """Connected components that are shaped like a digit.

    Shape is the only discriminator available: the ceiling light strips share the mask
    exactly. They are rejected as near-solid (fill ~1.0) and elongated, where a glyph is
    a thin stroke pattern inside its box.
    """
    m = ink_mask(img)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (MIN_H <= h <= MAX_H) or w < 2:
            continue
        if not (AR_LO <= w / h <= AR_HI):
            continue
        fill = a / float(w * h)
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        # The glyph sits on a column, so its immediate surround must be DARK. A digit
        # painted on a bright surface would be a reflection or HUD text, not signage.
        pad = max(2, h // 3)
        y0, y1 = max(0, y - pad), min(img.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(img.shape[1], x + w + pad)
        ring = img[y0:y1, x0:x1].mean()
        if ring > 120:
            continue
        out.append({'bbox': (x, y, w, h), 'fill': fill, 'centre': cent[i],
                    'crop': (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255})
    return out


GLYPH = 20  # template size; digits arrive between 7 and 30 px tall


def glyph_norm(crop, n=GLYPH):
    """One digit, deskewed to a fixed box. Aspect is DISCARDED on purpose.

    A '1' and an '8' differ in aspect, but so does the same digit seen at a glancing angle,
    and the second effect is larger. Stretching to a square throws away a weak cue to kill a
    strong nuisance; the stroke pattern is what carries the identity.
    """
    return cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def pairs(cands):
    """Group candidates into two-digit numbers: same height band, horizontally adjacent."""
    out = []
    used = set()
    cs = sorted(cands, key=lambda c: c['bbox'][0])
    for i, a in enumerate(cs):
        if i in used:
            continue
        xa, ya, wa, ha = a['bbox']
        for j in range(i + 1, len(cs)):
            if j in used:
                continue
            xb, yb, wb, hb = cs[j]['bbox']
            if abs(ha - hb) > 0.35 * max(ha, hb):
                continue
            if abs(ya - yb) > 0.4 * ha:
                continue
            gap = xb - (xa + wa)
            if -1 <= gap <= PAIR_DX * wa:
                out.append((a, cs[j]))
                used.add(i)
                used.add(j)
                break
    return out


def pair_crop(img_shape, a, b, cell=56):
    """Both digits of one number, in one normalised bitmap.

    Kept as a PAIR rather than two glyphs because a two-digit station number is
    self-checking -- it must land in 1..40 -- so a labelling mistake is catchable, where a
    mislabelled loose digit would quietly poison a template.
    """
    xa, ya, wa, ha = a['bbox']
    xb, yb, wb, hb = b['bbox']
    x0, y0 = min(xa, xb), min(ya, yb)
    x1, y1 = max(xa + wa, xb + wb), max(ya + ha, yb + hb)
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for c in (a, b):
        x, y, w, h = c['bbox']
        np.maximum(canvas[y - y0:y - y0 + h, x - x0:x - x0 + w], c['crop'],
                   out=canvas[y - y0:y - y0 + h, x - x0:x - x0 + w])
    s = cell - 8
    sc = min(s / canvas.shape[1], s / canvas.shape[0])
    g = cv2.resize(canvas, (max(1, int(canvas.shape[1] * sc)), max(1, int(canvas.shape[0] * sc))),
                   interpolation=cv2.INTER_NEAREST)
    pad = np.zeros((cell, cell), np.uint8)
    yy, xx = (cell - g.shape[0]) // 2, (cell - g.shape[1]) // 2
    pad[yy:yy + g.shape[0], xx:xx + g.shape[1]] = g
    return pad


def harvest_pairs(session, out_png, out_npz, stride, limit, cell=56):
    """Contact sheet of two-digit numbers, indexed, for hand labelling."""
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    sel = frames[::stride][:limit]
    crops, meta, digits = [], [], []
    for r in sel:
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is None:
            continue
        for a, b in pairs(candidates(img)):
            crops.append(pair_crop(img.shape, a, b, cell))
            meta.append((r['file'], a['bbox'], b['bbox']))
            digits.append((glyph_norm(a['crop']), glyph_norm(b['crop'])))
    if not crops:
        print('no pairs')
        return
    cols, lab_h = 16, 18
    rows = (len(crops) + cols - 1) // cols
    sheet = np.zeros((rows * (cell + lab_h), cols * cell, 3), np.uint8)
    for k, g in enumerate(crops):
        rr, cc = divmod(k, cols)
        y0 = rr * (cell + lab_h)
        sheet[y0:y0 + cell, cc * cell:(cc + 1) * cell] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(sheet, (cc * cell, y0), ((cc + 1) * cell - 1, y0 + cell), (50, 50, 50), 1)
        cv2.putText(sheet, str(k), (cc * cell + 3, y0 + cell + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_png, cv2.resize(sheet, None, fx=1.5, fy=1.5,
                                    interpolation=cv2.INTER_NEAREST))
    np.savez_compressed(out_npz, digits=np.array([np.stack(d) for d in digits]),
                        files=np.array([m[0] for m in meta]))
    print(f'{len(crops)} pairs from {len(sel)} frames -> {out_png} / {out_npz}')


def harvest(session, out_png, stride, limit, cell=48):
    """Dump candidate glyph crops to a contact sheet, for eyeballing before classifying.

    The classifier needs templates in the sim's own font, and there is no labelled set --
    so the crops get looked at first and labelled by hand. Deliberately a separate step:
    guessing the glyph shapes and discovering the mask was wrong is two bugs at once.
    """
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    crops, meta = [], []
    for r in frames[::stride][:limit]:
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is None:
            continue
        for a, b in pairs(candidates(img)):
            for c in (a, b):
                g = c['crop']
                s = cell - 6
                sc = min(s / g.shape[1], s / g.shape[0])
                g = cv2.resize(g, (max(1, int(g.shape[1] * sc)), max(1, int(g.shape[0] * sc))),
                               interpolation=cv2.INTER_NEAREST)
                pad = np.zeros((cell, cell), np.uint8)
                y0 = (cell - g.shape[0]) // 2
                x0 = (cell - g.shape[1]) // 2
                pad[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
                crops.append(pad)
                meta.append((r['file'], c['bbox']))
    if not crops:
        print('no candidates')
        return
    cols = 24
    rows = (len(crops) + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell), np.uint8)
    for k, g in enumerate(crops):
        rr, cc = divmod(k, cols)
        sheet[rr * cell:(rr + 1) * cell, cc * cell:(cc + 1) * cell] = g
    sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for k in range(len(crops)):
        rr, cc = divmod(k, cols)
        cv2.rectangle(sheet, (cc * cell, rr * cell),
                      ((cc + 1) * cell - 1, (rr + 1) * cell - 1), (40, 40, 40), 1)
    cv2.imwrite(out_png, sheet)
    print(f'{len(crops)} glyph candidates from {len(frames[::stride][:limit])} frames'
          f' -> {out_png}  ({rows}x{cols} sheet)')


def overlay(session, out_png, index):
    """Draw the accepted pairs on one frame, so a miss is visible in context."""
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    r = frames[index]
    img = cv2.imread(os.path.join(session, 'frames', r['file']))
    for a, b in pairs(candidates(img)):
        xs = [a['bbox'][0], b['bbox'][0] + b['bbox'][2]]
        ys = [min(a['bbox'][1], b['bbox'][1]),
              max(a['bbox'][1] + a['bbox'][3], b['bbox'][1] + b['bbox'][3])]
        cv2.rectangle(img, (xs[0] - 2, ys[0] - 2), (xs[1] + 2, ys[1] + 2), (0, 220, 255), 1)
    big = cv2.resize(img, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_png, big)
    print(f'frame {index} ({r["file"]}) -> {out_png}')


def build_templates(npz, labels_json, out_npz):
    """Average the labelled glyphs into one template per digit.

    An average, not an exemplar: the same digit arrives at a range of scales and viewing
    angles, and averaging keeps what is common to all of them. Digits with few examples
    stay noisier, which the report below makes visible rather than hiding.
    """
    d = np.load(npz)
    glyphs = d['digits']                       # (N, 2, GLYPH, GLYPH)
    spec = json.load(open(labels_json))['labels']
    bins = collections.defaultdict(list)
    for num, idxs in spec.items():
        for i in idxs:
            if i >= len(glyphs):
                raise SystemExit(f'label index {i} past the {len(glyphs)} harvested pairs '
                                 f'-- labels and npz are from different harvests')
            bins[num[0]].append(glyphs[i][0])
            bins[num[1]].append(glyphs[i][1])
    missing = [str(k) for k in range(10) if str(k) not in bins]
    if missing:
        raise SystemExit(f'no examples for digit(s) {",".join(missing)} -- label more cells')
    keys = sorted(bins)
    T = np.stack([np.mean(bins[k], axis=0) for k in keys])
    np.savez_compressed(out_npz, templates=T, digits=np.array(keys))
    print(f'templates -> {out_npz}')
    for k in keys:
        print(f'  digit {k}: {len(bins[k]):3d} examples')
    # Worst-case confusion between the templates themselves. If two templates are already
    # closer to each other than a noisy glyph is to its own, no amount of matching helps.
    n = len(keys)
    S = np.array([[float(np.sum(T[i] * T[j]) /
                         (np.linalg.norm(T[i]) * np.linalg.norm(T[j]) + 1e-9))
                   for j in range(n)] for i in range(n)])
    np.fill_diagonal(S, 0)
    i, j = np.unravel_index(S.argmax(), S.shape)
    print(f'  closest template pair: {keys[i]} vs {keys[j]}  cos {S[i, j]:.3f}')
    return T, keys


def classify(glyph, T, keys):
    """Nearest template by cosine similarity; returns (digit, score, margin).

    The MARGIN over the runner-up is returned because it, not the score, is what says the
    read is trustworthy -- a blurred glyph can sit close to everything at once.
    """
    v = glyph.ravel()
    M = T.reshape(len(keys), -1)
    s = M @ v / (np.linalg.norm(M, axis=1) * (np.linalg.norm(v) + 1e-9) + 1e-9)
    o = np.argsort(s)[::-1]
    return keys[o[0]], float(s[o[0]]), float(s[o[0]] - s[o[1]])


def read_frame(img, T, keys, min_score=0.80, min_margin=0.02):
    """Every station number legible in one frame -> [(number, score, bbox)]."""
    out = []
    for a, b in pairs(candidates(img)):
        da, sa, ma = classify(glyph_norm(a['crop']), T, keys)
        db, sb, mb = classify(glyph_norm(b['crop']), T, keys)
        if min(sa, sb) < min_score or min(ma, mb) < min_margin:
            continue
        num = int(da + db)
        if not (1 <= num <= 40):     # a station number outside the hangar is a misread
            continue
        xa, ya, wa, ha = a['bbox']
        xb, yb, wb, hb = b['bbox']
        out.append({'station': num, 'score': min(sa, sb),
                    'bbox': (min(xa, xb), min(ya, yb),
                             max(xa + wa, xb + wb) - min(xa, xb),
                             max(ya + ha, yb + hb) - min(ya, yb))})
    return out


def report(session, tmpl_npz, stride, limit):
    """Read a session and check the reads against the sum-to-41 invariant."""
    d = np.load(tmpl_npz)
    T, keys = d['templates'], [str(x) for x in d['digits']]
    frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
    sel = frames[::stride][:limit]
    hits, nframe, pair_ok, pair_bad = collections.Counter(), 0, 0, 0
    for r in sel:
        img = cv2.imread(os.path.join(session, 'frames', r['file']))
        if img is None:
            continue
        rd = read_frame(img, T, keys)
        if rd:
            nframe += 1
        for x in rd:
            hits[x['station']] += 1
        # The invariant: two columns read in the SAME frame from opposite rows sum to 41.
        for i in range(len(rd)):
            for j in range(i + 1, len(rd)):
                s = rd[i]['station'] + rd[j]['station']
                if s == 41:
                    pair_ok += 1
                elif rd[i]['station'] != rd[j]['station']:
                    pair_bad += 1
    print(f'frames with >=1 station read : {nframe} / {len(sel)}  '
          f'({100.0 * nframe / max(1, len(sel)):.0f}%)')
    print(f'distinct stations seen       : {len(hits)}  {sorted(hits)}')
    print(f'co-visible pairs summing 41  : {pair_ok}   other cross-row pairs: {pair_bad}')
    print('reads per station:', dict(sorted(hits.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--harvest', default='')
    ap.add_argument('--harvest-pairs', default='')
    ap.add_argument('--npz', default='')
    ap.add_argument('--build-templates', default='')
    ap.add_argument('--labels', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     'station_labels.json'))
    ap.add_argument('--templates', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'station_templates.npz'))
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--overlay', default='')
    ap.add_argument('--index', type=int, default=500)
    ap.add_argument('--stride', type=int, default=60)
    ap.add_argument('--limit', type=int, default=20)
    args = ap.parse_args()
    if args.harvest:
        harvest(args.session, args.harvest, args.stride, args.limit)
    if args.harvest_pairs:
        harvest_pairs(args.session, args.harvest_pairs, args.npz, args.stride, args.limit)
    if args.build_templates:
        build_templates(args.npz, args.labels, args.build_templates)
    if args.report:
        report(args.session, args.templates, args.stride, args.limit)
    if args.overlay:
        overlay(args.session, args.overlay, args.index)


if __name__ == '__main__':
    main()
