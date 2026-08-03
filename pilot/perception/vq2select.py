"""Choose the VQ2 frames worth hand-labelling, and stage them for `labelui.html`.

WHY SELECTION MATTERS MORE THAN THE TOOL. gatenet is trained entirely on VQ1, which has
no wordmark, no checkerboard, no hangar clutter and no cyan ribbon across an aperture, and
it has no confidence head, so decoration false positives are wholly unaddressed
(`TRAINING.md`, caveats 3 and 4). Claire has about half an hour. A uniform sample of a
6303-frame lap would spend most of it re-labelling the easy near gate. So frames are
scored into the five categories VQ1 CANNOT TEACH and sampled from each:

  deco     a decoration false positive is present -- `labelgates.drop_decorations_by_
           parent()` rejected at least one detection. These are the negatives a
           confidence head needs; there is no other source of them.
  ribbon   the cyan guidance ribbon crosses the aperture OUTLINE. Measured as the cyan
           fraction of a 7 px ring on the detected quad's outline, which is the same
           measure TRAINING.md uses -- and deliberately NOT the interior, because an
           interior proxy already produced one confounded non-result in this project.
  merged   two detections overlap in image space (failure mode 3). Nothing detects this
           today; `quality()` does not catch it, because nothing about the blob is
           individually anomalous.
  clipped  a detection touches the image border (failure mode 2).
  empty    no detection at all, and negligible orange. labelui records an empty array as
           a VERIFIED NEGATIVE, which is a different fact from an unlabelled frame, and
           hard negatives are what teach a confidence head to check pixels.

Detections come from `vq2cache.py`'s pickles, so no frame is re-detected here.

DE-DUPLICATION. 30 Hz, so neighbours are near-duplicates: a frame is rejected if an
already-selected frame in the same session is within MIN_GAP frames.

ORDERING. Output files are prefixed `NNN_` in priority order, interleaved across
categories, because `labelui.html` sorts by filename. If Claire stops at 40% she gets a
balanced 40% rather than the first 40% of the lap. The merge step strips the prefix.

    python3 pilot/perception/vq2select.py [--out pilot/perception/vq2_label] [--n 170]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L  # noqa: E402

SESSIONS = os.path.abspath(os.path.join(HERE, '..', 'sessions'))
CACHE = os.path.join(HERE, 'cache')

# Sessions to draw from, with the frame ranges session-notes.txt says are usable.
# EXCLUDES are the two regions Claire's notes flag in the primary lap: a real race-index
# reversal, and a 32 s backtrack the index does not record.
SOURCES = [
    ('20260801-121520-vq2-lap-0-15', (102853, 108061),
     [(102799, 102987), (105273, 106229)]),
    ('20260801-115735-vq2-lap-0-11', (71665, 76349), []),
    ('20260801-114948-vq2-strafing-15-16', (59217, 59975), []),
]

MIN_GAP = 30          # frames (1.0 s) between two selected frames of one session
RIBBON_RING = 7       # px, ring width on the aperture outline
# TRAINING.md's "heavy" bucket is >=0.15. Measured here that is only 45 frames in 4779
# and NOTHING above 0.30 -- and the count is biased low by construction, because the ring
# is measured on the DETECTOR's quad and the detector's recall collapses precisely when
# the ribbon crosses the outline (recall 0.91 -> 0.65 in TRAINING.md). The frames worst
# affected are the ones with no detection to measure. So the pool is opened to 0.05 and
# candidates are taken heaviest-first, which takes all 45 heavy frames before any lighter
# one.
RIBBON_MIN = 0.05
BORDER = 2            # px, a quad this close to the edge counts as clipped
MERGE_IOU = 0.10
# Orange below which a frame with no detection is plausibly EMPTY rather than a detector
# miss. Set from the measured distribution, not chosen: of 1066 frames where the detector
# returns nothing, only 11 have <=150 px of orange and 27 have <=400 -- the median is
# 15934. A lap almost always has a gate in view, so verified negatives are intrinsically
# scarce and `empty` takes everything it can get rather than filling a quota.
EMPTY_ORANGE_MAX = 400
SEED = 20260802

# CORRECTED AFTER LOOKING AT THE FRAMES. The first pass called every no-detection frame
# `empty`, and the contact sheet showed three of four such frames containing plainly
# visible gates -- they are DETECTOR MISSES, not negatives. Labelling them is valuable,
# but as positives; recording them as verified negatives would have written wrong labels
# into the training set. The two are now separate categories.
QUOTA = {'deco': 0.22, 'ribbon': 0.22, 'merged': 0.16, 'clipped': 0.16,
         'missed': 0.14, 'empty': 0.10}


def cyan_mask(bgr):
    """NOTES.md's measured cyan class: hue 85-100, saturated and bright."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (((h >= 85) & (h <= 100)) & (s > 110) & (v > 110)).astype(np.uint8)


def ring_fraction(mask, quad, width=RIBBON_RING):
    """Fraction of a `width` px ring on the quad OUTLINE that is set in `mask`.

    The outline, not the interior: the ribbon corrupts the edge the contour follower
    tracks, and an interior measure answers a different question.
    """
    ring = np.zeros(mask.shape, np.uint8)
    cv2.polylines(ring, [np.asarray(quad, np.int32).reshape(-1, 1, 2)], True, 1, width)
    n = int(ring.sum())
    if n < 30:
        return 0.0, n
    return float((mask & ring).sum()) / n, n


def quad_contain(a, b):
    """Intersection over the SMALLER area -- catches a far gate seen through a near
    gate's aperture, which IoU scores near zero."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    m = min(abs(cv2.contourArea(a)), abs(cv2.contourArea(b)))
    return float(inter / m) if m > 0 else 0.0


def quad_iou(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    aa = abs(cv2.contourArea(a))
    ab = abs(cv2.contourArea(b))
    u = aa + ab - inter
    return float(inter / u) if u > 0 else 0.0


def touches_border(quad, w=L.W, h=L.H):
    q = np.asarray(quad, float)
    return bool(q[:, 0].min() <= BORDER or q[:, 1].min() <= BORDER or
                q[:, 0].max() >= w - 1 - BORDER or q[:, 1].max() >= h - 1 - BORDER)


def score_session(session, rng_lo, rng_hi, excludes, stride):
    """-> list of per-frame dicts with the category flags actually measured."""
    name = os.path.basename(session)
    pkl = os.path.join(CACHE, name + '.det.pkl')
    if not os.path.exists(pkl):
        print(f'  {name}: no detection cache, skipped (run vq2cache.py)')
        return []
    with open(pkl, 'rb') as fh:
        dets = pickle.load(fh)

    fdir = os.path.join(SESSIONS, name, 'frames')
    out = []
    files = sorted(dets)
    for n, fn in enumerate(files):
        num = int(os.path.splitext(fn)[0])
        if not (rng_lo <= num <= rng_hi):
            continue
        if any(lo <= num <= hi for lo, hi in excludes):
            continue
        if n % stride:
            continue
        det = dets[fn]
        path = os.path.join(fdir, fn)
        img = cv2.imread(path)
        if img is None:
            continue

        real = [d for d in det if not d['deco']]
        deco = [d for d in det if d['deco']]
        cy = cyan_mask(img)

        rib, ribn = 0.0, 0
        for d in real:
            f, k = ring_fraction(cy, d['quad'])
            if f > rib:
                rib, ribn = f, k

        # OVERLAP, not "merged", and the distinction is a measurement-setup error worth
        # recording. A truly merged pair collapses into ONE orange blob, so the detector
        # returns one detection and pairwise overlap of detections cannot express the
        # effect at all -- it found 11 frames in 4779, which is a null result about the
        # probe, not about the course. What IS measurable from detections is gates
        # CROWDING each other: nested (one seen through another's aperture) or close
        # relative to their own size. Those are the frames where amodal separation is the
        # hard part, which is the labelling value either way.
        merged = 0.0
        for i in range(len(real)):
            for j in range(i + 1, len(real)):
                a, b = real[i], real[j]
                merged = max(merged, quad_contain(a['quad'], b['quad']))
                d = float(np.linalg.norm(np.asarray(a['centre']) - np.asarray(b['centre'])))
                near = 0.75 * (a['size_px'] + b['size_px']) / 2.0
                if d < near and min(a['size_px'], b['size_px']) > 10:
                    merged = max(merged, 1.0 - d / max(near, 1e-6))

        clipped = any(touches_border(d['quad']) for d in real)

        # "empty" must be checked on pixels, not on the detector returning nothing:
        # the detector missing a gate is exactly the case we must NOT record as a
        # verified negative.
        from detect import orange_mask
        orange = int(orange_mask(img).sum())

        out.append({
            'session': name, 'file': fn, 'num': num, 'ordinal': n, 'path': path,
            'n_det': len(real), 'n_deco': len(deco),
            'ribbon': round(rib, 3), 'ring_px': ribn,
            'merged_iou': round(merged, 3), 'clipped': clipped,
            'orange_px': orange,
            'max_size': round(max([d['size_px'] for d in real], default=0.0), 1),
        })
    return out


def categorise(f):
    """Every category the frame qualifies for. A frame can serve several."""
    c = []
    if f['n_det'] == 0:
        # Either a verified negative or a detector miss, and orange is what separates
        # them. Both are OFFERED rather than asserted -- Claire is the judge, and the
        # `empty` frames still have to be confirmed by eye before Enter records a
        # negative.
        c.append('empty' if f['orange_px'] <= EMPTY_ORANGE_MAX else 'missed')
        return c
    if f['n_deco'] > 0:
        c.append('deco')
    if f['ribbon'] >= RIBBON_MIN:
        c.append('ribbon')
    if f['merged_iou'] >= MERGE_IOU:
        c.append('merged')
    if f['clipped']:
        c.append('clipped')
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'vq2_label'))
    ap.add_argument('--n', type=int, default=170)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--survey', action='store_true', help='measure and report only')
    ap.add_argument('--rescore', action='store_true', help='ignore the score cache')
    args = ap.parse_args()

    rng = random.Random(SEED)
    sc = os.path.join(CACHE, 'vq2select.scores.%d.pkl' % args.stride)
    if os.path.exists(sc) and not args.rescore:
        with open(sc, 'rb') as fh:
            frames = pickle.load(fh)
        print(f'{len(frames)} frames from the score cache ({sc})')
    else:
        frames = []
        for name, (lo, hi), ex in SOURCES:
            s = os.path.join(SESSIONS, name)
            if not os.path.isdir(s):
                print(f'  {name}: missing, skipped')
                continue
            got = score_session(s, lo, hi, ex, args.stride)
            print(f'  {name}: {len(got)} frames scored')
            frames += got
        os.makedirs(CACHE, exist_ok=True)
        with open(sc, 'wb') as fh:
            pickle.dump(frames, fh)

    for f in frames:
        f['cats'] = categorise(f)

    rb = np.array([f['ribbon'] for f in frames])
    mg = np.array([f['merged_iou'] for f in frames])
    print('\nribbon (cyan fraction of the aperture-outline ring): '
          '>0 %d, >=0.05 %d, >=0.15 %d, >=0.30 %d'
          % ((rb > 0).sum(), (rb >= .05).sum(), (rb >= .15).sum(), (rb >= .30).sum()))
    print('crowding score: >=0.10 %d, >=0.30 %d, >=0.50 %d'
          % ((mg >= .1).sum(), (mg >= .3).sum(), (mg >= .5).sum()))
    print(f'\n{len(frames)} frames scored (stride {args.stride})')
    print('%-9s %7s %7s' % ('category', 'frames', 'after dedup'))
    pools = {}
    for c in QUOTA:
        pool = [f for f in frames if c in f['cats']]
        ded = dedup(pool, {})
        pools[c] = pool
        print('%-9s %7d %7d' % (c, len(pool), len(ded)))
    if args.survey:
        return

    # ---- sample: rarest category first, so a frame that serves two goes to the scarce one
    # heaviest ribbon first (scarce), emptiest first for the negatives (least likely to
    # be a missed gate); everything else in time order from a random phase
    PREFER = {
        'ribbon': lambda f: -f['ribbon'],
        'empty': lambda f: f['orange_px'],
        'missed': lambda f: -f['orange_px'],
        'merged': lambda f: -f['merged_iou'],
    }
    taken, chosen = {}, {}
    order = sorted(QUOTA, key=lambda c: len(pools[c]))
    for c in order:
        want = max(1, int(round(args.n * QUOTA[c])))
        # verified negatives are intrinsically scarce (27 candidates in 4779 frames,
        # 3 after the standard gap) and cost ONE keystroke each, so the duplicate
        # penalty that justifies MIN_GAP barely applies to them
        gap = 8 if c == 'empty' else MIN_GAP
        got = dedup(pools[c], taken, rng, want, PREFER.get(c), gap)
        chosen[c] = got
        print(f'  picked {len(got):3d}/{want} for {c}')

    # ---- interleave so any prefix of the output is balanced
    lists = {c: list(v) for c, v in chosen.items()}
    for v in lists.values():
        rng.shuffle(v)
    seq, i = [], 0
    while any(lists.values()):
        for c in order:
            if lists[c]:
                seq.append((lists[c].pop(), c))
        i += 1

    outdir = args.out
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    fdir = os.path.join(outdir, 'frames')
    os.makedirs(fdir)
    index = []
    for i, (f, c) in enumerate(seq):
        dst = '%03d_%s' % (i, f['file'])
        shutil.copyfile(f['path'], os.path.join(fdir, dst))
        index.append({'labelui_key': dst, 'session': f['session'], 'file': f['file'],
                      'category': c, 'all_categories': f['cats'],
                      'n_det': f['n_det'], 'n_deco': f['n_deco'],
                      'ribbon': f['ribbon'], 'merged_iou': f['merged_iou'],
                      'clipped': f['clipped'], 'orange_px': f['orange_px'],
                      'max_size_px': f['max_size']})
    with open(os.path.join(outdir, 'index.json'), 'w') as fh:
        json.dump({'min_gap': MIN_GAP, 'stride': args.stride, 'seed': SEED,
                   'sources': [s[0] for s in SOURCES], 'frames': index}, fh, indent=1)
    print(f'\nwrote {len(index)} frames to {fdir}')
    print(f'  index: {os.path.join(outdir, "index.json")}')


def dedup(pool, taken, rng=None, want=10 ** 9, prefer=None, gap=None):
    """Greedy, keeping MIN_GAP frames between picks in one session. `taken` maps
    session -> list of ordinals already used, and is shared across categories.

    `prefer` ranks candidates (most wanted first) instead of walking in time order --
    used so the scarce heavy-ribbon frames are all taken before any light one.
    """
    if prefer is not None:
        order = sorted(pool, key=prefer)
    else:
        order = sorted(pool, key=lambda f: (f['session'], f['ordinal']))
        if rng is not None:
            # walk in time order but from a random phase, so one category does not always
            # win the start of the lap
            k = rng.randrange(max(1, len(order)))
            order = order[k:] + order[:k]
    got = []
    for f in order:
        if len(got) >= want:
            break
        used = taken.setdefault(f['session'], [])
        if any(abs(o - f['ordinal']) < (MIN_GAP if gap is None else gap) for o in used):
            continue
        got.append(f)
        used.append(f['ordinal'])
    return got


if __name__ == '__main__':
    main()
