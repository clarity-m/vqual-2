"""Build a label-triage slideshow: are the AUTO-LABELS wrong, or is the model wrong?

WHY. `TRAINING.md` reports clipped corner p90 76-96 px and >=120 px median 19-44 px,
i.e. ~10% of gate width against 3.4-4.8% everywhere else. Nothing in the training run can
tell whether that is the net failing or the net faithfully reproducing a bad projected
label. A human looking at a couple of hundred crops settles it for the cost of half an
hour, and no experiment settles it more cheaply.

WHAT IS SAMPLED. From `autolabels_vq1.json`, restricted to the **held-out** side of
`gatenet.split_index(items, 'block')` so verdicts land on exactly the instances
TRAINING.md's tables are computed over.

  A  clipped, apparent size < 120 px   ~150, stratified over gatenet.SIZE_BANDS
  B  apparent size >= 120 px           ~50
  C  unclipped, 15-60 px               ~30   HIDDEN CONTROL

MEASURED, NOT ASSUMED: the block val set contains **no unclipped instance >= 120 px** --
all 166 are clipped. So "large gates" and "clipped gates" are not two weak bands, they are
one, and stratum B cannot be the unclipped-large group the brief asked for. It is the
>=120 px band as it actually exists.

THE CONTROL IS THE POINT of stratum C. Without it a low pass-rate on clipped crops is
unattributable: it could mean the labels are bad, or that the judge is strict. C is
ordinary, easy, unclipped mid-size geometry, so its pass rate is the baseline that A and B
are read against. The three strata are interleaved in a fixed shuffled order and the UI is
given no way to tell them apart -- `manifest.json` holds the mapping and the UI never
reads it.

DE-DUPLICATION. 30 Hz video, so neighbouring frames are near-duplicates. A candidate is
rejected if an already-selected instance in the same session sits within MIN_GAP frames
AND its crop centre is within NEAR_PX. Both conditions are required: two different gates
in one frame are legitimately different crops, and the same gate 4 s later is a different
view.

    python3 pilot/perception/triage_build.py [--out pilot/perception/triage]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gatenet as G  # noqa: E402

LABELS = os.path.join(HERE, 'autolabels_vq1.json')
SESSIONS = os.path.abspath(os.path.join(HERE, '..', 'sessions'))

N_CLIPPED, N_LARGE, N_CONTROL = 150, 50, 30
# Separation. CALIBRATED, not guessed (`--calibrate` prints the table). Restricted to the
# held-out 20% there are only 29 independent clipped views at 45 frames / 70 px, because a
# gate stays clipped for a contiguous run and its amodal centre barely moves -- nowhere
# near 150. Over the FULL index the same setting yields 117, and 30 frames yields 144.
# 30 frames is 1.0 s, over which the camera moves metres at race pace; the second
# condition (60 px of centre motion) keeps two gates in ONE frame as two valid crops.
MIN_GAP = 30            # frames (1.0 s)
NEAR_PX = 60.0          # crop-centre distance below which two picks are near-duplicates
OUT_W = 1180            # rendered slide width
SEED = 20260802


# ------------------------------------------------------------------------------ index

def load_items():
    """Instance dicts with the join key. Mirrors gatenet.load_index(), plus `inst`,
    which gatenet drops and the triage export needs."""
    with open(LABELS) as fh:
        raw = json.load(fh)
    keys = sorted(raw)
    ordinal, seen = {}, {}
    for k in keys:
        s = k.split('/', 1)[0]
        ordinal[k] = seen.get(s, 0)
        seen[s] = ordinal[k] + 1

    out = []
    for k in keys:
        sess, fname = k.split('/', 1)
        for i, inst in enumerate(raw[k]):
            c = np.asarray(inst['corners'], np.float32)
            e = [float(np.linalg.norm(c[(j + 1) % 4] - c[j])) for j in range(4)]
            out.append({
                'key': k, 'inst': i, 'session': sess, 'file': fname,
                'path': os.path.join(SESSIONS, sess, 'frames', fname),
                'ordinal': ordinal[k], 'corners': c,
                'clipped': bool(inst['clipped']), 'occluded': bool(inst['occluded']),
                'size_px': float(max(e)),
                # carried through untouched for the join, and never shown to the labeller:
                # autolabel.py's automated visibility verdict on the same instance.
                'auto': {f: inst[f] for f in
                         ('gate', 'visibility', 'reason', 'occl_frac', 'orange_band',
                          'pale_band', 'band_px', 'orange_box', 'vis_px')
                         if f in inst},
            })
    return out


def centre(d):
    return np.asarray(d['corners'], np.float32).mean(axis=0)


def pick(cands, n, taken, rng, gap=None, near=None):
    """Greedy min-separation sample. `taken` is shared across strata, so a control crop
    can never be a near-duplicate of a clipped one either.

    BOTH conditions are required for a rejection. Time alone would throw away two
    different gates visible in the same frame, which are genuinely different crops;
    position alone would throw away the same gate seen 10 s later from somewhere else.
    """
    gap = MIN_GAP if gap is None else gap
    near = NEAR_PX if near is None else near
    order = list(cands)
    rng.shuffle(order)
    got = []
    for d in order:
        if len(got) >= n:
            break
        c = centre(d)
        dup = any(t['session'] == d['session']
                  and abs(t['ordinal'] - d['ordinal']) < gap
                  and float(np.linalg.norm(centre(t) - c)) < near
                  for t in taken)
        if dup:
            continue
        got.append(d)
        taken.append(d)
    return got


# ----------------------------------------------------------------------------- render

PANEL = 600


def _fit(img, w, h):
    """Letterbox into w x h without changing the aspect ratio."""
    s = min(w / float(img.shape[1]), h / float(img.shape[0]))
    r = cv2.resize(img, (max(1, int(round(img.shape[1] * s))),
                         max(1, int(round(img.shape[0] * s)))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    out = np.full((h, w, 3), 21, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def render(d):
    """Two panels: the whole frame for context, and a zoom on the instance.

    ONE PANEL IS NOT ENOUGH, and the contact sheet is what showed it. A 12 px gate in a
    640 px frame is simply not judgeable however the slide is scaled, so the context view
    alone would have collected verdicts that meant nothing; a zoom alone would hide which
    gate is being judged when several are in shot. Both, side by side.

    The margin round the frame is adaptive because an amodal quad round a 700 px gate lies
    far outside the image, and a fixed margin would crop away the thing being judged.
    """
    img = cv2.imread(d['path'])
    if img is None:
        return None
    h, w = img.shape[:2]
    q = np.asarray(d['corners'], np.float32)
    need = max(0.0, -q[:, 0].min(), -q[:, 1].min(),
               q[:, 0].max() - w, q[:, 1].max() - h)
    # Capped at 200: letting the margin follow a 900 px amodal quad shrank the frame in
    # the context panel to a thumbnail, which is the one thing that panel is for. The quad
    # simply runs off the context view when it is that big; the zoom panel still shows all
    # of it, because the zoom pads the canvas further on demand.
    m = int(min(200, max(40, need + 30)))

    canvas = np.full((h + 2 * m, w + 2 * m, 3), 34, np.uint8)
    canvas[m:m + h, m:m + w] = img
    cv2.rectangle(canvas, (m, m), (m + w - 1, m + h - 1), (110, 110, 110), 1)

    def overlay(im, off, thick, rad):
        p = (q + off).astype(np.int32)
        cv2.polylines(im, [p.reshape(-1, 1, 2)], True, (60, 240, 60), thick, cv2.LINE_AA)
        for j, pt in enumerate(p):
            cv2.circle(im, tuple(pt), rad, (255, 0, 255) if j == 0 else (60, 240, 60), -1)
            cv2.circle(im, tuple(pt), rad, (0, 0, 0), 1)

    ctx = canvas.copy()
    overlay(ctx, m, 2, 5)
    left = _fit(ctx, PANEL, PANEL)

    # zoom: the amodal quad plus 40% context, floored so a tiny gate still fills the panel
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    cx, cy = (x0 + x1) / 2.0 + m, (y0 + y1) / 2.0 + m
    side = max(max(x1 - x0, y1 - y0) * 1.9, 70.0)
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = int(round(side))
    pad = max(0, -a, -b, a + side - canvas.shape[1], b + side - canvas.shape[0])
    if pad:
        canvas = cv2.copyMakeBorder(canvas, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=(21, 21, 21))
        a, b = a + pad, b + pad
        zoff = m + pad
    else:
        zoff = m
    zq = canvas[b:b + side, a:a + side].copy()
    # overlay in the zoom's own coordinates
    p = (q + zoff - np.array([a, b], np.float32)).astype(np.int32)
    zs = PANEL / float(max(1, side))
    zq = cv2.resize(zq, (PANEL, PANEL),
                    interpolation=cv2.INTER_AREA if zs < 1 else cv2.INTER_NEAREST)
    p = (p * zs).astype(np.int32)
    cv2.polylines(zq, [p.reshape(-1, 1, 2)], True, (60, 240, 60), 2, cv2.LINE_AA)
    for j, pt in enumerate(p):
        cv2.circle(zq, tuple(pt), 6, (255, 0, 255) if j == 0 else (60, 240, 60), -1)
        cv2.circle(zq, tuple(pt), 6, (0, 0, 0), 1)

    slide = np.full((PANEL + 26, PANEL * 2 + 6, 3), 21, np.uint8)
    slide[26:, :PANEL] = left
    slide[26:, PANEL + 6:] = zq
    cv2.putText(slide, 'whole frame', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1)
    cv2.putText(slide, 'zoom  (x%.1f)' % zs, (PANEL + 14, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return slide


# -------------------------------------------------------------------------------- ui

HTML = r"""<meta charset="utf-8">
<title>gatenet label triage</title>
<!--
  Auto-label triage. One crop, one verdict, next crop.
  Y = the green quad is on the gate aperture.   N = it is not.   S = cannot tell.
  Built by triage_build.py; the slide order is fixed and the strata are deliberately
  not shown, because the pass rate of the hidden control is the baseline the other
  groups are read against. Verdicts autosave to localStorage on every keypress.
-->
<style>
 html,body{margin:0;height:100%;background:#15171c;color:#e6e8ee;
   font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",sans-serif;overflow:hidden}
 body{display:flex;flex-direction:column;align-items:center}
 #bar{width:100%;display:flex;gap:16px;align-items:center;padding:7px 12px;
   background:#1e2129;border-bottom:1px solid #333844;flex:0 0 auto}
 #bar b{color:#ffa53b}
 .sp{flex:1}
 button{background:#2a2f3a;color:#e6e8ee;border:1px solid #333844;border-radius:4px;
   padding:5px 11px;cursor:pointer;font:inherit}
 button:hover{background:#39404e}
 #wrap{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center;
   width:100%;overflow:hidden}
 img{max-width:100%;max-height:100%;object-fit:contain;image-rendering:auto}
 #foot{flex:0 0 auto;padding:5px 12px;color:#98a0b0}
 kbd{background:#2a2f3a;border:1px solid #333844;border-bottom-width:2px;border-radius:3px;
   padding:1px 6px;font:12px ui-monospace,Consolas,monospace}
 .y{color:#4be36a}.n{color:#ff5d5d}.s{color:#e765ff}
</style>
<div id="bar">
  <span><b id="i">0</b> / <span id="n">0</span></span>
  <span>done <b id="done">0</b></span>
  <span class="y">Y <b id="cy">0</b></span>
  <span class="n">N <b id="cn">0</b></span>
  <span class="s">S <b id="cs">0</b></span>
  <span id="verdict"></span>
  <span class="sp"></span>
  <button id="prev">&larr; back</button>
  <button id="next">skip &rarr;</button>
  <button id="exp">Export JSON (E)</button>
</div>
<div id="wrap"><img id="im"></div>
<div id="foot">
  <kbd>Y</kbd> quad is on the aperture &nbsp; <kbd>N</kbd> it is not &nbsp;
  <kbd>S</kbd> cannot tell &nbsp; <kbd>&larr;</kbd><kbd>&rarr;</kbd> move &nbsp;
  <kbd>E</kbd> export. Magenta dot = corner 0. Judge the GREEN QUAD against the gate's
  inner aperture; corners outside the image are meant to be there.
</div>
<script>
"use strict";
var SLIDES = __SLIDES__;
var LS = 'vqual2.triage.v1';
var v = {}, i = 0;
try { var r = localStorage.getItem(LS); if (r) { var d = JSON.parse(r);
      v = d.v || {}; i = Math.min(d.i || 0, SLIDES.length - 1); } } catch (e) {}

function save(){ try { localStorage.setItem(LS, JSON.stringify({v:v, i:i, ts:Date.now()})); }
                 catch (e) {} }
function counts(){ var c={y:0,n:0,s:0}; for (var k in v) if (c[v[k]]!==undefined) c[v[k]]++;
                   return c; }
function show(){
  var s = SLIDES[i];
  document.getElementById('im').src = 'crops/' + s.img;
  document.getElementById('i').textContent = i + 1;
  document.getElementById('n').textContent = SLIDES.length;
  var c = counts();
  document.getElementById('cy').textContent = c.y;
  document.getElementById('cn').textContent = c.n;
  document.getElementById('cs').textContent = c.s;
  document.getElementById('done').textContent = c.y + c.n + c.s;
  var cur = v[s.id];
  var e = document.getElementById('verdict');
  e.textContent = cur ? ('this one: ' + cur.toUpperCase()) : '';
  e.className = cur || '';
  // prefetch the next few so the slideshow never stalls on a JPEG decode
  for (var k = 1; k <= 3; k++) if (SLIDES[i+k]) { var p = new Image();
    p.src = 'crops/' + SLIDES[i+k].img; }
}
function set(x){ v[SLIDES[i].id] = x; save();
                 if (i < SLIDES.length - 1) i++; save(); show(); }
function go(d){ i = Math.max(0, Math.min(SLIDES.length - 1, i + d)); save(); show(); }
function exp(){
  var out = {};
  for (var j = 0; j < SLIDES.length; j++){ var s = SLIDES[j];
    if (v[s.id]) out[s.id] = v[s.id]; }
  var b = new Blob([JSON.stringify(out, null, 1)], {type:'application/json'});
  var u = URL.createObjectURL(b), a = document.createElement('a');
  a.href = u; a.download = 'triage_verdicts.json';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(function(){ URL.revokeObjectURL(u); }, 4000);
}
document.addEventListener('keydown', function(ev){
  var k = ev.key.toLowerCase();
  if (k === 'y') set('y');
  else if (k === 'n') set('n');
  else if (k === 's') set('s');
  else if (ev.key === 'ArrowRight' || k === ' ') go(1);
  else if (ev.key === 'ArrowLeft') go(-1);
  else if (k === 'e') exp();
  else return;
  ev.preventDefault();
});
document.getElementById('prev').addEventListener('click', function(){ go(-1); });
document.getElementById('next').addEventListener('click', function(){ go(1); });
document.getElementById('exp').addEventListener('click', exp);
window.addEventListener('beforeunload', save);
show();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'triage'))
    ap.add_argument('--split', default='block')
    ap.add_argument('--calibrate', action='store_true',
                    help='report how many instances each separation setting yields, '
                         'then exit -- the basis for MIN_GAP / NEAR_PX')
    args = ap.parse_args()

    rng = random.Random(SEED)
    items = load_items()
    _, va = G.split_index(items, args.split)
    print(f'{len(items)} instances, {len(va)} held out on split={args.split}')

    if args.calibrate:
        pools = {
            'clipped <120': [d for d in va if d['clipped'] and d['size_px'] < 120],
            '>=120 px': [d for d in va if d['size_px'] >= 120],
            'control 15-60 unclipped': [d for d in va if not d['clipped']
                                        and 15 <= d['size_px'] < 60],
        }
        print('%-26s %6s' % ('pool', 'all'), end='')
        settings = [(45, 70.), (30, 60.), (15, 40.), (10, 30.), (5, 20.)]
        for g, n in settings:
            print('  %3df/%2.0fpx' % (g, n), end='')
        print()
        for nm, pool in pools.items():
            print('%-26s %6d' % (nm, len(pool)), end='')
            for g, n in settings:
                got = pick(pool, 10 ** 6, [], random.Random(SEED), g, n)
                print('  %9d' % len(got), end='')
            print()
        return

    sz = np.array([d['size_px'] for d in va])
    cl = np.array([d['clipped'] for d in va])
    print('  clipped <120 px: %d   >=120 px: %d (unclipped >=120: %d)   unclipped 15-60: %d'
          % (int((cl & (sz < 120)).sum()), int((sz >= 120).sum()),
             int(((~cl) & (sz >= 120)).sum()), int(((~cl) & (sz >= 15) & (sz < 60)).sum())))

    # SAMPLED FROM THE FULL INDEX, not just the held-out side. Whether a projected label
    # sits on the aperture is a property of the label, not of which split it fell in, and
    # the held-out 20% simply does not contain 150 independent clipped views (29 at the
    # separation used here). Each pick records which side it came from, so the held-out
    # subset can still be read against TRAINING.md's tables on its own.
    val_ids = {id(d) for d in va}
    pool = items

    taken, chosen = [], []
    # A: clipped under 120 px, spread over the size bands rather than let the small
    # band dominate by sheer count
    bands = [(lo, hi) for lo, hi in G.SIZE_BANDS if lo < 120]
    per = N_CLIPPED // len(bands)
    for lo, hi in bands:
        c = [d for d in pool if d['clipped'] and lo <= d['size_px'] < hi]
        got = pick(c, per, taken, rng)
        chosen += [(d, 'clipped') for d in got]
        print(f'  A clipped {lo:>3.0f}-{hi:<6.0f}: {len(got):3d} of {len(c)} available')
    short = N_CLIPPED - sum(1 for _, g in chosen if g == 'clipped')
    if short > 0:
        ids = {id(t) for t in taken}
        c = [d for d in pool if d['clipped'] and d['size_px'] < 120 and id(d) not in ids]
        got = pick(c, short, taken, rng)
        chosen += [(d, 'clipped') for d in got]
        print(f'  A top-up (bands that ran dry, refilled from any size <120): {len(got)}')

    got = pick([d for d in pool if d['size_px'] >= 120], N_LARGE, taken, rng)
    chosen += [(d, 'large') for d in got]
    print(f'  B >=120 px: {len(got)}')

    got = pick([d for d in pool if not d['clipped'] and 15 <= d['size_px'] < 60],
               N_CONTROL, taken, rng)
    chosen += [(d, 'control') for d in got]
    print(f'  C control (unclipped 15-60 px): {len(got)}')

    rng.shuffle(chosen)

    crops = os.path.join(args.out, 'crops')
    os.makedirs(crops, exist_ok=True)
    slides, manifest, bad = [], [], 0
    for n, (d, group) in enumerate(chosen):
        im = render(d)
        if im is None:
            bad += 1
            continue
        fn = '%03d.jpg' % len(slides)
        cv2.imwrite(os.path.join(crops, fn), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sid = '%s#%d' % (d['key'], d['inst'])
        slides.append({'img': fn, 'id': sid})
        manifest.append({'img': fn, 'id': sid, 'key': d['key'], 'inst': d['inst'],
                         'group': group, 'size_px': round(d['size_px'], 1),
                         'clipped': d['clipped'], 'occluded': d['occluded'],
                         'session': d['session'], 'ordinal': d['ordinal'],
                         'split': 'val' if id(d) in val_ids else 'train',
                         'auto': d['auto']})

    with open(os.path.join(args.out, 'triage.html'), 'w') as fh:
        fh.write(HTML.replace('__SLIDES__', json.dumps(slides)))
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump({'split': args.split, 'seed': SEED, 'min_gap': MIN_GAP,
                   'near_px': NEAR_PX, 'items': manifest}, fh, indent=1)

    print(f'\nwrote {len(slides)} slides to {args.out}  ({bad} unreadable frames)')
    print(f'  open  {os.path.join(args.out, "triage.html")}')


if __name__ == '__main__':
    main()
