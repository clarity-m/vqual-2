"""gatenet.py -- CNN corner regression for the gate INNER 1500 mm aperture.

WHY A NET AT ALL. `detect.py` (colour -> contour -> quad) has four failure modes that
more pixels cannot fix, all of them recorded in `NOTES.md`:

  1. decoration false positives (orange things that are not gates),
  2. clipped gates vanish -- the aperture stops being an enclosed contour at the image
     edge, and 46% of labelled instances are clipped,
  3. merged gates -- two gates on the view axis form one blob and the outer-boundary
     fallback measures the pair as one gate,
  4. the cyan guidance ribbon is drawn OVER the aperture edge, so the contour follows the
     ribbon silhouette (one measured quad had a side pulled ~30% short).

(2), (3) and (4) are AMODAL problems: the right answer is where the edge WOULD be, which
a contour follower cannot express and a regressor can. So this file regresses the four
aperture corners directly, and the corners are allowed to land outside the crop.

SCOPE. Corner regression only -- no real/not-real confidence head. VQ1 gate truth covers
5 of the 6 race gates, so frames showing race gate 5 contain an UNLABELLED real gate. That
is a poison pill for a confidence head (it would be taught that a real gate is background)
but harmless for a regressor trained on positive crops, which never looks at gate 5.

LABELS. `autolabels_vq1.json` from `autolabel.py`: corners are the inner aperture in
original 640x360 px, AMODAL (never clamped), wound clockwise on screen starting at the
min(x+y) corner. That winding is a deterministic function of the four points, which is
exactly what makes corner regression well posed -- see the docstrings of `autolabel.py`
and `labelui.html`. Nothing here re-derives it; `canon()` is imported from `autolabel`.

THE CROP MUST BE DERIVABLE AT INFERENCE TIME. A crop centred on the amodal bounding box
would be cheating: at race time nothing knows where an off-screen corner is. So the crop
is built from the VISIBLE part of the quad (quad ^ image rect) -- the same information a
detector or a tracker's previous-frame box would supply -- and the amodal target is then
expressed in that crop's frame, freely exceeding it. This is the whole trick: the input
box is modal, the output is amodal.

COORDINATE FRAME. With crop centre (cx, cy) and side S (original px):

    u = 2 * (x - cx) / S,   v = 2 * (y - cy) / S

so the crop interior is [-1, 1]^2 and a corner outside the crop simply has |u| > 1. The
net's output layer is linear with no squashing, so points outside are representable --
which a sigmoid- or heatmap-based head could not do without extra machinery.

LOSS is smooth-L1 in those normalised units, i.e. SCALE INVARIANT: a 100 px gate and a
20 px gate contribute equally. That is deliberate. Absolute-pixel loss would be dominated
by the handful of near gates (apparent size runs to 1774 px), and what solvePnP actually
needs is the corner positions relative to the gate's own size. Pixel error is still what
gets REPORTED, broken down by size band, because that is the number to compare against
the detector.

NO HORIZONTAL FLIP, NO ROTATION augmentation. The winding convention is defined in IMAGE
space -- index 0 is the min(x+y) corner -- so any transform that moves corners around must
re-canonicalise the target, and a silent bug there trains the net on two different orders
for one picture. Flip is implemented (`--hflip`) with the re-canonicalisation done, but is
off by default: this run is unattended and the correctness risk outweighs the diversity
gain on 13k instances. Photometric augmentation and crop jitter/scale carry the load.

SPLIT BY TIME, NEVER RANDOMLY. Frames are 30 Hz video and adjacent frames are near
duplicates, so a random split leaks and reports a fantasy. Two splits are supported:

  --split block    contiguous blocks of BLOCK frames per session, every 5th block held
                   out (~20%), plus a GUARD band of frames adjacent to a held-out block
                   dropped from TRAIN so that no training frame is within GUARD/30 s of
                   any validation frame.
  --split session  train on 20260731-195307, validate on 20260731-204841-vq1-lap-slow.
                   Harsher and more honest: a different flight, flown differently.

USAGE
    python3 pilot/perception/gatenet.py --mode train --split block   --tag block
    python3 pilot/perception/gatenet.py --mode train --split session --tag session
    python3 pilot/perception/gatenet.py --mode train --resume --tag block
    python3 pilot/perception/gatenet.py --mode eval  --tag block --ckpt best
    python3 pilot/perception/gatenet.py --mode bench --tag block
    python3 pilot/perception/gatenet.py --mode sanity --tag block   # crop/target overlays

OVERNIGHT SAFETY: checkpoints (model + optimizer + scaler + epoch + rng) every epoch,
--resume, a CSV metrics line appended every epoch, --max-hours (default 8) with a clean
exit and a written report, CUDA OOM caught and the batch size halved rather than dying,
and nothing that can ever prompt for input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
SESSIONS = os.path.join(ROOT, 'pilot', 'sessions')
LABELS = os.path.join(HERE, 'autolabels_vq1.json')
RUNS = os.path.join(HERE, 'gatenet_runs')
REPORT = os.path.join(HERE, 'TRAINING.md')

sys.path.insert(0, HERE)
from autolabel import canon  # noqa: E402  (single source of the winding convention)

W, H = 640, 360
RES = 160                # net input side (--res). Corner precision on LARGE gates is
                         # resolution-limited, not capacity-limited: a 300 px gate gets a
                         # ~480 px crop, so one crop pixel is 480/RES original pixels and
                         # that is a hard floor on its error. 160 buys 1.25x over 128 for
                         # 1.6x the multiplies and still runs in single-digit ms on CPU,
                         # which is the 30-Hz-alongside-control constraint.
MARGIN = 1.60            # crop side / visible-bbox side, at eval (deterministic)
MIN_VIS_PX = 12.0        # floor on the visible bbox side, so a sliver at the edge still
                         # gets a crop with some context in it
BLOCK = 240              # frames per time block (8 s at 30 Hz)
VAL_EVERY = 5            # 1 block in 5 held out -> ~20% val
GUARD = 45               # 1.5 s of frames either side of a val block, dropped from train
SESSION_TRAIN = '20260731-195307'
SESSION_VAL = '20260731-204841-vq1-lap-slow'
SIZE_BANDS = ((0, 15), (15, 30), (30, 60), (60, 120), (120, 1e9))


# ---------------------------------------------------------------------------------------
# data


def load_index():
    """-> list of instance dicts, one per labelled gate, ordered by (session, frame)."""
    with open(LABELS) as fh:
        raw = json.load(fh)
    keys = sorted(raw)                       # '<session>/<file>'; filenames are numeric
    # frame ordinal WITHIN each session, from the sorted filename order == time order
    ordinal, seen = {}, {}
    for k in keys:
        sess = k.split('/', 1)[0]
        n = seen.get(sess, 0)
        ordinal[k] = n
        seen[sess] = n + 1

    out = []
    for k in keys:
        sess, fname = k.split('/', 1)
        for inst in raw[k]:
            c = np.asarray(inst['corners'], np.float32)
            edges = [float(np.linalg.norm(c[(j + 1) % 4] - c[j])) for j in range(4)]
            out.append({
                'path': os.path.join(SESSIONS, sess, 'frames', fname),
                'session': sess,
                'ordinal': ordinal[k],
                'corners': c,
                'clipped': bool(inst['clipped']),
                'occluded': bool(inst['occluded']),
                'size_px': max(edges),
            })
    return out


def split_index(items, how):
    """-> (train, val). Time-contiguous by construction; see the module docstring."""
    if how == 'session':
        tr = [d for d in items if d['session'] == SESSION_TRAIN]
        va = [d for d in items if d['session'] == SESSION_VAL]
        return tr, va

    tr, va = [], []
    for d in items:
        blk = d['ordinal'] // BLOCK
        if blk % VAL_EVERY == VAL_EVERY - 2:
            va.append(d)
            continue
        # distance in frames to the nearest held-out block; drop the guard band from train
        pos = d['ordinal'] % BLOCK
        prev_is_val = (blk - 1) % VAL_EVERY == VAL_EVERY - 2
        next_is_val = (blk + 1) % VAL_EVERY == VAL_EVERY - 2
        if (prev_is_val and pos < GUARD) or (next_is_val and pos >= BLOCK - GUARD):
            continue
        tr.append(d)
    return tr, va


def visible_box(corners):
    """Bounding box of quad ^ image rect -- the only part a detector could have seen."""
    rect = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    area, inter = cv2.intersectConvexConvex(np.asarray(corners, np.float32), rect)
    if inter is None or len(inter) < 3:
        p = np.asarray(corners, np.float32)      # degenerate; fall back to the clamp
        x0, y0 = np.clip(p.min(0), [0, 0], [W, H])
        x1, y1 = np.clip(p.max(0), [0, 0], [W, H])
    else:
        p = inter.reshape(-1, 2)
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
    return float(x0), float(y0), float(x1), float(y1)


def crop_params(corners, rng=None):
    """(cx, cy, S) for the square crop, in original px. `rng` -> training jitter."""
    x0, y0, x1, y1 = visible_box(corners)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = max(x1 - x0, y1 - y0, MIN_VIS_PX)
    S = s * MARGIN
    if rng is not None:
        # scale jitter is log-uniform: a detector's box is wrong by a RATIO, not by px
        S *= math.exp(rng.uniform(math.log(0.78), math.log(1.55)))
        cx += rng.uniform(-0.16, 0.16) * S
        cy += rng.uniform(-0.16, 0.16) * S
    return cx, cy, S


def make_crop(img, cx, cy, S, res=None):
    """Similarity warp of the crop square to res x res. Out-of-image area -> black,
    which is what the net will see at race time too (nothing exists beyond the sensor)."""
    res = RES if res is None else res
    a = res / S
    M = np.array([[a, 0.0, res / 2.0 - cx * a],
                  [0.0, a, res / 2.0 - cy * a]], np.float32)
    return cv2.warpAffine(img, M, (res, res), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def to_norm(corners, cx, cy, S):
    p = np.asarray(corners, np.float32)
    return np.stack([2.0 * (p[:, 0] - cx) / S, 2.0 * (p[:, 1] - cy) / S], 1)


def from_norm(uv, cx, cy, S):
    uv = np.asarray(uv, np.float32).reshape(4, 2)
    return np.stack([uv[:, 0] * S / 2.0 + cx, uv[:, 1] * S / 2.0 + cy], 1)


def photometric(img, rng):
    """Brightness / contrast / gamma / gaussian noise / mild blur.

    The hangar is dark (66.5% of a VQ2 frame is V<40) and the gates glow, so exposure and
    noise are the realistic nuisance axes; geometry is handled by the crop jitter."""
    x = img.astype(np.float32)
    if rng.random() < 0.9:
        x = x * rng.uniform(0.55, 1.55) + rng.uniform(-28, 28)
    if rng.random() < 0.5:
        g = rng.uniform(0.65, 1.5)
        x = 255.0 * np.power(np.clip(x, 0, 255) / 255.0, g)
    if rng.random() < 0.5:
        x = x + np.random.normal(0.0, rng.uniform(2.0, 10.0), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        k = rng.choice([3, 5])
        x = cv2.GaussianBlur(x, (k, k), 0)
    return x


class GateCrops(Dataset):
    def __init__(self, items, train, hflip=False, res=None):
        self.items, self.train, self.hflip = items, train, hflip
        self.res = RES if res is None else res

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        d = self.items[i]
        img = cv2.imread(d['path'], cv2.IMREAD_COLOR)
        if img is None:                              # never die on a bad frame overnight
            img = np.zeros((H, W, 3), np.uint8)
        corners = d['corners']
        rng = random.Random((i * 2654435761 + int(time.time() * 1e6)) & 0xFFFFFFFF) \
            if self.train else None
        cx, cy, S = crop_params(corners, rng)
        crop = make_crop(img, cx, cy, S, self.res)
        tgt = to_norm(corners, cx, cy, S)
        if self.train:
            crop = photometric(crop, rng)
            if self.hflip and rng.random() < 0.5:
                crop = crop[:, ::-1].copy()
                # mirror in the CROP frame, then re-canonicalise: the winding rule is an
                # image-space function of the four points, so flipping without re-running
                # canon() would hand the net two orders for one picture.
                tgt = np.stack([-tgt[:, 0], tgt[:, 1]], 1)
                tgt = canon(tgt).astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        x = x.div_(255.0).sub_(0.45).div_(0.25)
        return x, torch.from_numpy(tgt.reshape(8).astype(np.float32)), \
            torch.tensor([cx, cy, S], dtype=torch.float32), i


# ---------------------------------------------------------------------------------------
# model


class DWSep(nn.Module):
    """Depthwise-separable conv. Chosen over plain 3x3 because the target is 30 Hz
    alongside a control loop: same receptive field, ~8x fewer multiplies."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.silu(self.bn1(self.dw(x)), inplace=True)
        return F.silu(self.bn2(self.pw(x)), inplace=True)


class GateNet(nn.Module):
    """Five stride-2 stages (RES/32 spatial), pooled to 4x4, then a FLATTENED head.

    The head deliberately does not global-average-pool to 1x1. Pooling to a point throws
    away exactly the information being regressed (where things are); a 4x4 spatial code
    costs ~0.7 M params, which is affordable at this size. The adaptive pool makes the
    head independent of --res, so resolution is a free parameter."""

    def __init__(self, width=1.0):
        super().__init__()
        c = [max(8, int(round(w * width))) for w in (24, 48, 96, 128, 160)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c[0], 3, 2, 1, bias=False), nn.BatchNorm2d(c[0]), nn.SiLU(True))
        self.body = nn.Sequential(
            DWSep(c[0], c[1], 2), DWSep(c[1], c[1], 1),
            DWSep(c[1], c[2], 2), DWSep(c[2], c[2], 1),
            DWSep(c[2], c[3], 2), DWSep(c[3], c[3], 1),
            DWSep(c[3], c[4], 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(c[4] * 4 * 4, 256), nn.SiLU(True), nn.Linear(256, 8))
        # start near the identity-ish prior: a gate filling the crop. Speeds the first
        # epochs and keeps early predictions inside a sane range.
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor(
                [-0.6, -0.6, 0.6, -0.6, 0.6, 0.6, -0.6, 0.6]))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# ---------------------------------------------------------------------------------------
# metrics


def corner_px_errors(pred, tgt, geo):
    """Per-corner euclidean error in ORIGINAL image pixels. pred/tgt (N,8), geo (N,3)."""
    S = geo[:, 2:3]
    p = pred.reshape(-1, 4, 2) * (S[:, :, None] / 2.0)
    t = tgt.reshape(-1, 4, 2) * (S[:, :, None] / 2.0)
    return torch.linalg.norm(p - t, dim=2)               # (N, 4)


def centre_px_error(pred, tgt, geo):
    S = geo[:, 2:3]
    p = pred.reshape(-1, 4, 2).mean(1) * (S / 2.0)
    t = tgt.reshape(-1, 4, 2).mean(1) * (S / 2.0)
    return torch.linalg.norm(p - t, dim=1)               # (N,)


@torch.no_grad()
def evaluate(model, loader, dev, items=None):
    model.eval()
    ce, cn, idx, loss_sum, n = [], [], [], 0.0, 0
    for x, y, g, i in loader:
        x, y, g = x.to(dev, non_blocking=True), y.to(dev), g.to(dev)
        p = model(x)
        loss_sum += float(F.smooth_l1_loss(p, y, beta=0.05, reduction='sum'))
        n += y.numel()
        ce.append(corner_px_errors(p, y, g).cpu())
        cn.append(centre_px_error(p, y, g).cpu())
        idx.append(i)
    ce = torch.cat(ce).numpy()
    cn = torch.cat(cn).numpy()
    idx = torch.cat(idx).numpy()
    return {'loss': loss_sum / max(n, 1), 'corner': ce, 'centre': cn, 'idx': idx}


def _q(a, p):
    return float(np.percentile(a, p)) if len(a) else float('nan')


def breakdown(res, items):
    """Median / p90 corner + centre error, by size band and clipped vs not."""
    ce, cn, idx = res['corner'], res['centre'], res['idx']
    per_inst = ce.mean(1)
    size = np.array([items[i]['size_px'] for i in idx])
    clip = np.array([items[i]['clipped'] for i in idx])
    occl = np.array([items[i]['occluded'] for i in idx])
    rows = []

    def add(name, m):
        if m.sum() == 0:
            return
        # RELATIVE error as well as pixels. Absolute px flatters small gates and punishes
        # big ones for the same geometric quality: a 300 px gate can absorb far more
        # pixels of corner error than a 20 px one before solvePnP's pose moves. Neither
        # number alone is the honest one, so both are reported.
        rel = per_inst[m] / np.maximum(size[m], 1e-6)
        rows.append({'group': name, 'n': int(m.sum()),
                     'corner_med': _q(ce[m].ravel(), 50), 'corner_p90': _q(ce[m].ravel(), 90),
                     'inst_med': _q(per_inst[m], 50), 'inst_p90': _q(per_inst[m], 90),
                     'centre_med': _q(cn[m], 50), 'centre_p90': _q(cn[m], 90),
                     'rel_med': _q(rel, 50), 'rel_p90': _q(rel, 90)})

    add('ALL', np.ones(len(size), bool))
    for lo, hi in SIZE_BANDS:
        add(f'size {lo:.0f}-{hi:.0f} px' if hi < 1e8 else f'size >={lo:.0f} px',
            (size >= lo) & (size < hi))
    add('clipped', clip)
    add('unclipped', ~clip)
    add('occluded (gate-on-gate)', occl)
    # the comparable-to-detector subset: detect.py only ever reports these
    add('detector-comparable (unclipped, unoccluded, >=15 px)',
        (~clip) & (~occl) & (size >= 15))
    return rows


def fmt_rows(rows):
    out = ['| group | n | corner med | corner p90 | centre med | centre p90 | rel med |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        out.append(f"| {r['group']} | {r['n']} | {r['corner_med']:.2f} | "
                   f"{r['corner_p90']:.2f} | {r['centre_med']:.2f} | "
                   f"{r['centre_p90']:.2f} | {100*r.get('rel_med', float('nan')):.1f}% |")
    return '\n'.join(out)


# ---------------------------------------------------------------------------------------
# train


def make_loaders(tr_items, va_items, bs, workers, hflip):
    tr = DataLoader(GateCrops(tr_items, True, hflip), batch_size=bs, shuffle=True,
                    num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    va = DataLoader(GateCrops(va_items, False), batch_size=max(bs, 128), shuffle=False,
                    num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    return tr, va


def log_line(path, s):
    with open(path, 'a') as fh:
        fh.write(s + '\n')
        fh.flush()


def train(args):
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    outdir = os.path.join(RUNS, args.tag)
    os.makedirs(outdir, exist_ok=True)
    logf = os.path.join(outdir, 'log.csv')

    items = load_index()
    tr_items, va_items = split_index(items, args.split)
    print(f'[{args.tag}] split={args.split}  train={len(tr_items)}  val={len(va_items)} '
          f'instances (of {len(items)})', flush=True)

    model = GateNet(args.width).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'[{args.tag}] parameters: {nparam:,}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.02)
    start, best = 0, float('inf')
    bs = args.batch

    last_ck = os.path.join(outdir, 'last.pt')
    if args.resume and os.path.exists(last_ck):
        ck = torch.load(last_ck, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start, best, bs = ck['epoch'] + 1, ck['best'], ck.get('batch', bs)
        print(f'[{args.tag}] resumed from epoch {start}, best={best:.3f} px', flush=True)
        # RESUMING WITH A DIFFERENT --epochs RESHAPES THE COSINE, on purpose. Epoch time
        # here is set by JPEG decode on a thermally throttled 4-core laptop and is only
        # knowable once the run is under way, so the horizon that fits inside --max-hours
        # is not knowable when the run starts. Keeping the original T_max would stop the
        # run at high LR with the anneal never applied, which UNDERSTATES the model --
        # a false negative, the one error this exercise must not make.
        if args.epochs != sched.T_max:
            for g in opt.param_groups:
                g['initial_lr'] = args.lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=args.lr * 0.02, last_epoch=start - 1)
            print(f'[{args.tag}] cosine reshaped to T_max={args.epochs}, '
                  f'lr now {opt.param_groups[0]["lr"]:.3g}', flush=True)

    if not os.path.exists(logf):
        log_line(logf, 'epoch,train_loss,val_loss,val_corner_med_px,val_corner_p90_px,'
                       'val_centre_med_px,lr,batch,secs,elapsed_h')

    tr_loader, va_loader = make_loaders(tr_items, va_items, bs, args.workers, args.hflip)
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600.0
    stop_reason = 'completed all epochs'

    for ep in range(start, args.epochs):
        model.train()
        ep_t = time.time()
        tot, cnt = 0.0, 0
        it = iter(tr_loader)
        while True:
            try:
                batch = next(it)
            except StopIteration:
                break
            except Exception as e:                       # a corrupt JPEG must not end the night
                print(f'[{args.tag}] loader error, skipping batch: {e}', flush=True)
                continue
            x, y, g, _ = batch
            try:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
                    loss = F.smooth_l1_loss(model(x), y, beta=0.05)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                tot += float(loss.detach()) * len(y)
                cnt += len(y)
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower() or bs <= 4:
                    raise
                bs = max(4, bs // 2)
                print(f'[{args.tag}] CUDA OOM -> batch {bs}, continuing', flush=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                del it
                tr_loader, va_loader = make_loaders(tr_items, va_items, bs,
                                                    args.workers, args.hflip)
                it = iter(tr_loader)
                continue
            if time.time() > deadline:
                break

        try:
            res = evaluate(model, va_loader, dev, va_items)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            res = evaluate(model, DataLoader(GateCrops(va_items, False), batch_size=32,
                                             num_workers=0), dev, va_items)
        med = _q(res['corner'].ravel(), 50)
        p90 = _q(res['corner'].ravel(), 90)
        cmed = _q(res['centre'], 50)
        sched.step()
        secs = time.time() - ep_t
        elapsed_h = (time.time() - t0) / 3600.0
        trl = tot / max(cnt, 1)
        log_line(logf, f'{ep},{trl:.6f},{res["loss"]:.6f},{med:.3f},{p90:.3f},{cmed:.3f},'
                       f'{opt.param_groups[0]["lr"]:.6g},{bs},{secs:.1f},{elapsed_h:.3f}')
        print(f'[{args.tag}] ep {ep:3d}  train {trl:.5f}  val {res["loss"]:.5f}  '
              f'corner med {med:.2f} px p90 {p90:.2f}  centre {cmed:.2f}  {secs:.0f}s',
              flush=True)

        ck = {'model': model.state_dict(), 'opt': opt.state_dict(),
              'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
              'epoch': ep, 'best': best, 'batch': bs, 'args': vars(args),
              'val_corner_med': med, 'nparam': nparam}
        torch.save(ck, last_ck + '.tmp')
        os.replace(last_ck + '.tmp', last_ck)
        if med < best:
            best = med
            ck['best'] = best
            torch.save(ck, os.path.join(outdir, 'best.pt.tmp'))
            os.replace(os.path.join(outdir, 'best.pt.tmp'), os.path.join(outdir, 'best.pt'))

        if time.time() > deadline:
            stop_reason = f'hit --max-hours {args.max_hours}'
            break

    # final report from the BEST checkpoint
    bp = os.path.join(outdir, 'best.pt')
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location=dev,
                                         weights_only=False)['model'])
    res = evaluate(model, va_loader, dev, va_items)
    rows = breakdown(res, va_items)
    bench_txt = bench_str(model, dev)
    body = [
        f'\n## Run `{args.tag}` -- split `{args.split}`',
        '',
        f'* {stop_reason}; {(time.time()-t0)/3600.0:.2f} h, {ep+1-start} epochs this invocation',
        f'* parameters: {nparam:,}   input {RES}x{RES}   batch {bs}   crop margin {MARGIN}',
        f'* train {len(tr_items)} / val {len(va_items)} instances '
        f'({len(items)} labelled total)',
        f'* best val median corner error: **{best:.2f} px**',
        f'* {bench_txt}',
        '',
        fmt_rows(rows),
        '',
    ]
    with open(REPORT, 'a') as fh:
        fh.write('\n'.join(body) + '\n')
    print('\n'.join(body), flush=True)
    json.dump(rows, open(os.path.join(outdir, 'breakdown.json'), 'w'), indent=1)
    return rows


# ---------------------------------------------------------------------------------------
# bench / eval / sanity


def bench_str(model, dev):
    parts = []
    model.eval()
    for device, label, batches in ((dev, str(dev), (1, 8)), (torch.device('cpu'), 'cpu', (1,))):
        m = GateNet().to(device)
        m.load_state_dict({k: v.to(device) for k, v in model.state_dict().items()})
        m.eval()
        for b in batches:
            x = torch.randn(b, 3, RES, RES, device=device)
            with torch.no_grad():
                for _ in range(10):
                    m(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(50):
                    m(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t) / 50.0
            parts.append(f'{label} batch{b}: {1000*dt:.2f} ms ({1000*dt/b:.2f} ms/crop)')
    return 'inference -- ' + '; '.join(parts)


def bench(args):
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    m = GateNet(args.width).to(dev)
    ck = os.path.join(RUNS, args.tag, f'{args.ckpt}.pt')
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)['model'])
    n = sum(p.numel() for p in m.parameters())
    print(f'parameters: {n:,}')
    print(bench_str(m, dev))


def eval_only(args):
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    items = load_index()
    tr_items, va_items = split_index(items, args.split)
    m = GateNet(args.width).to(dev)
    m.load_state_dict(torch.load(os.path.join(RUNS, args.tag, f'{args.ckpt}.pt'),
                                 map_location=dev, weights_only=False)['model'])
    _, va = make_loaders(tr_items, va_items, args.batch, args.workers, False)
    res = evaluate(m, va, dev, va_items)
    print(fmt_rows(breakdown(res, va_items)))
    print(bench_str(m, dev))


def sanity(args):
    """Render crops with GT (green) and, if a checkpoint exists, predicted (magenta)
    corners. The one check that catches a frame/winding bug that loss curves cannot."""
    dev = torch.device('cpu')
    items = load_index()
    tr_items, va_items = split_index(items, args.split)
    m = None
    ck = os.path.join(RUNS, args.tag, f'{args.ckpt}.pt')
    if os.path.exists(ck):
        m = GateNet(args.width)
        m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)['model'])
        m.eval()
    sel = [va_items[int(k)] for k in np.linspace(0, len(va_items) - 1, 24)]
    tiles = []
    for d in sel:
        img = cv2.imread(d['path'])
        if img is None:
            continue
        cx, cy, S = crop_params(d['corners'])
        crop = make_crop(img, cx, cy, S)
        vis = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST)
        g = (to_norm(d['corners'], cx, cy, S) + 1.0) * 128.0
        cv2.polylines(vis, [g.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.circle(vis, tuple(g[0].astype(int)), 6, (0, 255, 255), -1)
        if m is not None:
            x = torch.from_numpy(crop.transpose(2, 0, 1)).float().div(255).sub(.45).div(.25)
            with torch.no_grad():
                p = m(x[None]).numpy().reshape(4, 2)
            pp = (p + 1.0) * 128.0
            cv2.polylines(vis, [pp.astype(np.int32).reshape(-1, 1, 2)], True,
                          (255, 0, 255), 1)
        cv2.putText(vis, f'{d["size_px"]:.0f}px{" CLIP" if d["clipped"] else ""}',
                    (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(vis)
    cols = 6
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows_n * 256, cols * 256, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = t
    os.makedirs(os.path.join(RUNS, args.tag), exist_ok=True)
    p = os.path.join(RUNS, args.tag, 'sanity.png')
    cv2.imwrite(p, sheet)
    print(f'wrote {p}  ({len(tiles)} tiles)')


def detector_baseline(args):
    """`detect.py` measured on the SAME held-out instances, so the comparison is made
    rather than quoted.

    Matching rule is `autolabel.verify`'s and the size gate is load-bearing, not a fudge:
    where the detector MISSES a gate the nearest detection is some other gate metres away,
    and scoring that as "error" measures recall, not accuracy. So a detection counts as a
    match only if it is within 2x of the labelled apparent size and within one gate width
    of the labelled centre; everything else is a MISS and is reported as recall.

    Note the asymmetry that favours the detector here: its errors are computed only where
    it fired, while the net is scored on every instance including the ones the detector
    never found."""
    import detect as D
    items = load_index()
    _, va = split_index(items, args.split)
    by_frame = {}
    for k, d in enumerate(va):
        by_frame.setdefault(d['path'], []).append(k)

    n = len(va)
    cn = np.full(n, np.nan)
    ce = np.full((n, 4), np.nan)
    paths = sorted(by_frame)
    for j, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        dets = D.detections(img)
        if not dets:
            continue
        dc = np.array([d['centre'] for d in dets], np.float32)
        ds = np.array([d['size_px'] for d in dets], np.float32)
        for k in by_frame[p]:
            gt = va[k]['corners']
            c = gt.mean(0)
            sz = va[k]['size_px']
            dist = np.linalg.norm(dc - c, axis=1)
            ok = (ds > sz / 2) & (ds < sz * 2) & (dist < max(sz, 12.0))
            if not ok.any():
                continue
            m = int(np.argmin(np.where(ok, dist, 1e9)))
            cn[k] = dist[m]
            ce[k] = np.linalg.norm(canon(dets[m]['quad']) - canon(gt), axis=1)
        if j % 200 == 0:
            print(f'  {j}/{len(paths)} frames', flush=True)

    hit = ~np.isnan(cn)
    size = np.array([d['size_px'] for d in va])
    clip = np.array([d['clipped'] for d in va])
    lines = [f'\n## Detector baseline (`detect.py`) on the same held-out set '
             f'-- split `{args.split}`', '',
             f'* {int(hit.sum())} of {n} labelled instances matched a detection '
             f'-> **recall {hit.mean():.3f}**', '',
             '| group | n | recall | corner med | corner p90 | centre med | centre p90 |',
             '|---|---:|---:|---:|---:|---:|---:|']

    def row(name, m):
        if m.sum() == 0:
            return
        h = m & hit
        if h.sum() == 0:
            lines.append(f'| {name} | {int(m.sum())} | 0.000 | - | - | - | - |')
            return
        lines.append(f'| {name} | {int(m.sum())} | {h.sum()/m.sum():.3f} | '
                     f'{_q(ce[h].ravel(), 50):.2f} | {_q(ce[h].ravel(), 90):.2f} | '
                     f'{_q(cn[h], 50):.2f} | {_q(cn[h], 90):.2f} |')

    row('ALL', np.ones(n, bool))
    for lo, hi in SIZE_BANDS:
        row(f'size {lo:.0f}-{hi:.0f} px' if hi < 1e8 else f'size >={lo:.0f} px',
            (size >= lo) & (size < hi))
    row('clipped', clip)
    row('unclipped', ~clip)
    occl = np.array([d['occluded'] for d in va])
    row('detector-comparable (unclipped, unoccluded, >=15 px)',
        (~clip) & (~occl) & (size >= 15))
    txt = '\n'.join(lines) + '\n'
    with open(REPORT, 'a') as fh:
        fh.write(txt)
    print(txt)


def ribbon(args):
    """Net vs detector, split by how much CYAN GUIDANCE RIBBON crosses the aperture.

    This is failure mode 4 from `NOTES.md`, and none of the other tables isolate it: the
    autolabels' `occluded` flag models gate-on-gate overlap ONLY, so a gate whose edge is
    painted over by the ribbon is recorded as unoccluded. The ribbon matters more than its
    frequency suggests because it marks the racing line -- it passes through the aperture
    of exactly the gate being flown at, and so preferentially corrupts the nearest and most
    important gate.

    Contamination is measured the way NOTES measured it: the cyan class from the HSV table
    (85<=H<=100, S>110, V>110), as a fraction of the GT quad's visible interior. On the one
    frame NOTES analysed by hand that fraction was 17.3% and the contaminated edge came out
    ~30% short, so the >=15% bucket below is the regime that measurement came from."""
    import detect as D
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    items = load_index()
    _, va = split_index(items, args.split)
    model = GateNet(args.width).to(dev)
    model.load_state_dict(torch.load(os.path.join(RUNS, args.tag, f'{args.ckpt}.pt'),
                                     map_location=dev, weights_only=False)['model'])
    model.eval()

    by_frame = {}
    for k, d in enumerate(va):
        by_frame.setdefault(d['path'], []).append(k)
    n = len(va)
    frac = np.zeros(n)
    net_e = np.full(n, np.nan)
    det_e = np.full(n, np.nan)

    for j, p in enumerate(sorted(by_frame)):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cyan = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 100) &
                (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 110))
        dets = D.detections(img)
        dc = np.array([d['centre'] for d in dets], np.float32) if dets else None
        ds = np.array([d['size_px'] for d in dets], np.float32) if dets else None
        crops, keys = [], []
        for k in by_frame[p]:
            gt = va[k]['corners']
            # CONTAMINATION IS MEASURED ON THE EDGE BAND, NOT THE INTERIOR. The failure
            # mode is the ribbon drawn OVER the aperture boundary, so the contour follows
            # the ribbon silhouette. Cyan in the middle of an aperture is harmless to a
            # contour follower -- an interior-area proxy would measure the wrong thing and
            # would also be biased by gate size. So: a ~3 px ring on the quad outline.
            ring = np.zeros(cyan.shape, np.uint8)
            cv2.polylines(ring, [np.round(gt).astype(np.int32).reshape(-1, 1, 2)], True,
                          1, 7)
            a = int(ring.sum())
            frac[k] = float((ring & cyan.astype(np.uint8)).sum()) / a if a else 0.0
            cx, cy, S = crop_params(gt)
            c = make_crop(img, cx, cy, S)
            x = torch.from_numpy(np.ascontiguousarray(c.transpose(2, 0, 1))).float()
            crops.append(x.div_(255.0).sub_(0.45).div_(0.25))
            keys.append((k, cx, cy, S))
            if dets:
                sz = va[k]['size_px']
                dist = np.linalg.norm(dc - gt.mean(0), axis=1)
                ok = (ds > sz / 2) & (ds < sz * 2) & (dist < max(sz, 12.0))
                if ok.any():
                    mi = int(np.argmin(np.where(ok, dist, 1e9)))
                    det_e[k] = float(np.mean(np.linalg.norm(
                        canon(dets[mi]['quad']) - canon(gt), axis=1)))
        with torch.no_grad():
            pr = model(torch.stack(crops).to(dev)).cpu().numpy()
        for (k, cx, cy, S), q in zip(keys, pr):
            net_e[k] = float(np.mean(np.linalg.norm(
                from_norm(q, cx, cy, S) - va[k]['corners'], axis=1)))
        if j % 250 == 0:
            print(f'  {j}/{len(by_frame)} frames', flush=True)

    size = np.array([d['size_px'] for d in va])
    clip = np.array([d['clipped'] for d in va])
    lines = [f'\n## Cyan-ribbon contamination -- split `{args.split}`', '',
             'Cyan fraction of a ~7 px ring on the GT aperture OUTLINE (the edge is what '
             'the ribbon corrupts, not the interior); net error is over ALL instances in '
             'the bucket, detector error only over the ones it found (recall column).',
             '',
             'THE RAW BUCKETS ARE CONFOUNDED and must not be read as an effect: a clipped '
             'or huge gate has most of its outline off-image and lands in the "none" '
             'bucket, which is also where the hardest instances live. The second table '
             'controls for it -- unclipped, 15-60 px only, so size and clipping are held '
             'roughly fixed and only the ribbon varies.', '']

    def table(title, keep):
        lines.extend([f'**{title}**', '',
                      '| ribbon coverage | n | net corner med | net p90 | det recall '
                      '| det corner med |', '|---|---:|---:|---:|---:|---:|'])
        for name, sel in (('none (<2%)', frac < 0.02),
                          ('slight (2-15%)', (frac >= 0.02) & (frac < 0.15)),
                          ('heavy (>=15%)', frac >= 0.15)):
            m = sel & keep
            if m.sum() == 0:
                continue
            h = m & ~np.isnan(det_e)
            det = f'{_q(det_e[h], 50):.2f}' if h.sum() else '-'
            lines.append(f'| {name} | {int(m.sum())} | {_q(net_e[m], 50):.2f} | '
                         f'{_q(net_e[m], 90):.2f} | {h.sum()/m.sum():.3f} | {det} |')
        lines.append('')

    table('all held-out instances (confounded -- see above)', np.ones(n, bool))
    table('CONTROLLED: unclipped, apparent size 15-60 px',
          (~clip) & (size >= 15) & (size < 60))
    txt = '\n'.join(lines) + '\n'
    with open(REPORT, 'a') as fh:
        fh.write(txt)
    print(txt)


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='train',
                    choices=['train', 'eval', 'bench', 'sanity', 'detector', 'ribbon'])
    ap.add_argument('--split', default='block', choices=['block', 'session'])
    ap.add_argument('--tag', default='block')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--width', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-hours', type=float, default=8.0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--hflip', action='store_true')
    ap.add_argument('--ckpt', default='best')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--res', type=int, default=RES)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    RES = args.res
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True

    {'train': train, 'eval': eval_only, 'bench': bench, 'sanity': sanity,
     'detector': detector_baseline, 'ribbon': ribbon}[args.mode](args)


if __name__ == '__main__':
    main()
