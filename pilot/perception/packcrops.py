"""packcrops.py -- pre-rendered crop cache for `gatenet.py`, so a training run needs 4.4 GB
of tiles instead of 200k JPEGs and can therefore leave this laptop entirely.

WHY. `gatenet_runs/block/log.csv`: epochs 1-5 ran 27.6-34 s, epochs 185-189 ran 72.5-73.0 s,
median 72.5 s over 190 epochs, stepping at epoch 29 and never recovering. Every epoch
re-decodes 6598 JPEGs to rebuild the same 13008 crops and throws the work away, so caching
those crops is obviously worth doing -- and the obvious diagnosis, that the epoch is
JPEG-decode bound, is WRONG. It was measured before and after, and it does not survive:

    pure GPU, synthetic batches, no dataset at all, batch 64 : 440 ms/step, 6.88 ms/crop
                                                   -> 64.3 s of GPU per train epoch
    cached epoch, same laptop, batch 64, 4 workers : 72.0-72.7 s (train 66, val 6)
    uncached baseline, from log.csv               : 72.5 s median

66 s of measured train time against a 64.3 s GPU floor means the loader contributes about
2 s and the MX150 is ~98% of the epoch. **The cache does not make training faster on this
laptop, and this file does not claim it does.** The 27.6 s -> 72.5 s step at epoch 29 is
thermal too, but it is the GPU being throttled in a thin chassis, not the decode threads:
the synthetic-batch GPU floor measured on an already-hot machine is 64.3 s, i.e. 2.3x
slower than the whole cool epoch was.

WHAT THE CACHE ACTUALLY BUYS, all measured:

  1. **It makes the Colab move possible.** `gatenet_colab.ipynb` needs no JPEGs, no
     `pilot/sessions/`, no label file for anything but a staleness check -- one 4.4 GB
     memmap and a 1 MB index. Moving 200k frames to Drive was never going to happen.
     Prediction to test there, with a kill criterion: on a T4 the GPU floor should drop to
     roughly 8 s/epoch, at which point an 18-20 s JPEG loader WOULD be the bottleneck and
     the cache's 13-15 s would be marginal too. If Colab epochs do not land under ~20 s,
     the loader is the new limit and the next move is to hold the tiles in RAM, not to
     tune anything here.
  2. **~30% less loader CPU**, which is what competes with flying the simulator. One full
     train-epoch pass over the loader alone, no model, 4 workers, warm: cached 13.3 / 14.7 s
     vs source 18.0 / 20.7 s. Less than the 20x the decode story predicted, because
     `photometric()` -- float32 brightness/gamma/noise over 160x160x3 -- costs more per
     crop than `cv2.imread` did, and the cached path still pays a warp out of the tile.
  3. **Eval passes get ~4.7x faster** once the file is warm: 28.1 s -> 6 s over the 2670
     held-out instances. Ablations and re-scoring are eval-shaped work.

One cost, stated: the FIRST pass over a cold 4.4 GB cache is slower than decoding JPEGs
(95.4 s vs 28.1 s standalone) because it is 2670 random 339 KB reads off spinning-rust-speed
cold cache. It warms into RAM (24 GB here) and stays warm.

So: decode once, warp once, keep the pixels. One tile per labelled instance in a uint8
memmap, plus a sidecar index carrying everything `split_index()` and `breakdown()` need, so
a cached run's tables are directly comparable to the ones already in `TRAINING.md`.

-----------------------------------------------------------------------------------------
THE TILE GEOMETRY -- this is the part that decides whether the cache is honest
-----------------------------------------------------------------------------------------

Eval crops are deterministic: `crop_params()` with `rng=None` gives (cx, cy, S) and
`make_crop()` warps that square to RES x RES with scale a = RES / S. Training instead draws

    S' = S * exp(U(ln 0.78, ln 1.55))          # log-uniform: a box is wrong by a RATIO
    cx' = cx + U(-0.16, 0.16) * S'             # NOTE: the JITTERED S', not S
    cy' = cy + U(-0.16, 0.16) * S'

so the training window can reach, measured from the eval centre,

    half-extent = S' * (0.5 + 0.16) = 1.55 * 0.66 * S = 1.023 * S

The "1.87x" figure that falls out of 1.55 + 2*0.16 is wrong for THIS code, because the
translation is scaled by S' rather than S; the requirement is 2.046x, not 1.87x. Confirmed
by sampling 2e6 draws from the real expressions: max half-extent 1.02248 * S.

TILE_SCALE = 2.10 (half-side 1.05 * S) therefore covers every draw with 2.7% to spare, and
a tile crop is a crop WITHIN the tile -- never a fresh warp from the JPEG.

RESOLUTION IS NOT A FREE CHOICE -- IT IS THE WHOLE DECISION. Setting

    TILE_RES = round(RES * TILE_SCALE) = 160 * 2.10 = 336

makes the tile's warp scale a_t = TILE_RES / (TILE_SCALE * S) = RES / S -- *identical* to
the eval warp's. The tile is then literally the eval crop's own sampling grid extended to a
2.10x field of view, and the eval crop is the integer sub-window

    tile[OFF : OFF+RES, OFF : OFF+RES]   with OFF = (TILE_RES - RES) // 2 = 88

taken with no resampling at all. Any other tile resolution makes eval a warp of a warp.
Measured cost of getting that wrong, val split, same `best.pt`, 2670 held-out instances:

    tile 336 (on-grid) :  0.77% of eval samples differ, max 7 grey levels
                          -> worst table cell 0.049 px, p99 per-instance 0.013 px
    tile 256 (off-grid): 40.3% of eval samples differ, max 217 grey levels
                          -> worst table cell 3.65 px, p99 per-instance 6.54 px,
                             one instance off by 58.8 px

3.65 px on a table whose headline number is 0.89 px is not a rounding difference, it is a
different experiment. So the tile is 336 and the file is 4.4 GB, and the 2.0-2.6 GB the
size budget would have preferred is not on offer.

WHY IT IS 0.77% AND NOT ZERO. It should have been zero -- same scale, integer offset -- and
the first version of this file claimed so. `--mode verify` said otherwise, and the reason is
inside `cv2.warpAffine`: it does not evaluate the affine map per pixel in floating point, it
builds a PER-COLUMN integer table `adelta[x] = round(iM00 * x * INTER_TAB_SIZE)` with
INTER_TAB_SIZE = 32, i.e. the source coordinate is quantised to 1/32 px separately for each
destination column. Shifting the destination index by 88 re-rounds every entry, and a few
columns land on the far side of a 1/32 boundary. Localised exactly as that predicts: on a
probe instance, **7 columns of 160 differ and all 160 rows within them do**, max 5 levels.
It is a 1/32-px grid dither, not a blur or a shift -- mean signed difference 0.005 levels --
and `tile_warp()` below still reuses make_crop's own float32 translation so nothing worse
than that is left. The residual is measured and reported rather than tuned away.

FIDELITY COST OF THE TRAINING PATH, stated rather than assumed. A training draw that ZOOMS
IN (r < 1) asks for density RES/(r*S) and the tile carries RES/S, so it resamples up by at
most 1/0.78 = 1.28x. That is a real loss ONLY where the source JPEG could have supplied the
difference, i.e. r*S >= 160, i.e. S >= 205 px: **1110 of 13008 instances (8.5%)**, and only
on the zoomed end of their jitter draws. The other 91.5% are upsampled from the source in
BOTH pipelines, so the tile holds everything the frame had.

SIZE ON DISK: 13008 x 336 x 336 x 3 uint8 = **4.41 GB** (4.10 GiB), plus a ~1 MB index.

-----------------------------------------------------------------------------------------
USAGE
    python3 pilot/perception/packcrops.py --mode pack                    # build the cache
    python3 pilot/perception/packcrops.py --mode verify --tag block      # cached vs source
    python3 pilot/perception/packcrops.py --mode train --split block --tag block-cached
    python3 pilot/perception/packcrops.py --mode epochbench --split block

`--mode train` does NOT reimplement training. It monkey-patches `gatenet.load_index` and
`gatenet.make_loaders` and then calls `gatenet.train()` unchanged, so the optimiser,
schedule, checkpointing, --resume, --max-hours, OOM handling and the TRAINING.md report all
stay in one place. The notebook does the same thing. gatenet.py is not edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gatenet as G  # noqa: E402

# scale jitter tops out at 1.55 and translation is +/-0.16 of the JITTERED side, so the
# worst window reaches 1.55*(0.5+0.16) = 1.023 of the eval side from the eval centre.
TILE_SCALE = 2.10
CACHE = os.path.join(HERE, 'crop_cache')


def tile_res_for(res, scale=TILE_SCALE):
    """Tile side that keeps the tile ON the eval crop's sampling grid.

    res*scale must be an integer and (tile_res - res) must be even, or the centre RES x RES
    window stops being an exact integer sub-window and eval becomes a warp of a warp."""
    tr = int(round(res * scale))
    if (tr - res) % 2:
        tr += 1
    return tr


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _fingerprint():
    """Everything the cached pixels depend on. A cache whose fingerprint no longer matches
    the live code/labels is STALE and must not be silently used -- that is exactly the class
    of bug this file exists to avoid."""
    return {
        'labels_sha256': _sha256(G.LABELS),
        'RES': G.RES, 'MARGIN': G.MARGIN, 'MIN_VIS_PX': G.MIN_VIS_PX,
        'W': G.W, 'H': G.H,
    }


def tile_warp(img, cx, cy, S, res, tres):
    """The eval warp, extended to a (tres/res)x field of view WITHOUT moving the grid.

    Rendering the tile as an independent warp with a = tres / (scale * S) is the same map
    algebraically but not the same FLOATS. This builds M from make_crop's OWN expressions --
    a = res / S, tx = res/2 - cx*a in float32 -- and adds the INTEGER offset, so the two
    inverse maps differ by exactly (off, off) whole pixels and no last-ulp translation error
    is introduced.

    That removes one source of drift but NOT all of it: warpAffine's per-column 1/32-px
    coordinate table re-rounds when the column index shifts by `off`, which is why
    `--mode verify` still reports 0.77% of eval samples differing by <= 7 grey levels. See
    the module docstring -- it is a grid dither worth 0.05 px on a reported table cell, and
    it is measured every time rather than assumed away."""
    off = (tres - res) // 2
    a = res / S
    M = np.array([[a, 0.0, res / 2.0 - cx * a],
                  [0.0, a, res / 2.0 - cy * a]], np.float32)
    M[0, 2] += off
    M[1, 2] += off
    return cv2.warpAffine(img, M, (tres, tres), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ---------------------------------------------------------------------------------------
# pack


def pack(args):
    res = args.res
    tres = args.tile_res or tile_res_for(res, args.tile_scale)
    off = (tres - res) // 2
    exact = abs(tres - res * args.tile_scale) < 1e-6 and (tres - res) % 2 == 0

    items = G.load_index()
    keep = list(range(len(items)))
    if args.subset != 'all':
        tr, va = G.split_index(items, args.split)
        want = set(id(d) for d in (tr if args.subset == 'train' else va))
        keep = [i for i, d in enumerate(items) if id(d) in want]
    n = len(keep)

    out = args.out or CACHE
    os.makedirs(out, exist_ok=True)
    mmpath = os.path.join(out, 'tiles.npy')
    nbytes = n * tres * tres * 3
    print(f'packing {n} instances -> {mmpath}  '
          f'({tres}x{tres}, scale {args.tile_scale}, {nbytes/1e9:.2f} GB, '
          f'eval-exact={exact})', flush=True)

    mm = np.lib.format.open_memmap(mmpath, mode='w+', dtype=np.uint8,
                                   shape=(n, tres, tres, 3))

    geo = np.zeros((n, 3), np.float64)   # cx,cy,S in float32 would perturb
                                         # to_norm() at the 1e-4 px level; free to avoid
    corners = np.zeros((n, 4, 2), np.float32)
    size_px = np.zeros(n, np.float32)
    clipped = np.zeros(n, bool)
    occluded = np.zeros(n, bool)
    ordinal = np.zeros(n, np.int32)
    body_rate = np.zeros(n, np.float32)   # rad/s at capture; drives --weight-mode rate
    srcs = []                             # 'auto' | 'hand' provenance (mergelabels.py)
    orig_idx = np.asarray(keep, np.int32)
    sessions = []

    # decode each JPEG once: instances are ordered by (session, frame), so grouping by path
    # is a single pass with no dict of decoded images to hold.
    by_path = {}
    for slot, i in enumerate(keep):
        by_path.setdefault(items[i]['path'], []).append((slot, i))

    t0 = time.time()
    missing = 0
    for j, path in enumerate(sorted(by_path)):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((G.H, G.W, 3), np.uint8)
            missing += 1
        for slot, i in by_path[path]:
            d = items[i]
            cx, cy, S = G.crop_params(d['corners'])
            mm[slot] = (tile_warp(img, cx, cy, S, res, tres) if exact else
                        G.make_crop(img, cx, cy, S * args.tile_scale, tres))
            geo[slot] = (cx, cy, S)
            corners[slot] = d['corners']
            size_px[slot] = d['size_px']
            clipped[slot] = d['clipped']
            occluded[slot] = d['occluded']
            ordinal[slot] = d['ordinal']
            body_rate[slot] = d.get('body_rate', 0.0)
            srcs.append(d.get('src', 'auto'))
            sessions.append(d['session'])
        if j % 500 == 0:
            print(f'  {j}/{len(by_path)} frames  {time.time()-t0:.0f}s', flush=True)
    mm.flush()
    del mm

    np.savez_compressed(
        os.path.join(out, 'index.npz'),
        geo=geo, corners=corners, size_px=size_px, clipped=clipped, occluded=occluded,
        ordinal=ordinal, orig_idx=orig_idx, session=np.array(sessions),
        body_rate=body_rate, src=np.array(srcs))
    meta = {'n': n, 'tile_res': tres, 'tile_scale': args.tile_scale, 'res': res,
            'offset': off, 'eval_exact': bool(exact), 'subset': args.subset,
            'split': args.split if args.subset != 'all' else None,
            'missing_frames': missing, 'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'bytes': nbytes, **_fingerprint()}
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(f'done in {time.time()-t0:.0f}s; {missing} frames unreadable', flush=True)
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------------------------------
# read side


def load_cache(path=None, strict=True, open_tiles=False):
    """-> (items, tiles_or_None, meta). `items` are shaped exactly like
    `gatenet.load_index()` output, so `split_index` / `breakdown` / the report code all work
    untouched.

    `open_tiles` DEFAULTS TO FALSE and callers should leave it there. Measured on this
    laptop (24 GB RAM, 31 GB commit limit, 10 GB commit free): with the 4.41 GB memmap held
    open in the PARENT, spawning 4 DataLoader workers dies with

        OSError: [Errno 22] Invalid argument  /  UnpicklingError: pickle data was truncated

    while 0 and 2 workers succeed, and the same code on the 0.90 GB val-only cache succeeds
    at 4. It is not the pickle -- the dataset pickles to 476 KB, measured. It is Windows
    commit charge: `spawn` clones the parent's address space reservations, so the mapping is
    charged once per child. Drop the mapping in the parent and each worker maps it itself
    (`CachedGateCrops.tiles`), which is what you want anyway for the page cache."""
    path = path or CACHE
    meta = json.load(open(os.path.join(path, 'meta.json')))
    fp = _fingerprint()
    for k, v in fp.items():
        if meta.get(k) != v:
            msg = (f'cache {path} is STALE: {k} is {meta.get(k)!r} in the cache but '
                   f'{v!r} now. Repack.')
            if strict:
                raise SystemExit(msg)
            print('WARNING: ' + msg, flush=True)
    meta['_path'] = path
    z = np.load(os.path.join(path, 'index.npz'), allow_pickle=False)
    tiles = (np.load(os.path.join(path, 'tiles.npy'), mmap_mode='r')
             if open_tiles else None)
    # Materialise each column ONCE, then hand out COPIES rather than views. Two distinct
    # bugs, both measured here: `z['corners']` decompresses the whole array on every
    # subscript, so indexing it inside the loop is 13008 full decompressions; and a view
    # keeps its 13008-row base object alive, so every item drags the entire array into any
    # pickle of it. A spawned Windows DataLoader worker pickles the dataset -- with views it
    # died on the 13008-instance cache with `OSError [Errno 22]` / `pickle data was
    # truncated` while working fine on the 2670-instance one, which is exactly the shape of
    # bug that passes a small test and fails the real run.
    cols = {k: np.array(z[k]) for k in
            ('session', 'ordinal', 'corners', 'clipped', 'occluded', 'size_px', 'geo')}
    # body_rate / src arrived with the v3 labels; a cache packed before then simply lacks
    # the arrays. Defaults (0.0 / 'auto') make --weight-mode rate weight everything 1.0
    # there, so it is called out loudly rather than silently doing nothing.
    for k in ('body_rate', 'src'):
        if k in z.files:
            cols[k] = np.array(z[k])
        else:
            print(f'NOTE: cache index has no {k!r} column (packed by an older '
                  f'packcrops.py) -- weight-mode rate and the rate/src eval rows will '
                  f'see defaults. Repack to carry it.', flush=True)
    items = []
    for k in range(meta['n']):
        items.append({
            'slot': k,
            'path': None,                      # the cache exists so nothing reads a JPEG
            'session': str(cols['session'][k]),
            'ordinal': int(cols['ordinal'][k]),
            'corners': cols['corners'][k].copy(),
            'clipped': bool(cols['clipped'][k]),
            'occluded': bool(cols['occluded'][k]),
            'size_px': float(cols['size_px'][k]),
            'geo': cols['geo'][k].copy(),
            'body_rate': float(cols['body_rate'][k]) if 'body_rate' in cols else 0.0,
            'src': str(cols['src'][k]) if 'src' in cols else 'auto',
        })
    return items, tiles, meta


class CachedGateCrops(Dataset):
    """`gatenet.GateCrops` with the JPEG replaced by its pre-rendered tile.

    Everything that touches the TARGET is the untouched gatenet code path -- same
    `crop_params` jitter draw, same `to_norm`, same `photometric`, same hflip
    re-canonicalisation. Only the source of pixels changes, and at eval not even that:
    the centre window is sliced out, so no second resampling happens at all."""

    def __init__(self, items, cache_path, meta, train, hflip=False):
        self.items, self.train, self.hflip = items, train, hflip
        self.cache_path = cache_path
        self._tiles = None       # opened lazily -- see tiles()
        self.res = meta['res']
        self.tres = meta['tile_res']
        self.tscale = meta['tile_scale']
        self.off = meta['offset']
        self.exact = meta['eval_exact']

    @property
    def tiles(self):
        """Open the memmap per PROCESS, never at construction.

        On Windows DataLoader workers are SPAWNED, so the dataset is pickled to each one.
        A `np.memmap` pickles by VALUE -- numpy materialises the whole array -- so holding
        it as an attribute tries to ship 4.4 GB down a pipe per worker and dies with
        `OSError: [Errno 22]` / `pickle data was truncated`. Measured, not guessed: that is
        exactly how the first version failed. Opening in the child costs one mmap syscall
        and shares the OS page cache between workers."""
        if self._tiles is None:
            self._tiles = np.load(os.path.join(self.cache_path, 'tiles.npy'),
                                  mmap_mode='r')
        return self._tiles

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        d = self.items[i]
        tile = self.tiles[d['slot']]
        cx, cy, S = (float(v) for v in d['geo'])
        corners = d['corners']

        if not self.train:
            if self.exact:
                o, r = self.off, self.res
                crop = np.ascontiguousarray(tile[o:o + r, o:o + r])
            else:
                # off-grid tile: the eval window is not an integer sub-window, so it has to
                # be resampled. Reported, never hidden -- meta['eval_exact'] is False.
                c = self.tres / 2.0
                crop = G.make_crop(np.asarray(tile), c, c, self.tres / self.tscale,
                                   self.res)
            tgt = G.to_norm(corners, cx, cy, S)
        else:
            rng = random.Random((i * 2654435761 + int(time.time() * 1e6)) & 0xFFFFFFFF)
            jx, jy, jS = G.crop_params(corners, rng)
            # original px -> tile px is a pure similarity centred on (cx, cy). On an
            # on-grid tile the scale is literally the eval crop's own a = RES/S; the
            # jittered window is a square inside the tile and never leaves it (see the
            # module docstring's measured 1.0225 * S bound vs the 1.05 * S half-side).
            a = self.res / S if self.exact else self.tres / (self.tscale * S)
            crop = G.make_crop(np.asarray(tile),
                               (jx - cx) * a + self.tres / 2.0,
                               (jy - cy) * a + self.tres / 2.0,
                               jS * a, self.res)
            tgt = G.to_norm(corners, jx, jy, jS)
            cx, cy, S = jx, jy, jS
            crop = G.photometric(crop, rng)
            if self.hflip and rng.random() < 0.5:
                crop = crop[:, ::-1].copy()
                tgt = np.stack([-tgt[:, 0], tgt[:, 1]], 1)
                tgt = G.canon(tgt).astype(np.float32)

        x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        x = x.div_(255.0).sub_(0.45).div_(0.25)
        return x, torch.from_numpy(tgt.reshape(8).astype(np.float32)), \
            torch.tensor([cx, cy, S], dtype=torch.float32), i


def attach(path=None, strict=True):
    """Point `gatenet` at the cache. Returns the loaded (items, tiles, meta).

    Monkey-patching rather than editing gatenet.py is deliberate: the training loop,
    checkpoint format, resume logic and report writer must stay byte-identical between a
    laptop run, a cached run and a Colab run, or the tables stop being comparable."""
    items, tiles, meta = load_cache(path, strict)

    def load_index():
        return items

    def make_loaders(tr_items, va_items, bs, workers, hflip):
        tr = DataLoader(CachedGateCrops(tr_items, meta['_path'], meta, True, hflip),
                        batch_size=bs, shuffle=True, num_workers=workers,
                        pin_memory=True, drop_last=True,
                        persistent_workers=workers > 0,
                        prefetch_factor=4 if workers else None)
        va = DataLoader(CachedGateCrops(va_items, meta['_path'], meta, False),
                        batch_size=max(bs, 128), shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0,
                        prefetch_factor=4 if workers else None)
        return tr, va

    G.load_index = load_index
    G.make_loaders = make_loaders
    return items, tiles, meta


# ---------------------------------------------------------------------------------------
# verify -- deliverable: the cache must not have changed the data


def _sub(items, idx):
    return [items[i] for i in idx]


def verify(args):
    """Same checkpoint, same held-out instances, once through JPEGs and once through the
    cache. If a cache changes the numbers, every future measurement silently stops being
    comparable to TRAINING.md and nobody finds out -- so this runs the whole `evaluate()`
    /`breakdown()` path both ways and prints both tables."""
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    citems, tiles, meta = load_cache(args.out or CACHE, open_tiles=True)
    print(json.dumps(meta, indent=1), flush=True)

    src_items = G.load_index()
    if meta['subset'] == 'all':
        _, va_src = G.split_index(src_items, args.split)
        _, va_cac = G.split_index(citems, args.split)
    else:
        va_src = [src_items[i] for i in
                  np.load(os.path.join(args.out or CACHE, 'index.npz'))['orig_idx']]
        va_cac = citems
    assert len(va_src) == len(va_cac), (len(va_src), len(va_cac))
    print(f'held-out instances: {len(va_src)}', flush=True)

    # --- 1. pixels. The strongest check available, and it is free: at TILE_RES on-grid the
    #        cached eval crop should be BIT-equal to the freshly warped one.
    ds_c = CachedGateCrops(va_cac, meta['_path'], meta, False)
    ndiff, maxdiff, npx = 0, 0, 0
    probe = np.linspace(0, len(va_src) - 1, min(400, len(va_src))).astype(int)
    for k in probe:
        d = va_src[int(k)]
        img = cv2.imread(d['path'], cv2.IMREAD_COLOR)
        if img is None:
            continue
        cx, cy, S = G.crop_params(d['corners'])
        a = G.make_crop(img, cx, cy, S)
        b = (ds_c[int(k)][0] * 0.25 + 0.45).mul_(255.0).numpy() \
            .transpose(1, 2, 0)
        b = np.clip(np.round(b), 0, 255).astype(np.uint8)
        dif = np.abs(a.astype(np.int16) - b.astype(np.int16))
        ndiff += int((dif > 0).sum())
        maxdiff = max(maxdiff, int(dif.max()))
        npx += dif.size
    print(f'\nPIXEL CHECK on {len(probe)} eval crops: {ndiff}/{npx} differing samples '
          f'({100.0*ndiff/max(npx,1):.4f}%), max |diff| = {maxdiff} levels', flush=True)

    # --- 2. the actual reported tables, both ways
    m = G.GateNet(args.width).to(dev)
    ck = os.path.join(G.RUNS, args.tag, f'{args.ckpt}.pt')
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)['model'])
    m.eval()

    ld_src = DataLoader(G.GateCrops(va_src, False), batch_size=128, shuffle=False,
                        num_workers=args.workers)
    ld_cac = DataLoader(ds_c, batch_size=128, shuffle=False, num_workers=args.workers)

    t = time.time()
    r_src = G.evaluate(m, ld_src, dev, va_src)
    t_src = time.time() - t
    t = time.time()
    r_cac = G.evaluate(m, ld_cac, dev, va_cac)
    t_cac = time.time() - t

    rows_src = G.breakdown(r_src, va_src)
    rows_cac = G.breakdown(r_cac, va_cac)

    print(f'\n### ORIGINAL pipeline (decode JPEG -> warp)   [{t_src:.1f} s]')
    print(G.fmt_rows(rows_src))
    print(f'\n### CACHED pipeline (memmap tile)             [{t_cac:.1f} s]')
    print(G.fmt_rows(rows_cac))

    # per-instance deltas: an aggregate table can agree while individuals disagree, and it
    # is the individuals that a downstream consumer sees.
    o_s = np.argsort(r_src['idx'])
    o_c = np.argsort(r_cac['idx'])
    ce_s, ce_c = r_src['corner'][o_s], r_cac['corner'][o_c]
    cn_s, cn_c = r_src['centre'][o_s], r_cac['centre'][o_c]
    dce = np.abs(ce_s - ce_c).ravel()
    dcn = np.abs(cn_s - cn_c)
    print('\n### per-instance agreement (cached minus original, absolute px)')
    print(f'* corner error: max {dce.max():.6f} px, p99 {np.percentile(dce,99):.6f}, '
          f'median {np.median(dce):.6f}')
    print(f'* centre error: max {dcn.max():.6f} px, p99 {np.percentile(dcn,99):.6f}, '
          f'median {np.median(dcn):.6f}')
    print(f'* val loss: original {r_src["loss"]:.8f}  cached {r_cac["loss"]:.8f}  '
          f'delta {abs(r_src["loss"]-r_cac["loss"]):.2e}')

    worst = np.argsort(-np.abs(ce_s - ce_c).mean(1))[:5]
    print('* largest-disagreement instances (size_px, |delta| mean corner px):')
    for w in worst:
        print(f'    {va_cac[int(w)]["size_px"]:8.1f} px  '
              f'{np.abs(ce_s[w]-ce_c[w]).mean():.6f}')

    tabdiff = max(abs(a[k] - b[k]) for a, b in zip(rows_src, rows_cac)
                  for k in ('corner_med', 'corner_p90', 'centre_med', 'centre_p90'))
    print(f'\n* worst table-cell disagreement across all groups: {tabdiff:.6f} px')


# ---------------------------------------------------------------------------------------
# train / epochbench


def train(args):
    attach(args.out or CACHE, strict=not args.allow_stale)
    # --runs / --report exist so a smoke test cannot append junk to the real TRAINING.md.
    # gatenet.train() writes the report unconditionally, which is correct for a real run
    # and wrong for a two-epoch check that the plumbing works.
    if args.runs:
        G.RUNS = args.runs
    if args.report:
        G.REPORT = args.report
    G.train(args)


def epochbench(args):
    """Measured epoch wall time, cached vs the 72.5 s median in gatenet_runs/block/log.csv.

    Times the real training loop for --bench-epochs epochs. Thermal state matters on this
    machine (27.6 s epoch 1 -> 72.5 s epoch 29, same code), so a single fast epoch proves
    nothing; run enough of them that the laptop has settled, and IGNORE epoch 0 -- it also
    pays for warming 4.4 GB of cold page cache.

    Also prints the GPU floor: the same optimiser step on a synthetic batch that never
    touches the dataset. Without it, a wall-clock number cannot distinguish "the loader is
    slow" from "the GPU is saturated", and on this laptop it is the second one -- which is
    the finding that sends the work to Colab rather than to a faster loader."""
    items, tiles, meta = attach(args.out or CACHE, strict=not args.allow_stale)
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tr_items, va_items = G.split_index(items, args.split)
    tr_loader, va_loader = G.make_loaders(tr_items, va_items, args.batch, args.workers,
                                          False)
    model = G.GateNet(args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    import torch.nn.functional as F
    secs = []
    for ep in range(args.bench_epochs):
        model.train()
        t0 = time.time()
        for x, y, g, _ in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
                loss = F.smooth_l1_loss(model(x), y, beta=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        t_train = time.time() - t0
        G.evaluate(model, va_loader, dev, va_items)
        secs.append(time.time() - t0)
        print(f'  epoch {ep}: {secs[-1]:.1f} s  (train {t_train:.1f} s, '
              f'val {secs[-1]-t_train:.1f} s)', flush=True)
    warm = secs[1:] or secs
    print(f'\ncached epoch time: median {np.median(warm):.1f} s over {len(warm)} warm '
          f'epochs ({min(warm):.1f}-{max(warm):.1f}); epoch 0 was {secs[0]:.1f} s cold')
    print(f'baseline (gatenet_runs/block/log.csv): 72.5 s median over 190 epochs')

    # the GPU floor, on data that never existed: everything above this is loader.
    steps = len(tr_items) // args.batch
    x = torch.randn(args.batch, 3, G.RES, G.RES, device=dev)
    y = torch.randn(args.batch, 8, device=dev)
    for _ in range(10):
        with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
            loss = F.smooth_l1_loss(model(x), y, beta=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(40):
        with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
            loss = F.smooth_l1_loss(model(x), y, beta=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    step = (time.perf_counter() - t) / 40.0
    print(f'GPU floor ({dev}, synthetic batches, no dataset): {1000*step:.1f} ms/step, '
          f'{1000*step/args.batch:.2f} ms/crop -> {steps*step:.1f} s per train epoch')
    print(f'-> the loader accounts for {np.median(warm) - steps*step:.1f} s of a '
          f'{np.median(warm):.1f} s epoch; the rest is the GPU.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='pack',
                    choices=['pack', 'verify', 'train', 'epochbench'])
    ap.add_argument('--out', default=None, help='cache directory (default crop_cache/)')
    ap.add_argument('--tile-scale', type=float, default=TILE_SCALE)
    ap.add_argument('--tile-res', type=int, default=None,
                    help='override; off-grid values make eval non-exact, on purpose only')
    ap.add_argument('--subset', default='all', choices=['all', 'train', 'val'])
    ap.add_argument('--allow-stale', action='store_true')
    ap.add_argument('--bench-epochs', type=int, default=6)
    ap.add_argument('--runs', default=None, help='override gatenet.RUNS (checkpoint dir)')
    ap.add_argument('--report', default=None, help='override gatenet.REPORT (TRAINING.md)')
    # everything below mirrors gatenet.py's parser, because --mode train hands `args`
    # straight to gatenet.train() and it must find the fields it expects.
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
    ap.add_argument('--res', type=int, default=G.RES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--labels', default=None,
                    help='label file for G.LABELS (default: $GATENET_LABELS or '
                         'autolabels_vq1.json); part of the cache fingerprint')
    ap.add_argument('--weight-mode', default='none', choices=['none', 'rate'])
    args = ap.parse_args()

    if args.labels:
        G.LABELS = os.path.abspath(args.labels)
    G.RES = args.res
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True
    {'pack': pack, 'verify': verify, 'train': train, 'epochbench': epochbench}[args.mode](
        args)


if __name__ == '__main__':
    main()
