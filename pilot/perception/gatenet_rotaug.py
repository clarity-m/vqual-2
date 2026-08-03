"""gatenet_rotaug.py -- camera-roll (in-plane rotation) augmentation for gatenet training.

WHY. The VQ1 training footage is slow, level flying; the VQ2 race is acro with large bank, so
the net has never seen a rolled gate. For a pinhole camera a pure rotation about the optical
centre is an EXACT homography H = K R K^-1 (K: fx=fy=320, cx=320, cy=180), so warping image and
labels by the SAME map keeps the pair geometrically consistent -- no new label noise.

ONE VARIABLE, TILE-LEVEL ONLY. A true camera roll rotates the whole 640x360 frame about the
principal point, which also TRANSLATES an off-centre gate. Doing that needs a full-frame
re-warp from the JPEGs and breaks the cache. This run instead rotates each cached tile about
its own centre -- the leading-order term of roll for a gate near frame centre -- because it
is free (the tile is already in hand) and changes exactly one thing versus colab-v3-nw.
Full-frame roll is deliberately out of scope here.

THE HEADROOM, AND THE SAFE ANGLE. packcrops packs tiles at TILE_SCALE=2.10 (half-side
1.05*S) but the crop trained/evaluated on is MARGIN=1.60 (half-side 0.80*S). Rotating that
0.80*S square about its centre, its corner reaches 0.80*S*(cos+sin) along the tile axes; it
stays inside the 1.05*S tile while

    cos A + sin A  <=  TILE_SCALE / MARGIN = 1.3125
    sqrt(2) sin(A + 45 deg) <= 1.3125  ->  A <= 23.14 deg

so SAFE_ANGLE_DEG = 23.14 deg is the largest roll for which the EVAL-sized crop is sourced
entirely from real tile pixels. It is conservative for this run: the actual tile/crop side
ratio is TILE_SCALE = 2.10 (the tile is TILE_SCALE * S, S already = MARGIN * visible box), so
even a 45-deg roll of the eval crop clears the tile -- 23 deg is the tighter figure the two
constants give and it is what we cap at.

ROLL ABOUT THE TILE CENTRE, NOT THE CROP CENTRE. The training crop is jittered off the tile
centre, so rolling about the *crop* centre swings the aperture -- which sits near the *tile*
centre -- toward the tile edge, and a large-jitter draw then samples black onto the aperture
(measured: up to 21% blacked at 23 deg). Rolling about the TILE centre keeps the aperture near
the pivot, where the 2.10x headroom always covers it; the jittered window is extracted from
the rolled tile. BORDER is BORDER_CONSTANT black (matching make_crop and inference -- nothing
beyond the sensor); rotaug_check.py verifies the roll never blacks an unclipped aperture.

WINDING. Corners wind clockwise-on-screen from the min(x+y) corner (autolabel.canon). Image
and target go through the SAME affine M, so they agree by construction; a rotation preserves
the shoelace sign (canon never reverses) but moves which corner is min(x+y), so the target is
re-canonicalised after warping -- exactly as gatenet's --hflip path already does.

INSTALL. `attach_rotaug()` runs `packcrops.attach()` (points gatenet.load_index/make_loaders
at the cache), then overrides make_loaders once more so ONLY the train dataset is wrapped in
RotAugCrops; the val loader stays the untouched CachedGateCrops. gatenet.py and packcrops.py
are not edited -- pure runtime monkey-patch -- so the optimiser, schedule, checkpointing and
report code stay byte-identical and the tables stay comparable to TRAINING.md.
"""

from __future__ import annotations

import math
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

import gatenet as G      # noqa: E402  -- canon, crop_params, to_norm, photometric, make_crop
import packcrops as P    # noqa: E402  -- CachedGateCrops, attach, TILE_SCALE


def safe_angle_deg(tile_scale=None, margin=None):
    """Largest roll (deg) that keeps a MARGIN-sized crop inside a TILE_SCALE tile.

    Derived from cos A + sin A <= tile_scale/margin (see the module docstring)."""
    tile_scale = P.TILE_SCALE if tile_scale is None else tile_scale
    margin = G.MARGIN if margin is None else margin
    r = tile_scale / margin
    assert r > 1.0, (tile_scale, margin, r)          # a tile smaller than the crop is a bug
    x = r / math.sqrt(2.0)
    assert 0.0 < x <= 1.0, x                          # else the crop already fills the tile
    return math.degrees(math.asin(x) - math.pi / 4.0)


SAFE_ANGLE_DEG = safe_angle_deg()                    # 23.19 deg for 2.10 / 1.60


def roll_transform(cx, cy, S, jx, jy, jS, deg, res, tres, tscale, exact):
    """Forward affine tile-px -> output-crop-px for a roll of `deg` about the TILE centre
    followed by extraction of the jittered (jx, jy, jS) window. Returned so the image warp
    and the target both use ONE matrix and cannot drift."""
    a = res / S if exact else tres / (tscale * S)
    ctrx, ctry = (jx - cx) * a + tres / 2.0, (jy - cy) * a + tres / 2.0
    a2 = res / (jS * a)                               # make_crop(tile, ctr, jS*a, res)'s scale
    Mb = np.array([[a2, 0.0, res / 2.0 - ctrx * a2],
                   [0.0, a2, res / 2.0 - ctry * a2]], np.float32)
    Rt = cv2.getRotationMatrix2D((tres / 2.0, tres / 2.0), deg, 1.0)   # roll about TILE centre
    return (np.vstack([Mb, [0, 0, 1]]) @ np.vstack([Rt, [0, 0, 1]]))[:2].astype(np.float32), a


class RotAugCrops(Dataset):
    """Wrap a TRAIN `packcrops.CachedGateCrops` and add per-sample roll about the TILE centre.

    Everything except the roll is the untouched cached train path: same RNG seeding, the same
    `crop_params` jitter draw consumed FIRST (so the jitter distribution is bit-for-bit the
    baseline's), the same tile->crop similarity, the same `photometric`, the same optional
    hflip re-canonicalisation. The roll is folded into the SAME warpAffine that extracts the
    crop, so a sample is resampled exactly ONCE (no warp-of-a-warp), and the target goes
    through the identical matrix, so image and label cannot disagree."""

    def __init__(self, base, max_angle_deg=SAFE_ANGLE_DEG):
        assert isinstance(base, P.CachedGateCrops), type(base)
        assert base.train, 'RotAugCrops is train-only; the val loader must stay untouched'
        assert 0.0 < max_angle_deg <= SAFE_ANGLE_DEG + 1e-9, (max_angle_deg, SAFE_ANGLE_DEG)
        self.base = base
        self.A = float(max_angle_deg)
        self.res, self.tres = base.res, base.tres
        self.tscale, self.exact, self.hflip = base.tscale, base.exact, base.hflip

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        b = self.base
        d = b.items[i]
        tile = np.asarray(b.tiles[d['slot']])        # opens the memmap per-process (property)
        assert tile.shape == (self.tres, self.tres, 3), tile.shape
        cx, cy, S = (float(v) for v in d['geo'])
        corners = np.asarray(d['corners'], np.float32)

        # IDENTICAL seeding + jitter draw to CachedGateCrops.__getitem__, drawn first so the
        # jitter is bit-for-bit the baseline; the roll is the only added draw.
        rng = random.Random((i * 2654435761 + int(time.time() * 1e6)) & 0xFFFFFFFF)
        jx, jy, jS = G.crop_params(corners, rng)
        assert jS > 0.0, jS
        deg = rng.uniform(-self.A, self.A)

        M, a = roll_transform(cx, cy, S, jx, jy, jS, deg, self.res, self.tres,
                              self.tscale, self.exact)
        crop = cv2.warpAffine(tile, M, (self.res, self.res), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # target through the SAME M: corner px -> tile px -> crop px -> norm -> canon
        tile_px = np.stack([(corners[:, 0] - cx) * a + self.tres / 2.0,
                            (corners[:, 1] - cy) * a + self.tres / 2.0], 1)
        crop_px = np.c_[tile_px, np.ones(4, np.float32)] @ M.T
        tgt = G.canon((crop_px - self.res / 2.0) / (self.res / 2.0)).astype(np.float32)

        crop = G.photometric(crop, rng)
        if self.hflip and rng.random() < 0.5:
            crop = crop[:, ::-1].copy()
            tgt = np.stack([-tgt[:, 0], tgt[:, 1]], 1)
            tgt = G.canon(tgt).astype(np.float32)

        x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        x = x.div_(255.0).sub_(0.45).div_(0.25)
        return x, torch.from_numpy(tgt.reshape(8).astype(np.float32)), \
            torch.tensor([jx, jy, jS], dtype=torch.float32), i


def attach_rotaug(cache_path=None, max_angle_deg=SAFE_ANGLE_DEG, strict=True):
    """Point gatenet at the cache (via packcrops.attach) and wrap ONLY the train loader.

    Returns (items, tiles, meta), same as packcrops.attach. Call BEFORE gatenet.train():
    G.make_loaders is what train() uses to build its loaders, and this replaces it with a
    version whose train dataset is RotAugCrops and whose val dataset is the plain
    CachedGateCrops -- so the only thing that changes versus a cached run is train-time roll."""
    items, tiles, meta = P.attach(cache_path, strict=strict)
    mpath = meta['_path']
    A = float(max_angle_deg)

    def make_loaders(tr_items, va_items, bs, workers, hflip):
        base_tr = P.CachedGateCrops(tr_items, mpath, meta, True, hflip)
        tr = DataLoader(RotAugCrops(base_tr, A), batch_size=bs, shuffle=True,
                        num_workers=workers, pin_memory=True, drop_last=True,
                        persistent_workers=workers > 0,
                        prefetch_factor=4 if workers else None)
        va = DataLoader(P.CachedGateCrops(va_items, mpath, meta, False),
                        batch_size=max(bs, 128), shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0,
                        prefetch_factor=4 if workers else None)
        return tr, va

    G.make_loaders = make_loaders
    print(f'[rotaug] installed: train loader rolls +/-{A:.2f} deg about the crop centre '
          f'(safe angle {SAFE_ANGLE_DEG:.2f}); val loader untouched', flush=True)
    return items, tiles, meta


if __name__ == '__main__':
    # data-free self-check: the safe angle, and that the target rotation is the SAME map as
    # the image warp (analytic round-trip). The heavy, data-backed checks live in
    # rotaug_check.py; this one runs anywhere in a second.
    print(f'SAFE_ANGLE_DEG = {SAFE_ANGLE_DEG:.4f}  (TILE_SCALE {P.TILE_SCALE} / MARGIN '
          f'{G.MARGIN} = {P.TILE_SCALE / G.MARGIN:.4f})')
    A = SAFE_ANGLE_DEG
    assert abs(math.cos(math.radians(A)) + math.sin(math.radians(A))
               - P.TILE_SCALE / G.MARGIN) < 1e-9
    # a rolled crop of eval size must have its far corner land exactly on the tile edge
    half = (G.MARGIN / 2.0) * (math.cos(math.radians(A)) + math.sin(math.radians(A)))
    assert abs(half - P.TILE_SCALE / 2.0) < 1e-9, (half, P.TILE_SCALE / 2.0)
    print('self-check OK: safe angle puts the rolled eval-crop corner on the tile edge')
