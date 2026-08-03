"""rotaug_check.py -- data-backed verification of gatenet_rotaug.py. CPU, ~1-2 min.

Runs five things and prints PASS/FAIL for each:
  1. synthetic pixel round-trip: the computed target lands on a painted quad after the roll;
  2. deg=0 identity: RotAugCrops reduces to the cached train path (bit-exact warp);
  3. aperture safety: the roll never paints black onto an unclipped gate's aperture;
  4. a 3x4 sheet of augmented tiles with the rolled GT corners drawn -> vercheck/rotaug_check.png
  5. a 2-epoch smoke on a small subset (loss decreases) + per-sample CPU overhead (<2 ms).

    python3 pilot/perception/rotaug_check.py
"""
from __future__ import annotations

import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gatenet as G
import packcrops as P
import gatenet_rotaug as R

CACHE = os.path.join(HERE, 'crop_cache')
VERCHECK = os.path.join(HERE, 'vercheck')
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
cv2.setNumThreads(0)


def check1_synthetic():
    """Exact-projection round-trip through PIXELS: paint a bright quad on a synthetic tile at
    known corners, run the real roll_transform (image + target), and confirm the computed
    target polygon lands on the painted quad in the output crop (IoU ~ 1)."""
    res, tres, tscale = 160, 336, 2.10
    rng = random.Random(1)
    worst_iou = 1.0
    for _ in range(400):
        cx, cy = rng.uniform(120, 520), rng.uniform(90, 270)
        S = rng.uniform(50, 180)
        wx, wy = S * rng.uniform(.30, .48), S * rng.uniform(.30, .48)   # aperture well inside
        corners = G.canon(np.array([[cx - wx, cy - wy], [cx + wx, cy - wy],
                                    [cx + wx, cy + wy], [cx - wx, cy + wy]], np.float32)
                          ).astype(np.float32)
        a = res / S
        tile = np.zeros((tres, tres, 3), np.uint8)
        tp = np.stack([(corners[:, 0] - cx) * a + tres / 2, (corners[:, 1] - cy) * a + tres / 2], 1)
        cv2.fillPoly(tile, [np.round(tp).astype(np.int32).reshape(-1, 1, 2)], (255, 255, 255))
        jx, jy, jS = cx + rng.uniform(-.1, .1) * S, cy + rng.uniform(-.1, .1) * S, \
            S * rng.uniform(1.5, 1.9)
        deg = rng.uniform(-R.SAFE_ANGLE_DEG, R.SAFE_ANGLE_DEG)
        M, aa = R.roll_transform(cx, cy, S, jx, jy, jS, deg, res, tres, tscale, True)
        crop = cv2.warpAffine(tile, M, (res, res), flags=cv2.INTER_LINEAR)
        tile_px = np.stack([(corners[:, 0] - cx) * aa + tres / 2,
                            (corners[:, 1] - cy) * aa + tres / 2], 1)
        crop_px = np.c_[tile_px, np.ones(4, np.float32)] @ M.T
        tgt = G.canon((crop_px - res / 2) / (res / 2))
        qpx = (np.asarray(tgt) + 1.0) * (res / 2)
        bright = crop.sum(2) > 200
        tgtmask = np.zeros((res, res), np.uint8)
        cv2.fillPoly(tgtmask, [np.round(qpx).astype(np.int32).reshape(-1, 1, 2)], 1)
        tm = tgtmask.astype(bool)
        iou = float((bright & tm).sum()) / max(int((bright | tm).sum()), 1)
        worst_iou = min(worst_iou, iou)
    ok = worst_iou > 0.95
    print(f'[1] synthetic pixel round-trip: worst target-vs-painted IoU over 400 draws = '
          f'{worst_iou:.4f} -> {"PASS" if ok else "FAIL"}')
    return ok


def check2_identity(items, meta):
    """deg=0 must reproduce the cached train warp exactly: same jitter draw => identity roll
    => M == make_crop's affine => bit-identical crop, and the target equals to_norm (canon is
    idempotent on the already-canonical label)."""
    base = P.CachedGateCrops(items, meta['_path'], meta, True, hflip=False)
    worst_px, worst_t = 0, 0.0
    for i in range(64):
        seed = (i * 2654435761 + 12345) & 0xFFFFFFFF
        d = base.items[i]
        cx, cy, S = (float(v) for v in d['geo'])
        corners = np.asarray(d['corners'], np.float32)
        # baseline cached train path
        jx, jy, jS = G.crop_params(corners, random.Random(seed))
        a = base.res / S
        crop_b = G.make_crop(np.asarray(base.tiles[d['slot']]),
                             (jx - cx) * a + base.tres / 2.0,
                             (jy - cy) * a + base.tres / 2.0, jS * a, base.res)
        tgt_b = G.to_norm(corners, jx, jy, jS)
        # rotaug path, same jitter, deg forced to 0
        jx2, jy2, jS2 = G.crop_params(corners, random.Random(seed))
        M, aa = R.roll_transform(cx, cy, S, jx2, jy2, jS2, 0.0, base.res, base.tres,
                                 base.tscale, base.exact)
        crop_a = cv2.warpAffine(np.asarray(base.tiles[d['slot']]), M, (base.res, base.res),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        tile_px = np.stack([(corners[:, 0] - cx) * aa + base.tres / 2,
                            (corners[:, 1] - cy) * aa + base.tres / 2], 1)
        tgt_a = G.canon((np.c_[tile_px, np.ones(4, np.float32)] @ M.T - base.res / 2)
                        / (base.res / 2))
        worst_px = max(worst_px, int(np.abs(crop_a.astype(int) - crop_b.astype(int)).max()))
        worst_t = max(worst_t, float(np.abs(np.asarray(tgt_a) - tgt_b).max()))
    ok = worst_px == 0 and worst_t < 1e-4
    print(f'[2] deg=0 identity vs cached train: max pixel diff {worst_px}, max target diff '
          f'{worst_t:.2e} -> {"PASS" if ok else "FAIL"}')
    return ok


def _render(base, i, deg):
    """Render one train crop at a chosen roll, reusing a FIXED jitter draw so deg is the only
    change, plus the GT-corner polygon in crop pixels. Uses the module's own transform."""
    d = base.items[i]
    cx, cy, S = (float(v) for v in d['geo'])
    corners = np.asarray(d['corners'], np.float32)
    jx, jy, jS = G.crop_params(corners, random.Random((i * 999 + 7) & 0xFFFFFFFF))
    M, a = R.roll_transform(cx, cy, S, jx, jy, jS, deg, base.res, base.tres,
                            base.tscale, base.exact)
    crop = cv2.warpAffine(np.asarray(base.tiles[d['slot']]), M, (base.res, base.res),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    tile_px = np.stack([(corners[:, 0] - cx) * a + base.tres / 2,
                        (corners[:, 1] - cy) * a + base.tres / 2], 1)
    tgt = G.canon((np.c_[tile_px, np.ones(4, np.float32)] @ M.T - base.res / 2) / (base.res / 2))
    quad = (np.asarray(tgt).reshape(4, 2) + 1.0) * (base.res / 2.0)
    return crop, quad


def check3_border(items, meta):
    """The claim to verify: the TILE HEADROOM (safe angle) means rolling never pulls
    MISSING-DATA black (from beyond the tile edge) onto the gate aperture.

    Black on a rolled crop has two sources: (a) OFF-SENSOR void already baked into the tile
    (gates near the frame edge -- the camera really saw black there, and it is the same
    BORDER_CONSTANT the net already trains on across 46% clipped crops), and (b) beyond-tile
    MISSING data, which the headroom is supposed to prevent. They are told apart by whether the
    rolled sample coordinate falls inside the tile raster. The PASS criterion is on (b) only,
    over unclipped gates; (a) is characterised but not failed, because it is in-distribution."""
    base = P.CachedGateCrops(items, meta['_path'], meta, True, hflip=False)
    res, tres = base.res, base.tres
    cand = [k for k, d in enumerate(items) if not d['clipped'] and 15 <= d['size_px'] <= 120]
    idx = [cand[int(k)] for k in np.linspace(0, len(cand) - 1, 400)]
    # crop-pixel grid, to test where each rolled sample lands in the tile
    ys, xs = np.mgrid[0:res, 0:res].astype(np.float32)
    hom = np.stack([xs.ravel(), ys.ravel(), np.ones(res * res, np.float32)], 1)  # (res*res,3)
    beyond_hits, offsensor_frac, worst_beyond = 0, [], 0.0
    for i in idx:
        d = base.items[i]
        cx, cy, S = (float(v) for v in d['geo'])
        corners = np.asarray(d['corners'], np.float32)
        jx, jy, jS = G.crop_params(corners, random.Random((i * 999 + 7) & 0xFFFFFFFF))
        M, a = R.roll_transform(cx, cy, S, jx, jy, jS, R.SAFE_ANGLE_DEG, res, tres,
                                base.tscale, base.exact)
        Minv = cv2.invertAffineTransform(M)                     # crop px -> tile px (source)
        src = hom @ Minv.T                                      # where each crop px samples
        outside = ((src[:, 0] < 0) | (src[:, 0] > tres - 1) |
                   (src[:, 1] < 0) | (src[:, 1] > tres - 1)).reshape(res, res)
        crop_r, quad = _render(base, i, R.SAFE_ANGLE_DEG)
        black_r = crop_r.sum(2) == 0
        ap = np.zeros((res, res), np.uint8)
        cv2.fillPoly(ap, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)], 1)
        apb = ap.astype(bool)
        ap_px = int(apb.sum())
        if not ap_px:
            continue
        beyond = float((black_r & outside & apb).sum()) / ap_px       # (b) missing data
        offs = float((black_r & ~outside & apb).sum()) / ap_px        # (a) off-sensor void
        offsensor_frac.append(offs)
        worst_beyond = max(worst_beyond, beyond)
        beyond_hits += beyond > 0.001
    ok = beyond_hits == 0
    off = np.array(offsensor_frac)
    print(f'[3] aperture safety over {len(idx)} unclipped gates:')
    print(f'      (b) beyond-tile MISSING-data black on aperture: {beyond_hits} gates >0.1% '
          f'(worst {100*worst_beyond:.4f}%) -> {"PASS" if ok else "FAIL"}')
    print(f'      (a) in-tile black on aperture (dark hangar / off-sensor -- REAL content '
          f'from inside the tile, present un-rolled too): '
          f'median {100*np.median(off):.3f}%  p90 {100*np.percentile(off, 90):.3f}%  '
          f'max {100*off.max():.2f}%')
    return ok


def check4_sheet(items, meta):
    """3x4 sheet: augmented tile + rotated GT corners (green) + corner0 (yellow). Prefer
    unclipped mid-size gates so the aperture is visible and the winding is easy to eyeball."""
    base = P.CachedGateCrops(items, meta['_path'], meta, True, hflip=False)
    aug = R.RotAugCrops(base, R.SAFE_ANGLE_DEG)
    cand = [k for k, d in enumerate(items)
            if not d['clipped'] and 25 <= d['size_px'] <= 90]
    sel = [cand[int(k)] for k in np.linspace(0, len(cand) - 1, 12)]
    tiles = []
    for k in sel:
        x, y, g, _ = aug[k]
        img = ((x * 0.25 + 0.45) * 255).clamp(0, 255).byte().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(img)
        q = (y.numpy().reshape(4, 2) + 1.0) * (base.res / 2.0)
        cv2.polylines(img, [q.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        cv2.circle(img, tuple(q[0].astype(int)), 4, (0, 255, 255), -1)
        cv2.putText(img, f"{items[k]['size_px']:.0f}px", (3, base.res - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(img)
    cols, side = 4, base.res
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * side, cols * side, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * side:(r + 1) * side, c * side:(c + 1) * side] = t
    os.makedirs(VERCHECK, exist_ok=True)
    out = os.path.join(VERCHECK, 'rotaug_check.png')
    cv2.imwrite(out, sheet)
    print(f'[4] wrote {out} ({len(tiles)} tiles)')
    return True


def check5_smoke(items, meta):
    """2-epoch train on a small subset: train loss must fall. Plus per-sample CPU overhead
    of the wrapper vs the plain cached train dataset."""
    tr = items[:640]
    base = P.CachedGateCrops(tr, meta['_path'], meta, True, hflip=False)
    aug = R.RotAugCrops(base, R.SAFE_ANGLE_DEG)

    # timing: wrapper vs base, single-process, warm
    for _ in range(10):
        base[_]; aug[_]
    t = time.perf_counter()
    for i in range(200):
        base[i]
    tb = (time.perf_counter() - t) / 200
    t = time.perf_counter()
    for i in range(200):
        aug[i]
    ta = (time.perf_counter() - t) / 200
    over = (ta - tb) * 1000
    ok_time = over < 2.0
    print(f'[5a] per-sample CPU: base {tb*1000:.2f} ms, rotaug {ta*1000:.2f} ms, '
          f'overhead {over:+.2f} ms -> {"PASS" if ok_time else "FAIL"} (<2 ms)')

    dev = torch.device('cpu')
    ld = DataLoader(aug, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    model = G.GateNet(1.0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for ep in range(2):
        model.train()
        tot, n = 0.0, 0
        for x, ytg, g, _ in ld:
            loss = F.smooth_l1_loss(model(x), ytg, beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(ytg)
            n += len(ytg)
        losses.append(tot / n)
        print(f'[5b] epoch {ep}: train loss {losses[-1]:.5f}')
    ok_loss = losses[1] < losses[0]
    print(f'[5b] loss {"decreased" if ok_loss else "did NOT decrease"} '
          f'({losses[0]:.5f} -> {losses[1]:.5f}) -> {"PASS" if ok_loss else "FAIL"}')
    return ok_time and ok_loss


def main():
    print(f'SAFE_ANGLE_DEG = {R.SAFE_ANGLE_DEG:.4f} deg  '
          f'(TILE_SCALE {P.TILE_SCALE} / MARGIN {G.MARGIN})\n')
    items, tiles, meta = P.load_cache(CACHE, strict=False)   # local cache predates v3 relabel
    print(f'cache: {len(items)} instances, tile {meta["tile_res"]} '
          f'eval_exact={meta["eval_exact"]}\n')
    results = [
        check1_synthetic(),
        check2_identity(items, meta),
        check3_border(items, meta),
        check4_sheet(items, meta),
        check5_smoke(items, meta),
    ]
    print(f'\n=== {"ALL PASS" if all(results) else "FAILURES PRESENT"} ===')


if __name__ == '__main__':
    main()
