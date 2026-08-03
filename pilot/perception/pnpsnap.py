"""pnpsnap.py -- does PnP-projecting gatenet's quads onto the manifold of
physically-realizable square projections reduce corner error, and is the
raw-vs-snapped residual a usable confidence signal?

The net regresses 8 free numbers; a true gate corner set has only 6 DOF (pose of a
1.5 m square through the known ideal pinhole). solvePnP + reproject = projection onto
that manifold. Run on the block-split held-out set, exactly as gatenet.evaluate does.

    python3 pilot/perception/pnpsnap.py
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gatenet as G  # noqa: E402

FX = FY = 320.0
CX, CY = 320.0, 180.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float64)
HALF = 0.75  # 1500 mm inner aperture

# IPPE_SQUARE's documented object-point order. For a square, any roll/winding of the
# correspondence is absorbed by a valid 3D rotation (incl. viewing from behind), so
# correspondence cannot silently ruin things HERE -- but this is verified empirically
# below on ground-truth quads, which are exact pinhole projections of the real square.
OBJP = np.array([[-HALF, HALF, 0], [HALF, HALF, 0],
                 [HALF, -HALF, 0], [-HALF, -HALF, 0]], np.float64)

SIZE_BANDS = G.SIZE_BANDS


def snap(quad_px):
    """quad (4,2) full-image px -> (snapped (4,2), rms residual, method str)."""
    imgp = np.ascontiguousarray(quad_px, np.float64).reshape(4, 1, 2)
    rvec = tvec = None
    method = 'ippe'
    try:
        ok, rvec, tvec = cv2.solvePnP(OBJP, imgp, K, None,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            raise cv2.error('ippe returned not-ok')
    except cv2.error:
        method = 'iter'
        try:
            ok, rvec, tvec = cv2.solvePnP(OBJP, imgp, K, None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise cv2.error('iterative returned not-ok')
        except cv2.error:
            return quad_px.copy(), float('nan'), 'fail'
    proj, _ = cv2.projectPoints(OBJP, rvec, tvec, K, None)
    proj = proj.reshape(4, 2)
    rms = float(np.sqrt(np.mean((proj - quad_px) ** 2)))
    return proj, rms, method


def sanity_gt(va, n=60):
    """PnP on GROUND-TRUTH quads of easy unclipped 15-120 px gates. Labels are exact
    projections of the real 1.5 m square through this exact pinhole, so the snapped
    quad must land ~0 px from the label; anything else is a correspondence/K bug."""
    easy = [d for d in va if not d['clipped'] and 15 <= d['size_px'] < 120]
    sel = easy[:: max(1, len(easy) // n)][:n]
    res = []
    for d in sel:
        s, rms, m = snap(np.asarray(d['corners'], np.float64))
        err = float(np.abs(s - d['corners']).max())
        res.append((err, rms, m))
    errs = np.array([r[0] for r in res])
    print(f'[sanity] PnP on {len(sel)} GT quads: max |snapped-label| corner dev '
          f'med {np.median(errs):.4f} px, max {errs.max():.4f} px, '
          f'fallbacks {sum(1 for r in res if r[2] != "ippe")}')
    if errs.max() > 1.0:
        print('[sanity] FAILED -- correspondence or intrinsics wrong. Aborting.')
        sys.exit(1)


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float('nan')


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    items = G.load_index()
    _, va = G.split_index(items, 'block')
    print(f'val instances: {len(va)} (block split)')

    sanity_gt(va)

    model = G.GateNet(1.0).to(dev)
    ck = os.path.join(G.RUNS, 'block', 'best.pt')
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)['model'])
    model.eval()

    loader = DataLoader(G.GateCrops(va, False), batch_size=128, shuffle=False,
                        num_workers=0)
    preds, geos, idxs = [], [], []
    with torch.no_grad():
        for x, y, g, i in loader:
            p = model(x.to(dev)).cpu().numpy()
            preds.append(p)
            geos.append(g.numpy())
            idxs.append(i.numpy())
    preds = np.concatenate(preds)          # (N,8) normalised crop coords
    geos = np.concatenate(geos)            # (N,3) cx, cy, S
    idxs = np.concatenate(idxs)

    N = len(preds)
    raw_q = np.zeros((N, 4, 2))
    snap_q = np.zeros((N, 4, 2))
    resid = np.zeros(N)
    methods = []
    for j in range(N):
        cx, cy, S = geos[j]
        quad = G.from_norm(preds[j], cx, cy, S).astype(np.float64)  # full-image px
        raw_q[j] = quad
        s, r, m = snap(quad)
        snap_q[j], resid[j] = s, r
        methods.append(m)

    n_iter = sum(1 for m in methods if m == 'iter')
    n_fail = sum(1 for m in methods if m == 'fail')
    print(f'PnP: {N - n_iter - n_fail} IPPE_SQUARE, {n_iter} ITERATIVE fallbacks, '
          f'{n_fail} total failures (snapped=raw there)')

    gt = np.stack([np.asarray(va[i]['corners'], np.float64) for i in idxs])
    size = np.array([va[i]['size_px'] for i in idxs])
    clip = np.array([va[i]['clipped'] for i in idxs])

    raw_ce = np.linalg.norm(raw_q - gt, axis=2)    # (N,4)
    snp_ce = np.linalg.norm(snap_q - gt, axis=2)
    raw_inst = raw_ce.mean(1)
    snp_inst = snp_ce.mean(1)

    # ---------------- bucket table ----------------
    rows = []

    def add(name, m):
        if m.sum() == 0:
            return
        rows.append((name, int(m.sum()),
                     q(raw_ce[m].ravel(), 50), q(raw_ce[m].ravel(), 90),
                     q(snp_ce[m].ravel(), 50), q(snp_ce[m].ravel(), 90)))

    add('ALL', np.ones(N, bool))
    for lo, hi in SIZE_BANDS:
        add(f'size {lo:.0f}-{hi:.0f}' if hi < 1e8 else f'size >={lo:.0f}',
            (size >= lo) & (size < hi))
    add('clipped', clip)
    add('unclipped', ~clip)

    print('\n| group | n | RAW med | RAW p90 | SNAP med | SNAP p90 |')
    print('|---|---:|---:|---:|---:|---:|')
    for r in rows:
        print('| %s | %d | %.2f | %.2f | %.2f | %.2f |' % r)

    # per-instance improvement stats
    dlt = snp_inst - raw_inst
    print(f'\nper-instance mean-corner delta (snap - raw): '
          f'med {np.median(dlt):+.3f} px, improved on {np.mean(dlt < 0)*100:.1f}% '
          f'of instances, worsened >0.5 px on {np.mean(dlt > 0.5)*100:.1f}%')

    # ---------------- residual as confidence ----------------
    ok_r = np.isfinite(resid)
    resid_f = np.where(ok_r, resid, np.inf)      # failures rank worst
    rel_resid = resid_f / np.maximum(size, 1e-6)

    from scipy.stats import spearmanr
    for name, key in (('residual px', resid_f), ('residual/size', rel_resid)):
        rho_raw = spearmanr(key[ok_r], raw_inst[ok_r]).statistic
        rho_snp = spearmanr(key[ok_r], snp_inst[ok_r]).statistic
        print(f'\nspearman({name}, err): raw {rho_raw:.3f}  snapped {rho_snp:.3f}')
        order = np.argsort(key)                  # ascending: keep the front
        print('| drop worst | RAW p90 (pooled) | SNAP p90 | RAW inst p90 | SNAP inst p90 |')
        print('|---|---:|---:|---:|---:|')
        for frac in (0.0, 0.05, 0.10, 0.20):
            keep = order[: int(round(N * (1 - frac)))]
            print('| %3.0f%% | %.2f | %.2f | %.2f | %.2f |' % (
                frac * 100, q(raw_ce[keep].ravel(), 90), q(snp_ce[keep].ravel(), 90),
                q(raw_inst[keep], 90), q(snp_inst[keep], 90)))

    good = raw_inst < 2.0
    bad = raw_inst > 10.0
    print(f'\nresidual distribution | true raw err <2 px  (n={good.sum()}): '
          f'med {q(resid_f[good & ok_r], 50):.3f}  p90 {q(resid_f[good & ok_r], 90):.3f}')
    print(f'residual distribution | true raw err >10 px (n={bad.sum()}): '
          f'med {q(resid_f[bad & ok_r], 50):.3f}  p90 {q(resid_f[bad & ok_r], 90):.3f}')
    print(f'relative residual     | <2 px: med {q(rel_resid[good & ok_r], 50):.4f}  '
          f'p90 {q(rel_resid[good & ok_r], 90):.4f}')
    print(f'relative residual     | >10px: med {q(rel_resid[bad & ok_r], 50):.4f}  '
          f'p90 {q(rel_resid[bad & ok_r], 90):.4f}')

    # ---------------- sheet: biggest snap movement ----------------
    top = np.argsort(-resid_f)[:12]
    tiles = []
    for j in top:
        d = va[idxs[j]]
        img = cv2.imread(d['path'], cv2.IMREAD_COLOR)
        if img is None:
            continue
        cx, cy, S = geos[j]
        S = max(S * 1.5, 100.0)
        pad = 60
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                 value=(0, 0, 0))

        def draw(quad, col, t=2):
            p = np.asarray(quad, np.float32) + pad
            if np.isfinite(p).all():
                cv2.polylines(big, [p.astype(np.int32)], True, col, t, cv2.LINE_AA)

        draw(gt[j], (90, 230, 90), 2)          # green truth
        draw(raw_q[j], (230, 90, 230), 2)      # magenta raw
        draw(snap_q[j], (0, 230, 230), 2)      # yellow snapped
        x0, y0 = int(cx - S / 2) + pad, int(cy - S / 2) + pad
        tile = big[max(0, y0):y0 + int(S), max(0, x0):x0 + int(S)]
        if tile.size == 0:
            continue
        tile = cv2.resize(tile, (260, 260), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, 'sz%.0f%s res %.1f raw %.1f snap %.1f' % (
            size[j], ' CLIP' if clip[j] else '', resid[j], raw_inst[j], snp_inst[j]),
            (4, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (259, 259), (70, 70, 70), 1)
        tiles.append(tile)
    cols = 4
    rn = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rn * 260, cols * 260, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 260:(r + 1) * 260, c * 260:(c + 1) * 260] = t
    out = os.path.join(HERE, 'pnpsnap_sheet.png')
    cv2.imwrite(out, sheet)
    print(f'\nwrote {out} ({len(tiles)} tiles: green=truth magenta=raw yellow=snapped)')

    json.dump({'rows': rows, 'n_iter': n_iter, 'n_fail': n_fail},
              open(os.path.join(HERE, 'pnpsnap_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
