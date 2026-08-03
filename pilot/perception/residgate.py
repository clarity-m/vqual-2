"""residgate.py -- derive a SIZE-RELATIVE PnP snap-residual gate from data (Bug 1).

The deployed HUD trust gate is 0.3 px ABSOLUTE (hud.PNP_RESID_GATE, from pnpsnap.py's
population study). PnP reprojection residual scales with apparent size, so on a ~500 px
frame-filling gate a proportionally-excellent 0.87 px pose is refused exactly on final
approach. This script derives the replacement rule

    accept iff resid <= max(ABS_PX, k * size_px)

from labelled instances, evaluating with the DIHEDRAL-CORRECTED residual (producer.py's
pnp_pose scores the residual modulo the square's 4-fold symmetry; the raw residual
rejects exact head-on quads at ~34 px).

Populations, per size bucket:
  * NET quads: deployed trunk colab-v3-rot on eval crops of the autolabels_vq1_v3
    block-split val set (2259 instances), exactly as gatenet.evaluate builds them.
  * CONTOUR quads: detect.py inner quads on the same val frames, matched to GT --
    the producer runs the same residual gate on these, so the rule must not starve them.
  * VQ2 hand labels (labels_gates_all.json): net quads on Claire's quads, incl. the
    big/clipped/upward instances the VQ1 set lacks. GT here is hand-placed (~1-2 px),
    so only >10 px catastrophe classification is trusted, not the <2 px "good" cut.

GOOD pose = mean corner error < 2 px (vs pose-projected GT); CATASTROPHE = > 10 px.
Pick k: keep pnpsnap's catastrophe-catch (>=80% of >10 px rejected) while accepting
good big-gate poses that the 0.3 px absolute gate refuses.

    python3 pilot/perception/residgate.py            # full derivation + tables
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PILOT)

import gatenet as G                    # noqa: E402
import gatenet_conf as GC              # noqa: E402
import detect as D                     # noqa: E402
from producer import pnp_pose          # noqa: E402  (dihedral-corrected residual)

import torch                           # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

TRUNK = os.path.join(HERE, 'gatenet_runs', 'colab-v3-rot', 'best.pt')
HANDLABELS = os.path.join(HERE, 'labels_gates_all.json')
HANDFRAMES = os.path.join(HERE, 'vq2_label', 'frames')

BANDS = ((0, 15), (15, 30), (30, 60), (60, 120), (120, 250), (250, 1e9))


def corrected_resid(quad):
    """Dihedral-corrected snap residual + range via producer.pnp_pose. NaN on failure."""
    pose = pnp_pose(np.ascontiguousarray(D.order_quad(np.asarray(quad, np.float64))))
    if pose is None:
        return float('nan'), False
    return pose['rms'], pose['normal_valid']


def raw_resid(quad):
    """The HUD's current residual: solvePnP + raw rms, NO symmetry correction."""
    q = D.order_quad(np.asarray(quad, np.float64))
    K = np.array([[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1.0]])
    try:
        ok, rvec, tvec = cv2.solvePnP(D.OBJ, q.reshape(4, 1, 2), K, None,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return float('nan')
        proj, _ = cv2.projectPoints(D.OBJ, rvec, tvec, K, None)
        return float(np.sqrt(np.mean((proj.reshape(4, 2) - q) ** 2)))
    except cv2.error:
        return float('nan')


def net_quads_on(items, trunk):
    """-> (quads (N,4,2), err_px mean-corner, size_px, resid_corr, resid_raw)."""
    loader = DataLoader(G.GateCrops(items, False), batch_size=64, shuffle=False,
                        num_workers=0)
    preds, geos, idxs = [], [], []
    with torch.no_grad():
        for x, y, g, i in loader:
            preds.append(trunk(x).numpy())
            geos.append(g.numpy())
            idxs.append(i.numpy())
    preds, geos = np.concatenate(preds), np.concatenate(geos)
    idxs = np.concatenate(idxs)
    n = len(preds)
    err = np.zeros(n)
    size = np.zeros(n)
    rc = np.zeros(n)
    rr = np.zeros(n)
    for j in range(n):
        cx, cy, S = geos[j]
        quad = G.from_norm(preds[j], cx, cy, S).astype(np.float64)
        d = items[idxs[j]]
        gt = np.asarray(d['corners'], np.float64)
        # mean corner error, matched on the canonical ordering both share
        err[j] = float(np.linalg.norm(D.order_quad(quad) - D.order_quad(gt),
                                      axis=1).mean())
        size[j] = d['size_px']
        rc[j], _ = corrected_resid(quad)
        rr[j] = raw_resid(quad)
    return err, size, rc, rr


def contour_quads_on(items):
    """detect.py inner quads on the val FRAMES, matched to GT instances."""
    byframe = {}
    for d in items:
        byframe.setdefault(d['path'], []).append(d)
    err, size, rc = [], [], []
    for path, insts in byframe.items():
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            dets = D.detections(img)
        except Exception:
            continue
        for d in insts:
            gt = np.asarray(d['corners'], np.float64)
            ctr = gt.mean(0)
            best = None
            for det in dets:
                if det['source'] != 'inner':
                    continue
                q = np.asarray(det['quad'], np.float64)
                dist = float(np.linalg.norm(q.mean(0) - ctr))
                if dist < 0.5 * max(d['size_px'], 20.0) and (best is None
                                                             or dist < best[0]):
                    best = (dist, q)
            if best is None:
                continue
            q = best[1]
            err.append(float(np.linalg.norm(D.order_quad(q) - D.order_quad(gt),
                                            axis=1).mean()))
            size.append(d['size_px'])
            r, _ = corrected_resid(q)
            rc.append(r)
    return np.array(err), np.array(size), np.array(rc)


def hand_items():
    raw = json.load(open(HANDLABELS))
    out = []
    for key, insts in sorted(raw.items()):
        path = os.path.join(HANDFRAMES, key)
        for inst in insts:
            if inst.get('unsure'):
                continue                        # a guessed GT cannot referee a residual
            c = np.asarray(inst['corners'], np.float32)
            edges = [float(np.linalg.norm(c[(j + 1) % 4] - c[j])) for j in range(4)]
            out.append({'path': path, 'corners': c, 'clipped': bool(inst['clipped']),
                        'occluded': bool(inst['occluded']), 'size_px': max(edges),
                        'session': 'hand', 'ordinal': 0, 'src': 'hand',
                        'body_rate': 0.0})
    return out


def q(a, p):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else float('nan')


def bucket_table(tag, err, size, resid):
    print(f'\n== {tag}: residual (dihedral-corrected) by size bucket ==')
    print('| bucket | n | n good(<2px) | resid med/p90 good | n bad(>10px) | '
          'resid med/p10 bad |')
    for lo, hi in BANDS:
        m = (size >= lo) & (size < hi)
        g = m & (err < 2.0)
        b = m & (err > 10.0)
        rg, rb = resid[g], resid[b]
        name = f'{lo:.0f}-{hi:.0f}' if hi < 1e8 else f'>={lo:.0f}'
        print(f'| {name} | {m.sum()} | {g.sum()} | {q(rg,50):.3f}/{q(rg,90):.3f} | '
              f'{b.sum()} | {q(rb,50):.3f}/{q(rb,10):.3f} |')


def eval_rule(name, err, size, resid, thr_fn):
    ok = np.isfinite(resid)
    rej = ~ok | (resid > thr_fn(size))
    good = err < 2.0
    bad = err > 10.0
    catch = float(rej[bad].mean()) if bad.any() else float('nan')
    keep_good = float((~rej)[good].mean()) if good.any() else float('nan')
    big = good & (size >= 120)
    keep_big = float((~rej)[big].mean()) if big.any() else float('nan')
    print(f'  {name:34s} catch(>10px) {100*catch:5.1f}%  keep good {100*keep_good:5.1f}%'
          f'  keep good>=120px {100*keep_big:5.1f}%')
    return catch, keep_good, keep_big


def main():
    dev = torch.device('cpu')
    trunk = GC.load_trunk(TRUNK, dev)

    items = G.load_index()
    _, va = G.split_index(items, 'block')
    print(f'val instances (block split): {len(va)}')

    err, size, rc, rr = net_quads_on(va, trunk)
    np.savez(os.path.join(HERE, 'residgate_net_val.npz'),
             err=err, size=size, resid_corr=rc, resid_raw=rr)
    bucket_table('NET quads, VQ1 val', err, size, rc)

    # raw vs corrected on big gates: what the HUD's uncorrected residual does
    big = size >= 120
    print(f'\nraw-vs-corrected residual on >=120 px good poses (n={int((big & (err<2)).sum())}): '
          f'raw med {q(rr[big & (err<2)],50):.2f} p90 {q(rr[big & (err<2)],90):.2f} | '
          f'corrected med {q(rc[big & (err<2)],50):.2f} p90 {q(rc[big & (err<2)],90):.2f}')

    print('\n== rule sweep, NET quads (corrected residual) ==')
    eval_rule('HUD today: 0.3 abs (raw resid)', err, size,
              rr, lambda s: 0.3 + 0 * s)
    eval_rule('0.3 abs (corrected)', err, size, rc, lambda s: 0.3 + 0 * s)
    eval_rule('producer today: max(6, .05*s)', err, size, rc,
              lambda s: np.maximum(6.0, 0.05 * s))
    for k in (0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030):
        eval_rule(f'max(0.3, {k}*size)', err, size, rc,
                  lambda s, k=k: np.maximum(0.3, k * s))

    # per-bucket keep/catch for the shortlist
    print('\n== per-bucket keep(good)/catch(bad), NET quads ==')
    for k in (0.006, 0.008, 0.010, 0.015):
        line = [f'k={k:.3f}']
        for lo, hi in BANDS:
            m = (size >= lo) & (size < hi)
            g = m & (err < 2.0)
            b = m & (err > 10.0)
            thr = np.maximum(0.3, k * size)
            rej = ~np.isfinite(rc) | (rc > thr)
            kg = 100 * (~rej)[g].mean() if g.any() else float('nan')
            cb = 100 * rej[b].mean() if b.any() else float('nan')
            line.append(f'{lo:.0f}+:{kg:.0f}/{cb:.0f}')
        print('  ' + '  '.join(line))

    # contour arm: the producer's residual gate also sees detector quads
    cerr, csize, crc = contour_quads_on(va)
    np.savez(os.path.join(HERE, 'residgate_contour_val.npz'),
             err=cerr, size=csize, resid_corr=crc)
    bucket_table('CONTOUR quads, VQ1 val', cerr, csize, crc)
    print('\n== rule sweep, CONTOUR quads ==')
    eval_rule('producer today: max(6, .05*s)', cerr, csize, crc,
              lambda s: np.maximum(6.0, 0.05 * s))
    for k in (0.006, 0.008, 0.010, 0.015, 0.020, 0.030):
        eval_rule(f'max(0.3, {k}*size)', cerr, csize, crc,
                  lambda s, k=k: np.maximum(0.3, k * s))
        eval_rule(f'max(1.0, {k}*size)', cerr, csize, crc,
                  lambda s, k=k: np.maximum(1.0, k * s))

    # VQ2 hand-label arm: Claire's regime (big/clipped/upward)
    hi = hand_items()
    print(f'\nVQ2 hand instances (unsure excluded): {len(hi)}')
    herr, hsize, hrc, hrr = net_quads_on(hi, trunk)
    np.savez(os.path.join(HERE, 'residgate_net_hand.npz'),
             err=herr, size=hsize, resid_corr=hrc, resid_raw=hrr)
    bucket_table('NET quads, VQ2 hand', herr, hsize, hrc)
    print('\n== rule sweep, VQ2 hand (GT is hand-placed: catastrophe class only) ==')
    eval_rule('HUD today: 0.3 abs (raw resid)', herr, hsize, hrr,
              lambda s: 0.3 + 0 * s)
    for k in (0.006, 0.008, 0.010, 0.015):
        eval_rule(f'max(0.3, {k}*size)', herr, hsize, hrc,
                  lambda s, k=k: np.maximum(0.3, k * s))


if __name__ == '__main__':
    main()
