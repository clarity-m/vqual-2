"""Side-by-side frames where gatenet and the contour detector DISAGREE.

The aggregate tables in TRAINING.md say the net wins on coverage and precision, but an
aggregate cannot show WHY, and it cannot show the cases where the net is the one that is
wrong. This renders the disagreements so they can be judged by eye, which is how three of
this project's real bugs were actually found.

Four categories, chosen so the selection is not flattering:

    net-wins      both fired, net closer by the largest margin
    DETECTOR-WINS both fired, detector closer -- included deliberately, and first in the
                  report, because a comparison that only shows its own side winning is
                  advocacy rather than evidence
    detector-miss detector found nothing; the net's answer is all there is, so the question
                  is whether that answer is usable, not merely present
    both-bad      both far from truth -- the cases neither method handles

Green = ground truth, magenta = gatenet, orange = detect.py.

    python3 pilot/perception/compare_net_vs_detector.py --per-group 4 --out sheet.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gatenet as G  # noqa: E402
import detect as D  # noqa: E402

GT_COL = (90, 230, 90)
NET_COL = (230, 90, 230)
DET_COL = (60, 170, 255)


def net_predict(model, dev, img, corners):
    """Run the net on one instance, exactly as evaluate() does — same crop geometry."""
    cx, cy, S = G.crop_params(corners)
    crop = G.make_crop(img, cx, cy, S)
    x = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(dev)
    with torch.no_grad():
        uv = model(x).cpu().numpy().reshape(4, 2)
    return G.from_norm(uv, cx, cy, S), (cx, cy, S)


def detector_match(img, corners, size_px):
    """detect.py's best match under the SAME rule the baseline uses: within 2x apparent
    size and one gate width of the labelled centre. Anything else is a miss, because
    scoring a different gate metres away would measure recall, not accuracy."""
    dets = D.detections(img)
    if not dets:
        return None
    c = np.asarray(corners, np.float32).mean(0)
    best, bd = None, 1e18
    for d in dets:
        dist = float(np.linalg.norm(np.asarray(d['centre'], np.float32) - c))
        ratio = d['size_px'] / max(size_px, 1e-6)
        if dist > size_px or ratio > 2.0 or ratio < 0.5:
            continue
        if dist < bd:
            best, bd = d, dist
    return best


def draw(img, quad, col, label=None, thick=1):
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    ok = np.isfinite(q).all()
    if not ok:
        return
    cv2.polylines(img, [q.astype(np.int32)], True, col, thick, cv2.LINE_AA)
    if label:
        p = q[np.argmin(q[:, 1])]
        cv2.putText(img, label, (int(p[0]) + 3, max(10, int(p[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='block', choices=['block', 'session'])
    ap.add_argument('--tag', default='block')
    ap.add_argument('--ckpt', default='best')
    ap.add_argument('--per-group', type=int, default=4)
    ap.add_argument('--zoom', type=float, default=2.6)
    ap.add_argument('--out', default='compare_sheet.png')
    args = ap.parse_args()

    dev = torch.device('cpu')
    model = G.GateNet(1.0).to(dev)
    ck = os.path.join(G.RUNS, args.tag, f'{args.ckpt}.pt')
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)['model'])
    model.eval()

    items = G.load_index()
    _, va = G.split_index(items, args.split)
    print(f'{len(va)} held-out instances ({args.split} split)')

    rows = []
    by_frame = {}
    for k, d in enumerate(va):
        by_frame.setdefault(d['path'], []).append(k)
    for p in sorted(by_frame):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        for k in by_frame[p]:
            it = va[k]
            gt = np.asarray(it['corners'], np.float32)
            pred, geo = net_predict(model, dev, img, gt)
            ne = float(np.median(np.linalg.norm(pred - gt, axis=1)))
            dm = detector_match(img, gt, it['size_px'])
            if dm is None:
                de = np.inf
            else:
                # detect.py gives a quad; compare centres, the only quantity both produce
                de = float(np.linalg.norm(np.asarray(dm['centre'], np.float32) - gt.mean(0)))
            nc = float(np.linalg.norm(pred.mean(0) - gt.mean(0)))
            rows.append({'k': k, 'path': p, 'gt': gt, 'pred': pred, 'det': dm,
                         'net_corner': ne, 'net_centre': nc, 'det_centre': de,
                         'size': it['size_px'], 'clipped': bool(it.get('clipped'))})

    fired = [r for r in rows if np.isfinite(r['det_centre'])]
    missed = [r for r in rows if not np.isfinite(r['det_centre'])]
    print(f'detector fired on {len(fired)}/{len(rows)}  ({100.0*len(fired)/max(1,len(rows)):.0f}%)')

    groups = {
        'DETECTOR-WINS': sorted([r for r in fired if r['det_centre'] < r['net_centre']],
                                key=lambda r: r['net_centre'] - r['det_centre'])[::-1],
        'net-wins': sorted(fired, key=lambda r: r['det_centre'] - r['net_centre'])[::-1],
        'detector-miss': sorted(missed, key=lambda r: -r['size']),
        'both-bad': sorted([r for r in fired
                            if r['net_centre'] > 5 and r['det_centre'] > 5],
                           key=lambda r: -(r['net_centre'] + r['det_centre'])),
    }

    tiles, caps = [], []
    for name, rs in groups.items():
        print(f'\n{name}: {len(rs)} instances')
        for r in rs[:args.per_group]:
            print('   size %6.1f px  clipped %-5s  net centre %6.2f  det centre %s   %s'
                  % (r['size'], r['clipped'], r['net_centre'],
                     'MISS' if not np.isfinite(r['det_centre']) else '%6.2f' % r['det_centre'],
                     os.path.basename(r['path'])))
            img = cv2.imread(r['path'], cv2.IMREAD_COLOR)
            cx, cy, S = G.crop_params(r['gt'])
            S = max(S * 1.35, 90.0)
            x0, y0 = int(cx - S / 2), int(cy - S / 2)
            pad = 40
            big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            sh = lambda q: np.asarray(q, np.float32) + pad  # noqa: E731
            draw(big, sh(r['gt']), GT_COL, 'truth', 2)
            draw(big, sh(r['pred']), NET_COL, 'net %.1f' % r['net_centre'], 2)
            if r['det'] is not None and r['det'].get('quad') is not None:
                draw(big, sh(np.asarray(r['det']['quad']).reshape(-1, 2)), DET_COL,
                     'det %.1f' % r['det_centre'], 2)
            tile = big[max(0, y0 + pad):y0 + pad + int(S), max(0, x0 + pad):x0 + pad + int(S)]
            if tile.size == 0:
                continue
            tile = cv2.resize(tile, (240, 240), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(tile, (0, 0), (239, 239), (70, 70, 70), 1)
            cv2.putText(tile, '%s %.0fpx%s' % (name, r['size'], ' CLIP' if r['clipped'] else ''),
                        (5, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
            caps.append(name)

    if not tiles:
        print('nothing to render')
        return
    cols = args.per_group
    rowsn = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rowsn * 240, cols * 240, 3), np.uint8)
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sheet[rr * 240:(rr + 1) * 240, cc * 240:(cc + 1) * 240] = t
    cv2.imwrite(args.out, sheet)
    print(f'\nwrote {args.out}  ({len(tiles)} tiles)')


if __name__ == '__main__':
    main()
