"""Render the identity assignments inside each marker window of ONE session -- the LOOK
step for the 2026-08-01 link laps, before any of their rows are believed.

Same discipline as mapedges_markers.py (which refereed the targeting session and caught
its migrated fragments): for each marker, render a few frames across the window with

    orange + range   contour detection passing mapvq2.clean()
    thin orange      contour detection failing clean()
    cyan g<k>        crossing-anchored track identity present in that frame

so the staged pair, the named gates, and the ranges can be judged against the picture.
A pair row is only as good as these labels.

    python3 pilot/perception/vercheck_markers.py <session-dir> [--per 3] [--out name.png]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mapvq2 as M              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--per', type=int, default=3, help='frames rendered per marker window')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    res = M.run(a.session, verbose=False)
    S = res['S']
    good, ident = res['good'], res['ident']
    ident_at = collections.defaultdict(dict)
    for tid, obs in good.items():
        g = ident.get(tid)
        if g is None:
            continue
        for i, d in obs:
            # nearer observation wins a frame collision, as in mapedges.gate_obs_index
            if g not in ident_at[i] or d['range_m'] < ident_at[i][g]['range_m']:
                ident_at[i][g] = d

    pad = M.MARKER_PAD_OVERRIDE_S.get(S.name, M.MARKER_PAD_S)
    tiles = []
    for mi, (t, g_act) in enumerate(S.markers, 1):
        lo = int(np.searchsorted(S.t, t - pad * 1e9))
        hi = int(np.searchsorted(S.t, t + pad * 1e9))
        hi = min(hi, len(S.frames))
        if hi <= lo:
            continue
        sel = sorted({int(k) for k in np.linspace(lo, hi - 1, a.per)})
        for i in sel:
            img = cv2.imread(os.path.join(S.path, 'frames', S.frames[i]['file']),
                             cv2.IMREAD_COLOR)
            if img is None:
                continue
            big = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_NEAREST)

            def draw(quad, col, th, label=None):
                q = np.asarray(quad, np.float32).reshape(-1, 2) * 2.0
                q = np.clip(q, -2000, 4000)
                cv2.polylines(big, [q.astype(np.int32)], True, col, th, cv2.LINE_AA)
                if label:
                    c = q.mean(0)
                    cv2.putText(big, label,
                                (int(c[0]) - 28, int(np.clip(c[1], 14, 700))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

            for d in S.det.get(S.frames[i]['file'], []):
                if M.clean(d):
                    draw(d['quad'], (60, 170, 255), 2, '%.0fm' % d['range_m'])
                else:
                    draw(d['quad'], (40, 90, 140), 1)
            for g, d in sorted(ident_at.get(i, {}).items()):
                q = np.asarray(d['quad'], np.float32).reshape(-1, 2) * 2.0
                c = q.mean(0)
                cv2.putText(big, 'g%d %.0fm' % (g, d['range_m']),
                            (int(c[0]) - 10, int(np.clip(c[1] - 18, 14, 700))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 60), 2, cv2.LINE_AA)
            cv2.putText(big, 'marker %d  active %d  fid %d' % (mi, g_act, S.fid[i]),
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            tiles.append(big)
            print('marker %2d active %2d fid %d: named %s  clean %d'
                  % (mi, g_act, S.fid[i],
                     sorted((g, round(d['range_m'])) for g, d
                            in ident_at.get(i, {}).items()),
                     sum(M.clean(d) for d in S.det.get(S.frames[i]['file'], []))))
    out = a.out or os.path.join(HERE, 'vercheck',
                                'markers_%s.png' % S.name.split('-')[-1])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.vstack(tiles))
    print('wrote', out)


if __name__ == '__main__':
    main()
