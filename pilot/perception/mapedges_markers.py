"""One annotated frame per marker of the targeting session -- the identity referee.

Claire's stated intent: marker 1 = gates 2+3, markers 2-4 = 6+7, 5-8 = 7+8,
9-12 = 9+10, 13-15 = 13+14, both gates fully in frame at each marker. This renders the
frame at each marker instant with every detection labelled:

    orange + range   contour detection passing clean()
    thin orange      contour detection FAILING clean() (with reason omitted)
    green            accepted gatenet measurement (from mapedges_accepted.json)
    cyan g<k>        named-gate identity (crossing-anchored track) present in frame

so the claimed pair, its measured separation, and the sim's own active gate can be
judged against what the picture shows.

    python3 pilot/perception/mapedges_markers.py
"""

from __future__ import annotations

import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mapvq2 as M              # noqa: E402
from mapedges import TARGETING  # noqa: E402

OUT = os.path.join(HERE, 'vercheck', 'targeting_markers.png')


def main():
    res = M.run(TARGETING, verbose=False)
    S = res['S']
    good, ident = res['good'], res['ident']
    ident_at = collections.defaultdict(dict)
    for tid, obs in good.items():
        g = ident.get(tid)
        if g is None:
            continue
        for i, d in obs:
            ident_at[i][g] = d
    try:
        acc = json.load(open(os.path.join(HERE, 'mapedges_accepted.json')))
    except FileNotFoundError:
        acc = []
    acc_by = collections.defaultdict(list)
    for m in acc:
        if m['session'] == S.name:
            acc_by[m['i']].append(m)

    tiles = []
    for mi, (t, g_act) in enumerate(S.markers, 1):
        i = int(np.argmin(np.abs(S.t - t)))
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
                cv2.putText(big, label, (int(c[0]) - 28, int(np.clip(c[1], 14, 700))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        for d in S.det.get(S.frames[i]['file'], []):
            if M.clean(d):
                draw(d['quad'], (60, 170, 255), 2, '%.0fm' % d['range_m'])
            else:
                draw(d['quad'], (40, 90, 140), 1)
        for g, d in sorted(ident_at.get(i, {}).items()):
            q = np.asarray(d['quad'], np.float32).reshape(-1, 2) * 2.0
            c = q.mean(0)
            cv2.putText(big, 'g%d' % g, (int(c[0]) - 10, int(np.clip(c[1] - 18, 14, 700))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 60), 2, cv2.LINE_AA)
        for m in acc_by.get(i, []):
            draw(m['quad'], (90, 230, 90), 2, 'net g%d %.0fm' % (m['g'], m['range_m']))
        cv2.putText(big, 'marker %d  active %d  fid %d' % (mi, g_act, S.fid[i]),
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        tiles.append(big)
        print('marker %2d active %2d fid %d: named %s  clean %d  net %d'
              % (mi, g_act, S.fid[i], sorted(ident_at.get(i, {})),
                 sum(M.clean(d) for d in S.det.get(S.frames[i]['file'], [])),
                 len(acc_by.get(i, []))))
    sheet = np.vstack(tiles)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, sheet)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
