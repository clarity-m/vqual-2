"""Render the frames behind the two direction anomalies mapdir.py exposed.

(a) lap 121520's 4-6 rows: the 4-6 vector (22.74 m @ 67.7 deg, dz +0.19) is the SAME
    displacement as 4-5 (22.46 @ 66.4, dz -0.31) -- two static gates cannot share a
    displacement, so one channel's partner is misidentified. Draw every named quad in
    a spread of the 4-6 row frames and look.
(b) the 1-2 conflict: pausing (n=5, bearing 106.9) vs tiebreak 202923 (n=6, bearing
    287.7) -- opposite directions. Draw both sets.

    python3 pilot/perception/mapdir_check.py
"""
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mapvq2 as M       # noqa: E402
import mapedges as ME    # noqa: E402

rows = json.load(open(os.path.join(HERE, 'mapdir_rows.json')))

WANT = [
    ('20260801-121520-vq2-lap-0-15', (4, 6), 6),
    ('20260801-121520-vq2-lap-0-15', (4, 5), 3),
    ('20260801-121520-vq2-lap-0-15', (5, 6), 3),
    ('20260801-144858-vqual2-lap-pausing', (1, 2), 5),
    ('20260801-202923-vq2-1213-1214', (1, 2), 6),
]

bysess = collections.defaultdict(list)
for r in rows:
    for sname, pair, k in WANT:
        if r['session'] == sname and tuple(r['pair']) == pair:
            bysess[sname].append((pair, r))

outs = []
for sname in sorted(bysess):
    res = M.run(os.path.join(M.SESS, sname), verbose=False)
    S = res['S']
    byframe, _c = ME.named_clean_by_frame(S, res['good'], res['ident'])
    sel = []
    per = collections.defaultdict(list)
    for pair, r in bysess[sname]:
        per[pair].append(r)
    for sn, pair, k in WANT:
        if sn != sname:
            continue
        rs = sorted(per[pair], key=lambda r: r['fid'])
        idxs = np.unique(np.linspace(0, len(rs) - 1, min(k, len(rs))).astype(int))
        sel += [(pair, rs[j]) for j in idxs]
    seen = set()
    for pair, r in sel:
        if r['fid'] in seen:
            continue
        seen.add(r['fid'])
        img = cv2.imread(os.path.join(S.path, 'frames', r['file']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        big = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_NEAREST)
        for g, d in byframe.get(r['i'], {}).items():
            q = (np.asarray(d['quad'], np.float32).reshape(4, 2) * 2).astype(np.int32)
            col = (90, 230, 90) if g in pair else (60, 170, 255)
            cv2.polylines(big, [q], True, col, 2, cv2.LINE_AA)
            c0 = q.mean(0)
            cv2.putText(big, 'g%d %.1fm' % (g, d['range_m']),
                        (int(np.clip(c0[0] - 40, 0, 1180)),
                         int(np.clip(c0[1], 16, 700))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.putText(big, '%s  row %d-%d  fid %d  d=%.2f dz=%+.2f' %
                    (sname[-14:], pair[0], pair[1], r['fid'], r['d'],
                     r.get('dz_up', float('nan'))),
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        outs.append(big)

out = os.path.join(HERE, 'vercheck', 'mapdir_anomalies.png')
cv2.imwrite(out, np.vstack(outs))
print('wrote %s (%d frames)' % (out, len(outs)))
