"""Two-channel decisive-frame count for the 2-3-c strafe (beyond the template check).

Per marker window (+-1.5 s), stride 3, classify each frame:
  GOLD:    >=2 detections with size>=26 px, inner-source, unclipped (contour channel)
  NETOK:   >=2 detections with size>=26 px, at least 2 total but not GOLD, where the
           non-gold members are clipped/outer (gatenet+PnP channel tolerates these)
Prints per-marker counts and totals.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detect as D   # noqa: E402
import label as L    # noqa: E402

SESS_NAME = sys.argv[1] if len(sys.argv) > 1 else '20260802-010914-vq2-strafe-2-3-c'
SESS = os.path.join(HERE, '..', 'sessions', SESS_NAME)
W, H = 640, 360
PAD = 1.5e9


def clipped(d):
    if d['source'] == 'outer':
        return True
    q = d['quad']
    return bool((q[:, 0] < 3).any() or (q[:, 0] > W - 4).any()
                or (q[:, 1] < 3).any() or (q[:, 1] > H - 4).any())


frames = [r for r in L.load_csv(os.path.join(SESS, 'frames.csv')) if r['file']]
for r in frames:
    r['t'] = float(r['t_recv_wall_ns'])
ft = np.array([f['t'] for f in frames])

markers = []
with open(os.path.join(SESS, 'events.jsonl')) as fh:
    for line in fh:
        e = json.loads(line)
        if e.get('kind') == 'marker':
            markers.append(e)

tot_gold = tot_one = tot_both = 0
print('marker  frames  GOLD  ONECLIP  BOTHCLIP  neither')
for m in markers:
    tm = m['t_wall_ns']
    lo, hi = np.searchsorted(ft, [tm - PAD, tm + PAD])
    idxs = list(range(lo, hi, 3))
    ngold = none_ = nboth = 0
    for i in idxs:
        img = cv2.imread(os.path.join(SESS, 'frames', frames[i]['file']))
        ds = [] if img is None else D.detections(img)
        big = [d for d in ds if d['size_px'] >= 26]
        gold = [d for d in big if not clipped(d) and d['source'] == 'inner']
        if len(gold) >= 2:
            ngold += 1
        elif len(big) >= 2 and len(gold) == 1:
            none_ += 1      # one trusted anchor + one clipped >=26 px partner
        elif len(big) >= 2:
            nboth += 1      # both clipped but big: net-only
    tot_gold += ngold
    tot_one += none_
    tot_both += nboth
    print(f'  {m["index"]}     {len(idxs):4d}   {ngold:4d}   {none_:4d}     {nboth:4d}'
          f'      {len(idxs)-ngold-none_-nboth:4d}')
print(f'\nTOTAL gold {tot_gold}, one-clipped {tot_one}, both-clipped {tot_both}; '
      f'gold-or-net {tot_gold + tot_one + tot_both}')
