"""Exploratory: what does the S<50 & V>200 mask contain, frame by frame?

Dumps per-component stats and a render for a handful of frames so the light-quad
detector's thresholds are chosen from data rather than assumed.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.abspath(os.path.join(HERE, '..', 'sessions'))
OUT = os.path.join(HERE, 'vercheck')
os.makedirs(OUT, exist_ok=True)

PICKS = [
    ('20260801-144858-vqual2-lap-pausing', '00024877.jpg'),
    ('20260801-144858-vqual2-lap-pausing', '00029000.jpg'),
    ('20260801-144858-vqual2-lap-pausing', '00032000.jpg'),
    ('20260801-144858-vqual2-lap-pausing', '00026000.jpg'),
    ('20260801-121520-vq2-lap-0-15', '00101000.jpg'),
    ('20260801-121520-vq2-lap-0-15', '00104000.jpg'),
]

for sess, fname in PICKS:
    p = os.path.join(SESS, sess, 'frames', fname)
    img = cv2.imread(p)
    if img is None:
        print('MISSING', p)
        continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 200)).astype(np.uint8) * 255
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    vis = img.copy()
    vis[m > 0] = (0, 255, 0)
    big = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 15:
            continue
        fill = a / float(w * h)
        big.append((a, w, h, fill, x, y))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    big.sort(reverse=True)
    print(f'{sess}/{fname}: mask {100*np.count_nonzero(m)/m.size:.2f}% of frame, '
          f'{len(big)} comps >=15px')
    for a, w, h, fill, x, y in big[:14]:
        print(f'   area {a:5d}  {w:3d}x{h:3d}  fill {fill:.2f}  at ({x},{y})')
    cv2.imwrite(os.path.join(OUT, f'skylight_explore_{sess[:15]}_{fname[:-4]}.png'), vis)
print('renders in', OUT)
