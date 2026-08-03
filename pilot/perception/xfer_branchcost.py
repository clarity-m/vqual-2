"""What a wrong mod-90 branch actually costs, in metres, on each of the three legs.

Each contested leg is a BRIDGE in the map graph, so rotating its bearing by +-90 deg
rotates the whole downstream sub-chain rigidly about the upstream gate.
"""
import json
import numpy as np

C = json.load(open('C:/Users/USER/Projects/vqual-2/pilot/course/course_vq2.json'))
P = np.array([g['position_m'] for g in C['gates']] if True
             else None, float)
print('gates', P.shape)
LEGS = [(2, 3), (6, 7), (15, 16)]
for a, b in LEGS:
    down = list(range(b, len(P)))
    piv = P[a, :2]
    for k in (90, -90, 180):
        th = np.radians(k)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Q = (P[down, :2] - piv) @ R.T + piv
        d = np.linalg.norm(Q - P[down, :2], axis=1)
        print('leg %2d-%-2d  branch error %+4d deg -> %2d downstream gates move, '
              'median %6.1f m, max %6.1f m (gate %d)'
              % (a, b, k, len(down), np.median(d), d.max(), down[int(np.argmax(d))]))
    print()
