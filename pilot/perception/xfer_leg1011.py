"""xfer_leg1011.py -- audit the map's longest edge, 10-11 = 33.88 m, which the drag
path-length channel says is impossible (path/chord 0.53 and 0.60 on the two race laps).
Re-detect the frames the edge was measured from and look at what was actually seen.
"""
import os, sys
import numpy as np, cv2
sys.path.insert(0, 'C:/Users/USER/Projects/vqual-2/pilot/perception')
sys.path.insert(0, 'C:/Users/USER/Projects/vqual-2/pilot')
import detect as D

SESS = 'C:/Users/USER/Projects/vqual-2/pilot/sessions/20260801-144858-vqual2-lap-pausing/'
VC = 'C:/Users/USER/Projects/vqual-2/pilot/perception/vercheck/'
FIDS = [32235, 32261, 32290, 32335, 32352, 32403]
tiles = []
for f in FIDS:
    p = SESS + 'frames/%08d.jpg' % f
    img = cv2.imread(p)
    dets = D.detections(img)
    dets = sorted(dets, key=lambda d: -d['size_px'])[:4]
    print('%08d.jpg  %d detections' % (f, len(dets)))
    for d in dets:
        c = np.asarray(d['quad'], float).reshape(4, 2).mean(0)
        print('    size %6.2f px  range %6.2f m  src %-6s  centre (%.0f,%.0f)'
              % (d['size_px'], d['range_m'], d['source'], c[0], c[1]))
        cv2.polylines(img, [np.asarray(d['quad'], np.int32).reshape(-1, 1, 2)], True,
                      (0, 255, 0), 1)
        cv2.putText(img, '%.0fpx %.0fm' % (d['size_px'], d['range_m']),
                    (int(c[0]) - 30, int(c[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, '%08d.jpg  act=10, gate 10 crossed 1-6 s later' % f, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(img)
h, w = tiles[0].shape[:2]
sheet = np.zeros((2 * h, 3 * w, 3), np.uint8)
for k, t in enumerate(tiles):
    r, c = divmod(k, 3)
    sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
cv2.imwrite(VC + 'xfer_leg1011.png', sheet)
print('wrote ' + VC + 'xfer_leg1011.png')


# ---- alternative measurement: gate 10 paired with the NEXT-nearest gate, not the
# third-nearest. act=10 throughout, so the largest/nearest detection IS gate 10.
print()
print('--- separation from gate 10 to each further detection, per frame ---')
SESS2 = SESS
import glob
fids = list(range(32235, 32410))
rows = []
for f in fids:
    p = SESS2 + 'frames/%08d.jpg' % f
    if not os.path.exists(p):
        continue
    img = cv2.imread(p)
    dets = [d for d in D.detections(img) if d['size_px'] >= 14.0
            and d.get('pos_body') is not None]
    if len(dets) < 2:
        continue
    dets = sorted(dets, key=lambda d: d['range_m'])
    near = dets[0]
    if near['range_m'] > 20.0:
        continue                      # gate 10 must be the near one
    pa = np.asarray(near['pos_body'], float)
    seps = [float(np.linalg.norm(np.asarray(d['pos_body'], float) - pa))
            for d in dets[1:]]
    rows.append((f, near['range_m'], near['size_px'], seps))
print('n frames with gate 10 near and at least one further gate: %d' % len(rows))
s1 = np.array([r[3][0] for r in rows])
print('separation gate10 -> NEXT-nearest detection: median %.2f m  MAD %.2f  '
      'p10 %.2f  p90 %.2f  n=%d'
      % (np.median(s1), np.median(np.abs(s1 - np.median(s1))),
         np.percentile(s1, 10), np.percentile(s1, 90), len(s1)))
s2 = np.array([r[3][1] for r in rows if len(r[3]) > 1])
if len(s2):
    print('separation gate10 -> SECOND-next detection:  median %.2f m  MAD %.2f  n=%d'
          % (np.median(s2), np.median(np.abs(s2 - np.median(s2))), len(s2)))
s3 = np.array([r[3][2] for r in rows if len(r[3]) > 2])
if len(s3):
    print('separation gate10 -> THIRD-next detection:   median %.2f m  MAD %.2f  n=%d'
          % (np.median(s3), np.median(np.abs(s3 - np.median(s3))), len(s3)))
print('(map: 10-11 = 33.88, 11-12 = 19.67, 12-13 = 13.45)')
