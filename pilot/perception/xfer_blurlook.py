"""Is there any motion blur in this sim's frames at all? Look at the highest-rate frames."""
import numpy as np, cv2, csv, os
PR = 'C:/Users/USER/Projects/vqual-2/pilot/perception/producer_runs/XFER-fast161838/'
SESS = 'C:/Users/USER/Projects/vqual-2/pilot/sessions/20260802-161838/'
VC = 'C:/Users/USER/Projects/vqual-2/pilot/perception/vercheck/'
d = np.loadtxt(PR + 'diag.csv', delimiter=',', skiprows=1)
frames = [r for r in csv.DictReader(open(SESS + 'frames.csv')) if r['file']]
import sys
sys.path.insert(0, 'C:/Users/USER/Projects/vqual-2/pilot')
import producer as P
race = P.load_csv(SESS + 'race.csv')
rr = P.find_last_reset(race, frames)
frames = [f for f in frames if float(f['t_recv_wall_ns']) >= rr[0]]
gy = d[:, 9]
order = np.argsort(gy)
picks = [('lowest |w|', order[0]), ('median |w|', order[len(order)//2]),
         ('p99 |w|', order[int(0.99*len(order))]), ('max |w|', order[-1])]
tiles = []
for lbl, i in picks:
    img = cv2.imread(SESS + 'frames/' + frames[i]['file'])
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    txt = '%s = %.2f rad/s   Laplacian var %.0f   %s' % (lbl, gy[i], lap, frames[i]['file'])
    cv2.putText(img, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    print(txt)
    tiles.append(img)
h, w = tiles[0].shape[:2]
sheet = np.zeros((2*h, 2*w, 3), np.uint8)
for k, t in enumerate(tiles):
    r, c = divmod(k, 2)
    sheet[r*h:(r+1)*h, c*w:(c+1)*w] = t
cv2.imwrite(VC + 'xfer_blur_look.png', sheet)
# population: sharpness vs rate
lv = []
step = max(1, len(frames)//200)
for i in range(0, len(frames), step):
    img = cv2.imread(SESS + 'frames/' + frames[i]['file'], cv2.IMREAD_GRAYSCALE)
    lv.append((gy[i], float(cv2.Laplacian(img, cv2.CV_64F).var())))
lv = np.array(lv)
print('\nframe sharpness (Laplacian variance) vs |gyro|, n=%d sampled frames' % len(lv))
for a, b in [(0, .3), (.3, .7), (.7, 1.1), (1.1, 1.6), (1.6, 9)]:
    m = (lv[:, 0] >= a) & (lv[:, 0] < b)
    if m.sum() > 3:
        print('   |w| %4.1f-%4.1f  n=%3d  Laplacian var median %.0f'
              % (a, b, m.sum(), np.median(lv[m, 1])))
print('corr(|w|, sharpness) = %+.2f' % np.corrcoef(lv[:, 0], lv[:, 1])[0, 1])
print('wrote ' + VC + 'xfer_blur_look.png')
