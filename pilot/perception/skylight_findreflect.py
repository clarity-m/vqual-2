"""Find frames where the horizon filter rejects genuine floor REFLECTIONS of lights:
below-horizon bright components that are NOT orange-surrounded (so not decoration),
area >= 40 px. Rank frames by total such area; print the top ones."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import label as L
import skylight as SK

session = sys.argv[1]
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 12
frames = [r for r in L.load_csv(os.path.join(session, 'frames.csv')) if r['file']]
imu = SK.Imu(session)
scores = []
for r in frames[::stride]:
    t = float(r['t_recv_wall_ns'])
    g, _n = imu.gravity(t)
    if g is None:
        continue
    R_lb = SK.level_rotation(g)
    if R_lb is None:
        continue
    img = cv2.imread(os.path.join(session, 'frames', r['file']))
    if img is None:
        continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] < SK.SAT_MAX) & (hsv[:, :, 2] > SK.VAL_MIN)).astype(np.uint8)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    orange = SK._orange_mask(img)
    tot, cnt = 0, 0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 40:
            continue
        # TOP corners: component entirely below horizon = its highest point below margin
        corners = [(x, y), (x + w, y)]
        rl = (SK._rays_cam(corners) @ SK._R_BC.T) @ R_lb.T
        elev = np.degrees(np.arcsin(np.clip(-rl[:, 2], -1, 1)))
        if elev.max() >= -SK.HORIZON_MARGIN_DEG:
            continue
        x0, y0 = max(0, x - 5), max(0, y - 5)
        x1, y1 = min(img.shape[1], x + w + 5), min(img.shape[0], y + h + 5)
        own = (lab[y0:y1, x0:x1] == i)
        ofrac = float(orange[y0:y1, x0:x1][~own].sum()) / max(int((~own).sum()), 1)
        if ofrac > SK.ORANGE_FRAC_MAX:
            continue
        tot += a; cnt += 1
    if cnt:
        scores.append((tot, cnt, r['file']))
scores.sort(reverse=True)
for s in scores[:15]:
    print(s, flush=True)
