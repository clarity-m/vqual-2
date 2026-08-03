"""Part 2: model checks + Monte-Carlo occlusion + clearance with uncertainty."""
import json, math
import numpy as np

ROOT = 'C:/Users/USER/Projects/vqual-2/'
OUT = 'C:/Users/USER/AppData/Local/Temp/xfercols/'
ap = json.load(open(ROOT + 'pilot/perception/map_approx.json'))
cv = json.load(open(ROOT + 'pilot/course/course_vq2.json'))
gl = sorted(cv['gates'], key=lambda g: g['id'])
P = np.array([g['position_m'] for g in gl], float)
SX = np.array([g['sigma_x_m'] for g in gl]); SY = np.array([g['sigma_y_m'] for g in gl])
G = sorted(ap['gates'], key=lambda g: g['race_index'])
S = np.array([[g['along_station'], g['across']] for g in G], float)
A = np.column_stack([S[:, 0], S[:, 1], np.ones(17)])

def affine(A_, P_):
    M, *_ = np.linalg.lstsq(A_, P_, rcond=None)
    return M

M = affine(A, P[:, :2])
res = P[:, :2] - A @ M
print('affine   rms residual %.2f m' % np.sqrt((res**2).sum()/17))

# --- homography (sketch could be a perspective photo trace) -----------------
def homography(src, dst):
    rows = []
    for (u, v), (x, y) in zip(src, dst):
        rows.append([u, v, 1, 0, 0, 0, -x*u, -x*v, -x])
        rows.append([0, 0, 0, u, v, 1, -y*u, -y*v, -y])
    _, _, Vt = np.linalg.svd(np.array(rows))
    return Vt[-1].reshape(3, 3)

H = homography(S, P[:, :2])
def applyH(H, pts):
    q = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return q[:, :2] / q[:, 2:3]
rh = P[:, :2] - applyH(H, S)
print('homography rms residual %.2f m  (8 dof vs 6)' % np.sqrt((rh**2).sum()/17))

# --- similarity (isotropic) -------------------------------------------------
def similarity(src, dst):
    s0, d0 = src.mean(0), dst.mean(0)
    X, Y = src - s0, dst - d0
    U, s, Vt = np.linalg.svd(X.T @ Y)
    R = (U @ Vt).T
    sc = s.sum() / (X**2).sum()
    return sc, R, d0 - sc * (R @ s0)
sc, R, t = similarity(S, P[:, :2])
rs = P[:, :2] - (sc * (S @ R.T) + t)
print('similarity rms residual %.2f m, isotropic scale %.2f m/unit (4 dof)' % (np.sqrt((rs**2).sum()/17), sc))

# --- independent along-scale from near-pure-along gate pairs ---------------
print('\n--- pair-wise along scale, |d across| small ---')
vals = []
for i in range(17):
    for j in range(i+1, 17):
        da, dc = S[j,0]-S[i,0], S[j,1]-S[i,1]
        if abs(da) < 0.5 or abs(dc) > 0.15*abs(da):
            continue
        d = np.linalg.norm(P[j,:2]-P[i,:2])
        vals.append((abs(d/da), i, j, abs(da)))
vals.sort()
for v, i, j, da in vals:
    print('   %2d-%-2d  d_along %.2f st  ->  %.2f m/station' % (i, j, da, v))
if vals:
    a = np.array([v[0] for v in vals])
    print('   n=%d  median %.2f  p10 %.2f  p90 %.2f' % (len(a), np.median(a), *np.percentile(a,[10,90])))

# --- Monte Carlo over fit + map error --------------------------------------
u_along, u_across, c0 = M[0], M[1], M[2]
def cols_from(Mx, lo=6, hi=36):
    d = {}
    for N in range(lo, hi):
        d[('L', N)] = Mx[2] + N*Mx[0]
        d[('R', 41-N)] = Mx[2] + N*Mx[0] + Mx[1]
    return d
def spd(a, b, q):
    ab = b - a
    t = float(np.clip(np.dot(q-a, ab)/np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(a + t*ab - q)), t

rng = np.random.default_rng(1)
NMC = 3000
mind = {6: [], 7: [], 8: []}
edge_min = np.zeros((NMC, 16))
for k in range(NMC):
    idx = rng.integers(0, 17, 17)
    Mb = affine(A[idx], P[idx, :2])
    # gate positions jittered by their own map sigma (independent approximation)
    Pj = P[:, :2] + np.column_stack([rng.normal(0, SX), rng.normal(0, SY)])
    C = cols_from(Mb)
    for tgt in (6, 7, 8):
        mind[tgt].append(min(spd(Pj[5], Pj[tgt], v)[0] for v in C.values()))
    for i in range(16):
        edge_min[k, i] = min(spd(Pj[i], Pj[i+1], v)[0] for v in C.values())

print('\n=== MC (n=%d): perpendicular distance from the gate-5 sightline to the nearest column ===' % NMC)
for tgt in (6, 7, 8):
    a = np.array(mind[tgt])
    print('  5->%d : median %.2f m  [p10 %.2f, p90 %.2f]   P(<1.0 m)=%.2f  P(<2.0 m)=%.2f' %
          (tgt, np.median(a), *np.percentile(a, [10, 90]), (a < 1.0).mean(), (a < 2.0).mean()))

print('\n=== MC min clearance per race edge (m) ===')
for i in range(16):
    a = edge_min[:, i]
    print('  %2d-%-2d  median %5.2f  p10 %5.2f  P(<1.5 m)=%.2f  P(<3 m)=%.2f' %
          (i, i+1, np.median(a), np.percentile(a, 10), (a < 1.5).mean(), (a < 3.0).mean()))
allmin = edge_min.min(1)
print('\nwhole-lap worst clearance: median %.2f m  p10 %.2f  P(any edge < 1.5 m)=%.2f' %
      (np.median(allmin), np.percentile(allmin, 10), (allmin < 1.5).mean()))
