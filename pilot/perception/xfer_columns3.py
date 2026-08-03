"""Part 3 (corrected + vectorised): column rows exist only for station 1..20 (left)
and 21..40 (right, at along = 41-N).  Model checks, MC occlusion, MC clearance."""
import json, math, sys
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
print('affine rms residual %.2f m' % np.sqrt((res**2).sum()/17))

def homography(src, dst):
    rows = []
    for (u, v), (x, y) in zip(src, dst):
        rows += [[u, v, 1, 0, 0, 0, -x*u, -x*v, -x], [0, 0, 0, u, v, 1, -y*u, -y*v, -y]]
    _, _, Vt = np.linalg.svd(np.array(rows)); return Vt[-1].reshape(3, 3)
H = homography(S, P[:, :2])
q = np.column_stack([S, np.ones(17)]) @ H.T
rh = P[:, :2] - q[:, :2]/q[:, 2:3]
print('homography rms residual %.2f m (8 dof vs 6)' % np.sqrt((rh**2).sum()/17))

s0, d0 = S.mean(0), P[:, :2].mean(0)
X, Y = S - s0, P[:, :2] - d0
U, sv, Vt = np.linalg.svd(X.T @ Y); R = (U @ Vt).T; sc = sv.sum()/(X**2).sum()
rs = P[:, :2] - (sc*(S @ R.T) + (d0 - sc*(R @ s0)))
print('similarity rms residual %.2f m, isotropic scale %.2f m/unit (4 dof)' % (np.sqrt((rs**2).sum()/17), sc))

print('\n--- pair-wise along scale, near-pure-along pairs ---')
vals = []
for i in range(17):
    for j in range(i+1, 17):
        da, dc = S[j,0]-S[i,0], S[j,1]-S[i,1]
        if abs(da) < 0.4 or abs(dc) > 0.2*abs(da): continue
        vals.append((abs(np.linalg.norm(P[j,:2]-P[i,:2])/da), i, j, abs(da), abs(dc)))
vals.sort()
for v,i,j,da,dc in vals:
    print('   %2d-%-2d  d_along %.2f st, d_across %.2f  ->  %.2f m/station' % (i,j,da,dc,v))
if vals:
    a = np.array([v[0] for v in vals])
    print('   n=%d  median %.2f  p10 %.2f  p90 %.2f' % (len(a), np.median(a), *np.percentile(a,[10,90])))

NL, NR = np.arange(1, 21), np.arange(21, 41)
def cols_from(Mx):
    left = Mx[2] + NL[:, None]*Mx[0]
    right = Mx[2] + (41-NR)[:, None]*Mx[0] + Mx[1]
    return np.vstack([left, right])
C0 = cols_from(M)
names = [('L', n) for n in NL] + [('R', n) for n in NR]

print('\n-- modelled columns (m) --')
for k, (r, n) in enumerate(names):
    if n % 2 == 0 or n in (11, 21):
        print('  %s%02d x=%8.2f y=%7.2f' % (r, n, C0[k,0], C0[k,1]))

def seg_dists(a, b, Q):
    ab = b - a
    t = np.clip(((Q - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(a + t[:, None]*ab - Q, axis=1), t

print('\n=== nominal: gate-5 sightlines ===')
for tgt in (6, 7, 8):
    d, t = seg_dists(P[5,:2], P[tgt,:2], C0)
    o = np.argsort(d)[:4]
    print('  5->%-2d len %6.2f m :' % (tgt, np.linalg.norm(P[tgt,:2]-P[5,:2])),
          ', '.join('%s%02d %.2fm(t=%.2f, range %.1f m, off-axis %.1f deg)' %
                    (names[i][0], names[i][1], d[i], t[i],
                     t[i]*np.linalg.norm(P[tgt,:2]-P[5,:2]),
                     math.degrees(math.atan2(d[i], max(t[i]*np.linalg.norm(P[tgt,:2]-P[5,:2]), 1e-6)))) for i in o))

print('\n=== nominal: min clearance per race edge ===')
for i in range(16):
    d, t = seg_dists(P[i,:2], P[i+1,:2], C0)
    k = int(np.argmin(d))
    print('  %2d-%-2d len %6.2f  clearance %5.2f m to %s%02d (t=%.2f)' %
          (i, i+1, np.linalg.norm(P[i+1,:2]-P[i,:2]), d[k], names[k][0], names[k][1], t[k]))

rng = np.random.default_rng(1)
NMC = 2000
mind = {6: np.zeros(NMC), 7: np.zeros(NMC), 8: np.zeros(NMC)}
edge_min = np.zeros((NMC, 16))
for k in range(NMC):
    idx = rng.integers(0, 17, 17)
    Mb = affine(A[idx], P[idx, :2])
    Pj = P[:, :2] + np.column_stack([rng.normal(0, SX), rng.normal(0, SY)])
    C = cols_from(Mb)
    for tgt in (6, 7, 8):
        mind[tgt][k] = seg_dists(Pj[5], Pj[tgt], C)[0].min()
    for i in range(16):
        edge_min[k, i] = seg_dists(Pj[i], Pj[i+1], C)[0].min()

print('\n=== MC n=%d : perpendicular distance from gate-5 sightline to nearest column ===' % NMC)
for tgt in (6, 7, 8):
    a = mind[tgt]
    print('  5->%d  median %.2f  [p10 %.2f, p90 %.2f]  P(<1.0)=%.2f  P(<2.0)=%.2f' %
          (tgt, np.median(a), *np.percentile(a,[10,90]), (a<1).mean(), (a<2).mean()))

print('\n=== MC min clearance per race edge ===')
for i in range(16):
    a = edge_min[:, i]
    print('  %2d-%-2d median %5.2f  p10 %5.2f  P(<1.5)=%.2f  P(<3)=%.2f' %
          (i, i+1, np.median(a), np.percentile(a,10), (a<1.5).mean(), (a<3).mean()))
am = edge_min.min(1)
print('\nwhole lap worst clearance: median %.2f  p10 %.2f  P(any edge <1.5 m)=%.2f' %
      (np.median(am), np.percentile(am,10), (am<1.5).mean()))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(15, 8))
ax = axs[0]
ax.plot(C0[:20,0], C0[:20,1], 's', ms=8, color='0.3', label='row A: Station 1-20 (-y)')
ax.plot(C0[20:,0], C0[20:,1], 's', ms=8, color='0.65', label='row B: Station 21-40 (+y)')
for k,(r,n) in enumerate(names):
    ax.annotate(str(n), C0[k], fontsize=6, color='0.25')
ax.plot(P[:,0], P[:,1], '-o', color='tab:orange', label='race line')
for i in range(17): ax.annotate(str(i), P[i,:2], fontsize=9, color='tab:red')
for tgt,c in ((6,'tab:green'),(7,'tab:red'),(8,'tab:blue')):
    ax.plot([P[5,0],P[tgt,0]],[P[5,1],P[tgt,1]],'--',color=c,lw=1.3,label='LOS 5->%d'%tgt)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(fontsize=7,ncol=3)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Column model vs course: %.1f m/station, row sep %.1f m, fit residual %.1f m rms'
             % (np.linalg.norm(M[0]), np.linalg.norm(M[1]), np.sqrt((res**2).sum()/17)))
ax2 = axs[1]
ax2.plot(C0[:20,0], C0[:20,1], 's', ms=9, color='0.3')
ax2.plot(C0[20:,0], C0[20:,1], 's', ms=9, color='0.65')
for k,(r,n) in enumerate(names): ax2.annotate('%s%d'%(r,n), C0[k], fontsize=7, color='0.25')
ax2.plot(P[:,0], P[:,1], '-o', color='tab:orange')
for i in range(17): ax2.annotate(str(i), P[i,:2], fontsize=10, color='tab:red')
for tgt,c in ((6,'tab:green'),(7,'tab:red'),(8,'tab:blue')):
    ax2.plot([P[5,0],P[tgt,0]],[P[5,1],P[tgt,1]],'--',color=c,lw=1.5)
ax2.set_xlim(-125,-60); ax2.set_ylim(-20,15); ax2.set_aspect('equal'); ax2.grid(alpha=.3)
ax2.set_title('zoom: gates 5-9 and the LOS 5->7 grazing column L16 (red dashed)')
fig.tight_layout()
for p in (ROOT+'pilot/perception/vercheck/xfer_columns.png', OUT+'xfer_columns.png'):
    try: fig.savefig(p, dpi=110); print('wrote', p)
    except Exception as e: print('FAILED', p, e)
