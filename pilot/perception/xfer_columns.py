"""READ-ONLY analysis: derive a column/obstacle model in the measured course frame."""
import json, math
import numpy as np

ROOT = 'C:/Users/USER/Projects/vqual-2/'
OUT = 'C:/Users/USER/AppData/Local/Temp/xfercols/'
ap = json.load(open(ROOT + 'pilot/perception/map_approx.json'))
cv = json.load(open(ROOT + 'pilot/course/course_vq2.json'))

P = np.array([g['position_m'] for g in sorted(cv['gates'], key=lambda g: g['id'])], float)
SIG = np.array([g['sigma_xy_m'] for g in sorted(cv['gates'], key=lambda g: g['id'])], float)

G = sorted(ap['gates'], key=lambda g: g['race_index'])
S = np.array([[g['along_station'], g['across']] for g in G], float)

print('\ncorr(x, along) = %+.4f   corr(y, across) = %+.4f' %
      (np.corrcoef(P[:, 0], S[:, 0])[0, 1], np.corrcoef(P[:, 1], S[:, 1])[0, 1]))

A = np.column_stack([S[:, 0], S[:, 1], np.ones(17)])
M, *_ = np.linalg.lstsq(A, P[:, :2], rcond=None)
res = P[:, :2] - A @ M
dof = 17 - 3
s2 = (res ** 2).sum(0) / dof
cov = np.linalg.inv(A.T @ A)
se = np.sqrt(np.outer(np.diag(cov), s2))

print('\naffine M (rows: along, across, const; cols: x, y)')
for i, nm in enumerate(['along ', 'across', 'const ']):
    print('  %s  x=%+9.3f +-%5.3f   y=%+9.3f +-%5.3f' % (nm, M[i,0], se[i,0], M[i,1], se[i,1]))

u_along, u_across = M[0], M[1]
mps = np.linalg.norm(u_along); rowsep = np.linalg.norm(u_across)
ang = math.degrees(math.acos(np.dot(u_along, u_across) / (mps * rowsep)))
print('\n|along|  = %.3f m/station\n|across| = %.3f m per across-unit (row separation)\naxis angle %.2f deg' % (mps, rowsep, ang))
rss = (res**2).sum(); tss = ((P[:,:2]-P[:,:2].mean(0))**2).sum()
print('R^2 (2-D) = %.4f' % (1 - rss/tss))
r = np.linalg.norm(res, axis=1)
print('residual: rms %.2f  med %.2f  max %.2f (gate %d)' % (np.sqrt((r**2).mean()), np.median(r), r.max(), int(r.argmax())))
print('per-gate:', ' '.join('%.1f' % v for v in r))

# 1-D checks
sx = np.polyfit(S[:,0], P[:,0], 1)
print('\n1-D  x vs along: slope %.3f m/station, resid rms %.2f m' % (sx[0], np.std(P[:,0]-np.polyval(sx,S[:,0]))))
sy = np.polyfit(S[:,1], P[:,1], 1)
print('1-D  y vs across: slope %.3f m/unit, resid rms %.2f m' % (sy[0], np.std(P[:,1]-np.polyval(sy,S[:,1]))))

rng = np.random.default_rng(0)
bm, br, bang = [], [], []
for _ in range(4000):
    idx = rng.integers(0, 17, 17)
    Mb, *_ = np.linalg.lstsq(A[idx], P[idx, :2], rcond=None)
    a, c = Mb[0], Mb[1]
    na, nc = np.linalg.norm(a), np.linalg.norm(c)
    if na < 1e-6 or nc < 1e-6: continue
    bm.append(na); br.append(nc)
    bang.append(math.degrees(math.acos(np.clip(np.dot(a,c)/(na*nc), -1, 1))))
bm, br, bang = map(np.array, (bm, br, bang))
print('\nbootstrap m/station %.2f [%.2f, %.2f]' % (np.median(bm), *np.percentile(bm,[16,84])))
print('bootstrap row sep   %.2f [%.2f, %.2f]' % (np.median(br), *np.percentile(br,[16,84])))
print('bootstrap axis ang  %.1f [%.1f, %.1f]' % (np.median(bang), *np.percentile(bang,[16,84])))

cols = {}
for N in range(6, 36):
    cols[('L', N)] = M[2] + N * u_along
    cols[('R', 41 - N)] = M[2] + N * u_along + u_across
print('\n-- columns in map metres --')
for k in sorted(cols, key=lambda k: (k[0], k[1])):
    if k[1] % 4 == 0 or k[1] in (12,20,21,29):
        print('  %s%02d  x=%8.2f  y=%7.2f' % (k[0], k[1], cols[k][0], cols[k][1]))

def spd(a, b, q):
    ab = b - a
    t = float(np.clip(np.dot(q-a, ab)/np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(a + t*ab - q)), t

print('\n=== occlusion check from gate 5 ===')
for tgt in (6, 7, 8):
    a, b = P[5,:2], P[tgt,:2]
    ranked = sorted(((spd(a,b,v)[0], k, spd(a,b,v)[1]) for k, v in cols.items()))[:4]
    print('  gate5->g%-2d len %6.2f m :' % (tgt, np.linalg.norm(b-a)),
          ', '.join('%s%02d %.2fm(t=%.2f)' % (k[0], k[1], d, t) for d, k, t in ranked))

print('\n=== min clearance of each race edge to nearest modelled column ===')
worst = []
for i in range(16):
    a, b = P[i,:2], P[i+1,:2]
    cand = [(spd(a,b,v)[0], kk, spd(a,b,v)[1]) for kk, v in cols.items()]
    d, k, t = min(cand, key=lambda z: z[0])
    worst.append((d, i, k))
    print('  %2d-%-2d len %6.2f  min clearance %5.2f m to %s%02d (t=%.2f)' % (i, i+1, np.linalg.norm(b-a), d, k[0], k[1], t))
print('\nsorted clearances:', ', '.join('%d-%d:%.1f' % (i, i+1, d) for d, i, k in sorted(worst)))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 6))
cl = np.array([cols[('L', N)] for N in range(6, 36)])
cr = np.array([cols[('R', 41-N)] for N in range(6, 36)])
ax.plot(cl[:,0], cl[:,1], 's', ms=7, color='0.35', label='column row 12-20+ (-y side)')
ax.plot(cr[:,0], cr[:,1], 's', ms=7, color='0.7', label='column row 21-29+ (+y side)')
for N in range(8, 34, 2):
    ax.annotate(str(N), cols[('L',N)][:2], fontsize=6, color='0.3')
    ax.annotate(str(41-N), cols[('R',41-N)][:2], fontsize=6, color='0.55')
ax.plot(P[:,0], P[:,1], '-o', color='tab:orange', label='race line / gates')
for i in range(17):
    ax.annotate(str(i), P[i,:2], fontsize=9, color='tab:red')
for tgt, c in ((6,'tab:green'), (7,'tab:red'), (8,'tab:blue')):
    ax.plot([P[5,0], P[tgt,0]], [P[5,1], P[tgt,1]], '--', color=c, lw=1.3, label='LOS 5->%d' % tgt)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=3)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Derived column rows vs measured course  (%.2f m/station, row sep %.2f m)' % (mps, rowsep))
fig.tight_layout()
for p in (ROOT + 'pilot/perception/vercheck/xfer_columns.png', OUT + 'xfer_columns.png'):
    try:
        fig.savefig(p, dpi=110); print('wrote', p)
    except Exception as e:
        print('FAILED', p, e)
