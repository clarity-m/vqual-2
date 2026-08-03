"""Final stratified summary + plot for the gate-9 in-plane-roll question."""
from __future__ import annotations
import collections, json, math, sys
import numpy as np
sys.path.insert(0, 'C:/Users/USER/xfer_scratch')
from xfer_gate9roll import (levelling, quad_orientation, plumb_dir, wrap90, ROWS)
sys.path.insert(0, 'C:/Users/USER/Projects/vqual-2/pilot')
import producer as P

D = json.load(open(ROWS))
R = []
for gs, rows in D.items():
    g = int(gs)
    for r in rows:
        q = np.asarray(r['quad'], float)
        if q.shape != (4, 2): continue
        gb = np.asarray(r['g_body'], float); Rlb = levelling(gb)
        if Rlb is None: continue
        gh = gb/np.linalg.norm(gb)
        ctr = q.mean(0); Pb = P.pixel_ray_body(*ctr)*float(r['range'])
        pu = plumb_dir(Pb, gh)
        if pu is None: continue
        phi = quad_orientation(q, pu)
        if phi is None: continue
        e = [np.linalg.norm(q[(i+1) % 4]-q[i]) for i in range(4)]
        a1, a2 = (e[0]+e[2])/2, (e[1]+e[3])/2
        prox = math.degrees(math.acos(min(1., min(a1,a2)/max(a1,a2))))
        R.append(dict(g=g, s=r['s'], phi=phi, prox=prox, size=float(r['size_px'])))
print('total rows %d' % len(R))

def cell(sel):
    v = [r['phi'] for r in R if sel(r)]
    if len(v) < 3: return '   --  n%-4d|' % len(v), len(v)
    by = collections.defaultdict(list)
    for r in R:
        if sel(r): by[(r['g'], r['s'])].append(r['phi'])
    ms = [np.median(x) for x in by.values() if len(x) >= 3]
    sp = np.std(ms) if len(ms) >= 2 else float('nan')
    return '%+5.1f+-%4.1f n%-4d|' % (np.median(v), sp, len(v)), len(v)

GR = [('9', lambda g: g == 9), ('3', lambda g: g == 3), ('8', lambda g: g == 8),
      ('13', lambda g: g == 13), ('all-9', lambda g: g != 9)]
for smin in (26, 60, 100):
    for bins, nm, key in (([(0,10),(10,20),(20,35),(35,90)], 'PnP-FREE foreshortening obliquity', 'prox'),):
        print('\n=== median phi +- sd(session medians), deg   [%s, size_px>=%d]' % (nm, smin))
        print('gate  |' + ''.join('  %2d-%2d deg      |' % b for b in bins))
        for gname, gsel in GR:
            row = ''
            for lo, hi in bins:
                c, _ = cell(lambda r, gs=gsel, lo=lo, hi=hi, sm=smin:
                            gs(r['g']) and lo <= r[key] < hi and r['size'] >= sm)
                row += c
            print('%-5s |' % gname + row)

# bootstrap CI on gate 9 pooled (size>=60)
v = np.array([r['phi'] for r in R if r['g'] == 9 and r['size'] >= 60])
bs = [np.median(np.random.choice(v, len(v))) for _ in range(4000)]
print('\ngate 9 pooled size>=60: n=%d median %+.2f deg, 95%% CI [%+.2f, %+.2f], IQR %.2f'
      % (len(v), np.median(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
         np.subtract(*np.percentile(v, [75, 25]))))
for g in (3, 8, 13):
    w = np.array([r['phi'] for r in R if r['g'] == g and r['size'] >= 60])
    bs = [np.median(np.random.choice(w, len(w))) for _ in range(4000)]
    print('gate %-2d pooled size>=60: n=%d median %+.2f deg, 95%% CI [%+.2f, %+.2f], IQR %.2f'
          % (g, len(w), np.median(w), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
             np.subtract(*np.percentile(w, [75, 25]))))
allg = collections.defaultdict(list)
for r in R:
    if r['size'] >= 60: allg[r['g']].append(r['phi'])
print('\nper-gate medians (size>=60): ' + '  '.join(
    '%d:%+.1f(n%d)' % (g, np.median(v), len(v)) for g, v in sorted(allg.items()) if len(v) >= 10))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
cols = {9: 'tab:red', 3: 'tab:blue', 8: 'tab:green', 13: 'tab:orange'}
for g, c in cols.items():
    xs = [r['prox'] for r in R if r['g'] == g and r['size'] >= 60]
    ys = [r['phi'] for r in R if r['g'] == g and r['size'] >= 60]
    ax[0].scatter(xs, ys, s=7, alpha=.35, color=c, label='gate %d (n=%d)' % (g, len(xs)))
    bx, by = [], []
    for lo, hi in ((0,10),(10,20),(20,35),(35,90)):
        w = [y for x, y in zip(xs, ys) if lo <= x < hi]
        if len(w) >= 5: bx.append((lo+min(hi,50))/2); by.append(np.median(w))
    ax[0].plot(bx, by, '-o', color=c, lw=2)
ax[0].set_xlabel('PnP-free obliquity proxy (deg)'); ax[0].set_ylabel('in-plane angle phi (deg)')
ax[0].set_title('quad orientation vs plumb, size_px>=60'); ax[0].set_ylim(-30, 30)
ax[0].axhline(0, color='k', lw=.8); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
meds = sorted(((g, np.median(v), len(v)) for g, v in allg.items() if len(v) >= 10),
              key=lambda t: t[0])
ax[1].bar([str(g) for g, _, _ in meds], [m for _, m, _ in meds],
          color=['tab:red' if g == 9 else 'tab:gray' for g, _, _ in meds])
ax[1].axhline(0, color='k', lw=.8)
ax[1].set_xlabel('gate'); ax[1].set_ylabel('median phi (deg)')
ax[1].set_title('median in-plane angle per gate (size_px>=60)'); ax[1].grid(alpha=.3, axis='y')
ax[1].set_ylim(-12, 12)
fig.tight_layout(); fig.savefig('C:/Users/USER/xfer_scratch/xfer_gate9roll_summary.png', dpi=110)
print('wrote summary png')
