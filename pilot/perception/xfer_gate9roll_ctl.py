"""Positive controls + diagnostics for xfer_gate9roll.py.

1. INJECTION TEST: take each row's resolved pose, synthesise the quad WITH a known
   in-plane roll, and push it through the identical measurement. If the recovered angle
   does not track the injected one, a null result means nothing.
2. LEAN SENSITIVITY: synthesise a leaning-but-unrolled gate and read what the estimator
   returns, per obliquity bin -- the contamination term.
3. PnP-FREE obliquity proxy (quad foreshortening) as a second stratifier.
"""
from __future__ import annotations
import collections, json, math, sys
import numpy as np
sys.path.insert(0, 'C:/Users/USER/xfer_scratch')
from xfer_gate9roll import (levelling, quad_orientation, plumb_dir, synth_quad, wrap90,
                            ROWS, GATE_M)
sys.path.insert(0, 'C:/Users/USER/Projects/vqual-2/pilot')
import producer as P

D = json.load(open(ROWS))
INJ = [0.0, 5.0, 10.0, 20.0]
res = collections.defaultdict(list)
lean_by = collections.defaultdict(list)
prox = []
for gs, rows in D.items():
    g = int(gs)
    for r in rows:
        q = np.asarray(r['quad'], float)
        if q.shape != (4, 2):
            continue
        gb = np.asarray(r['g_body'], float); Rlb = levelling(gb)
        if Rlb is None: continue
        gh = gb/np.linalg.norm(gb)
        ctr = q.mean(0); rhat = P.pixel_ray_body(*ctr); Pb = rhat*float(r['range'])
        pu = plumb_dir(Pb, gh)
        if pu is None: continue
        n_lev = np.asarray(r['n_lev'], float); l_lev = np.asarray(r['l_lev'], float)
        th = math.degrees(math.acos(max(-1., min(1., abs(float(n_lev@l_lev))))))
        lean = math.degrees(math.asin(min(1., abs(float(n_lev[2])))))
        nb = Rlb.T @ n_lev
        for a in INJ:
            sq = synth_quad(Pb, nb, gh, a)
            if sq is None: continue
            v = quad_orientation(sq, pu)
            if v is None: continue
            res[(g, a)].append((th, v, float(r['size_px'])))
        # PnP-free obliquity proxy: edge-pair length ratio
        e = [np.linalg.norm(q[(i+1) % 4]-q[i]) for i in range(4)]
        a1, a2 = (e[0]+e[2])/2, (e[1]+e[3])/2
        if max(a1, a2) > 1e-6:
            prox.append((g, th, math.degrees(math.acos(min(1., min(a1,a2)/max(a1,a2)))),
                         lean, float(r['size_px'])))

TB = [(0,5),(5,10),(10,15),(15,25),(25,90)]
print('INJECTION TEST -- recovered angle (deg) for a synthetic quad at the row pose')
print('gate  inj |' + ''.join(' th %2d-%2d   |' % b for b in TB))
for g in (9,3,8,13):
    for a in INJ:
        cells=[]
        for lo,hi in TB:
            v=[x[1] for x in res[(g,a)] if lo<=x[0]<hi and x[2]>=80]
            cells.append('%6.1f n%-3d|'%(np.median(v),len(v)) if len(v)>=3 else '  --  n%-3d|'%len(v))
        print('%-5d %4.0f|'%(g,a)+''.join(cells))
print()
print('LEAN of the row PnP normal (deg) and PnP-free obliquity proxy, by theta bin, size>=80')
print('gate |' + ''.join(' th %2d-%2d          |' % b for b in TB))
for g in (9,3,8,13):
    cells=[]
    for lo,hi in TB:
        v=[(x[3],x[2]) for x in prox if x[0]==g and lo<=x[1]<hi and x[4]>=80]
        cells.append('lean%5.1f prox%5.1f|'%(np.median([a for a,_ in v]),np.median([b for _,b in v])) if len(v)>=3 else '   --  n%-3d       |'%len(v))
    print('%-4d |'%g+''.join(cells))
