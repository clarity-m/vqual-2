"""xfer_quadrant2.py -- pin the 2-3 leg from the RESOLVED-session graph alone.

Independent of the two thin 202923 (2,3) rows: gates 2 and 3 are BOTH tied into the
branch-resolved chain by other pairs measured in branch-resolved sessions, so the 2-3
vector is predicted by the chain and can referee the sketch branch.
"""
import json, collections
import numpy as np

HERE = 'C:/Users/USER/Projects/vqual-2/pilot/perception/'
rows = json.load(open(HERE + 'mapdir_rows.json'))
M = json.load(open(HERE + 'map_vq2.json'))
L = M['layout_directions_2026_08_02']
OFF, PROV = L['session_branch_offsets_deg'], L['quadrant_provenance']
ROT, REFL = L['frame']['grid_to_sketch_rotation_deg'], L['frame']['grid_to_sketch_reflection']
ACC = {tuple(int(x) for x in k.split('-')): v['dist_m'] for k, v in M['measured_pairs'].items()}
RESOLVED = {s for s, p in PROV.items() if 'SKETCH-PRIOR' not in p}


def to_sketch(b):
    return ((-(b + ROT)) if REFL else (b + ROT)) % 360.0


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def circmed(v):
    v = np.asarray(v, float); b = v[0]
    return float((b + np.median(wrap(v - b))) % 360.0)


by = collections.defaultdict(list)
for r in rows:
    if r.get('bearing') is None or r['session'] not in RESOLVED:
        continue
    by[tuple(r['pair'])].append((r['session'], r))

print('--- resolved-session pairs among gates 0..6 (leg 2-3 EXCLUDED from the solve) ---')
use = {}
for p, rs in sorted(by.items()):
    if max(p) > 6:
        continue
    d = np.median([r['d'] for _, r in rs])
    b = circmed([to_sketch(r['bearing'] + OFF[s]) for s, r in rs])
    acc = ACC.get(p)
    flag = ''
    if p == (4, 6):
        flag = 'REFUSED by the map (re-measures 4-5)'
    if acc is None:
        flag = (flag + ' ; not an accepted map pair').strip(' ;')
    print('  %-8s n=%4d  d=%6.2f  acc=%s  bearing_sk=%7.2f  %s'
          % (str(p), len(rs), d, ('%.2f' % acc) if acc else '  --  ', b, flag))
    use[p] = (len(rs), d, b, acc)

for label, keep in [('A: accepted map pairs only, n>=5, leg 2-3 excluded',
                     lambda p, n, acc: acc is not None and n >= 5 and p != (2, 3) and p != (4, 6)),
                    ('B: same but allow thin (n>=1) non-accepted pairs too',
                     lambda p, n, acc: p != (2, 3) and p != (4, 6))]:
    E = [(p, v) for p, v in use.items() if keep(p, v[0], v[3])]
    idx = sorted({g for p, _ in E for g in p})
    ii = {g: k for k, g in enumerate(idx)}
    if 2 not in ii or 3 not in ii:
        print('\n%s -> gates 2 and 3 not both present' % label); continue
    A = np.zeros((2 * len(E) + 2, 2 * len(idx)))
    y = np.zeros(2 * len(E) + 2)
    for k, (p, (n, d, b, acc)) in enumerate(E):
        a, c = ii[p[0]], ii[p[1]]
        th = np.radians(b)
        for ax in (0, 1):
            A[2 * k + ax, 2 * c + ax] = 1.0
            A[2 * k + ax, 2 * a + ax] = -1.0
        y[2 * k] = d * np.cos(th); y[2 * k + 1] = d * np.sin(th)
    A[-2, 0] = 1.0; A[-1, 1] = 1.0            # gauge: gate 0 at origin
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    P = sol.reshape(-1, 2)
    v = P[ii[3]] - P[ii[2]]
    pred_b = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    res = A @ sol - y
    print('\n%s' % label)
    print('  edges used: %s' % [p for p, _ in E])
    print('  solve residual rms %.3f m, max %.3f m' % (np.sqrt((res**2).mean()), np.abs(res).max()))
    print('  PREDICTED 2-3: d = %.2f m, bearing_sk = %.2f deg  (accepted d 13.25, bearing 211.69)'
          % (np.linalg.norm(v), pred_b))
    for k in (0, 90, 180, 270):
        cand = to_sketch(circmed([r['bearing'] for r in
                                  [x for x in rows if tuple(x['pair']) == (2, 3)
                                   and x['session'].startswith('20260802-010914')]]) + k)
        print('     2-3-c branch k=%3d -> %7.2f  |err vs chain| %6.2f%s'
              % (k, cand, abs(wrap(cand - pred_b)),
                 '  <== CHOSEN' if k == OFF['20260802-010914-vq2-strafe-2-3-c'] % 360 else ''))
