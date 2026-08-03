"""xfer_quadrant.py -- can the 2-3 / 6-7 / 15-16 mod-90 quadrant branches be PINNED?

Read-only analysis. The skylight compass is absolute only mod 90; psi_unwrapped is
continuous WITHIN a session, so every pair measured in one session shares ONE branch
offset. Sessions whose branch was fixed against the pausing-lap reference by a shared
pair are MEASUREMENT; the rest fall back to the sketch prior. This asks, for each of the
three sketch-branch legs, whether any BRANCH-RESOLVED session also carries rows for that
leg -- which would turn the branch into a measurement.
"""
import json, math, collections
import numpy as np

HERE = 'C:/Users/USER/Projects/vqual-2/pilot/perception/'

rows = json.load(open(HERE + 'mapdir_rows.json'))
M = json.load(open(HERE + 'map_vq2.json'))
L = M['layout_directions_2026_08_02']
OFF = L['session_branch_offsets_deg']
PROV = L['quadrant_provenance']
ROT = L['frame']['grid_to_sketch_rotation_deg']
REFL = L['frame']['grid_to_sketch_reflection']
PB = L['pair_bearings_grid_deg']


def to_sketch(b):
    return ((-(b + ROT)) if REFL else (b + ROT)) % 360.0


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def circmed(v):
    v = np.asarray(v, float)
    b = v[0]
    return float((b + np.median(wrap(v - b))) % 360.0)


RESOLVED = {s for s, p in PROV.items() if 'SKETCH-PRIOR' not in p}

by = collections.defaultdict(list)
for r in rows:
    by[(r['session'], tuple(r['pair']))].append(r)

print('=== branch provenance ===')
for s in sorted(PROV):
    tag = 'RESOLVED' if s in RESOLVED else 'sketch  '
    print('  [%s] off=%6.1f  %s' % (tag, OFF[s], s))
    print('              %s' % PROV[s])

LEGS = [(2, 3), (6, 7), (15, 16)]
print('\n=== who measured each contested leg ===')
for leg in LEGS:
    k = '%d-%d' % leg
    print('\n-- leg %s   accepted sketch-frame bearing %.2f deg (%s)'
          % (k, PB[k]['bearing_sketch_frame'], PB[k]['mode']))
    for (s, p), rs in sorted(by.items()):
        if p != leg:
            continue
        b = [to_sketch(r['bearing'] + OFF[s]) for r in rs]
        d = [r['d'] for r in rs]
        src = collections.Counter(r['src'] for r in rs)
        tag = 'RESOLVED' if s in RESOLVED else 'sketch'
        print('   %-9s n=%4d  bearing_sk med %7.2f  MAD %5.2f  d med %6.2f  src %s  [%s]'
              % (tag, len(rs), circmed(b),
                 float(np.median(np.abs(wrap(np.array(b) - circmed(b))))),
                 float(np.median(d)), dict(src), s))

print('\n=== REFEREE TEST: resolved-session rows vs the four branch candidates ===')
for leg in LEGS:
    key = '%d-%d' % leg
    ref = [(s, r) for (s, p), rs in by.items() if p == leg and s in RESOLVED for r in rs]
    if not ref:
        print('\n-- leg %s: NO rows in any branch-resolved session -> NOT PINNABLE' % key)
        continue
    bs = [to_sketch(r['bearing'] + OFF[s]) for s, r in ref]
    med = circmed(bs)
    ds = [r['d'] for _, r in ref]
    print('\n-- leg %s: referee n=%d from %s' % (key, len(ref), sorted(set(s for s, _ in ref))))
    print('   referee bearing_sk = %.2f deg  rows %s  d %s m'
          % (med, [round(x, 1) for x in bs], [round(x, 2) for x in ds]))
    for (s, p), rs in sorted(by.items()):
        if p != leg or s in RESOLVED:
            continue
        base = circmed([r['bearing'] for r in rs])
        print('   sketch session %s (n=%d, chosen off=%.0f):' % (s, len(rs), OFF[s]))
        for kk in (0, 90, 180, 270):
            cand = to_sketch(base + kk)
            print('      k=%3d -> %7.2f   |err vs referee| %6.2f%s'
                  % (kk, cand, abs(wrap(cand - med)),
                     '   <== CHOSEN' if (kk % 360) == (OFF[s] % 360) else ''))

print('\n=== connectivity of the resolved-session-only measurement graph ===')
edges_res = sorted({p for (s, p) in by if s in RESOLVED})
print('  pairs measured in resolved sessions:', edges_res)
for leg in LEGS:
    E = [e for e in edges_res if e != leg]
    adj = collections.defaultdict(set)
    for a, b in E:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {leg[0]}, [leg[0]]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    print('  leg %s: with the leg itself removed, is %d still reachable from %d through '
          'resolved-session edges?  %s'
          % (leg, leg[1], leg[0], 'YES (indirectly pinned)' if leg[1] in seen else 'NO (bridge)'))
