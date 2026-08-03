"""mapedges_strafe1011.py -- the 10-11 edge, measured from four staged vantages.

WHY THIS SESSION EXISTS. The map's 10-11 edge read 33.84 m and was wrong. Its 76 rows came
from ONE session, ONE contiguous 171-frame window, 1.5 deg of bearing spread: a single
vantage, so the static-pair viewpoint referee -- two fixed gates cannot change separation
with viewpoint -- never fired. Three channels said the number was impossible (drag
path/chord 0.53-0.67 against it, i.e. a flown path SHORTER than the chord; the station-
number ruler's rigid -19.8 m step at gate 11; frame re-detection showing the rows pairing
the near gate with a gate two along). Claire then flew
20260802-180755-vm-strafe-10-11 for this edge alone: markers 1-2, 3-4, 5-6, 7-8 delimit
FOUR intervals of stability, each at a different viewing angle on the 10/11 pair. That is
the multi-vantage evidence the original measurement lacked.

WHAT HAD TO BE FIXED TO MEASURE IT AT ALL. detect.detections() returns NOTHING for the near
gate here. At 4-7 m its aperture stops forming a hole contour (ribbon glow and bloom fill
it) while its CHECKERBOARD squares do become holes, so the frame yields half a dozen
decoration squares with fictitious 20-47 m ranges -- and the outer-boundary fallback then
refuses the gate body itself, because that body is now a 'used parent'. big_outer() below
fits that outer boundary anyway, with detect.py's own 2700 mm model. Taking the naive
'nearest detection' as gate 10 (the first pass here did) measures decoration and yields
nonsense; recorded because the failure is silent.

IDENTITY. Three static objects sit at 13.8 / 16.4 / 34.9 m horizontally from gate 10, each
reproduced across vantages. The 13.8 and 16.4 objects are the SAME gate: 85% of the 13.8
detections come from the biased outer-boundary fallback and 92% of the 16.4 ones from the
inner aperture, their apparent sizes are equal (29.5 vs 30.6 px), and detect.py's own note
says the outer fit is biased outward by bloom and signage -- which shortens the range and
so shortens the separation. The inner-aperture value is the measurement. Which far object
is gate 11 is then settled by an INDEPENDENT number: the map's 11-12 = 19.67 m (28 rows, a
different session). The inner-aperture object reproduces it at 19.14 m; the 34.9 m object
does not (21.96 m to the third). The cyan ribbon confirms it by eye --
vercheck/strafe1011_ident.png, the ribbon threads gate 10 then the inner-aperture gate.

RESULT: 10-11 = 16.44 m horizontal, dz_up -3.63 m; and the 34.9 m object is GATE 12, i.e.
the old 33.84 m rows were 10-12 under a wrong name (mapvq2.RELABEL_ROWS).

    python3 pilot/perception/mapedges_strafe1011.py
"""
import collections
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import detect as D          # noqa: E402
import label as L           # noqa: E402

SESS = os.path.join(os.path.dirname(HERE), 'sessions',
                    '20260802-180755-vm-strafe-10-11')
MIN_BIG_AREA = 2500.0       # px^2: the near gate at 4-7 m; nothing else is this large
MIN_FAR_PX = 11.0
MAX_ABS_DZ = 6.0            # m: above this it is ceiling structure, not a gate
CLUSTER_TOL = 1.3           # m
TARGETS = {'g11_outer_fit': 13.8, 'g11': 16.5, 'g12': 34.9}


def big_outer(img):
    """The near gate via its OUTER boundary, even when it is a 'used parent'."""
    m = D.orange_mask(img)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(cnts):
        if hier[i][3] >= 0 or cv2.contourArea(c) < MIN_BIG_AREA:
            continue
        q = D.fit_quad(c)
        if q is None:
            continue
        qa = cv2.contourArea(q.astype(np.float32))
        if qa <= 1.0 or cv2.contourArea(c) / qa < D.MIN_FILL_OUTER:
            continue
        e = [np.linalg.norm(q[(j + 1) % 4] - q[j]) for j in range(4)]
        if max(e) / max(min(e), 1e-6) > D.MAX_ASPECT:
            continue
        ok, _rv, tv = cv2.solvePnP(D.OBJ_OUTER, D.order_quad(q), D.K, D.DIST,
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        t = tv.reshape(3)
        out.append({'pos_body': L.body_to_cam().T @ t,
                    'range_m': float(np.linalg.norm(t)),
                    'area': float(cv2.contourArea(c)), 'quad': D.order_quad(q)})
    out.sort(key=lambda d: -d['area'])
    return out


def load():
    frames = [r for r in csv.DictReader(open(os.path.join(SESS, 'frames.csv'), newline=''))
              if r['file']]
    ft = np.array([float(r['t_recv_wall_ns']) for r in frames])
    imu = list(csv.DictReader(open(os.path.join(SESS, 'imu.csv'), newline='')))
    it = np.array([float(r['t_wall_ns']) for r in imu])
    ia = np.array([[float(r['xacc']), float(r['yacc']), float(r['zacc'])] for r in imu])
    mk = sorted(float(json.loads(l)['t_wall_ns'])
                for l in open(os.path.join(SESS, 'events.jsonl'))
                if json.loads(l).get('kind') == 'marker')
    return frames, ft, it, ia, mk


def gravity(it, ia, t, tol=0.35):
    j = int(np.argmin(np.abs(it - t)))
    if abs(it[j] - t) > 0.1e9:
        return None
    a = ia[j]
    n = float(np.linalg.norm(a))
    return None if abs(n - 9.81) > tol else -a / n


def rows_for_interval(frames, ft, it, ia, lo, hi, comp=None):
    out = []
    for i in range(lo, hi):
        img = cv2.imread(os.path.join(SESS, 'frames', frames[i]['file']))
        if img is None:
            continue
        bg = big_outer(img)
        gh = gravity(it, ia, ft[i])
        if not bg or gh is None:
            continue
        pa = np.asarray(bg[0]['pos_body'], float)
        psi = None
        if comp is not None and comp.ok:
            psi, _why = comp.bearing_terms(frames[i]['file'], ft[i])
        found = {}
        for d in D.detections(img):
            if d['size_px'] < MIN_FAR_PX or d.get('pos_body') is None:
                continue
            p = np.asarray(d['pos_body'], float)
            v = p - pa
            dd = float(np.linalg.norm(v))
            if abs(float(v @ gh)) > MAX_ABS_DZ:
                continue
            for nm, t in TARGETS.items():
                if abs(dd - t) < CLUSTER_TOL and nm not in found:
                    found[nm] = d
        rec = {'i': i, 'fid': int(frames[i]['frame_id']),
               'near_range': bg[0]['range_m'], 'objs': {}}
        pts = {'g10': (pa, None)}
        for nm, d in found.items():
            pts[nm] = (np.asarray(d['pos_body'], float), d)
        for a in pts:
            for b in pts:
                if a >= b:
                    continue
                v = pts[b][0] - pts[a][0]
                dd = float(np.linalg.norm(v))
                dz = -float(v @ gh)
                rec['objs'][(a, b)] = {
                    'd': dd, 'h': math.sqrt(max(dd * dd - dz * dz, 0.0)), 'dz_up': dz,
                    'src': (pts[b][1] or {}).get('source'),
                    'px': (pts[b][1] or {}).get('size_px')}
                if psi is not None:
                    psi_u, R_lb = psi
                    v_lev = R_lb @ v
                    rec['objs'][(a, b)]['bearing'] = (
                        psi_u + math.degrees(math.atan2(v_lev[1], v_lev[0])))
        out.append(rec)
    return out


def stat(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not len(v):
        return None
    m = float(np.median(v))
    return m, float(np.median(np.abs(v - m))), len(v)


PAIRS = (('g10', 'g11'), ('g10', 'g12'), ('g11', 'g12'))


def select(rows, pair):
    """Inner-aperture detections only for gate 11 (the outer fit is biased -- see header)."""
    return [r['objs'][pair] for r in rows if pair in r['objs']
            and (pair[1] != 'g11' or r['objs'][pair]['src'] == 'inner')]


def main():
    frames, ft, it, ia, mk = load()
    comp = None
    try:
        import mapdir
        comp = mapdir.Compass(SESS)
    except Exception as exc:
        print('no compass (%s)' % exc)
    print('compass available: %s' % bool(comp is not None and comp.ok))
    pooled = collections.defaultdict(list)
    for k in range(0, 8, 2):
        lo = int(np.searchsorted(ft, mk[k]))
        hi = int(np.searchsorted(ft, mk[k + 1]))
        rows = rows_for_interval(frames, ft, it, ia, lo, hi, comp)
        print('')
        print('=== VANTAGE %d (markers %d-%d)  frames %d..%d  usable %d'
              % (k // 2 + 1, k + 1, k + 2, int(frames[lo]['frame_id']),
                 int(frames[hi - 1]['frame_id']), len(rows)))
        nr = [r['near_range'] for r in rows]
        if nr:
            print('    gate 10 at %.1f-%.1f m' % (min(nr), max(nr)))
        for pair in PAIRS:
            sel = select(rows, pair)
            if not sel:
                print('    %s-%s: not co-visible in this vantage' % pair)
                continue
            h = stat([s['h'] for s in sel])
            z = stat([s['dz_up'] for s in sel])
            b = [s['bearing'] for s in sel if 'bearing' in s]
            line = ('    %-3s-%-3s horiz %6.2f m (MAD %4.2f, n=%3d)  dz_up %+6.2f (MAD %4.2f)'
                    % (pair[0][1:], pair[1][1:], h[0], h[1], h[2], z[0], z[1]))
            if b:
                bb = np.asarray(b)
                line += '  bearing %7.2f (spread %5.2f, n=%d)' % (
                    float(np.median(bb)), float(bb.max() - bb.min()), len(bb))
            px = [s['px'] for s in sel if s['px']]
            print(line + ('  far %4.1f px' % float(np.median(px)) if px else ''))
            pooled[pair].extend(sel)
    print('')
    print('=== POOLED over the four vantages -- THIS IS THE MEASUREMENT')
    for pair in PAIRS:
        sel = pooled[pair]
        if not sel:
            continue
        h = stat([s['h'] for s in sel])
        d = stat([s['d'] for s in sel])
        z = stat([s['dz_up'] for s in sel])
        b = [s['bearing'] for s in sel if 'bearing' in s]
        line = ('  %-3s-%-3s horiz %6.2f (MAD %4.2f)  3-D %6.2f  dz_up %+6.2f (MAD %4.2f)  n=%d'
                % (pair[0][1:], pair[1][1:], h[0], h[1], d[0], z[0], z[1], h[2]))
        if b:
            bb = np.asarray(b)
            line += '  bearing %7.2f (MAD %4.2f)' % (
                float(np.median(bb)), float(np.median(np.abs(bb - np.median(bb)))))
        print(line)
    print('')
    print('  REFEREE: 11-12 = 19.67 m horiz, dz +2.32 m was measured in a DIFFERENT')
    print('  session (n=28). Reproducing it here is what NAMES gate 11.')


if __name__ == '__main__':
    main()
