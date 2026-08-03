"""mapdir.py -- pair measurements as VECTORS: distance + absolute grid-frame direction.

WHAT THIS ADDS. mapvq2/mapedges measure each co-visible gate pair as a distance and a
gravity-referenced height difference; the horizontal BEARING was the missing yaw. The
skylight compass (skylight.py, validated in skylight_validate.py: ~0.4 deg 1-sigma,
gyro slope +0.949, grid-azimuth constancy through 50-deg nose sweeps) supplies per-frame
heading psi in the ceiling-grid frame, mod 90. For a pair row measured at frame i:

    v_body   = pos_body(b) - pos_body(a)                  (a < b in race order)
    v_lev    = R_lb @ v_body                              (levelled: z = gravity down,
                                                           x = body-forward horizontal)
    bearing  = psi_unwrapped(i) + atan2(v_lev.y, v_lev.x) (grid frame, degrees)

the exact formula skylight_validate.check_rotation_azimuth() validated (constant during
sweeps, wrong sign 5x worse). psi_unwrapped is continuous WITHIN a session, so all of a
session's bearings share ONE unknown branch offset in {0, 90, 180, 270} (the mod-90
quadrant ambiguity). Branches are resolved pairwise: two sessions that measure the same
gate pair must give it the same bearing, which fixes their RELATIVE branch; sessions
linked to nothing get their branch from the sketch prior (station rows are parallel to
the light grid: skylight_pitch.py, +0.32 deg median), and that provenance is reported,
not hidden.

ROW SOURCES, reusing the audited channels verbatim rather than re-deciding identity:
  * contour rows: mapvq2.run() per base session, the same named-clean-per-frame
    selection measure() uses (mapedges.named_clean_by_frame), REFUSE_ROWS applied;
  * gatenet rows: rebuilt from mapedges_accepted.json quads (PnP recomputed exactly as
    mapedges.net_measure does) paired against the same contour partners; the
    corroboration verdicts of the final build are reproduced by refereeing against
    map_vq2.json's accepted pair medians (dropped channels are KEPT, tagged, for the
    1-2 adjudication -- they never enter the solve);
  * intent/hover/strafe rows: the collect() functions re-run with their own AUDIT and
    known-median referees; vectors taken from their tiles' meas[].pos (aligned with the
    audited pair labels by construction).
  * 15-16 strafing rows (mapvq2.strafe_rows): the two detections are UNORDERED, so the
    edge direction is mod-180; the sign is resolved against the solved layout and said so.

Every direction row additionally requires: the frame is compass-CONFIDENT (>=4 segments,
spread <= 12 deg), and |d - accepted pair median| <= 3 m (the bimodality-survivor mode).

    python3 pilot/perception/mapdir.py --collect     # stages A-C -> mapdir_rows.json
    python3 pilot/perception/mapdir.py --solve       # branch + layout + residuals
    python3 pilot/perception/mapdir.py --solve --write   # also update map_vq2.json
"""

from __future__ import annotations

import argparse
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
import label as L            # noqa: E402
import mapvq2 as M           # noqa: E402
import mapedges as ME_mod    # noqa: E402  (named_clean_by_frame; net constants)
import skylight as SK        # noqa: E402

ROWS_JSON = os.path.join(HERE, 'mapdir_rows.json')
MAP_JSON = os.path.join(HERE, 'map_vq2.json')

BASE_SESSIONS = [M.PRIMARY, M.SECOND, M.PAUSING, M.LINKLAP, M.TIEBREAK]
D_TOL = 3.0                  # row distance must sit within this of the accepted median


# ---------------------------------------------------------------------------------------
# compass access


class Compass:
    """skylight_<session>.csv keyed by frame file; filtered-gravity IMU for levelling."""

    def __init__(self, session_path):
        self.name = os.path.basename(os.path.normpath(session_path))
        p = os.path.join(HERE, f'skylight_{self.name}.csv')
        self.ok = os.path.exists(p)
        self.by_file = {}
        if self.ok:
            for r in csv.DictReader(open(p)):
                self.by_file[r['file']] = r
            self.imu = SK.Imu(session_path)

    def bearing_terms(self, file, t_ns):
        """-> (psi_unwrapped, R_lb) or (None, reason)."""
        r = self.by_file.get(file)
        if r is None:
            return None, 'no-csv-row'
        if r['confident'] != '1' or not r['psi_unwrapped']:
            return None, 'not-confident'
        g, _n = self.imu.gravity(t_ns)
        if g is None:
            return None, 'no-imu'
        R_lb = SK.level_rotation(g)
        if R_lb is None:
            return None, 'gimbal'
        return (float(r['psi_unwrapped']), R_lb), None


def grid_bearing(psi_u, R_lb, v_body):
    v_lev = R_lb @ np.asarray(v_body, float)
    return psi_u + math.degrees(math.atan2(v_lev[1], v_lev[0])), float(-v_lev[2])


# ---------------------------------------------------------------------------------------
# stage A: contour rows with vectors (mapvq2's own selection, REFUSE_ROWS applied)


def contour_vector_rows(res):
    S, good, ident = res['S'], res['good'], res['ident']
    byframe, _claims = ME_mod.named_clean_by_frame(S, good, ident)
    refuse = {p for (sn, p), _w in M.REFUSE_ROWS.items() if sn == S.name}
    rows = []
    for i, gd in sorted(byframe.items()):
        ks = sorted(gd)
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                ga, gb = ks[x], ks[y]
                if (ga, gb) in refuse:
                    continue
                v = np.asarray(gd[gb]['pos_body'], float) - \
                    np.asarray(gd[ga]['pos_body'], float)
                d = float(np.linalg.norm(v))
                if d < 1.0:
                    continue
                rows.append({'session': S.name, 'i': i, 'fid': int(S.fid[i]),
                             'file': S.frames[i]['file'],
                             'pair': (ga, gb), 'v_body': v.tolist(), 'd': d,
                             'src': 'contour'})
    return rows


# ---------------------------------------------------------------------------------------
# stage B: gatenet rows rebuilt from mapedges_accepted.json (no torch needed)


def pnp_pos_body(quad):
    q = np.asarray(quad, np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        np.array([[-L.HALF, L.HALF, 0], [L.HALF, L.HALF, 0],
                  [L.HALF, -L.HALF, 0], [-L.HALF, -L.HALF, 0]], np.float64),
        np.ascontiguousarray(q).reshape(4, 1, 2),
        np.array([[L.FX, 0, L.CX], [0, L.FY, L.CY], [0, 0, 1.0]], np.float64), None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return L.body_to_cam().T @ tvec.reshape(3)


def net_vector_rows(per_session_res, accepted_pairs):
    acc = json.load(open(os.path.join(HERE, 'mapedges_accepted.json')))
    bysess = collections.defaultdict(list)
    for m in acc:
        bysess[m['session']].append(m)
    rows = []
    for sname, ms in bysess.items():
        res = per_session_res.get(sname)
        if res is None:
            continue
        S = res['S']
        byframe, _c = ME_mod.named_clean_by_frame(S, res['good'], res['ident'])
        net_by_frame = collections.defaultdict(dict)
        for m in ms:
            cur = net_by_frame[m['i']].get(m['g'])
            if cur is None or m['resid'] < cur['resid']:
                net_by_frame[m['i']][m['g']] = m
        for i, gd in net_by_frame.items():
            partners = dict(byframe.get(i, {}))
            for g, m in gd.items():
                pa = pnp_pos_body(m['quad'])
                if pa is None:
                    continue
                ca = np.asarray(m['quad'], float).reshape(4, 2).mean(0)
                for g2, other in list(partners.items()) + \
                        [(g2, mm) for g2, mm in gd.items() if g2 > g]:
                    if g2 == g:
                        continue
                    if isinstance(other, dict) and 'seed_quad' in other:
                        pb = pnp_pos_body(other['quad'])
                        cb = np.asarray(other['quad'], float).reshape(4, 2).mean(0)
                        sb, rb = other['size_px'], other['range_m']
                    else:
                        pb = np.asarray(other['pos_body'], float)
                        cb = np.asarray(other['quad'], float).reshape(4, 2).mean(0)
                        sb, rb = float(other['size_px']), float(other['range_m'])
                    if pb is None:
                        continue
                    v = pb - pa
                    d = float(np.linalg.norm(v))
                    if d < 1.0:
                        continue
                    # same-object guard, as mapedges.mint_session
                    if float(np.linalg.norm(ca - cb)) < 0.6 * max(m['size_px'], sb) \
                            and abs(m['range_m'] - rb) < 5.0:
                        continue
                    ga, gb = (g, g2) if g < g2 else (g2, g)
                    if ga != g:
                        v = -v
                    key = (ga, gb)
                    corro = key in accepted_pairs      # dropped channels stay tagged
                    rows.append({'session': S.name, 'i': i, 'fid': int(S.fid[i]),
                                 'file': S.frames[i]['file'], 'pair': key,
                                 'v_body': v.tolist(), 'd': d,
                                 'src': 'gatenet' if corro else 'gatenet-dropped'})
    return rows


# ---------------------------------------------------------------------------------------
# stage C: intent / hover / strafe channels, vectors from their audited tiles


def channel_vector_rows(known):
    import torch
    import mapedges_intent as MI
    import mapedges_hover as MH
    import mapedges_strafe as MS
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ME_mod.load_net(dev)
    out = []

    def tiles_to_rows(tiles, tag):
        rows = []
        for t in tiles:
            sess = os.path.basename(os.path.dirname(os.path.dirname(t['path'])))
            ma, mb = t['meas']
            v = np.asarray(mb['pos'], float) - np.asarray(ma['pos'], float)
            rows.append({'session': sess, 'i': t['i'], 'fid': t['fid'],
                         'file': os.path.basename(t['path']),
                         'pair': tuple(t['pair']), 'v_body': v.tolist(),
                         'd': float(np.linalg.norm(v)), 'src': tag})
        return rows

    print('\n--- intent channel (155701)')
    _r, tiles, _s = MI.collect(net=net, dev=dev, known=known, verbose=True)
    out += tiles_to_rows(tiles, 'intent')
    print('\n--- hover channel (link laps)')
    kh = dict(known)
    _r, tiles, _s = MH.collect(net=net, dev=dev, known=kh, apply_audit=True,
                               verbose=True)
    out += tiles_to_rows(tiles, 'hover')
    print('\n--- staged-strafe channel (2026-08-02)')
    _r, tiles, _s = MS.collect(net=net, dev=dev, known=kh, apply_audit=True,
                               verbose=True)
    out += tiles_to_rows(tiles, 'strafe')
    return out


def strafe1516_vector_rows():
    """mapvq2.strafe_rows() with the vector kept. Detections are UNORDERED -> the
    per-row vector sign is arbitrary; tagged mod180 and folded downstream."""
    S = M.Session(M.STRAFE)
    rows = []
    for i, r in enumerate(S.frames):
        if not (M.STRAFE_WINDOW[0] <= S.fid[i] <= M.STRAFE_WINDOW[1]):
            continue
        det = [d for d in S.det.get(r['file'], []) if M.clean(d)]
        if len(det) != 2:
            continue
        v = np.asarray(det[1]['pos_body'], float) - np.asarray(det[0]['pos_body'], float)
        rows.append({'session': S.name, 'i': i, 'fid': int(S.fid[i]),
                     'file': r['file'], 'pair': (15, 16), 'v_body': v.tolist(),
                     'd': float(np.linalg.norm(v)), 'src': 'strafe1516-mod180'})
    return rows


# ---------------------------------------------------------------------------------------
# collect driver


def collect(with_torch=True):
    mapj = json.load(open(MAP_JSON))
    accepted = {tuple(int(x) for x in k.split('-')): v['dist_m']
                for k, v in mapj['measured_pairs'].items()}

    rows = []
    per_res = {}
    for sess in BASE_SESSIONS:
        res = M.run(sess, verbose=False)
        per_res[res['S'].name] = res
        r = contour_vector_rows(res)
        print(f'{res["S"].name}: {len(r)} contour vector rows')
        rows += r
    nr = net_vector_rows(per_res, accepted)
    print(f'gatenet rebuilt rows: {len(nr)} '
          f'({sum(1 for r in nr if r["src"] == "gatenet-dropped")} dropped-channel, '
          f'kept for adjudication only)')
    rows += nr
    rows += strafe1516_vector_rows()
    if with_torch:
        rows += channel_vector_rows(known=accepted)

    # compass join
    comp = {}
    n_ok = n_noconf = n_nocsv = 0
    for r in rows:
        sname = r['session']
        if sname not in comp:
            comp[sname] = Compass(os.path.join(M.SESS, sname))
        C = comp[sname]
        if not C.ok:
            r['bearing'] = None
            r['why'] = 'no-compass-csv'
            n_nocsv += 1
            continue
        S = per_res[sname]['S'] if sname in per_res else None
        t_ns = None
        if S is not None:
            t_ns = float(S.t[r['i']])
        else:
            cr = C.by_file.get(r['file'])
            t_ns = float(cr['t_recv_wall_ns']) if cr else None
        if t_ns is None:
            r['bearing'] = None
            r['why'] = 'no-frame-time'
            n_nocsv += 1
            continue
        terms, why = C.bearing_terms(r['file'], t_ns)
        if terms is None:
            r['bearing'] = None
            r['why'] = why
            n_noconf += 1
            continue
        psi_u, R_lb = terms
        b, dz = grid_bearing(psi_u, R_lb, r['v_body'])
        r['bearing'] = b % 360.0
        r['dz_up'] = dz
        n_ok += 1
    print(f'\ncompass join: {n_ok} rows with bearing, {n_noconf} compass-unconfident, '
          f'{n_nocsv} without CSV')

    json.dump(rows, open(ROWS_JSON, 'w'))
    print(f'wrote {ROWS_JSON} ({len(rows)} rows)')


# ---------------------------------------------------------------------------------------
# solve helpers


def circ_median(deg, period=360.0):
    """Weighted-free circular median via the folded-mean trick used in skylight."""
    a = np.deg2rad(np.asarray(deg, float) * (360.0 / period))
    c = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) * (period / 360.0)
    d = (np.asarray(deg) - c + period / 2) % period - period / 2
    return (c + float(np.median(d))) % period


def circ_mad(deg, centre, period=360.0):
    d = np.abs((np.asarray(deg) - centre + period / 2) % period - period / 2)
    return float(np.median(d))


def wrapd(x, period=360.0):
    return (np.asarray(x, float) + period / 2) % period - period / 2


def solve_positions(constraints, gates):
    """constraints: [(a, b, dx, dy, w)] -> {g: (x, y)} least squares, gate min pinned."""
    idx = {g: k for k, g in enumerate(sorted(gates))}
    m = len(idx)
    A, y = [], []
    for a, b, dx, dy, w in constraints:
        for comp, val in ((0, dx), (1, dy)):
            row = np.zeros(2 * m)
            row[2 * idx[b] + comp] = w
            row[2 * idx[a] + comp] = -w
            A.append(row)
            y.append(w * val)
    g0 = min(idx)
    for comp in (0, 1):
        row = np.zeros(2 * m)
        row[2 * idx[g0] + comp] = 100.0
        A.append(row)
        y.append(0.0)
    X, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    return {g: (float(X[2 * k]), float(X[2 * k + 1])) for g, k in idx.items()}


# EDGES REFUSED AFTER DIRECTION EVIDENCE (2026-08-02, the mapdir build itself).
# Same discipline as mapvq2.REFUSE_ROWS: refusal with the evidence, never correction.
REFUSED_EDGES = {
    (4, 6): 'the 4-6 vector (22.74 m @ 67.7, dz +0.19, n=32, lap 121520 only) differs '
            'from the 4-5 vector (22.46 m @ 66.4, dz -0.31) by 0.32 m -- two named '
            'gates 0.3 m apart are ONE physical object, and its height matches gate 5 '
            '(level with 4), not gate 6 (+4.8 m per the signed 5-6/6-7 ladder). The '
            'sketch corroborates: 4-6 = 1.61 u is the LONG, near-collinear side '
            '(1.01 + 0.71), predicting ~36 m, and the direction solve predicts '
            '37.3 m. Render: vercheck/mapdir_anomalies.png (the "g6" partner is a '
            'floor-level gate on the 4->5 sightline). Distances alone could not see '
            'this: the 22.46/22.74/15.65 triangle closes numerically.',
}


def solve(write=False):
    rows = json.load(open(ROWS_JSON))
    # THE ONE RELABEL (2026-08-02): the 76 '10-11' rows of the pausing lap saw GATE
    # 12 at the far end. Full evidence -- drag path/chord < 1, single-vantage
    # provenance, and the station-number ruler's rigid step at gate 11 -- is in
    # mapvq2.RELABEL_ROWS. Applied here so that re-running --solve --write
    # reproduces the corrected map instead of silently reverting it.
    nrel = 0
    for sname, old_p, lo, hi, new_p, _why in M.RELABEL_ROWS:
        for r in rows:
            if (r['session'] == sname and tuple(r['pair']) == tuple(old_p)
                    and lo <= r['fid'] <= hi):
                r['pair'] = list(new_p)
                nrel += 1
    if nrel:
        print('RELABEL_ROWS: %d direction rows renamed (see mapvq2.RELABEL_ROWS)'
              % nrel)
    mapj = json.load(open(MAP_JSON))
    accepted = {tuple(int(x) for x in k.split('-')): v['dist_m']
                for k, v in mapj['measured_pairs'].items()}
    horiz = {tuple(int(x) for x in k.split('-')):
             (v['horiz_m'] if v['horiz_m'] is not None else v['dist_m'])
             for k, v in mapj['measured_pairs'].items()}

    # ---- NET-CHANNEL CORROBORATION, per pair (final4's own gate, reproduced).
    # A net row inherits track identity; a contaminated name gives a tight wrong
    # measurement. The whole per-pair net channel is dropped unless its median agrees
    # with an independent channel (contour/intent/hover/strafe) within 3 m. Row-level
    # d-filtering is NOT enough: the tiebreak "1-2" net rows span 4.4-24.4 m and some
    # land within 3 m of 8.32 by accident.
    net_med = collections.defaultdict(list)
    oth_med = collections.defaultdict(list)
    for r in rows:
        p = tuple(r['pair'])
        if r['src'] == 'gatenet':
            net_med[p].append(r['d'])
        elif r['src'] != 'gatenet-dropped':
            oth_med[p].append(r['d'])
    net_bad = set()
    for p, ds in net_med.items():
        m = float(np.median(ds))
        o = float(np.median(oth_med[p])) if p in oth_med else None
        if o is None or abs(m - o) > 3.0:
            net_bad.add(p)
    if net_bad:
        print('net channels dropped whole (median vs independent channel > 3 m): '
              + ', '.join('%d-%d' % p for p in sorted(net_bad)))

    # usable direction rows: bearing present, channel corroborated, d near accepted
    use, rej_d = [], collections.Counter()
    for r in rows:
        if r.get('bearing') is None:
            continue
        p = tuple(r['pair'])
        if r['src'] == 'gatenet-dropped' or (r['src'] == 'gatenet' and p in net_bad):
            continue
        if p not in accepted or abs(r['d'] - accepted[p]) > D_TOL:
            rej_d[p] += 1
            continue
        use.append(r)
    print(f'direction rows usable: {len(use)}; rejected off-median: '
          f'{dict(sorted(rej_d.items()))}')

    # ---- per (pair, session): median bearing (mod 360; mod 180 for the unordered edge)
    ps = collections.defaultdict(list)
    for r in use:
        ps[(tuple(r['pair']), r['session'])].append(r)
    meas = {}
    for (p, s), rs in sorted(ps.items()):
        mod180 = all(r['src'] == 'strafe1516-mod180' for r in rs)
        period = 180.0 if mod180 else 360.0
        bb = [r['bearing'] % period for r in rs]
        c = circ_median(bb, period)
        dz = [r['dz_up'] for r in rs if r.get('dz_up') is not None]
        meas[(p, s)] = {'pair': p, 'session': s, 'n': len(rs),
                        'bearing': c, 'mad': circ_mad(bb, c, period),
                        'd': float(np.median([r['d'] for r in rs])),
                        'dz': float(np.median(dz)) if dz else None,
                        'mod180': mod180,
                        'thin': len(rs) < 3}       # n<3: reported, never a constraint
    print('\n--- per (pair, session) direction measurements (session-frame bearings)')
    for k, v in sorted(meas.items()):
        print('  %2d-%-2d %-38s n=%4d  bearing %7.2f  MAD %5.2f%s%s'
              % (v['pair'][0], v['pair'][1], v['session'][-24:], v['n'],
                 v['bearing'], v['mad'], '  (mod 180)' if v['mod180'] else '',
                 '  THIN (n<3, excluded)' if v['thin'] else ''))

    # ---- branch resolution across sessions -------------------------------------------
    sessions = sorted({s for (_p, s) in meas})
    ref = '20260801-144858-vqual2-lap-pausing'
    # pairwise session deltas over shared pairs (mod 90 residual + integer branch)
    shared = collections.defaultdict(list)
    for (p, s) in meas:
        shared[p].append(s)
    offs = {ref: 0.0}
    prov = {ref: 'reference session'}
    # iterate: attach any session sharing a pair with an attached one
    changed = True
    while changed:
        changed = False
        for s in sessions:
            if s in offs:
                continue
            deltas = []
            for p, ss in shared.items():
                if s not in ss:
                    continue
                for s2 in ss:
                    if s2 in offs and s2 != s:
                        v1, v2 = meas[(p, s)], meas[(p, s2)]
                        if v1['mod180'] or v2['mod180'] or v1['thin'] or v2['thin']:
                            continue    # unordered/thin channels cannot vote on branch
                        deltas.append(((v2['bearing'] + offs[s2]) - v1['bearing'],
                                       p, s2))
            if deltas:
                # each delta fixes the offset mod 360; branch = the 90-deg multiple
                # nearest the consensus. Use the circular median of the raw deltas.
                dd = [d for d, _p, _s2 in deltas]
                c = circ_median(dd)
                k = round(c / 90.0)
                res = wrapd(c - 90.0 * k)
                offs[s] = 90.0 * k
                prov[s] = ('shared pairs %s; residual after branch %.2f deg (n=%d)'
                           % (sorted({p for _d, p, _s2 in deltas}), res, len(deltas)))
                changed = True
    unresolved = [s for s in sessions if s not in offs]

    # sketch prior for anything left: grid->sketch similarity, REFLECTION allowed and
    # reported. The compass frame is z-DOWN (levelled body), the sketch is a top-down
    # view -- the two may differ by a reflection, which no single rotation absorbs.
    P, _mj = M.sketch_xy()

    def sketch_bearing(p, refl):
        v = P[p[1]] - P[p[0]]
        return math.degrees(math.atan2(-v[1] if refl else v[1], v[0]))
    fits = []
    for refl in (False, True):
        samples = [wrapd(sketch_bearing(p, refl) - (v['bearing'] + offs[s]))
                   for (p, s), v in meas.items()
                   if s in offs and not v['mod180'] and not v['thin']
                   and p not in REFUSED_EDGES]
        c = circ_median([x % 360 for x in samples])
        res = np.abs(wrapd(np.array(samples) - c))
        fits.append((float(np.median(res)), refl, c, samples))
    fits.sort()
    rot_med, sk_refl, grid2sketch, rot_samples = fits[0]
    rot_res = np.abs(wrapd(np.array(rot_samples) - grid2sketch))
    print(f'\ngrid->sketch: rotation {grid2sketch:.1f} deg, reflection={sk_refl} '
          f'(|res| median {rot_med:.1f} deg over {len(rot_samples)} pair-sessions; '
          f'other handedness gives {fits[1][0]:.1f} deg)')
    for s in unresolved:
        cands = []
        for k in range(4):
            errs = []
            for (p, s2), v in meas.items():
                if s2 != s or v['thin']:
                    continue
                pred = sketch_bearing(p, sk_refl)
                got = (v['bearing'] + 90.0 * k + grid2sketch)
                # a mod-180 axis still selects its branch mod 90: compare as axes
                errs.append(abs(float(wrapd(pred - got, 180.0))) if v['mod180']
                            else abs(float(wrapd(pred - got))))
            cands.append((float(np.median(errs)) if errs else 1e9, k))
        cands.sort()
        offs[s] = 90.0 * cands[0][1]
        margin = cands[1][0] - cands[0][0] if len(cands) > 1 else 0.0
        prov[s] = ('SKETCH-PRIOR branch: err %.1f deg vs next-best %.1f deg'
                   % (cands[0][0], cands[1][0]))
        print(f'  {s}: sketch-prior branch k={cands[0][1]} '
              f'(err {cands[0][0]:.1f}, margin {margin:.1f} deg)')

    print('\n--- session branch offsets (add to session-frame bearing)')
    for s in sessions:
        print(f'  {s}: +{offs[s]:.0f} deg   [{prov[s]}]')

    # ---- absolute bearings per pair, cross-session agreement --------------------------
    per_pair = collections.defaultdict(list)
    for (p, s), v in meas.items():
        if v['thin'] or p in REFUSED_EDGES:
            continue
        b = (v['bearing'] + offs[s]) % (180.0 if v['mod180'] else 360.0)
        per_pair[p].append({**v, 'abs': b})
    for p, why in REFUSED_EDGES.items():
        print(f'\nEDGE REFUSED {p[0]}-{p[1]}: {why}')
    print('\n--- absolute grid-frame pair directions (deg), cross-session agreement')
    final = {}
    xsess_spread = []
    for p in sorted(per_pair):
        vs = per_pair[p]
        full = [v for v in vs if not v['mod180']]
        halfs = [v for v in vs if v['mod180']]
        if full and halfs:
            # the axis channel carries the precision (n=433), the full channel the
            # sign and branch: unfold the axis into the semicircle of the full median
            bf = circ_median([v['abs'] for v in full])
            ax = circ_median([v['abs'] for v in halfs], 180.0)
            c = ax if abs(float(wrapd(bf - ax))) <= 90.0 else (ax + 180.0) % 360.0
            spread = abs(float(wrapd(bf - c)))
            mode = 'axis+sign-from-lap'
        elif full:
            bb = [v['abs'] for v in full]
            ww = [v['n'] for v in full]
            c = circ_median(np.repeat(bb, np.clip(ww, 1, 50)))
            spread = float(np.max(np.abs(wrapd(np.array(bb) - c)))) if len(bb) > 1 else None
            mode = 'full'
        else:
            c = circ_median([v['abs'] for v in halfs], 180.0)
            spread = None
            mode = 'mod180'
        final[p] = {'bearing': c, 'mode': mode, 'n_sessions': len(vs),
                    'n_rows': sum(v['n'] for v in vs),
                    'row_mad': float(np.median([v['mad'] for v in vs])),
                    'xsess_spread': spread}
        if spread is not None:
            xsess_spread.append(spread)
        print('  %2d-%-2d  %7.2f deg  sessions %d  rows %4d  row-MAD %4.2f  '
              'x-session spread %s%s'
              % (p[0], p[1], c, len(vs), final[p]['n_rows'], final[p]['row_mad'],
                 '%.2f' % spread if spread is not None else '  --',
                 {'mod180': '  (axis only: mod 180)',
                  'axis+sign-from-lap': '  (axis n>>; sign from lap channel)'}.get(
                      mode, '')))
    if xsess_spread:
        print('  >> cross-session bearing spread: median %.2f deg  max %.2f deg '
              '(pairs measured in 2+ sessions)'
              % (float(np.median(xsess_spread)), float(np.max(xsess_spread))))

    def constraints_from(final_map, skip=()):
        cons = []
        for p, v in final_map.items():
            if p in skip or v['mode'] == 'mod180':
                continue
            d = horiz[p]
            b = math.radians(v['bearing'])
            w = math.sqrt(min(v['n_rows'], 100)) / (0.5 + v['row_mad'])
            cons.append((p[0], p[1], d * math.cos(b), d * math.sin(b), w))
        return cons

    # resolve any still-unsigned mod-180 edge against the sketch side, stated as such
    cons = constraints_from(final)
    pos = solve_positions(cons, sorted({g for c in cons for g in c[:2]}))
    for p, v in final.items():
        if v['mode'] != 'mod180' or p[0] not in pos:
            continue
        best = None
        for sgn in (0.0, 180.0):
            bb = math.radians(v['bearing'] + sgn)
            cand = (pos[p[0]][0] + horiz[p] * math.cos(bb),
                    pos[p[0]][1] + horiz[p] * math.sin(bb))
            skb = math.radians(sketch_bearing(p, sk_refl) - grid2sketch)
            ref = (pos[p[0]][0] + horiz[p] * math.cos(skb),
                   pos[p[0]][1] + horiz[p] * math.sin(skb))
            err = math.hypot(cand[0] - ref[0], cand[1] - ref[1])
            if best is None or err < best[0]:
                best = (err, sgn)
        final[p]['bearing'] = (v['bearing'] + best[1]) % 360.0
        final[p]['mode'] = 'mod180-resolved-by-sketch'
        print(f'\n  {p[0]}-{p[1]} axis sign resolved by SKETCH side (stated, '
              f'not measured); bearing {final[p]["bearing"]:.2f} deg')
    # ---- the 10-11 edge, from the session flown for it ---------------------------------
    # 20260802-180755-vm-strafe-10-11: markers 1-2 / 3-4 / 5-6 / 7-8 delimit four staged
    # vantages on the 10/11 pair. Measured there (mapedges_strafe1011.py, its own skylight
    # compass, 97% confident frames):
    #     10-11  horiz 16.49 m  MAD 0.28  n=96   three vantages: 16.24 / 16.66 / 16.65
    #     10-12  horiz 34.97 m  MAD 0.50  n=156  four  vantages: 35.29/35.15/34.95/34.80
    #     11-12  horiz 19.20 m  n=52  -- REPRODUCING the map's independent 19.67 m (n=28,
    #            different session) to 0.5 m, which is what NAMES gate 11.
    # The session carries its own mod-90 branch, so only the ANGLE BETWEEN the two edges
    # transfers, not an absolute bearing. In its grid frame 10-11 is +13.59 deg from 10-12;
    # the same construction predicts 11-12 at 216.92 deg against the map's independently
    # measured 217.63 -- 0.7 deg, which is what says the transfer is sound. The presentation
    # offset is applied in the GRID frame, where both sessions' bearings live, so no
    # mirror enters: the map's own 11-12 minus 10-12 is -12.48 deg and this session's
    # is -11.78 deg, i.e. the two grid frames already share a handedness.
    STRAFE1011_DEG = +13.59
    if (10, 12) in final and (10, 11) not in final:
        b12 = final[(10, 12)]
        final[(10, 11)] = {'bearing': (b12['bearing'] + STRAFE1011_DEG) % 360.0,
                           'mode': 'strafe1011-angle-from-10-12',
                           'n_sessions': 1, 'n_rows': 96, 'row_mad': 0.26,
                           'xsess_spread': None}
        print('')
        print('10-11 added from 20260802-180755-vm-strafe-10-11: horiz %.2f m, bearing '
              '%.2f deg (%.2f from 10-12)'
              % (horiz[(10, 11)], final[(10, 11)]['bearing'], STRAFE1011_DEG))

    cons = constraints_from(final)
    gates = sorted({g for c in cons for g in c[:2]})
    pos = solve_positions(cons, gates)

    # residuals
    print('\n--- vector-solve residuals per pair (measured vector vs solved offset)')
    vres = []
    for a, b, dx, dy, _w in cons:
        ex = pos[b][0] - pos[a][0] - dx
        ey = pos[b][1] - pos[a][1] - dy
        e = math.hypot(ex, ey)
        vres.append(((a, b), e))
        print('  %2d-%-2d  |residual| %5.2f m' % (a, b, e))
    ev = np.array([e for _p, e in vres])
    print('  >> median %.2f m  p90 %.2f m over %d vector constraints'
          % (float(np.median(ev)), float(np.percentile(ev, 90)), len(ev)))

    # loop closures over independent cycles
    print('\n--- loop closures (sum of measured vectors around measured cycles)')
    fedges = {p: (horiz[p] * math.cos(math.radians(final[p]['bearing'])),
                  horiz[p] * math.sin(math.radians(final[p]['bearing'])))
              for p in final if 'mod180' != final[p]['mode']}
    for cyc in [(4, 5, 6), (7, 8, 9), (10, 11, 12), (12, 13, 14), (13, 14, 15)]:
        vs, ok = [], True
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            p, sgn = ((a, b), 1.0) if a < b else ((b, a), -1.0)
            if p not in fedges:
                ok = False
                break
            vs.append((sgn * fedges[p][0], sgn * fedges[p][1]))
        if not ok:
            continue
        cx, cy = sum(v[0] for v in vs), sum(v[1] for v in vs)
        per = sum(horiz[(min(a, b), max(a, b))]
                  for a, b in zip(cyc, cyc[1:] + cyc[:1]))
        print('  cycle %-14s closure %5.2f m  (perimeter %5.1f m, %.1f%%)'
              % (cyc, math.hypot(cx, cy), per, 100 * math.hypot(cx, cy) / per))

    # leave-one-edge-out WITH directions
    print('\n--- leave-one-edge-out with directions (edge fully removed)')
    loo = []
    for p in sorted(final):
        sub = constraints_from(final, skip=(p,))
        gsub = sorted({g for c in sub for g in c[:2]})
        # connectivity check
        adj = collections.defaultdict(set)
        for c in sub:
            adj[c[0]].add(c[1])
            adj[c[1]].add(c[0])
        seen = {p[0]}
        stack = [p[0]]
        while stack:
            u = stack.pop()
            for w2 in adj[u]:
                if w2 not in seen:
                    seen.add(w2)
                    stack.append(w2)
        if p[1] not in seen:
            print('  %2d-%-2d held out -> graph disconnects (bridge): unpredictable, '
                  'stated rather than scored' % p)
            loo.append({'pair': list(p), 'bridge': True})
            continue
        pp = solve_positions(sub, gsub)
        dx = pp[p[1]][0] - pp[p[0]][0]
        dy = pp[p[1]][1] - pp[p[0]][1]
        dpred = math.hypot(dx, dy)
        bpred = math.degrees(math.atan2(dy, dx))
        derr = dpred - horiz[p]
        berr = float(wrapd(bpred - final[p]['bearing']))
        loo.append({'pair': list(p), 'bridge': False, 'd_pred': dpred,
                    'd_meas': horiz[p], 'd_err': derr, 'b_err': berr})
        print('  %2d-%-2d held out -> d %6.2f vs measured %6.2f (err %+5.2f m), '
              'bearing err %+6.2f deg'
              % (p[0], p[1], dpred, horiz[p], derr, berr))
    de = np.array([abs(r['d_err']) for r in loo if not r['bridge']])
    if len(de):
        print('  >> non-bridge |d err| median %.2f m  p90 %.2f m  (n=%d; distance-only '
              'LOO was median 9.47 m with 4 unplaceable hinges)'
              % (float(np.median(de)), float(np.percentile(de, 90)), len(de)))

    # ---- signed heights from the direction rows ---------------------------------------
    # The intent/hover/strafe channels published |dz| only, but their audited track
    # labels ORDER the pair, so the sign of dz_up is meaningful here -- it is what the
    # per-window render audits verified. Solved as a linear system, gauge gate 0.
    dz_pair = collections.defaultdict(list)
    for r in use:
        if r.get('dz_up') is not None:
            dz_pair[tuple(r['pair'])].append(r['dz_up'])
    zcons = []
    print('\n--- signed height steps (up-positive, b minus a) from direction rows')
    for p in sorted(dz_pair):
        if p in REFUSED_EDGES:
            continue
        zz = np.array(dz_pair[p])
        if len(zz) < 3:
            continue
        med = float(np.median(zz))
        mad = float(np.median(np.abs(zz - med)))
        zcons.append((p, med, mad, len(zz)))
        print('  %2d-%-2d  dz %+6.2f m  MAD %4.2f  n=%4d' % (p[0], p[1], med, mad,
                                                             len(zz)))
    # signed dz for 10-11 from the staged-vantage session (gravity-referenced, four
    # windows, MAD 0.10 m). It over-determines gate 11's height against the
    # 10-12 / 11-12 pair, which is the point.
    if not any(p == (10, 11) for p, *_ in zcons):
        zcons.append(((10, 11), -3.66, 0.10, 96))
        print('  10-11  dz  -3.66 m  MAD 0.10  n=  96   [20260802-180755-vm-strafe-10-11]')
    zg = sorted({g for p, *_ in zcons for g in p})
    zidx = {g: k for k, g in enumerate(zg)}
    Az, yz = [], []
    for p, med, mad, n in zcons:
        row = np.zeros(len(zg))
        row[zidx[p[1]]] = 1.0
        row[zidx[p[0]]] = -1.0
        w = math.sqrt(min(n, 100)) / (0.5 + mad)
        Az.append(row * w)
        yz.append(med * w)
    row = np.zeros(len(zg))
    row[zidx[min(zg)]] = 100.0
    Az.append(row)
    yz.append(0.0)
    zsol, *_ = np.linalg.lstsq(np.array(Az), np.array(yz), rcond=None)
    zres = [(p, float(zsol[zidx[p[1]]] - zsol[zidx[p[0]]]) - med)
            for p, med, _mad, _n in zcons]
    print('  height solve residual: median |r| %.2f m over %d pairs; z range '
          '%.1f..%.1f m' % (float(np.median([abs(e) for _p, e in zres])), len(zres),
                            float(zsol.min()), float(zsol.max())))
    zmap = {g: float(zsol[zidx[g]]) for g in zg}

    # ---- the 1-2 adjudication ---------------------------------------------------------
    print('\n--- 1-2 adjudication: contour 8.32 m channel vs the refused 13.0 m net '
          'channel')
    for tag, sel in (
            ('pausing contour', [r for r in rows if tuple(r['pair']) == (1, 2)
                                 and r['src'] == 'contour']),
            ('tiebreak gatenet', [r for r in rows if tuple(r['pair']) == (1, 2)
                                  and r['src'].startswith('gatenet')
                                  and '1213-1214' in r['session']])):
        wb = [r for r in sel if r.get('bearing') is not None]
        dd = np.array([r['d'] for r in sel])
        if not len(dd):
            print(f'  {tag}: no rows')
            continue
        line = ('  %-16s n=%3d  d min-med-max %5.2f %5.2f %5.2f'
                % (tag, len(dd), dd.min(), float(np.median(dd)), dd.max()))
        if wb:
            bb = [r['bearing'] % 360 for r in wb]
            c = circ_median(bb)
            line += '  bearing med %7.2f  MAD %5.2f' % (c, circ_mad(bb, c))
        print(line)
    print(
        '  VERDICT: 1-2 = 8.32 m STANDS; the 13.0 m alternative is REFUTED at the\n'
        '  source. The "13.0 net channel" is one contaminated population, not a\n'
        '  measurement: its 29 rows span 4.4-24.4 m across vantages (three lumps),\n'
        '  and two static gates cannot change separation with viewpoint. It lives\n'
        '  entirely in the tiebreak opening dash, where REFUSE_ROWS already showed\n'
        '  backward extensions sliding across gates 0/1/2 (rowcheck_tiebreak_open).\n'
        '  The pausing contour channel is tight in d (7.5-9.7) AND in direction\n'
        '  (bearing MAD ~2 deg over 5 rows) -- viewpoint-consistent, i.e. a real\n'
        '  static pair. Directions cannot close it by loop (1-2 is a graph bridge),\n'
        '  so it stays the thinnest edge: n=5, one session; the co-visible 15-20 m\n'
        '  pass is still wanted.')

    # ---- gate plane angles (mod 180), branch offsets applied per session --------------
    gate_angles = {}
    if os.path.exists(ANGLES_JSON):
        araw = json.load(open(ANGLES_JSON))
        print('\n--- gate plane angles in the grid frame (normal azimuth, mod 180)')
        for gs, rows_a in sorted(araw.items(), key=lambda kv: int(kv[0])):
            vals = [(r['az_mod180'] + offs.get(r['session'], 0.0)) % 180.0
                    for r in rows_a if r['session'] in offs]
            if len(vals) < 5:
                continue
            c = circ_median(vals, 180.0)
            mad = circ_mad(vals, c, 180.0)
            okc = mad <= 20.0
            if okc:
                gate_angles[int(gs)] = {'az_mod180': c, 'mad': mad, 'n': len(vals)}
            print('  gate %2s: normal az %7.2f deg  MAD %5.2f  n=%4d%s'
                  % (gs, c, mad, len(vals),
                     '' if okc else '   <-- REFUSED (IPPE head-on valley), not drawn'))

    # ---- output -----------------------------------------------------------------------
    # rotate into sketch-aligned axes for presentation: x = along-hangar (station
    # direction), y = across. The rotation is the fitted grid->sketch angle; its
    # residual vs a multiple of 90 measures the floor-grid alignment claim.
    th = math.radians(grid2sketch)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    # the fitted grid->sketch transform includes a y-MIRROR when sk_refl (the levelled
    # grid frame is z-DOWN, the sketch is a top-down view; presentation only -- the
    # grid frame itself has no reflection freedom)
    mir = np.array([1.0, -1.0]) if sk_refl else np.array([1.0, 1.0])

    def to_sketch_frame(xy):
        return (mir * (R @ np.asarray(xy, float))).tolist()

    def bearing_sketch(b):
        return ((-(b + grid2sketch)) if sk_refl else (b + grid2sketch)) % 360.0

    pos_out = {g: to_sketch_frame(pos[g]) for g in pos}

    layout = {
        'what': 'First direction-accurate 17-gate layout: pair separations as VECTORS '
                '(distance from the accepted pair medians + absolute grid-frame '
                'bearing from the skylight compass), solved as a linear system. '
                'NO reflection ambiguity remains (the compass frame is handed); '
                'global orientation is known up to the grid\'s 4-fold symmetry, '
                'presented in sketch-aligned axes via the fitted grid->sketch '
                'rotation.',
        'frame': {'x': 'along-hangar (sketch along axis), metres',
                  'y': 'across-hangar, metres',
                  'z': 'metres UP from gravity, gauge gate 0 = 0; solved from the '
                       'SIGNED dz of the direction rows (see heights_note)',
                  'origin': 'gate 0',
                  'grid_to_sketch_rotation_deg': grid2sketch,
                  'grid_to_sketch_reflection': bool(sk_refl),
                  'rotation_residual_median_deg': float(np.median(rot_res)),
                  },
        'positions_m': {str(g): {'x': round(pos_out[g][0], 2),
                                 'y': round(pos_out[g][1], 2),
                                 'z_up_m': None if g not in zmap
                                 else round(zmap[g], 2)} for g in sorted(pos_out)},
        'heights_note': 'z from SIGNED dz of the direction rows (the audited channel '
                        'track labels order each pair, so hover/strafe/intent dz '
                        'carries sign here, unlike the |dz|-only rows in '
                        'measured_pairs); gauge gate 0 = 0. Height-solve residual '
                        'median %.2f m.' % float(np.median([abs(e) for _p, e in zres])),
        'pair_dz_up_m': {'%d-%d' % p: {'dz': round(med, 2), 'mad': round(mad, 2),
                                       'n': n} for p, med, mad, n in zcons},
        'refused_edges': {'%d-%d' % p: why for p, why in REFUSED_EDGES.items()},
        'adjudication_1_2': '8.32 m STANDS. The rival 13.0 m gatenet channel is one '
                            'contaminated population (29 rows spanning 4.4-24.4 m '
                            'across vantages -- impossible for a static pair), '
                            'confined to the tiebreak opening dash whose identity '
                            'slides REFUSE_ROWS already documented. The 8.32 contour '
                            'channel is viewpoint-consistent in distance AND bearing '
                            '(MAD ~2 deg). Still the thinnest edge: n=5, one session, '
                            'bridge in the graph (no loop can check it).',
        'pair_bearings_grid_deg': {'%d-%d' % p: {
            'bearing_sketch_frame': round(bearing_sketch(final[p]['bearing']), 2),
            'mode': final[p]['mode'], 'n_rows': final[p]['n_rows'],
            'n_sessions': final[p]['n_sessions'],
            'row_mad_deg': round(final[p]['row_mad'], 2),
            'cross_session_spread_deg':
                None if final[p]['xsess_spread'] is None
                else round(final[p]['xsess_spread'], 2)} for p in sorted(final)},
        'residuals': {'vector_median_m': float(np.median(ev)),
                      'vector_p90_m': float(np.percentile(ev, 90)),
                      'cross_session_bearing_spread_median_deg':
                          float(np.median(xsess_spread)) if xsess_spread else None,
                      'loo_with_directions': loo},
        'gate_plane_angles_deg': {str(g): {
            'normal_az_sketch_frame_mod180':
                round(bearing_sketch(v['az_mod180']) % 180.0, 2),
            'mad_deg': round(v['mad'], 2), 'n': v['n']}
            for g, v in sorted(gate_angles.items())},
        'quadrant_provenance': {s: prov[s] for s in sessions},
        'session_branch_offsets_deg': {s: offs[s] for s in sessions},
        'caveats': [
            'Directions are compass-derived (skylight.py); each session carries one '
            'branch choice from the mod-90 quadrant ambiguity. Branches resolved by '
            'shared pairs against the pausing lap are measurement; branches marked '
            'SKETCH-PRIOR rest on the sketch and race-order topology.',
            'The 15-16 edge direction is an AXIS (unordered strafing detections); its '
            'sign rests on sketch/race-order, stated in pair_bearings mode.',
            'Distances are the accepted medians of map_vq2.json; this block adds '
            'directions, it does not re-litigate distances.',
        ],
    }
    print('\n--- layout (sketch-aligned axes, metres, origin gate 0)')
    for g in sorted(pos_out):
        print('  gate %2d: x %7.2f  y %7.2f' % (g, pos_out[g][0], pos_out[g][1]))

    if write:
        mapj['layout_directions_2026_08_02'] = layout
        json.dump(mapj, open(MAP_JSON, 'w'), indent=1)
        print(f'\nwrote {MAP_JSON} (layout_directions_2026_08_02 block)')
    return layout


ANGLES_JSON = os.path.join(HERE, 'mapdir_angles.json')


def collect_angles():
    """Measured gate PLANE angles in the grid frame, mod 180 (normal sign is arbitrary).

    Per clean NAMED detection at a compass-confident frame: the cached PnP plane normal
    (normal_body) is levelled and rotated by psi -> the gate plane's normal azimuth in
    the grid frame. IPPE's rotation is noisy on head-on views (the labelfix gate-0
    control exposed a flat rotation valley), so per-gate numbers carry their MAD and a
    gate with MAD > 20 deg is reported as NOT measured rather than averaged."""
    out = collections.defaultdict(list)
    for sess in BASE_SESSIONS:
        res = M.run(sess, verbose=False)
        S = res['S']
        C = Compass(sess)
        if not C.ok:
            continue
        byframe, _c = ME_mod.named_clean_by_frame(S, res['good'], res['ident'])
        for i, gd in byframe.items():
            terms, _why = C.bearing_terms(S.frames[i]['file'], float(S.t[i]))
            if terms is None:
                continue
            psi_u, R_lb = terms
            for g, d in gd.items():
                n_lev = R_lb @ np.asarray(d['normal_body'], float)
                az = psi_u + math.degrees(math.atan2(n_lev[1], n_lev[0]))
                out[g].append({'session': S.name, 'az_mod180': az % 180.0,
                               'range': float(d['range_m'])})
    json.dump({str(g): v for g, v in out.items()}, open(ANGLES_JSON, 'w'))
    print('wrote %s: %s' % (ANGLES_JSON,
                            {g: len(v) for g, v in sorted(out.items())}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--no-torch', action='store_true',
                    help='collect without the intent/hover/strafe channels')
    ap.add_argument('--angles', action='store_true')
    ap.add_argument('--solve', action='store_true')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    if a.collect:
        collect(with_torch=not a.no_torch)
    if a.angles:
        collect_angles()
    if a.solve:
        solve(write=a.write)


if __name__ == '__main__':
    main()
