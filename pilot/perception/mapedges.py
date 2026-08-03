"""mapedges.py -- mint the missing inter-component edges with gatenet + a PnP gate.

THE GAP THIS CLOSES. mapvq2.py leaves the 17 gates in five internally-consistent
components ([0-2] [3-6] [7-9] [10-13] [14-16]) because at every one of Claire's 25 marker
moments the LINKING gate was in frame but clipped, and clean() refuses clipped contour
quads outright -- correctly, because the contour detector's clipped quads are garbage.
gatenet is an AMODAL corner regressor: trained to place corners where the edge WOULD be,
including outside the image. So exactly the detections mapvq2 refuses are the ones the
net was built for.

WHAT RUNS WHERE. mapvq2's whole pipeline (tracking, crossing-anchored identity, all its
refusals) runs UNCHANGED and its rows are kept as-is. This file only ADDS rows: in marker
windows and near-crossing windows, for gates that are NAMED but have no clean() detection
in the frame, it runs gatenet on a seeded crop and, if the result survives three
independent gates, emits a pair row tagged source='gatenet' into the same aggregation
path (bimodality split, marker referee, MIN_PAIR_OBS -- all of mapvq2.aggregate()).

CROP SEEDING (the caution from TRAINING.md: the net was trained on crops centred by
ground-truth boxes; a bad seed degrades it). The seed is the named track's OWN detection
in that frame -- it exists at the marker moments by construction, since the refusal was
"in frame but clipped", not "not detected". If the track has no detection in the exact
frame, the nearest observation within SEED_MAX_GAP frames is used (marker windows are
hovers, so the box is nearly static). An outer-source seed quad is shrunk by 1500/2700
about its centroid so the crop scale matches the inner aperture the net regresses.
The crop is then built exactly as training's eval path: visible-box -> MARGIN 1.60,
similarity warp, /255 - 0.45 / 0.25 normalisation (compare_net_vs_detector.py's
missing-normalisation bug is NOT copied here).

THREE ACCEPTANCE GATES, each refusing a different failure:
  1. PnP residual (pnpsnap.py): the net emits 8 free numbers but a real gate projection
     has 6 DOF; solvePnP(IPPE_SQUARE) + reproject measures the distance to that manifold.
     pnpsnap measured: residual > 0.3 px flags 84% of catastrophes at a 12.6% flag rate.
     Default gate 0.3 px; sensitivity at 0.5 / 1.0 is reported, not silently used.
  2. Jitter stability: the net runs three times (base crop, S*1.10, S*0.90 with a 5%
     centre shift); the three full-image quads must agree within max(2 px, 3% of size).
     A net answering from a bad seed answers DIFFERENTLY when the seed moves.
  3. Range continuity vs the track's own crossing-anchored profile: the net's PnP range
     must sit within 3 m + 0.7 m/frame of the nearest clean-inner observation of the same
     gate within +/-60 frames. A static gate's range cannot jump; rows with no nearby
     clean observation are kept but counted separately (continuity_unchecked).
Plus mapvq2's own floors: size >= MIN_SIZE_PX, PnP-vs-apparent-size range agreement
<= MAX_RANGE_DISAGREE, pair distance >= 1 m, partner-gate frame-ambiguity refusal.

The claims-ambiguity check is NOT applied to the net gate itself, deliberately: the
track's raw in-frame detection of a clipped gate carries a garbage outer-fallback range,
and refusing the net row for disagreeing with the very reading it exists to replace would
kill the target population by construction. The partner gate keeps the full check.

    python3 pilot/perception/mapedges.py                # full build, writes map_vq2.json
    python3 pilot/perception/mapedges.py --dry          # stats only, no JSON write
    python3 pilot/perception/mapedges.py --resid 0.5    # move the residual gate
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label as L            # noqa: E402
import mapvq2 as M           # noqa: E402
import gatenet as G          # noqa: E402
from pnpsnap import snap     # noqa: E402

CKPT = os.path.join(G.RUNS, 'colab-v3-nw', 'best.pt')

# Claire's 2026-08-01 targeting session: flown SPECIFICALLY at the missing linking pairs.
# 15 markers -- 1 @active 2 (pair 2+3), 2-4 @6 (6+7), 5-8 @7 (7+8), 9-12 @9 (9+10),
# 13-15 @13 (13+14) -- and, better than the brief promised, the session CROSSES gates
# 0-13 (active_gate_index reaches 14), so mapvq2's crossing-anchored identity applies
# unchanged and her per-marker intent is corroborated rather than load-bearing.
# One sim_reset ~16 s in; excluded in mapvq2.EXCLUDE so nothing spans the teleport.
TARGETING = os.path.join(M.SESS, '20260801-155701-vq2-targeting-pairs')

MARKER_PAD_NET_S = 1.5       # net-window pad around markers; wider than mapvq2's 0.75
                             # because the targeting hovers drift (multiple viewpoints)

RESID_PX = 0.3               # pnpsnap: >0.3 px flags 84% of catastrophes @ 12.6% rate
RESID_REPORT = (0.3, 0.5, 1.0)
SEED_MAX_GAP = 15            # frames a seed box may be borrowed across (hover ~ static)
CROSS_WIN = (90, 45)         # frames (before, after) a crossing that count as a window
STAB_PX = 2.0                # jitter stability floor ...
STAB_FRAC = 0.03             # ... and fraction of apparent size
CONT_WIN = 60                # frames searched for a clean-inner range to check continuity
OUTER_TO_INNER = 1500.0 / 2700.0


# ---------------------------------------------------------------------------------------
# net inference on a seeded crop


def load_net(dev):
    net = G.GateNet(1.0).to(dev)
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    net.load_state_dict(ck['model'])
    net.eval()
    return net


def seed_box(det):
    """Detection -> (cx, cy, S) crop params, training's eval geometry."""
    q = np.asarray(det['quad'], np.float32).reshape(4, 2)
    if det['source'] == 'outer':
        c = q.mean(0)
        q = c + (q - c) * OUTER_TO_INNER
    x0, y0, x1, y1 = G.visible_box(q)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = max(x1 - x0, y1 - y0, G.MIN_VIS_PX)
    return cx, cy, s * G.MARGIN


def predict_quads(net, dev, img, cx, cy, S):
    """Net on (base, +10% scale, -10% scale + 5% shift) crops -> three (4,2) px quads."""
    geoms = [(cx, cy, S),
             (cx, cy, S * 1.10),
             (cx + 0.05 * S, cy + 0.05 * S, S * 0.90)]
    xs = []
    for gcx, gcy, gS in geoms:
        crop = G.make_crop(img, gcx, gcy, gS)
        x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        xs.append(x.div_(255.0).sub_(0.45).div_(0.25))
    with torch.no_grad():
        uv = net(torch.stack(xs).to(dev)).cpu().numpy()
    return [G.from_norm(uv[j], *geoms[j]) for j in range(3)]


def net_measure(net, dev, img, det):
    """One seeded inference -> dict with quad, stability, PnP pose, or refusal reason."""
    cx, cy, S = seed_box(det)
    quads = predict_quads(net, dev, img, cx, cy, S)
    q = quads[0].astype(np.float64)
    e = [float(np.linalg.norm(q[(k + 1) % 4] - q[k])) for k in range(4)]
    size = max(e)
    if size < M.MIN_SIZE_PX:
        return {'ok': False, 'why': 'small'}
    if max(e) / max(min(e), 1e-6) > M.MAX_ASPECT:
        return {'ok': False, 'why': 'aspect'}
    stab = max(float(np.linalg.norm(quads[a] - quads[b], axis=1).max())
               for a, b in ((0, 1), (0, 2), (1, 2)))
    if stab > max(STAB_PX, STAB_FRAC * size):
        return {'ok': False, 'why': 'unstable', 'stab': stab}
    snapped, resid, method = snap(q)
    if not np.isfinite(resid):
        return {'ok': False, 'why': 'pnp-fail'}
    # pose exactly as detect.py computes it for inner quads (same OBJ, same K)
    ok, rvec, tvec = cv2.solvePnP(
        np.array([[-L.HALF, L.HALF, 0], [L.HALF, L.HALF, 0],
                  [L.HALF, -L.HALF, 0], [-L.HALF, -L.HALF, 0]], np.float64),
        np.ascontiguousarray(q, np.float64).reshape(4, 1, 2), M.np.array(
            [[L.FX, 0, L.CX], [0, L.FY, L.CY], [0, 0, 1.0]], np.float64), None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return {'ok': False, 'why': 'pnp-fail'}
    t = tvec.reshape(3)
    rng = float(np.linalg.norm(t))
    rng_size = 480.0 / size
    if abs(rng - rng_size) / max(rng, rng_size, 1e-6) > M.MAX_RANGE_DISAGREE:
        return {'ok': False, 'why': 'range-disagree', 'resid': resid}
    return {'ok': True, 'quad': q, 'size_px': size, 'stab': stab, 'resid': resid,
            'pos_body': L.body_to_cam().T @ t, 'range_m': rng, 'range_size_m': rng_size,
            'seed': det}


# ---------------------------------------------------------------------------------------
# candidate frames and gates


def windows(S, cross):
    """Frame indices worth running the net in: marker windows + near-crossing windows."""
    pad = max(MARKER_PAD_NET_S, M.MARKER_PAD_OVERRIDE_S.get(S.name, 0.0))
    out = set(S.marker_frames(pad))
    for _k, ic in cross.items():
        out.update(range(max(0, ic - CROSS_WIN[0]),
                         min(ic + CROSS_WIN[1], len(S.frames))))
    return out


def gate_obs_index(good, ident):
    """gate -> {frame: raw det} over ALL observations of tracks named that gate."""
    gobs = collections.defaultdict(dict)
    for tid, obs in good.items():
        g = ident.get(tid)
        if g is None:
            continue
        for i, d in obs:
            # nearer observation wins a frame collision (fragments were already pruned
            # for >3 m in-frame contradictions by anchor())
            if i not in gobs[g] or d['range_m'] < gobs[g][i]['range_m']:
                gobs[g][i] = d
    return gobs


def named_clean_by_frame(S, good, ident):
    """measure()'s named-clean-detection-per-frame logic, reproduced verbatim:
    same-gate duplicates take min range, and a gate is refused in any frame where its
    claimed detections (clean or not) disagree by >3 m."""
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    claims = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, obs in good.items():
        g = ident.get(tid)
        if g is None:
            continue
        for i, d in obs:
            claims[i][g].append(d['range_m'])
            if M.clean(d):
                raw[i][g].append(d)
    byframe = collections.defaultdict(dict)
    for i, gd in raw.items():
        for g, ds in gd.items():
            cl = claims[i][g]
            if max(cl) - min(cl) > 3.0:
                continue
            rr = [d['range_m'] for d in ds]
            byframe[i][g] = ds[int(np.argmin(rr))]
    return byframe, claims


def clean_range_series(good, ident):
    """gate -> sorted [(frame, range)] over clean-inner observations, for continuity."""
    ser = collections.defaultdict(list)
    for tid, obs in good.items():
        g = ident.get(tid)
        if g is None:
            continue
        for i, d in obs:
            if M.clean(d):
                ser[g].append((i, d['range_m']))
    return {g: sorted(v) for g, v in ser.items()}


# ---------------------------------------------------------------------------------------
# per-session minting


def mint_session(S, res, net, dev, verbose=True):
    """-> (rows, accepted, stats). rows are mapvq2-shaped pair rows, src='gatenet'."""
    good, ident, cross = res['good'], res['ident'], res['cross']
    win = windows(S, cross)
    gobs = gate_obs_index(good, ident)
    byframe, _claims = named_clean_by_frame(S, good, ident)
    series = clean_range_series(good, ident)
    mk = S.marker_frames()

    stats = collections.Counter()
    accepted = []                    # per-frame accepted net measurements
    attempts = []                    # every candidate with its outcome, for diagnosis
    img_cache = {}
    for i in sorted(win):
        fpath = os.path.join(S.path, 'frames', S.frames[i]['file'])
        img = None
        for g in byframe.get(i, {}):
            attempts.append({'i': i, 'g': g, 'fid': int(S.fid[i]), 'why': 'clean',
                             'mk': i in mk, 'session': S.name})
        for g, fobs in gobs.items():
            if g in byframe.get(i, {}):
                continue             # clean contour detection exists: nothing refused
            # seed: own detection in-frame, else nearest within SEED_MAX_GAP
            if i in fobs:
                seed, gap = fobs[i], 0
            else:
                near = [j for j in fobs if abs(j - i) <= SEED_MAX_GAP]
                if not near:
                    continue
                j = min(near, key=lambda j: abs(j - i))
                seed, gap = fobs[j], abs(j - i)
            stats['candidates'] += 1
            if img is None:
                img = img_cache.get(fpath)
                if img is None:
                    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
                    img_cache.clear()
                    img_cache[fpath] = img
            if img is None:
                stats['no-image'] += 1
                continue
            m = net_measure(net, dev, img, seed)
            if not m['ok']:
                stats['refused-' + m['why']] += 1
                attempts.append({'i': i, 'g': g, 'fid': int(S.fid[i]),
                                 'why': m['why'], 'mk': i in mk, 'session': S.name})
                continue
            # continuity vs the gate's own crossing-anchored clean ranges
            ser = series.get(g, [])
            nearby = [(abs(j - i), r) for j, r in ser if abs(j - i) <= CONT_WIN]
            if nearby:
                dgap, rref = min(nearby)
                if abs(m['range_m'] - rref) > 3.0 + M.CLOSING_M_PER_FRAME * dgap:
                    stats['refused-continuity'] += 1
                    attempts.append({'i': i, 'g': g, 'fid': int(S.fid[i]),
                                     'why': 'continuity', 'mk': i in mk,
                                     'session': S.name})
                    continue
                m['cont_checked'] = True
            else:
                m['cont_checked'] = False
                stats['continuity-unchecked'] += 1
            m.update(i=i, g=g, fid=int(S.fid[i]), seed_gap=gap, session=S.name,
                     path=fpath)
            accepted.append(m)
            stats['accepted'] += 1
            attempts.append({'i': i, 'g': g, 'fid': int(S.fid[i]), 'why': 'accepted',
                             'mk': i in mk, 'session': S.name})

    # pair rows: net gate x (clean partner | other net gate), same guards as measure()
    net_by_frame = collections.defaultdict(dict)
    for m in accepted:
        cur = net_by_frame[m['i']].get(m['g'])
        if cur is None or m['resid'] < cur['resid']:
            net_by_frame[m['i']][m['g']] = m
    rows = []
    for i, gd in net_by_frame.items():
        gh = S.gravity(i)
        partners = dict(byframe.get(i, {}))          # gate -> clean det
        for g, m in gd.items():
            for g2, other in list(partners.items()) + \
                    [(g2, mm) for g2, mm in gd.items() if g2 > g]:
                if g2 == g:
                    continue
                pa, pb = m['pos_body'], other['pos_body']
                v = pb - pa
                d = float(np.linalg.norm(v))
                if d < 1.0:
                    stats['pair-sub-metre'] += 1
                    continue
                # SAME-OBJECT GUARD, learned from this file's own first dry run: gates
                # 11 and 13 came out 1.70 m apart (n=3, marker rows) -- impossible, the
                # frames are 2.7 m wide, so two centres under ~3 m would interpenetrate.
                # Two seeds had landed on ONE physical gate and PnP noise separated the
                # duplicate readings by 1.7 m, sailing past the 1 m floor. A duplicate
                # shows as: same place in the IMAGE and same range. Genuine overlapping
                # pairs (near gate framing a far one, e.g. 0-1's far rows) differ in
                # range by many metres and are untouched.
                ca = np.asarray(m['quad'], np.float64).reshape(4, 2).mean(0)
                cb = (np.asarray(other['quad'], np.float64).reshape(4, 2).mean(0)
                      if 'quad' in other else np.asarray(other['centre'], np.float64))
                sa = m['size_px']
                sb = float(other['size_px'])
                px_apart = float(np.linalg.norm(ca - cb))
                if px_apart < 0.6 * max(sa, sb) and abs(
                        m['range_m'] - other['range_m']) < 5.0:
                    stats['pair-same-object'] += 1
                    continue
                dz = float(-(v @ gh)) if gh is not None else np.nan
                h = float(np.sqrt(max(d * d - dz * dz, 0.0))) if gh is not None else np.nan
                ga, gb = (g, g2) if g < g2 else (g2, g)
                if ga != g:
                    dz = -dz if np.isfinite(dz) else dz
                rows.append({'i': i, 'fid': int(S.fid[i]),
                             'pair': (('g', ga), ('g', gb)),
                             'd': d, 'dz': dz, 'h': h, 'mk': i in mk,
                             'ra': float(m['range_m'] if ga == g else other['range_m']),
                             'rb': float(other['range_m'] if ga == g else m['range_m']),
                             'src': 'gatenet', 'resid': m['resid'],
                             'session': S.name})
                stats['pair-rows'] += 1
    if verbose:
        print(f'  [mapedges {S.name[-10:]}] ' +
              '  '.join(f'{k}={v}' for k, v in sorted(stats.items())))
    return rows, accepted, stats, attempts


# ---------------------------------------------------------------------------------------
# main


def main():
    ap = argparse.ArgumentParser()
    # Identity-based rows come from the three sessions whose identity survived frame
    # verification. The TARGETING session is deliberately NOT here: rendering its marker
    # frames (vercheck/targeting_markers.png) showed named fragments migrated onto
    # neighbouring gates during the long drifting hovers (marker 6: a 'g7' fragment on a
    # 19 m gate while the real gate 7 fills the frame; marker 1: 'g2' at 17 m with the
    # 2->3 crossing 7 s out), and the migrated fragments never co-occur with a true one,
    # so the per-gate contradiction pruning cannot catch them. Its evidence enters as
    # INTENT rows instead (mapedges_intent.py): identity from Claire's word per marker,
    # measurement identity-free.
    ap.add_argument('--sessions',
                    default=','.join([M.PRIMARY, M.SECOND, M.PAUSING,
                                      M.LINKLAP, M.TIEBREAK]))
    ap.add_argument('--no-intent', action='store_true',
                    help='skip the targeting-session intent rows')
    ap.add_argument('--resid', type=float, default=RESID_PX)
    ap.add_argument('--out', default=os.path.join(HERE, 'map_vq2.json'))
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--quick', action='store_true', help='skip leave-one-out')
    a = ap.parse_args()
    sessions = a.sessions.split(',')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_net(dev)

    # base map, exactly as mapvq2.main builds it (strafe rows included)
    base = M.build(sessions, verbose=True, strafe=True)

    all_rows, all_acc, all_att = [], [], []
    for sess in sessions:
        name = os.path.basename(os.path.normpath(sess))
        res = base['per'][name]
        rows, acc, _st, att = mint_session(res['S'], res, net, dev)
        all_rows += rows
        all_acc += acc
        all_att += att

    # ---- why is each still-missing linking edge still missing?
    MISSING = [(2, 3), (6, 7), (9, 10), (13, 14)]
    print('\n--- missing-edge diagnosis (both gates over the run windows)')
    byfr = collections.defaultdict(dict)      # (session, i) -> {gate: outcome}
    for t in all_att:
        cur = byfr[(t['session'], t['i'])].get(t['g'])
        if cur is None or t['why'] in ('accepted', 'clean'):
            byfr[(t['session'], t['i'])][t['g']] = t['why']
    for ka, kb in MISSING:
        co = collections.Counter()
        solo = {ka: collections.Counter(), kb: collections.Counter()}
        for _fr, gd in byfr.items():
            oa, ob = gd.get(ka), gd.get(kb)
            if oa is not None and ob is not None:
                co[(oa, ob)] += 1
            elif oa is not None:
                solo[ka][oa] += 1
            elif ob is not None:
                solo[kb][ob] += 1
        print(f'  edge {ka}-{kb}: co-present frames {sum(co.values())}')
        for k, v in co.most_common(6):
            print(f'      both in frame: g{ka}={k[0]:<16s} g{kb}={k[1]:<16s} x{v}')
        for g in (ka, kb):
            top = '  '.join(f'{w}x{c}' for w, c in solo[g].most_common(4))
            print(f'      g{g} alone: {sum(solo[g].values())} frames  ({top})')

    # ---- residual-gate sensitivity (reported at fixed thresholds, applied at --resid)
    print('\n--- residual-gate sensitivity (accepted net measurements / pair rows)')
    for thr in sorted(set(RESID_REPORT + (a.resid,))):
        na = sum(1 for m in all_acc if m['resid'] <= thr)
        nr = sum(1 for r in all_rows if r['resid'] <= thr)
        print(f'    resid <= {thr:.2f} px: {na:4d} measurements, {nr:4d} pair rows')
    use_rows = [r for r in all_rows if r['resid'] <= a.resid]
    use_acc = [m for m in all_acc if m['resid'] <= a.resid]

    # ---- per-edge report on the net rows alone, target edges first
    TARGETS = [(2, 3), (6, 7), (9, 10), (13, 14), (7, 8), (12, 13)]
    print(f'\n--- gatenet rows at resid <= {a.resid} px, by pair (targets flagged *)')
    per = collections.defaultdict(list)
    for r in use_rows:
        per[(r['pair'][0][1], r['pair'][1][1])].append(r)
    for p in sorted(per):
        ds = np.array([r['d'] for r in per[p]])
        rr = np.array([r['resid'] for r in per[p]])
        tag = ' *' if p in TARGETS else ''
        print('    %2d-%-2d n=%3d  d med %6.2f  MAD %5.2f  p10-p90 %5.1f-%-5.1f  '
              'resid med %.3f max %.3f  markers %d%s'
              % (p[0], p[1], len(ds), np.median(ds),
                 np.median(np.abs(ds - np.median(ds))),
                 np.percentile(ds, 10), np.percentile(ds, 90),
                 np.median(rr), rr.max(),
                 sum(r['mk'] for r in per[p]), tag))

    # ---- intent rows from the targeting session (identity = Claire's marker intent)
    intent_rows = []
    if not a.no_intent:
        import mapedges_intent as MI
        print('\n--- intent-anchored rows from %s' % os.path.basename(TARGETING))
        known_med = {(a2[1], b2[1]): v['d'] for (a2, b2), v in base['agg'].items()
                     if a2[0] == 'g' and b2[0] == 'g'}
        intent_rows, intent_tiles, _ist = MI.collect(net=net, dev=dev, known=known_med)
        MI.montage(intent_tiles)

    # ---- hover-intent rows from the 2026-08-01 evening link laps (mapedges_hover.py):
    # same identity discipline as the intent rows (Claire's word + per-window render
    # audit + known-median referee), measurement identity-free. The intent medians join
    # the referee so 13-14 checks against the targeting session's 12.54.
    hover_rows = []
    if not a.no_intent:
        import mapedges_hover as MH
        print('\n--- hover-intent rows from the link laps')
        hov_known = dict(known_med)
        per_int0 = collections.defaultdict(list)
        for r in intent_rows:
            per_int0[(r['pair'][0][1], r['pair'][1][1])].append(r['d'])
        for p, v in per_int0.items():
            hov_known.setdefault(p, float(np.median(v)))
        hover_rows, hover_tiles, _hst = MH.collect(net=net, dev=dev, known=hov_known,
                                                   apply_audit=True)
        MI.montage(hover_tiles, out=os.path.join(HERE, 'vercheck',
                                                 'hover_pairs_montage.png'))

    # ---- staged-strafe intent rows from the 2026-08-02 sessions (mapedges_strafe.py):
    # same discipline again -- Claire's staged pair per session, per-window render
    # audit, known-median referee, measurement identity-free.
    strafe_rows = []
    if not a.no_intent:
        import mapedges_strafe as MS
        print('\n--- staged-strafe intent rows from the 2026-08-02 sessions')
        ms_known = dict(known_med)
        per_prev = collections.defaultdict(list)
        for r in intent_rows + hover_rows:
            per_prev[(r['pair'][0][1], r['pair'][1][1])].append(r['d'])
        for p, v in per_prev.items():
            ms_known.setdefault(p, float(np.median(v)))
        strafe_rows, strafe_tiles, _sst = MS.collect(net=net, dev=dev, known=ms_known,
                                                     apply_audit=True)
        MI.montage(strafe_tiles, out=os.path.join(HERE, 'vercheck',
                                                  'strafe0802_pairs_montage.png'))

    # ---- CORROBORATION GATE on the identity-net rows. The 1-5 lesson (see NOTES):
    # net rows inherit track identity, and a contaminated name produces a tight,
    # low-residual, perfectly stable measurement of the WRONG pair. A net-row pair is
    # kept only if an independent channel agrees with it within 3 m: the contour rows
    # of the same pair, or the intent rows. Uncorroborated net pairs are dropped and
    # listed -- refusal, not correction.
    base_med = {(a2[1], b2[1]): v['d'] for (a2, b2), v in base['agg'].items()
                if a2[0] == 'g' and b2[0] == 'g'}
    per_int = collections.defaultdict(list)
    for r in intent_rows + hover_rows + strafe_rows:   # intent-anchored channels referee
        per_int[(r['pair'][0][1], r['pair'][1][1])].append(r['d'])
    int_med = {p: float(np.median(v)) for p, v in per_int.items()}
    per_net = collections.defaultdict(list)
    for r in use_rows:
        per_net[(r['pair'][0][1], r['pair'][1][1])].append(r)
    kept_net, dropped_pairs = [], []
    for p, rs in sorted(per_net.items()):
        med = float(np.median([r['d'] for r in rs]))
        ok = (p in base_med and abs(med - base_med[p]) <= 3.0) or \
             (p in int_med and abs(med - int_med[p]) <= 3.0)
        if ok:
            kept_net += rs
        else:
            dropped_pairs.append((p, med, len(rs)))
    print('\n--- net-row corroboration: kept %d rows; dropped pairs:' % len(kept_net))
    for p, med, n in dropped_pairs:
        print('    %2d-%-2d med %6.2f m n=%3d  (no contour/intent channel within 3 m)'
              % (p[0], p[1], med, n))
    use_rows = kept_net

    # ---- merged aggregation through mapvq2's own path (bimodality guard intact)
    merged_rows = base['rows'] + use_rows + intent_rows + hover_rows + strafe_rows
    agg = M.aggregate(merged_rows)
    ge = M.gate_report(agg)

    # which gate-gate edges exist ONLY because of net rows?
    base_agg = base['agg']
    minted = sorted((a2[1], b2[1]) for (a2, b2) in agg
                    if a2[0] == 'g' and b2[0] == 'g' and (a2, b2) not in base_agg)
    print(f'\n--- edges minted by gatenet (absent from the contour-only map): {minted}')

    # ---- components + per-component solve, as mapvq2.build does -- plus one honesty
    # check mapvq2 never needed: intent rows carry |dz| only (no sign), so a component
    # fused through them can be HEIGHT-disconnected while distance-connected. Heights in
    # a dz-block that does not contain the gauge gate are minimum-norm output, not
    # measurement; they are reported as None.
    comps = []
    for c in M.components(agg):
        gs = sorted(n[1] for n in c if n[0] == 'g')
        if len(gs) < 2:
            continue
        sub = {k: v for k, v in agg.items() if k[0] in c and k[1] in c}
        nodes = M.node_list(sub)
        z, zres, _zu = M.solve_heights(sub, nodes)
        n2, X, st = M.solve_layout(sub, nodes)
        idx = {n: i for i, n in enumerate(n2)}
        idz = {n: i for i, n in enumerate(nodes)}
        # dz-connectivity from the gauge (lowest-indexed gate of the component)
        zadj = collections.defaultdict(set)
        for (na, nb), v in sub.items():
            if v['dz'] is not None:
                zadj[na].add(nb)
                zadj[nb].add(na)
        gauge = ('g', gs[0])
        zseen, stack = {gauge}, [gauge]
        while stack:
            u = stack.pop()
            for w in zadj[u]:
                if w not in zseen:
                    zseen.add(w)
                    stack.append(w)
        pos = {}
        for g in gs:
            if ('g', g) not in idx:
                continue
            zg = z[idz[('g', g)]] if (z is not None and ('g', g) in zseen) else np.nan
            pos[g] = np.array([X[idx[('g', g)]][0], X[idx[('g', g)]][1], zg])
        comps.append({'gates': gs, 'sub': sub, 'pos': pos, 'stats': st,
                      'zres': zres, 'nodes': n2,
                      'z_unknown': sorted(g for g in gs if ('g', g) in idx
                                          and ('g', g) not in zseen)})
    print('\n--- %d connected components with 2+ gates (was 5 without gatenet rows)'
          % len(comps))
    for c in comps:
        st = c['stats']
        print('    gates %-24s %3d nodes %3d edges  MDS stress med %5.2f m  p90 %5.2f m'
              % (c['gates'], st['n_nodes'], st['n_edges'],
                 st['stress_median_m'], st['stress_p90_m']))

    # ---- scale
    sc, mj = M.scale_check(agg)
    ss = np.array([r['scale'] for r in sc])
    print('\n--- METRIC SCALE from %d identified gate pairs' % len(sc))
    for r in sorted(sc, key=lambda r: -r['h']):
        print('    gates %2d-%-2d  measured %6.2f m  sketch %5.2f u -> %6.2f m/station '
              '(n=%d, MAD %.2f)' % (r['pair'][0], r['pair'][1], r['h'], r['u'],
                                    r['scale'], r['n'], r['mad']))
    if len(ss):
        print('    median %.2f m/station  p10 %.2f  p90 %.2f'
              % (np.median(ss), np.percentile(ss, 10), np.percentile(ss, 90)))

    # ---- leave-one-out on the merged solve
    loo_json = []
    if not a.quick:
        print('\n--- LEAVE-ONE-EDGE-OUT, per component')
        for c in comps:
            for r in sorted(M.loo_edges(c['sub'], M.node_list(c['sub'])),
                            key=lambda r: -abs(r['err'] or 0)):
                if r['err'] is None:
                    print('    %2d-%-2d held out -> hinge (gate unplaceable)' % r['pair'])
                else:
                    print('    %2d-%-2d measured %6.2f m  predicted %6.2f m  err %+6.2f m'
                          % (r['pair'][0], r['pair'][1], r['measured'], r['predicted'],
                             r['err']))
                loo_json.append(dict(r, pair=list(r['pair'])))
        e = np.array([abs(r['err']) for r in loo_json if r['err'] is not None])
        if len(e):
            print('    |err| median %.2f m  p90 %.2f m' % (np.median(e),
                                                           np.percentile(e, 90)))

    # dump the accepted measurements + rows for the montage / later inspection,
    # in dry runs too -- the whole point of a dry run is to look at what happened
    acc_out = os.path.join(HERE, 'mapedges_accepted.json')
    json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in m.items() if k not in ('seed', 'pos_body')}
               | {'seed_quad': np.asarray(m['seed']['quad']).tolist(),
                  'seed_source': m['seed']['source']}
               for m in use_acc], open(acc_out, 'w'), indent=1)
    rows_out = os.path.join(HERE, 'mapedges_rows.json')
    json.dump([{**r, 'pair': [list(r['pair'][0]), list(r['pair'][1])]}
               for r in use_rows], open(rows_out, 'w'), indent=1)
    print('\nwrote %s (%d accepted), %s (%d rows)'
          % (acc_out, len(use_acc), rows_out, len(use_rows)))

    if a.dry:
        print('[dry] not writing map JSON')
        return

    # ---- JSON, mapvq2.main()'s schema with the gatenet provenance added
    net_per_pair = collections.defaultdict(int)
    for r in use_rows:
        net_per_pair[(r['pair'][0][1], r['pair'][1][1])] += 1
    intent_per_pair = collections.defaultdict(int)
    for r in intent_rows:
        intent_per_pair[(r['pair'][0][1], r['pair'][1][1])] += 1
    hover_per_pair = collections.defaultdict(int)
    for r in hover_rows:
        hover_per_pair[(r['pair'][0][1], r['pair'][1][1])] += 1
    strafe_per_pair = collections.defaultdict(int)
    for r in strafe_rows:
        strafe_per_pair[(r['pair'][0][1], r['pair'][1][1])] += 1
    status = dict(M.STATUS)
    status['targeting_session_2026_08_01'] = {
        'session': os.path.basename(TARGETING),
        'what': 'Flown at the missing linking pairs (markers: 1=2+3, 2-4=6+7, 5-8=7+8, '
                '9-12=9+10, 13-15=13+14). Crossing-anchored identity RUNS on it but was '
                'REFUSED after frame verification showed named fragments migrating onto '
                'neighbouring gates during the drifting hovers (targeting_markers.png). '
                'Its rows enter as intent-anchored, cluster-rank-labeled measurements '
                'instead (mapedges_intent.py), refereed by the known pair medians.',
        'collinearity_finding': 'Gates 7-8-9 are nearly collinear: 7-8=11.0 + 8-9=15.2 '
                                '= 26.2 = the fast-lap "7-8" mode (26.6). Every 7-8 '
                                'disagreement was the same line under different names.',
    }
    status['staged_strafes_2026_08_02'] = {
        'sessions': ['20260802-004153-vq2-strafe-12-13-14 (REFUSED whole: no ribbon '
                     'renders that far down-course and a 12<->14 swap is inside the '
                     'referee tolerance)',
                     '20260802-005208-vq2-strafe-2-3 (refused: no window with both '
                     'staged gates measurable)',
                     '20260802-005431-vq2-strafe-6-7 (5 markers, all accepted)',
                     '20260802-005725-vq2-2-3-b (marker 1 accepted, 6 refused)',
                     '20260802-010914-vq2-strafe-2-3-c (2 gold windows + 1 marker '
                     'accepted, 7 refused incl. the merged-gate t~98-104 window)'],
        'what_they_settled': [
            '6-7 measured from five hover viewpoints: the partner gate sits ~5.3 m '
            'ABOVE gate 6 (signed height step, all four checked windows), while gate '
            '5 sits 4.8 m BELOW it -- the 202110 "partner was gate 5" trap is '
            'excluded by sign, not just by distance (16.4 vs known 5-6 = 15.4).',
            '2-3 measured from three vantages across two sessions (near gate 1 '
            'hovering, and from beyond gate 3 looking back); gate 2 anchored by its '
            'own crossing at the end of 010914.',
        ],
        'identity': 'Claire\'s staged pair per session name; per-window render audit '
                    '(mapedges_strafe.AUDIT, sheets vercheck/strafe0802_windows_*.png);'
                    ' known-pair-median referee; measurement identity-free.',
    }
    status['gatenet_edges_2026_08_01'] = {
        'what': 'Rows tagged source=gatenet: amodal corner regression (gatenet.py, '
                'colab-v3-nw/best.pt) run on clipped/refused NAMED gates in marker and '
                'near-crossing windows, seeded from the track\'s own detection box, '
                'accepted only past a PnP-residual gate (<= %.2f px), a 3-way crop-jitter '
                'stability check, and range continuity vs the gate\'s crossing-anchored '
                'clean ranges. detect.py contour rows are untouched.' % a.resid,
        'edges_minted': ['%d-%d' % p for p in minted],
        'net_rows_used': len(use_rows),
        'net_measurements_accepted': len(use_acc),
        'caveat': 'gatenet was trained on VQ1 auto-labels only; VQ2 decoration/hangar '
                  'clutter is out of distribution for it, which is why every accepted '
                  'row passed a physics gate rather than a confidence head. Verified by '
                  'rendering accepted quads on frames: '
                  'vercheck/gatenet_edges_montage.png.',
    }
    result = {
        'status': status,
        'source': 'mapedges.py (gatenet edges) on top of mapvq2.py -- vision + gravity, '
                  'no pose stream',
        'sessions': [os.path.basename(os.path.normpath(s)) for s in sessions],
        'frame': {'note': 'Each component has ITS OWN frame -- coordinates are NOT '
                          'comparable across components. Within one: x/y metres, '
                          'horizontal, arbitrary rotation and reflection; z metres UP '
                          'from gravity, lowest-indexed gate of the component = 0.'},
        'components': [{
            'gates': c['gates'],
            'local_positions': {str(g): {'x': float(p[0]), 'y': float(p[1]),
                                         'z_up_m': None if not np.isfinite(p[2])
                                         else float(p[2])}
                                for g, p in c['pos'].items()},
            'z_unknown_gates': c['z_unknown'],
            'layout_stats': c['stats'],
            'height_residual_median_m': None if c['zres'] is None else
                float(np.median(np.abs(c['zres'])))} for c in comps],
        'metres_per_station': {'median': float(np.median(ss)),
                               'p10': float(np.percentile(ss, 10)),
                               'p90': float(np.percentile(ss, 90)),
                               'n_pairs': len(sc),
                               'per_pair': {'%d-%d' % r['pair']: round(r['scale'], 2)
                                            for r in sc},
                               'map_approx_says': mj.get('metres_per_station')},
        'measured_pairs': {'%d-%d' % (a2[1], b2[1]): {
            'n': v['n'], 'dist_m': v['d'], 'horiz_m': v['h'], 'dz_m': v['dz'],
            'mad_m': v['d_mad'], 'n_height': v['n_h'],
            'n_marker_rows': v.get('n_marker', 0),
            'bimodal_rows_dropped': v.get('bimodal_dropped', 0),
            'n_gatenet_rows': net_per_pair.get((a2[1], b2[1]), 0),
            'n_intent_rows': intent_per_pair.get((a2[1], b2[1]), 0),
            'n_hover_rows': hover_per_pair.get((a2[1], b2[1]), 0),
            'n_strafe_rows': strafe_per_pair.get((a2[1], b2[1]), 0),
            'gatenet_only': (a2, b2) not in base_agg}
            for (a2, b2), v in sorted(agg.items()) if a2[0] == 'g' and b2[0] == 'g'},
    }
    if loo_json:
        e = np.array([abs(r['err']) for r in loo_json if r['err'] is not None])
        result['leave_one_out'] = {'median_abs_err_m': float(np.median(e)),
                                   'p90_abs_err_m': float(np.percentile(e, 90)),
                                   'edges': loo_json}
    with open(a.out, 'w') as fh:
        json.dump(result, fh, indent=1)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
