"""Validate the skylight compass three independent ways, and render the evidence.

  (a) STATIC: during Claire's marker hovers the aircraft drifts but barely rotates, so
      the grid heading should be constant. Reported per marker: peak-to-peak and std of
      psi over the marker window (confident frames only).
  (b) GYRO: between confident frames, the change in grid heading should equal the
      integral of the gravity-projected canonical yaw rate (gyro is mirrored, x -1 per
      CONVENTIONS.md). Reported: RMS mismatch over frame pairs, the best camera-IMU time
      offset (scanned, since CONVENTIONS.md lists it as unmeasured), and a plot of both
      signals over a rotating segment.
  (c) MAP/ROTATION: a static gate's azimuth IN THE GRID FRAME (psi + levelled body
      bearing of the detection) must stay constant while the aircraft yaws. This is the
      test with teeth: if the compass sign or scale were wrong the gate azimuth would
      sweep by twice the yaw. Also, for co-visible identified pairs, the grid-azimuth
      DIFFERENCE between the two gates is compared against the angle predicted by the
      map's measured pair distance (law of cosines with the two PnP ranges).

    python3 skylight_validate.py            # all checks, uses skylight_<session>.csv
"""

from __future__ import annotations

import collections
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label as L  # noqa: E402
import skylight as SK  # noqa: E402
import vq2cache  # noqa: E402
import mapvq2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.abspath(os.path.join(HERE, '..', 'sessions'))
OUT = os.path.join(HERE, 'vercheck')

MARKER_SESSIONS = [
    ('20260801-144858-vqual2-lap-pausing', 0.75),
    ('20260801-155701-vq2-targeting-pairs', 0.75),
    ('20260801-202110-vq2-23-67-1314', 2.0),
    ('20260801-202923-vq2-1213-1214', 2.0),
]


def wrap90(x):
    return (np.asarray(x, float) + 45.0) % 90.0 - 45.0


def load_skylight(name):
    p = os.path.join(HERE, f'skylight_{name}.csv')
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        r['t'] = float(r['t_recv_wall_ns'])
        r['psi'] = float(r['psi_mod90']) if r['psi_mod90'] else None
        r['conf'] = r['confident'] == '1'
        r['uw'] = float(r['psi_unwrapped']) if r['psi_unwrapped'] else None
        r['nseg'] = int(r['n_seg'])
        r['spread'] = float(r['spread_deg']) if r['spread_deg'] else None
    return rows


def markers(session):
    out = []
    p = os.path.join(SESS, session, 'events.jsonl')
    for line in open(p):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get('kind') == 'marker':
            out.append(float(e['t_wall_ns']))
    return out


# ------------------------------------------------------------------ (a) static markers

def check_static():
    print('=' * 78)
    print('(a) STATIC CHECK: heading constancy over marker hovers')
    allpp = []
    for name, pad in MARKER_SESSIONS:
        rows = load_skylight(name)
        t = np.array([r['t'] for r in rows])
        for k, tm in enumerate(markers(name)):
            lo, hi = np.searchsorted(t, [tm - pad * 1e9, tm + pad * 1e9])
            win = [r for r in rows[lo:hi] if r['conf']]
            if len(win) < 5:
                print(f'  {name} marker {k:2d}: only {len(win)} confident frames, skip')
                continue
            psi = np.array([r['psi'] for r in win])
            c = psi[len(psi) // 2]
            d = wrap90(psi - c)
            pp, sd = float(d.max() - d.min()), float(d.std())
            allpp.append((pp, sd))
            flag = '' if pp < 1.0 else '   <-- exceeds 1 deg'
            print(f'  {name[-24:]} marker {k:2d}: n={len(win):3d}  p-p {pp:5.2f} deg  '
                  f'std {sd:4.2f} deg{flag}')
    if allpp:
        pp = np.array([x[0] for x in allpp])
        sd = np.array([x[1] for x in allpp])
        print(f'  >> {len(allpp)} marker windows: p-p median {np.median(pp):.2f} deg '
              f'(p90 {np.percentile(pp, 90):.2f}), std median {np.median(sd):.3f} deg; '
              f'{int((pp < 1.0).sum())}/{len(pp)} windows under 1 deg p-p')
    return allpp


# ------------------------------------------------------------------ (b) gyro agreement

def check_gyro(name='20260801-144858-vqual2-lap-pausing', plot=True):
    print('=' * 78)
    print('(b) GYRO CHECK: d(heading)/dt vs gravity-projected canonical yaw rate')
    rows = load_skylight(name)
    imu = SK.Imu(os.path.join(SESS, name))
    conf = [r for r in rows if r['conf']]

    def rms_at(offset_ns, min_rate=0.0):
        errs, drs = [], []
        for i in range(1, len(conf)):
            a, b = conf[i - 1], conf[i]
            dt = (b['t'] - a['t']) / 1e9
            if not (0.02 <= dt <= 1.0):
                continue
            dg = imu.yaw_lev_integral(a['t'] + offset_ns, b['t'] + offset_ns)
            if abs(dg) > 40.0:
                continue
            if abs(dg) / dt < min_rate:
                continue
            dm = float(wrap90(b['psi'] - a['psi']))
            errs.append(dm - dg)
            drs.append((dm, dg))
        return (float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan,
                len(errs), drs)

    # sign: the derivation says psi increases nose-right; measure it anyway
    r_pos, n, drs = rms_at(0.0)
    dm = np.array([d[0] for d in drs])
    dg = np.array([d[1] for d in drs])
    corr = float(np.corrcoef(dm, dg)[0, 1])
    slope = float((dm @ dg) / max(dg @ dg, 1e-9))
    print(f'  {name}: {n} confident frame pairs')
    print(f'  corr(d psi_grid, d psi_gyro) = {corr:+.4f}   slope = {slope:+.4f} '
          f'(sign +1 confirmed)' if corr > 0 else
          f'  corr = {corr:+.4f} slope {slope:+.4f}  <-- SIGN PROBLEM')
    # camera-IMU offset scan (listed unmeasured in CONVENTIONS.md)
    best = (np.inf, 0.0)
    for ms in range(-120, 121, 10):
        r, _n, _ = rms_at(ms * 1e6)
        if r < best[0]:
            best = (r, ms)
    r_rot, n_rot, _ = rms_at(best[1] * 1e6, min_rate=20.0)
    print(f'  RMS(frame-to-frame mismatch): {r_pos:.3f} deg at offset 0; '
          f'best offset {best[1]:+d} ms -> {best[0]:.3f} deg')
    print(f'  rotating subset (|rate| > 20 deg/s): RMS {r_rot:.3f} deg over {n_rot} pairs')

    if plot:
        # a segment with real rotation: cumulative grid heading vs pure gyro integral
        t0 = conf[0]['t']
        uw_t = np.array([r['t'] for r in rows if r['uw'] is not None])
        uw = np.array([r['uw'] for r in rows if r['uw'] is not None])
        # pure gyro, anchored at the first confident frame
        gy = [uw[0]]
        for i in range(1, len(uw_t)):
            gy.append(gy[-1] + imu.yaw_lev_integral(uw_t[i - 1], uw_t[i]))
        gy = np.array(gy)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ts = (uw_t - t0) / 1e9
        # choose the busiest 90 s window
        span = uw.max() - uw.min()
        best_w = (0, ts[-1])
        if ts[-1] > 100:
            var, w0 = -1, 0.0
            for s in np.arange(ts[0], ts[-1] - 90, 10):
                m = (ts >= s) & (ts < s + 90)
                if m.sum() > 50 and uw[m].std() > var:
                    var, w0 = uw[m].std(), s
            best_w = (w0, w0 + 90)
        m = (ts >= best_w[0]) & (ts <= best_w[1])
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
        ax[0].plot(ts[m], uw[m], lw=1.0, label='grid heading (unwrapped)')
        ax[0].plot(ts[m], gy[m], lw=0.8, ls='--', label='pure gyro integral')
        ax[0].set_ylabel('heading [deg]')
        ax[0].legend()
        ax[0].set_title(f'{name}: skylight heading vs gyro integral '
                        f'(span {span:.0f} deg total)')
        ax[1].plot(ts[m], (uw - gy)[m], lw=0.8, color='crimson')
        ax[1].set_ylabel('grid - gyro [deg]')
        ax[1].set_xlabel('t [s]')
        out = os.path.join(OUT, 'skylight_gyro_check.png')
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f'  plot -> {out}')
    return r_pos, best, corr


# ------------------------------------------- (c) gate azimuth in the grid frame + map

_RUN_CACHE = {}


def _mapvq2_run(name):
    """mapvq2's full identity pass (tracks + crossing anchoring + backward extension),
    cached -- the same machinery the map itself was built and frame-verified with."""
    if name not in _RUN_CACHE:
        _RUN_CACHE[name] = mapvq2.run(os.path.join(SESS, name), verbose=False)
    return _RUN_CACHE[name]


def _named_per_frame(res):
    """frame index -> {race gate: det}, clean detections of identified tracks only,
    frames where one gate is claimed at contradictory ranges refused (measure()'s rule)."""
    named = collections.defaultdict(dict)
    claims = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, obs in res['good'].items():
        if tid not in res['ident']:
            continue
        g = res['ident'][tid]
        for i, d in obs:
            claims[i][g].append(d['range_m'])
            if mapvq2.clean(d):
                cur = named[i].get(g)
                if cur is None or d['range_m'] < cur['range_m']:
                    named[i][g] = d
    for i in list(named):
        for g in list(named[i]):
            cl = claims[i][g]
            if max(cl) - min(cl) > 3.0:
                del named[i][g]
    return named


def check_rotation_azimuth(name='20260801-144858-vqual2-lap-pausing'):
    print('=' * 78)
    print('(c1) ROTATION CHECK: a static gate\'s grid-frame azimuth while the nose sweeps')
    rows = load_skylight(name)
    res = _mapvq2_run(name)
    S = res['S']
    imu = SK.Imu(os.path.join(SESS, name))
    byfile = {r['file']: r for r in rows}
    named = _named_per_frame(res)
    # per gate: time series of (frame, psi_unwrapped, levelled azimuth of the detection)
    series = collections.defaultdict(list)
    for i in sorted(named):
        r = byfile.get(S.frames[i]['file'])
        if r is None or r['uw'] is None or not r['conf']:
            continue
        g, _n = imu.gravity(r['t'])
        if g is None:
            continue
        R_lb = SK.level_rotation(g)
        for gate, d in named[i].items():
            p_lev = R_lb @ np.asarray(d['pos_body'], float)
            az = math.degrees(math.atan2(p_lev[1], p_lev[0]))
            series[gate].append((i, r['uw'], az, float(d['range_m'])))
    results = []
    for gate, ss in sorted(series.items()):
        # contiguous runs (gap <= 15 frames), inside which look for >= 20 deg of sweep
        runs, cur = [], [ss[0]]
        for x in ss[1:]:
            if x[0] - cur[-1][0] <= 15:
                cur.append(x)
            else:
                runs.append(cur)
                cur = [x]
        runs.append(cur)
        for run_ in runs:
            if len(run_) < 12:
                continue
            psi = np.array([x[1] for x in run_])
            if psi.max() - psi.min() < 20.0:
                continue
            az = np.unwrap(np.deg2rad([x[2] for x in run_])) * 180 / np.pi
            grid_az = psi + az
            anti = psi - az           # what a sign error would make constant instead
            results.append({
                'gate': gate, 'fid0': int(S.fid[run_[0][0]]),
                'sweep': float(psi.max() - psi.min()), 'n': len(run_),
                'range': float(np.median([x[3] for x in run_])),
                'std_sum': float(np.std(grid_az)), 'pp_sum': float(np.ptp(grid_az)),
                'std_diff': float(np.std(anti)),
            })
    if not results:
        print('  no usable sweep windows found')
        return results
    for w in results:
        print(f'  gate {w["gate"]:2d} @fid {w["fid0"]}: sweep {w["sweep"]:5.1f} deg, '
              f'n={w["n"]:3d}, range {w["range"]:4.1f} m -> grid azimuth std '
              f'{w["std_sum"]:5.2f} deg (p-p {w["pp_sum"]:5.2f}); wrong-sign std '
              f'{w["std_diff"]:6.2f}')
    s = np.array([w['std_sum'] for w in results])
    a = np.array([w['std_diff'] for w in results])
    print(f'  >> {len(results)} sweeps: grid-azimuth std median {np.median(s):.2f} deg; '
          f'wrong-sign std median {np.median(a):.2f} deg')
    return results


def check_map_pairs(name='20260801-144858-vqual2-lap-pausing'):
    print('=' * 78)
    print('(c2) MAP CHECK: co-visible pair angular separation vs measured_pairs distance')
    m = json.load(open(os.path.join(HERE, 'map_vq2.json')))
    pairs = {tuple(int(x) for x in k.split('-')): v['dist_m']
             for k, v in m['measured_pairs'].items()}
    res = _mapvq2_run(name)
    S = res['S']
    named = _named_per_frame(res)
    errs = []
    for i, gd in named.items():
        if len(gd) < 2:
            continue
        for (ga, da), (gb, db) in [(x, y) for x in gd.items() for y in gd.items()
                                   if x[0] < y[0]]:
            key = (min(ga, gb), max(ga, gb))
            if key not in pairs:
                continue
            ra = float(np.linalg.norm(da['pos_body']))
            rb = float(np.linalg.norm(db['pos_body']))
            dmap = pairs[key]
            cosc = (ra * ra + rb * rb - dmap * dmap) / (2 * ra * rb)
            if not (-1 <= cosc <= 1):
                continue
            pred = math.degrees(math.acos(cosc))
            va = np.asarray(da['pos_body']) / ra
            vb = np.asarray(db['pos_body']) / rb
            meas = math.degrees(math.acos(float(np.clip(va @ vb, -1, 1))))
            errs.append((key, pred, meas, meas - pred))
    if not errs:
        print('  no usable co-visible named pairs')
        return errs
    bypair = collections.defaultdict(list)
    for key, pred, meas, e in errs:
        bypair[key].append(e)
    for key in sorted(bypair):
        e = np.array(bypair[key])
        print(f'  pair {key[0]:2d}-{key[1]:<2d}: n={len(e):3d}  '
              f'angle error median {np.median(e):+5.2f} deg  MAD '
              f'{np.median(np.abs(e - np.median(e))):4.2f}')
    alle = np.array([e for _k, _p, _m, e in errs])
    print(f'  >> {len(alle)} pair-frames: |angle error| median '
          f'{np.median(np.abs(alle)):.2f} deg (p90 {np.percentile(np.abs(alle), 90):.2f})'
          f' -- tests PnP + levelling, heading cancels; kept for honesty')
    return errs


def main():
    os.makedirs(OUT, exist_ok=True)
    check_static()
    check_gyro()
    check_rotation_azimuth()
    check_map_pairs()


if __name__ == '__main__':
    main()
