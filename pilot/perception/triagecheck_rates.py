"""Measurement only: does body rate / measured offset predict Claire's triage N's?

Rate at frame time reuses labelgates.py's matching path: frames.csv t_recv_wall_ns ->
nearest imu.csv row by t_wall_ns. |rate| = norm(xgyro, ygyro, zgyro); we take the median
over samples within +/-50 ms (imu ~50+ Hz) with nearest-sample fallback.

    python3 pilot/perception/triagecheck_rates.py
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(os.path.dirname(HERE), 'sessions')

BINS = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
BIN_NAMES = ['<0.2', '0.2-0.5', '0.5-1', '1-2', '>2']


def load_imu(sess):
    ts, gy = [], []
    with open(os.path.join(SESSIONS, sess, 'imu.csv')) as fh:
        for r in csv.DictReader(fh):
            ts.append(float(r['t_wall_ns']))
            gy.append((float(r['xgyro']), float(r['ygyro']), float(r['zgyro'])))
    ts = np.asarray(ts)
    mag = np.linalg.norm(np.asarray(gy), axis=1)
    return ts, mag


def load_frames(sess):
    out = {}
    with open(os.path.join(SESSIONS, sess, 'frames.csv')) as fh:
        for r in csv.DictReader(fh):
            if r['file']:
                out[r['file']] = float(r['t_recv_wall_ns'])
    return out


def rate_at(imu, t_ns):
    ts, mag = imu
    k = np.abs(ts - t_ns) <= 50e6
    if k.any():
        return float(np.median(mag[k]))
    return float(mag[int(np.argmin(np.abs(ts - t_ns)))])


def bin_of(r):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= r < hi:
            return i
    return len(BINS) - 1


def auc(pos, neg):
    """P(score(pos) > score(neg)), ties 0.5. pos = N-slides (what we try to flag)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def nrate_table(rows, title, keyfun):
    print(f'\n{title}')
    groups = {}
    for r in rows:
        groups.setdefault(keyfun(r), []).append(r)
    hdr = f'{"":>10}' + ''.join(f'{b:>14}' for b in BIN_NAMES) + f'{"all":>14}'
    print(hdr)
    for g in sorted(groups, key=str):
        rs = groups[g]
        cells = []
        for i in range(len(BINS)):
            sub = [r for r in rs if r['bin'] == i]
            n = len(sub)
            nn = sum(1 for r in sub if r['v'] == 'n')
            cells.append(f'{100 * nn / n:5.0f}% ({nn}/{n})' if n else f'{"--":>12}')
        nn = sum(1 for r in rs if r['v'] == 'n')
        cells.append(f'{100 * nn / len(rs):5.0f}% ({nn}/{len(rs)})')
        print(f'{str(g):>10}' + ''.join(f'{c:>14}' for c in cells))


def main():
    verdicts = json.load(open(os.path.join(HERE, 'triage', 'triage_verdicts.json')))
    manifest = json.load(open(os.path.join(HERE, 'triage', 'manifest.json')))
    items = {it['id']: it for it in manifest['items']}

    imu_cache, frames_cache = {}, {}
    rows = []
    for vid, v in verdicts.items():
        it = items.get(vid)
        if it is None:
            print(f'  WARNING: verdict {vid} not in manifest')
            continue
        sess = it['session']
        if sess not in imu_cache:
            imu_cache[sess] = load_imu(sess)
            frames_cache[sess] = load_frames(sess)
        fname = it['key'].split('/', 1)[1]
        t = frames_cache[sess].get(fname)
        if t is None:
            print(f'  WARNING: no frame row for {vid}')
            continue
        r = rate_at(imu_cache[sess], t)
        rows.append({'id': vid, 'key': it['key'], 'inst': it['inst'], 'v': v,
                     'group': it['group'], 'gate': it['auto']['gate'],
                     'size_px': it['size_px'], 'session': sess,
                     'rate': r, 'bin': bin_of(r)})

    print(f'{len(rows)} triaged slides with rate '
          f'({sum(1 for r in rows if r["v"] == "n")} N)')

    # Q1: rate bins, overall and per group
    nrate_table(rows, 'N-rate by rate bin (ALL):', lambda r: 'all')
    nrate_table(rows, 'N-rate by rate bin, per GROUP:', lambda r: r['group'])

    npos = [r['rate'] for r in rows if r['v'] == 'n']
    nneg = [r['rate'] for r in rows if r['v'] == 'y']
    print(f'\nrate AUC (N vs Y): {auc(npos, nneg):.3f}   '
          f'median rate N={np.median(npos):.2f}  Y={np.median(nneg):.2f} rad/s')
    print('threshold sweep (flag if rate > T):')
    for T in (0.2, 0.5, 1.0, 1.5, 2.0):
        caught = sum(1 for r in rows if r['v'] == 'n' and r['rate'] > T)
        false = sum(1 for r in rows if r['v'] == 'y' and r['rate'] > T)
        print(f'  T={T:4.1f}  catches {caught}/{len(npos)} N ({100*caught/len(npos):.0f}%)  '
              f'false-flags {false}/{len(nneg)} Y ({100*false/len(nneg):.0f}%)')

    # Q2: gate 1 vs rest, conditioned on rate bin
    nrate_table(rows, 'N-rate by rate bin, gate 1 vs others:',
                lambda r: 'gate1' if r['gate'] == 1 else 'other')
    g1 = [r for r in rows if r['gate'] == 1]
    print(f'\ngate 1: n={len(g1)}, median rate {np.median([r["rate"] for r in g1]):.2f}; '
          f'others median rate '
          f'{np.median([r["rate"] for r in rows if r["gate"] != 1]):.2f} rad/s')
    nrate_table(rows, 'N-rate by rate bin, per GATE:', lambda r: r['gate'])

    # Q3: offsets join
    off = {}
    with open(os.path.join(HERE, 'gatebias_offsets.csv')) as fh:
        for r in csv.DictReader(fh):
            off[(r['key'], int(r['idx']))] = float(np.hypot(float(r['dx']), float(r['dy'])))
    joined = [dict(r, off=off[(r['key'], r['inst'])]) for r in rows
              if (r['key'], r['inst']) in off]
    jn = [r['off'] for r in joined if r['v'] == 'n']
    jy = [r['off'] for r in joined if r['v'] == 'y']
    print(f'\noffset join: {len(joined)}/{len(rows)} slides matched '
          f'({len(jn)} N, {len(jy)} Y)')
    if jn and jy:
        print(f'median |offset|: N={np.median(jn):.2f} px  Y={np.median(jy):.2f} px   '
              f'AUC={auc(jn, jy):.3f}')
        for T in (2.0, 3.0, 4.0, 5.0, 7.0):
            c = sum(1 for x in jn if x > T)
            f = sum(1 for x in jy if x > T)
            print(f'  |off|>{T:3.1f}px  catches {c}/{len(jn)} N ({100*c/len(jn):.0f}%)  '
                  f'false-flags {f}/{len(jy)} Y ({100*f/len(jy):.0f}%)')

    # Q4: dump the 6 lowest-rate N slides (offset shown when known) for the sheet step
    ns = sorted([r for r in rows if r['v'] == 'n'], key=lambda r: r['rate'])
    picks = ns[:6]
    print('\n6 lowest-rate N slides:')
    for r in picks:
        o = off.get((r['key'], r['inst']))
        print(f"  {r['id']}  rate={r['rate']:.3f}  off={'%.2f' % o if o else 'n/a'}  "
              f"gate={r['gate']} group={r['group']} size={r['size_px']:.0f}")
    json.dump([r['id'] for r in picks],
              open(os.path.join(HERE, 'triagecheck_picks.json'), 'w'))


if __name__ == '__main__':
    main()
