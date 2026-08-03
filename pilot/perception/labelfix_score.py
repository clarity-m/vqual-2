"""Acceptance test: score autolabels_vq1_v2.json against Claire's 223 triage verdicts.

For each triaged slide (id = <key>#<v1-idx>), classify what v2 did:
    removed    -- instance moved to labelfix_negatives_v2.json (behind-gate);
    corrected  -- corners moved > MOVED_PX (gate-1 pose regeneration);
    unchanged  -- byte-identical corners.
Join is (key, gate): a gate appears at most once per frame, and v2 re-indexes z.

The confusion table asked for: of the 70 N's, how many did v2 fix or remove; of the
153 Y's, how many did v2 wrongly disturb. Unchanged N's are then split by body_rate
(mechanism A is handled by down-weighting, not by editing labels).

    python3 pilot/perception/labelfix_score.py
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOVED_PX = 2.0      # mean corner move above which a slide's label is "corrected"
RATE_T = 1.0        # rad/s; mechanism A's established knee


def inst_map(d):
    return {(key, i['gate']): i for key, lst in d.items() for i in lst}


def main():
    verdicts = json.load(open(os.path.join(HERE, 'triage', 'triage_verdicts.json')))
    manifest = json.load(open(os.path.join(HERE, 'triage', 'manifest.json')))
    items = {it['id']: it for it in manifest['items']}
    v1 = json.load(open(os.path.join(HERE, 'autolabels_vq1.json')))
    v2 = inst_map(json.load(open(os.path.join(HERE, 'autolabels_vq1_v2.json'))))
    neg = inst_map(json.load(open(os.path.join(HERE, 'labelfix_negatives_v2.json'))))

    rows = []
    for vid, verdict in verdicts.items():
        it = items[vid]
        key, idx = it['key'], it['inst']
        old = v1[key][idx]
        jk = (key, old['gate'])
        if jk in neg:
            status, move, rate = 'removed', None, neg[jk].get('body_rate')
        elif jk in v2:
            d = v2[jk]
            move = float(np.linalg.norm(
                np.asarray(d['corners']) - np.asarray(old['corners']), axis=1).mean())
            status = 'corrected' if move > MOVED_PX else 'unchanged'
            rate = d.get('body_rate')
        else:
            status, move, rate = 'missing', None, None
        rows.append({'id': vid, 'v': verdict, 'gate': old['gate'], 'status': status,
                     'move': move, 'rate': rate})

    print(f'{len(rows)} triaged slides scored (MOVED_PX = {MOVED_PX})\n')
    print(f'{"":>12}{"removed":>9}{"corrected":>11}{"unchanged":>11}{"total":>7}')
    for v, name in (('n', "Claire N"), ('y', "Claire Y")):
        rs = [r for r in rows if r['v'] == v]
        c = {s: sum(1 for r in rs if r['status'] == s)
             for s in ('removed', 'corrected', 'unchanged')}
        print(f'{name:>12}{c["removed"]:>9}{c["corrected"]:>11}{c["unchanged"]:>11}'
              f'{len(rs):>7}')

    print('\nN slides by gate x status:')
    for g in sorted({r['gate'] for r in rows}):
        rs = [r for r in rows if r['v'] == 'n' and r['gate'] == g]
        if not rs:
            continue
        c = {s: sum(1 for r in rs if r['status'] == s)
             for s in ('removed', 'corrected', 'unchanged')}
        print(f'  gate {g}: removed {c["removed"]:>2}  corrected {c["corrected"]:>2}  '
              f'unchanged {c["unchanged"]:>2}   of {len(rs)}')

    un_n = [r for r in rows if r['v'] == 'n' and r['status'] == 'unchanged']
    hi = [r for r in un_n if r['rate'] is not None and r['rate'] > RATE_T]
    print(f'\nunchanged N: {len(un_n)}; of those {len(hi)} have body_rate > {RATE_T} '
          f'rad/s (mechanism A, handled by down-weighting)')
    resid = [r for r in un_n if r['rate'] is None or r['rate'] <= RATE_T]
    print(f'RESIDUAL unexplained N (unchanged, rate <= {RATE_T}): {len(resid)}')
    for r in sorted(resid, key=lambda r: (r['gate'], r['id'])):
        print(f'  {r["id"]}  gate {r["gate"]}  rate '
              f'{r["rate"] if r["rate"] is not None else float("nan"):.2f}')

    dist_y = [r for r in rows if r['v'] == 'y' and r['status'] != 'unchanged']
    print(f'\nY slides disturbed: {len(dist_y)}')
    for r in dist_y:
        print(f'  {r["id"]}  gate {r["gate"]}  {r["status"]}  '
              f'move {r["move"] if r["move"] is not None else "-"}'
              f'  rate {r["rate"]}')

    # corrected-N move sizes, for the record
    mv = [r['move'] for r in rows if r['v'] == 'n' and r['status'] == 'corrected']
    if mv:
        print(f'\ncorrected-N move px: median {np.median(mv):.1f}  '
              f'p90 {np.percentile(mv, 90):.1f}')
    json.dump(rows, open(os.path.join(HERE, 'labelfix_score.json'), 'w'), indent=1)
    print('wrote labelfix_score.json')


if __name__ == '__main__':
    main()
