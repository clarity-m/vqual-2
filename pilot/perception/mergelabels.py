"""Merge VQ2 hand labels into the VQ1 auto-labels to make one training set.

    python3 pilot/perception/mergelabels.py \
        --hand ~/Downloads/labels_gates.json \
        --auto pilot/perception/autolabels_vq1.json \
        --index pilot/perception/vq2_label/index.json \
        --out pilot/perception/labels_merged.json

THE TWO KEY SPACES DIFFER, and it is not only a prefix.

  autolabel.py  keys '<session>/<file>', because it spans sessions.
  labelui.html  keys BASENAME, because it labels one folder at a time -- and the folder
                `vq2select.py` stages has files renamed `NNN_<original>.jpg` so that
                labelui's filename sort puts a balanced sample first. So a hand key is
                `037_00104512.jpg` and has to be mapped back through `index.json` to
                `20260801-121520-vq2-lap-0-15/00104512.jpg`.

`index.json` is therefore REQUIRED rather than optional, and a hand key it cannot resolve
is an error, not a warning: silently dropping labels Claire spent her evening making is
the worst failure this script has available. `--allow-unmapped` downgrades it if a key
genuinely comes from somewhere else.

PROVENANCE, per instance. A hand label and a projected label do not deserve equal weight:
the projection is exact where pose is exact but models only gate-on-gate occlusion and has
no truth for VQ1 race gate 5, while the hand label is amodal by judgement and carries a
real `unsure` flag. Every emitted instance gets

    'src': 'hand' | 'auto'

so a future run can weight, filter or ablate them apart. `--drop-unsure` exists because
the corner-regression loss probably should not see the ones Claire flagged, while a
detection or confidence head probably should.

VERIFIED NEGATIVES SURVIVE THE MERGE. labelui writes `[]` for "reviewed, contains no
labellable gate"; autolabel.py never writes one, because it cannot honestly claim one
(race gate 5 has no truth). An empty array is therefore always a hand fact and is kept as
an empty array -- consumers that treat "key present, list empty" as a hard negative
depend on it, and dropping empty keys would silently delete exactly the negatives the
confidence head is being built for.

COLLISIONS: if a frame is present in both files the HAND labels win outright for that
frame, and the count is reported. Merging two label sets for one image would double-count
gates; a human who looked at the picture beats a projection.

The output schema is the union of both inputs' fields, so nothing autolabel.py adds later
(it is under active edit) is dropped -- unknown keys are copied through untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CORE = ('corners', 'occluded', 'clipped', 'unsure', 'z')


def load_index(path):
    """-> {labelui_key: '<session>/<file>'} from vq2select.py's index."""
    with open(path) as fh:
        d = json.load(fh)
    return {f['labelui_key']: '%s/%s' % (f['session'], f['file'])
            for f in d['frames']}, d


def canon(uv):
    """labelui.html's winding, re-applied. Both inputs claim to be canonical already;
    this is the cheap assertion that they are, and it costs nothing to make it true."""
    p = np.asarray(uv, float)
    sl = float(sum(p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1]
                   for i in range(4)))
    if sl <= 0:
        p = p[[0, 3, 2, 1]]
    return np.roll(p, -int(np.argmin(p[:, 0] + p[:, 1])), axis=0)


def normalise(inst, src, i):
    """One instance in the merged schema. Unknown fields are carried through."""
    c = canon(inst['corners'])
    out = dict(inst)
    out['corners'] = [[round(float(x), 2), round(float(y), 2)] for x, y in c]
    out['occluded'] = bool(inst.get('occluded', False))
    out['clipped'] = bool(inst.get('clipped', False))
    out['unsure'] = bool(inst.get('unsure', False))
    out['z'] = i
    out['src'] = src
    return out


def check_schema(hand, auto):
    """Verify the two schemas match rather than assuming it."""
    def fields(d):
        s = set()
        for v in d.values():
            for it in v:
                s |= set(it)
        return s
    fh, fa = fields(hand), fields(auto)
    # A hand file can legitimately contain nothing but verified negatives, in which case
    # there are no instances to have fields and "missing core fields" is a false alarm --
    # it also masked the unresolvable-key error it should have run after.
    missing = (set(CORE) - fh if fh else set()), set(CORE) - fa
    print('schema check')
    print('  hand fields: %s' % sorted(fh))
    print('  auto fields: %s' % sorted(fa))
    print('  core %s present in both: %s'
          % (list(CORE), not missing[0] and not missing[1]))
    if missing[0] or missing[1]:
        print('  MISSING hand=%s auto=%s' % (sorted(missing[0]), sorted(missing[1])))
    extra = fa - fh - {'src'}
    if extra:
        print('  auto-only fields (carried through, hand instances will lack them): %s'
              % sorted(extra))
    return not (missing[0] or missing[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hand', required=True, help='labels_gates.json from labelui.html')
    ap.add_argument('--auto', default=os.path.join(HERE, 'autolabels_vq1.json'))
    ap.add_argument('--index', default=os.path.join(HERE, 'vq2_label', 'index.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'labels_merged.json'))
    ap.add_argument('--drop-unsure', action='store_true',
                    help='omit instances Claire flagged unsure (corner-regression use)')
    ap.add_argument('--allow-unmapped', action='store_true',
                    help='keep hand keys index.json cannot resolve, under their own name')
    args = ap.parse_args()

    with open(args.hand) as fh:
        hand = json.load(fh)
    with open(args.auto) as fh:
        auto = json.load(fh)
    ok = check_schema(hand, auto)
    if not ok:
        print('\nrefusing to merge: the two schemas do not share the core fields')
        return 1

    keymap, idx = load_index(args.index)

    # ---- map hand keys into '<session>/<file>'
    mapped, unmapped = {}, []
    for k, v in hand.items():
        base = os.path.basename(k)
        if base in keymap:
            mapped[keymap[base]] = v
        elif '/' in k:
            mapped[k] = v                       # already fully qualified
        else:
            unmapped.append(k)
    if unmapped:
        print('\n%d hand key(s) not resolvable through %s:' % (len(unmapped), args.index))
        for k in unmapped[:10]:
            print('   ', k)
        if not args.allow_unmapped:
            print('refusing to merge and silently lose them; pass --allow-unmapped to '
                  'keep them under their own key')
            return 1
        for k in unmapped:
            mapped[k] = hand[k]

    # ---- merge
    out, stats = {}, Counter()
    for k, v in auto.items():
        out[k] = [normalise(it, 'auto', i) for i, it in enumerate(v)]
        stats['auto_inst'] += len(out[k])
    collide = 0
    for k, v in mapped.items():
        if k in out:
            collide += 1
            stats['auto_inst'] -= len(out[k])
        inst = [it for it in v if not (args.drop_unsure and it.get('unsure'))]
        stats['unsure_dropped'] += len(v) - len(inst)
        # An empty array means VERIFIED NO GATE HERE. A frame whose every instance was
        # `unsure` is the opposite claim -- "I could not label these" -- so emitting it
        # empty would turn an admission of doubt into a negative. Drop the frame instead.
        if v and not inst:
            stats['emptied_by_unsure'] += 1
            if k in out:
                collide -= 1          # nothing replaced it after all
                out.pop(k)
            continue
        out[k] = [normalise(it, 'hand', i) for i, it in enumerate(inst)]
        stats['hand_inst'] += len(out[k])
        if not out[k]:
            stats['verified_negative'] += 1

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)

    # ---- report
    per_sess = Counter(k.split('/', 1)[0] for k in out)
    print('\nmerged -> %s' % args.out)
    print('  frames        %d  (auto %d, hand %d, collisions resolved to hand %d)'
          % (len(out), len(auto), len(mapped), collide))
    print('  instances     %d  (auto %d, hand %d)'
          % (stats['auto_inst'] + stats['hand_inst'], stats['auto_inst'],
             stats['hand_inst']))
    print('  verified negatives (empty arrays, hand only)  %d'
          % stats['verified_negative'])
    if args.drop_unsure:
        print('  unsure instances dropped  %d' % stats['unsure_dropped'])
        print('  frames dropped (every instance unsure -- NOT emitted as negatives)  %d'
              % stats['emptied_by_unsure'])
    print('  frames per session:')
    for s, n in per_sess.most_common():
        print('    %-40s %5d' % (s, n))

    # category coverage of what she actually got through
    cats = {f['labelui_key']: f['category'] for f in idx['frames']}
    done = Counter(cats[os.path.basename(k)] for k in hand
                   if os.path.basename(k) in cats)
    if done:
        print('  hand-labelled frames by selection category:')
        for c, n in done.most_common():
            print('    %-10s %4d' % (c, n))
    return 0


if __name__ == '__main__':
    sys.exit(main())
