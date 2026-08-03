# Cross-eval on the CLEAN val subset only -- frames that are held out under BOTH block
# splits (the old model's v1-file split AND the current v3/merged split).
#
# Why this exists: deleting labels shifted the block boundaries, so 35% of the current
# val (68% of the >=120 px bucket, 45% of clipped) sits inside the OLD model's training
# blocks. Every ACCEPT/REJECT so far graded the incumbent partly on its own training
# data, in exactly the buckets that decided the verdict. val_clean_keys.json carries the
# frame lists, computed on the laptop from the v1-era key set (filtered + negatives).
#
# Run INSIDE a gatenet_colab session after the setup cells (needs G, P, LOCAL_CACHE,
# meta, va_items, RUNS_DRIVE):
#   exec(open(f"{CODE_DRIVE}/crosseval_clean.py").read())
#
# Scores every checkpoint in CKPTS on the clean subset, prints per-bucket AND per-gate
# tables, and writes crosseval_clean.png (grouped bars) next to the checkpoints on Drive.

import json as _json
import os as _os
import numpy as _np
import torch as _torch
from torch.utils.data import DataLoader as _DL

CKPTS = [('OLD  block', f'{RUNS_DRIVE}/block/best.pt'),
         ('v3+rate    ', f'{RUNS_DRIVE}/colab-v3/best.pt'),
         ('v3 no-wt   ', f'{RUNS_DRIVE}/colab-v3-nw/best.pt')]

_clean = set(_json.load(open(f'{CODE_DRIVE}/val_clean_keys.json'))['clean'])
_key = lambda it: '/'.join(_os.path.normpath(it['path']).replace('\\', '/').split('/')[-3::2])
_sub = [i for i, it in enumerate(va_items) if _key(it) in _clean]
_items = [va_items[i] for i in _sub]
print(f'clean subset: {len(_items)} of {len(va_items)} val instances '
      f'({len(set(_key(i) for i in _items))} frames)\n')

_dev = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
_loader = _DL(P.CachedGateCrops(_items, LOCAL_CACHE, meta, train=False),
              batch_size=256, shuffle=False, num_workers=2)

_res = {}
for _name, _ck in CKPTS:
    _m = G.GateNet(1.0).to(_dev)
    _m.load_state_dict(_torch.load(_ck, map_location=_dev, weights_only=False)['model'])
    _m.eval()
    _r = G.evaluate(_m, _loader, _dev, _items)
    _res[_name] = _r['corner'].mean(1)          # per-instance corner err
    del _m

_size = _np.array([it['size_px'] for it in _items])
_clip = _np.array([bool(it.get('clipped')) for it in _items])
_gate = _np.array([int(it.get('gate', -1)) for it in _items])

def _row(mask, label):
    if mask.sum() == 0:
        return
    cells = '  '.join(f'{_np.median(_res[n][mask]):6.2f}/{_np.percentile(_res[n][mask],90):7.2f}'
                      for n, _ in CKPTS)
    print(f'  {label:22s} n={mask.sum():4d}   {cells}')

print(' ' * 32 + '   '.join(f'{n:>14s}' for n, _ in CKPTS) + '     (med/p90 px)')
_row(_np.ones(len(_items), bool), 'ALL (clean)')
for _lo, _hi, _lab in [(0, 15, 'size 0-15'), (15, 30, 'size 15-30'), (30, 60, 'size 30-60'),
                       (60, 120, 'size 60-120'), (120, 1e9, 'size >=120')]:
    _row((_size >= _lo) & (_size < _hi), _lab)
_row(_clip, 'clipped')
_row(~_clip, 'unclipped')
print()
for _g in sorted(set(_gate)):
    _row(_gate == _g, f'gate {_g}')
_row(_clip & (_size >= 60), 'clipped & >=60 px')

# grouped bar chart: per-gate medians, plus the decision buckets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
_groups = [(f'gate {g}', _gate == g) for g in sorted(set(_gate)) if g >= 0]
_groups += [('clipped', _clip), ('>=120px', _size >= 120), ('ALL', _np.ones(len(_items), bool))]
_fig, _ax = _plt.subplots(figsize=(13, 5.5))
_w = 0.26
for _i, (_name, _) in enumerate(CKPTS):
    _vals = [_np.median(_res[_name][m]) for _, m in _groups]
    _ax.bar(_np.arange(len(_groups)) + (_i - 1) * _w, _vals, _w, label=_name.strip())
_ax.set_xticks(range(len(_groups)))
_ax.set_xticklabels([g for g, _ in _groups], rotation=20)
_ax.set_ylabel('corner median, px (log)')
_ax.set_yscale('log')
_ax.set_title(f'Checkpoints on the CLEAN val subset ({len(_items)} instances held out of '
              f'EVERY training run) -- per gate and decision buckets')
_ax.legend(); _ax.grid(alpha=0.3, axis='y')
_plt.tight_layout()
_out = f'{RUNS_DRIVE}/crosseval_clean.png'
_plt.savefig(_out, dpi=110)
print(f'\nwrote {_out}')
