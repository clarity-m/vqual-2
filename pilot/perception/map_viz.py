"""Regenerate map_current_viz.png from map_vq2.json -- MEASURED layout.

Since 2026-08-02 the node positions and gate angles are MEASURED (skylight-compass
directions + metric pair distances, layout_directions_2026_08_02 in map_vq2.json), not
Claire's sketch. The sketch keeps a small inset for topological comparison. Edges keep
the d +/- MAD / dz / n labels from measured_pairs; the direction-refused 4-6 edge is
drawn crossed out. Falls back to the old sketch layout if the block is absent.

    python3 pilot/perception/map_viz.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = json.load(open(os.path.join(HERE, 'map_vq2.json')))
SK = json.load(open(os.path.join(HERE, 'map_approx.json')))

LAY = MAP.get('layout_directions_2026_08_02')
pairs = MAP['measured_pairs']

sketch_pos = {g['race_index']: (g['along_station'], g['across']) for g in SK['gates']}

fig, ax = plt.subplots(figsize=(17.5, 9.3))

if LAY is not None:
    pos = {int(g): (v['x'], v['y']) for g, v in LAY['positions_m'].items()}
    zup = {int(g): v.get('z_up_m') for g, v in LAY['positions_m'].items()}
    refused = set(LAY.get('refused_edges', {}))
    dzs_map = LAY.get('pair_dz_up_m', {})
    angles = LAY.get('gate_plane_angles_deg', {})
else:                                    # fallback: the pre-2026-08-02 sketch layout
    pos = sketch_pos
    zup, refused, dzs_map, angles = {}, set(), {}, {}

# race-order course arrows (thin, under everything)
for k in range(16):
    if k in pos and k + 1 in pos:
        xa, ya = pos[k]
        xb, yb = pos[k + 1]
        ax.annotate('', (xb, yb), (xa, ya), zorder=0,
                    arrowprops=dict(arrowstyle='-|>', color='#b0bec5', lw=1.0,
                                    shrinkA=14, shrinkB=14))

# edges
for name, v in pairs.items():
    a, b = (int(x) for x in name.split('-'))
    if a not in pos or b not in pos:
        continue
    xa, ya = pos[a]
    xb, yb = pos[b]
    if name in refused:
        ax.plot([xa, xb], [ya, yb], ':', color='#c0392b', lw=1.6, zorder=1)
        ax.annotate('REFUSED %s (re-measures 4-5)' % name,
                    ((xa + xb) / 2, (ya + yb) / 2), fontsize=8, color='#c0392b',
                    ha='center', va='center',
                    bbox=dict(fc='white', ec='#c0392b', alpha=0.85, pad=1.2), zorder=3)
        continue
    n = v['n']
    lw = 1.0 + 1.2 * np.log10(max(n, 1))
    ax.plot([xa, xb], [ya, yb], '-', color='#2e8b57', lw=lw, alpha=0.75, zorder=1)
    dz = dzs_map.get(name, {}).get('dz', v.get('dz_m'))
    dzs = '' if dz is None else ' dz%+.1f' % dz
    lab = '%.1fm%s\n±%.1f n=%d' % (v['dist_m'], dzs, v['mad_m'], n)
    ax.annotate(lab, ((xa + xb) / 2, (ya + yb) / 2), fontsize=8,
                color='#1b5e20', ha='center', va='center',
                bbox=dict(fc='white', ec='none', alpha=0.55, pad=0.6), zorder=3)

# nodes + measured gate-plane bars + heights
BAR = 5.0
for g, (x, y) in pos.items():
    a = angles.get(str(g))
    if a is not None:
        # gate plane line = normal azimuth + 90 (mod 180)
        t = math.radians(a['normal_az_sketch_frame_mod180'] + 90.0)
        dx, dy = BAR * math.cos(t), BAR * math.sin(t)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], '-', color='#e65100', lw=3.0,
                zorder=4, solid_capstyle='round')
    ax.scatter([x], [y], s=520, c='#1565c0', edgecolors='black', zorder=5)
    ax.annotate(str(g), (x, y), color='white', ha='center', va='center',
                fontsize=11, fontweight='bold', zorder=6)
    z = zup.get(g)
    if z is not None:
        ax.annotate('z%+.1f' % z, (x, y), xytext=(0, -17),
                    textcoords='offset points', fontsize=7.5, color='#37474f',
                    ha='center', va='top', zorder=6)

ax.plot([], [], '-', color='#2e8b57', lw=3, label='measured edge d±MAD, dz (signed, '
                                                  'up+), n (width ~ rows)')
ax.plot([], [], '-', color='#e65100', lw=3, label='measured gate-plane orientation '
                                                  '(mod 180; PnP normals + compass)')
ax.plot([], [], ':', color='#c0392b', lw=1.6, label='edge refused by direction '
                                                    'evidence')
ax.plot([], [], '-', color='#b0bec5', lw=1.0, label='race order 0 -> 16 (arrows)')

if LAY is not None:
    res = LAY['residuals']
    prov = LAY['quadrant_provenance']
    n_sketch = sum('SKETCH-PRIOR' in p for p in prov.values())
    ax.set_title(
        'VQ2 gate map -- MEASURED layout (2026-08-02): skylight-compass pair '
        'DIRECTIONS + metric distances (vision + IMU, no pose stream)\n'
        '19 pair vectors; residual p90 %.1f m; x-session bearing spread med '
        '%.1f deg; loop closures 1.2-1.6%%; sketch-aligned axes (%.1f deg + '
        'mirror); %d sessions on sketch-prior quadrant (the 2-3, 6-7, 15-16 legs)'
        % (res['vector_p90_m'],
           res['cross_session_bearing_spread_median_deg'],
           LAY['frame']['grid_to_sketch_rotation_deg'], n_sketch), fontsize=10.5)
    ax.set_xlabel('along hangar, METRES (measured; origin gate 0)')
    ax.set_ylabel('across hangar, METRES (measured)')
else:
    ax.set_title('VQ2 gate map -- sketch layout (no measured layout block found)')

ax.legend(loc='upper center', fontsize=9, ncol=2,
          bbox_to_anchor=(0.5, -0.10))
ax.set_aspect('equal')
ax.invert_xaxis()
# +y points toward the 21-29 column row, which is on the drone's RIGHT leaving
# the start pad (confirmed against the sim). With x inverted so the race runs
# left->right, y must inverted too or the transform is a REFLECTION (-x,+y)
# and every turn reads backwards. (-x,-y) is a 180 deg rotation and is correct.
ax.invert_yaxis()
ax.grid(alpha=0.25)

# sketch inset for topological comparison
if LAY is not None:
    ins = fig.add_axes([0.045, 0.055, 0.21, 0.24])
    for k in range(16):
        if k in sketch_pos and k + 1 in sketch_pos:
            xa, ya = sketch_pos[k]
            xb, yb = sketch_pos[k + 1]
            ins.plot([xa, xb], [ya, yb], '-', color='#b0bec5', lw=0.8, zorder=1)
    for g, (x, y) in sketch_pos.items():
        ins.scatter([x], [y], s=90, c='#8e44ad', edgecolors='black', zorder=2)
        ins.annotate(str(g), (x, y), color='white', ha='center', va='center',
                     fontsize=6, fontweight='bold', zorder=3)
    ins.invert_xaxis()
    ins.invert_yaxis()
    ins.set_title("Claire's sketch (station units) -- topology check", fontsize=8)
    ins.tick_params(labelsize=6)

out = os.path.join(HERE, 'map_current_viz.png')
fig.savefig(out, dpi=110)
print('wrote', out)
