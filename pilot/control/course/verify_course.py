"""verify_course.py -- the export's own check: sanity_check + the randomization envelope.

    python3 pilot/course/verify_course.py

(a) sanity_check on the nominal course and on 100 samples, with the error distribution;
(b) exported positions vs map_vq2.json's measured pair distances (max deviation);
(c) course_samples.png -- nominal top-down plus 20 translucent sampled instances.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import course_vq2 as C            # noqa: E402

MAP = os.path.normpath(os.path.join(HERE, '..', 'perception', 'map_vq2.json'))
PNG = os.path.join(HERE, 'course_samples.png')


def main():
    nom = C.load()
    print('--- (a) sanity_check')
    e0, z0 = C.sanity_check(nom)
    print('nominal: worst edge %.3f m, worst dz %.3f m' % (e0, z0))
    errs, zerrs = [], []
    for s in range(100):
        a, b = C.sanity_check(C.sample(s))
        errs.append(a)
        zerrs.append(b)
    errs, zerrs = np.array(errs), np.array(zerrs)
    print('100 samples, worst-edge reconstruction error:  med %.3f  p90 %.3f  max %.3f m'
          % (np.median(errs), np.percentile(errs, 90), errs.max()))
    print('100 samples, worst-edge dz error:              med %.3f  p90 %.3f  max %.3f m'
          % (np.median(zerrs), np.percentile(zerrs, 90), zerrs.max()))

    print('\n--- (b) exported positions vs map_vq2.json measured_pairs')
    mapj = json.load(open(MAP))
    MP = mapj['measured_pairs']
    P = nom.positions
    worst = ('', 0.0)
    for e in nom.raw['edges']:
        a, c = e['pair']
        key = '%d-%d' % (a, c)
        d = float(np.hypot(*(P[c][:2] - P[a][:2])))
        dev = abs(d - float(MP[key]['horiz_m']))
        if dev > worst[1]:
            worst = (key, dev)
    print('max |exported horizontal distance - map measured_pairs horiz_m| = '
          '%.3f m  (edge %s)' % (worst[1], worst[0]))
    # heights too
    DZ = mapj['layout_directions_2026_08_02']['pair_dz_up_m']
    wz = ('', 0.0)
    for e in nom.raw['edges']:
        a, c = e['pair']
        key = '%d-%d' % (a, c)
        dev = abs(float(P[c][2] - P[a][2]) - float(DZ[key]['dz']))
        if dev > wz[1]:
            wz = (key, dev)
    print('max |exported dz - map pair_dz_up_m|                            = '
          '%.3f m  (edge %s)' % (wz[1], wz[0]))
    gp = mapj['layout_directions_2026_08_02']['positions_m']
    dmax = max(np.hypot(P[g][0] - gp[str(g)]['x'], P[g][1] - gp[str(g)]['y'])
               for g in range(17))
    print('max |exported gate position - map positions_m|                  = '
          '%.4f m' % dmax)

    print('\n--- (b2) tilt: gate 9 leans, the other 16 do not, and it leans the SAME WAY '
          'in every sample')
    q = C.gate_corners(9, nom, size_m=nom.outer_m)
    d = 0.5 * (q[2] + q[3]) - 0.5 * (q[0] + q[1])       # top edge midpoint - bottom
    az = math.degrees(math.atan2(d[1], d[0])) % 360.0
    print('gate 9 nominal: tilt %.1f deg, top displaced %.2f m horizontally toward '
          'azimuth %.1f deg (exported lean azimuth %.1f)'
          % (nom.tilt_deg[9], float(np.hypot(d[0], d[1])), az, nom.tilt_lean_deg[9]))
    assert nom.tilt_deg[9] > 10.0, 'gate 9 must be exported tilted'
    assert abs((az - nom.tilt_lean_deg[9] + 180.0) % 360.0 - 180.0) < 15.0, \
        'gate 9 corners lean the wrong way'
    others = [nom.tilt_deg[g] for g in range(17) if g != 9]
    print('other 16 gates, nominal tilt: max %.1f deg' % max(others))
    assert max(others) == 0.0
    S = [C.sample(s) for s in range(200)]
    t9 = np.array([s.tilt_deg[9] for s in S])
    l9 = np.array([s.tilt_lean_deg[9] for s in S])
    tv = np.array([[s.tilt_deg[g] for g in range(17) if g != 9] for s in S])
    print('200 samples: gate 9 tilt %.1f +- %.1f deg (range %.0f-%.0f), lean azimuth '
          '%.1f +- %.1f deg' % (t9.mean(), t9.std(), t9.min(), t9.max(),
                                l9.mean(), l9.std()))
    print('200 samples: every other gate tilt max %.1f deg (all clipped at 3 sigma)'
          % tv.max())
    assert t9.min() > 10.0, 'a sample must never make gate 9 nearly vertical'
    assert tv.max() < 12.5, 'a "vertical" gate drew an implausible tilt'
    assert float(np.abs((l9 - 129.7 + 180) % 360 - 180).max()) < 25.0

    print('\n--- (c) envelope plot')
    samples = [C.sample(s) for s in range(20)]
    fig, axes = plt.subplots(2, 1, figsize=(15, 11.0), dpi=130)
    ax = axes[0]
    for s in samples:
        Q = s.positions
        ax.plot(Q[:, 0], Q[:, 1], '-', color='#e8663d', lw=1.0, alpha=0.22, zorder=1)
        ax.plot(Q[:, 0], Q[:, 1], '.', color='#e8663d', ms=4, alpha=0.30, zorder=1)
    ax.plot(P[:, 0], P[:, 1], '-', color='#1f2933', lw=2.0, zorder=3, label='nominal')
    ax.plot(P[:, 0], P[:, 1], 'o', color='#1f2933', ms=6, zorder=4)
    for g in range(17):
        sxy = nom.raw['gates'][g]['sigma_xy_m']
        ax.annotate('%d' % g, (P[g, 0], P[g, 1]), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=9, zorder=5)
        ax.annotate('%.1f' % sxy, (P[g, 0], P[g, 1]), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=6.5, color='#7b8794',
                    zorder=5)
    # mark the edges the map itself calls weak
    for e in nom.raw['edges']:
        a, c = e['pair']
        if e['n_rows'] <= 15 or 'CONTESTED' in ' '.join(e['flags']):
            ax.plot([P[a, 0], P[c, 0]], [P[a, 1], P[c, 1]], '-', color='#b91c1c',
                    lw=1.6, alpha=0.85, zorder=2)
            mx, my = 0.5 * (P[a, 0] + P[c, 0]), 0.5 * (P[a, 1] + P[c, 1])
            ax.annotate('%d-%d n=%d' % (a, c, e['n_rows']), (mx, my),
                        textcoords='offset points', xytext=(0, 6), ha='center',
                        fontsize=6.5, color='#b91c1c', zorder=5)
    ax.set_aspect('equal')
    ax.invert_xaxis()          # race runs toward -x; draw it left to right
    ax.set_xlabel('x  [m]   (race runs 0 -> 16 toward -x; axis inverted so the race '
                  'reads left to right)')
    ax.set_ylabel('y  [m]  (+y toward the Station 21-29 column row)')
    ax.set_title('VQ2 course: nominal (black) + 20 domain-randomized samples (orange). '
                 'Red = thin/contested edges. Small grey number = sigma_xy from gate 0.')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc='upper right')

    # ---- panel 2: same samples, each rigidly aligned (Kabsch) to the nominal layout.
    # The top panel is dominated by chain drift accumulating away from the gate-0 gauge,
    # which hides LOCAL shape uncertainty. Removing the best-fit rigid motion shows what
    # is actually uncertain about the SHAPE -- and the thin edges stand out.
    ax2 = axes[1]
    A0 = P[:, :2] - P[:, :2].mean(axis=0)
    for s in samples:
        B = s.positions[:, :2]
        B0 = B - B.mean(axis=0)
        U, _S, Vt = np.linalg.svd(B0.T @ A0)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        Q = (B0 @ R) + P[:, :2].mean(axis=0)
        ax2.plot(Q[:, 0], Q[:, 1], '-', color='#e8663d', lw=1.0, alpha=0.22, zorder=1)
        ax2.plot(Q[:, 0], Q[:, 1], '.', color='#e8663d', ms=4, alpha=0.30, zorder=1)
    ax2.plot(P[:, 0], P[:, 1], '-', color='#1f2933', lw=2.0, zorder=3)
    ax2.plot(P[:, 0], P[:, 1], 'o', color='#1f2933', ms=6, zorder=4)
    for g in range(17):
        ax2.annotate('%d' % g, (P[g, 0], P[g, 1]), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=9, zorder=5)
    for e in nom.raw['edges']:
        a, c = e['pair']
        if e['n_rows'] <= 15 or 'CONTESTED' in ' '.join(e['flags']):
            ax2.plot([P[a, 0], P[c, 0]], [P[a, 1], P[c, 1]], '-', color='#b91c1c',
                     lw=1.6, alpha=0.85, zorder=2)
    ax2.set_aspect('equal')
    ax2.invert_xaxis()
    ax2.set_xlabel('x  [m]')
    ax2.set_ylabel('y  [m]')
    ax2.set_title('same 20 samples, rigidly aligned to the nominal (gauge freedom '
                  'removed): LOCAL shape uncertainty')
    ax2.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    fig.savefig(PNG)
    print('wrote %s' % PNG)

    # per-gate spread actually realized in the plotted samples
    S = np.array([s.positions for s in samples])
    print('\nrealized spread over the 20 plotted samples (std of x,y / m):')
    print('  ' + '  '.join('%d:%.2f' % (g, np.hypot(S[:, g, 0].std(), S[:, g, 1].std()))
                           for g in range(17)))


if __name__ == '__main__':
    main()
